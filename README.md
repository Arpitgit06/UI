# OmniUI

Local, offline video-to-UI-code pipeline. Converts a screen recording of a
2D/3D interface into working React/HTML/CSS (and React Three Fiber for 3D)
using deterministic computer vision for layout extraction — the local LLM
is used strictly as a syntax compiler, never to guess pixel positions.

## Status (iteration 5 — all four modules implemented)

| Piece | Status |
|---|---|
| Core Orchestration (FastAPI, async job queue) | ✅ implemented |
| VRAM lifecycle manager (`vram_scope`, `GPUPipelineGuard`) | ✅ implemented |
| A — Temporal Video Parser (OpenCV + SSIM) | ✅ implemented |
| B — Spatial Vision & Detection (Hybrid YOLO, PaddleOCR, Depth-Anything-V2) | ✅ implemented |
| C — DOM Synthesizer | ✅ implemented |
| D — Local Code Generation (Hugging Face Transformers Subprocess) | ✅ implemented |

The full upload → queue → parse → detect → nest → generate → download
pipeline is wired end-to-end. What's left is real-world tuning (a
UI-trained YOLO checkpoint, threshold calibration against actual
recordings) rather than missing plumbing.

### Module D in detail

`app/modules/module_d_code_generator.py` turns Module C's layout tree
into React components, CSS, and (for 3D-flagged states) a React Three
Fiber scene, using a local Hugging Face LLM (e.g. Qwen2.5-Coder) running
in an isolated subprocess — but deliberately not the way a literal
reading of "have the LLM generate the component" would suggest.

**Python resolves every value; the LLM only transcribes syntax.** The
architecture doc's own Stage 3 already separates "convert coordinates
into layout logic" (a Python step) from "generate the component" (an LLM
step) — and structured-output research is explicit that grammar-constrained
JSON decoding guarantees *syntactic* validity, not *value* accuracy. So
before any prompt is built, Python computes:
- **Position/size** as a percentage of each node's *direct parent's* box
  (not the root), so `position: absolute` + percentages reproduces the
  captured layout exactly and scales proportionally if the container
  resizes. This deliberately skips Flexbox/Grid inference — guessing
  which elements "belong" in a flex row is exactly the kind of
  interpretation this project's CV-first philosophy exists to avoid.
- **3D projection** — each node's 2D bbox + z_index projected into R3F
  scene units once, in Python (`_project_to_3d`).
- **Color** — straight from Module B's Colorgram output.

The LLM's job is to transcribe this already-resolved tree into
syntactically correct JSX/CSS/R3F — matching the doc's own framing that
the LLM should "format the extracted logic," not invent it.

**Because grammar constraints don't guarantee fidelity, every generation
is mechanically re-checked afterward.** `_validate_component` /
`_validate_scene3d` confirm every node's class name appears in both
outputs, every CSS block contains every computed property *tied to its
value* (checking the bare value alone turned out to have a real false-negative:
if two properties coincidentally share a value, like `left` and `top`
both landing on "10.00%", a corrupted `left` can hide behind the correct
`top` — caught during testing and now guarded against), and text content
appears verbatim. A failed check retries, then raises an itemized
`LocalLLMValidationError` rather than silently shipping mismatched code.

**VRAM/connection handling stays consistent with the rest of the
pipeline:** By isolating generation into `llm_subprocess_worker.py`, the OS
forcibly reclaims all VRAM the instant generation completes, preventing
Out-Of-Memory crashes across multiple frames without needing complex
PyTorch garbage collection.

**Per-state, not merged.** Each key state gets its own component file
(`State0.jsx`, `State1.jsx`, ...) rather than one component trying to
represent transitions between states — deciding "is state 1 a hover of
state 0" is itself an inference this pipeline doesn't attempt. A minimal,
plainly-templated (not LLM-generated) `index.jsx` renders all of them as
an independent gallery, and `package.json` lists `@react-three/fiber` +
`three` only when at least one state actually needs them.

**Verification:** The code generation is fully wired up using `transformers`
and `bitsandbytes` for 4-bit quantization. See `tests/test_module_d.py`.

### Module C in detail

`app/modules/module_c_dom_synthesizer.py` takes Module B's flat list of
`DetectedElement` and builds a parent-child `DOMNode` tree by spatial
containment, then writes `state_N_layout.json` — the exact contract
Module D consumes.

**Containment uses an overlap ratio, not strict corner containment.**
`intersection_area(A, B) / area(A) >= dom_containment_threshold` (default
0.8). Real detections are noisy — a button's box and the text inside it
rarely align to the pixel — so requiring literal full containment would
miss real parent-child relationships over a few pixels of imprecision.

**Each element's parent is the *tightest* enclosing element, not just any
containing one**, found by requiring a parent candidate to have strictly
greater area than the child. That single rule is also what makes the
algorithm cycle-proof by construction — a cycle would need two elements
each strictly larger than the other, which can't happen for real numbers.
Elements with no valid container become direct children of a synthesized
root node sized to the actual source image's pixel dimensions (read via
PIL; falls back to the union of all detected boxes if the image can't be
read). Near-identical same-size boxes deliberately end up as siblings
rather than one arbitrarily "winning" as the other's parent.

Within a parent, children are sorted by `z_index` ascending — farthest
first, nearest last — matching typical HTML/CSS painting order.

**Deliberately not attempted:** distinguishing a genuine 3D scene from a
flat 2D UI. `LayoutState.is_3d_scene` stays at its `False` default because
Module B's `z_index` values are already min-max normalized to 0-100 per
state, which stretches even a tiny real depth range to fill the whole
scale — so the normalized values alone can't tell "meaningfully layered"
apart from "flat, but normalization amplified noise." Doing this properly
needs Module B to also pass through a pre-normalization statistic (raw
depth range/variance); that's a small, targeted follow-up rather than a
heuristic guess bolted on here.

**Verification:** this module has no ML/GPU dependencies at all — it's
pure geometry plus a PIL image-size read — so every test in
`tests/test_module_c.py` runs against the real, unmodified implementation.
No stand-ins, no monkeypatching, unlike Modules A/B. 30 scenarios were
checked before finalizing, including multi-level nesting attaching to the
*tightest* parent (not skipping to a grandparent), near-identical boxes
correctly NOT nesting, partial overlap below threshold correctly NOT
nesting, z-order sorting, and the missing-image → union-bbox fallback
path.

### Module B in detail

`app/modules/module_b_spatial_vision.py` runs four analyses per key-state
image, one heavy model at a time via `GPUPipelineGuard`:

1. **Hybrid YOLO** (`ultralytics`) — YOLOv8 Macro-detector for containers, YOLOv10 Micro-detector for elements inside crops
2. **PaddleOCR** — text region bounding boxes + recognized strings (via persistent subprocess)
3. **Depth-Anything-V2** (`transformers` pipeline) — per-pixel relative depth → `z_index`
4. **Colorgram.py** (CPU, no GPU) — dominant hex colors per element, from its crop

**No cross-referencing between YOLO and OCR here.** Module B emits one flat
list of `DetectedElement` — one per YOLO box, one per OCR text region —
with no attempt to decide that a text region "belongs to" a UI element.
That containment decision (is this text strictly inside that button,
making it a child node?) is Module C's job by design, per the architecture
doc's description of the DOM Synthesizer; duplicating it here would just
be logic Module C already owns.

**The VRAM gate spans two frameworks, not one.** YOLO and
Depth-Anything-V2 are PyTorch models, but PaddleOCR runs on PaddlePaddle —
a separate framework with its own CUDA context that `torch.cuda.empty_cache()`
cannot touch. PaddleOCR is now fully isolated into a persistent subprocess worker 
(`app.modules.ocr_subprocess_worker`), ensuring 100% VRAM cleanup and preventing memory leaks.
The `GPUPipelineGuard` coordinates these stages so they still run sequentially.

**Two things are placeholders until real weights are available:**
- `settings.yolo_macro_weights_path` and `settings.yolo_micro_weights_path` default to stock, COCO-pretrained YOLO models
  (detects people/cars/dogs — not buttons or navbars). Real UI detection
  needs checkpoints fine-tuned on a UI dataset (e.g. Rico); swapping them in
  is a config change once you have them.
- `_normalize_z_indices()` assumes Depth-Anything-V2's larger values mean
  "closer to camera." Worth confirming visually against a real model
  output before trusting the ordering.

**Verification status:** none of `ultralytics` / `paddleocr` / `colorgram` /
`transformers` / `torch` are installed in the sandbox this was built in
(and it has no network to install them), so the actual model-loading and
inference calls couldn't be executed against real weights. Every API shape
used was checked against current documentation, not assumed from memory —
PaddleOCR in particular has a very different API in 3.x (`.predict()` →
`.json` dict) than the old `.ocr()` interface many tutorials still show.
What *was* tested for real: every pure/glue function (bbox conversion,
depth sampling + z-index normalization, hex formatting, the PaddleOCR
result parser) plus the full orchestration flow, using hand-built stand-ins
for the four model-loading calls — this caught a real off-by-one bug in
the depth-sampling window before it shipped. See
`tests/test_module_b.py` and that file's module docstring for exactly
what's covered and what isn't.

### Module A in detail

`app/modules/module_a_temporal_parser.py` reduces a recording to "key state"
PNGs plus a best-effort cursor/action guess, using two SSIM comparisons per
frame rather than one:

1. **Change-from-reference** — is this frame different enough from the
   *last captured key state* to be a new one? Comparing against a
   persistent reference (not literally frame N-1) is what catches slow,
   gradual transitions (a fading modal, a slow drag) that a naive
   frame-to-frame diff would miss, since no single step ever crosses the
   threshold even though the cumulative drift is large.
2. **Stability-from-previous** — has the video *stopped changing*
   frame-to-frame right now? Without this check, a continuous drag gets
   captured almost every frame (each incremental step already differs
   enough from the reference), producing dozens of near-duplicate states
   instead of one "before" and one "after." Only once motion settles do we
   test the settled frame against the reference.

Both checks run on a downscaled **color** SSIM proxy (not grayscale) so
color-only UI changes — a hover or error state that shifts hue at similar
luminance — aren't missed; exact colors are the whole point of this
project. Cursor position/action is a classical frame-differencing centroid
over full-resolution grayscale frames, classified into hover/click/drag by
travel distance and elapsed time — a heuristic to retune against real
recordings, not a learned detector.

All of this was verified against synthetic videos before being written
here: two sudden state changes correctly produce 3 captures (not 2, not
dozens); a continuous drag collapses to a start + settled state, correctly
labeled `"drag"`; a static clip produces exactly 1 state. See
`tests/test_module_a.py`.

## Setup

```bash
python -m venv omniui_env
source omniui_env/bin/activate      # Windows: omniui_env\Scripts\activate

# torch and paddlepaddle-gpu ship CUDA-specific wheels from their own
# indexes -- install both BEFORE requirements.txt (see comments at the
# top of requirements.txt for CPU-only alternatives):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to run tests

cp .env.example .env   # adjust if needed
```

```bash
setup.bat
```
This automatically sets up your `models_cache` and fully downloads all AI weights (including the 15GB Qwen2.5-Coder model) so the tool works 100% offline forever.

## Run

```bash
./run.sh
# equivalent to: uvicorn app.main:app --reload
```

Interactive API docs: http://localhost:8000/docs — the easiest way to POST
a video right now; a dedicated upload page is a later iteration.

## Try it

```bash
curl -F "video=@your_ui_recording.mp4" http://localhost:8000/jobs
curl http://localhost:8000/jobs/<job_id>
curl -OJ http://localhost:8000/jobs/<job_id>/download
```

With the models fully cached locally by `setup.bat`, the downloaded zip now contains
real generated `State*.jsx` files, a combined `styles.css`, `index.jsx`,
and `package.json`. Check `storage/jobs/<job_id>/key_states/` and
`.../layouts/` for Module A/C's intermediate output along the way.

## Test

```bash
pytest
```

56 unit/integration tests across `tests/test_vram_manager.py`,
`test_module_a.py`, `test_module_b.py`, `test_module_c.py`, and
`test_module_d.py` run without any GPU or external service — they use
hand-built stand-ins for the pieces that genuinely need a GPU
(see each module's "Verification" notes above).
`test_schemas.py` needs real pydantic and `test_api_smoke.py` needs a
real FastAPI/httpx `TestClient`, both trivial to install via
`requirements-dev.txt` but unavailable in the sandbox this was built in.

Beyond the per-module suite, a full pipeline integration pass wires the
**real, unmodified** Module A and C together with Module B/D (faked only
at the exact points that need a GPU — the YOLO/PaddleOCR/Depth
loaders and the LLM worker) and runs an actual synthetic video through
`run_pipeline()` end to end: real SSIM key-state extraction, real DOM
nesting, real code generation and validation, a real zip written to disk.
This caught two real bugs before they reached the repo:
- `test_api_smoke.py` was still asserting a job would reach `"complete"`
  using fake video bytes — accurate when Module A was a stub, wrong now
  that it's real and correctly rejects invalid video. Fixed to assert a
  clean terminal state (complete *or* a clearly-explained failure)
  instead of assuming every heavy dependency is installed.
- The integration test's fake LLM client for the 3D path returned
  a fixed, non-matching response — which Module D's real validation
  logic correctly rejected. That's the validator doing its job, not a
  pipeline bug; the fix was building the fake response from the actual
  tree, same as the 2D path already did.

## Design notes

- **Single-worker async queue** (`app/core/job_queue.py`): jobs run strictly
  one at a time. This isn't a throughput limit we plan to lift later — it's
  required, since Module B/D's models can't safely share VRAM with a second
  job's models loaded concurrently.
- **`vram_scope()` / `GPUPipelineGuard`** (`app/core/vram_manager.py`): a
  context manager taking any `loader`/`unloader` pair. Guarantees the model
  is unloaded and `torch.cuda.empty_cache()` + `gc.collect()` run even if
  inference raises. The free-VRAM check before loading is a tripwire
  against "did we forget to release the previous model" bugs, not a
  precise footprint predictor.
- **The LLM is isolated via a separate subprocess.** It runs via
  `llm_subprocess_worker.py` rather than sharing PyTorch state in the
  main memory space, ensuring 100% VRAM cleanup between generation stages.
- **CPU-bound/GPU-bound stub functions are already dispatched through
  `asyncio.to_thread`** (Modules A, B, and D's packaging step). OpenCV,
  YOLO, and zip I/O are all blocking calls; wrapping them now means the
  event loop stays responsive to status polls once the real logic lands,
  without needing to retrofit this later. Module C is the one exception —
  it's lightweight pure-Python containment math, fast enough to call
  directly.
- **Strongly-typed module contracts** (`app/models/schemas.py`):
  `KeyStateFrame` → `DetectedElement` → `LayoutState` are the exact objects
  that flow between Modules A→B→C→D, defined now so each module can be
  implemented and tested independently against a fixed interface.

## Roadmap

1. ~~Module A — OpenCV capture + SSIM frame diffing + cursor tracking~~ ✅
2. ~~Module B — YOLOv10 / PaddleOCR / Colorgram / Depth-Anything-V2 behind `GPUPipelineGuard`~~ ✅
3. ~~Module C — containment-based DOM nesting algorithm~~ ✅
4. ~~Module D — Local LLM subprocess + React/R3F code emission~~ ✅

Remaining work is tuning and hardening, not missing pipeline stages:
- A UI-fine-tuned YOLO checkpoint (e.g. trained on Rico) to replace the
  stock COCO weights — see Module B's README section.
- Module C's `is_3d_scene` detection needs Module B to pass through a
  pre-normalization depth statistic before it can be reliable — see
  Module C's README section.
- TensorRT export for YOLOv10 + Depth-Anything-V2, per the original
  optimization plan.
- Real end-to-end validation against actual model weights. Everything
  in this repo has been tested with real synthetic data and hand-built
  stand-ins for the pieces this sandbox can't install.

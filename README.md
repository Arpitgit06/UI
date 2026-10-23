# OmniUI

OmniUI turns screen recordings into local React projects. It combines UI recognition,
measured layouts, native controls, preserved artwork, and generated visual assets
with browser verification and a correction workflow.

The output is a frontend starting point for another project. OmniUI reconstructs
visible UI; application services and backend integration belong to the consuming project.

## Features

- Upload MP4, WebM, or MKV recordings through a local dashboard.
- Detect text and control roles, group elements, and compile measured React/CSS layouts.
- Export native form controls with browser validation and local interaction hooks.
- Preserve complex artwork and isolate observed motion where possible.
- Generate separate artwork candidates with LTX-Video 2B.
- Review detected elements and confirm navigation between captured screens.
- Build and inspect output in Chromium, with screenshot comparisons and diagnostic reports.
- Download a standalone Vite/React project with source, assets, and design metadata.

## Requirements

| Requirement | Local setup |
| --- | --- |
| Operating system | Windows; the automated installer uses Windows batch scripts |
| Python | Python 3.11 recommended; installer requires 3.10 or newer |
| Node.js | Node.js 20+ with npm available on `PATH` |
| GPU | NVIDIA CUDA GPU; the bounded local profile targets 8 GB VRAM |
| Memory and disk | Allow for large model downloads, Python packages, and intermediate video assets |
| Network | Required during installation and model caching |

Models run in separate sequential processes to limit concurrent GPU memory use.
Memory requirements increase with input complexity, token counts, and generation
settings. The 8 GB profile does not establish that every resolution or video will fit.

## Installation

From the repository root in PowerShell:

```powershell
.\setup.bat
```

The installer creates `.venv`, installs the tested Python dependencies, configures
`.env`, caches the vision/coder models and LTX components, and installs frontend
verification dependencies and Chromium. Model and browser downloads live in
`models_cache/`. Existing cached files are reused.

Setup sets the model choices in `.env` and preserves unrelated settings.
Preview the installation commands without downloading anything:

```powershell
.\setup.bat --plan
```

Verify that the port, dependencies, model files, and browser are available:

```powershell
.\start.bat --check
```

## Running the application

```powershell
.\start.bat
```

Open the dashboard at <http://127.0.0.1:8000/>. API documentation is available at
<http://127.0.0.1:8000/docs>.

1. Upload a recording of the interface.
2. Wait for analysis, code generation, and browser verification.
3. Open the reconstructed UI and inspect the review notes.
4. Use the review editor to correct roles, text, bounds, styling, or navigation.
5. Build a reviewed revision or download the generated project.

A completed pipeline run may still produce a draft requiring review. Check
`reconstruction_status` and the acceptance report before relying on the result.

If port 8000 is occupied or reserved by Windows, select another port:

```powershell
.\start.bat --port 8001
```

An existing OmniUI service on the requested port is reused. Automatic reload is
disabled by default; `--reload` enables development reload. Stop the server with
`Ctrl+C`. `run.sh` uses the same launcher with a separately prepared Unix virtual
environment; it is not a Linux dependency installer.

## Architecture and models

| Stage | Implementation | Role |
| --- | --- | --- |
| Video analysis | OpenCV and temporal sampling | Select stable UI states across the recording |
| Text and region extraction | CPU PaddleOCR; optional UI-trained YOLO checkpoints | Recover text and grounded element bounds |
| UI recognition | `Qwen/Qwen2-VL-7B-Instruct`, NF4 | Classify grounded UI regions |
| Layout synthesis | Contextual control recovery and measured layout compiler | Build semantic trees, groups, and style data |
| Artwork encoding | T5, INT8 | Encode the artwork prompt in a separate GPU process |
| Artwork generation | Original LTX-Video 2B v0.9, FP16 by default | Generate an image-conditioned visual candidate |
| Code generation | `Qwen/Qwen2.5-Coder-7B-Instruct`, NF4; measured React compiler | Generate shared native panel/form JSX and compile measured scene JSX/CSS |
| Verification | esbuild, Playwright/Chromium, structural and screenshot checks | Check builds, native editing, layout, and reconstruction acceptance |

The GPU order is recognition, prompt encoding, artwork generation, then code
generation. Each worker exits before the next model stage begins. Jobs are queued
sequentially. Inference uses cached models with offline mode enabled by default.

LTX generates artwork, while React renders text and controls. Generated artwork
is packaged as a separate candidate requiring visual review; normal previews keep
source-aligned artwork. LTX-Video 2B v0.9 is a distinct model family from LTX-2.3/2.5.

YOLO is optional and requires compatible UI-trained checkpoints. Generic COCO
checkpoints are excluded from UI detection.

## Generated projects

Each export contains React components, styles, a Vite project, extracted assets,
and design/verification metadata. Important artifacts include:

| Artifact | Purpose |
| --- | --- |
| `design/` | Measured element trees, roles, text, and bounds |
| `interaction_map.json` | Confirmed navigation between captured states |
| `motion_manifest.json` | Isolated artwork motion and sampling information |
| `generated_artwork.json` | LTX candidate metadata and generation profiles |
| `code_generation.json` | Coder stage provenance |
| `verification_report.json` | Build, visual, semantic, and acceptance results |

Local results are stored in `storage/jobs/<job_id>/generated/`. Downloadable
archives are stored in `storage/outputs/`.

After extracting an export, run these commands inside its folder:

```powershell
npm install
npm run dev
# Production build:
npm run build
```

The default preview opens the recovered application screen. Diagnostic state
selection and rendering controls are available with `?inspect=1`. Inspector modes
include hybrid native UI with isolated artwork, source imagery with control
overlays, and an editable approximation. Their fidelity and editability differ.

Forms use native browser validation and emit a bubbling `omniui:submit` event
with field values for integration. They do not call application services. Navigation
between recovered screens requires confirmed mappings in `interaction_map.json`.

## Configuration

See [`.env.example`](.env.example) for the configuration template.

| Setting | Default | Purpose |
| --- | --- | --- |
| `OMNIUI_OFFLINE` | `true` | Use cached recognition weights during inference |
| `OMNIUI_CUDA_DEVICE` | `0` | Select the CUDA device |
| `OMNIUI_CODE_GENERATOR` | `qwen` | Use the coder; `deterministic` skips it for diagnostics |
| `OMNIUI_ARTWORK_GENERATOR` | `ltx-2b-fp16` | Select FP16 artwork generation, `ltx-2b-q8`, or `preserve` |
| `OMNIUI_MAX_UPLOAD_MB` | `512` | Maximum upload size |
| `OMNIUI_MAX_VIDEO_SECONDS` | `600` | Maximum video duration |
| `OMNIUI_MAX_KEY_STATES` | `12` | Selected UI state budget |
| `OMNIUI_MOTION_MAX_SAMPLES` | `80` | Isolated-motion sampling budget |
| `OMNIUI_MOTION_MAX_REGIONS` | `4` | Maximum isolated motion regions |
| `OMNIUI_WORKER_TIMEOUT_SEC` | `600` | Inference and browser verification timeout |
| `OMNIUI_REQUIRE_VISUAL_VERIFICATION` | `true` | Require available browser verification for completion |

To use Q8 artwork, download its matching weights first:

```powershell
.\.venv\Scripts\python.exe tools/setup_ltx.py --precision q8
```

Then set `OMNIUI_ARTWORK_GENERATOR=ltx-2b-q8` in `.env`. Set it to `preserve` to
retain source artwork without running LTX. Restart the service after configuration changes.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /jobs` | Upload a recording and queue reconstruction |
| `GET /jobs/{job_id}` | Read progress, warnings, and reconstruction status |
| `GET /jobs/{job_id}/preview/` | Open a completed generated preview |
| `GET /jobs/{job_id}/download` | Download the exported project |
| `GET /jobs/{job_id}/review/ui` | Open the correction editor |
| `GET /jobs/{job_id}/review` | Read review data |
| `POST /jobs/{job_id}/review` | Build a corrected revision |
| `GET /health` | Inspect service, queue, and GPU status |

`POST /jobs` accepts multipart `video` and `render_mode` fields. Render modes are
`hybrid` (default), `asset-backed`, and `editable`. Refer to `/docs` for schemas.
The service binds to localhost by default.

## Development and validation

Run the test suite after setup:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Unit and API tests isolate storage and stub model inference. Windows cleanup tests
use disposable project copies. GPU generation and visual accuracy require separate
integration checks:

```powershell
# Build and browser-check a synthetic UI without loading models:
.\.venv\Scripts\python.exe tools/smoke_renderer.py

# Exercise the sequential vision, LTX, and coder stack with a reference image:
.\.venv\Scripts\python.exe tools/smoke_ltx.py "path\to\reference.png" --precision fp16

# Reconstruct a complete recording without starting the dashboard server:
.\.venv\Scripts\python.exe tools/regenerate.py "path\to\recording.mp4" --max-states 6
```

Run model integration commands while the server is stopped to avoid competing
GPU workloads. The authored [Forge Hub example](examples/forge-hub/README.md)
demonstrates component and animation techniques; it is not an automatic pipeline benchmark.

## Project structure

```text
app/                 API, job queue, inference workers, reconstruction, and verification
static/              Upload dashboard and review editor
examples/            Authored reference interfaces
models_cache/        Downloaded model and browser files (ignored by Git)
storage/             Uploads, jobs, reports, and exports (ignored by Git)
tests/               Unit, API, regression, and cleanup tests
tools/               Setup, launch, cleanup, evaluation, and browser tooling
setup.bat            Install dependencies and cache models
start.bat            Check and launch the local service
clean.bat            Reset project-local generated data and dependencies
```

## Cleanup and fresh setup

Stop the server and project workers before cleanup. Preview the deletion list:

```powershell
.\clean.bat --dry-run
```

Run cleanup and type `yes` when prompted:

```powershell
.\clean.bat
```

Cleanup permanently removes project-local model/browser downloads, virtual
environments, frontend dependencies, all storage contents, generated builds, caches,
and `.env`. Source code, example source/assets, lockfiles, `.env.example`, and Git
history are preserved. Shared user/system caches are not removed. `--yes` skips
the confirmation prompt.

Reinstall and start:

```powershell
.\setup.bat
.\start.bat
```

## Limitations

- Recognition, OCR, grouping, typography, and responsive behavior are estimates;
  some recordings require manual correction.
- Complex artwork may remain raster assets. Editable 3D meshes and original
  animation code are not recovered from video.
- Source-frame similarity and successful builds alone do not establish native UI
  reconstruction. Image-only output cannot pass reconstruction acceptance.
- LTX may alter visual details; generated candidates need independent review.
- Long clips, scene changes, scrolling, and sampling budgets can limit motion fidelity.
- Hidden application logic, authentication, native OS integration, and backend
  contracts cannot be inferred reliably from a screen recording.
- The exported target is a browser-based React frontend. Other application targets
  require their own integration and packaging.

## License

Project source is available under the [MIT License](LICENSE). Model weights and
third-party dependencies remain subject to their respective licenses.

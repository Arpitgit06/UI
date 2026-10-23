"""Output verification: distinguish tree retention from actual browser fidelity."""
import json
import shutil
import subprocess
import os
from pathlib import Path
from app.config import settings


def _nodes(node):
    return [node] + [n for child in node.children for n in _nodes(child)]


def build_verification_report(layouts, detected_counts=None, generated_files=None, visual_result=None, warnings=None):
    states = []
    for i, layout in enumerate(layouts):
        nodes = _nodes(layout.root)[1:]
        detected = detected_counts[i] if detected_counts is not None else len(nodes)
        states.append({"state_name":layout.state_name,"timestamp_sec":layout.source_frame.timestamp_sec,
            "detected_elements":detected,"layout_nodes":len(nodes),
            "tree_retention_ratio":len(nodes)/max(1,detected),
            "text_nodes":sum(bool(n.text_content) for n in nodes),
            "native_controls":sum(n.tag_hint in {'button','input','select','a'} for n in nodes),
            "vision_verifier_approved":layout.vision_verifier_approved,
            "semantic_status":"approved_by_model" if layout.vision_verifier_approved else "needs_review"})
    visual = visual_result or {"status":"not_run","reason":"Browser verification was not requested"}
    motion=json.loads(generated_files.get('motion_manifest.json','{}')) if generated_files else {}
    return json.dumps({"states":states,"warnings":warnings or [],"output_verification":visual,
        "reconstruction_acceptance":visual.get('reconstruction_acceptance',{'status':'not_assessed','accepted':False}),
        "generated_artwork":json.loads(generated_files.get('generated_artwork.json','{"status":"not_run"}')) if generated_files else {'status':'not_run'},
        "motion_preservation":{"status":motion.get('status','not_run'),'region_count':len(motion.get('regions',[])),
            'effective_sample_hz':motion.get('effective_sample_hz'),'warnings':motion.get('warnings',[])},
        "summary":{"total_states":len(states),"total_detected_elements":sum(s["detected_elements"] for s in states),
            "visual_status":visual["status"],"semantic_reconstruction":"needs_review" if any(not l.vision_verifier_approved for l in layouts) else "model_approved",
            "review_required":bool(warnings) or any(not l.vision_verifier_approved for l in layouts) or visual.get('hybrid_status')!='passed',
            "application_behavior":"unconfirmed unless interaction_map entries have been independently supplied and tested",
            "note":"Tree retention is not visual coverage. Asset-backed screenshot fidelity does not measure editability or application behavior."}},indent=2)


def run_visual_verification(layouts, output_dir):
    output = Path(output_dir)
    runtime = settings.project_root / "tools" / "frontend"
    if not (runtime / "node_modules" / "esbuild").exists() or not shutil.which("node"):
        return {"status":"not_run","reason":"Install the local validator: npm install --prefix tools/frontend and npx --prefix tools/frontend playwright install chromium"}
    request = {"states":[{"id":l.state_name,"width":round(l.root.bbox.width),"height":round(l.root.bbox.height)} for l in layouts]}
    (output / "verification_request.json").write_text(json.dumps(request),encoding="utf-8")
    try:
        result = subprocess.run([shutil.which("node"),str(runtime/"verify.mjs"),str(output.resolve())],
            capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=settings.worker_timeout_sec)
        parsed = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, ValueError) as exc:
        return {"status":"failed","errors":[f"Validator failed: {exc}"]}
    if parsed.get("build") == "passed" and not parsed.get("states") and any("Executable doesn't exist" in e for e in parsed.get("errors",[])):
        return {"status":"not_run","build":"passed","reason":"Chromium is missing; run tools/frontend/node_modules/.bin/playwright install chromium"}
    if parsed.get("status") != "passed":
        return parsed
    import cv2
    from skimage.metrics import structural_similarity
    sources = {l.state_name:l.source_frame.image_path for l in layouts}
    for state in parsed["states"]:
        original = cv2.imread(sources[state["state_name"]])
        rendered = cv2.imread(str(output/state["screenshot"]))
        if original is None or rendered is None or original.shape != rendered.shape:
            state.update({"ssim":None,"visual_pass":False,"reason":"Screenshot dimensions differ"})
            parsed["status"] = "failed"
            continue
        score, difference = structural_similarity(original,rendered,channel_axis=-1,data_range=255,full=True)
        state["ssim"] = round(float(score),5)
        state["visual_pass"] = bool(score >= .97)
        state["comparison_kind"] = {"asset-backed":"preserved_reference_pixels","hybrid":"native_controls_with_preserved_artwork","editable":"editable_reconstruction"}[state["mode"]]
        # An inaccurate editable approximation is reported independently, never
        # hidden behind a perfect preserved-image score.
        if state["mode"] == "asset-backed" and not state["visual_pass"]:
            parsed["status"] = "failed"
        heatmap = cv2.applyColorMap(((1-difference.mean(axis=2)).clip(0,1)*255).astype("uint8"),cv2.COLORMAP_JET)
        diff = state["screenshot"].replace(".png","-diff.png")
        cv2.imwrite(str(output/diff),heatmap)
        state["difference_image"] = diff
    parsed["artwork_status"] = "passed" if all(s.get("visual_pass") for s in parsed["states"] if s["mode"] == "asset-backed") else "failed"
    parsed["editable_status"] = "passed" if all(s.get("visual_pass") for s in parsed["states"] if s["mode"] == "editable") else "approximation_differs"
    hybrids=[s for s in parsed['states'] if s['mode']=='hybrid']
    parsed['hybrid_status']='passed' if hybrids and all(s.get('visual_pass') for s in hybrids) else 'needs_visual_review'
    parsed['layout_finding_count']=sum(len(state.get('layout_findings',[])) for state in parsed['states'] if state['mode']=='hybrid')
    from app.modules.reconstruction_acceptance import assess_reconstruction
    parsed['reconstruction_acceptance']=assess_reconstruction(layouts,parsed)
    generated_manifest=output/'generated_artwork.json'
    if generated_manifest.exists() and json.loads(generated_manifest.read_text()).get('status')=='generated_candidate':
        acceptance=parsed['reconstruction_acceptance']
        acceptance.update({'accepted':False,'status':'needs_review' if acceptance['status']=='accepted' else acceptance['status']})
        acceptance['reasons'].append('Generated artwork requires separate visual review; source screenshots do not validate its fidelity')
    return parsed

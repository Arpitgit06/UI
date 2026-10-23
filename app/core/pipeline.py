"""Sequential video analysis, grounded semantics, hybrid artwork and browser QA.

Only compatible UI detectors are used. Code compilation is deterministic; source
artwork preservation and model consistency are not proof of semantic correctness.
"""
from datetime import datetime, timezone

from app.config import settings
from app.models.schemas import Job, JobStatus
from app.modules import (
    module_a_temporal_parser as module_a,
    module_b_spatial_vision as module_b,
    module_c_dom_synthesizer as module_c,
    module_d_code_generator as module_d,
    verification as verify,
)
from app.utils.logger import get_logger

logger = get_logger("omniui.pipeline")


def _advance(job: Job, status: JobStatus, detail: str = "", progress: float = 0.0) -> None:
    job.status = status
    job.current_stage_detail = detail
    job.progress_percent = progress
    job.updated_at = datetime.now(timezone.utc)
    logger.info(f"[{job.job_id}] -> {status.value} ({detail}) [{progress:.1f}%]")
    from app.core.job_queue import persist_job
    persist_job(job)


async def run_pipeline(job: Job) -> None:
    import asyncio
    import json
    from app.core.process_runner import shutdown_workers, resume_workers
    resume_workers()
    work = settings.jobs_dir / job.job_id
    try:
        _advance(job, JobStatus.PARSING_VIDEO, "scanning the complete video and selecting stable states", 5)
        key_states_dir = work / "key_states"
        states = await module_a.extract_key_states(job.source_video_path,key_states_dir,max_key_states=settings.max_key_states)
        job.key_states_detected = len(states)
        _advance(job, JobStatus.DETECTING_ELEMENTS, "UI detection, spatial OCR fusion and verification",20)
        detections, approvals = await module_b.analyze_states(states,enable_3d=False)
        if len(detections) != len(states) or len(approvals) != len(states):
            raise RuntimeError("Detection/state cardinality mismatch; refusing a partial positional pairing")
        diagnostics_path = key_states_dir / "detection_diagnostics.json"
        if diagnostics_path.exists():
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            job.warnings.extend(diagnostics.get("warnings",[]))
        if not all(approvals):
            job.warnings.append("Some semantic detections need review; image preservation is not semantic verification")
        if not any(detections):
            job.warnings.append("No semantic controls recovered; artwork preview only")
        if job.enable_3d:
            job.warnings.append("3D artwork is preserved as imagery; editable 3D geometry is not inferred")
        _advance(job, JobStatus.SYNTHESIZING_DOM,"building semantic trees and measured styles",50)
        layouts = []
        for state, elements, approved in zip(states,detections,approvals):
            layouts.append(await module_c.synthesize_dom(state,elements,work/"layouts",
                is_3d_scene=False,vision_verifier_approved=approved))
        from app.modules.ltx_assets import generate_artwork
        _advance(job, JobStatus.GENERATING_CODE,"recognition unloaded; preparing artwork generation",60)
        generated_assets,asset_report=await asyncio.to_thread(generate_artwork,layouts,work)
        job.generated_artwork_available=asset_report.get('status')=='generated_candidate'
        if job.generated_artwork_available:job.warnings.append('LTX artwork is a separate generated candidate requiring visual review; normal UI retains source-aligned artwork')
        _advance(job, JobStatus.GENERATING_CODE,"artwork models unloaded; generating native JSX with coder and measured layout compiler",65)
        files = await module_d.generate_code(layouts,fast_mode=job.fast_mode,render_mode=job.render_mode)
        files.update(generated_assets)
        files['generated_artwork.json']=json.dumps(asset_report,indent=2)
        files['pipeline_plan.json']=json.dumps({'scope':'UI reconstruction only; no application backend services',
            'recognition':settings.vision_detector_model_name,'artwork':settings.artwork_generator,
            'gpu_execution':'recognition exits -> prompt encoder exits -> artwork generator exits -> coder exits',
            'code_generation':{'model':settings.local_llm_model_name,'mode':settings.code_generator,
                'scope':'Coder writes shared native JSX; measured layout compiler emits scene JSX/CSS'},
            'verification':'geometry and browser checks; no duplicate verifier model'},indent=2)
        profile_path=key_states_dir/'gpu_profile.json'
        if profile_path.exists():files['recognition_gpu_profile.json']=profile_path.read_text(encoding='utf-8')
        from app.modules.artwork_extractor import extract_motion
        _advance(job,JobStatus.GENERATING_CODE,"isolating artwork motion outside semantic controls",72)
        motion_dir=work / "artwork"
        motion_report=await asyncio.to_thread(extract_motion,job.source_video_path,layouts,motion_dir,
            settings.motion_max_samples,settings.motion_max_regions)
        files["motion_manifest.json"]=json.dumps(motion_report,indent=2)
        from app.modules.scene_plan import configure_scenes
        configure_scenes(files,layouts,job.source_video_path,motion_report)
        for asset in (motion_dir/"assets").rglob("*.webp"):
            files["public/"+asset.relative_to(motion_dir).as_posix()]=asset.read_bytes()
        job.warnings.extend(motion_report["warnings"])
        generated = work / "generated"
        generated.mkdir(parents=True,exist_ok=True)
        files["timeline.json"] = (key_states_dir/"timeline.json").read_text(encoding="utf-8")
        if diagnostics_path.exists():
            files["detection_diagnostics.json"] = diagnostics_path.read_text(encoding="utf-8")
        for name,content in files.items():
            target = (generated/name).resolve()
            if not target.is_relative_to(generated.resolve()):
                raise ValueError("Unsafe output path")
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content if isinstance(content,bytes) else content.encode("utf-8"))
        _advance(job,JobStatus.GENERATING_CODE,"building and comparing source/rendered screenshots",80)
        visual = await asyncio.to_thread(verify.run_visual_verification,layouts,str(generated)) if settings.visual_verification else {"status":"not_run","reason":"Disabled in configuration"}
        (generated/'browser_result.json').write_text(json.dumps(visual,indent=2),encoding='utf-8')
        if visual["status"] == "failed":
            details=visual.get('errors') or [state for state in visual.get('states',[]) if state.get('mode')=='asset-backed' and not state.get('visual_pass')]
            raise RuntimeError("Generated project failed browser/build verification: " + json.dumps(details)[:1500])
        acceptance=visual.get('reconstruction_acceptance',{})
        job.reconstruction_status=acceptance.get('status','not_assessed')
        if not acceptance.get('accepted'):
            job.warnings.append('Reconstruction not accepted: '+ '; '.join(acceptance.get('reasons',['Product acceptance not assessed'])))
        if visual["status"] != "passed":
            if settings.require_visual_verification:
                raise RuntimeError("Visual verification required: " + visual.get("reason","not available"))
            job.warnings.append("Browser verification not completed: " + visual.get("reason","not available"))
        if any(s.get("mode") == "editable" and not s.get("visual_pass") for s in visual.get("states",[])):
            job.warnings.append("Editable approximation differs from the reference; inspect screenshot comparisons")
        if visual.get('hybrid_status')=='needs_visual_review':
            job.warnings.append('Hybrid controls/artwork differ from source; visual review is required')
        if visual.get('layout_finding_count'):
            job.warnings.append(f"{visual['layout_finding_count']} hybrid text/overlap findings need review; inspect the selected nodes in the review editor")
        files["verification_report.json"] = verify.build_verification_report(layouts,[len(d) for d in detections],
            files,visual,job.warnings)
        (generated/"verification_report.json").write_text(files["verification_report.json"],encoding="utf-8")
        for artifact in (generated/"verification").glob("*.png"):
            files[artifact.relative_to(generated).as_posix()] = artifact.read_bytes()
        _advance(job,JobStatus.PACKAGING,"packaging source, assets, diagnostics and comparison images",95)
        job.output_zip_path = str(await module_d.package_output(job.job_id,files))
        _advance(job,JobStatus.COMPLETE,"Draft preview generated; reconstruction not accepted" if job.reconstruction_status!='accepted' else "verified output ready",100)
    finally:
        shutdown_workers()
        module_b._global_ocr_worker.shutdown()

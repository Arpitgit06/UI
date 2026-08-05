"""
Coordinates the full Module A → B → C → D execution flow for a
single job. Deliberately thin: all the actual logic lives in
app/modules/*.

v2: Hybrid parallel pipeline. YOLO + Vision LLM 3B + Vision LLM 7B
    run in Module B regardless of fast_mode. Only Module D differs
    between fast and LLM modes.
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


async def run_pipeline(job: Job) -> None:
    # -----------------------------------------------------------------------
    # Module A: Temporal Parsing
    # -----------------------------------------------------------------------
    _advance(job, JobStatus.PARSING_VIDEO, "extracting key state frames via SSIM diffing", 5.0)
    key_states_dir = settings.jobs_dir / job.job_id / "key_states"
    key_states = await module_a.extract_key_states(job.source_video_path, key_states_dir)
    job.key_states_detected = len(key_states)
    logger.info(f"Module A: {len(key_states)} key states extracted")

    # -----------------------------------------------------------------------
    # Module B: Hybrid Parallel Detection
    # YOLO (CPU) + Vision LLM 3B (detector) + Vision LLM 7B (verifier)
    # + PaddleOCR + Colorgram + Depth — runs IDENTICALLY in fast & LLM mode
    # -----------------------------------------------------------------------
    _advance(
        job, JobStatus.DETECTING_ELEMENTS,
        "YOLO (CPU) + Vision LLM 3B detector + 7B verifier + PaddleOCR + Depth",
        20.0,
    )
    detections_by_state, approval_flags = await module_b.analyze_states(
        key_states, enable_3d=True  # always run depth for z-ordering
    )

    # -----------------------------------------------------------------------
    # Module C: DOM Synthesis
    # is_3d_scene: if user explicitly enabled 3D → force True
    #              otherwise → None (let Module C auto-detect)
    # -----------------------------------------------------------------------
    _advance(job, JobStatus.SYNTHESIZING_DOM, "building parent-child layout tree", 50.0)
    layouts_dir = settings.jobs_dir / job.job_id / "layouts"
    layouts = []
    for i, (state, detections) in enumerate(zip(key_states, detections_by_state)):
        is_3d = True if job.enable_3d else None  # None = auto-detect
        approved = approval_flags[i] if i < len(approval_flags) else False
        
        # Get raw depth variance from the first detection (all detections in a state share it)
        raw_depth_var = None
        if detections:
            raw_depth_var = detections[0].raw_depth_range  # stored per-element but same per-state
        
        layout = await module_c.synthesize_dom(
            state, detections, layouts_dir,
            is_3d_scene=is_3d,
            vision_verifier_approved=approved,
            raw_depth_variance=raw_depth_var,
        )
        layouts.append(layout)

    # -----------------------------------------------------------------------
    # Module D: Code Generation
    # THIS is where fast_mode matters: deterministic Python vs. LLM
    # -----------------------------------------------------------------------
    _advance(job, JobStatus.GENERATING_CODE, "running code generation", 60.0)
    generated_files = await module_d.generate_code(layouts, fast_mode=job.fast_mode)

    # Build verification report
    detected_counts = [len(dets) for dets in detections_by_state]
    generated_files["verification_report.json"] = verify.build_verification_report(
        layouts, detected_counts
    )

    # -----------------------------------------------------------------------
    # Packaging
    # -----------------------------------------------------------------------
    _advance(job, JobStatus.PACKAGING, "zipping project output", 95.0)
    zip_path = await module_d.package_output(job.job_id, generated_files)

    job.output_zip_path = str(zip_path)
    _advance(job, JobStatus.COMPLETE, "done", 100.0)

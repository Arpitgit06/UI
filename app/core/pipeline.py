"""
Coordinates the full Module A -> B -> C -> D execution flow for a
single job. Deliberately thin: all the actual logic lives in
app/modules/*.
"""
from datetime import datetime, timezone

from app.config import settings
from app.models.schemas import Job, JobStatus
from app.modules import (
    module_a_temporal_parser as module_a,
    module_b_spatial_vision as module_b,
    module_b_vision_llm as module_b_fallback,
    module_c_dom_synthesizer as module_c,
    module_d_code_generator as module_d,
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
    _advance(job, JobStatus.PARSING_VIDEO, "extracting key state frames via SSIM diffing", 5.0)
    key_states_dir = settings.jobs_dir / job.job_id / "key_states"
    key_states = await module_a.extract_key_states(job.source_video_path, key_states_dir)
    job.key_states_detected = len(key_states)

    detail = "running YOLOv10 / PaddleOCR"
    if job.enable_3d:
        detail += " / Depth-Anything-V2"
    _advance(job, JobStatus.DETECTING_ELEMENTS, detail, 20.0)
    detections_by_state = await module_b.analyze_states(key_states, enable_3d=job.enable_3d)
    fallback_indices = []
    for i, (state, detections) in enumerate(zip(key_states, detections_by_state)):
        if len([d for d in detections if d.element_type != "text"]) == 0:
            logger.info(f"Fallback: YOLO found no UI elements for {state.image_path}, queueing for Vision LLM.")
            fallback_indices.append(i)

    if fallback_indices:
        _advance(job, JobStatus.DETECTING_ELEMENTS, "running Vision LLM fallback", 40.0)
        fallback_states = [key_states[i] for i in fallback_indices]
        fallback_detections_list = await module_b_fallback.analyze_states_fallback(fallback_states)
        for i, fallback_detections in zip(fallback_indices, fallback_detections_list):
            detections_by_state[i] = fallback_detections

    _advance(job, JobStatus.SYNTHESIZING_DOM, "building parent-child layout tree", 50.0)
    layouts_dir = settings.jobs_dir / job.job_id / "layouts"
    layouts = [
        await module_c.synthesize_dom(state, detections, layouts_dir, is_3d_scene=job.enable_3d)
        for state, detections in zip(key_states, detections_by_state)
    ]

    _advance(job, JobStatus.GENERATING_CODE, "running local 7B LLM code generation", 60.0)
    generated_files = await module_d.generate_code(layouts)

    _advance(job, JobStatus.PACKAGING, "zipping project output", 95.0)
    zip_path = await module_d.package_output(job.job_id, generated_files)

    job.output_zip_path = str(zip_path)
    _advance(job, JobStatus.COMPLETE, "done", 100.0)


"""Retry build/browser verification and packaging without repeating inference."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.core.job_queue import persist_job
from app.models.schemas import Job, JobStatus, LayoutState
from app.modules.verification import run_visual_verification, build_verification_report
from app.modules.module_d_code_generator import package_output


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    args = parser.parse_args()
    work = (settings.jobs_dir / args.job_id).resolve()
    if work.parent != settings.jobs_dir.resolve():
        raise ValueError('Invalid job ID')
    job = Job.model_validate_json((work / 'job.json').read_text(encoding='utf-8'))
    if job.status not in {JobStatus.FAILED, JobStatus.COMPLETE}:
        raise ValueError('Cannot reverify an active job')
    generated = work / 'generated'
    layouts = [LayoutState.model_validate_json(p.read_text(encoding='utf-8'))
               for p in (work / 'layouts').glob('*.json')]
    layouts.sort(key=lambda state: state.source_frame.timestamp_sec)
    if not layouts:
        raise ValueError('No saved layouts to verify')
    visual = await asyncio.to_thread(run_visual_verification, layouts, str(generated))
    (generated / 'browser_result.json').write_text(json.dumps(visual, indent=2), encoding='utf-8')
    if visual['status'] != 'passed':
        raise RuntimeError(json.dumps(visual.get('errors', visual)))
    files = {p.relative_to(generated).as_posix(): p.read_bytes()
             for p in generated.rglob('*') if p.is_file()
             and '.verified-build' not in p.relative_to(generated).parts}
    report_files = {'motion_manifest.json': (generated / 'motion_manifest.json').read_text(encoding='utf-8')}
    if (generated/'generated_artwork.json').exists():report_files['generated_artwork.json']=(generated/'generated_artwork.json').read_text(encoding='utf-8')
    job.generated_artwork_available=json.loads(report_files.get('generated_artwork.json','{}')).get('status')=='generated_candidate'
    job.reconstruction_status=visual.get('reconstruction_acceptance',{}).get('status','not_assessed')
    if visual.get('hybrid_status') == 'needs_visual_review':
        warning = 'Hybrid controls/artwork differ from source; visual review is required'
        if warning not in job.warnings:
            job.warnings.append(warning)
    files['verification_report.json'] = build_verification_report(
        layouts, generated_files=report_files, visual_result=visual, warnings=job.warnings)
    (generated / 'verification_report.json').write_text(files['verification_report.json'], encoding='utf-8')
    job.output_zip_path = str(await package_output(job.job_id, files))
    job.status = JobStatus.COMPLETE
    job.error_message = None
    job.progress_percent = 100
    job.current_stage_detail = 'verified output ready with review notes'
    job.updated_at = datetime.now(timezone.utc)
    persist_job(job)
    print(json.dumps({'job_id': job.job_id, 'status': job.status.value,
                      'interaction_checks': visual.get('interaction_checks', [])}))


if __name__ == '__main__':
    asyncio.run(main())

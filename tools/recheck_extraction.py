"""Rebuild saved real detections after compiler/asset changes, without re-inference.

The report records extraction reuse explicitly. Useful for reviewing exported
designs; not a substitute for independent end-to-end input evaluations.
"""
import asyncio
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.core.job_queue import persist_job
from app.models.schemas import Job,JobStatus,LayoutState,DetectedElement
from app.modules.module_c_dom_synthesizer import synthesize_dom
from app.modules.module_d_code_generator import generate_code,package_output
from app.modules.artwork_extractor import extract_motion
from app.modules.verification import run_visual_verification,build_verification_report


async def recheck(job_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',job_id):raise ValueError('Invalid job ID')
    source=settings.jobs_dir/job_id
    previous=Job.model_validate_json((source/'job.json').read_text())
    now=datetime.now(timezone.utc)
    job=Job(job_id='recheck_'+uuid.uuid4().hex[:8],source_video_path=previous.source_video_path,created_at=now,updated_at=now,render_mode='hybrid')
    work=settings.jobs_dir/job.job_id;generated=work/'generated';layouts=[];counts=[]
    anchors={}
    diagnostics=source/'key_states'/'detection_diagnostics.json'
    diagnostic_states=json.loads(diagnostics.read_text()).get('states',[]) if diagnostics.exists() else []
    def flatten(node):
        result=[]
        for child in node.children:
            if not child.node_id.startswith('geometry-'):
                attributes=dict(child.attributes)
                if child.text_content in anchors:attributes['data-text-bbox']=json.dumps(anchors[child.text_content])
                result.append(DetectedElement(element_id=child.node_id,element_type=child.element_type or 'text',bbox=child.bbox,text_content=child.text_content,styles=child.styles,attributes=attributes,confidence=child.confidence,detection_source='ocr'))
            result.extend(flatten(child))
        return result
    for path in sorted((source/'layouts').glob('*.json'),key=lambda p:int(re.search(r'state_(\d+)',p.name)[1])):
        old=LayoutState.model_validate_json(path.read_text())
        raw=next((s.get('raw_detector',[]) for s in diagnostic_states if s.get('frame_index')==old.source_frame.frame_index),[])
        anchors={d['text']:d['text_bbox'] for d in raw if d.get('text') and d.get('text_bbox')}
        detections=flatten(old.root);counts.append(len(detections))
        layouts.append(await synthesize_dom(old.source_frame,detections,work/'layouts'))
    files=await generate_code(layouts)
    motion=await asyncio.to_thread(extract_motion,job.source_video_path,layouts,work/'artwork')
    files['motion_manifest.json']=json.dumps(motion,indent=2)
    from app.modules.scene_plan import configure_scenes
    configure_scenes(files,layouts,job.source_video_path,motion)
    for asset in (work/'artwork'/'assets').rglob('*.webp'):files['public/'+asset.relative_to(work/'artwork').as_posix()]=asset.read_bytes()
    for name,content in files.items():
        target=generated/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content if isinstance(content,bytes) else content.encode())
    visual=await asyncio.to_thread(run_visual_verification,layouts,str(generated))
    warnings=['Semantic detections require review',f'Reused real extraction from {job_id}; compiler, layout, motion and browser stages rerun']+motion['warnings']
    files['verification_report.json']=build_verification_report(layouts,counts,files,visual,warnings)
    files['extraction_provenance.json']=json.dumps({'reused_from':job_id,'stages_rerun':['layout','compiler','artwork','browser','package']},indent=2)
    for name in ['verification_report.json','extraction_provenance.json']:(generated/name).write_text(files[name],encoding='utf-8')
    for asset in (generated/'verification').glob('*.png'):files[asset.relative_to(generated).as_posix()]=asset.read_bytes()
    job.status=JobStatus.COMPLETE if visual['status']=='passed' else JobStatus.FAILED;job.warnings=warnings
    job.reconstruction_status=visual.get('reconstruction_acceptance',{}).get('status','not_assessed')
    if job.status==JobStatus.COMPLETE:job.output_zip_path=str(await package_output(job.job_id,files))
    else:job.error_message=json.dumps(visual)
    persist_job(job)
    summary={'source_job':job_id,'job_id':job.job_id,'status':job.status.value,'motion':motion['status'],'regions':len(motion['regions']),'visual':[(s['mode'],s.get('ssim')) for s in visual.get('states',[])]}
    print(json.dumps(summary),flush=True);return summary


async def main():
    results=[]
    for job_id in sys.argv[1:]:results.append(await recheck(job_id))
    target=settings.jobs_dir/'generalization_evaluation'/'rechecked.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(r['status']!='complete' for r in results):raise SystemExit(1)


if __name__=='__main__':asyncio.run(main())

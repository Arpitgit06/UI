"""Validated human corrections and confirmed UI transitions, without re-inference."""
import asyncio
from datetime import datetime,timezone
import json
from typing import Literal
import uuid

from pydantic import BaseModel,Field,ConfigDict

from app.config import settings
from app.core.job_queue import persist_job
from app.models.schemas import BoundingBox,Job,JobStatus,LayoutState
from app.modules.module_c_dom_synthesizer import _tag_hint_for
from app.modules.module_d_code_generator import generate_code,package_output
from app.modules.artwork_extractor import extract_motion
from app.modules.verification import run_visual_verification,build_verification_report


Role=Literal['text','button','input','checkbox','radio','slider','dropdown','link','image','icon','container']


class NodeCorrection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    state_id:str=Field(min_length=1,max_length=100)
    node_id:str=Field(min_length=1,max_length=150)
    element_type:Role|None=None
    text_content:str|None=Field(default=None,max_length=4000)
    bbox:BoundingBox|None=None
    input_type:Literal['text','email','password','search','number','date']|None=None
    font_size:float|None=Field(default=None,ge=8,le=256,allow_inf_nan=False)
    color:str|None=Field(default=None,pattern=r'^#[a-fA-F0-9]{6}$')


class ConfirmedTransition(BaseModel):
    model_config=ConfigDict(extra='forbid')
    state_id:str=Field(min_length=1,max_length=100)
    node_id:str=Field(min_length=1,max_length=150)
    target_state:str=Field(min_length=1,max_length=100)


class ReviewRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    corrections:list[NodeCorrection]=Field(default_factory=list,max_length=500)
    transitions:list[ConfirmedTransition]=Field(default_factory=list,max_length=500)
    clear_transitions:bool=False
    exclude_motion_regions:list[str]=Field(default_factory=list,max_length=8)
    note:str=Field(default='',max_length=2000)


def nodes(root):
    return list(root.children)+[descendant for child in root.children for descendant in nodes(child)]


def load_layouts(job_id):
    directory=settings.jobs_dir/job_id/'generated'/'design'
    paths=list(directory.glob('*.json'))
    if not paths:paths=list((settings.jobs_dir/job_id/'layouts').glob('*_layout.json'))
    layouts=[LayoutState.model_validate_json(path.read_text(encoding='utf-8')) for path in paths]
    if not layouts:raise ValueError('This job has no editable design records')
    return sorted(layouts,key=lambda layout:layout.source_frame.timestamp_sec)


def prepare_review(layouts,request,existing_actions=None):
    """Validate everything before producing a new independent artifact."""
    layouts=[layout.model_copy(deep=True) for layout in layouts]
    states={layout.state_name:layout for layout in layouts}
    if len(states)!=len(layouts):raise ValueError('Duplicate state IDs')
    index={state:{node.node_id:node for node in nodes(layout.root)} for state,layout in states.items()}
    if any(len(index[state])!=len(nodes(layout.root)) for state,layout in states.items()):raise ValueError('Duplicate node IDs in design')
    touched=set()
    for correction in request.corrections:
        key=(correction.state_id,correction.node_id)
        if key in touched:raise ValueError('Duplicate correction for a node')
        touched.add(key)
        if correction.node_id not in index.get(correction.state_id,{}):raise ValueError('Correction references an unknown node')
        node=index[correction.state_id][correction.node_id];layout=states[correction.state_id]
        if correction.bbox is not None:
            b=correction.bbox;root=layout.root.bbox
            if b.width<=0 or b.height<=0 or b.x<root.x or b.y<root.y or b.x+b.width>root.x+root.width or b.y+b.height>root.y+root.height:raise ValueError('Corrected bounds must stay inside the captured frame')
            if node.text_content or node.tag_hint in {'button','input','select','a'}:
                node.attributes.setdefault('data-original-semantic-bbox',json.dumps([node.bbox.x,node.bbox.y,node.bbox.width,node.bbox.height]))
            node.bbox=b;node.attributes.pop('data-text-bbox',None)
        if correction.element_type is not None:
            if node.children and correction.element_type not in {'container'}:raise ValueError('Move child nodes before changing a container into a leaf/control')
            if correction.element_type=='text' and correction.bbox is None and node.attributes.get('data-text-bbox'):
                text_bounds=json.loads(node.attributes['data-text-bbox'])
                restored=BoundingBox(x=text_bounds[0],y=text_bounds[1],width=text_bounds[2],height=text_bounds[3])
                frame=layout.root.bbox
                if restored.x<frame.x or restored.y<frame.y or restored.x+restored.width>frame.x+frame.width or restored.y+restored.height>frame.y+frame.height:raise ValueError('Original text anchor is outside the frame')
                node.bbox=restored
            node.element_type=correction.element_type;node.tag_hint=_tag_hint_for(correction.element_type)
            node.attributes.pop('type',None)
            if node.tag_hint=='input':node.attributes['type']={'checkbox':'checkbox','radio':'radio','slider':'range'}.get(correction.element_type,'text')
        if 'text_content' in correction.model_fields_set:
            if correction.text_content!=node.text_content:node.attributes.pop('data-text-bbox',None)
            node.text_content=correction.text_content
        if correction.input_type is not None:
            if node.tag_hint!='input' or node.element_type in {'checkbox','radio','slider'}:raise ValueError('Input type correction requires a text-like input')
            node.attributes['type']=correction.input_type
        if correction.font_size is not None:node.styles['font-size']=f'{correction.font_size:g}px'
        if correction.color is not None:node.styles['color']=correction.color
    actions={} if request.clear_transitions else json.loads(json.dumps(existing_actions or {}))
    if not isinstance(actions,dict) or any(not isinstance(value,dict) for value in actions.values()):raise ValueError('Existing interaction map must contain state/node mappings')
    transition_keys=set()
    for transition in request.transitions:
        transition_key=(transition.state_id,transition.node_id)
        if transition_key in transition_keys:raise ValueError('Duplicate confirmed transition for a control')
        transition_keys.add(transition_key)
        if transition.target_state not in states:raise ValueError('Transition references an unknown destination state')
        node=index.get(transition.state_id,{}).get(transition.node_id)
        if node is None or node.tag_hint not in {'button','a'}:raise ValueError('Transition must originate from a button or link')
        actions.setdefault(transition.state_id,{})[transition.node_id]=transition.target_state
    # Also reject stale previously confirmed mappings invalidated by a role edit.
    for state,mappings in actions.items():
        for node_id,target in mappings.items():
            node=index.get(state,{}).get(node_id)
            if not node or node.tag_hint not in {'button','a'} or target not in states:raise ValueError('Existing transition is incompatible with the corrected design')
    return layouts,actions


def review_data(job):
    layouts=load_layouts(job.job_id);generated=settings.jobs_dir/job.job_id/'generated'
    report_path=generated/'verification_report.json'
    report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    manifest_path=generated/'motion_manifest.json'
    motion=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'regions':[]}
    action_path=generated/'interaction_map.json'
    actions=json.loads(action_path.read_text(encoding='utf-8')) if action_path.exists() else {}
    return {'job_id':job.job_id,'states':[{'id':layout.state_name,'time':layout.source_frame.timestamp_sec,'size':[layout.root.bbox.width,layout.root.bbox.height],
        'frame_url':f'/jobs/{job.job_id}/review/frame/{layout.state_name}',
        'nodes':[{'id':node.node_id,'role':node.element_type,'tag':node.tag_hint,'text':node.text_content,'bbox':node.bbox.model_dump(),'attributes':node.attributes,'styles':node.styles,'confidence':node.confidence} for node in nodes(layout.root)]} for layout in layouts],
        'actions':actions,'motion':motion,'verification':report,'warnings':job.warnings}


async def build_review(parent,child,layouts,actions,request):
    work=settings.jobs_dir/child.job_id;generated=work/'generated'
    try:
        child.status=JobStatus.GENERATING_CODE;child.current_stage_detail='Compiling reviewed controls and confirmed transitions';persist_job(child)
        files=await generate_code(layouts,render_mode=child.render_mode)
        files['interaction_map.json']=json.dumps(actions,indent=2)
        motion=await asyncio.to_thread(extract_motion,parent.source_video_path,layouts,work/'artwork',settings.motion_max_samples,settings.motion_max_regions)
        excluded=set(request.exclude_motion_regions)
        # Exclusions refer to the parent's reviewed geometry, not re-numbered
        # extraction IDs: omit any new region intersecting an excluded parent box.
        parent_manifest=settings.jobs_dir/parent.job_id/'generated'/'motion_manifest.json'
        old=json.loads(parent_manifest.read_text()) if parent_manifest.exists() else {'regions':[]}
        boxes=[r['bbox'] for r in old['regions'] if r['id'] in excluded]
        def intersects(region):
            x,y,w,h=region['bbox']
            return any(min(x+w,bx+bw)>max(x,bx) and min(y+h,by+bh)>max(y,by) for bx,by,bw,bh in boxes)
        motion['regions']=[r for r in motion['regions'] if not intersects(r)]
        if excluded:
            motion['excluded_parent_regions']=sorted(excluded)
            motion['warnings'].append('Human review excluded artwork regions intersecting selected parent regions')
            if not motion['regions']:motion['status']='excluded_by_review'
        files['motion_manifest.json']=json.dumps(motion,indent=2)
        from app.modules.scene_plan import configure_scenes
        configure_scenes(files,layouts,parent.source_video_path,motion)
        for region in motion['regions']:files['public/'+region['src']]=(work/'artwork'/region['src']).read_bytes()
        files['human_review.json']=json.dumps({'source_job':parent.job_id,'scope':'explicit corrections and confirmed UI transitions only','request':request.model_dump(mode='json',exclude_unset=True)},indent=2)
        for name,content in files.items():
            target=(generated/name).resolve()
            if not target.is_relative_to(generated.resolve()):raise ValueError('Unsafe compiler output path')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content if isinstance(content,bytes) else content.encode('utf-8'))
        child.current_stage_detail='Browser-checking reviewed output';persist_job(child)
        visual=await asyncio.to_thread(run_visual_verification,layouts,str(generated))
        (generated/'browser_result.json').write_text(json.dumps(visual,indent=2),encoding='utf-8')
        if visual['status']!='passed':raise ValueError('Reviewed output failed browser verification; inspect browser_result.json')
        child.reconstruction_status=visual.get('reconstruction_acceptance',{}).get('status','not_assessed')
        child.warnings=['Human review covers supplied corrections only; unreviewed semantics and hidden application behavior remain unconfirmed']+motion['warnings']
        if visual.get('hybrid_status')!='passed':child.warnings.append('Hybrid output still differs from the source; visual review remains necessary')
        if visual.get('layout_finding_count'):child.warnings.append(f"{visual['layout_finding_count']} text/overlap findings remain in hybrid mode")
        files['verification_report.json']=build_verification_report(layouts,None,files,visual,child.warnings)
        (generated/'verification_report.json').write_text(files['verification_report.json'],encoding='utf-8')
        for artifact in (generated/'verification').glob('*.png'):files[artifact.relative_to(generated).as_posix()]=artifact.read_bytes()
        child.output_zip_path=str(await package_output(child.job_id,files));child.status=JobStatus.COMPLETE
        child.current_stage_detail='Reviewed output ready with remaining review notes';child.progress_percent=100
    except Exception as exc:
        child.status=JobStatus.FAILED;child.error_message=str(exc)
    finally:
        child.updated_at=datetime.now(timezone.utc);persist_job(child)


def make_review_job(parent):
    now=datetime.now(timezone.utc)
    return Job(job_id='review_'+uuid.uuid4().hex[:12],source_video_path=parent.source_video_path,created_at=now,updated_at=now,render_mode='hybrid',status=JobStatus.GENERATING_CODE)

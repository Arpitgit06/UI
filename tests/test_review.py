import asyncio
from datetime import datetime,timezone
import json

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.config import settings
from app.core import review
from app.core.job_queue import persist_job
from app.main import app
from app.models.schemas import BoundingBox,DOMNode,Job,JobStatus,KeyStateFrame,LayoutState
from app.modules import module_d_code_generator as compiler
from app.modules.artwork_extractor import semantic_mask


def example():
    root=DOMNode(node_id='root',tag_hint='div',bbox=BoundingBox(x=0,y=0,width=400,height=250),children=[DOMNode(node_id='open',tag_hint='button',element_type='button',bbox=BoundingBox(x=20,y=40,width=120,height=35),text_content='Open')])
    first=LayoutState(state_name='state_0',source_frame=KeyStateFrame(frame_index=0,timestamp_sec=0,image_path='unused'),root=root)
    second=first.model_copy(deep=True);second.state_name='state_12';second.source_frame.timestamp_sec=1;second.source_frame.frame_index=12
    return [first,second]


def test_review_is_independent_and_validates_navigation():
    layouts=example()
    request=review.ReviewRequest(corrections=[review.NodeCorrection(state_id='state_0',node_id='open',text_content='Continue',font_size=18)],transitions=[review.ConfirmedTransition(state_id='state_0',node_id='open',target_state='state_12')])
    result,actions=review.prepare_review(layouts,request)
    assert layouts[0].root.children[0].text_content=='Open'
    assert result[0].root.children[0].text_content=='Continue'
    assert actions=={'state_0':{'open':'state_12'}}


@pytest.mark.parametrize('payload',[
    {'corrections':[{'state_id':'unknown','node_id':'open','text_content':'x'}]},
    {'corrections':[{'state_id':'state_0','node_id':'open','bbox':{'x':390,'y':0,'width':50,'height':20}}]},
    {'transitions':[{'state_id':'state_0','node_id':'open','target_state':'missing'}]},
    {'corrections':[{'state_id':'state_0','node_id':'open','element_type':'text'}],'transitions':[{'state_id':'state_0','node_id':'open','target_state':'state_12'}]},
])
def test_invalid_review_fails_before_build(payload):
    with pytest.raises(ValueError):review.prepare_review(example(),review.ReviewRequest.model_validate(payload))


def test_stale_transition_must_be_explicitly_cleared():
    request=review.ReviewRequest(corrections=[review.NodeCorrection(state_id='state_0',node_id='open',element_type='text')])
    with pytest.raises(ValueError,match='Existing transition'):review.prepare_review(example(),request,{'state_0':{'open':'state_12'}})
    request.clear_transitions=True
    assert review.prepare_review(example(),request,{'state_0':{'open':'state_12'}})[1]=={}


def test_selected_render_mode_reaches_compiler(monkeypatch):
    monkeypatch.setattr(compiler,'_generate_code_sync',lambda layouts,fast_mode,**kwargs:kwargs['render_mode'])
    assert asyncio.run(compiler.generate_code([],render_mode='editable'))=='editable'


def test_moved_control_masks_original_and_reviewed_positions():
    request=review.ReviewRequest(corrections=[review.NodeCorrection(state_id='state_0',node_id='open',bbox=BoundingBox(x=240,y=40,width=120,height=35))])
    layouts,_=review.prepare_review(example(),request)
    mask=semantic_mask(layouts[0],(400,250))
    assert mask[50,50]==255 and mask[50,260]==255
    assert mask[50,200]==0


def test_reclassified_text_uses_original_glyph_anchor():
    layouts=example();layouts[0].root.children[0].attributes['data-text-bbox']='[30,45,60,20]'
    request=review.ReviewRequest(corrections=[review.NodeCorrection(state_id='state_0',node_id='open',element_type='text')])
    corrected,_=review.prepare_review(layouts,request)
    assert corrected[0].root.children[0].bbox==BoundingBox(x=30,y=45,width=60,height=20)


def test_uncued_ocr_noun_does_not_become_dropdown():
    from app.modules.module_b_spatial_vision import _supported_role
    assert _supported_role('dropdown',{'source':'ocr','text':'AppFlowy'})=='text'
    assert _supported_role('dropdown',{'source':'ocr','text':'Choose country'})=='dropdown'
    assert _supported_role('button',{'source':'ocr','text':'AppFlowy'})=='text'
    assert _supported_role('button',{'source':'ocr','text':'Execute intent'})=='button'


def test_review_api_builds_separate_artifact(tmp_path,monkeypatch):
    import app.main as main
    now=datetime.now(timezone.utc)
    parent=Job(job_id='source_review',status=JobStatus.COMPLETE,created_at=now,updated_at=now,source_video_path='unused')
    work=settings.jobs_dir/parent.job_id;design=work/'generated'/'design';design.mkdir(parents=True)
    image=work/'frame.png';Image.new('RGB',(400,250),'white').save(image)
    parent_motion={'regions':[{'id':'artwork-0','bbox':[180,50,80,80],'src':'assets/motion/a.webp'}],'sample_times':[0,1],'source_size':[400,250],'warnings':[],'status':'extracted'}
    (work/'generated'/'motion_manifest.json').write_text(json.dumps(parent_motion))
    for layout in example():
        layout.source_frame.image_path=str(image)
        (design/f'{layout.state_name}.json').write_text(layout.model_dump_json(),encoding='utf-8')
    persist_job(parent)
    monkeypatch.setattr(review,'extract_motion',lambda *args:json.loads(json.dumps(parent_motion)))
    monkeypatch.setattr(review,'run_visual_verification',lambda *args:{'status':'passed','states':[],'hybrid_status':'needs_visual_review'})
    with TestClient(app) as client:
        assert client.get('/jobs/source_review/review').status_code==200
        assert client.get('/jobs/source_review/review/frame/state_0').status_code==200
        invalid=client.post('/jobs/source_review/review',json={'transitions':[{'state_id':'state_0','node_id':'open','target_state':'bad'}]})
        assert invalid.status_code==422
        assert client.post('/jobs/source_review/review',json={'exclude_motion_regions':['missing']}).status_code==422
        result=client.post('/jobs/source_review/review',json={'corrections':[{'state_id':'state_0','node_id':'open','text_content':'Continue'}],'transitions':[{'state_id':'state_0','node_id':'open','target_state':'state_12'}],'exclude_motion_regions':['artwork-0']})
        assert result.status_code==202
        child=client.get('/jobs/'+result.json()['job_id']).json()
        assert child['status']=='complete'
        files=settings.jobs_dir/child['job_id']/'generated'
        assert json.loads((files/'interaction_map.json').read_text())=={'state_0':{'open':'state_12'}}
        assert json.loads((design/'state_0.json').read_text())['root']['children'][0]['text_content']=='Open'
        assert json.loads((files/'design'/'state_0.json').read_text())['root']['children'][0]['text_content']=='Continue'
        assert (files/'human_review.json').exists()
        assert json.loads((files/'motion_manifest.json').read_text())['regions']==[]
        assert json.loads((work/'generated'/'motion_manifest.json').read_text())['regions']==parent_motion['regions']
        assert not main.active_reviews

from PIL import Image,ImageDraw
from app.models.schemas import BoundingBox,DetectedElement,DOMNode,LayoutState,KeyStateFrame
from app.modules.visual_groups import augment_visual_groups
from app.modules.scene_plan import application_entry,configure_scenes
from app.modules.reconstruction_acceptance import assess_reconstruction


def layout(index,controls=False):
    root=DOMNode(node_id='root',tag_hint='div',bbox=BoundingBox(x=0,y=0,width=400,height=300))
    if controls:
        root.children=[DOMNode(node_id=tag,tag_hint=tag,bbox=BoundingBox(x=10,y=10,width=100,height=30)) for tag in ['input','button']]
    return LayoutState(state_name=f'state_{index}',source_frame=KeyStateFrame(frame_index=index,timestamp_sec=index,image_path='unused'),root=root)


def test_context_requires_border_and_adjacent_action():
    image=Image.new('RGB',(640,300),'#101820');d=ImageDraw.Draw(image)
    field=DetectedElement(element_id='f',element_type='text',text_content='Paste a website link',bbox=BoundingBox(x=50,y=110,width=230,height=20))
    action=DetectedElement(element_id='a',element_type='text',text_content='Inspect link',bbox=BoundingBox(x=380,y=110,width=100,height=20))
    assert all(e.element_type=='text' for e in augment_visual_groups(image,[field,action]))
    d.rectangle((35,95,340,150),outline='white',width=2);d.rectangle((365,95,500,150),outline='white',width=2)
    result=augment_visual_groups(image,[field,action])
    assert next(e for e in result if e.element_id=='f').element_type=='input'
    assert next(e for e in result if e.element_id=='a').element_type=='button'


def test_intro_ends_at_first_contiguous_final_application_screen(tmp_path):
    states=[layout(0),layout(2),layout(8,True),layout(9,True)]
    source=tmp_path/'source.mp4';source.write_bytes(b'test fixture')
    files={};plan=configure_scenes(files,states,source,{'status':'review_required'})
    assert application_entry(states)==2
    assert plan['entry_state']=='state_8' and plan['intro']['end']==8
    assert 'public/assets/intro-source.mp4' in files


def test_initial_usable_interface_is_not_forced_into_intro(tmp_path):
    states=[layout(0,True),layout(1,True)]
    assert configure_scenes({},states,tmp_path/'absent.mp4',{})['intro'] is None


def test_context_refines_previously_expanded_controls_using_original_text():
    import json
    image=Image.new('RGB',(640,300),'#101820');d=ImageDraw.Draw(image)
    d.rectangle((35,95,340,150),outline='white',width=2);d.rectangle((365,95,500,150),outline='white',width=2)
    panel=BoundingBox(x=30,y=90,width=480,height=70)
    field=DetectedElement(element_id='f',element_type='input',text_content='Paste a website link',bbox=panel,attributes={'data-text-bbox':json.dumps([50,110,230,20])})
    action=DetectedElement(element_id='a',element_type='button',text_content='Inspect link',bbox=panel,attributes={'data-text-bbox':json.dumps([380,110,100,20])})
    result=augment_visual_groups(image,[field,action])
    f=next(e for e in result if e.element_id=='f');a=next(e for e in result if e.element_id=='a')
    assert f.bbox.x+f.bbox.width<=a.bbox.x
    assert f.attributes['type']=='url'


def test_perfect_screenshot_cannot_accept_image_only_output():
    result=assess_reconstruction([layout(0)],{'hybrid_status':'passed','states':[{'ssim':1.0}]})
    assert result['status']=='preview_only' and result['accepted'] is False


def test_unreviewed_native_form_is_not_certified_frontend():
    result=assess_reconstruction([layout(0,True)],{'hybrid_status':'passed','interaction_checks':[{'check':'native_form_validation_and_local_submission'}]})
    assert result['status']=='needs_review' and not result['accepted']


def test_reviewed_frontend_with_visual_and_interaction_checks_can_pass():
    state=layout(0,True);state.vision_verifier_approved=True
    result=assess_reconstruction([state],{'hybrid_status':'passed','interaction_checks':[{'check':'native_form_validation_and_local_submission'}]})
    assert result['accepted'] and result['status']=='accepted'

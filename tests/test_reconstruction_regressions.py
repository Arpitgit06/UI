import asyncio
import json
import numpy as np
import pytest
from PIL import Image

from app.models.schemas import BoundingBox, DetectedElement, DOMNode, KeyStateFrame
from app.modules.ocr_adapter import _parse_paddleocr_result
from app.modules.module_a_temporal_parser import _extract_key_states_sync
from app.modules.module_b_spatial_vision import _merge_detections, _run_vision_verifier, _run_vision_detector, _valid_detection
from app.modules.module_c_dom_synthesizer import _assign_parents, _detect_3d_scene
from app.modules import module_b_spatial_vision as vision, module_d_code_generator as code
from app.modules.verification import build_verification_report
from tests.test_module_a import _make_frame, _draw_square, _write_video


def test_budget_covers_end_of_video(tmp_path):
    frames = []
    for i in range(12):
        frames += [_draw_square(_make_frame(),30+i*20,90,30,(0,200,0))]*10
    path = tmp_path/"clip.mp4"
    _write_video(path,frames)
    states = _extract_key_states_sync(str(path),tmp_path/"frames",max_key_states=3)
    timeline = json.loads((tmp_path/"frames/timeline.json").read_text())
    assert timeline["frames_scanned"] == 120
    assert timeline["omitted_candidates"] > 0
    assert len(states) == 3 and states[-1].timestamp_sec > 3.5


def test_paddle_arrays_and_real_plural_keys():
    result = {"rec_texts":["Save"],"rec_scores":np.array([.95]),
              "rec_polys":np.array([[[10,20],[80,20],[80,40],[10,40]]])}
    assert _parse_paddleocr_result(result)[0]["bbox"] == (10,20,70,20)


def test_iou_fuses_without_id():
    result = _merge_detections([{"bbox":(10,10,100,40),"label":"button","confidence":.9}],
        [{"bbox":[11,10,100,40],"type":"button","text":"Save"}])
    assert len(result) == 1 and result[0]["text"] == "Save"


@pytest.mark.parametrize("raw",['not json','[]','{}','{"approved":"true"}'])
def test_verifier_never_approves_invalid_result(raw,monkeypatch):
    monkeypatch.setattr(vision,"_run_vision_subprocess",lambda *a:[raw])
    assert _run_vision_verifier([KeyStateFrame(frame_index=0,timestamp_sec=0,image_path="unused.png")],[[{"type":"button","bbox":[0,0,10,10]}]])[1] == [False]


def test_retired_verifier_preserves_detections_without_inference(monkeypatch):
    monkeypatch.setattr(vision,"_run_vision_subprocess",lambda *a:[json.dumps({"approved":False,"deleted_ids":[0]})])
    result,approved = _run_vision_verifier([KeyStateFrame(frame_index=0,timestamp_sec=0,image_path="unused.png")],[[{"bbox":[0,0,10,10],"type":"button"}]])
    assert result == [[{"bbox":[0,0,10,10],"type":"button"}]] and approved == [False]


def test_text_cannot_own_other_text():
    a = DetectedElement(element_id="a",element_type="text",bbox=BoundingBox(x=0,y=0,width=100,height=100))
    b = DetectedElement(element_id="b",element_type="text",bbox=BoundingBox(x=10,y=10,width=20,height=20))
    assert _assign_parents([a,b],.8) == {"a":None,"b":None}
    a.bbox.z_index=0; b.bbox.z_index=100
    assert not _detect_3d_scene([a,b],10000)


def test_repeated_ocr_words_in_different_locations_survive(tmp_path,monkeypatch):
    path=tmp_path/"screen.png"; Image.new("RGB",(400,100),"white").save(path)
    monkeypatch.setattr(vision,"_run_vision_detector",lambda *a:[[{"type":"text","text":"Save","bbox":[10,10,50,20]}]])
    monkeypatch.setattr(vision._global_ocr_worker,"analyze",lambda *a:[[{"text":"Save","confidence":.9,"bbox":[200,10,50,20]}]])
    result,_=vision._analyze_states_sync([KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(path))])
    assert len(result[0]) == 2


def test_dense_screen_is_not_capped(tmp_path,monkeypatch):
    path=tmp_path/"screen.png"; Image.new("RGB",(1000,200),"white").save(path)
    monkeypatch.setattr(vision,"_run_vision_detector",lambda *a:[[{"type":"button","bbox":[i*10,0,8,20]} for i in range(80)]])
    result,_=vision._analyze_states_sync([KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(path))])
    assert len(result[0]) == 80


def test_invalid_coordinates_cannot_enter_layout():
    assert _valid_detection({"bbox":[0,0,float('nan'),20]}) is None
    assert _valid_detection({"bbox":[200,0,20,20]},(100,100)) is None


def test_control_bounds_snap_to_border_instead_of_label():
    from app.modules.visual_styles import snap_control_bbox
    assert snap_control_bbox((120,110,100,30),[(100,100,200,60),(0,0,640,360)],(640,360)) == (100,100,200,60)
    assert snap_control_bbox((120,110,100,30),[],(640,360)) == (120,110,100,30)


def test_classifier_cannot_invent_text_or_coordinates(tmp_path,monkeypatch):
    path=tmp_path/"screen.png"; Image.new("RGB",(400,200),"white").save(path)
    frame=KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(path))
    monkeypatch.setattr(vision,"_run_vision_subprocess",lambda *a:['{"0":"input"}'])
    results=_run_vision_detector([frame],[[]],[[{"bbox":[20,30,100,20],"text":"Enter query","confidence":.9}]])
    assert results[0][0]["type"] == "input"
    assert results[0][0]["text"] == "Enter query"
    assert results[0][0]["bbox"] == (12,22,116,36)
    monkeypatch.setattr(vision,"_run_vision_subprocess",lambda *a:['{"0":{"type":"button","bbox":[900,900,200,200]}}'])
    results=_run_vision_detector([frame],[[]],[[{"bbox":[20,30,100,20],"text":"Heading"}]])
    assert results[0][0]["type"] == "text"
    assert results[0][0]["bbox"] == [20,30,100,20]


def test_ocr_cache_is_bound_to_source_pixels(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path=tmp_path/"screen.png"; Image.new("RGB",(40,30),"white").save(path)
    calls=[]
    def worker(*args,**kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=0,stderr="",stdout='__OCR_JSON_START__'+json.dumps({"status":"success","detections":[[{"text":"Save","bbox":[1,1,10,10],"confidence":.9}]]})+'__OCR_JSON_END__')
    monkeypatch.setattr(vision,"run_worker",worker)
    ocr=vision.PersistentOCRWorker()
    assert ocr.analyze([str(path)],"en",.5) == ocr.analyze([str(path)],"en",.5)
    assert len(calls) == 1
    Image.new("RGB",(40,30),"black").save(path)
    ocr.analyze([str(path)],"en",.5)
    assert len(calls) == 2


def test_assets_safe_jsx_and_truthful_report(tmp_path):
    from app.modules.module_c_dom_synthesizer import _synthesize_dom_sync
    path=tmp_path/"screen.png"; Image.new("RGB",(400,200),"black").save(path)
    frame=KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(path))
    elements=[DetectedElement(element_id="input",element_type="input",text_content='Hello {name} < Back',
                              bbox=BoundingBox(x=20,y=20,width=150,height=30))]
    layout=_synthesize_dom_sync(frame,elements,is_3d_scene=False)
    files=asyncio.run(code.generate_code([layout]))
    assert 'placeholder={"Hello {name} < Back"}' in files['State0.jsx']
    assert '</input>' not in files['State0.jsx']
    assert isinstance(files['public/assets/state_0-reference.png'],bytes)
    report=json.loads(build_verification_report([layout],[1],files))
    assert report['summary']['visual_status'] == 'not_run'
    assert 'coverage_score' not in report['states'][0]

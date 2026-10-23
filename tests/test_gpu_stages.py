import json
from types import SimpleNamespace
from PIL import Image
import pytest
from app.core.model_budget import configure_cuda_stage,configure_recognition
from app.modules import ltx_assets
from app.models.schemas import DOMNode,BoundingBox,LayoutState,KeyStateFrame


class CUDA:
    def __init__(self,free=7*1024**3):self.free=free;self.fraction=None
    def is_available(self):return True
    def mem_get_info(self,index):return self.free,8*1024**3
    def set_device(self,index):pass
    def set_per_process_memory_fraction(self,fraction,index):self.fraction=fraction
    def reset_peak_memory_stats(self,index):pass


def test_budget_leaves_driver_headroom_and_stays_below_eight_gib():
    cuda=CUDA();profile=configure_cuda_stage(SimpleNamespace(cuda=cuda),'generation')
    assert profile['allocator_budget_bytes']==6.5*1024**3
    assert cuda.fraction<1


def test_large_recognition_model_is_not_automatically_loaded():
    with pytest.raises(ValueError):configure_recognition(SimpleNamespace(cuda=CUDA()),'Lightricks/LTX-2.5')


def test_low_memory_refuses_generation():
    with pytest.raises(RuntimeError):configure_cuda_stage(SimpleNamespace(cuda=CUDA(2*1024**3)),'generation')


def test_encoder_exits_before_generation_and_art_is_separate_from_ui(tmp_path,monkeypatch):
    monkeypatch.setattr(ltx_assets.settings,'artwork_generator','ltx-2b-fp16')
    image=tmp_path/'frame.png';Image.new('RGB',(320,192),'navy').save(image)
    layout=LayoutState(state_name='state_0',source_frame=KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(image)),
                       root=DOMNode(node_id='root',tag_hint='div',bbox=BoundingBox(x=0,y=0,width=320,height=192)))
    phases=[]
    def worker(command,**kw):
        phase=command[-1];request=json.loads(open(command[-2]).read());out=tmp_path/'ltx'
        assert request['precision']=='fp16'
        if phase=='generate':assert phases==['encode']
        phases.append(phase);(out/f'{phase}-profile.json').write_text('{}')
        if phase=='generate':(out/'generated.mp4').write_bytes(b'fixture')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ltx_assets,'run_worker',worker)
    files,report=ltx_assets.generate_artwork([layout],tmp_path)
    assert phases==['encode','generate']
    assert report['editable_ui'] is False and report['requires_visual_review']
    assert files['public/assets/generated-artwork.mp4']==b'fixture'


def test_source_preservation_does_not_load_generation_models(tmp_path,monkeypatch):
    monkeypatch.setattr(ltx_assets.settings,'artwork_generator','preserve')
    monkeypatch.setattr(ltx_assets,'run_worker',lambda *a,**k:pytest.fail('Model unexpectedly started'))
    assert ltx_assets.generate_artwork([],tmp_path)[1]['status']=='disabled'

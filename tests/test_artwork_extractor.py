import io
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from app.models.schemas import BoundingBox,DOMNode,KeyStateFrame,LayoutState
from app.modules.artwork_extractor import extract_motion,semantic_mask,static_artwork


def layout_for(tmp_path,width,height):
    source=tmp_path/'frame.png';Image.new('RGB',(width,height),(30,40,60)).save(source)
    control=DOMNode(node_id='input',tag_hint='input',bbox=BoundingBox(x=8,y=8,width=45,height=20),text_content='Email')
    return LayoutState(state_name='state_0',source_frame=KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(source)),root=DOMNode(node_id='root',tag_hint='div',bbox=BoundingBox(x=0,y=0,width=width,height=height),children=[control]))


def write_clip(path,width,height,kind):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),12,(width,height))
    assert writer.isOpened()
    for i in range(24):
        frame=np.full((height,width,3),35,np.uint8)
        cv2.rectangle(frame,(8,8),(53,28),(220,220,220),1)
        if kind=='motion':cv2.circle(frame,(width//2+int(np.sin(i*.5)*10),height//2),14,(30,210,250),-1)
        if kind=='cut' and i>=12:frame[:]=(230,210,180)
        if kind=='camera':
            for x in range(-20,width+20,20):cv2.line(frame,((x+i*4)%width,0),((x+i*4)%width,height-1),(180,190,200),3)
        writer.write(frame)
    writer.release()


@pytest.mark.parametrize('size',[(320,180),(180,320),(512,256)])
def test_local_motion_is_preserved_without_covering_controls(tmp_path,size):
    width,height=size;layout=layout_for(tmp_path,width,height);source=tmp_path/'source.avi'
    write_clip(source,width,height,'motion')
    out=tmp_path/'art';report=extract_motion(source,[layout],out,max_samples=16)
    assert report['status']=='extracted'
    assert report['sample_times'][0]==0
    assert report['sample_times'][-1]==pytest.approx(23/12,abs=.000001)
    for region in report['regions']:
        x,y,w,h=region['bbox']
        assert w<width and h<height
        assert not (x<53 and x+w>8 and y<28 and y+h>8)
        with Image.open(out/region['src']) as atlas:
            assert max(atlas.size)<=2048
    assert json.loads((out/'motion_manifest.json').read_text())['source_size']==[width,height]


def test_static_clip_exports_no_fake_animation(tmp_path):
    layout=layout_for(tmp_path,320,180);source=tmp_path/'static.avi';write_clip(source,320,180,'static')
    assert extract_motion(source,[layout],tmp_path/'art')['regions']==[]


def test_scene_cut_is_not_exported_as_local_motion(tmp_path):
    layout=layout_for(tmp_path,320,180);source=tmp_path/'cut.avi';write_clip(source,320,180,'cut')
    result=extract_motion(source,[layout],tmp_path/'art')
    assert result['status']=='review_required'
    assert result['regions']==[] and result['global_change_pairs']>0


def test_semantic_regions_are_removed_from_artwork(tmp_path):
    layout=layout_for(tmp_path,320,180)
    mask=semantic_mask(layout,(320,180))
    assert mask[15,15]==255 and mask[100,100]==0
    artwork,report=static_artwork(layout)
    assert Image.open(io.BytesIO(artwork)).size==(320,180)
    assert report['native_pixel_fraction']>0


def test_scrolling_is_withheld_from_local_artwork(tmp_path):
    layout=layout_for(tmp_path,320,180);source=tmp_path/'scroll.avi';write_clip(source,320,180,'camera')
    report=extract_motion(source,[layout],tmp_path/'art')
    assert report['status']=='review_required'
    assert report['regions']==[]


def test_motion_budget_is_bounded(tmp_path):
    with pytest.raises(ValueError,match='budgets'):
        extract_motion('unused',[],tmp_path,max_samples=10000)

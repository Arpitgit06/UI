import json

import cv2
import numpy as np
import pytest
from PIL import Image

from tools.analyze_reference_motion import analyze


def test_observed_contours_change_with_source_motion(tmp_path):
    source=tmp_path/'motion.avi'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),12,(800,450))
    assert writer.isOpened()
    for frame in range(12):
        image=np.zeros((450,800,3),dtype=np.uint8)
        cv2.circle(image,(396,200),55,(160,100,35),2)
        cv2.line(image,(360,175),(420,220),(230,230,200),2)
        cv2.ellipse(image,(396,200),(120,24),frame*4,0,360,(70,200,235),4)
        writer.write(image)
    writer.release()
    target=tmp_path/'observations.json'
    analyze(source,target,12)
    data=json.loads(target.read_text())
    assert len(data['samples'])==12
    assert data['samples'][0]['time']==0
    assert data['samples'][-1]['time']==pytest.approx(11/12,abs=.00001)
    assert data['samples'][0]['ring_paths']!=data['samples'][-1]['ring_paths']
    assert any(s['orb_paths'] for s in data['samples'])
    assert all(0<=s['wave_gain']<=1 for s in data['samples'])
    assert all(.65<=s['glow_gain']<=1.5 for s in data['samples'])
    for sample in data['samples']:
        for ring in sample['rings']:
            extent_y=np.hypot(ring['rx']*np.sin(ring['angle']),ring['ry']*np.cos(ring['angle']))
            assert extent_y<=192.01
    with Image.open(tmp_path/'public'/'artwork'/'sphere-atlas.webp') as atlas:
        assert atlas.mode=='RGBA'
        assert atlas.size==(3072,2560)


def test_invalid_video_is_reported(tmp_path):
    with pytest.raises(ValueError,match='Unable to read'):
        analyze(tmp_path/'missing.mp4',tmp_path/'output.json')

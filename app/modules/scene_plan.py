"""Keep observed noninteractive intro media separate from a recovered application screen."""
import json
from pathlib import Path


def nodes(root):
    yield root
    for child in root.children:
        yield from nodes(child)


def application_entry(layouts):
    # A field and an action distinguish a usable form from decorative intro cards.
    paired = [any(n.tag_hint=='input' for n in nodes(s.root)) and
              any(n.tag_hint=='button' for n in nodes(s.root)) for s in layouts]
    if not paired or not paired[-1]:
        return 0
    index=len(layouts)-1
    while index>0 and paired[index-1]:
        index-=1
    return index


def configure_scenes(files, layouts, video_path, motion):
    index=application_entry(layouts)
    plan={'version':1,'entry_state':layouts[index].state_name,'intro':None,
          'sequencing':'application controls select screens; no frame slideshow',
          'artwork_representation':'preserved assets, not recovered editable 3D'}
    source=Path(video_path)
    if index>0 and motion.get('status')=='review_required' and source.suffix.lower() in {'.mp4','.webm'}:
        asset='assets/intro-source'+source.suffix.lower()
        files['public/'+asset]=source.read_bytes()
        plan['intro']={'src':asset,'start':0,'end':layouts[index].source_frame.timestamp_sec,
                       'kind':'preserved_source_intro','editable':False}
    files['scene_manifest.json']=json.dumps(plan,indent=2)
    return plan

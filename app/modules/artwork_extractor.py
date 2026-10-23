"""Bounded, content-independent artwork preservation around semantic UI regions.

No named tools, fixed screen coordinates, color-specific ring models, or scene
templates. Native text/controls are cut out of artwork layers. Motion extraction
is conservative: frame-wide transitions and camera movement are not called art.
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SEMANTIC_TAGS = {'span','p','label','button','input','select','a'}


def semantic_mask(layout, size):
    w,h=size
    mask=np.zeros((h,w),np.uint8)
    def paint(x,y,bw,bh):
        x1=max(0,int(np.floor(x))-2);y1=max(0,int(np.floor(y))-2)
        x2=min(w,int(np.ceil(x+bw))+2);y2=min(h,int(np.ceil(y+bh))+2)
        mask[y1:y2,x1:x2]=255
    def walk(node):
        def tags(n):
            return {n.tag_hint} | set().union(*(tags(child) for child in n.children))
        def grounded(n):
            return bool(n.attributes.get('data-role-evidence')) or any(grounded(child) for child in n.children)
        # Rebuild the smallest form panel, including its border/background, rather
        # than retaining a screenshot of the same panel underneath native fields.
        if node is not layout.root and grounded(node) and {'input','button'} <= tags(node) and not any({'input','button'} <= tags(c) for c in node.children) and node.bbox.width*node.bbox.height < w*h*.2:
            b=node.bbox;paint(b.x,b.y,b.width,b.height)
        original=node.attributes.get('data-original-semantic-bbox')
        if original:
            paint(*json.loads(original))
        if (node.text_content or node.tag_hint in {'button','input','select','a'}) and node.tag_hint in SEMANTIC_TAGS:
            b=node.bbox;paint(b.x,b.y,b.width,b.height)
        for child in node.children:walk(child)
    walk(layout.root)
    return mask


def static_artwork(layout):
    """Remove native semantic regions, retaining unparsed artwork as explicit media."""
    with Image.open(layout.source_frame.image_path) as source:
        image=np.asarray(source.convert('RGB'))
    h,w=image.shape[:2]
    mask=semantic_mask(layout,(w,h))
    # Inpainting is an approximation behind real components, never recovered design.
    painted=cv2.inpaint(image,mask,3,cv2.INPAINT_TELEA) if mask.any() else image
    stream=io.BytesIO();Image.fromarray(painted).save(stream,format='PNG')
    return stream.getvalue(),{'native_pixel_fraction':round(float(np.count_nonzero(mask)/mask.size),5),
        'artwork_method':'source artwork with semantic text/control regions inpainted',
        'editability':'hybrid; unparsed graphics remain raster assets'}


def extract_motion(video_path, layouts, output_dir, max_samples=80, max_regions=4, tile_edge=256):
    """Find persistent local changes and export transparent source-artwork atlases.

    Long recordings are sampled across their complete duration with reported
    temporal resolution; no unbounded atlas or per-frame full-resolution cache.
    """
    if not 2<=max_samples<=120 or not 1<=max_regions<=8 or not 32<=tile_edge<=256:
        raise ValueError('Motion budgets exceed supported bounds')
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video_path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not np.isfinite(fps) or fps<=0 or count<1:raise ValueError('Cannot decode motion source')
        selected=np.unique(np.linspace(0,count-1,min(max_samples,count),dtype=int))
        frames=[];times=[];width=height=0
        for index in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
            if not ok:raise ValueError('Motion source failed to decode at a selected sample')
            height,width=frame.shape[:2]
            ratio=min(1,640/max(width,height))
            proxy=cv2.resize(frame,(max(1,round(width*ratio)),max(1,round(height*ratio))))
            if frames and proxy.shape!=frames[0].shape:raise ValueError('Changing source dimensions are unsupported')
            frames.append(proxy);times.append(round(float(index/fps),6))
    finally:cap.release()
    if not layouts:raise ValueError('Motion extraction requires semantic layouts')
    ph,pw=frames[0].shape[:2]
    protected=np.zeros((ph,pw),np.uint8)
    for layout in layouts:
        protected|=cv2.resize(semantic_mask(layout,(width,height)),(pw,ph),interpolation=cv2.INTER_NEAREST)
    activity=np.zeros((ph,pw),np.float32);comparisons=0;global_changes=0;camera_moves=0
    previous=None
    for frame in frames:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if previous is not None:
            diff=cv2.absdiff(gray,previous)>18
            if diff.mean()>.4:
                global_changes+=1
            else:
                # A coherent translation must not be baked into local artwork motion.
                shift,response=cv2.phaseCorrelate(previous.astype(np.float32),gray.astype(np.float32))
                if response>.2 and diff.mean()>.06 and np.hypot(*shift)>max(2,pw*.008):camera_moves+=1
                else:activity+=diff;comparisons+=1
        previous=gray
    report={'version':1,'duration_sec':round(count/fps,6),'source_size':[width,height],
            'sample_times':times,'effective_sample_hz':round((len(times)-1)/max(times[-1],1/fps),3),
            'regions':[],'warnings':[],'method':'automatic local temporal change outside semantic regions',
            'global_change_pairs':global_changes,'camera_motion_pairs':camera_moves}
    if len(times)>1 and max(np.diff(times))>.25:report['warnings'].append('Motion is temporally undersampled; rapid animation may differ')
    if global_changes:report['warnings'].append('Frame-wide changes were excluded from local artwork motion')
    if camera_moves:report['warnings'].append('Camera/scroll movement detected; those samples are excluded from local motion detection')
    # Be conservative across scene cuts, scrolling and state transitions. Keep the
    # state views rather than placing one scene's moving crop over another scene.
    if global_changes or camera_moves:
        report['status']='review_required'
        report['warnings'].append('Automatic motion preservation withheld across scene/camera changes; use captured states')
    elif comparisons>=2:
        active=(activity>=max(2,comparisons*.025)).astype(np.uint8)*255
        active[protected>0]=0
        active=cv2.morphologyEx(active,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        active=cv2.morphologyEx(active,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        active[protected>0]=0
        contours,_=cv2.findContours(active,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True):
            x,y,w,h=cv2.boundingRect(contour)
            if w*h<ph*pw*.001 or w<6 or h<6:continue
            if w*h>ph*pw*.4:
                report['warnings'].append('Broad motion region needs manual separation; not exported as an isolated asset')
                continue
            boxes.append((max(0,x-3),max(0,y-3),min(pw,x+w+3),min(ph,y+h+3)))
            if len(boxes)>=max_regions:break
        for region_index,(x1,y1,x2,y2) in enumerate(boxes):
            w,h=x2-x1,y2-y1
            scale=min(1,tile_edge/max(w,h));tw,th=max(1,round(w*scale)),max(1,round(h*scale))
            columns=min(8,len(frames));rows=int(np.ceil(len(frames)/columns))
            atlas=Image.new('RGBA',(columns*tw,rows*th))
            alpha=255-protected[y1:y2,x1:x2]
            for i,frame in enumerate(frames):
                rgba=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGBA);rgba[:,:,3]=alpha
                tile=Image.fromarray(rgba).resize((tw,th),Image.Resampling.LANCZOS)
                atlas.paste(tile,((i%columns)*tw,(i//columns)*th))
            relative=f'assets/motion/region-{region_index}.webp'
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);atlas.save(target,lossless=True)
            report['regions'].append({'id':f'artwork-{region_index}','src':relative,'columns':columns,
                'tile_size':[tw,th],'bbox':[round(x1/pw*width,3),round(y1/ph*height,3),round(w/pw*width,3),round(h/ph*height,3)],
                'kind':'preserved_artwork','editable_geometry':False})
        report['status']='extracted' if report['regions'] else 'no_isolated_motion'
    else:report['status']='no_isolated_motion'
    report['warnings'].append('Preserved motion assets do not establish semantic accuracy or recovered 3D geometry')
    (destination/'motion_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

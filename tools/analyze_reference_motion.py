"""Measure reviewed artwork regions; export editable 2D motion observations.

No 3D rotation is inferred from optical flow. Gold silhouettes are source-derived
vector paths, not a video texture. Regions and color selection are reviewed for
the Forge Hub reference and are not an automatic general-purpose scene parser.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def fit_rings(mask):
    """Robust partial-ellipse fit to observed gold pixels; no depth inference."""
    yy,xx=np.where(mask>0)
    points=np.column_stack((xx,yy)).astype('float32')
    if len(points)>1800:points=points[::max(1,len(points)//1800)]
    rng=np.random.default_rng(147)
    found=[]
    for _ in range(3):
        if len(points)<25:break
        best=None
        for attempt in range(220):
            subset=points[rng.choice(len(points),8,replace=False)].reshape(-1,1,2)
            (cx,cy),(a,b),angle=cv2.fitEllipse(subset)
            if not (abs(cx-396)<8 and abs(cy-200)<8 and 12<min(a,b)<210 and 180<max(a,b)<300):continue
            theta=np.deg2rad(angle)
            extent_x=np.sqrt((a/2*np.cos(theta))**2+(b/2*np.sin(theta))**2)
            extent_y=np.sqrt((a/2*np.sin(theta))**2+(b/2*np.cos(theta))**2)
            if not (90<extent_x<150 and extent_y<96):continue
            major_angle=np.deg2rad(angle)+(np.pi/2 if a<b else 0)
            if any(abs((major_angle-(ring['angle']+(np.pi/2 if ring['rx']<ring['ry'] else 0))+np.pi/2)%np.pi-np.pi/2)<.3 for ring in found):continue
            dx=points[:,0]-cx;dy=points[:,1]-cy
            u=dx*np.cos(theta)+dy*np.sin(theta);v=-dx*np.sin(theta)+dy*np.cos(theta)
            distance=np.abs(np.sqrt((u/(a/2))**2+(v/(b/2))**2)-1)*min(a,b)/2
            inliers=distance<1.8
            score=int(inliers.sum())
            if best is None or score>best[0]:best=(score,(cx,cy,a,b,angle),inliers)
        if best is None or best[0]<25:break
        score,(cx,cy,a,b,angle),inliers=best
        found.append({'cx':round(cx*2,3),'cy':round(cy*2,3),'rx':round(a,3),'ry':round(b,3),'angle':round(float(np.deg2rad(angle)),5),'support':score})
        points=points[~inliers]
    return found


def analyze(video, destination, sample_hz=12):
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or fps <= 0 or count <= 0:
        raise ValueError("Unable to read source video")
    previous = None
    samples = []
    atlas=Image.new('RGBA',(256*12,256*10))
    alpha=Image.new('L',(256,256));ImageDraw.Draw(alpha).ellipse((14,14,242,242),fill=255)
    alpha=alpha.filter(ImageFilter.GaussianBlur(1.2))
    for frame_index in range(0, count, max(1, round(fps/sample_hz))):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Unable to decode frame {frame_index}")
        image = cv2.resize(frame, (800, 450), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b, g, r = [x.astype(float) for x in cv2.split(image)]
        yy, xx = np.indices(gray.shape)
        sphere = ((xx-396)**2+(yy-200)**2 < 57**2)
        ring_region = (xx>250)&(xx<550)&(yy>105)&(yy<300)
        gold = ((r>b*1.025)&(g>b*.995)&(r>115)&(g>105)&ring_region&~sphere).astype('uint8')*255
        gold = cv2.morphologyEx(gold, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        orb_edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),45,110)
        orb_edges[~sphere]=0
        orb_contours,_=cv2.findContours(orb_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        orb_paths=[]
        for contour in sorted(orb_contours,key=lambda c:cv2.arcLength(c,False),reverse=True)[:60]:
            if cv2.arcLength(contour,False)<9:
                continue
            points=cv2.approxPolyDP(contour,.65,False).reshape(-1,2)
            if len(points)>=2:
                orb_paths.append((points*2).tolist())
        contours, _ = cv2.findContours(gold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        paths=[]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.arcLength(contour, True)<30 or cv2.contourArea(contour)<5:
                continue
            points=cv2.approxPolyDP(contour, .65, True).reshape(-1, 2)
            if len(points)>=3:
                paths.append((points*2).tolist())
            if len(paths)>=18:
                break
        # Static/background motion is separately reported rather than called orb rotation.
        flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0) if previous is not None else np.zeros((*gray.shape,2),np.float32)
        background = (yy<110)&(xx>220)&(xx<650)
        global_flow=np.median(flow[background],axis=0)
        orb_flow=np.median(flow[sphere],axis=0)-global_flow
        magnitude=cv2.magnitude(flow[...,0]-global_flow[0],flow[...,1]-global_flow[1])
        # Wave envelope is transient in this recording. Measure cyan excess outside
        # controls, instead of continuously drawing an invented sinusoidal wave.
        wave=(xx>275)&(yy>280)&(yy<440)
        cyan=np.maximum(0,(b+r*.1)-r)*np.maximum(0,(g-r))/255
        wave_energy=float(np.percentile(cyan[wave],95))
        source=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA))
        fw,fh=source.size
        sprite=source.crop((round(332/800*fw),round(136/450*fh),round(460/800*fw),round(264/450*fh))).resize((256,256),Image.Resampling.LANCZOS)
        sprite.putalpha(alpha)
        sprite_index=len(samples)
        if sprite_index>=120:raise ValueError('Reviewed atlas supports at most 120 observations; reduce sample-hz')
        atlas.paste(sprite,((sprite_index%12)*256,(sprite_index//12)*256))
        samples.append({"time":round(frame_index/fps,5),"ring_paths":paths,"rings":fit_rings(gold),"orb_paths":orb_paths,
                        "glow":round(float(gray[sphere].mean()),3),
                        "orb_flow":[round(float(x)*2,4) for x in orb_flow],
                        "orb_motion":round(float(np.median(magnitude[sphere]))*2,4),
                        "wave_energy":round(wave_energy,4)})
        previous=gray
    capture.release()
    # Normalize measured brightness/envelope against the observed clip range.
    glows=np.array([s['glow'] for s in samples]);waves=np.array([s['wave_energy'] for s in samples])
    for sample in samples:
        sample['glow_gain']=round(float(np.clip(sample['glow']/max(np.median(glows),1),.65,1.5)),4)
        sample['wave_gain']=round(float(np.clip((sample['wave_energy']-np.percentile(waves,10))/max(np.percentile(waves,95)-np.percentile(waves,10),1),0,1)),4)
    report={"version":1,"duration":round(count/fps,5),"sample_hz":fps/max(1,round(fps/sample_hz)),
            "coordinate_space":[1600,900],"source":Path(video).name,
            "method":"Reviewed regions; gold-color vector contours; relative optical flow and luminance",
            "limitations":["2D observations only; not recovered 3D meshes or angular velocity", "Wave amplitude and glow are measured envelopes; their rendered geometry is approximate", "Color segmentation may miss pale or occluded ring segments", "One-shot source timing; hold final observations at the end rather than inventing a seamless loop"],
            "samples":samples}
    destination.parent.mkdir(parents=True,exist_ok=True)
    assets=destination.parent/'public'/'artwork';assets.mkdir(parents=True,exist_ok=True)
    atlas.save(assets/'sphere-atlas.webp',lossless=True)
    report['sphere_atlas']={'src':'./artwork/sphere-atlas.webp','columns':12,'tile_size':256,'draw_rect':[664,272,256,256],
                            'provenance':'Isolated source artwork frames; not reconstructed geometry'}
    destination.write_text(json.dumps(report,separators=(',',':')),encoding='utf-8')
    print(json.dumps({"samples":len(samples),"duration":report['duration'],"vector_frames":sum(bool(s['ring_paths']) for s in samples),"output":str(destination)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--sample-hz',type=float,default=12)
    args=parser.parse_args()
    if not 1<=args.sample_hz<=30:parser.error('sample-hz must be between 1 and 30')
    analyze(args.video,args.output,args.sample_hz)

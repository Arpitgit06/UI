"""Conservative visual grouping around grounded text; never invent control roles."""
import cv2
import numpy as np
import re
import json

from app.models.schemas import BoundingBox, DetectedElement
from app.modules.visual_styles import measure_styles


def augment_visual_groups(image, detections):
    width,height=image.size
    ratio=min(1,1280/max(width,height))
    rgb=np.asarray(image.convert('RGB'))
    proxy=cv2.resize(rgb,(max(1,round(width*ratio)),max(1,round(height*ratio))))
    gray=cv2.cvtColor(proxy,cv2.COLOR_RGB2GRAY)
    edges=cv2.morphologyEx(cv2.Canny(gray,45,130),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[];rectangles=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if w<45*ratio or h<23*ratio or w*h>proxy.shape[0]*proxy.shape[1]*.45:continue
        hull=cv2.convexHull(contour)
        area=cv2.contourArea(hull)
        polygon=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
        if not 4<=len(polygon)<=8 or area/max(w*h,1)<.72:continue
        box=tuple(float(v/ratio) for v in (x,y,w,h))
        rectangles.append(box)
        children=[d for d in detections if d.text_content and d.bbox.x>=box[0]-3 and d.bbox.y>=box[1]-3 and d.bbox.x+d.bbox.width<=box[0]+box[2]+3 and d.bbox.y+d.bbox.height<=box[1]+box[3]+3]
        # Two text anchors justify grouping, not a button/backend-action inference.
        if len(children)<2:continue
        if any(abs(box[0]-old[0])<5 and abs(box[1]-old[1])<5 and abs(box[2]-old[2])<8 and abs(box[3]-old[3])<8 for old,_ in candidates):continue
        candidates.append((box,children))
    from app.modules.contextual_controls import recover_field_actions
    from app.modules.visual_styles import control_rectangles
    detections=recover_field_actions(image,detections,control_rectangles(image))
    extra=[];replacements={}
    # OCR often sees a field's label, not its empty editable box. Associate a
    # grounded label with a measured border below it instead of typing on the label.
    for i,detection in enumerate(detections):
        text=(detection.text_content or '').strip()
        if not re.fullmatch(r'(email(?: address)?|e-mail|password|username|phone|name|address)',text,re.I):continue
        b=detection.bbox
        if detection.attributes.get('data-text-bbox'):
            try:
                tx,ty,tw,th=json.loads(detection.attributes['data-text-bbox'])
                b=BoundingBox(x=tx,y=ty,width=tw,height=th)
            except (ValueError,TypeError):pass
        below=[box for box in rectangles if box[1]>=b.y+b.height*.5 and box[1]<=b.y+b.height+65 and 18<=box[3]<=90 and box[2]>=max(60,b.width) and box[0]<=b.x+15 and box[0]+box[2]>=b.x+b.width-10]
        if not below:continue
        box=min(below,key=lambda box:(box[1]-b.y,box[2]*box[3]))
        replacements[detection.element_id]=detection.model_copy(update={'element_type':'text','bbox':b,'styles':measure_styles(image,(b.x,b.y,b.width,b.height),text,'text')})
        attributes={'aria-label':text,'type':'password' if 'password' in text.lower() else 'email' if 'mail' in text.lower() else 'text'}
        extra.append(DetectedElement(element_id=f'geometry-field-{i}',element_type='input',bbox=BoundingBox(x=box[0],y=box[1],width=box[2],height=box[3]),confidence=.7,detection_source='geometry',styles=measure_styles(image,box,None,'input'),attributes=attributes))
    # Keep bounded geometry; preferring smaller boxes retains tool cards before panels.
    for i,(box,children) in enumerate(sorted(candidates,key=lambda x:x[0][2]*x[0][3])[:40]):
        x,y,w,h=box
        extra.append(DetectedElement(element_id=f'geometry-group-{i}',element_type='container',bbox=BoundingBox(x=x,y=y,width=w,height=h),confidence=.6,detection_source='geometry',styles=measure_styles(image,box,None,'container')))
        first=min(child.bbox.x for child in children)
        gap=first-x
        # Decorative leading region is extracted as imagery, never a guessed icon name.
        if gap>=max(18,h*.45) and gap<min(w*.5,h*1.8):
            icon_box=(x+2,y+2,max(1,gap-5),max(1,h-4))
            crop=np.asarray(image.crop(tuple(map(int,(icon_box[0],icon_box[1],icon_box[0]+icon_box[2],icon_box[1]+icon_box[3])))).convert('RGB'))
            if crop.size and np.std(crop)>22:
                extra.append(DetectedElement(element_id=f'geometry-art-{i}',element_type='image',bbox=BoundingBox(x=icon_box[0],y=icon_box[1],width=icon_box[2],height=icon_box[3]),confidence=.55,detection_source='geometry'))
    return [replacements.get(d.element_id,d) for d in detections]+extra

"""Recover field/action pairs using both text intent and nearby measured bounds."""
import json
import re
from app.models.schemas import BoundingBox
from app.modules.visual_styles import measure_styles

FIELD = re.compile(r'^(?:paste|enter|type|search|provide)\b.+', re.I)
ACTION = re.compile(r'^(?:scan|inspect|analy[sz]e|check|search|submit|send|continue|verify|generate|execute|run)\b', re.I)


def recover_field_actions(image, detections, rectangles):
    result = list(detections)
    anchors=[]
    for element in detections:
        try:
            x,y,w,h=json.loads(element.attributes['data-text-bbox'])
            anchors.append(element.model_copy(update={'bbox':BoundingBox(x=x,y=y,width=w,height=h)}))
        except (KeyError,ValueError,TypeError):
            anchors.append(element)
    for i, field in enumerate(anchors):
        if not FIELD.match((field.text_content or '').strip()):
            continue
        b = field.bbox
        actions = [(j, action) for j, action in enumerate(anchors)
                   if j != i and ACTION.match((action.text_content or '').strip())
                   and action.bbox.x >= b.x + b.width * .65
                   and action.bbox.x - (b.x+b.width) < max(100, b.height*8)
                   and abs(action.bbox.y+action.bbox.height/2-b.y-b.height/2) < max(b.height, action.bbox.height)]
        if not actions:
            continue
        j, action = min(actions, key=lambda pair: pair[1].bbox.x)
        for index, element, role in [(i, field, 'input'), (j, action, 'button')]:
            if element.attributes.get('data-human-reviewed'):
                continue
            e = element.bbox
            # Require a measured border around the label; never turn prose alone into a form.
            boxes = [box for box in rectangles if box[0] <= e.x+3 and box[1] <= e.y+3
                     and box[0]+box[2] >= e.x+e.width-3 and box[1]+box[3] >= e.y+e.height-3
                     and box[2] >= e.width*.9 and max(16,e.height) <= box[3] <= max(100,e.height*4)
                     and box[2]*box[3] < image.width*image.height*.12]
            if not boxes:
                continue
            x,y,w,h = min(boxes, key=lambda box: box[2]*box[3])
            if role == 'input':
                w = min(x+w, action.bbox.x-12)-x
            if w < e.width*.9:
                continue
            attributes = dict(element.attributes)
            attributes.update({'data-text-bbox':json.dumps([e.x,e.y,e.width,e.height]),
                               'data-role-evidence':'adjacent field/action labels with enclosing border'})
            if role == 'input':
                attributes['type'] = 'url' if re.search(r'\b(link|url|website)\b', field.text_content, re.I) else 'text'
            styles = measure_styles(image,(x,y,w,h),element.text_content,role)
            text_styles = measure_styles(image,(e.x,e.y,e.width,e.height),element.text_content,'text')
            styles.update({k:v for k,v in text_styles.items() if k in {'font-size','font-family','color'}})
            styles['font-weight']=text_styles.get('font-weight','400')
            if role=='input':
                styles.update({'background-color':'transparent','border':'0','text-align':'left','padding-left':f'{max(4,e.x-x)}px'})
            else:
                styles['border-radius']=f'{round(h*.18)}px'
                if 'border' in styles:
                    color=styles['border'].split()[-1]
                    styles['box-shadow']=f'0 0 {round(h*.25)}px {color}, inset 0 0 {round(h*.18)}px {color}'
            result[index] = element.model_copy(update={'element_type':role,
                'bbox':BoundingBox(x=x,y=y,width=w,height=h),'styles':styles,'attributes':attributes})
    return result

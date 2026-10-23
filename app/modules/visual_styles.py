"""Measure simple styling; preserve complex visual regions as packaged assets."""
import numpy as np
from PIL import Image,ImageFont
from functools import lru_cache


@lru_cache(maxsize=1)
def _measurement_font():
    for name in ('arial.ttf','DejaVuSans.ttf'):
        try:return ImageFont.truetype(name,64)
        except OSError:pass
    return None


def _hex(rgb):
    return "#" + "".join(f"{int(v):02x}" for v in rgb)


def control_rectangles(image):
    import cv2
    gray=cv2.cvtColor(np.asarray(image.convert("RGB")),cv2.COLOR_RGB2GRAY)
    edges=cv2.morphologyEx(cv2.Canny(gray,40,120),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    rectangles=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        hull=cv2.convexHull(contour)
        polygon=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
        if w>=25 and h>=14 and 4<=len(polygon)<=8 and cv2.contourArea(hull)/max(w*h,1)>.72:
            rectangles.append((x,y,w,h))
    return rectangles


def snap_control_bbox(bbox, rectangles, image_size):
    """Prefer a measured enclosing control border over the label's text box."""
    x,y,w,h=bbox
    candidates=[]
    for rx,ry,rw,rh in rectangles:
        if (rx<=x+5 and ry<=y+5 and rx+rw>=x+w-5 and ry+rh>=y+h-5
            and rw>=w and rh>=h and rh<max(180,h*3)
            and rw*rh<image_size[0]*image_size[1]*.15):
            candidates.append((rw*rh,(float(rx),float(ry),float(rw),float(rh))))
    return min(candidates,key=lambda item:item[0])[1] if candidates else bbox


def measure_styles(image: Image.Image, bbox, text: str | None, kind: str) -> dict[str, str]:
    x, y, w, h = bbox
    crop = image.crop((int(x), int(y), int(x+w), int(y+h))).convert("RGB")
    a = np.asarray(crop)
    if a.size == 0:
        return {}
    edges = np.concatenate((a[:3].reshape(-1,3),a[-3:].reshape(-1,3),a[:,:3].reshape(-1,3),a[:,-3:].reshape(-1,3)))
    interior=a[3:-3,3:-3] if min(a.shape[:2])>8 else a
    background = np.median(interior.reshape(-1,3), axis=0)
    styles = {}
    if kind != "text":
        styles["background-color"] = _hex(background)
        contrasting=edges[np.linalg.norm(edges.astype(float)-background,axis=1)>22]
        border=np.median(contrasting,axis=0) if len(contrasting)>len(edges)*.15 else background
        if kind!='page' and np.linalg.norm(border-background)>22:
            styles['border']=f'1px solid {_hex(border)}'
    if text:
        distance = np.linalg.norm(a.astype(float)-background, axis=2)
        ink = a[distance > max(35, float(np.percentile(distance, 75)))]
        foreground = np.median(ink, axis=0) if len(ink) else (255-background)
        import cv2
        binary=(distance>max(35,float(np.percentile(distance,75)))).astype('uint8')
        if kind!='text' and min(binary.shape)>8:
            binary[:3]=0;binary[-3:]=0;binary[:,:3]=0;binary[:,-3:]=0
        count,_,stats,_=cv2.connectedComponentsWithStats(binary)
        glyphs=[s for s in stats[1:] if s[4]>=2 and not(s[3]<=3 and s[2]>w*.6) and not(s[2]<=3 and s[3]>h*.8)]
        if kind!='text':
            glyphs=[s for s in glyphs if s[0]>3 and s[1]>3 and s[0]+s[2]<a.shape[1]-3 and s[1]+s[3]<a.shape[0]-3]
        ink_width=max((s[0]+s[2] for s in glyphs),default=w)-min((s[0] for s in glyphs),default=0)
        font=_measurement_font()
        advance=font.getlength(text) if font else len(text)*35
        font_size=max(8,min(float(h)*(1.4 if kind=='text' else .65),ink_width*64/max(advance,1)))
        alignment='left' if kind=='text' else 'center'
        if kind!='text' and glyphs:
            left=min(s[0] for s in glyphs);right=a.shape[1]-max(s[0]+s[2] for s in glyphs)
            if left<w*.18 and right>w*.35:
                alignment='left';styles['padding-left']=f'{left}px'
        styles.update({"color": _hex(foreground), "font-size": f"{round(font_size)}px",
                       "font-family": "Arial, sans-serif", "line-height": "1.2",
                       "text-align": alignment})
        if kind=='text':
            from app.modules.font_matching import match_font
            matched=match_font(a,text)
            if matched:styles.update(matched)
    return styles

"""Match available font shapes to observed glyphs; never package licensed fonts."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


@lru_cache(maxsize=1)
def candidates():
    root=Path('C:/Windows/Fonts')
    choices=[('arial.ttf','Arial',400),('arialbd.ttf','Arial',700),('ARIALN.TTF','Arial Narrow',400),('ARIALNB.TTF','Arial Narrow',700),('bahnschrift.ttf','Bahnschrift',400)]
    return [(ImageFont.truetype(str(root/file),64),family,weight) for file,family,weight in choices if (root/file).is_file()]


def match_font(rgb,text):
    if len(text)<4 or rgb.shape[0]<12 or not candidates():return None
    background=np.median(rgb.reshape(-1,3),axis=0)
    distance=np.linalg.norm(rgb.astype(float)-background,axis=2)
    mask=(distance>max(40,float(np.percentile(distance,70)))).astype('uint8')*255
    points=cv2.findNonZero(mask)
    if points is None:return None
    x,y,w,h=cv2.boundingRect(points)
    if h<8 or w<12:return None
    source=cv2.resize(mask[y:y+h,x:x+w],(max(32,min(800,w)),48))>100
    best=None
    for font,family,weight in candidates():
        box=font.getbbox(text);tw=box[2]-box[0];th=box[3]-box[1]
        if not tw or not th:continue
        image=Image.new('L',(tw,th));ImageDraw.Draw(image).text((-box[0],-box[1]),text,font=font,fill=255)
        template=cv2.resize(np.asarray(image),(source.shape[1],48))>100
        iou=np.count_nonzero(source&template)/max(1,np.count_nonzero(source|template))
        score=iou-.7*abs(np.log((tw/th)/(w/h)))
        size=h*64/th
        spacing=max(-size*.08,min(size*.08,(w-tw*size/64)/max(1,len(text)-1)))
        if best is None or score>best[0]:best=(score,family,weight,size,spacing)
    if best and best[0]>.30:
        return {'font-family':f'"{best[1]}", Arial, sans-serif','font-weight':str(best[2]),'font-size':f'{round(best[3])}px','letter-spacing':f'{best[4]:.2f}px'}
    return None

"""Exercise the real upload pipeline on varied deterministic UI recordings.

These generated fixtures test generalization boundaries, not arbitrary real-world
fidelity. OCR, vision classification, compilation and Chromium checks are real.
"""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from app.config import settings
from app.core.job_queue import persist_job
from app.core.pipeline import run_pipeline
from app.models.schemas import Job,JobStatus


def make_clip(path,size,kind):
    w,h=size
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(w,h))
    if not writer.isOpened():raise RuntimeError('Fixture encoder unavailable')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
    for frame in range(36):
        image=Image.new('RGB',size,'#101e30' if kind=='animated' else '#edf1f7')
        draw=ImageDraw.Draw(image)
        if kind=='animated':
            draw.rounded_rectangle((25,25,300,400),12,fill='#1e344b',outline='#78a9c9',width=2)
            for text,y in [('Analytics',45),('Weekly activity',105),('Live chart',150)]:draw.text((45,y),text,font=title if y==45 else font,fill='white')
            draw.rounded_rectangle((45,300,270,345),6,fill='#407ce6');draw.text((62,310),'Generate report',font=font,fill='white')
            center=(550,225);radius=75
            draw.ellipse((475,150,625,300),outline='#71dfed',width=3)
            angle=frame*.1
            draw.line((center[0],center[1],center[0]+np.cos(angle)*radius,center[1]+np.sin(angle)*radius),fill='#8ef5d3',width=5)
        else:
            margin=30 if w<500 else 150
            draw.rounded_rectangle((margin-15,40,w-margin+15,h-30),12,fill='white',outline='#bcc9d8',width=2)
            draw.text((margin,65),'Welcome back',font=title,fill='#18324a')
            for text,y in [('Email address',120),('Password',200)]:
                draw.text((margin,y),text,font=font,fill='#1b3049')
                draw.rounded_rectangle((margin,y+28,w-margin,y+65),5,outline='#71869c',width=2)
            draw.rounded_rectangle((margin,290,w-margin,335),6,fill='#246bd6');draw.text((margin+15,300),'Sign in',font=font,fill='white')
            if kind=='portrait':
                draw.text((margin,390),'Your workspace',font=title,fill='#18324a')
                draw.text((margin,435),'Projects and messages',font=font,fill='#4b617b')
        writer.write(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
    writer.release()


async def main():
    settings.ensure_directories();settings.max_key_states=2
    evaluation=settings.jobs_dir/'generalization_evaluation';evaluation.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,size in [('login',(640,400)),('portrait',(360,640)),('animated',(800,450))]:
        source=settings.uploads_dir/f'evaluation_{name}.mp4';make_clip(source,size,name)
        now=datetime.now(timezone.utc);job=Job(job_id=f'eval_{name}_{uuid.uuid4().hex[:6]}',created_at=now,updated_at=now,source_video_path=str(source),render_mode='hybrid')
        persist_job(job);print(f'RUN {name}: {job.job_id}',flush=True)
        try:await run_pipeline(job)
        except Exception as exc:job.status=JobStatus.FAILED;job.error_message=str(exc)
        finally:persist_job(job)
        result={'case':name,'size':size,'job_id':job.job_id,'status':job.status.value,'warnings':job.warnings,'error':job.error_message}
        if job.status==JobStatus.COMPLETE:
            generated=settings.jobs_dir/job.job_id/'generated'
            report=json.loads((generated/'verification_report.json').read_text())
            result['verification']=report['output_verification']
            result['motion']=json.loads((generated/'motion_manifest.json').read_text())['status']
        results.append(result)
        (evaluation/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='verification'}),flush=True)
    if any(r['status']!='complete' for r in results):raise SystemExit(1)


if __name__=='__main__':asyncio.run(main())

"""Sequential LTX-Video 2B FP16/Q8 artwork generation, separate from UI semantics."""
import json
import sys
import time
from pathlib import Path
from app.config import settings
from app.core.process_runner import run_worker

BASE_REPO='Lightricks/LTX-Video'
Q8_REPO='city96/LTX-Video-gguf'
Q8_FILE='ltx-video-2b-v0.9-Q8_0.gguf'
FP16_FILE='ltx-video-2b-v0.9-F16.gguf'


def generate_artwork(layouts,work):
    destination=Path(work)/'ltx';destination.mkdir(parents=True,exist_ok=True)
    if settings.artwork_generator=='preserve':
        return {},{'status':'disabled','reason':'Observed source artwork preserved; LTX generation not enabled'}
    if settings.artwork_generator not in {'ltx-2b-q8','ltx-2b-fp16'}:raise ValueError('Unsupported artwork generator')
    precision='fp16' if settings.artwork_generator.endswith('fp16') else 'q8'
    # Recognition subprocesses finish before this function is entered. Each
    # generation phase also exits before the next is started: no model overlap.
    from app.modules.scene_plan import application_entry
    from app.modules.artwork_extractor import static_artwork,semantic_mask
    from PIL import Image
    layout=layouts[application_entry(layouts)]
    reference=destination/'reference.png';reference.write_bytes(static_artwork(layout)[0])
    mask=semantic_mask(layout,(round(layout.root.bbox.width),round(layout.root.bbox.height)))
    Image.fromarray(255-mask).save(destination/'artwork-mask.png')
    payload={'base_path':str(settings.models_cache_dir/'ltx-2b'/'base'),
             'q8_path':str(settings.models_cache_dir/'ltx-2b'/(FP16_FILE if precision=='fp16' else Q8_FILE)),
             'precision':precision,
             'reference':str(reference.resolve()),'output_dir':str(destination.resolve()),
             'prompt':'Subtle continuous ambient animation of the provided abstract interface artwork. Keep the composition and camera fixed. Preserve colors and lighting. No text, letters, buttons, panels or new objects.',
             'width':320,'height':192,'frames':17,'steps':20,'fps':12,'seed':0}
    path=destination/'request.json';path.write_text(json.dumps(payload),encoding='utf-8')
    trace=[]
    for phase in ('encode','generate'):
        started=time.time()
        result=run_worker([sys.executable,'-m','app.modules.ltx_worker',str(path.resolve()),phase],timeout=settings.worker_timeout_sec)
        trace.append({'stage':phase,'started_unix':started,'exited_unix':time.time(),'returncode':result.returncode})
        if result.returncode:raise RuntimeError(f'LTX {phase} failed: '+result.stderr[-1500:])
    video=destination/'generated.mp4'
    if not video.is_file():raise RuntimeError('LTX returned without an animation asset')
    manifest={'status':'generated_candidate','model':'LTX-Video 2B v0.9 '+precision,
              'state_id':layout.state_name,'src':'assets/generated-artwork.mp4','mask':'assets/generated-artwork-mask.png',
              'size':[320,192],'frames':17,'fps':12,'seed':0,
              'editable_ui':False,'requires_visual_review':True,
              'gpu_stage_trace':trace,'preview_query':'artwork=generated',
              'stages':['recognition process exited','T5 INT8 process exited','LTX '+precision+' + VAE process exited'],
              'profiles':{phase:json.loads((destination/f'{phase}-profile.json').read_text()) for phase in ('encode','generate')}}
    return {'public/assets/generated-artwork.mp4':video.read_bytes(),
            'public/assets/generated-artwork-mask.png':(destination/'artwork-mask.png').read_bytes()},manifest

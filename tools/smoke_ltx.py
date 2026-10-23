"""Measure the sequential VL-7B, T5, LTX-2B and Coder-7B stack on CUDA."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import settings
from app.models.schemas import KeyStateFrame


async def main():
    from app.modules.module_b_spatial_vision import analyze_states
    from app.modules.module_c_dom_synthesizer import synthesize_dom
    from app.modules.ltx_assets import generate_artwork
    from app.modules.coder_runtime import generate_primitives
    from app.core.process_runner import resume_workers,shutdown_workers
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--precision',choices=['fp16','q8'],default='fp16')
    args=parser.parse_args()
    work=settings.jobs_dir/'model_stack_smoke';work.mkdir(parents=True,exist_ok=True)
    # Keep source-frame diagnostics inside this run, never overwrite the source job.
    import shutil
    target=work/'reference.png';shutil.copyfile(args.image,target)
    resume_workers()
    try:
        state=KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(target.resolve()))
        detections,_=await analyze_states([state])
        layout=await synthesize_dom(state,detections[0],work/'layouts')
        settings.artwork_generator='ltx-2b-'+args.precision
        files,report=await asyncio.to_thread(generate_artwork,[layout],work)
        import time
        started=time.time()
        source,coder_report=await asyncio.to_thread(generate_primitives,[layout])
        report['coder_trace']={'started_unix':started,'exited_unix':time.time()}
        report['coder']=coder_report
        (work/'MeasuredComponents.jsx').write_text(source,encoding='utf-8')
        for name,data in files.items():
            destination=work/'generated'/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        report['recognition_profile']=json.loads((work/'gpu_profile.json').read_text())
        (work/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    finally:shutdown_workers()


if __name__=='__main__':asyncio.run(main())

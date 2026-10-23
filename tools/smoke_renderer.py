"""Exercise real React bundling/browser capture without inference models."""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageDraw
from app.config import settings
from app.models.schemas import KeyStateFrame, DetectedElement, BoundingBox
from app.modules.module_c_dom_synthesizer import _synthesize_dom_sync
from app.modules.module_d_code_generator import generate_code
from app.modules.verification import run_visual_verification


async def main():
    settings.code_generator='deterministic'  # This renderer-only check deliberately runs no model.
    directory=settings.jobs_dir/"renderer_smoke"
    directory.mkdir(parents=True,exist_ok=True)
    source=directory/"source.png"
    image=Image.new("RGB",(640,360),"#152334")
    draw=ImageDraw.Draw(image)
    draw.rectangle((100,50,500,280),fill="#25415c")
    draw.rectangle((130,100,330,145),fill="#59bbff")
    draw.text((140,110),'Save {name} < Back',fill="black")
    image.save(source)
    elements=[DetectedElement(element_id="card",element_type="card",bbox=BoundingBox(x=100,y=50,width=400,height=230)),
              DetectedElement(element_id="button",element_type="button",text_content='Save {name} < Back',bbox=BoundingBox(x=130,y=100,width=200,height=45)),
              DetectedElement(element_id="input",element_type="input",text_content='Type here',bbox=BoundingBox(x=130,y=180,width=200,height=35))]
    layout=_synthesize_dom_sync(KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(source)),elements,is_3d_scene=False)
    second=layout.model_copy(deep=True)
    second.state_name="state_120"
    second.source_frame.frame_index=120
    second.source_frame.timestamp_sec=4
    files=await generate_code([layout,second])
    files["interaction_map.json"]=json.dumps({"state_0":{"button":"state_120"}})
    output=directory/"generated"
    for name,content in files.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content if isinstance(content,bytes) else content.encode("utf-8"))
    result=run_visual_verification([layout,second],str(output))
    (output/"browser_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

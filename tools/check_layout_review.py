"""Real-browser regression: clipping and overlapping controls must be review findings."""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from PIL import Image
from app.config import settings
from app.models.schemas import BoundingBox,DOMNode,KeyStateFrame,LayoutState
from app.modules.module_d_code_generator import generate_code
from app.modules.verification import run_visual_verification


async def main():
    root=settings.jobs_dir/'layout_review_check';root.mkdir(parents=True,exist_ok=True)
    source=root/'frame.png';Image.new('RGB',(480,300),'#152334').save(source)
    nodes=[DOMNode(node_id='clipped',tag_hint='span',element_type='text',text_content='A deliberately overflowing label',bbox=BoundingBox(x=10,y=10,width=60,height=15),styles={'font-size':'32px','color':'#ffffff'}),
        DOMNode(node_id='fittable',tag_hint='span',element_type='text',text_content='Fit this text',bbox=BoundingBox(x=100,y=10,width=95,height=24),styles={'font-size':'24px','color':'#ffffff'}),
        DOMNode(node_id='first',tag_hint='button',element_type='button',text_content='First',bbox=BoundingBox(x=150,y=150,width=100,height=40)),
        DOMNode(node_id='second',tag_hint='button',element_type='button',text_content='Second',bbox=BoundingBox(x=160,y=155,width=100,height=40))]
    layout=LayoutState(state_name='state_0',source_frame=KeyStateFrame(frame_index=0,timestamp_sec=0,image_path=str(source)),root=DOMNode(node_id='root',tag_hint='div',bbox=BoundingBox(x=0,y=0,width=480,height=300),children=nodes))
    output=root/'generated'
    for name,content in (await generate_code([layout])).items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content if isinstance(content,bytes) else content.encode())
    result=await asyncio.to_thread(run_visual_verification,[layout],str(output))
    assert result['status']=='passed',result
    findings=[finding for state in result['states'] if state['mode']=='hybrid' for finding in state['layout_findings']]
    assert any(f['node_id']=='clipped' and f['kind']=='text_clipping' for f in findings),findings
    assert any(f['kind']=='overlapping_controls' for f in findings),findings
    assert not any(f['node_id']=='fittable' for f in findings),findings
    adjusted=[item for state in result['states'] if state['mode']=='hybrid' for item in state['typography_adjustments']]
    assert any(item['node_id']=='fittable' and .65<=item['scale']<1 for item in adjusted),adjusted
    (output/'review_findings.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','findings':findings}))


if __name__=='__main__':asyncio.run(main())

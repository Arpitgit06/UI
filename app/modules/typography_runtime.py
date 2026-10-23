"""Browser font metrics complement source OCR bounds without guessing font files."""
TYPOGRAPHY_HOOK = r"""import {useLayoutEffect} from 'react';
export function useMeasuredTypography(ref,mode){
  useLayoutEffect(()=>{
    let live=true;
    const fit=()=>{
      if(!live||!ref.current||mode==='asset-backed')return;
      const ctx=document.createElement('canvas').getContext('2d');
      for(const node of ref.current.querySelectorAll('.omniui-editable-layer :is(span,p,label,button,a)')){
        if(!node.textContent.trim()||node.querySelector('[data-node-id],[data-owner-node-id]'))continue;
        node.style.fontSize='';delete node.dataset.fontFit;
        const style=getComputedStyle(node),size=parseFloat(style.fontSize);
        ctx.font=`${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
        if('letterSpacing' in ctx)ctx.letterSpacing=style.letterSpacing==='normal'?'0px':style.letterSpacing;
        const metric=ctx.measureText(node.textContent),height=metric.actualBoundingBoxAscent+metric.actualBoundingBoxDescent;
        const width=node.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight);
        const availableHeight=node.clientHeight-parseFloat(style.paddingTop)-parseFloat(style.paddingBottom);
        if(width<=0||availableHeight<=0||!size)continue;
        const ratio=Math.max(.65,Math.min(1,width/Math.max(metric.width,1),availableHeight/Math.max(height,1)));
        if(ratio<.995){node.style.fontSize=`${size*ratio}px`;node.dataset.fontFit=ratio.toFixed(4);}
      }
    };
    fit();document.fonts.ready.then(fit);
    const observer=new ResizeObserver(fit);if(ref.current)observer.observe(ref.current);
    return()=>{live=false;observer.disconnect();};
  },[mode]);
}
"""

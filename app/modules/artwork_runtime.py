"""Generated, reusable React player for observed artwork sprite atlases."""

ARTWORK_COMPONENT = r"""import React, {useEffect,useRef} from 'react';
import manifest from './motion_manifest.json';
export default function ArtworkMotion({time=0}) {
  const ref=useRef(null),cache=useRef(new Map());
  useEffect(()=>{
    const canvas=ref.current,ctx=canvas.getContext('2d');let live=true;
    const draw=()=>{
      if(!live)return;
      const [w,h]=manifest.source_size||[1,1];
      const ratio=Math.min(1,1280/Math.max(w,h));canvas.width=Math.max(1,Math.round(w*ratio));canvas.height=Math.max(1,Math.round(h*ratio));
      ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
      let index=0;manifest.sample_times.forEach((t,i)=>{if(t<=time)index=i;});
      let loaded=0;
      for(const region of manifest.regions){
        const image=cache.current.get(region.src);if(!image?.complete||!image.naturalWidth)continue;
        loaded++;
        const [tw,th]=region.tile_size;ctx.drawImage(image,(index%region.columns)*tw,Math.floor(index/region.columns)*th,tw,th,...region.bbox);
      }
      canvas.dataset.sourceTime=String(time);canvas.dataset.regionCount=String(manifest.regions.length);
      canvas.dataset.loadedRegionCount=String(loaded);
    };
    for(const region of manifest.regions){
      if(!cache.current.has(region.src)){const image=new Image();cache.current.set(region.src,image);image.src=region.src;}
      cache.current.get(region.src).onload=draw;
    }
    draw();return()=>{live=false;for(const image of cache.current.values())image.onload=null;};
  },[time]);
  return <canvas ref={ref} className="omniui-artwork-motion" aria-label="Preserved isolated artwork motion"/>;
}
"""

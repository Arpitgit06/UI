import React, {useEffect, useRef} from 'react';
import scene from './scene.json';
import motion from './motion.json';

// Independent procedural artwork, not recovered source geometry. Fixed seeds keep
// comparisons reproducible; elapsed time survives pause/resume and effect cleanup.
export default function Universe({paused}) {
  const ref=useRef(null), phase=useRef(Math.max(0,Math.min(motion.samples.at(-1).time,Number(new URLSearchParams(location.search).get('time'))||0))), artwork=useRef(null);
  useEffect(()=>{
    const canvas=ref.current,ctx=canvas.getContext('2d'),sprite=new Image();
    const preference=window.matchMedia('(prefers-reduced-motion: reduce)');
    let reduced=preference.matches,raf,previous;
    let seed=147;const random=()=>{seed=seed*16807%2147483647;return seed/2147483647;};
    const stars=Array.from({length:850},()=>[random()*1600,random()*900,random()*1.35+.15,random()]);
    if(!artwork.current){
      // Fractal cloud texture is generated once, not re-evaluated every frame.
      const cloud=document.createElement('canvas');cloud.width=600;cloud.height=338;
      const cc=cloud.getContext('2d'),pixels=cc.createImageData(cloud.width,cloud.height);
      const hash=(x,y)=>{const n=Math.sin(x*127.1+y*311.7)*43758.5453;return n-Math.floor(n);};
      const noise=(x,y)=>{const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy,sx=fx*fx*(3-2*fx),sy=fy*fy*(3-2*fy);return (hash(ix,iy)*(1-sx)+hash(ix+1,iy)*sx)*(1-sy)+(hash(ix,iy+1)*(1-sx)+hash(ix+1,iy+1)*sx)*sy;};
      for(let y=0;y<338;y++)for(let x=0;x<600;x++){
        let f=0,weight=.52,frequency=.012;
        for(let o=0;o<5;o++){f+=noise(x*frequency,y*frequency)*weight;weight*=.52;frequency*=2.05;}
        const ridge=Math.max(0,1-Math.abs(y/338-(.42+.24*Math.sin(x*.017)))*2.7);
        const intensity=Math.pow(Math.max(0,f-.23),1.5)*ridge;
        const warm=Math.exp(-((x-510)**2/6500+(y-140)**2/12000));
        const i=(y*600+x)*4;pixels.data[i]=25+warm*90;pixels.data[i+1]=100-warm*40;pixels.data[i+2]=135-warm*65;pixels.data[i+3]=Math.min(210,intensity*650);
      }
      cc.putImageData(pixels,0,0);artwork.current=cloud;
    }
    const ellipse=(x,y,rx,ry,rot,color,width=1)=>{ctx.beginPath();ctx.ellipse(x,y,rx,ry,rot,0,Math.PI*2);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();};
    const path=(points,color,width=1,fill)=>{ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));if(fill){ctx.closePath();ctx.fillStyle=fill;ctx.fill();}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();};
    const draw=now=>{
      if(previous&&!paused&&!reduced)phase.current=Math.min(motion.samples.at(-1).time,phase.current+Math.max(now-previous,0)/1000);previous=now;
      const t=phase.current;
      let index=motion.samples.findIndex(sample=>sample.time>t)-1;
      if(index<0)index=t>=motion.samples.at(-1).time?motion.samples.length-1:0;
      const observation=motion.samples[index],next=motion.samples[Math.min(index+1,motion.samples.length-1)];
      const blend=next.time>observation.time?Math.min(1,(t-observation.time)/(next.time-observation.time)):0;
      const gain=observation.glow_gain+(next.glow_gain-observation.glow_gain)*blend;
      const waveGain=observation.wave_gain+(next.wave_gain-observation.wave_gain)*blend;
      canvas.dataset.sourceTime=t.toFixed(3);canvas.dataset.observation=String(index);
      const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio||1,2);
      if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
      ctx.setTransform(canvas.width/1600,0,0,canvas.height/900,0,0);
      ctx.fillStyle='#03111c';ctx.fillRect(0,0,1600,900);
      for(const [x,y,r,color] of [[410,150,490,'#075166'],[1320,330,370,'#44312c'],[730,430,370,'#06354e'],[1100,60,300,'#302339']]){
        const g=ctx.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,color);g.addColorStop(1,'transparent');ctx.fillStyle=g;ctx.fillRect(0,0,1600,900);
      }
      ctx.drawImage(artwork.current,0,0,1600,900);
      stars.forEach(([x,y,r,p])=>{ctx.globalAlpha=.32+.55*(.5+.5*Math.sin(t*.22+p*15));ctx.fillStyle='#d9f3ff';ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill();if(r>1.43){ctx.fillStyle='#b9efff50';ctx.fillRect(x-3,y-.3,6,.6);ctx.fillRect(x-.3,y-3,.6,6);}});ctx.globalAlpha=1;
      // Wider grid and luminous floor projection, as in the reference.
      ctx.strokeStyle='#5192ab60';ctx.lineWidth=.85;
      for(let x=-2000;x<3600;x+=210)path([[790+(x-790)*.35,505],[x,900]],'#5d8c9e45');
      for(let i=0;i<13;i++){const y=510+Math.pow(i/12,1.7)*390;path([[0,y],[1600,y]],'#6792a045');}
      // Layered, gently travelling waves; stable envelope avoids a spinning floor.
      ctx.globalAlpha=waveGain;
      for(let band=0;band<3;band++){
        const points=[];
        for(let i=0;i<=500;i++){
          const a=Math.PI+i/500*Math.PI*2,x=790+Math.cos(a)*(620+band*68),y=850+Math.sin(a)*(240+band*26);
          const ripple=Math.sin(i*.46-t*.65+band)*4;
          points.push([x,y+ripple]);
        }
        ctx.shadowBlur=12;ctx.shadowColor='#71dfff';path(points,'#52c9ff45',7);path(points,'#a9ecffb0',1.5);
        for(let i=0;i<points.length;i+=3){const [x,y]=points[i];ctx.fillStyle='#b9f2ffb0';ctx.fillRect(x,y,1.4,3+Math.sin(i*.4-t*.6)*1.5);}
      }
      ctx.globalAlpha=1;
      for(let i=1;i<=9;i++){ctx.shadowBlur=8;ellipse(790,729,i*24,i*6.3,0,i<5?'#94e9ffc0':'#7ad8ff60',i===4?2:1);}
      path([[790,520],[790,729]],'#67dcff75',1.2);
      // The source's floor beam intensifies as the wave envelope recedes.
      ctx.globalAlpha=Math.max(.1,(1-waveGain)*.65)*gain;
      for(let beam=0;beam<11;beam++){
        const x=790+(beam-5)*5;path([[x,729],[x,590+(beam%4)*20]],'#8eeaff75',beam%3?1:2);
      }
      ctx.globalAlpha=1;
      ctx.shadowBlur=0;
      scene.tools.forEach((tool,i)=>{
        const x=(tool.x+1.7)*16,y=(tool.y+3)*9;
        for(let strand=0;strand<2;strand++){
          ctx.beginPath();ctx.moveTo(790,400+strand*18);ctx.bezierCurveTo(790+(x-790)*.6,390+(i%3)*35,x+(strand?18:-10),y+110,x,y);ctx.strokeStyle=strand?'#a2e9ff62':tool.color+'95';ctx.lineWidth=strand?.6:1.3;ctx.stroke();
        }
      });
      ctx.globalAlpha=gain;
      const halo=ctx.createRadialGradient(790,400,25,790,400,180);halo.addColorStop(0,'#89eaff50');halo.addColorStop(.6,'#288de530');halo.addColorStop(1,'transparent');ctx.fillStyle=halo;ctx.fillRect(605,215,370,370);
      ctx.globalAlpha=1;
      if(sprite.complete&&sprite.naturalWidth){
        const paint=(sampleIndex,alpha)=>{
          const tile=motion.sphere_atlas.tile_size,columns=motion.sphere_atlas.columns;
          ctx.globalAlpha=alpha;ctx.drawImage(sprite,(sampleIndex%columns)*tile,Math.floor(sampleIndex/columns)*tile,tile,tile,...motion.sphere_atlas.draw_rect);
        };
        paint(index,1);paint(Math.min(index+1,motion.samples.length-1),blend);
        canvas.dataset.artworkLoaded='true';
      }
      ctx.globalAlpha=1;
      // Crossfade observed source contours. This matches 2D ring silhouettes and
      // timing without claiming the contours establish 3D rotation or topology.
      const ringLayer=(sample,alpha)=>{
        if(alpha<=0)return;
        ctx.globalAlpha=alpha;ctx.shadowColor='#ffe2a0';ctx.shadowBlur=9;
        sample.rings.forEach(ring=>{
          ellipse(ring.cx,ring.cy,ring.rx,ring.ry,ring.angle,'#d9c29a65',6);
          ellipse(ring.cx,ring.cy,ring.rx,ring.ry,ring.angle,'#f9e9bd',1.5);
          ellipse(ring.cx,ring.cy,ring.rx+4,ring.ry+3,ring.angle,'#b8d9ed50',.8);
        });
      };
      ringLayer(observation,1-blend);ringLayer(next,blend);ctx.globalAlpha=1;
      ctx.shadowBlur=0;
      if(!paused&&!reduced&&t<motion.samples.at(-1).time)raf=requestAnimationFrame(draw);
    };
    const resize=()=>{if(paused||reduced||phase.current>=motion.samples.at(-1).time)draw(performance.now());};
    const change=e=>{reduced=e.matches;cancelAnimationFrame(raf);previous=undefined;raf=requestAnimationFrame(draw);};
    sprite.onload=()=>{if(paused||reduced||phase.current>=motion.samples.at(-1).time)draw(performance.now());};sprite.src=motion.sphere_atlas.src;
    raf=requestAnimationFrame(draw);window.addEventListener('resize',resize);preference.addEventListener('change',change);
    return ()=>{sprite.onload=null;cancelAnimationFrame(raf);window.removeEventListener('resize',resize);preference.removeEventListener('change',change);};
  },[paused]);
  return <canvas ref={ref} className="universe" role="img" aria-label="Preserved animated sphere artwork with fitted ring paths, source-timed glow and floor waves"/>;
}

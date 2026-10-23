"""Deterministic React output with explicit artwork preservation and editable view."""
import html
import json
import re
from pathlib import Path

from PIL import Image

VOID_TAGS = {"input", "img", "br", "hr", "source"}
ALLOWED_TAGS = {"div","span","button","input","img","nav","aside","header","footer",
                "ul","li","a","select","video","i","label","p","section"}
CONTROL_TAGS = {"button", "input", "select", "a"}


def literal(value):
    # JSX expression containers keep OCR text as data, including braces and '<'.
    return "{" + json.dumps(str(value), ensure_ascii=True) + "}"


def render_component(tree, component_name, reference_src=None, artwork_src=None):
    rules = []
    hotspots = []
    def roles(node):
        return {node.get('tag')} | set().union(*(roles(c) for c in node.get('children',[])))
    def grounded_form(node):
        return bool(node.get('attributes',{}).get('data-role-evidence')) or any(grounded_form(c) for c in node.get('children',[]))

    def render(node):
        cls, text = node["class_name"], node.get("text")
        tag = node.get("tag") if node.get("tag") in ALLOWED_TAGS else "div"
        child_roles=roles(node)
        form_owner=tag=='div' and grounded_form(node) and {'input','button'}<=child_roles and not any({'input','button'}<=roles(c) for c in node.get('children',[]))
        attrs = f' className="{cls}" data-node-id={literal(node["node_id"])} data-element-type={literal(node.get("element_type") or "unknown")}'
        if form_owner:
            attrs += ' data-rebuilt-panel="true"'
            if len(node.get('children',[]))==2 and {c.get('tag') for c in node['children']}=={'input','button'}:
                attrs += ' data-layout="row-form"'
        styles = dict(node["css"])
        rules.append(f".{cls} {{ " + " ".join(f"{k}: {v};" for k,v in styles.items()) + " }")
        if tag in CONTROL_TAGS:
            attrs += f' aria-label={literal(node.get("attributes",{}).get("aria-label") or text or node.get("element_type") or tag)}'
            if tag in {"button", "a"}:
                attrs += f' onClick={{() => onAction?.({json.dumps(node["node_id"])})}}'
                if tag == "button":
                    attrs += ' type="submit"' if node.get('attributes',{}).get('data-role-evidence') else ' type="button"'
                else:
                    attrs += ' role="button" tabIndex={0}'
                    attrs += ' onKeyDown={(e) => { if (e.key === "Enter") onAction?.(' + json.dumps(node["node_id"]) + '); }}'
            if tag == "input":
                input_type = node.get("attributes",{}).get("type","text")
                input_type = input_type if input_type in {"url","text","password","checkbox","radio","range","date","number","email","search"} else "text"
                attrs += f' type="{input_type}"'
                attrs += ' name='+literal(node['node_id'])
                if node.get('attributes',{}).get('data-role-evidence'):
                    attrs += ' required'
                if input_type not in {"checkbox","radio","range"}:
                    attrs += f' placeholder={literal(text or "")}'
            if tag == "select":
                control_content = f'<option value="">{literal(text or "Select")}</option>'
            else:
                control_content = literal(text) if text and tag not in VOID_TAGS else ""
            hotspot_attrs = attrs.replace(f'className="{cls}"', f'className="{cls} omniui-hotspot"')
            box, root = node["bbox"], tree["bbox"]
            position = {"left":f"{(box['x']-root['x'])/max(root['width'],1)*100:.5f}%",
                        "top":f"{(box['y']-root['y'])/max(root['height'],1)*100:.5f}%",
                        "width":f"{box['width']/max(root['width'],1)*100:.5f}%",
                        "height":f"{box['height']/max(root['height'],1)*100:.5f}%"}
            hotspot_attrs += " style={" + json.dumps(position) + "}"
            hotspots.append(f"<{tag}{hotspot_attrs}" + (" />" if tag in VOID_TAGS else f">{control_content}</{tag}>"))
        if tag == "img":
            src = node.get("asset_src")
            if src:
                attrs += f' src={literal(src)} alt={literal(text or "")}'
            else:
                # Never generate a broken image URL.
                tag = "div"
        content = ""
        if tag == "select":
            content = f'<option value="">{literal(text or "Select")}</option>'
        elif tag not in VOID_TAGS:
            content = (literal(text) if text else "") + "".join(render(c) for c in node.get("children",[]))
            if tag in {'button','a'} and text and node.get('attributes',{}).get('data-text-bbox') and not node.get('children'):
                tx,ty,tw,th=json.loads(node['attributes']['data-text-bbox']);b=node['bbox']
                position={'left':f"{(tx-b['x'])/max(b['width'],1)*100:.5f}%",'top':f"{(ty-b['y'])/max(b['height'],1)*100:.5f}%",'width':f"{tw/max(b['width'],1)*100:.5f}%",'height':f"{th/max(b['height'],1)*100:.5f}%"}
                content='<span className="omniui-control-label" data-owner-node-id='+literal(node['node_id'])+' style={'+json.dumps(position)+'}>'+literal(text)+'</span>'
        output_tag='MeasuredForm' if form_owner else 'MeasuredPanel' if tag=='div' and node.get('element_type')=='container' else tag
        return f"<{output_tag}{attrs}" + (" />" if tag in VOID_TAGS else f">{content}</{output_tag}>")

    # Render children without placing them inside a second root-sized box.
    root_cls = tree["class_name"]
    rules.append(f".{root_cls} {{ " + " ".join(f"{k}: {v};" for k,v in tree["css"].items()) + " }")
    editable = "".join(render(child) for child in tree.get("children",[]))
    if tree.get("text"):
        editable = literal(tree["text"]) + editable
    reference = f'<img className="omniui-reference-layer" src={literal(reference_src)} alt="" aria-hidden="true" />' if reference_src else ""
    effective_mode = 'mode' if reference_src else '"editable"'
    artwork = (f'<img className="omniui-artwork-layer" src={literal(artwork_src)} alt="" aria-hidden="true" />' if artwork_src else '')
    jsx = ("import React from 'react';\nimport generatedArtwork from './generated_artwork.json';\nimport {MeasuredPanel,MeasuredForm} from './MeasuredComponents';\nimport ArtworkMotion from './ArtworkMotion';\nimport {useMeasuredTypography} from './useMeasuredTypography';\n\n"
           f"export default function {component_name}({{ mode = 'hybrid', onAction, artworkTime = 0 }}) {{\n"
           "  const stage=React.useRef(null);useMeasuredTypography(stage,mode);\n"
           f"  return (<div ref={{stage}} className=\"{root_cls} omniui-stage\" data-mode={{{effective_mode}}} data-state=\"{component_name}\">"
           f"{reference}{artwork}{{mode === 'hybrid' && <ArtworkMotion time={{artworkTime}} />}}<div className=\"omniui-editable-layer\">{editable}</div>"
           f"{{mode === 'hybrid' && generatedArtwork.state_id?.replaceAll('_','').toLowerCase() === {json.dumps(component_name.lower())} && new URLSearchParams(location.search).get('artwork') === 'generated' && new URLSearchParams(location.search).get('capture') !== '1' && <video className=\"omniui-generated-artwork\" src={{generatedArtwork.src}} style={{{{maskImage:'url('+generatedArtwork.mask+')',maskSize:'100% 100%'}}}} autoPlay={{!matchMedia('(prefers-reduced-motion: reduce)').matches}} muted playsInline />}}"
           f"<div className=\"omniui-hotspot-layer\">{''.join(hotspots)}</div></div>);\n}}\n")
    return jsx, "\n".join(rules)


def collect_assets(layout, tree):
    assets = {}
    reference_src = None
    source = Path(layout.source_frame.image_path)
    if source.is_file():
        reference_src = f"assets/{layout.state_name}-reference.png"
        # Re-encode rather than assuming the source extension/content is PNG.
        import io
        with Image.open(source) as image:
            image = image.convert("RGB")
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            assets[reference_src] = stream.getvalue()

            def walk(node):
                if node.get("tag") in {"img","i"} or node.get("element_type") in {"image","icon","avatar","canvas","3d_view","webgl"}:
                    box = node["bbox"]
                    crop = image.crop((int(box["x"]),int(box["y"]),int(box["x"]+box["width"]),int(box["y"]+box["height"])))
                    if crop.width and crop.height:
                        path = f"assets/{node['class_name']}.png"
                        stream = io.BytesIO(); crop.save(stream,format="PNG")
                        assets[path] = stream.getvalue()
                        if node.get("children"):
                            node["css"].update({"background-image":f'url("{path}")', "background-size":"100% 100%"})
                        else:
                            node["tag"],node["asset_src"] = "img",path
                for child in node.get("children",[]):
                    walk(child)
            walk(tree)
    if source.is_file():
        from app.modules.artwork_extractor import static_artwork
        artwork, _ = static_artwork(layout)
        assets[f"assets/{layout.state_name}-artwork.png"] = artwork
    return assets, reference_src


BASE_CSS = """* { box-sizing: border-box; }
html, body, #root { margin: 0; min-height: 100%; }
body { background: #10151e; color: #e7edf5; font-family: Arial, sans-serif; }
button, input, select { font: inherit; }
form {margin:0;}
.omniui-form-status {position:absolute;top:100%;left:0;z-index:10;background:#10151e;color:#fff;padding:6px;font-size:12px;}
.omniui-editable-layer [data-rebuilt-panel="true"] {border-radius:.9cqw;background-image:linear-gradient(110deg,#ffffff14,#00000030);}
.omniui-intro {position:relative;width:100%;background:#000;}
.omniui-intro video {display:block;width:100%;max-height:100dvh;object-fit:contain;}
.omniui-intro button {position:absolute;right:12px;bottom:12px;padding:8px;}
@container (max-width:600px) {
  .omniui-editable-layer [data-layout="row-form"] {display:grid;grid-template-columns:minmax(60px,1fr) minmax(64px,auto);gap:4px;padding:4px;min-height:40px;align-items:center;}
  .omniui-editable-layer [data-layout="row-form"]>input,.omniui-editable-layer [data-layout="row-form"]>button {position:static;width:100%;height:32px;min-width:0;font-size:clamp(9px,1.5cqw,14px);}
  .omniui-editable-layer [data-layout="row-form"]>button {padding:4px;}
  .omniui-editable-layer [data-layout="row-form"] .omniui-control-label {position:static;width:auto!important;height:auto!important;}
}
.omniui-stage { container-type:inline-size; overflow: hidden; isolation: isolate; flex: none; }
.omniui-stage :where(button,input,select) { margin: 0; padding: 0; border: 0; border-radius: 0; }
.omniui-artwork-layer,.omniui-artwork-motion {position:absolute;inset:0;width:100%;height:100%;pointer-events:none;}
.omniui-generated-artwork {position:absolute;inset:0;width:100%;height:100%;pointer-events:none;object-fit:fill;}
.omniui-stage:not([data-mode="hybrid"]) .omniui-artwork-layer {display:none;}
.omniui-stage[data-mode="hybrid"] .omniui-reference-layer,.omniui-stage[data-mode="hybrid"] .omniui-hotspot-layer {display:none;}
.omniui-stage[data-mode="hybrid"] .omniui-editable-layer [data-element-type="container"]:not([data-rebuilt-panel="true"]) {background:transparent;border-color:transparent;}
.omniui-editable-layer :is(span,p,label) {white-space:nowrap;overflow:hidden;display:flex;align-items:center;line-height:1;}
.omniui-control-label {position:absolute;pointer-events:none;text-align:left;}
.omniui-reference-layer { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: fill; pointer-events: none; }
.omniui-editable-layer, .omniui-hotspot-layer { position: absolute; inset: 0; }
.omniui-stage[data-mode="asset-backed"] .omniui-editable-layer { display: none; }
.omniui-stage[data-mode="editable"] .omniui-reference-layer,
.omniui-stage[data-mode="editable"] .omniui-hotspot-layer { display: none; }
.omniui-hotspot { opacity: 0; cursor: pointer; }
.omniui-hotspot:focus-visible { opacity: 1; outline: 2px solid #59bbff; outline-offset: 2px; }
.omniui-hotspot:is(input,select):focus { opacity: 1; }
.omniui-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 12px 20px; background: #192331; }
.omniui-toolbar button, .omniui-toolbar select { padding: 6px 10px; }
.omniui-view { width: 100%; overflow: auto; }
.omniui-note { padding: 8px 20px; margin: 0; color: #aab8cd; font-size: 13px; }
body[data-capture="true"] .omniui-toolbar, body[data-capture="true"] .omniui-note { display: none; }
"""


def render_app(layouts, default_mode="asset-backed", motion_src=None):
    names = ["".join(x.capitalize() for x in re.split(r"[^a-zA-Z0-9]+", l.state_name) if x) for l in layouts]
    imports = "\n".join(f"import {n} from './{n}';" for n in names)
    meta = [{"id":l.state_name,"time":l.source_frame.timestamp_sec} for l in layouts]
    default_index = 0
    return imports + "\nimport React, { useState, useEffect } from 'react';\nimport actions from './interaction_map.json';\nimport artworkManifest from './motion_manifest.json';\nimport scenePlan from './scene_manifest.json';\nimport './styles.css';\n" + f"const components = [{', '.join(names)}];\nconst states = {json.dumps(meta)};\n" + r"""
export default function App() {
  const params = new URLSearchParams(window.location.search);
  const inspect = params.get('inspect') === '1' && params.get('capture') !== '1';
  const [intro,setIntro]=useState(() => !!scenePlan.intro && !inspect && params.get('capture') !== '1' && !params.has('state') && !matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [index, setIndex] = useState(() => params.has('state') ? Math.max(0, states.findIndex(s => s.id === params.get('state'))) : Math.max(0,states.findIndex(s=>s.id===scenePlan.entry_state)));
  const [mode, setMode] = useState(['hybrid','editable','asset-backed'].includes(params.get('mode')) ? params.get('mode') : DEFAULT_MODE);
  const [playing, setPlaying] = useState(false);
  const [motion, setMotion] = useState(false);
  const [artworkTime,setArtworkTime] = useState(() => states[index]?.time||0);
  const [artworkPlaying,setArtworkPlaying] = useState(() => !inspect && params.get('capture') !== '1' && artworkManifest.regions.length > 0 && !matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [actionStatus,setActionStatus] = useState('');
  const reviewMatch=window.location.pathname.match(/^\/jobs\/([A-Za-z0-9_-]+)\/preview\//);
  useEffect(()=>{if(!artworkPlaying)setArtworkTime(states[index].time);setActionStatus('');},[index]);
  useEffect(()=>{
    if(!artworkPlaying || !inspect)return;
    let next=0;states.forEach((state,i)=>{if(state.time<=artworkTime)next=i;});setIndex(next);
  },[artworkTime,artworkPlaying]);
  useEffect(()=>{
    const preference=matchMedia('(prefers-reduced-motion: reduce)');
    const stop=()=>{if(preference.matches)setArtworkPlaying(false);};
    preference.addEventListener('change',stop);return()=>preference.removeEventListener('change',stop);
  },[]);
  useEffect(()=>{
    if(!artworkPlaying)return;
    let start,raf;
    const initial=artworkTime,last=artworkManifest.sample_times.at(-1)||0;
    const tick=now=>{
      if(start===undefined)start=now;
      const next=Math.min(last,initial+(now-start)/1000);
      setArtworkTime(next);
      if(next>=last){setArtworkPlaying(false);return;}
      raf=requestAnimationFrame(tick);
    };
    raf=requestAnimationFrame(tick);return()=>cancelAnimationFrame(raf);
  },[artworkPlaying]);
  const syncMotion = e => {
    const time = e.currentTarget.currentTime;
    let next = 0;
    states.forEach((s,i) => { if (s.time <= time) next = i; });
    setIndex(next);
  };
  useEffect(() => { document.body.dataset.capture = params.get('capture') === '1' ? 'true' : 'false'; }, []);
  useEffect(() => {
    if (!playing || index >= states.length-1) { if (playing) setPlaying(false); return; }
    const delay = Math.max(100, (states[index+1].time-states[index].time)*1000);
    const timer = setTimeout(() => setIndex(i => i+1), delay);
    return () => clearTimeout(timer);
  }, [playing,index]);
  const Current = components[index];
  const onAction = id => {
    const target = actions[states[index].id]?.[id];
    const next = states.findIndex(s => s.id === target);
    if (next >= 0) { setPlaying(false);setArtworkPlaying(false);setActionStatus('');setIndex(next); }
    else setActionStatus('No confirmed action is assigned to this control. Add its transition in the reconstruction review.');
  };
  if(intro)return <main className="omniui-intro"><video src={scenePlan.intro.src} autoPlay muted playsInline onTimeUpdate={e=>{if(e.currentTarget.currentTime>=scenePlan.intro.end)setIntro(false);}} onEnded={()=>setIntro(false)} onError={()=>setIntro(false)} /><button onClick={()=>setIntro(false)}>Skip intro</button></main>;
  return <>
    {inspect && <><header className="omniui-toolbar">
      <strong>Recording reconstruction</strong>
      {reviewMatch && <a href={'/jobs/'+reviewMatch[1]+'/review/ui'}>Review reconstruction</a>}
      {artworkManifest.regions.length>0 && <button onClick={()=>{setMode('hybrid');setMotion(false);if(artworkTime>=(artworkManifest.sample_times.at(-1)||0))setArtworkTime(0);setArtworkPlaying(!artworkPlaying);}}>{artworkPlaying?'Pause artwork':'Play isolated artwork'}</button>}
      {MOTION_SRC && <button onClick={() => {setPlaying(false);setMotion(!motion);setMode('asset-backed');}}> {motion ? 'Stop recorded motion' : 'Play recorded motion'} </button>}
      <button disabled={!index} onClick={() => {setPlaying(false);setIndex(index-1);}}>Previous</button>
      <button onClick={() => {if(index === states.length-1) setIndex(0);setPlaying(!playing);}}>{playing ? 'Pause' : 'Play states'}</button>
      <button disabled={index === states.length-1} onClick={() => {setPlaying(false);setIndex(index+1);}}>Next</button>
      <select aria-label="Captured state" value={index} onChange={e => {setPlaying(false);setIndex(Number(e.target.value));}}>
        {states.map((s,i) => <option key={s.id} value={i}>{s.time.toFixed(2)}s â€” {s.id}</option>)}
      </select>
      <select aria-label="Rendering mode" value={mode} onChange={e => setMode(e.target.value)}>
        <option value="hybrid">Rebuilt controls + isolated artwork</option><option value="asset-backed">Preserved artwork + controls</option><option value="editable">Editable approximation</option>
      </select>
    </header>
    <p className="omniui-note">Hybrid view rebuilds measured text/controls and preserves remaining artwork. Motion and semantics require review. Reference view preserves captured pixels with overlays. Editable view uses measured HTML/CSS. State playback follows captured timestamps; application transitions require confirmed mappings.</p>
    {actionStatus && <p className="omniui-note" role="status">{actionStatus}</p>}
    </>}
    <main className="omniui-view" style={{position:'relative'}}>
      <div className={motion ? 'omniui-motion-active' : ''} style={{position:'relative',width:'fit-content',maxWidth:'100%'}}>
        <Current key={states[index].id} mode={mode} onAction={onAction} artworkTime={artworkTime} />
        {motion && <video className="omniui-motion-video" src={MOTION_SRC} autoPlay muted playsInline onTimeUpdate={syncMotion} onEnded={() => setMotion(false)} />}
      </div>
    </main>
  </>;
}
""".replace("MOTION_SRC", json.dumps(motion_src)).replace("DEFAULT_MODE", json.dumps(default_mode)).replace("DEFAULT_INDEX",str(default_index))


def attach_recorded_motion(files, layouts, video_path, default_mode="asset-backed"):
    """Preserve source motion explicitly; no claim of recovered geometry/animation."""
    source = Path(video_path)
    # Browser-native formats only. Unsupported recordings keep the still-state view.
    if source.suffix.lower() not in {".mp4", ".webm"}:
        return False
    asset = "assets/recording" + source.suffix.lower()
    files["public/" + asset] = source.read_bytes()
    files["App.jsx"] = render_app(layouts, default_mode, asset)
    files["styles.css"] += "\n.omniui-motion-video {position:absolute;inset:0;width:100%;height:100%;object-fit:fill;pointer-events:none;z-index:1;}\n.omniui-motion-active .omniui-stage {z-index:2;background:transparent!important;}\n.omniui-motion-active .omniui-reference-layer {visibility:hidden;}\n"
    files["README.md"] += "\nRecorded motion replays the packaged source video beneath native control overlays. It is preserved footage, not synthesized animation or recovered 3D. Control positions follow sampled states; motion between samples is not tracked.\n"
    return True

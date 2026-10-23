import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import scene from './scene.json';
import './styles.css';
import Universe from './universe';

function Icon({name, ...props}) {
  const paths = {
    video: <><rect x="3" y="6" width="12" height="12" rx="2"/><path d="m15 10 6-4v12l-6-4Z"/><path d="m8 10 4 2-4 2Z"/></>,
    hex: <><path d="m12 2 9 5v10l-9 5-9-5V7Z"/><path d="m12 6 5 3v6l-5 3-5-3V9Z"/></>,
    aperture: <><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="m12 3 4 6m5 3-7 2m-2 7-3-7M3 12l7-3"/></>,
    audio: <><path d="M8 4a4 4 0 0 1 4 4v8a4 4 0 0 1-4 4M16 4a4 4 0 0 0-4 4v8a4 4 0 0 0 4 4M6 8v8m12-8v8M3 10v4m18-4v4"/></>,
    pen: <><path d="M4 3h16v18H4ZM8 7l4 10 4-10ZM12 17v4"/></>,
    wave: <><circle cx="12" cy="12" r="10"/><path d="M8 7v10m4-13v16m4-13v10M5 10v4m14-4v4"/></>,
    globe: <><circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18M5 7h14M5 17h14"/></>,
    file: <><path d="M6 2h9l4 4v16H6ZM14 2v5h5M9 11h7m-7 4h7m-7 4h5"/></>,
    moon: <path d="M20 15A9 9 0 0 1 9 3a9 9 0 1 0 11 12Z"/>,
    shield: <><path d="m12 2 8 4v6c0 5-8 10-8 10S4 17 4 12V6Z"/><path d="m8 11 3 3 5-6"/></>,
    attach: <path d="m8 12 7-7a4 4 0 0 1 6 6L10 22a6 6 0 0 1-8-8L13 3m-9 13 9-9"/>,
    mic: <><rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8"/></>,
  };
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name] || paths.hex}</svg>;
}

function App() {
  const inspect=new URLSearchParams(location.search).get('inspect')==='1';
  const [replay,setReplay]=useState(0);
  const [selected,setSelected]=useState(null),[intent,setIntent]=useState(''),[events,setEvents]=useState(['Local cockpit ready. Select a tool or enter an intent.']),[files,setFiles]=useState([]),[paused,setPaused]=useState(false),[voice,setVoice]=useState(false);
  const [clock,setClock]=useState(new Date(2026,0,1,23,58)),[notice,setNotice]=useState('');
  const fileInput=useRef(null),dialog=useRef(null),recognition=useRef(null),trigger=useRef(null),eventId=useRef(0);
  const log=text=>{eventId.current+=1;setEvents(old=>[`${new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})} · ${text}`, ...old].slice(0,20));};
  useEffect(()=>{const id=setInterval(()=>setClock(old=>new Date(old.getTime()+1000)),1000);return ()=>clearInterval(id);},[]);
  useEffect(()=>{if(selected)dialog.current?.showModal();},[selected]);
  useEffect(()=>()=>recognition.current?.abort(),[]);
  const openTool=(tool,event)=>{trigger.current=event.currentTarget;setSelected(tool);};
  const closeTool=()=>{dialog.current?.close();setSelected(null);trigger.current?.focus();};
  const submit=e=>{e.preventDefault();if(!intent.trim())return;log(`Intent queued locally${selected?' for '+selected.name:''}: ${intent.trim()}${files.length?' · '+files.length+' attachment(s)':''}`);setNotice('Intent added to the local activity feed. No external agent is connected.');setIntent('');setFiles([]);};
  const toggleVoice=()=>{
    if(voice){recognition.current?.stop();return;}
    const Speech=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!Speech){setNotice('Speech recognition is unavailable in this browser. You can type your intent.');return;}
    const listener=new Speech();recognition.current=listener;listener.lang='en-US';listener.onresult=e=>setIntent(old=>old+' '+e.results[0][0].transcript);listener.onend=()=>setVoice(false);listener.onerror=e=>{setVoice(false);setNotice(`Voice input: ${e.error}.`);};
    try{listener.start();setVoice(true);}catch{setNotice('Voice input could not start.');}
  };
  return <>
    {inspect && <div className="review-bar"><a href="/">OmniUI</a><span>Rebuilt controls · preserved sphere artwork · measured motion</span><button onClick={()=>setPaused(!paused)}>{paused?'Resume animation':'Pause animation'}</button><button onClick={()=>{history.replaceState(null,'',location.pathname+'?inspect=1');setReplay(value=>value+1);setPaused(false);}}>Replay source timing</button></div>}
    <main className="cockpit" aria-label="Forge Hub cockpit">
      <Universe key={replay} paused={paused}/>
      <aside className="telemetry panel">
        <div className="panel-title">CORE BRAIN OS – CONTROL HUB</div>
        <time className="clock">{clock.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',hour12:false})}</time>
        <h1>{scene.title}</h1><p className="muted">SYSTEM TELEMETRY FUST + PYTHON + REACT</p><p>WELCOME, RUBEN</p>
        <div className="divider"/><p className="muted">TOPOLOGY DATA – RUST + PYTHON<br/>DATABLOCKS, INTERNAL PROCESS</p>
        <div className="divider"/><p className="stat"><Icon name="hex"/><span>KERNEL: <b>ONLINE</b></span></p><p className="stat"><Icon name="globe"/><span>SATELLITES: <em>12</em></span></p>
        <p className="lead">MASTER LEAD — Ruben (Executive Architect)</p>
      </aside>
      <nav aria-label="Available tools" className="tools">
        {scene.tools.map(tool=><button key={tool.id} className={'tool '+(selected?.id===tool.id?'selected':'')} style={{left:tool.x+'%',top:tool.y+'%',width:tool.width+'%', '--accent':tool.color}} onClick={e=>openTool(tool,e)}>
          <span className="tool-emblem"><img src={`./icons/${tool.id}.png`} alt=""/></span><span><strong>{tool.name}</strong><small>{tool.detail}</small></span><span className="launch-mark">↗</span>
        </button>)}
      </nav>
      <p className="sphere-caption">{notice?'LOCAL INTENT RECEIVED':'WAVES STATIONARY — AWAITING INTENT ORDER'}</p>
      <section className="feed panel" aria-label="Activity feed"><div className="panel-title">FORGE-HUB INTENT ACTIVITY FEED</div><ol aria-live="polite">{events.map((event,i)=><li key={eventId.current-i}>{event}</li>)}</ol></section>
      <form className="command panel" onSubmit={submit}>
        <input ref={fileInput} type="file" multiple hidden onChange={e=>{setFiles(Array.from(e.target.files));log(`${e.target.files.length} attachment(s) selected locally`);e.target.value='';}}/>
        <button type="button" className="console-btn" onClick={()=>fileInput.current.click()}><Icon name="attach"/> {files.length?`${files.length} file(s)`:'Attach File / Image'}</button>
        <button type="button" className={'console-btn '+(voice?'listening':'')} onClick={toggleVoice}><Icon name="mic"/>Voice: {voice?'Listening':'Standby'}</button>
        <input className="intent" aria-label="Intent" placeholder="Enter your intent, add context, then execute…" value={intent} onChange={e=>setIntent(e.target.value)} required/>
        <button type="submit" className="execute">Execute Intent</button>
      </form>
    </main>
    {inspect && <p className="status-message" role="status">{notice||'Select any tool to inspect it. Actions run locally; external tools are not connected.'}</p>}
    <dialog ref={dialog} onCancel={closeTool} onClose={()=>{setSelected(null);trigger.current?.focus();}}>
      {selected&&<><div className="dialog-heading"><Icon name={selected.icon}/><div><h2>{selected.name}</h2><p>{selected.detail} workspace</p></div></div>
      <p>This reconstructed control opens a local workspace. The original service API is not connected.</p>
      <label>Task for {selected.name}<textarea aria-label="Tool task" value={intent} onChange={e=>setIntent(e.target.value)} placeholder="Describe a task…"/></label>
      <div className="dialog-actions"><button onClick={closeTool}>Close</button><button className="execute" disabled={!intent.trim()} onClick={()=>{log(`${selected.name}: ${intent.trim()} (local task)`);setNotice(`Task added locally for ${selected.name}.`);setIntent('');closeTool();}}>Add local task</button></div></>}
    </dialog>
  </>;
}
createRoot(document.getElementById('root')).render(<App/>);

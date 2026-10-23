document.addEventListener('DOMContentLoaded',async()=>{
  const $=id=>document.getElementById(id);
  const job=location.pathname.match(/^\/jobs\/([A-Za-z0-9_-]+)\/review\/ui$/)?.[1];
  const corrections=new Map(),transitions=new Map(),excluded=new Set();let data;
  const status=text=>{$('status').textContent=text;};
  const selectedState=()=>data.states.find(state=>state.id===$('state').value);
  const selectedNode=()=>selectedState()?.nodes.find(node=>node.id===$('node').value);
  const key=()=>[$('state').value,$('node').value].join(':');
  const draft=()=>({corrections:[...corrections.values()],transitions:[...transitions.values()],exclude_motion_regions:[...excluded],clear_transitions:$('clearTransitions').checked,note:$('note').value});
  const showDraft=()=>{$('draft').textContent=JSON.stringify(draft(),null,2);};
  const option=(select,value,label)=>{const item=document.createElement('option');item.value=value;item.textContent=label;select.append(item);};
  const overlay=()=>{for(const name of ['x','y','width','height'])$('box').setAttribute(name,$(name).value);};
  const syncInputType=()=>{$('inputType').disabled=$('role').value!=='input';if($('inputType').disabled)$('inputType').value='';};
  const showNode=()=>{
    const node=selectedNode();if(!node)return;
    const edit=corrections.get(key());const desired=edit?.element_type||node.role||'text';$('role').value=[...$('role').options].some(option=>option.value===desired)?desired:node.tag==='div'?'container':'text';
    $('text').value=edit&&Object.hasOwn(edit,'text_content')?edit.text_content??'':node.text??'';
    for(const name of ['x','y','width','height'])$(name).value=(edit?.bbox||node.bbox)[name];
    $('font').value=edit?.font_size??(parseFloat(node.styles['font-size'])||'');
    $('color').value=edit?.color??node.styles.color??'';$('inputType').value=edit?.input_type??'';
    const findings=(data.verification.output_verification?.states||[]).filter(state=>state.state_name===$('state').value&&state.mode==='hybrid').flatMap(state=>state.layout_findings||[]).filter(finding=>finding.node_id===node.id||finding.other_node_id===node.id);
    $('nodeInfo').textContent=`${node.id} · detection score ${Number(node.confidence).toFixed(2)}. Confirm the role visually. ${findings.map(finding=>finding.detail).join(' ')||'Groups with children must remain containers.'}`;syncInputType();overlay();
  };
  const showState=()=>{
    const state=selectedState();$('frame').src=state.frame_url;$('overlay').setAttribute('viewBox',`0 0 ${state.size.join(' ')}`);
    $('node').replaceChildren();for(const node of state.nodes)option($('node'),node.id,`${node.role||node.tag} · ${(node.text||node.id).slice(0,70)}`);showNode();
  };
  $('state').addEventListener('change',showState);$('node').addEventListener('change',showNode);
  $('role').addEventListener('change',syncInputType);
  for(const name of ['x','y','width','height'])$(name).addEventListener('input',overlay);
  $('stage').addEventListener('click',()=>{
    const node=selectedNode();if(!node)return;
    const patch={state_id:$('state').value,node_id:node.id,element_type:$('role').value,text_content:$('text').value||null};
    const bounds=Object.fromEntries(['x','y','width','height'].map(name=>[name,Number($(name).value)]));
    if(Object.keys(bounds).some(name=>bounds[name]!==node.bbox[name]))patch.bbox=bounds;
    if($('font').value)patch.font_size=Number($('font').value);if($('color').value)patch.color=$('color').value;
    if($('inputType').value)patch.input_type=$('inputType').value;
    corrections.set(key(),patch);showDraft();status('Correction added to draft. Build the revision to apply it.');
  });
  $('transition').addEventListener('click',()=>{if(!selectedNode())return;transitions.set(key(),{state_id:$('state').value,node_id:$('node').value,target_state:$('target').value});showDraft();status('Transition added to draft; it will be validated during the build.');});
  $('clearTransitions').addEventListener('change',showDraft);$('note').addEventListener('input',showDraft);
  $('build').addEventListener('click',async()=>{
    $('build').disabled=true;$('result').replaceChildren();
    try{
      status('Validating corrections…');const response=await fetch(`/jobs/${job}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(draft())});
      const revision=await response.json();if(!response.ok)throw Error(JSON.stringify(revision.detail));
      let current=revision;status(`Building ${revision.job_id}…`);
      while(!['complete','failed'].includes(current.status)){
        await new Promise(resolve=>setTimeout(resolve,1200));const check=await fetch(`/jobs/${revision.job_id}`);if(!check.ok)throw Error('Could not read revision status');current=await check.json();status(`${current.current_stage_detail||current.status} · ${revision.job_id}`);
      }
      if(current.status==='failed')throw Error(current.error_message||'Revision failed');
      for(const [label,url] of [['Open reviewed UI',`/jobs/${current.job_id}/preview/`],['Download code',`/jobs/${current.job_id}/download`],['Review revision',`/jobs/${current.job_id}/review/ui`]]){
        const link=document.createElement('a');link.textContent=label;link.href=url;link.target='_blank';link.rel='noopener';$('result').append(link);
      }
      status('Revision built and browser-checked. Remaining findings: '+current.warnings.join(' · '));
    }catch(error){status(error.message);}finally{$('build').disabled=false;}
  });
  try{
    if(!job)throw Error('Missing job ID');const response=await fetch(`/jobs/${job}/review`);data=await response.json();if(!response.ok)throw Error(JSON.stringify(data.detail));
    $('original').href=`/jobs/${job}/preview/`;
    for(const state of data.states){option($('state'),state.id,`${state.time.toFixed(2)}s · ${state.id}`);option($('target'),state.id,state.id);}
    for(const region of data.motion.regions){
      const label=document.createElement('label');label.className='inline';const input=document.createElement('input');input.type='checkbox';input.addEventListener('change',()=>{input.checked?excluded.add(region.id):excluded.delete(region.id);showDraft();});label.addEventListener('mouseenter',()=>{['x','y','width','height'].forEach((name,i)=>$('box').setAttribute(name,region.bbox[i]));});label.addEventListener('mouseleave',overlay);label.append(input,document.createTextNode(`Exclude ${region.id} · bounds ${region.bbox.join(', ')}`));$('motion').append(label);
    }
    if(!data.motion.regions.length)$('motion').textContent='No isolated motion regions were exported.';
    const summary=data.verification.summary||{};
    const confirmed=Object.values(data.actions).reduce((total,map)=>total+Object.keys(map).length,0);
    $('findings').textContent=`Build/runtime: ${summary.visual_status||'unverified'}\nSemantic review: ${summary.semantic_reconstruction||'needs review'}\nConfirmed transitions: ${confirmed}\n\n${data.warnings.map(warning=>'• '+warning).join('\n\n')}`;showState();showDraft();status('Ready. Add only corrections and transitions you have reviewed.');
  }catch(error){status(error.message);$('build').disabled=true;}
});

import { build } from 'esbuild';
import { chromium } from 'playwright';
import { readFile, writeFile, mkdir, cp, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import os from 'node:os';
import { existsSync, readdirSync } from 'node:fs';

const root = path.resolve(process.argv[2]);
const request = JSON.parse(await readFile(path.join(root, 'verification_request.json'), 'utf8'));
const motionManifest = JSON.parse(await readFile(path.join(root, 'motion_manifest.json'), 'utf8'));
const sceneManifest = JSON.parse(await readFile(path.join(root, 'scene_manifest.json'), 'utf8'));
const generatedManifest=JSON.parse(await readFile(path.join(root,'generated_artwork.json'),'utf8'));
const hasArtworkMotion = motionManifest.regions.length > 0 && motionManifest.sample_times.length > 1;
const out = path.join(root, '.verified-build');
const runtime = path.dirname(fileURLToPath(import.meta.url));
let server, browser;
const result = { status: 'failed', build: 'not_run', states: [], errors: [] };
try {
  await mkdir(out, { recursive: true });
  await build({ absWorkingDir: root, entryPoints: ['main.jsx'], bundle: true,
    outfile: path.join(out, 'main.js'), format: 'esm', platform: 'browser',
    jsx: 'automatic', nodePaths: [path.join(runtime, 'node_modules')], logLevel: 'silent' });
  await cp(path.join(root, 'public'), out, { recursive: true });
  await writeFile(path.join(out, 'index.html'), '<html><head><link rel="stylesheet" href="./main.css"></head><body><div id="root"></div><script type="module" src="./main.js"></script></body></html>');
  result.build = 'passed';
  server = http.createServer(async (req, res) => {
    try {
      let pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
      const target = path.resolve(out, '.' + (pathname === '/' ? '/index.html' : pathname));
      if (!target.startsWith(out + path.sep)) { res.writeHead(403).end(); return; }
      const data = await readFile(target);
      const mime = {'.js':'text/javascript','.css':'text/css','.html':'text/html','.png':'image/png','.mp4':'video/mp4'};
      res.setHeader('Content-Type', mime[path.extname(target)] || 'application/octet-stream');
      res.end(data);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let executablePath = process.env.OMNIUI_CHROMIUM_PATH;
  if (!executablePath && !existsSync(chromium.executablePath()) && process.platform === 'win32') {
    const cache = path.join(os.homedir(),'AppData','Local','ms-playwright');
    if (existsSync(cache)) {
      executablePath = readdirSync(cache).filter(n => /^chromium-\d+$/.test(n)).sort().reverse()
        .map(n => path.join(cache,n,'chrome-win64','chrome.exe')).find(p => existsSync(p));
    }
  }
  browser = await chromium.launch({headless:true, ...(executablePath ? {executablePath} : {})});
  for (const state of request.states) {
    for (const mode of ['asset-backed', 'editable', 'hybrid']) {
      const page = await browser.newPage({viewport:{width:state.width,height:state.height},deviceScaleFactor:1});
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('response', response => { if(response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
      await page.goto(`http://127.0.0.1:${server.address().port}/?capture=1&state=${encodeURIComponent(state.id)}&mode=${mode}`);
      await page.locator('.omniui-stage').waitFor();
      await page.evaluate(async () => { await document.fonts.ready; await Promise.all([...document.images].map(i => i.decode().catch(() => {}))); });
      const screenshot = `verification/${state.id}-${mode}.png`;
      await mkdir(path.join(root, 'verification'), {recursive:true});
      await page.locator('.omniui-stage').screenshot({path:path.join(root,screenshot),animations:'disabled'});
      const nodes = await page.locator('.omniui-editable-layer [data-node-id]').count();
      const controls = await page.locator('.omniui-hotspot').count();
      const layout_findings=mode==='asset-backed'?[]:await page.locator('.omniui-editable-layer').evaluate(layer=>{
        const findings=[],ctx=document.createElement('canvas').getContext('2d');
        const visible=[...layer.querySelectorAll('[data-node-id],[data-owner-node-id]')].filter(node=>node.getBoundingClientRect().width>0&&node.getBoundingClientRect().height>0);
        for(const node of visible){
          if(['SPAN','P','LABEL','BUTTON','A'].includes(node.tagName)&&node.textContent.trim()&&!node.querySelector('[data-node-id],[data-owner-node-id]')){
            const style=getComputedStyle(node);ctx.font=`${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
            if('letterSpacing' in ctx)ctx.letterSpacing=style.letterSpacing==='normal'?'0px':style.letterSpacing;
            const metric=ctx.measureText(node.textContent),w=node.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight),h=node.clientHeight-parseFloat(style.paddingTop)-parseFloat(style.paddingBottom);
            if(metric.width>w+2||metric.actualBoundingBoxAscent+metric.actualBoundingBoxDescent>h+2)findings.push({node_id:node.dataset.nodeId||node.dataset.ownerNodeId,kind:'text_clipping',detail:'Text exceeds its measured region; review font size or bounds'});
          }
        }
        const controls=visible.filter(node=>['BUTTON','INPUT','SELECT','A'].includes(node.tagName));
        for(let i=0;i<controls.length;i++)for(let j=i+1;j<controls.length;j++){
          const a=controls[i],b=controls[j];if(a.contains(b)||b.contains(a))continue;
          const x=a.getBoundingClientRect(),y=b.getBoundingClientRect();
          const area=Math.max(0,Math.min(x.right,y.right)-Math.max(x.left,y.left))*Math.max(0,Math.min(x.bottom,y.bottom)-Math.max(x.top,y.top));
          if(area/Math.max(1,Math.min(x.width*x.height,y.width*y.height))>.35)findings.push({node_id:a.dataset.nodeId,other_node_id:b.dataset.nodeId,kind:'overlapping_controls',detail:'Interactive regions overlap substantially; review roles and bounds'});
        }
        return findings;
      });
      const typography_adjustments=mode==='asset-backed'?[]:await page.locator('[data-font-fit]').evaluateAll(nodes=>nodes.map(node=>({node_id:node.dataset.nodeId||node.dataset.ownerNodeId,scale:Number(node.dataset.fontFit)})));
      result.states.push({state_name:state.id,mode,screenshot,rendered_nodes:nodes,controls,errors,layout_findings,typography_adjustments});
      result.errors.push(...errors);
      await page.close();
    }
  }
  result.interaction_checks = [];
  if(generatedManifest.status==='generated_candidate'){
    const state=request.states.find(state=>state.id===generatedManifest.state_id);
    if(!state)throw new Error('Generated artwork references an unknown UI state');
    const page=await browser.newPage({viewport:{width:state.width,height:state.height}});
    await page.goto(`http://127.0.0.1:${server.address().port}/?state=${encodeURIComponent(state.id)}&artwork=generated`);
    await page.waitForFunction(()=>document.querySelector('.omniui-generated-artwork')?.currentTime>.1);
    await page.screenshot({path:path.join(root,'verification/generated-artwork-candidate.png')});
    result.interaction_checks.push({check:'explicit_generated_artwork_candidate_loads_with_native_ui',status:'passed'});
    await page.close();
  }
  const actions = JSON.parse(await readFile(path.join(root, 'interaction_map.json'), 'utf8'));
  for (const [source, mappings] of Object.entries(actions)) {
    for (const [node, target] of Object.entries(mappings)) {
      const state = request.states.find(s => s.id === source);
      if (!state || !request.states.some(s => s.id === target)) throw new Error('Interaction map references an unknown state');
      const page = await browser.newPage({viewport:{width:state.width,height:state.height}});
      await page.goto(`http://127.0.0.1:${server.address().port}/?state=${encodeURIComponent(source)}&mode=asset-backed`);
      await page.locator(`.omniui-hotspot[data-node-id=${JSON.stringify(node)}]`).click();
      await page.waitForFunction(id => document.querySelector('.omniui-stage')?.dataset.state?.toLowerCase() === id.replaceAll('_','').toLowerCase(),target);
      result.interaction_checks.push({source,node,target,status:'passed'});
      await page.close();
    }
  }
  // Native editing is a separate check from visual similarity.
  const inputState = result.states.find(s => s.controls > 0);
  if (inputState) {
    const state = request.states.find(s => s.id === inputState.state_name);
    const page = await browser.newPage({viewport:{width:state.width,height:state.height}});
    await page.goto(`http://127.0.0.1:${server.address().port}/?state=${encodeURIComponent(state.id)}&mode=asset-backed`);
    const input = page.locator('.omniui-hotspot:is(input[type="text"],input[type="email"],input[type="search"],input[type="password"])').first();
    if (await input.count()) {
      await input.fill('OmniUI input check');
      if (await input.inputValue() !== 'OmniUI input check') throw new Error('Native input could not be edited');
      result.interaction_checks.push({state:state.id,check:'native_input_editing',status:'passed'});
    }
    await page.close();
  }
  const motionPage = await browser.newPage();
  await motionPage.goto(`http://127.0.0.1:${server.address().port}/`);
  const skipIntro=motionPage.getByRole('button',{name:'Skip intro'});
  if(await skipIntro.count()){
    await motionPage.waitForFunction(()=>document.querySelector('video')?.currentTime>0);
    await motionPage.locator('video').evaluate((video,end)=>{video.currentTime=end;},sceneManifest.intro.end);
    await motionPage.locator('.omniui-stage').waitFor();
    await motionPage.goto(`http://127.0.0.1:${server.address().port}/`);
    await skipIntro.click();
    await motionPage.locator('.omniui-stage').waitFor();
    result.interaction_checks.push({check:'preserved_intro_playback_and_skip_to_native_ui',status:'passed'});
  }
  if(await motionPage.locator('.omniui-toolbar,.omniui-note').count())throw new Error('Normal preview includes reconstruction controls');
  if(await motionPage.locator('.omniui-generated-artwork').count()){
    await motionPage.waitForFunction(()=>document.querySelector('.omniui-generated-artwork')?.currentTime>.1);
    result.interaction_checks.push({check:'generated_artwork_video_loads_behind_native_controls',status:'passed'});
  }
  if(hasArtworkMotion){
    const first=motionManifest.sample_times[0],last=motionManifest.sample_times.at(-1);
    const threshold=first+(last-first)*.01;
    await motionPage.waitForFunction(threshold=>{const c=document.querySelector('.omniui-artwork-motion');return Number(c?.dataset.sourceTime)>threshold&&Number(c?.dataset.loadedRegionCount)>0;},threshold);
    result.interaction_checks.push({check:'normal_preview_clean_with_automatic_artwork',status:'passed'});
  }else {
    const canvas=motionPage.locator('.omniui-artwork-motion');
    if(await canvas.count() && Number(await canvas.getAttribute('data-loaded-region-count'))>0)throw new Error('Unexpected motion assets in a static reconstruction');
    result.interaction_checks.push({check:'normal_preview_clean_without_motion_assets',status:'passed'});
  }
  await motionPage.goto(`http://127.0.0.1:${server.address().port}/?inspect=1`);
  const isolatedButton=motionPage.getByRole('button',{name:'Play isolated artwork'});
  if(await isolatedButton.count()){
    await isolatedButton.click();
    const threshold=motionManifest.sample_times.at(-1)*.01;
    await motionPage.waitForFunction(threshold=>{const c=document.querySelector('.omniui-artwork-motion');return Number(c?.dataset.sourceTime)>threshold&&Number(c?.dataset.loadedRegionCount)>0;},threshold);
    await motionPage.getByRole('button',{name:'Pause artwork'}).click();
    const paused=await motionPage.locator('.omniui-artwork-motion').getAttribute('data-source-time');
    await motionPage.waitForTimeout(100);
    if(await motionPage.locator('.omniui-artwork-motion').getAttribute('data-source-time')!==paused)throw new Error('Isolated artwork did not pause');
    result.interaction_checks.push({check:'isolated_artwork_asset_loading_playback_and_pause',status:'passed'});
  }
  await motionPage.locator('select[aria-label="Rendering mode"]').selectOption('hybrid');
  const hybridInput=motionPage.locator('.omniui-editable-layer input:is([type="url"],[type="text"],[type="email"],[type="password"],[type="search"])').first();
  if(await hybridInput.count()){
    await hybridInput.fill('Hybrid control check');
    if(await hybridInput.inputValue()!=='Hybrid control check')throw new Error('Hybrid native input could not be edited');
    result.interaction_checks.push({check:'hybrid_native_input_editing',status:'passed'});
  }
  const form=motionPage.locator('form[data-component="measured-form"]').first();
  if(await form.count()){
    await form.locator('input').evaluateAll(inputs=>inputs.forEach(input=>input.value=''));
    if(await form.evaluate(form=>form.checkValidity()))throw new Error('Empty recovered form passed validation');
    await form.locator('input').evaluateAll(inputs=>inputs.forEach(input=>{input.value=input.type==='url'?'https://example.com/job':input.type==='email'?'test@example.com':'Test request';input.dispatchEvent(new Event('input',{bubbles:true}));}));
    await motionPage.evaluate(()=>{window.__submission=null;document.addEventListener('omniui:submit',e=>window.__submission=e.detail,{once:true});});
    await form.getByRole('button').first().click();
    await motionPage.waitForFunction(()=>!!window.__submission?.values);
    await form.getByRole('status').waitFor();
    result.interaction_checks.push({check:'native_form_validation_and_local_submission',status:'passed'});
    await motionPage.addStyleTag({content:'.omniui-reference-layer,.omniui-artwork-layer,.omniui-artwork-motion{display:none!important}'});
    if(!await form.locator('input').first().isVisible()||!await form.getByRole('button').first().isVisible())throw new Error('Native form disappeared without source images');
    await motionPage.setViewportSize({width:390,height:844});
    const field=form.locator('input').first();
    await field.fill('https://example.com/mobile');
    await motionPage.waitForFunction(()=>{const box=document.querySelector('form[data-component="measured-form"] input')?.getBoundingClientRect();return box&&box.height>=28&&box.width>=20;},null,{timeout:3000});
    result.interaction_checks.push({check:'native_form_survives_without_source_images_and_supports_mobile_editing',status:'passed'});
  }
  const motionButton = motionPage.getByRole('button', {name:'Play recorded motion'});
  if (await motionButton.count()) {
    await motionButton.click();
    await motionPage.waitForFunction(() => document.querySelector('video')?.currentTime > 0.25, null, {timeout:15000});
    await motionPage.locator('video').evaluate(video => { video.currentTime = 5; });
    await motionPage.waitForFunction(() => document.querySelector('video')?.currentTime >= 5);
    result.interaction_checks.push({check:'recorded_motion_playback_and_seek',status:'passed'});
  }
  await motionPage.close();
  result.status = result.errors.length ? 'failed' : 'passed';
} catch(error) {
  result.errors.push(error.message);
} finally {
  await browser?.close();
  await new Promise(resolve => server ? server.close(resolve) : resolve());
}
console.log(JSON.stringify(result));
process.exitCode = result.status === 'passed' ? 0 : 1;

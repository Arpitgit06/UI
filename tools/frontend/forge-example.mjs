import {build} from 'esbuild';
import {chromium} from 'playwright';
import {mkdir,readFile,writeFile,cp} from 'node:fs/promises';
import {existsSync,readdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import http from 'node:http';

const runtime=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(runtime,'../../examples/forge-hub');
const out=path.join(root,'dist');
await mkdir(out,{recursive:true});
await cp(path.join(root,'public'),out,{recursive:true});
await build({absWorkingDir:root,entryPoints:['main.jsx'],outfile:path.join(out,'main.js'),bundle:true,format:'esm',jsx:'automatic',nodePaths:[path.join(runtime,'node_modules')],logLevel:'silent'});
await writeFile(path.join(out,'index.html'),(await readFile(path.join(root,'index.html'),'utf8')).replace('./main.jsx','./main.js').replace('</head>','<link rel="stylesheet" href="./main.css"/></head>'));
const server=http.createServer(async(req,res)=>{
  try{const requested=new URL(req.url,'http://localhost').pathname;const target=path.resolve(out,'.'+(requested==='/'?'/index.html':requested));
    if(!target.startsWith(out+path.sep)){res.writeHead(404).end();return;}
    res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.webp':'image/webp'})[path.extname(target)]||'application/octet-stream');res.end(await readFile(target));
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
const report={checks:[],errors:[]};
try{
  const cache=path.join(process.env.LOCALAPPDATA||'','ms-playwright');
  const installed=existsSync(cache)?readdirSync(cache).filter(x=>/^chromium-\d+$/.test(x)).sort().reverse().map(x=>path.join(cache,x,'chrome-win64','chrome.exe')).find(existsSync):undefined;
  browser=await chromium.launch({headless:true,...(installed?{executablePath:installed}:{})});
  const page=await browser.newPage({viewport:{width:1600,height:1000}});
  page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  if(await page.locator('.review-bar,.status-message').count())throw new Error('Normal cockpit includes review chrome');
  await page.goto(`http://127.0.0.1:${server.address().port}/?inspect=1`);
  await page.waitForFunction(()=>document.querySelector('canvas')?.dataset.artworkLoaded==='true');
  await page.getByRole('button',{name:/MoneyPrinter/}).click();
  await page.getByRole('textbox',{name:'Tool task'}).fill('Create an introduction video');
  await page.getByRole('button',{name:'Add local task'}).click();
  if(!await page.getByRole('region',{name:'Activity feed'}).textContent().then(x=>x.includes('Create an introduction video')))throw Error('Tool task missing from activity feed');
  report.checks.push('Tool dialog task submission');
  await page.getByRole('textbox',{name:'Intent',exact:true}).fill('Inspect the local memory workspace');
  await page.getByRole('button',{name:'Execute Intent'}).click();
  if(!await page.getByRole('region',{name:'Activity feed'}).textContent().then(x=>x.includes('Inspect the local memory workspace')))throw Error('Intent missing from feed');
  report.checks.push('Intent form submission');
  await page.locator('input[type=file]').setInputFiles({name:'context.txt',mimeType:'text/plain',buffer:Buffer.from('example context')});
  await page.getByRole('button',{name:/1 file/}).waitFor();report.checks.push('Local attachment selection');
  await page.getByRole('button',{name:'Pause animation'}).click();
  const pixels=()=>page.locator('canvas').evaluate(c=>c.toDataURL());
  await page.waitForTimeout(80);const paused=await pixels();await page.waitForTimeout(120);if(await pixels()!==paused)throw Error('Paused canvas changed');
  await page.getByRole('button',{name:'Resume animation'}).click();await page.waitForTimeout(100);const running=await pixels();await page.waitForTimeout(120);if(await pixels()===running)throw Error('Animated canvas did not change');
  report.checks.push('Procedural animation pause and resume');
  if(await page.locator('video').count())throw Error('Video backdrop unexpectedly used');
  if(await page.locator('img').evaluateAll(images=>images.some(img=>!img.closest('.tool-emblem')||!img.complete||!img.naturalWidth)))throw Error('An artwork crop is missing or used outside an icon');
  report.checks.push('Individual icon assets loaded; no full-frame or video backdrop');
  const obstacles=await page.locator('.tool,.telemetry,.feed,.command,.sphere-caption').evaluateAll(nodes=>nodes.map(n=>({name:n.textContent,rect:n.getBoundingClientRect().toJSON()})));
  for(let i=0;i<obstacles.length;i++)for(let j=i+1;j<obstacles.length;j++){
    const a=obstacles[i],b=obstacles[j],x=Math.min(a.rect.right,b.rect.right)-Math.max(a.rect.left,b.rect.left),y=Math.min(a.rect.bottom,b.rect.bottom)-Math.max(a.rect.top,b.rect.top);
    if(x>3&&y>3)throw Error(`Overlapping controls: ${a.name} / ${b.name}`);
  }
  report.checks.push('Desktop cards, caption and panels do not overlap');
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.waitForTimeout(80);const reduced=await pixels();await page.waitForTimeout(120);
  if(await pixels()!==reduced)throw Error('Reduced-motion preference was ignored');
  report.checks.push('Reduced-motion preference stops procedural animation');
  const referencePage=await browser.newPage({viewport:{width:1600,height:1000},reducedMotion:'reduce'});
  const observed=[];
  for(const time of [0,4.5,9.91667]){
    await referencePage.goto(`http://127.0.0.1:${server.address().port}/?inspect=1&time=${time}`);
    await referencePage.waitForFunction(()=>document.querySelector('canvas')?.dataset.artworkLoaded==='true');
    const actual=Number(await referencePage.locator('canvas').getAttribute('data-source-time'));
    if(Math.abs(actual-time)>.002)throw Error('Source timing seek did not select the requested observation');
    observed.push(await referencePage.locator('canvas').evaluate(c=>c.toDataURL()));
    await referencePage.screenshot({path:path.join(out,`source-time-${time}.png`),fullPage:true});
  }
  if(new Set(observed).size!==3)throw Error('Measured artwork did not change at source timestamps');
  await referencePage.emulateMedia({reducedMotion:'no-preference'});
  await referencePage.getByRole('button',{name:'Replay source timing'}).click();
  await referencePage.waitForFunction(()=>Number(document.querySelector('canvas')?.dataset.sourceTime)<.5);
  await referencePage.close();
  report.checks.push('Measured artwork at 0s, 4.5s and final sample; replay resets source timing');
  await page.emulateMedia({reducedMotion:'no-preference'});
  await page.setViewportSize({width:390,height:844});
  await page.getByRole('button',{name:/Guard Agent/}).click();
  await page.getByRole('button',{name:'Close',exact:true}).click();
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile horizontal overflow');
  await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});report.checks.push('Mobile tool interaction and no horizontal overflow');
  if(report.errors.length)throw Error(report.errors.join('; '));
  report.status='passed';
}catch(e){report.status='failed';report.errors.push(e.message);process.exitCode=1;}
finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
await writeFile(path.join(out,'verification.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));

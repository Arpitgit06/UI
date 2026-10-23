"""Generate native component primitives while retaining measured scene geometry."""
import json
import re
import shutil
import subprocess
import sys
from app.config import settings
from app.core.process_runner import run_worker
from app.modules.component_runtime import MEASURED_COMPONENTS


def validate_source(source):
    required = ('export function MeasuredPanel', 'export function MeasuredForm',
                'data-component="measured-panel"', 'data-component="measured-form"',
                'preventDefault()', 'reportValidity()', 'FormData(', 'omniui:submit',
                'bubbles:true', 'detail:{values}', 'role="status"', 'omniui-form-status',
                'Form interaction previewed.', '{children}', '{...props}')
    compact = re.sub(r'\s+', '', source)
    missing = [item for item in required if re.sub(r'\s+', '', item) not in compact]
    if missing:
        raise ValueError('Coder omitted native component contract: '+', '.join(missing))
    if re.search(r'\b(fetch|XMLHttpRequest|WebSocket|eval|Function)\s*\(|dangerouslySetInnerHTML|https?://|localStorage|sessionStorage|document\.|window\.', source):
        raise ValueError('Coder introduced behavior outside the local UI component contract')
    imports = re.findall(r'\bfrom\s*[\'"]([^\'"]+)', source)
    if imports != ['react'] or 'import(' in compact or 'require(' in compact:
        raise ValueError('Coder may import React only')
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js is required to validate generated component code')
    # Syntax validation is independent of model output and precedes packaging.
    script = "const fs=require('fs');require('esbuild').transformSync(fs.readFileSync(0,'utf8'),{loader:'jsx'})"
    result = subprocess.run([node,'-e',script], input=source, capture_output=True,text=True,
                            cwd=settings.project_root/'tools/frontend',timeout=20)
    if result.returncode:
        raise ValueError('Coder JSX failed syntax validation: '+result.stderr[-1000:])


def generate_primitives(layouts):
    summary = [{'state':layout.state_name,'controls':_controls(layout.root)} for layout in layouts]
    payload = {'model':settings.local_llm_model_name,'messages':[
        {'role':'system','content':'Write a React JSX module only, with no markdown. Implement the two named exports using the supplied reference contract. Preserve the exact data attributes, local submit event and status string, reportValidity, FormData, props spreading and children. Do not add network calls, backend logic, imports other than React, navigation, or styling. The measured scene compiler supplies all geometry and CSS. Use double quotes in JSX attributes and bubbles:true,detail:{values} in the event options.'},
        {'role':'user','content':json.dumps({'observed_ui':summary,'reference_contract':MEASURED_COMPONENTS})}]}
    result = run_worker([sys.executable,'-m','app.modules.llm_subprocess_worker'],
                        input=json.dumps(payload),timeout=settings.worker_timeout_sec)
    if result.returncode:
        raise RuntimeError('Coder model failed: '+result.stderr[-1500:])
    match = re.search(r'__CODER_JSON_START__(.*?)__CODER_JSON_END__',result.stdout,re.S)
    if not match:
        raise ValueError('Coder worker returned no structured response')
    response = json.loads(match.group(1))
    source = response['source'].strip()
    source = re.sub(r'^```(?:jsx|javascript|js)?\s*|\s*```$', '', source).strip()
    validate_source(source)
    return source, {'model':settings.local_llm_model_name,'status':'generated',
                    'scope':'Shared native panel/form JSX; measured scene JSX/CSS compiled from detections',
                    'profile':response['profile']}


def _controls(node):
    counts = {}
    pending = [node]
    while pending:
        current = pending.pop()
        counts[current.tag_hint] = counts.get(current.tag_hint,0)+1
        pending.extend(current.children)
    return counts

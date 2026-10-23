"""Exercise the real reset script only against disposable project copies."""
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows reset script')
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    root = tmp_path/'project with spaces'
    (root/'tools').mkdir(parents=True)
    shutil.copyfile(ROOT/'clean.bat', root/'clean.bat')
    shutil.copyfile(ROOT/'tools/clean_project.ps1',root/'tools/clean_project.ps1')
    for name in ('setup.bat','.env.example','README.md','app/config.py',
                 '.git/HEAD','.agents/AGENTS.md','.codex/config.toml','static/index.html',
                 'examples/demo/main.jsx','examples/demo/public/art.png','tools/frontend/package-lock.json'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('preserved')
    for name in ('.env','.venv/Lib/package.py','models_cache/ltx-2b/weights.gguf',
                 'storage/uploads/input.mp4','storage/jobs/example/output.jsx','storage/outputs/export.zip',
                 'storage/ocr_cache/entry.json','storage/server.log','tools/frontend/node_modules/lib/index.js',
                 'examples/demo/dist/main.js','app/__pycache__/config.pyc','.pytest_cache/cache',
                 'yolov8_macro.pt','omniui_test.zip'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('discarded')
    return root


def run_clean(root,*args,input=None):
    return subprocess.run(['cmd.exe','/d','/c',str(root/'clean.bat'),*args],
                          cwd=root.parent,input=input,capture_output=True,text=True,timeout=45)


def test_preview_and_cancel_never_delete(project):
    before={path.relative_to(project) for path in project.rglob('*')}
    result=run_clean(project,'--dry-run')
    assert result.returncode==0,result.stderr
    assert 'DRY RUN' in result.stdout
    assert before=={path.relative_to(project) for path in project.rglob('*')}
    result=run_clean(project,input='NO\n')
    assert result.returncode==0 and 'cancelled' in result.stdout
    assert (project/'models_cache/ltx-2b/weights.gguf').is_file()


def test_reset_removes_downloads_and_storage_but_preserves_source(project):
    result=run_clean(project,'--yes')
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Cleanup complete' in result.stdout
    for name in ('.env','.venv','storage/ocr_cache','storage/server.log',
                 'tools/frontend/node_modules','examples/demo/dist','app/__pycache__',
                 'yolov8_macro.pt','omniui_test.zip'):
        assert not (project/name).exists(),name
    for name in ('.git/HEAD','.agents/AGENTS.md','.codex/config.toml','app/config.py',
                 '.env.example','examples/demo/main.jsx','examples/demo/public/art.png',
                 'tools/frontend/package-lock.json'):
        assert (project/name).read_text()=='preserved',name
    for name in ('models_cache','storage/uploads','storage/jobs','storage/outputs'):
        assert [p.name for p in (project/name).iterdir()]==['.gitkeep']
    assert run_clean(project,'--yes').returncode==0  # Reset remains repeatable.


def test_missing_project_marker_prevents_deletion(project):
    (project/'app/config.py').unlink()
    result=run_clean(project,'--yes')
    assert result.returncode!=0
    assert (project/'storage/uploads/input.mp4').exists()


def test_linked_directory_cannot_redirect_reset(project,tmp_path):
    outside=tmp_path/'outside';outside.mkdir();(outside/'keep.txt').write_text('safe')
    link=project/'models_cache/redirect'
    # Arguments travel as data through an environment variable, not shell interpolation.
    import os
    command="New-Item -ItemType Junction -Path $env:OMNIUI_TEST_LINK -Target $env:OMNIUI_TEST_TARGET | Out-Null"
    subprocess.run(['powershell.exe','-NoProfile','-Command',command],check=True,capture_output=True,
                   env={**os.environ,'OMNIUI_TEST_LINK':str(link),'OMNIUI_TEST_TARGET':str(outside)})
    try:
        result=run_clean(project,'--yes')
        assert result.returncode!=0
        assert (outside/'keep.txt').read_text()=='safe'
        assert (project/'storage/uploads/input.mp4').exists()
    finally:
        link.rmdir()  # Unlink the junction without touching its target.


def test_unknown_option_does_not_delete(project):
    assert run_clean(project,'--everything').returncode==2
    assert (project/'.env').exists()


@pytest.mark.parametrize('answer',['yes','Yes','YES','  yes  '])
def test_confirmation_accepts_case_and_surrounding_spaces(project,answer):
    result=run_clean(project,input=answer+'\n')
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Cleanup complete' in result.stdout
    assert not (project/'.env').exists()


def test_running_project_process_prevents_reset(project):
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',str(project/'app/config.py')])
    try:
        result=run_clean(project,'--yes')
        assert result.returncode!=0
        assert 'Cleanup blocked: project processes are running' in result.stdout
        assert str(process.pid) in result.stdout
        assert 'Nothing has been deleted' in result.stdout
        assert 'WriteErrorException' not in result.stderr
        assert (project/'storage/uploads/input.mp4').exists()
        assert (project/'.env').exists()
    finally:
        process.terminate()
        process.wait(timeout=10)

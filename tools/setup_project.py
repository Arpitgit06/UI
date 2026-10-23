"""Repeatable installation for VL-7B, Coder-7B and LTX-Video 2B."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_ENV = {
    'OMNIUI_VISION_DETECTOR_MODEL': 'Qwen/Qwen2-VL-7B-Instruct',
    'OMNIUI_CODE_MODEL': 'Qwen/Qwen2.5-Coder-7B-Instruct',
    'OMNIUI_CODE_GENERATOR': 'qwen',
    'OMNIUI_LLM_4BIT': 'true',
    'OMNIUI_ARTWORK_GENERATOR': 'ltx-2b-fp16',
}


def migrate_config(root=ROOT):
    path = root / '.env'
    content = path.read_text(encoding='utf-8') if path.exists() else (root / '.env.example').read_text(encoding='utf-8')
    for key, value in MODEL_ENV.items():
        pattern = rf'(?m)^{key}=.*$'
        if re.search(pattern, content):
            content = re.sub(pattern, f'{key}={value}', content)
        else:
            content = content.rstrip() + f'\n{key}={value}\n'
    path.write_text(content, encoding='utf-8')


def commands(python=sys.executable):
    npm = shutil.which('npm.cmd') or shutil.which('npm') or 'npm.cmd'
    node = shutil.which('node') or 'node'
    return [
        [python, '-m', 'pip', 'install', 'torch==2.5.1', 'torchvision==0.20.1', '--index-url', 'https://download.pytorch.org/whl/cu121'],
        [python, '-m', 'pip', 'install', 'paddlepaddle==3.3.1', '-r', str(ROOT/'requirements.txt'), '-r', str(ROOT/'requirements-ltx.txt'), '-r', str(ROOT/'requirements-dev.txt')],
        [python, str(ROOT/'tools/cache_models.py')],
        [python, str(ROOT/'tools/setup_ltx.py'), '--precision', 'fp16'],
        [npm, 'ci', '--prefix', str(ROOT/'tools/frontend'), '--no-audit', '--no-fund'],
        [node, str(ROOT/'tools/frontend/node_modules/playwright/cli.js'), 'install', 'chromium'],
        [python, str(ROOT/'tools/start_server.py'), '--check'],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print install plan without changing files or downloading')
    args = parser.parse_args(argv)
    if args.plan:
        for command in commands():
            print(subprocess.list2cmdline(command))
        return 0
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Run setup.bat to install inside the project virtual environment.')
    if not shutil.which('node') or not (shutil.which('npm.cmd') or shutil.which('npm')):
        raise SystemExit('Install Node.js with npm before running setup.bat.')
    migrate_config()
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(ROOT/'models_cache/browsers'))
    for command in commands():
        print('Running: '+subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print('Setup complete. Run .\\start.bat. Models load sequentially; startup does not load them.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)

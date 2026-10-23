"""Start one local OmniUI service with an explicit socket preflight."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parent.parent


def omniui_running(port):
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/openapi.json', timeout=2) as response:
            return json.load(response).get('info', {}).get('title') == 'OmniUI'
    except (OSError, ValueError):
        return False


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(('127.0.0.1', port))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true', help='Development only; restarts interrupt active jobs')
    parser.add_argument('--check', action='store_true', help='Check startup without launching a server')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    url = f'http://127.0.0.1:{args.port}/'
    if omniui_running(args.port):
        print(f'OmniUI is already running: {url}\nUsing the existing service; no second server was started.')
        return 0
    try:
        check_port(args.port)
    except OSError as error:
        print(f'Cannot listen on {url}: {error}\n'
              'The port is occupied or reserved by Windows. No process was stopped.\n'
              'Try: .\\start.bat --port 8001', file=sys.stderr)
        return 1
    from tools.runtime_preflight import check_runtime
    errors = check_runtime(ROOT)
    if errors:
        print('Startup checks failed:\n- '+'\n- '.join(errors)+'\nRun .\\setup.bat to install/cache the configured stack.', file=sys.stderr)
        return 1
    print('Offline assets ready: VL-7B NF4 -> LTX T5 INT8 -> LTX-Video 2B -> Coder-7B NF4 (separate processes).')
    if args.check:
        print(f'Port available: {url}')
        return 0
    os.chdir(ROOT)
    command = [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(args.port)]
    if args.reload:
        command.append('--reload')
        print('Development reload enabled: file changes can interrupt jobs.')
    print(f'Starting OmniUI: {url}\nAPI documentation: {url}docs\nPress Ctrl+C to stop.', flush=True)
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.path.insert(0,str(ROOT))
    sys.exit(main())

"""Private loopback server owned by the native window; no acoustic work at startup."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--go-file', type=Path, required=True)
    parser.add_argument('--owner-pid', type=int, required=True)
    args = parser.parse_args()
    # The parent assigns this process to its Windows Job before releasing it.
    deadline = time.monotonic() + 30
    while not args.go_file.exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Desktop owner did not release the backend')
        time.sleep(.05)
    config = json.loads((ROOT / 'desktop_runtime.json').read_text(encoding='utf-8'))
    data = args.data_dir.resolve()
    data.mkdir(parents=True, exist_ok=True)
    os.environ['HARMONY_WEB_STORE'] = str(data / 'projects')
    os.environ['HARMONY_MODEL_PYTHON'] = config['model_python']
    os.environ['HARMONY_LEGACY_ROOT'] = config['legacy_root']
    os.environ['CHORD_DESKTOP_DATA'] = str(data)
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PATH'] = str(Path(json.loads((ROOT/'inference/runtime.local.json').read_text())['ffmpeg']['path']).parent) + os.pathsep + os.environ.get('PATH','')
    # Import once into a staging directory, then rename; the original is read-only.
    marker = data / 'legacy-import.json'
    old = Path(config['legacy_root']) / 'projects/web'
    target = data / 'projects'
    if not marker.exists() and not target.exists() and old.is_dir() and not os.environ.get('CHORD_DESKTOP_SKIP_IMPORT'):
        pending = data / 'projects.importing'
        if pending.exists():
            # Preserve an incomplete attempt for inspection rather than deleting it.
            pending.rename(data / ('projects.import-interrupted-' + secrets.token_hex(4)))
        shutil.copytree(old, pending, ignore=shutil.ignore_patterns('.server.lock', '*.pending', '__pycache__'))
        pending.rename(target)
        marker.write_text(json.dumps({'source':str(old),'copied_at':time.time(),'original_modified':False},ensure_ascii=False),encoding='utf-8')
    from app.web.server import create_app
    from fastapi.responses import JSONResponse
    import uvicorn
    app = create_app(target)
    token = secrets.token_urlsafe(32)
    @app.middleware('http')
    async def desktop_session(request, call_next):
        bootstrap = request.url.path == '/' and secrets.compare_digest(request.query_params.get('session',''), token)
        authorized = bootstrap or secrets.compare_digest(request.cookies.get('chord_session',''), token) or secrets.compare_digest(request.headers.get('x-chord-session',''), token)
        if not authorized:
            return JSONResponse({'detail':'请从 Chord Studio 桌面窗口打开此会话。'}, status_code=403)
        response = await call_next(request)
        if bootstrap:
            response.set_cookie('chord_session',token,httponly=True,samesite='strict')
        response.headers['Cache-Control'] = 'no-store'
        return response
    @app.get('/api/desktop/health')
    def health():
        return {'application':'Chord Studio','version':'1.0.0','backend_pid':os.getpid(),'model':config['model_version']}
    sock = socket.socket()
    port_file = data / 'server-port.txt'
    preferred_port = int(port_file.read_text()) if port_file.exists() else 0
    sock.bind(('127.0.0.1', preferred_port))
    port = sock.getsockname()[1]
    port_file.write_text(str(port), encoding='ascii')
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False, workers=1))
    def announce():
        while not server.started and not server.should_exit:
            time.sleep(.05)
        if server.started:
            value = {'schema':1,'pid':os.getpid(),'owner_pid':args.owner_pid,'port':port,'token':token,'url':f'http://127.0.0.1:{port}/?session={token}','data_dir':str(data)}
            temporary = args.ready_file.with_suffix('.pending')
            temporary.write_text(json.dumps(value),encoding='utf-8')
            temporary.replace(args.ready_file)
    threading.Thread(target=announce, daemon=True).start()
    server.run(sockets=[sock])

if __name__ == '__main__':
    main()

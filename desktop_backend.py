"""Private loopback server owned by the native window; no acoustic work at startup."""
from __future__ import annotations
import argparse
import errno
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


def bind_loopback(port_file):
    """The saved port is a preference, never a requirement for startup.

    Keep the bound socket until uvicorn takes over: checking a free port and
    opening it again leaves a race. Never reuse or terminate another listener.
    """
    try:
        preferred = int(port_file.read_text(encoding='ascii').strip())
        if not 0 <= preferred <= 65535:preferred = 0
    except (OSError, ValueError, UnicodeError):
        preferred = 0
    def attempt(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name == 'nt':sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(('127.0.0.1', port))
            return sock
        except BaseException:
            sock.close()
            raise
    try:
        return attempt(preferred)
    except OSError as exc:
        # Windows may report a reserved/exclusively bound port as access denied.
        conflicts = {errno.EADDRINUSE, errno.EACCES, 10048, 10013}
        if not preferred or (exc.errno not in conflicts and getattr(exc,'winerror',None) not in conflicts):
            raise
        return attempt(0)


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
    os.environ['CHORD_DESKTOP_DATA'] = str(data)
    from app.runtime_paths import prepare_environment
    prepare_environment()
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PATH'] = str(Path(json.loads((ROOT/'inference/runtime.local.json').read_text())['ffmpeg']['path']).parent) + os.pathsep + os.environ.get('PATH','')
    # Import once into a staging directory, then rename; the original is read-only.
    marker = data / 'legacy-import.json'
    old = Path(os.environ['HARMONY_LEGACY_ROOT']) / 'projects/web'
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
        return {'application':'Chord Studio','version':config.get('application_version','1.0.0'),'backend_pid':os.getpid(),'model':config['model_version']}
    port_file = data / 'server-port.txt'
    sock = bind_loopback(port_file)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False, workers=1))
    def announce():
        while not server.started and not server.should_exit:
            time.sleep(.05)
        if server.started:
            port_file.write_text(str(port), encoding='ascii')
            value = {'schema':1,'pid':os.getpid(),'owner_pid':args.owner_pid,'port':port,'token':token,'url':f'http://127.0.0.1:{port}/?session={token}','data_dir':str(data)}
            temporary = args.ready_file.with_suffix('.pending')
            temporary.write_text(json.dumps(value),encoding='utf-8')
            temporary.replace(args.ready_file)
    threading.Thread(target=announce, daemon=True).start()
    try:
        server.run(sockets=[sock])
    finally:
        sock.close()

if __name__ == '__main__':
    main()

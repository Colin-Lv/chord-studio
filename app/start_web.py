"""Start the local single-page workbench (one queue / one server process)."""
import argparse
import os
from pathlib import Path
import sys
import threading
import webbrowser
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    model_config = Path(__file__).resolve().parents[1] / ".tmp/harmony_web_model_python.txt"
    if model_config.is_file():
        os.environ.setdefault("HARMONY_MODEL_PYTHON", model_config.read_text(encoding="utf-8-sig").strip())
    try:
        with urlopen(f"http://127.0.0.1:{args.port}/api/config", timeout=1) as response:
            import json
            existing = json.load(response)
        if existing.get("profiles"):
            print(f"Workbench already running: http://127.0.0.1:{args.port}")
            if args.open:
                webbrowser.open(f"http://127.0.0.1:{args.port}")
            return
    except (OSError, ValueError):
        pass
    if args.open:
        timer = threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}"))
        timer.daemon = True
        timer.start()
    import uvicorn
    uvicorn.run("app.web.server:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()

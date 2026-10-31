#!/usr/bin/env python
"""Set up the optional, revision-pinned SheetSage2 CPU environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from sheetsage2_bridge import MODEL_REPOSITORY, MODEL_REVISION, PARENT_REPOSITORY, PARENT_REVISION, REPO_ROOT


DOWNLOAD_WORKER = r'''
import json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
model_repo,model_revision,parent_repo,parent_revision,model_dir=sys.argv[1:]
model=snapshot_download(model_repo,revision=model_revision,local_dir=model_dir,
    allow_patterns=["*.py","*.json","*.safetensors","LICENSE","THIRD_PARTY_NOTICES.md","README.md"],max_workers=2)
parent=snapshot_download(parent_repo,revision=parent_revision,
    allow_patterns=["*.py","*.json","*.safetensors","LICENSE","README.md"],max_workers=2)
manifest={"model_repository":model_repo,"model_revision":model_revision,"model_dir":model,
          "parent_repository":parent_repo,"parent_revision":parent_revision,"parent_cache":parent}
Path(model_dir,"chord_studio_download.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(json.dumps(manifest,indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=REPO_ROOT / ".tmp/sheetsage2_env")
    parser.add_argument("--base-python", default=sys.executable)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "tools/sheetsage2")
    parser.add_argument("--install", action="store_true", help="Create venv if missing and install the pinned optional packages.")
    parser.add_argument("--download", action="store_true", help="Download reviewed code/adapter and the pinned MERT parent (about 2.8 GB).")
    args = parser.parse_args()
    executable = args.env.resolve() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if args.install:
        if not executable.exists():
            subprocess.run([args.base_python, "-m", "venv", "--system-site-packages", str(args.env.resolve())], check=True)
        subprocess.run([str(executable), "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements_sheetsage2_cpu.txt")], check=True)
    if args.download:
        if not executable.exists():
            raise FileNotFoundError("Use --install first or supply an existing --env.")
        subprocess.run([str(executable), "-c", DOWNLOAD_WORKER, MODEL_REPOSITORY, MODEL_REVISION,
                        PARENT_REPOSITORY, PARENT_REVISION, str(args.model_dir.resolve())], check=True)
    if not args.install and not args.download:
        print(json.dumps({"python": str(executable), "model_repository": MODEL_REPOSITORY,
                          "model_revision": MODEL_REVISION, "parent_revision": PARENT_REVISION,
                          "next": "Use --install and/or --download explicitly; then run sheetsage2_bridge.py --check."}, indent=2))


if __name__ == "__main__":
    main()

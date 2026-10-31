"""Keep v88 byte-identical while suppressing consoles in its owned subprocesses."""
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / 'inference/audio_predict.py'
original = subprocess.Popen

class HiddenPopen(original):
    def __init__(self, args, *positional, **kwargs):
        if os.name == 'nt':
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | subprocess.CREATE_NO_WINDOW
        if isinstance(args, (list,tuple)):
            args = [str(value) for value in args]
            args = [str(Path(__file__).resolve()) if value == str(ENTRY) else value for value in args]
        super().__init__(args, *positional, **kwargs)

subprocess.Popen = HiddenPopen
sys.argv[0] = str(ENTRY)
runpy.run_path(str(ENTRY), run_name='__main__')

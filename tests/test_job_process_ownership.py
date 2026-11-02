import json
import os
from pathlib import Path
import sys
import time

import psutil
import pytest

from app.web.job_process import JobProcess, WindowsJob


@pytest.mark.skipif(os.name != 'nt', reason='Windows suspended launch and venv redirector')
def test_no_child_can_escape_before_job_assignment(tmp_path, monkeypatch):
    original = WindowsJob.assign
    observed = []
    def slow_assign(self, process):
        # Deliberately widen the race: an ordinary venv python.exe would already
        # have spawned the base interpreter, before Python startup gates run.
        time.sleep(.3)
        observed.append(psutil.Process(process.pid).children(recursive=True))
        original(self, process)
    monkeypatch.setattr(WindowsJob, 'assign', slow_assign)
    marker = tmp_path/'child.json'
    script = ('import os,subprocess,sys,time,json;from pathlib import Path;'
              "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
              f'Path({str(marker)!r}).write_text(json.dumps([os.getpid(),c.pid]));time.sleep(60)')
    with (tmp_path/'output.log').open('w') as log:
        process=JobProcess([sys.executable,'-c',script],cwd=tmp_path,env=os.environ,
                           log=log,gate=tmp_path/'release.go')
        try:
            deadline=time.monotonic()+10
            while not marker.exists() and time.monotonic()<deadline:time.sleep(.025)
            pids=json.loads(marker.read_text())
            descendants=[psutil.Process(pid) for pid in pids]
            assert observed == [[]]
        finally:process.close()
    # Windows may briefly keep a terminated PID visible in the process table.
    # wait_procs checks process termination rather than requiring instant PID reuse.
    _, alive=psutil.wait_procs(descendants, timeout=5)
    assert not alive
    assert not (tmp_path/'release.go').exists()

"""Own an entire worker process tree, including nested acoustic runtimes."""
import os
import signal
import subprocess
import time
from pathlib import Path


class WindowsJob:
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        class Basic(c.Structure):
            _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64),
                        ('flags', w.DWORD), ('min_ws', c.c_size_t), ('max_ws', c.c_size_t),
                        ('active_limit', w.DWORD), ('affinity', c.c_size_t),
                        ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class Extended(c.Structure):
            _fields_ = [('basic', Basic), ('io', c.c_uint64 * 6),
                        ('process_memory', c.c_size_t), ('job_memory', c.c_size_t),
                        ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]
        class Accounting(c.Structure):
            _fields_ = [('user_time', c.c_int64), ('kernel_time', c.c_int64),
                        ('period_user', c.c_int64), ('period_kernel', c.c_int64),
                        ('page_faults', w.DWORD), ('total', w.DWORD),
                        ('active', w.DWORD), ('terminated', w.DWORD)]
        self.Accounting = Accounting
        class ThreadEntry(c.Structure):
            _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('tid', w.DWORD),
                        ('pid', w.DWORD), ('priority', w.LONG), ('delta', w.LONG), ('flags', w.DWORD)]
        self.ThreadEntry = ThreadEntry
        self.c = c
        self.api = c.WinDLL('kernel32', use_last_error=True)
        for name, args, result in [
            ('CreateJobObjectW', [c.c_void_p, w.LPCWSTR], w.HANDLE),
            ('SetInformationJobObject', [w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            ('AssignProcessToJobObject', [w.HANDLE, w.HANDLE], w.BOOL),
            ('TerminateJobObject', [w.HANDLE, w.UINT], w.BOOL),
            ('QueryInformationJobObject', [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
            ('CreateToolhelp32Snapshot', [w.DWORD, w.DWORD], w.HANDLE),
            ('Thread32First', [w.HANDLE, c.c_void_p], w.BOOL),
            ('Thread32Next', [w.HANDLE, c.c_void_p], w.BOOL),
            ('OpenThread', [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            ('ResumeThread', [w.HANDLE], w.DWORD),
            ('CloseHandle', [w.HANDLE], w.BOOL)]:
            fn = getattr(self.api, name); fn.argtypes = args; fn.restype = result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:raise c.WinError(c.get_last_error())
        limits = Extended(); limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error()); self.close(); raise error

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.c.WinError(self.c.get_last_error())

    def resume(self, process):
        """Resume only after Job assignment, including before a venv redirector runs."""
        snapshot = self.api.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == self.c.c_void_p(-1).value:raise self.c.WinError(self.c.get_last_error())
        try:
            entry = self.ThreadEntry(); entry.size = self.c.sizeof(entry)
            present = self.api.Thread32First(snapshot, self.c.byref(entry))
            while present:
                if entry.pid == process.pid:
                    thread = self.api.OpenThread(2, False, entry.tid)  # THREAD_SUSPEND_RESUME
                    if not thread:raise self.c.WinError(self.c.get_last_error())
                    try:
                        if self.api.ResumeThread(thread) == 0xffffffff:
                            raise self.c.WinError(self.c.get_last_error())
                    finally:self.api.CloseHandle(thread)
                    return
                present = self.api.Thread32Next(snapshot, self.c.byref(entry))
            raise RuntimeError('无法恢复任务的启动线程')
        finally:self.api.CloseHandle(snapshot)

    def terminate(self):
        if self.handle and not self.api.TerminateJobObject(self.handle, 1):
            raise self.c.WinError(self.c.get_last_error())

    def close(self):
        if self.handle:self.api.CloseHandle(self.handle); self.handle = None

    def wait_empty(self, timeout=15):
        # TerminateJobObject is asynchronous; the root's process handle alone
        # says nothing about Python redirectors and GPU grandchildren.
        deadline = time.monotonic() + timeout
        while self.handle:
            info = self.Accounting()
            if not self.api.QueryInformationJobObject(self.handle, 1, self.c.byref(info), self.c.sizeof(info), None):
                raise self.c.WinError(self.c.get_last_error())
            if info.active == 0:return
            if time.monotonic() >= deadline:raise TimeoutError('任务子进程尚未完全退出')
            time.sleep(.025)


def wait_for_release():
    """The parent assigns ownership before allowing the worker to spawn children."""
    value = os.environ.pop('CHORD_JOB_RELEASE', None)
    if not value:return
    gate = Path(value); deadline = time.monotonic() + 60
    while not gate.exists():
        if time.monotonic() > deadline:raise RuntimeError('任务启动授权超时')
        time.sleep(.025)
    gate.unlink(missing_ok=True)


class JobProcess:
    def __init__(self, command, *, cwd, env, log, gate):
        self.gate = Path(gate); self.gate.unlink(missing_ok=True)
        self.scope = WindowsJob() if os.name == 'nt' else None
        self.process = None
        try:
            env = dict(env, CHORD_JOB_RELEASE=str(self.gate))
            self.process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=os.name != 'nt',
                creationflags=(subprocess.CREATE_NO_WINDOW | 4) if os.name == 'nt' else 0)
            if self.scope:self.scope.assign(self.process)
            self.gate.touch()
            if self.scope:self.scope.resume(self.process)
        except BaseException:
            if self.process:
                self.process.kill(); self.process.wait(timeout=10)
            self.close()
            raise

    def wait(self, timeout):return self.process.wait(timeout=timeout)

    def terminate(self):
        if self.scope:self.scope.terminate()
        else:
            try:os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:pass

    def close(self):
        if self.scope and self.scope.handle:
            try:
                self.scope.terminate()
                self.scope.wait_empty()
            finally:
                self.scope.close()
        elif self.process:self.terminate()
        if self.process and self.process.poll() is None:self.process.wait(timeout=10)
        self.gate.unlink(missing_ok=True)

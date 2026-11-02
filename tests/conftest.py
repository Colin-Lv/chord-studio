import pytest


@pytest.fixture
def fake_job_process(monkeypatch):
    """Replace acoustic execution while retaining the cancellable process contract."""
    def install(action=lambda *a,**k:None):
        class Process:
            def __init__(self,command,**kwargs):self.command=command;self.kwargs=kwargs
            def wait(self,timeout):
                action(self.command,**self.kwargs)
                return 0
            def close(self):pass
            def terminate(self):pass
        monkeypatch.setattr('app.web.jobs.JobProcess',Process)
    return install

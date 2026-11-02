import errno
import socket
import pytest
import desktop_backend as backend

def test_busy_port_uses_another_port_and_leaves_other_listener_alive(tmp_path):
    marker=tmp_path/'server-port.txt'
    owner=socket.socket()
    if backend.os.name=='nt':owner.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    owner.bind(('127.0.0.1',0));owner.listen()
    marker.write_text(str(owner.getsockname()[1]))
    try:
        sock=backend.bind_loopback(marker)
        try:
            assert sock.getsockname()[0]=='127.0.0.1'
            assert sock.getsockname()[1]!=owner.getsockname()[1]
            with socket.create_connection(owner.getsockname(),timeout=1):pass
            # Persist only after successful server startup, not during binding.
            assert int(marker.read_text())==owner.getsockname()[1]
        finally:sock.close()
    finally:owner.close()

@pytest.mark.parametrize('saved',[None,'','not-a-port','70000','-1'])
def test_missing_or_invalid_port_file_uses_os_assigned_port(tmp_path,saved):
    marker=tmp_path/'port'
    if saved is not None:marker.write_text(saved)
    sock=backend.bind_loopback(marker)
    try:assert 0<sock.getsockname()[1]<=65535
    finally:sock.close()

def test_unrelated_socket_failure_is_not_hidden(tmp_path,monkeypatch):
    marker=tmp_path/'port';marker.write_text('12345')
    class Broken:
        closed=False
        def setsockopt(self,*a):pass
        def bind(self,*a):raise OSError(errno.EMFILE,'too many files')
        def close(self):self.closed=True
    broken=Broken();monkeypatch.setattr(backend.socket,'socket',lambda *a:broken)
    with pytest.raises(OSError) as e:backend.bind_loopback(marker)
    assert e.value.errno==errno.EMFILE and broken.closed

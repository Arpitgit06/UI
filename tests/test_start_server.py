import pytest
from tools import start_server as launcher


@pytest.fixture(autouse=True)
def cached_runtime(monkeypatch):
    monkeypatch.setattr('tools.runtime_preflight.check_runtime',lambda root: [])


def test_existing_service_never_launches_duplicate(monkeypatch, capsys):
    monkeypatch.setattr(launcher, 'omniui_running', lambda port: True)
    monkeypatch.setattr(launcher.subprocess, 'call', lambda *a, **k: pytest.fail('Duplicate launch'))
    assert launcher.main([]) == 0
    assert 'already running' in capsys.readouterr().out


def test_blocked_port_is_actionable_without_launch(monkeypatch, capsys):
    monkeypatch.setattr(launcher, 'omniui_running', lambda port: False)
    def denied(port):
        raise PermissionError(10013, 'Socket access forbidden')
    monkeypatch.setattr(launcher, 'check_port', denied)
    monkeypatch.setattr(launcher.subprocess, 'call', lambda *a, **k: pytest.fail('Launch on blocked port'))
    assert launcher.main([]) == 1
    assert '--port 8001' in capsys.readouterr().err


def test_custom_port_uses_venv_without_reload(monkeypatch):
    monkeypatch.setattr(launcher, 'omniui_running', lambda port: False)
    monkeypatch.setattr(launcher, 'check_port', lambda port: None)
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'call', lambda command, **kw: calls.append(command) or 0)
    assert launcher.main(['--port', '8001']) == 0
    assert calls[0][0] == launcher.sys.executable
    assert calls[0][-2:] == ['--port', '8001']
    assert '--reload' not in calls[0]


def test_check_only_does_not_launch(monkeypatch):
    monkeypatch.setattr(launcher, 'omniui_running', lambda port: False)
    monkeypatch.setattr(launcher, 'check_port', lambda port: None)
    monkeypatch.setattr(launcher.subprocess, 'call', lambda *a, **k: pytest.fail('Check launched server'))
    assert launcher.main(['--check']) == 0


def test_incomplete_models_prevent_launch(monkeypatch,capsys):
    monkeypatch.setattr(launcher,'omniui_running',lambda port:False)
    monkeypatch.setattr(launcher,'check_port',lambda port:None)
    monkeypatch.setattr('tools.runtime_preflight.check_runtime',lambda root:['Missing LTX transformer'])
    monkeypatch.setattr(launcher.subprocess,'call',lambda *a,**kw:pytest.fail('Launch with missing models'))
    assert launcher.main([])==1
    assert 'Missing LTX transformer' in capsys.readouterr().err

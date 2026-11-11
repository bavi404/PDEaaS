"""scripts/deploy_zerogpu_space.py: proxy-only authentication, read-only preflight, upload guards. No network:
Hub calls go to a local mock server that records request headers, so we can assert that NO Authorization header
(no token) is ever sent by the script; on the real deployment the sandbox proxy adds the credential."""

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/deploy_zerogpu_space.py"


@pytest.fixture(scope="module")
def dz():
    spec = importlib.util.spec_from_file_location("deploy_zerogpu_space", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["deploy_zerogpu_space"] = m
    spec.loader.exec_module(m)
    return m


SPACE_JSON = {"_id": "x", "id": "bavi404/pdeflow", "author": "bavi404", "sdk": "gradio", "private": False,
              "lastModified": "2026-11-11T00:00:00.000Z", "tags": [], "likes": 0,
              "runtime": {"stage": "NO_APP_FILE", "hardware": {"current": None, "requested": "zero-a10g"},
                          "storage": None, "gcTimeout": 172800, "replicas": {"current": 1, "requested": 1}}}
WHOAMI_JSON = {"type": "user", "name": "bavi404", "auth": {"type": "access_token", "accessToken": {
    "displayName": "pdeflow-deployment", "role": "fineGrained"}}, "secret_like_field": "hf_SHOULD_NEVER_BE_PRINTED"}


@pytest.fixture()
def mock_hub():
    """Local Hub stand-in. Records (method, path, headers) of every request."""
    seen, routes = [], {}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, extra=None):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            seen.append(("GET", self.path, {k.lower(): v for k, v in self.headers.items()}))
            path = self.path.split("?")[0]
            if path in routes:
                code, body = routes[path]
                extra = {"X-Error-Code": "RepositoryNotFound"} if code == 404 else None
                return self._send(code, body, extra)
            return self._send(404, {"error": "not found"}, {"X-Error-Code": "RepositoryNotFound"})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen, routes
    srv.shutdown()


# ------------------------------------------------------------------ token secrecy
def test_script_never_reads_or_sends_a_token(dz):
    src = SCRIPT.read_text()
    assert "os.environ" not in src and "getenv" not in src and "get_token" not in src
    assert "HfApi(token=False)" in src and "create_repo(" not in src  # no local token; never creates a Space
    from huggingface_hub.utils import build_hf_headers

    assert not any(k.lower() == "authorization" for k in build_hf_headers(token=False))


def test_summarize_whoami_exposes_only_identity(dz):
    s = dz.summarize_whoami(WHOAMI_JSON)
    assert s == {"name": "bavi404", "type": "user", "role": "fineGrained"}
    assert "hf_" not in json.dumps(s)


def test_whoami_via_proxy_sends_no_authorization_header(dz, mock_hub):
    url, seen, routes = mock_hub
    routes["/api/whoami-v2"] = (200, WHOAMI_JSON)
    assert dz.whoami_via_proxy(url) == {"name": "bavi404", "type": "user", "role": "fineGrained"}
    assert seen and all("authorization" not in h for _, _, h in seen)


def test_whoami_failure_refuses(dz, mock_hub):
    url, _, routes = mock_hub
    routes["/api/whoami-v2"] = (401, {"error": "Invalid credentials"})
    with pytest.raises(dz.PreflightError, match="HTTP 401"):
        dz.whoami_via_proxy(url)


# ------------------------------------------------------------------ preflight rules
GOOD = {"author": "bavi404", "sdk": "gradio", "private": False, "hardware": None, "requested_hardware": "zero-a10g"}


def test_check_space_accepts_public_gradio_zerogpu(dz):
    s = dz.check_space("bavi404/pdeflow", "bavi404", **GOOD)
    assert s["hardware"] == "zero-a10g" and s["sdk"] == "gradio" and s["private"] is False


@pytest.mark.parametrize("account,override,needle", [
    ("someone-else", {}, "authenticated account"),
    ("bavi404", {"author": "other"}, "Space author"),
    ("bavi404", {"sdk": "docker"}, "sdk 'docker'"),
    ("bavi404", {"sdk": "static"}, "sdk 'static'"),
    ("bavi404", {"private": True}, "private=True"),
    ("bavi404", {"private": None}, "private=None"),
    ("bavi404", {"requested_hardware": "cpu-basic"}, "not ZeroGPU"),
    ("bavi404", {"requested_hardware": None}, "not ZeroGPU"),
    ("bavi404", {"hardware": "t4-small", "requested_hardware": "zero-a10g"}, "not ZeroGPU"),
])
def test_check_space_refuses(dz, account, override, needle):
    with pytest.raises(dz.PreflightError, match=needle):
        dz.check_space("bavi404/pdeflow", account, **(GOOD | override))


def test_preflight_reads_space_without_token(dz, mock_hub):
    url, seen, routes = mock_hub
    routes["/api/spaces/bavi404/pdeflow"] = (200, SPACE_JSON)
    from huggingface_hub import HfApi

    s = dz.preflight(HfApi(token=False, endpoint=url), "bavi404/pdeflow", "bavi404")
    assert s["hardware"] == "zero-a10g"
    assert seen and all("authorization" not in h for _, _, h in seen)
    assert all(m == "GET" for m, _, _ in seen)  # read-only


def test_preflight_missing_space_refuses_without_creating(dz, mock_hub):
    url, seen, _ = mock_hub
    from huggingface_hub import HfApi

    with pytest.raises(dz.PreflightError, match="does not exist or is not accessible"):
        dz.preflight(HfApi(token=False, endpoint=url), "bavi404/pdeflow", "bavi404")
    assert all(m == "GET" for m, _, _ in seen)


def test_preflight_rejects_docker_space(dz, mock_hub):
    url, _, routes = mock_hub
    routes["/api/spaces/bavi404/pdeflow"] = (200, SPACE_JSON | {"sdk": "docker"})
    from huggingface_hub import HfApi

    with pytest.raises(dz.PreflightError, match="sdk 'docker'"):
        dz.preflight(HfApi(token=False, endpoint=url), "bavi404/pdeflow", "bavi404")


# ------------------------------------------------------------------ main(): upload guards
class FakeApi:
    instances: list = []  # noqa: RUF012 - reset by the fixture

    def __init__(self, token=None, **kw):
        self.token, self.endpoint, self.uploads = token, "https://huggingface.co", []
        FakeApi.instances.append(self)

    def upload_folder(self, **kw):
        self.uploads.append(kw)

    def create_repo(self, *a, **k):  # pragma: no cover - must never be called
        raise AssertionError("create_repo must never be called")

    def get_space_runtime(self, space):  # pragma: no cover
        raise AssertionError("not used without --wait")


@pytest.fixture()
def patched_main(dz, monkeypatch, tmp_path):
    import huggingface_hub

    FakeApi.instances = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(dz, "git", lambda *a, **k: "0" * 40)
    monkeypatch.setattr(dz, "stage", lambda dest, web: (dest / "app.py").write_text("x") and ["app.py"])
    monkeypatch.setattr(dz, "whoami_via_proxy", lambda endpoint: {"name": "bavi404", "type": "user", "role": "fineGrained"})
    monkeypatch.setattr(dz, "preflight", lambda api, space, acct: {"sdk": "gradio", "hardware": "zero-a10g"})

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["deploy_zerogpu_space.py", "--space", "bavi404/pdeflow", *args])
        return dz.main()
    return run


def test_main_preflight_only_uploads_nothing(patched_main):
    assert patched_main("--preflight-only") == 0
    assert FakeApi.instances[0].token is False and FakeApi.instances[0].uploads == []


@pytest.mark.parametrize("args", [[], ["--confirm-deploy", "bavi404/other"]])
def test_main_refuses_upload_without_matching_confirmation(patched_main, args):
    with pytest.raises(SystemExit, match="refusing to upload"):
        patched_main(*args)
    assert FakeApi.instances[0].uploads == []


def test_main_uploads_only_with_confirmation_and_no_token(patched_main):
    assert patched_main("--confirm-deploy", "bavi404/pdeflow") == 0
    api = FakeApi.instances[0]
    assert api.token is False and len(api.uploads) == 1 and api.uploads[0]["repo_id"] == "bavi404/pdeflow"


def test_main_dry_run_makes_no_hub_calls(patched_main):
    assert patched_main("--dry-run") == 0
    assert FakeApi.instances == []

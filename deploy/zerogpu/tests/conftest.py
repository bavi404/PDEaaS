"""Fixtures for the ZeroGPU Space app tests (run with the Python 3.12 environment, docs/ZEROGPU.md §5).

    .venv-zerogpu/bin/python -m pytest deploy/zerogpu/tests -q

The static web export must exist (deploy/zerogpu/build_frontend.py); a missing export FAILS, it does not skip.
"""

import importlib.util
import socket
import sys
import threading
import time
import warnings
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
APP_PATH = HERE.parent / "app.py"
REPO = HERE.parents[2]
WEB = HERE.parent / "web"
if str(REPO / "backend") not in sys.path:  # the pdeflow package (same code the Space serves)
    sys.path.insert(0, str(REPO / "backend"))


def load_space_module():
    spec = importlib.util.spec_from_file_location("pdeflow_space_app", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pdeflow_space_app"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def space():
    return load_space_module()


@pytest.fixture(scope="session")
def web_dir():
    if not (WEB / "index.html").is_file():
        pytest.fail(f"static export missing: run `python deploy/zerogpu/build_frontend.py` (expected {WEB})")
    return WEB


@pytest.fixture(scope="session")
def client(space, web_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    with TestClient(space.build_app(web_dir=web_dir)) as c:
        yield c


@pytest.fixture(scope="session")
def live_url(space, web_dir):
    """The integrated app served by uvicorn on a free port (for gradio_client end-to-end calls)."""
    import uvicorn

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(space.build_app(web_dir=web_dir), host="127.0.0.1", port=port,
                                           log_level="warning", access_log=False))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(600):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(timeout=15)

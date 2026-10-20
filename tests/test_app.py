import json
import threading
import urllib.request
from http.server import HTTPServer

from robot_manipulation import app


def test_scene_gives_an_image_and_a_task():
    s = app.scene(3)
    assert s["image"].startswith("data:image/png;base64,")
    assert "cube" in s["instruction"]


def test_run_job_with_offline_brain():
    r = app.run_job({"seed": 3, "brain": "color", "instruction": ""})
    assert r["success"] is True
    assert r["video"].startswith("data:image/gif;base64,")


def test_errors_come_back_as_readable_messages(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    server = HTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert b"Robot arm control" in urllib.request.urlopen(base + "/").read()
        request = urllib.request.Request(base + "/api/run", method="POST",
                                         data=json.dumps({"brain": "gemini"}).encode())
        reply = json.loads(urllib.request.urlopen(request).read())
        assert "GEMINI_API_KEY" in reply["error"]
    finally:
        server.shutdown()

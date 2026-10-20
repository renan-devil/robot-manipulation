"""A small web page to drive the robot from your browser.

    python -m robot_manipulation.app          # then open http://localhost:8000

It uses only Python's built-in web server, so there is nothing extra to
install. The server handles one request at a time, on purpose: there is one
robot, and two jobs must never move it at once. Requests simply wait their
turn, which is the "queue" in front of the robot.

API keys typed into the page are used for that one request and never saved.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .brains import make_brain
from .run import run, save_gif
from .sim import SimRobot

PAGE = Path(__file__).parent / "app.html"
KEY_VARIABLES = {"gemini": "GEMINI_API_KEY", "mistral": "MISTRAL_API_KEY"}


def scene(seed: int) -> dict:
    """What the camera sees for this scene number, and its suggested task."""
    robot = SimRobot(seed)
    return {"seed": seed, "instruction": robot.task, "image": _data_url(robot.observe().image, "PNG")}


def brains() -> dict:
    """Which brains can run right now without the user typing a key."""
    return {"color": True, **{name: bool(os.environ.get(var)) for name, var in KEY_VARIABLES.items()}}


def run_job(request: dict) -> dict:
    brain = make_brain(request.get("brain", "color"), request.get("model") or None,
                       request.get("api_key") or None)
    result = run(brain, int(request.get("seed", 0)), request.get("instruction") or None, gif=True)
    video = io.BytesIO()
    save_gif(result["frames"], video)
    return {
        "instruction": result["instruction"],
        "success": result["success"],
        "grabbed": result["grabbed"],
        "on_tray": result["on_tray"],
        "brain_said": result["brain_said"],
        "brain_view": _data_url(result["brain_view"], "PNG"),
        "video": "data:image/gif;base64," + base64.b64encode(video.getvalue()).decode(),
    }


def _data_url(image, fmt: str) -> str:
    from PIL import Image  # images travel to the page inline, as base64 text
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/scene"):
            seed = int(self.path.partition("seed=")[2] or 0)
            self._json(lambda: scene(seed))
        elif self.path == "/api/brains":
            self._json(brains)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, b"not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self._json(lambda: run_job(body))

    def _json(self, compute):
        try:
            self._send(200, json.dumps(compute()).encode(), "application/json")
        except Exception as error:  # show the reason on the page instead of a blank failure
            traceback.print_exc()
            self._send(200, json.dumps({"error": f"{type(error).__name__}: {error}"}).encode(), "application/json")

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot arm control page")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    # 127.0.0.1: reachable from this computer only, not from the network.
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Robot control page: {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

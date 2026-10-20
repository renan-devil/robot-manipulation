"""The robot's brain: look at an image, read an instruction, point at things.

Every brain answers the same question with the same format:

    plan(image, "Put the red cube on the blue tray.")
    -> Plan(pick=[y, x], place=[y, x])     # points in 0-1000 image coordinates

The [y, x] / 0-1000 convention is the one Gemini Robotics-ER uses natively,
so we adopt it everywhere. A brain knows nothing about joints or metres:
turning points into motion is the body's job (sim.py).

Three brains:
  ColorBrain    no AI, no network: finds coloured blobs. The free baseline.
  GeminiBrain   Google DeepMind's Gemini Robotics-ER model, via the Gemini API.
  MistralBrain  a Mistral vision model, via Mistral's API.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

PROMPT = """You are the vision system of a robot arm. The image comes from a camera
looking straight down at a table.

Task: {instruction}

Point to (1) the object the robot must pick up, at the centre of its top face,
and (2) the centre of the place where it must put it.
Answer with JSON only, no prose:
{{"pick": [y, x], "place": [y, x]}}
Points are [y, x] normalized to 0-1000."""


@dataclass
class Plan:
    pick: tuple[float, float]   # [y, x], 0-1000
    place: tuple[float, float]  # [y, x], 0-1000
    raw: str = ""               # what the model actually said, for debugging


def parse_plan(text: str) -> Plan:
    """Read a model's answer, tolerating ```json fences and {"point": [y, x]} objects."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in the model's answer: {text!r}")
    data = json.loads(match.group(0))

    def point(value) -> tuple[float, float]:
        if isinstance(value, dict):
            value = value["point"]
        y, x = (float(c) for c in value)
        if not (0 <= y <= 1000 and 0 <= x <= 1000):
            raise ValueError(f"point {value} is outside 0-1000")
        return y, x

    return Plan(point(data["pick"]), point(data["place"]), raw=text)


# ---------------------------------------------------------------- ColorBrain

# A pixel belongs to a colour when its channels pass these tests (r, g, b are 0-255).
COLORS = {
    "red": lambda r, g, b: (r > 120) & (r > 2 * g) & (r > 2 * b),
    "green": lambda r, g, b: (g > 100) & (g > 1.5 * r) & (g > 1.5 * b),
    "blue": lambda r, g, b: (b > 100) & (b > 1.5 * r) & (b > 1.2 * g),
    "yellow": lambda r, g, b: (r > 120) & (g > 120) & (b < 0.6 * np.minimum(r, g)),
}


class ColorBrain:
    """Classic computer vision: the first colour word is what to pick, the second where to put it."""

    def plan(self, image: np.ndarray, instruction: str) -> Plan:
        words = re.findall(r"[a-z]+", instruction.lower())
        colors = [w for w in words if w in COLORS]
        if len(colors) < 2:
            raise ValueError(f"ColorBrain needs two colour words, got {colors} in {instruction!r}")
        return Plan(self._centre(image, colors[0]), self._centre(image, colors[1]), raw=f"colours {colors[:2]}")

    @staticmethod
    def _centre(image: np.ndarray, color: str) -> tuple[float, float]:
        r, g, b = (image[..., i].astype(float) for i in range(3))
        ys, xs = np.nonzero(COLORS[color](r, g, b))
        if len(ys) < 20:
            raise ValueError(f"nothing {color} in the image")
        h, w = image.shape[:2]
        return float(np.median(ys)) / h * 1000, float(np.median(xs)) / w * 1000


# ------------------------------------------------------------ cloud brains

def _png_base64(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _post_json(url: str, headers: dict, body: dict, timeout: float = 120) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _api_key(variable: str) -> str:
    key = os.environ.get(variable)
    if not key:
        raise RuntimeError(f"set {variable} in your environment (see README, 'Use a real AI brain')")
    return key


class GeminiBrain:
    """Gemini Robotics-ER (Google DeepMind): a vision-language model trained for pointing and spatial reasoning."""

    def __init__(self, model: str = "gemini-robotics-er-2-preview"):
        self.model = model

    def plan(self, image: np.ndarray, instruction: str) -> Plan:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/png", "data": _png_base64(image)}},
                {"text": PROMPT.format(instruction=instruction)},
            ]}],
            "generationConfig": {"temperature": 0.2},
        }
        reply = _post_json(url, {"x-goog-api-key": _api_key("GEMINI_API_KEY")}, body)
        parts = reply["candidates"][0]["content"]["parts"]
        return parse_plan("".join(p.get("text", "") for p in parts))


class MistralBrain:
    """A Mistral vision model. General-purpose, not trained for robotics: expect rougher points."""

    def __init__(self, model: str = "mistral-medium-latest"):
        self.model = model

    def plan(self, image: np.ndarray, instruction: str) -> Plan:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT.format(instruction=instruction)},
                {"type": "image_url", "image_url": "data:image/png;base64," + _png_base64(image)},
            ]}],
        }
        reply = _post_json("https://api.mistral.ai/v1/chat/completions",
                           {"Authorization": "Bearer " + _api_key("MISTRAL_API_KEY")}, body)
        return parse_plan(reply["choices"][0]["message"]["content"])


BRAINS = {"color": ColorBrain, "gemini": GeminiBrain, "mistral": MistralBrain}


def make_brain(name: str, model: str | None = None):
    cls = BRAINS[name]
    return cls(model) if model else cls()

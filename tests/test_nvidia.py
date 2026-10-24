"""NVIDIA hosted endpoints against a local fake of build.nvidia.com (no key or network needed)."""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from PIL import Image

from asset_pipeline import nvidia_api
from asset_pipeline.config import Config
from asset_pipeline.llm_client import LLMClient

GLB = b"glTF" + b"\x02\x00\x00\x00" + b"\x00" * 400


def _png(color=(200, 30, 30), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class Fake(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def _send(self, code, body, headers=None):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.calls.append((self.path, dict(self.headers), body))
        if self.headers.get("Authorization") != "Bearer nvapi-test":
            return self._send(401, {"detail": "unauthorized"})
        if self.path.endswith("/black-forest-labs/flux.1-schnell"):
            return self._send(200, {"artifacts": [{"base64": base64.b64encode(_png()).decode(),
                                                   "finishReason": "SUCCESS", "seed": body["seed"]}]})
        if self.path.endswith("/microsoft/trellis"):  # slow job: 202 + poll
            return self._send(202, {}, {"NVCF-REQID": "req-1"})
        if self.path == "/v1/chat/completions":
            return self._send(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})
        self._send(404, {})

    def do_GET(self):
        if self.path == "/status/req-1":
            return self._send(200, {"artifacts": [{"base64": base64.b64encode(GLB).decode(), "finishReason": "SUCCESS"}]})
        self._send(404, {})


@pytest.fixture()
def fake(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Fake.calls = []
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_image_generation(fake):
    from asset_pipeline.reference_generator import NvidiaImageBackend

    cfg = Config().diffusion
    cfg.backend, cfg.nvidia_url, cfg.width, cfg.height = "nvidia", fake, 1000, 1400
    img = NvidiaImageBackend(cfg).generate("a crate", "blurry", 123)
    assert img.size == (64, 64)
    path, headers, body = Fake.calls[-1]
    assert path == "/black-forest-labs/flux.1-schnell"
    assert (body["width"], body["height"]) == (832, 1216)  # snapped to a FLUX size of the same aspect
    assert body["steps"] == 4 and "negative_prompt" not in body


def test_trellis_polls_until_the_glb_is_ready(fake, tmp_path, monkeypatch):
    from asset_pipeline.mesh_reconstruction import MeshReconstructor

    monkeypatch.setattr(nvidia_api, "STATUS_URL", f"{fake}/status")
    monkeypatch.setattr(nvidia_api.time, "sleep", lambda s: None)
    rgba = tmp_path / "in_rgba.png"
    Image.new("RGBA", (512, 512), (120, 90, 60, 255)).save(rgba)
    cfg = Config().reconstruction
    cfg.backend, cfg.nvidia_url = "nvidia", fake
    raw = MeshReconstructor(cfg, seed=5).reconstruct(str(rgba), str(rgba), tmp_path / "raw")
    assert Path(raw.path).read_bytes() == GLB and raw.has_texture and raw.camera is None
    _, _, body = Fake.calls[-1]
    assert body["image"].startswith("data:image/png;base64,") and len(body["image"]) < nvidia_api.INLINE_LIMIT + 40
    assert body["ss_sampling_steps"] == 12 and body["slat_cfg_scale"] == 3.0


def test_rejected_key_is_explained(fake, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-wrong")
    with pytest.raises(nvidia_api.NvidiaError, match="key rejected"):
        nvidia_api.invoke("black-forest-labs/flux.1-schnell", {"prompt": "x"}, "nvapi-wrong", base_url=fake)
    monkeypatch.delenv("NVIDIA_API_KEY")
    with pytest.raises(nvidia_api.NvidiaError, match="build.nvidia.com"):
        nvidia_api.api_key()


def test_vision_requests_send_one_small_image(fake, tmp_path):
    images = []
    for i in range(5):
        p = tmp_path / f"r{i}.png"
        Image.effect_noise((900, 900), 80).convert("RGB").save(p)  # incompressible: forces downscaling
        images.append(p)
    client = LLMClient("nvidia", "meta/llama-3.2-90b-vision-instruct", f"{fake}/v1", api_key_env="NVIDIA_API_KEY")
    assert client.chat_json("audit", images=images) == {"ok": True}
    _, _, body = Fake.calls[-1]
    content = body["messages"][-1]["content"]
    pics = [c for c in content if c["type"] == "image_url"]
    assert len(pics) == 1 and len(pics[0]["image_url"]["url"]) < nvidia_api.INLINE_LIMIT + 40
    assert "tiled into one sheet" in content[0]["text"]


def test_presets_and_ping_need_the_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    client = LLMClient("nvidia", nvidia_api.DEFAULT_PLANNER)
    assert client.base_url == nvidia_api.CHAT_URL
    assert client.ping() is False  # the catalog is public, but chats need a key
    assert nvidia_api.is_vision_model("meta/llama-3.2-90b-vision-instruct")
    assert not nvidia_api.is_vision_model("deepseek-ai/deepseek-v4.1-flash")
    assert not nvidia_api.is_vision_model("nvidia/llama-nemotron-embed-vl-1b-v2")

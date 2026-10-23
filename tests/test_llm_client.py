import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from asset_pipeline.llm_client import LLMClient, LLMError, extract_json

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here you go: {"a": [1, 2]} hope that helps', {"a": [1, 2]}),
    ('[{"x": 1}]', [{"x": 1}]),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here")


class _Server:
    """Tiny HTTP server that records requests and replies per path."""

    def __init__(self, replies):
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                server.requests.append({"path": self.path, "headers": dict(self.headers),
                                        "body": json.loads(self.rfile.read(length))})
                self._reply(replies[self.path])

            def do_GET(self):
                self._reply({"models": []})

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(PNG_1PX)
    return path


def test_ollama_payload(image):
    srv = _Server({"/api/chat": {"message": {"content": '{"ok": true}'}}})
    try:
        client = LLMClient("ollama", "qwen2.5vl:7b", srv.url)
        assert client.ping()
        assert client.chat_json("check", system="sys", images=[image]) == {"ok": True}
        body = srv.requests[0]["body"]
        assert body["format"] == "json" and body["stream"] is False
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert len(body["messages"][1]["images"]) == 1
    finally:
        srv.close()


def test_openai_compatible_payload(image):
    srv = _Server({"/v1/chat/completions": {"choices": [{"message": {"content": '{"ok": 1}'}}]}})
    try:
        client = LLMClient("openai_compatible", "Qwen2-VL-7B", srv.url)
        assert client.chat_json("check", images=[image]) == {"ok": 1}
        content = srv.requests[0]["body"]["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "response_format" not in srv.requests[0]["body"]
    finally:
        srv.close()


def test_anthropic_payload(image, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    srv = _Server({"/v1/messages": {"content": [{"type": "text", "text": '{"approved": true}'}]}})
    try:
        client = LLMClient("anthropic", "claude-sonnet-5-5", srv.url)
        assert client.chat_json("check", system="sys", images=[image]) == {"approved": True}
        req = srv.requests[0]
        assert req["headers"]["x-api-key"] == "test-key"
        assert req["body"]["system"] == "sys"
        assert req["body"]["messages"][0]["content"][0]["type"] == "image"
        assert req["body"]["messages"][0]["content"][-1] == {"type": "text", "text": "check"}
    finally:
        srv.close()


def test_retry_on_bad_json():
    srv = _Server({"/api/chat": {"message": {"content": "not json"}}})
    try:
        client = LLMClient("ollama", "m", srv.url)
        with pytest.raises(LLMError):
            client.chat_json("x", retries=1)
        assert len(srv.requests) == 2
        assert "not valid JSON" in srv.requests[1]["body"]["messages"][-1]["content"]
    finally:
        srv.close()


def test_unknown_provider():
    with pytest.raises(ValueError):
        LLMClient("nope", "m")


def test_presets_fill_url_and_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    client = LLMClient("deepseek", "deepseek-flash", "http://localhost:11434")  # config default URL is replaced
    assert client._openai_root() == "https://api.deepseek.com" and client.api_key == "ds-key"
    gemini = LLMClient("gemini", "gemini-2.5-flash")
    assert gemini._openai_root() == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert LLMClient("openrouter", "x", "https://my-proxy/v1")._openai_root() == "https://my-proxy/v1"


def test_preset_uses_openai_wire_format(image, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    srv = _Server({"/v1/chat/completions": {"choices": [{"message": {"content": '{"ok": 2}'}}]}})
    try:
        client = LLMClient("openrouter", "deepseek/deepseek-chat", srv.url + "/v1")
        assert client.chat_json("hi", images=[image]) == {"ok": 2}
        assert srv.requests[0]["headers"]["Authorization"] == "Bearer or-key"
    finally:
        srv.close()


def test_cli_provider(tmp_path, image):
    import sys

    script = tmp_path / "agent.py"
    script.write_text("import sys, json\ntext = sys.stdin.read()\nprint('thinking...')\n"
                      "print(json.dumps({'chars': len(text), 'saw_image': 'x.png' in text}))\n")
    client = LLMClient("cli", "", command=f"{sys.executable} {script}")
    assert client.ping()
    reply = client.chat_json("describe", system="be brief", images=[image])
    assert reply["saw_image"] is True and reply["chars"] > 10

    echo = LLMClient("cli", "", command=f"{sys.executable} -c 'import sys; print(sys.argv[1])' {{prompt}}")
    assert echo.chat("hello there").strip() == "hello there"
    with pytest.raises(ValueError):
        LLMClient("cli", "")


def test_cli_command_split_keeps_windows_paths(monkeypatch):
    import asset_pipeline.llm_client as llm_client

    with monkeypatch.context() as m:
        m.setattr(llm_client.os, "name", "nt")
        parts = llm_client.split_command(r'X:\a\b.exe -p "{prompt}" --dir "C:\Program Files\x"')
    assert parts == [r"X:\a\b.exe", "-p", "{prompt}", "--dir", r"C:\Program Files\x"]
    assert llm_client.split_command("claude -p '{prompt}'") == ["claude", "-p", "{prompt}"]

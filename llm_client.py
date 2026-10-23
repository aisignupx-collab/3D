"""Thin, dependency-light LLM client used by the concept and audit agents.

Two wire protocols cover every backend the pipeline targets:

* OpenAI-compatible ``/chat/completions`` - Ollama, vLLM, LM Studio, llama.cpp
  server, OpenAI. Images are sent as ``image_url`` data URIs.
* Anthropic Messages API - Claude, with base64 ``image`` content blocks.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import requests

from config import LLMConfig

log = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class LLMUnavailable(RuntimeError):
    """Raised when no LLM is configured or the endpoint cannot be reached."""


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text``.

    Local models often wrap JSON in prose or ```json fences, so try strict
    parsing first, then fenced blocks, then the outermost brace span.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in LLM response: {text[:300]!r}")


def _encode_image(path: Path, max_side: int) -> tuple[str, str]:
    """Return (media_type, base64) for an image, downscaled to ``max_side``."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return "image/png", base64.b64encode(buf.getvalue()).decode("ascii")


class LLMClient:
    def __init__(self, cfg: LLMConfig, max_image_side: int = 1024):
        self.cfg = cfg
        self.max_image_side = max_image_side
        self._disabled_reason: Optional[str] = None
        if cfg.provider == "none":
            self._disabled_reason = "provider is 'none'"

    @property
    def enabled(self) -> bool:
        return self._disabled_reason is None

    @property
    def api_key(self) -> Optional[str]:
        key = os.environ.get(self.cfg.api_key_env)
        if not key and self.cfg.provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
        if not key and "openai.com" in self.cfg.base_url:
            key = os.environ.get("OPENAI_API_KEY")
        return key

    def chat(
        self,
        system: str,
        user: str,
        images: Sequence[Path] = (),
        json_mode: bool = False,
    ) -> str:
        if not self.enabled:
            raise LLMUnavailable(self._disabled_reason)
        try:
            if self.cfg.provider == "anthropic":
                return self._chat_anthropic(system, user, images)
            if self.cfg.provider == "openai_compatible":
                return self._chat_openai(system, user, images, json_mode)
        except requests.ConnectionError as exc:
            # Don't retry an unreachable server on every agent call.
            self._disabled_reason = f"cannot reach {self.cfg.base_url}: {exc}"
            raise LLMUnavailable(self._disabled_reason) from exc
        raise LLMUnavailable(f"unknown provider '{self.cfg.provider}'")

    def chat_json(self, system: str, user: str, images: Sequence[Path] = ()) -> Any:
        raw = self.chat(system, user, images, json_mode=self.cfg.json_mode)
        return extract_json(raw)

    # -- OpenAI-compatible ---------------------------------------------------
    def _chat_openai(
        self, system: str, user: str, images: Sequence[Path], json_mode: bool
    ) -> str:
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}]
            for img in images:
                media, b64 = _encode_image(Path(img), self.max_image_side)
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}}
                )
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        resp = requests.post(url, json=body, headers=headers, timeout=self.cfg.timeout_s)
        if resp.status_code == 400 and json_mode:
            # Some servers reject response_format; the prompt still asks for JSON.
            body.pop("response_format")
            resp = requests.post(url, json=body, headers=headers, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""

    # -- Anthropic -----------------------------------------------------------
    def _chat_anthropic(self, system: str, user: str, images: Sequence[Path]) -> str:
        if not self.api_key:
            raise LLMUnavailable(
                f"set {self.cfg.api_key_env} or ANTHROPIC_API_KEY for the anthropic provider"
            )
        content: list[dict[str, Any]] = []
        for img in images:
            media, b64 = _encode_image(Path(img), self.max_image_side)
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}}
            )
        content.append({"type": "text", "text": user})
        body = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        # base_url only applies when it names a Messages endpoint (e.g. a gateway).
        url = self.cfg.base_url if self.cfg.base_url.rstrip("/").endswith("/messages") else ANTHROPIC_URL
        resp = requests.post(url, json=body, headers=headers, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        return "".join(
            block.get("text", "") for block in resp.json()["content"] if block.get("type") == "text"
        )

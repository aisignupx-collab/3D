"""Provider-agnostic chat client for text and vision LLMs.

Supported providers:
  * ``ollama``             - native Ollama API (``/api/chat``), e.g. qwen2.5, llava, qwen2.5vl
  * ``openai_compatible``  - vLLM, LM Studio, llama.cpp server, TGI (``/v1/chat/completions``)
  * ``openai``             - api.openai.com (same wire format as above)
  * ``anthropic``          - Claude Messages API
  * presets on the OpenAI wire format: ``deepseek``, ``gemini``, ``openrouter``,
    ``lmstudio``, ``vllm`` (see :data:`PRESETS`; keys come from environment variables)
  * ``cli``                - any agent CLI (Claude Code, Codex, OpenCode, ...) via a command
    template; the prompt goes to stdin or replaces ``{prompt}``

Only ``requests`` is required; no vendor SDKs.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

log = logging.getLogger(__name__)

ImageInput = Union[str, Path]


class LLMError(RuntimeError):
    pass


def _encode_image(path: ImageInput) -> tuple:
    path = Path(path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return media_type, base64.b64encode(path.read_bytes()).decode("ascii")


def split_command(command: str) -> List[str]:
    """Split a command line; on Windows keep backslashes (``C:\\tools\\claude.exe``) and strip quotes."""
    if os.name != "nt":
        return shlex.split(command)
    return [part[1:-1] if len(part) > 1 and part[0] == part[-1] and part[0] in "\"'" else part
            for part in shlex.split(command, posix=False)]


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in an LLM reply.

    Handles markdown fences, leading prose and trailing commentary, which local
    models emit even when asked not to.
    """
    if text is None:
        raise ValueError("empty LLM response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char in "{[":
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in LLM response: {text[:200]!r}")


# OpenAI-wire-format services: provider -> (API root, default API-key env var)
PRESETS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),  # deepseek-flash (V4.1 Flash), deepseek-v4-pro
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),  # gemini-*-flash
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),  # any hosted model, one key
    "lmstudio": ("http://localhost:1234/v1", ""),
    "vllm": ("http://localhost:8000/v1", ""),
}
OLLAMA_DEFAULT_URL = "http://localhost:11434"


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str = "",
        api_key_env: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_s: int = 300,
        command: str = "",
        seed: Optional[int] = None,
        keep_alive: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.command = command
        self.seed = seed
        self.keep_alive = keep_alive  # Ollama: "0" unloads the model right after the call (frees VRAM)
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if self.provider in PRESETS:
            root, key_env = PRESETS[self.provider]
            if not self.base_url or self.base_url == OLLAMA_DEFAULT_URL:
                self.base_url = root
                self._root = root  # preset roots are complete; don't append /v1
            self.api_key = self.api_key or (os.environ.get(key_env, "") if key_env else "")
        elif self.provider == "cli":
            if not command:
                raise ValueError("provider 'cli' needs a command, e.g. \"claude -p {prompt}\"")
        elif self.provider == "anthropic":
            self.base_url = self.base_url or "https://api.anthropic.com"
            self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        elif self.provider == "openai":
            self.base_url = self.base_url or "https://api.openai.com"
            self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        elif self.provider not in ("ollama", "openai_compatible"):
            raise ValueError(f"unknown LLM provider '{provider}'")

    @classmethod
    def from_config(cls, cfg) -> "LLMClient":
        return cls(
            provider=cfg.provider,
            model=cfg.model,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.timeout_s,
            command=getattr(cfg, "command", ""),
            seed=getattr(cfg, "seed", None),
        )

    # ------------------------------------------------------------------ public

    def chat(
        self,
        prompt: str,
        system: str = "",
        images: Sequence[ImageInput] = (),
        json_mode: bool = False,
    ) -> str:
        images = [Path(p) for p in images]
        for img in images:
            if not img.is_file():
                raise FileNotFoundError(img)
        handler = {
            "ollama": self._chat_ollama,
            "openai_compatible": self._chat_openai,
            "openai": self._chat_openai,
            "anthropic": self._chat_anthropic,
            "cli": self._chat_cli,
        }.get(self.provider, self._chat_openai)  # presets speak the OpenAI format
        log.debug("LLM %s/%s request (%d images)", self.provider, self.model, len(images))
        return handler(prompt, system, images, json_mode)

    def chat_json(
        self,
        prompt: str,
        system: str = "",
        images: Sequence[ImageInput] = (),
        retries: int = 2,
    ) -> Any:
        """Chat and parse a JSON reply, re-asking on malformed output."""
        last_error: Optional[Exception] = None
        current_prompt = prompt
        for _ in range(retries + 1):
            reply = self.chat(current_prompt, system=system, images=images, json_mode=True)
            try:
                return extract_json(reply)
            except ValueError as exc:
                last_error = exc
                current_prompt = (
                    prompt
                    + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
                )
        raise LLMError(f"LLM did not return valid JSON: {last_error}")

    def ping(self) -> bool:
        """Cheap reachability check used by ``main.py --doctor``."""
        try:
            if self.provider == "ollama":
                r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            elif self.provider == "anthropic":
                return bool(self.api_key)
            elif self.provider == "cli":
                return shutil.which(split_command(self.command)[0]) is not None
            else:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                r = requests.get(f"{self._openai_root()}/models", headers=headers, timeout=5)
            return r.ok
        except requests.RequestException:
            return False

    # --------------------------------------------------------------- providers

    def _post(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise LLMError(f"{self.provider} request to {url} failed: {exc}") from exc
        if not r.ok:
            raise LLMError(f"{self.provider} HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def _chat_ollama(self, prompt, system, images, json_mode) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user: Dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user["images"] = [_encode_image(p)[1] for p in images]
        messages.append(user)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens,
                        **({"seed": self.seed} if self.seed is not None else {})},
        }
        if json_mode:
            payload["format"] = "json"
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        data = self._post(f"{self.base_url}/api/chat", payload, {})
        return data.get("message", {}).get("content", "")

    def _openai_root(self) -> str:
        if getattr(self, "_root", None):
            return self._root
        if self.base_url.endswith(("/v1", "/openai")):
            return self.base_url
        return f"{self.base_url}/v1"

    def _chat_openai(self, prompt, system, images, json_mode) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                media_type, b64 = _encode_image(img)
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.provider == "lmstudio" and self.keep_alive == "0":
            # LM Studio's idle TTL (seconds) for just-in-time loaded models: frees the VRAM for
            # diffusion / TripoSR soon after the call. Models you loaded by hand stay loaded.
            payload["ttl"] = 30
        # response_format is not implemented by every local server, so JSON is
        # requested in the prompt and parsed leniently instead.
        if json_mode and self.provider in ("openai", "deepseek", "gemini"):
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(f"{self._openai_root()}/chat/completions", payload, headers)
        return data["choices"][0]["message"]["content"] or ""

    def _chat_cli(self, prompt, system, images, json_mode) -> str:
        """Run an agent CLI non-interactively. Images are passed as file paths for the agent to open."""
        text = (system + "\n\n" if system else "") + prompt
        if images:
            text += "\n\nLook at these image files (in this order):\n" + "\n".join(str(Path(i).resolve()) for i in images)
        if "{prompt}" in self.command:
            cmd = [part.replace("{prompt}", text) for part in split_command(self.command)]
            stdin = None
        else:
            cmd, stdin = split_command(self.command), text
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LLMError(f"cli provider failed: {exc}") from exc
        if proc.returncode != 0:
            raise LLMError(f"cli provider exited {proc.returncode}: {proc.stderr[-500:]}")
        return proc.stdout

    def _chat_anthropic(self, prompt, system, images, json_mode) -> str:
        if not self.api_key:
            raise LLMError("Anthropic provider selected but no API key is set (ANTHROPIC_API_KEY)")
        content: List[Dict[str, Any]] = []
        for img in images:
            media_type, b64 = _encode_image(img)
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}
            )
        content.append({"type": "text", "text": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            payload["system"] = system
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = self._post(f"{self.base_url}/v1/messages", payload, headers)
        return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")

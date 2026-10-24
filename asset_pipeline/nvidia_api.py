"""NVIDIA's hosted model API (build.nvidia.com): free developer keys, rate-limited.

One key (``NVIDIA_API_KEY``, starts with ``nvapi-``; create it at
https://build.nvidia.com/settings/api-keys) covers every stage:

* chat / vision LLMs - OpenAI-compatible at ``https://integrate.api.nvidia.com/v1``
  (used through ``llm_client`` with ``provider: nvidia``);
* image generation (FLUX.1, Stable Diffusion) and image-to-3D (TRELLIS) -
  ``POST https://ai.api.nvidia.com/v1/genai/<publisher>/<model>``.

Inline images are limited to about 180 KB; larger inputs are uploaded as NVCF assets
and referenced by id. Long jobs may answer ``202 Accepted`` with an ``NVCF-REQID``
header and are polled until done. The free tier is meant for prototyping: requests
are rate-limited, and prompts and images are sent to NVIDIA.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

KEY_ENV = "NVIDIA_API_KEY"
KEY_URL = "https://build.nvidia.com/settings/api-keys"
CHAT_URL = "https://integrate.api.nvidia.com/v1"
GENAI_URL = "https://ai.api.nvidia.com/v1/genai"
ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
STATUS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status"
INLINE_LIMIT = 180_000  # characters of base64 the API accepts inline

DEFAULT_PLANNER = "deepseek-ai/deepseek-v4.1-flash"
DEFAULT_VISION = "meta/llama-3.2-90b-vision-instruct"
DEFAULT_IMAGE = "black-forest-labs/flux.1-schnell"
DEFAULT_3D = "microsoft/trellis"
# Image sizes FLUX.1 accepts (others are snapped to the closest aspect ratio)
FLUX_SIZES = ((1024, 1024), (768, 1344), (1344, 768), (896, 1152), (1152, 896), (832, 1216), (1216, 832))
VISION_PATTERN = re.compile(r"vision|vl\b|-vl-|vila|neva|fuyu|kosmos|omni|multimodal|gemma-3|llama-4", re.I)


class NvidiaError(RuntimeError):
    pass


def api_key(env: str = KEY_ENV) -> str:
    key = os.environ.get(env, "").strip()
    if not key:
        raise NvidiaError(f"{env} is not set: create a free key at {KEY_URL} and set the {env} environment "
                          "variable (or paste it under System -> API keys in the web UI)")
    return key


def is_vision_model(model_id: str) -> bool:
    return bool(VISION_PATTERN.search(model_id)) and "embed" not in model_id and "retriever" not in model_id


def catalog(timeout: float = 10.0) -> List[str]:
    """Model ids on the hosted chat endpoint (public, no key needed)."""
    r = requests.get(f"{CHAT_URL}/models", timeout=timeout)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def snap_flux_size(width: int, height: int) -> Tuple[int, int]:
    aspect = width / max(height, 1)
    return min(FLUX_SIZES, key=lambda wh: abs(wh[0] / wh[1] - aspect))


# ----------------------------------------------------------------- images in

def fit_image_bytes(image, limit: int = INLINE_LIMIT, keep_alpha: bool = False) -> Tuple[str, bytes]:
    """Encode a PIL image small enough to send inline: (media type, bytes).

    PNG (alpha kept) when ``keep_alpha``, else JPEG; the image is shrunk until its base64 fits.
    """
    from PIL import Image

    img = image if keep_alpha else image.convert("RGB")
    side = max(img.size)
    for _ in range(12):
        buf = io.BytesIO()
        if keep_alpha:
            img.save(buf, "PNG", optimize=True)
            media = "image/png"
        else:
            img.save(buf, "JPEG", quality=85, optimize=True)
            media = "image/jpeg"
        data = buf.getvalue()
        if len(base64.b64encode(data)) <= limit or side <= 128:
            return media, data
        side = int(side * 0.8)
        img = img.copy()
        img.thumbnail((side, side), Image.LANCZOS)
    return media, data


def upload_asset(data: bytes, content_type: str, key: str, description: str = "asset_pipeline input",
                 url: str = ASSETS_URL, timeout: float = 60.0) -> str:
    """Upload a large input as an NVCF asset; returns its id."""
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                    "Accept": "application/json"},
                      json={"contentType": content_type, "description": description}, timeout=timeout)
    _raise(r, "asset upload")
    info = r.json()
    put = requests.put(info["uploadUrl"], data=data, timeout=timeout,
                       headers={"Content-Type": content_type, "x-amz-meta-nvcf-asset-description": description})
    _raise(put, "asset upload")
    return info["assetId"]


def image_reference(image, key: str, keep_alpha: bool = False, limit: int = INLINE_LIMIT,
                    assets_url: str = ASSETS_URL) -> Tuple[str, List[str]]:
    """``data:`` URL for an image input (inline, or an uploaded asset) and the asset ids to declare."""
    media, data = fit_image_bytes(image, limit, keep_alpha)
    encoded = base64.b64encode(data).decode("ascii")
    if len(encoded) <= limit:
        return f"data:{media};base64,{encoded}", []
    asset_id = upload_asset(data, media, key, url=assets_url)
    return f"data:{media};asset_id,{asset_id}", [asset_id]


# ----------------------------------------------------------------- calls

def _raise(r: requests.Response, what: str):
    if r.status_code == 401 or r.status_code == 403:
        raise NvidiaError(f"NVIDIA {what}: key rejected ({r.status_code}). Check {KEY_ENV} ({KEY_URL})")
    if r.status_code == 429:
        raise NvidiaError(f"NVIDIA {what}: rate limited (429); the free tier allows ~40 requests/min")
    if r.status_code >= 400:
        raise NvidiaError(f"NVIDIA {what} failed: {r.status_code} {r.text[:500]}")


def invoke(model: str, payload: Dict[str, Any], key: str, base_url: str = GENAI_URL, timeout: float = 600.0,
           asset_ids: Optional[List[str]] = None, status_url: Optional[str] = None,
           poll_s: float = 2.0) -> requests.Response:
    """POST to a genai model and wait for the result (handles 202 + polling)."""
    status_url = status_url or STATUS_URL
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json"}
    if asset_ids:
        headers["NVCF-INPUT-ASSET-REFERENCES"] = ",".join(asset_ids)
    url = f"{base_url.rstrip('/')}/{model}"
    deadline = time.time() + timeout
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    while r.status_code == 202:
        req_id = r.headers.get("NVCF-REQID") or r.headers.get("nvcf-reqid")
        if not req_id:
            raise NvidiaError(f"NVIDIA {model}: 202 without a request id")
        if time.time() > deadline:
            raise NvidiaError(f"NVIDIA {model}: no result after {timeout:.0f}s")
        time.sleep(poll_s)
        r = requests.get(f"{status_url.rstrip('/')}/{req_id}", headers=headers, timeout=timeout)
    _raise(r, model)
    return r


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def binary_outputs(r: requests.Response) -> Iterator[bytes]:
    """Every binary payload in a genai response: the raw body, or base64 strings inside its JSON."""
    ctype = r.headers.get("Content-Type", "")
    if "json" not in ctype:
        yield r.content
        return
    try:
        body = r.json()
    except ValueError:
        yield r.content
        return
    for s in _strings(body):
        if len(s) < 200:
            continue
        s = s.split(",", 1)[1] if s.startswith("data:") and "," in s else s
        try:
            yield base64.b64decode(s, validate=False)
        except (ValueError, binascii.Error):
            continue


def first_glb(r: requests.Response) -> bytes:
    """The GLB in a response (raw, base64 in JSON, or inside a zip)."""
    import zipfile

    for blob in binary_outputs(r):
        if blob[:4] == b"glTF":
            return blob
        if blob[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".glb"):
                        return zf.read(name)
    raise NvidiaError(f"NVIDIA response contained no GLB ({r.headers.get('Content-Type')}, {len(r.content)} bytes)")


def first_image(r: requests.Response):
    """The first image in a response, as a PIL image."""
    from PIL import Image

    for blob in binary_outputs(r):
        if blob[:8] == b"\x89PNG\r\n\x1a\n" or blob[:3] == b"\xff\xd8\xff" or blob[:4] == b"RIFF":
            return Image.open(io.BytesIO(blob)).convert("RGB")
    try:
        reason = next((a.get("finishReason") for a in r.json().get("artifacts", [])), None)
    except (ValueError, AttributeError):
        reason = None
    if reason and reason != "SUCCESS":
        raise NvidiaError(f"NVIDIA image generation finished with {reason} (content filter?)")
    raise NvidiaError("NVIDIA response contained no image")

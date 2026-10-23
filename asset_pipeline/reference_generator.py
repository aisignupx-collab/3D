"""Stage 1b - reference image generation and preprocessing.

* Text input  -> per-view concept images via local diffusion (diffusers or ComfyUI),
                 isolated with rembg and assembled into a concept sheet.
* Image input -> foreground isolation and reconstruction preprocessing only.

Every view is written in three flavours:
  ``<view>_raw.png``       diffusion output / user input as-is
  ``<view>_rgba.png``      rembg cutout, tight-cropped to the foreground
                           (used for projection texturing and VLM comparison)
  ``<view>_prepared.png``  square, centred, grey background at a fixed
                           foreground ratio - the input format TripoSR /
                           InstantMesh / OpenLRM were trained on
"""

from __future__ import annotations

import gc
import io
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageOps

from .concept_planner import AssetSpec

log = logging.getLogger(__name__)


@dataclass
class ReferenceSet:
    primary: str  # prepared image fed to the reconstruction model
    views: Dict[str, Dict[str, str]] = field(default_factory=dict)
    concept_sheet: Optional[str] = None
    source: str = "text"  # text | image

    def to_dict(self) -> Dict[str, Any]:
        return {"primary": self.primary, "views": self.views,
                "concept_sheet": self.concept_sheet, "source": self.source}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self._session = None

    def _rembg_session(self):
        if self._session is None:
            try:
                from rembg import new_session
            except ImportError as exc:
                raise RuntimeError("rembg is not installed: pip install 'rembg[gpu]'") from exc
            self._session = new_session(self.cfg.rembg_model)
        return self._session

    def isolate(self, image: Image.Image) -> Image.Image:
        """Return an RGBA cutout of the foreground object."""
        if image.mode == "RGBA" and image.getextrema()[3][0] < 250:
            log.info("Input already has transparency; skipping rembg")
            return image
        from rembg import remove

        return remove(image.convert("RGB"), session=self._rembg_session()).convert("RGBA")

    @staticmethod
    def tight_crop(rgba: Image.Image, alpha_threshold: int = 16) -> Image.Image:
        alpha = rgba.getchannel("A").point(lambda a: 255 if a > alpha_threshold else 0)
        bbox = alpha.getbbox()
        if bbox is None:
            raise ValueError("background removal produced an empty mask")
        return rgba.crop(bbox)

    def prepare(self, rgba: Image.Image, background: Optional[List[int]] = None) -> Image.Image:
        """Centre the object on a square canvas at ``foreground_ratio`` of its size."""
        crop = self.tight_crop(rgba)
        size = self.cfg.output_size
        scale = (size * self.cfg.foreground_ratio) / max(crop.size)
        new_size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        crop = crop.resize(new_size, Image.LANCZOS)
        bg = tuple(background or self.cfg.background_rgb) + (255,)
        canvas = Image.new("RGBA", (size, size), bg)
        offset = ((size - crop.width) // 2, (size - crop.height) // 2)
        canvas.alpha_composite(crop, offset)
        return canvas.convert("RGB")

    def process(self, image: Image.Image, view: str, out_dir: Path) -> Dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / f"{view}_raw.png"
        image.save(raw_path)
        rgba = self.isolate(image)
        rgba_path = out_dir / f"{view}_rgba.png"
        self.tight_crop(rgba).save(rgba_path)
        prepared_path = out_dir / f"{view}_prepared.png"
        self.prepare(rgba).save(prepared_path)
        return {"raw": str(raw_path), "rgba": str(rgba_path), "prepared": str(prepared_path)}


def build_concept_sheet(views: Dict[str, Dict[str, str]], out_path: Path, title: str = "") -> str:
    """Lay out the isolated views side by side on white."""
    cell = 512
    tiles = []
    for name, paths in views.items():
        cutout = Image.open(paths["rgba"]).convert("RGBA")
        cutout = ImageOps.contain(cutout, (cell - 32, cell - 64))
        tile = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
        tile.alpha_composite(cutout, ((cell - cutout.width) // 2, 48 + (cell - 64 - cutout.height) // 2))
        ImageDraw.Draw(tile).text((12, 12), name.upper(), fill=(40, 40, 40, 255))
        tiles.append(tile)
    header = 40 if title else 0
    sheet = Image.new("RGBA", (cell * max(1, len(tiles)), cell + header), (255, 255, 255, 255))
    if title:
        ImageDraw.Draw(sheet).text((12, 12), title, fill=(0, 0, 0, 255))
    for i, tile in enumerate(tiles):
        sheet.alpha_composite(tile, (i * cell, header))
    sheet.convert("RGB").save(out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Diffusion backends
# ---------------------------------------------------------------------------

class DiffusersBackend:
    """SD 1.5 / SDXL / FLUX through HuggingFace diffusers."""

    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.pipe = None
        self.kind = None

    def _resolve_kind(self) -> str:
        if self.cfg.pipeline_type != "auto":
            return self.cfg.pipeline_type
        model = self.cfg.model_id.lower()
        if "flux" in model:
            return "flux"
        if "xl" in model:
            return "sdxl"
        return "sd"

    def load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForText2Image

        self.kind = self._resolve_kind()
        if self.device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA not available - running diffusion on CPU (use a turbo model)")
            self.device = "cpu"
        on_gpu = self.device != "cpu"
        kwargs: Dict[str, Any] = {}
        if self.kind == "flux":
            from diffusers import FluxPipeline

            loader = FluxPipeline
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            loader = AutoPipelineForText2Image
            # fp16 kernels are GPU-only; CPU inference needs fp32
            kwargs["torch_dtype"] = torch.float16 if on_gpu else torch.float32
            if self.kind == "sdxl" and on_gpu:
                kwargs["variant"] = "fp16"
                kwargs["use_safetensors"] = True
        log.info("Loading diffusion model %s (%s)", self.cfg.model_id, self.kind)
        try:
            pipe = loader.from_pretrained(self.cfg.model_id, **kwargs)
        except (OSError, ValueError):
            kwargs.pop("variant", None)  # model repo without fp16 variant files
            pipe = loader.from_pretrained(self.cfg.model_id, **kwargs)
        if on_gpu and (self.cfg.enable_cpu_offload or self.kind == "flux"):
            # FLUX-dev needs ~33 GB in bf16; offload keeps it usable on 16-24 GB cards
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        self.pipe = pipe

    def generate(self, prompt: str, negative: str, seed: int) -> Image.Image:
        import torch

        self.load()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            width=self.cfg.width,
            height=self.cfg.height,
            generator=generator,
        )
        if self.kind != "flux":
            kwargs["negative_prompt"] = negative
        return self.pipe(**kwargs).images[0]

    def unload(self):
        if self.pipe is None:
            return
        self.pipe = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class ComfyUIBackend:
    """Queue prompts on a running ComfyUI server (``python main.py --listen``)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.url = cfg.comfyui_url.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def _default_workflow(self) -> Dict[str, Any]:
        # API-format SDXL txt2img graph ("Save (API Format)" in ComfyUI)
        return {
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.cfg.comfyui_checkpoint}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": "{{width}}", "height": "{{height}}", "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}", "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{negative}}", "clip": ["4", 1]}},
            "3": {"class_type": "KSampler", "inputs": {
                "seed": "{{seed}}", "steps": self.cfg.steps, "cfg": self.cfg.guidance_scale,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "asset_pipeline", "images": ["8", 0]}},
        }

    @staticmethod
    def _fill(node: Any, values: Dict[str, Any]) -> Any:
        """Replace ``{{placeholder}}`` strings; whole-string placeholders keep their type."""
        if isinstance(node, dict):
            return {k: ComfyUIBackend._fill(v, values) for k, v in node.items()}
        if isinstance(node, list):
            return [ComfyUIBackend._fill(v, values) for v in node]
        if isinstance(node, str):
            for key, value in values.items():
                token = "{{" + key + "}}"
                if node == token:
                    return value
                node = node.replace(token, str(value))
        return node

    def generate(self, prompt: str, negative: str, seed: int) -> Image.Image:
        if self.cfg.comfyui_workflow:
            workflow = json.loads(Path(self.cfg.comfyui_workflow).read_text(encoding="utf-8"))
        else:
            workflow = self._default_workflow()
        workflow = self._fill(workflow, {
            "prompt": prompt, "negative": negative, "seed": seed,
            "width": self.cfg.width, "height": self.cfg.height,
        })
        r = requests.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=30)
        if not r.ok:
            raise RuntimeError(f"ComfyUI rejected workflow: {r.status_code} {r.text[:500]}")
        prompt_id = r.json()["prompt_id"]
        deadline = time.time() + 900
        while time.time() < deadline:
            hist = requests.get(f"{self.url}/history/{prompt_id}", timeout=30).json()
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {status.get('messages')}")
                for output in entry.get("outputs", {}).values():
                    for img in output.get("images", []):
                        view = requests.get(f"{self.url}/view", params=img, timeout=60)
                        view.raise_for_status()
                        return Image.open(io.BytesIO(view.content)).convert("RGB")
                if status.get("completed"):
                    raise RuntimeError("ComfyUI finished without producing an image")
            time.sleep(1.0)
        raise TimeoutError("ComfyUI generation timed out")

    def unload(self):
        # Ask ComfyUI to free VRAM before the reconstruction model loads
        try:
            requests.post(f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=10)
        except requests.RequestException:
            pass


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class ReferenceGenerator:
    def __init__(self, diffusion_cfg, preprocess_cfg, device: str = "cuda", seed: int = 42):
        self.cfg = diffusion_cfg
        self.preprocessor = ImagePreprocessor(preprocess_cfg)
        self.device = device
        self.seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            if self.cfg.backend == "comfyui":
                self._backend = ComfyUIBackend(self.cfg)
            elif self.cfg.backend == "diffusers":
                self._backend = DiffusersBackend(self.cfg, self.device.split(":")[0])
            else:
                raise ValueError(f"unknown diffusion backend '{self.cfg.backend}'")
        return self._backend

    def from_text(self, spec: AssetSpec, out_dir: Path) -> ReferenceSet:
        out_dir.mkdir(parents=True, exist_ok=True)
        views: Dict[str, Dict[str, str]] = {}
        for view in self.cfg.views:
            prompt = spec.view_prompts.get(view) or f"{spec.prompt}, {view} view, white background"
            log.info("Generating %s view", view)
            # Same seed for every view keeps style/colour consistent across the sheet
            image = self.backend.generate(prompt, self.cfg.negative_prompt, self.seed)
            views[view] = self.preprocessor.process(image, view, out_dir)
        self.unload()
        # Image-to-3D models want a three-quarter view; fall back to whatever we have
        primary_view = "isometric" if "isometric" in views else next(iter(views))
        sheet = build_concept_sheet(views, out_dir / "concept_sheet.png", title=spec.prompt)
        return ReferenceSet(primary=views[primary_view]["prepared"], views=views,
                            concept_sheet=sheet, source="text")

    def from_image(self, image_path: str, out_dir: Path) -> ReferenceSet:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        views = {"input": self.preprocessor.process(image, "input", out_dir)}
        sheet = build_concept_sheet(views, out_dir / "concept_sheet.png")
        return ReferenceSet(primary=views["input"]["prepared"], views=views, concept_sheet=sheet, source="image")

    def unload(self):
        if self._backend is not None:
            self._backend.unload()

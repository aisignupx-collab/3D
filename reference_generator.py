"""Stage 1 - concept planning, reference sheet generation, foreground isolation.

Three cooperating pieces:

* :class:`ConceptAgent` asks the LLM for an :class:`AssetSpec` (real-world
  references, mechanics, metric dimensions, materials, moving parts). Falls
  back to the offline knowledge base when no LLM is reachable.
* :class:`ComfyUIGenerator` / :class:`DiffusersGenerator` render front / side /
  isometric views on a white background with SDXL, FLUX or any checkpoint.
* :class:`ForegroundIsolator` strips the background with ``rembg`` and
  :func:`preprocess_for_reconstruction` crops/pads the object the way image-to-3D
  models expect (square, centred, ~85 % fill, neutral grey background).
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

from asset_spec import AssetSpec, heuristic_spec, knowledge_base_dimensions
from config import DiffusionConfig, ReconstructionConfig
from llm_client import LLMClient, LLMUnavailable

log = logging.getLogger(__name__)

VIEW_PROMPTS = {
    "front": "front view, orthographic, facing the camera, symmetrical",
    "side": "side view, orthographic profile, full length visible",
    "back": "rear view, orthographic",
    "isometric": "three-quarter isometric view from slightly above",
    "top": "top-down orthographic view",
}

CONCEPT_SYSTEM_PROMPT = """You are a senior technical art director preparing a brief for an automated 3D asset pipeline.
Return ONLY a JSON object with these keys:
  name: short snake_case asset name
  category: one of weapon, vehicle, prop, architecture, character, creature, furniture, other
  description: one dense visual sentence for a text-to-image model (shape, materials, style; no background or camera words)
  real_world_references: list of real objects / models / standards this is based on
  functional_mechanics: list of how it works physically (moving parts, joints, load paths)
  dimensions_m: [x_width, y_length_or_depth, z_height] in metres at real-world scale
  primary_axis: "x", "y" or "z" - the dimension that is most reliably known (e.g. "y" for a rifle's overall length, "z" for a door's height)
  scale_reference: short sentence justifying the dimensions (cite real measurements, human scale clearances)
  origin: "bottom_center" for things that stand on the ground, "center" for hand-held items
  symmetry: "x" if bilaterally symmetric, else null
  materials: list of {name, base_color: [r,g,b] linear 0-1, roughness 0-1, metallic 0-1}, most dominant first, max 4
  moving_parts: list of {name, region_min: [x,y,z], region_max: [x,y,z], pivot: [x,y,z], motion: "rotate"|"translate"|"fixed", axis: "X"|"Y"|"Z", limits: [min,max]}
     Regions and pivots are NORMALIZED 0-1 coordinates inside the asset's bounding box (0,0,0 = min corner, -Y is the front).
     limits are degrees for rotate, metres for translate. Empty list for characters/creatures and static objects.
  rig_type: "humanoid", "quadruped", "chain" (snakes, tails, tentacles) or "none"
Axis convention: Blender Z-up, X = width (left/right), Y = length/depth (front/back), Z = height."""


@dataclass
class ReferenceResult:
    spec: AssetSpec
    views: dict[str, Path] = field(default_factory=dict)  # isolated RGBA views
    primary_input: Optional[Path] = None  # pre-processed image for Stage 2
    concept_sheet: Optional[Path] = None
    spec_source: str = "llm"


# ---------------------------------------------------------------------------
# Concept planning
# ---------------------------------------------------------------------------
class ConceptAgent:
    """Turns a free-text prompt (and optionally an image) into an AssetSpec."""

    def __init__(self, llm: LLMClient, vision: Optional[LLMClient] = None):
        self.llm = llm
        self.vision = vision

    def describe_image(self, image_path: Path) -> Optional[str]:
        if not self.vision or not self.vision.enabled:
            return None
        try:
            return self.vision.chat(
                "You describe objects for 3D artists. Be concise and factual.",
                "Identify the single main object in this image. Give its likely real-world "
                "identity, materials and notable parts in two sentences.",
                images=[image_path],
            ).strip()
        except (LLMUnavailable, requests.RequestException) as exc:
            log.warning("Vision captioning unavailable: %s", exc)
            return None

    def plan(self, prompt: str, image_caption: Optional[str] = None) -> tuple[AssetSpec, str]:
        user = f"Asset request: {prompt}"
        if image_caption:
            user += f"\nThe user supplied a reference image showing: {image_caption}"
        try:
            data = self.llm.chat_json(CONCEPT_SYSTEM_PROMPT, user)
            if not isinstance(data, dict):
                raise ValueError("LLM returned non-object JSON")
            spec = AssetSpec.from_dict(data, prompt=prompt)
            source = "llm"
        except (LLMUnavailable, requests.RequestException, ValueError, KeyError) as exc:
            log.warning("Concept LLM unavailable (%s); using knowledge-base heuristics", exc)
            spec, source = heuristic_spec(prompt), "heuristic"
        self._sanity_check_scale(spec, prompt)
        return spec, source

    @staticmethod
    def _sanity_check_scale(spec: AssetSpec, prompt: str) -> None:
        """Guard against hallucinated units (mm vs m) using the knowledge base."""
        ref = knowledge_base_dimensions(prompt)
        if not ref:
            return
        idx = "xyz".index(spec.primary_axis)
        ratio = spec.dimensions_m[idx] / max(ref[idx], 1e-6)
        if ratio > 5 or ratio < 0.2:
            log.warning(
                "LLM dimensions %s disagree with reference %s by %.1fx; using reference",
                spec.dimensions_m, ref, ratio,
            )
            spec.dimensions_m = ref


# ---------------------------------------------------------------------------
# Text-to-image backends
# ---------------------------------------------------------------------------
class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, negative: str, seed: int) -> Image.Image: ...

    def release(self) -> None:
        """Free VRAM so the image-to-3D model fits on the same GPU."""


def _default_comfy_workflow(cfg: DiffusionConfig) -> dict[str, Any]:
    """Minimal SDXL text-to-image graph in ComfyUI API format."""
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cfg.comfyui_checkpoint}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{PROMPT}}", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{NEGATIVE}}", "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": "{{SEED}}", "steps": cfg.steps, "cfg": cfg.guidance,
            "sampler_name": cfg.comfyui_sampler, "scheduler": cfg.comfyui_scheduler, "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "asset_pipeline", "images": ["8", 0]}},
    }


def fill_workflow_placeholders(node: Any, values: dict[str, Any]) -> Any:
    """Recursively substitute ``{{KEY}}`` placeholders in a ComfyUI workflow.

    A string that is exactly one placeholder takes the raw value (so seeds stay
    ints); placeholders embedded in longer strings are replaced textually.
    """
    if isinstance(node, dict):
        return {k: fill_workflow_placeholders(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [fill_workflow_placeholders(v, values) for v in node]
    if isinstance(node, str):
        for key, value in values.items():
            token = "{{" + key + "}}"
            if node == token:
                return value
            if token in node:
                node = node.replace(token, str(value))
    return node


class ComfyUIGenerator(ImageGenerator):
    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg
        self.url = cfg.comfyui_url.rstrip("/")
        if cfg.comfyui_workflow:
            self.template = json.loads(Path(cfg.comfyui_workflow).read_text(encoding="utf-8"))
        else:
            self.template = _default_comfy_workflow(cfg)

    def generate(self, prompt: str, negative: str, seed: int) -> Image.Image:
        workflow = fill_workflow_placeholders(copy.deepcopy(self.template), {
            "PROMPT": prompt, "NEGATIVE": negative, "SEED": seed,
            "WIDTH": self.cfg.width, "HEIGHT": self.cfg.height,
        })
        client_id = uuid.uuid4().hex
        resp = requests.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow: {resp.status_code} {resp.text[:500]}")
        prompt_id = resp.json()["prompt_id"]

        deadline = time.monotonic() + self.cfg.timeout_s
        while time.monotonic() < deadline:
            hist = requests.get(f"{self.url}/history/{prompt_id}", timeout=30).json()
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {status.get('messages')}")
                for output in entry.get("outputs", {}).values():
                    for img in output.get("images", []):
                        data = requests.get(f"{self.url}/view", params={
                            "filename": img["filename"], "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output")}, timeout=60)
                        data.raise_for_status()
                        return Image.open(BytesIO(data.content)).convert("RGB")
                if status.get("completed"):
                    raise RuntimeError("ComfyUI finished without an image output (needs a SaveImage node)")
            time.sleep(1.0)
        raise TimeoutError(f"ComfyUI did not finish within {self.cfg.timeout_s}s")

    def release(self) -> None:
        try:
            requests.post(f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
        except requests.RequestException as exc:
            log.warning("Could not ask ComfyUI to free VRAM: %s", exc)


class DiffusersGenerator(ImageGenerator):
    """In-process Diffusers pipeline (SDXL, SD3, FLUX via AutoPipeline)."""

    def __init__(self, cfg: DiffusionConfig):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.cfg = cfg
        self.torch = torch
        dtype = getattr(torch, cfg.diffusers_dtype)
        self.pipe = AutoPipelineForText2Image.from_pretrained(cfg.diffusers_model, torch_dtype=dtype)
        if cfg.diffusers_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, prompt: str, negative: str, seed: int) -> Image.Image:
        device = "cuda" if self.torch.cuda.is_available() else "cpu"
        gen = self.torch.Generator(device=device).manual_seed(seed)
        kwargs = dict(prompt=prompt, width=self.cfg.width, height=self.cfg.height,
                      num_inference_steps=self.cfg.steps, guidance_scale=self.cfg.guidance, generator=gen)
        try:
            out = self.pipe(negative_prompt=negative, **kwargs)
        except TypeError:  # FLUX pipelines have no negative prompt
            out = self.pipe(**kwargs)
        return out.images[0].convert("RGB")

    def release(self) -> None:
        del self.pipe
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def build_image_generator(cfg: DiffusionConfig) -> Optional[ImageGenerator]:
    if cfg.backend == "comfyui":
        return ComfyUIGenerator(cfg)
    if cfg.backend == "diffusers":
        return DiffusersGenerator(cfg)
    if cfg.backend == "none":
        return None
    raise ValueError(f"Unknown diffusion backend '{cfg.backend}'")


# ---------------------------------------------------------------------------
# Foreground isolation & pre-processing
# ---------------------------------------------------------------------------
class ForegroundIsolator:
    def __init__(self, model_name: str = "isnet-general-use"):
        self.model_name = model_name
        self._session = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA" and image.getextrema()[3][0] < 250:
            return image  # already has a meaningful alpha channel
        from rembg import new_session, remove

        if self._session is None:
            self._session = new_session(self.model_name)
        return remove(image.convert("RGB"), session=self._session).convert("RGBA")


def preprocess_for_reconstruction(
    rgba: Image.Image, size: int = 512, foreground_ratio: float = 0.85, background: float = 0.5
) -> tuple[Image.Image, Image.Image]:
    """Crop to the alpha bounding box, pad to a centred square and resize.

    Returns ``(rgba_square, rgb_on_grey)``; TripoSR/LRM-style models expect the
    object composited on 50 % grey filling ~85 % of the frame.
    """
    import numpy as np

    rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba)[:, :, 3]
    ys, xs = np.nonzero(alpha > 16)
    if len(xs) == 0:
        raise ValueError("Foreground isolation produced an empty mask")
    crop = rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = int(max(crop.size) / foreground_ratio)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    square = square.resize((size, size), Image.LANCZOS)

    arr = np.asarray(square).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + background * (1.0 - arr[:, :, 3:4])
    return square, Image.fromarray((rgb * 255.0).round().astype(np.uint8))


def build_concept_sheet(views: dict[str, Image.Image], title: str = "", cell: int = 512) -> Image.Image:
    """Tile isolated views side by side on white with labels."""
    from PIL import ImageDraw

    label_h = 28
    sheet = Image.new("RGB", (cell * len(views), cell + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(views.items()):
        tile = img.convert("RGBA").copy()
        tile.thumbnail((cell, cell))
        bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
        bg.alpha_composite(tile, ((cell - tile.width) // 2, (cell - tile.height) // 2))
        sheet.paste(bg.convert("RGB"), (i * cell, label_h))
        draw.text((i * cell + 8, 6), f"{title}  [{name}]" if title else name, fill="black")
    return sheet


# ---------------------------------------------------------------------------
# Stage 1 driver
# ---------------------------------------------------------------------------
class ReferenceGenerator:
    def __init__(
        self,
        diffusion: DiffusionConfig,
        recon: ReconstructionConfig,
        concept_agent: ConceptAgent,
    ):
        self.cfg = diffusion
        self.recon = recon
        self.agent = concept_agent
        self.isolator = ForegroundIsolator(diffusion.rembg_model)
        self._generator: Optional[ImageGenerator] = None

    @property
    def generator(self) -> Optional[ImageGenerator]:
        if self._generator is None:
            self._generator = build_image_generator(self.cfg)
        return self._generator

    def view_prompt(self, spec: AssetSpec, view: str) -> str:
        return (
            f"{spec.description}, {VIEW_PROMPTS.get(view, view)}, single isolated object, "
            "entire object in frame, plain pure white background, neutral studio lighting, "
            "product concept render, high detail, sharp focus"
        )

    def run(
        self,
        prompt: Optional[str],
        image_input: Optional[Path],
        out_dir: Path,
        need_images: bool = True,
    ) -> ReferenceResult:
        out_dir.mkdir(parents=True, exist_ok=True)

        caption = None
        if image_input:
            caption = self.agent.describe_image(image_input)
        request = prompt or caption or (image_input.stem if image_input else "object")
        spec, source = self.agent.plan(request, caption)
        (out_dir / "asset_spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        result = ReferenceResult(spec=spec, spec_source=source)
        log.info("Asset spec (%s): %s %s m, primary axis %s", source, spec.category,
                 spec.dimensions_m, spec.primary_axis)
        if not need_images:
            return result

        isolated: dict[str, Image.Image] = {}
        if image_input:
            isolated["input"] = self.isolator(Image.open(image_input))
            primary = "input"
        else:
            if self.generator is None:
                raise RuntimeError("Text-only input needs a diffusion backend (diffusion.backend)")
            for i, view in enumerate(self.cfg.views):
                log.info("Generating %s view", view)
                raw = self.generator.generate(self.view_prompt(spec, view), spec.negative_prompt, self.cfg.seed + i)
                raw.save(out_dir / f"{view}_raw.png")
                isolated[view] = self.isolator(raw)
            self.generator.release()
            self._generator = None
            primary = self.cfg.primary_view if self.cfg.primary_view in isolated else next(iter(isolated))

        for view, img in isolated.items():
            path = out_dir / f"{view}_rgba.png"
            img.save(path)
            result.views[view] = path

        _, rgb = preprocess_for_reconstruction(
            isolated[primary], self.recon.input_size, self.recon.foreground_ratio
        )
        result.primary_input = out_dir / "reconstruction_input.png"
        rgb.save(result.primary_input)
        sheet = build_concept_sheet(isolated, spec.name)
        result.concept_sheet = out_dir / "concept_sheet.png"
        sheet.save(result.concept_sheet)
        return result

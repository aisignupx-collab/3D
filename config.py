"""Pipeline configuration.

All tunables live in dataclasses with sane defaults. A YAML file (see
``config.example.yaml``) can override any field, and a handful of environment
variables override the YAML (useful for CI / shared machines):

    BLENDER_PATH        -> blender.executable
    LLM_BASE_URL        -> llm.base_url
    LLM_MODEL           -> llm.model
    VISION_BASE_URL     -> vision.base_url
    VISION_MODEL        -> vision.model
    COMFYUI_URL         -> diffusion.comfyui_url
    TRIPOSR_REPO        -> reconstruction.triposr_repo
    INSTANTMESH_REPO    -> reconstruction.instantmesh_repo
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LLMConfig:
    """Text LLM used for concept planning (Stage 1).

    ``provider`` is one of:
      * ``openai_compatible`` - Ollama (``http://localhost:11434/v1``), vLLM
        (``http://localhost:8000/v1``), LM Studio (``http://localhost:1234/v1``)
        or OpenAI itself (``https://api.openai.com/v1``).
      * ``anthropic`` - Claude Messages API.
      * ``none`` - no LLM; the built-in heuristic knowledge base is used.
    """

    provider: str = "openai_compatible"
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:14b-instruct"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_s: float = 300.0
    json_mode: bool = True


@dataclass
class VisionConfig(LLMConfig):
    """Vision LLM used by the Stage 6 auditor (and to caption input images)."""

    model: str = "qwen2.5vl:7b"
    api_key_env: str = "VISION_API_KEY"
    # Some local VLMs (LLaVA) handle one image much better than several:
    # tile all renders into a single contact sheet before sending.
    contact_sheet: bool = True
    max_image_side: int = 1024


@dataclass
class DiffusionConfig:
    """Text-to-image backend for reference concept sheets (Stage 1)."""

    backend: str = "comfyui"  # comfyui | diffusers | none
    comfyui_url: str = "http://127.0.0.1:8188"
    # Optional API-format workflow JSON (ComfyUI "Save (API Format)").
    # Placeholders: {{PROMPT}} {{NEGATIVE}} {{SEED}} {{WIDTH}} {{HEIGHT}}
    comfyui_workflow: Optional[str] = None
    comfyui_checkpoint: str = "sd_xl_base_1.0.safetensors"
    comfyui_sampler: str = "dpmpp_2m"
    comfyui_scheduler: str = "karras"
    diffusers_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    diffusers_dtype: str = "float16"  # float16 | bfloat16 (FLUX) | float32
    diffusers_cpu_offload: bool = False
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 6.0
    seed: int = 1234
    views: list[str] = field(default_factory=lambda: ["front", "side", "isometric"])
    # Which view is fed to the single-image reconstruction model.
    primary_view: str = "isometric"
    rembg_model: str = "isnet-general-use"
    timeout_s: float = 600.0


@dataclass
class ReconstructionConfig:
    """Image-to-3D backbone (Stage 2)."""

    backend: str = "triposr"  # triposr | instantmesh | proxy
    device: str = "cuda:0"
    # TripoSR (https://github.com/VAST-AI-Research/TripoSR) - cloned repo path.
    triposr_repo: Optional[str] = None
    triposr_model: str = "stabilityai/TripoSR"
    triposr_chunk_size: int = 8192
    triposr_mc_resolution: int = 320
    input_size: int = 512
    foreground_ratio: float = 0.85
    # InstantMesh (https://github.com/TencentARC/InstantMesh) - cloned repo,
    # usually with its own conda env (its pinned deps clash with TripoSR's).
    instantmesh_repo: Optional[str] = None
    instantmesh_python: Optional[str] = None
    instantmesh_config: str = "configs/instant-mesh-large.yaml"
    instantmesh_diffusion_steps: int = 75
    timeout_s: float = 1800.0


@dataclass
class BlenderConfig:
    """Headless Blender processing (Stages 3-6)."""

    executable: Optional[str] = None
    timeout_s: float = 7200.0
    use_gpu: bool = True
    compute_device: str = "AUTO"  # AUTO | OPTIX | CUDA | HIP | ONEAPI | METAL | CPU
    # Stage 3 - topology
    min_island_ratio: float = 0.002
    fill_holes: bool = True
    auto_orient: bool = True
    import_rotation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    source_max_faces: int = 400_000
    remesh_mode: str = "quadriflow"  # quadriflow | voxel | decimate | none
    voxel_resolution: int = 160
    target_faces: int = 12_000
    smooth_iterations: int = 2
    shrinkwrap: bool = True
    smooth_angle_deg: float = 35.0
    boolean_union: bool = True
    nonuniform_fit: bool = False
    # Stage 4 - UV / PBR
    uv_method: str = "smart"  # smart | seams | lightmap
    uv_angle_limit_deg: float = 66.0
    seam_angle_deg: float = 50.0
    island_margin: float = 0.004
    texture_size: int = 2048
    bake_samples: int = 16
    cage_extrusion_rel: float = 0.01
    bake_margin_px: int = 16
    # Stage 6 - renders
    render_engine: str = "CYCLES"  # CYCLES | EEVEE | WORKBENCH
    render_samples: int = 48
    render_resolution: int = 1024
    render_uv_checker: bool = True


@dataclass
class AuditConfig:
    """Stage 6 acceptance criteria."""

    max_iterations: int = 3
    dimension_tolerance: float = 0.10  # relative error on the primary axis
    proportion_tolerance: float = 0.30  # relative error on secondary axes
    require_watertight: bool = True
    max_uv_stretch: float = 0.45  # area-weighted mean |log(texel density ratio)|
    max_self_intersections: int = 0
    min_vision_score: float = 0.7
    use_vision_llm: bool = True
    export_unapproved: bool = True


@dataclass
class PipelineConfig:
    output_root: str = "outputs"
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    blender: BlenderConfig = field(default_factory=BlenderConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_ENV_OVERRIDES = {
    "BLENDER_PATH": ("blender", "executable"),
    "LLM_BASE_URL": ("llm", "base_url"),
    "LLM_MODEL": ("llm", "model"),
    "VISION_BASE_URL": ("vision", "base_url"),
    "VISION_MODEL": ("vision", "model"),
    "COMFYUI_URL": ("diffusion", "comfyui_url"),
    "TRIPOSR_REPO": ("reconstruction", "triposr_repo"),
    "INSTANTMESH_REPO": ("reconstruction", "instantmesh_repo"),
}


def _merge_into(obj: Any, data: dict[str, Any], path: str = "") -> None:
    """Recursively copy ``data`` onto dataclass ``obj``, rejecting unknown keys."""
    known = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{path}{key}'")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{path}{key}' must be a mapping")
            _merge_into(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: Optional[str | Path] = None) -> PipelineConfig:
    """Build a config from defaults, an optional YAML file, then env vars."""
    cfg = PipelineConfig()
    if path:
        import yaml  # local import: only needed when a file is given

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge_into(cfg, data)
    for env, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env)
        if value:
            setattr(getattr(cfg, section), key, value)
    return cfg

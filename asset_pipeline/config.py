"""Pipeline configuration.

Configuration is layered, lowest priority first:
  1. dataclass defaults below
  2. a YAML file (``config.yaml`` by default, see ``config.example.yaml``)
  3. environment variables (``ASSET_PIPELINE_<SECTION>__<KEY>``, e.g.
     ``ASSET_PIPELINE_BLENDER__EXECUTABLE=/opt/blender/blender``)
  4. CLI flags (applied by ``main.py``)
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ENV_PREFIX = "ASSET_PIPELINE_"


@dataclass
class LLMConfig:
    """Text LLM used for concept planning (Stage 1)."""

    # ollama | openai_compatible | openai | anthropic | deepseek | gemini | openrouter | lmstudio | vllm | cli
    provider: str = "ollama"
    model: str = "qwen2.5:14b-instruct"
    base_url: str = "http://localhost:11434"
    api_key_env: str = ""  # name of env var holding the key (cloud providers)
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_s: int = 300
    command: str = ""  # provider "cli": e.g. 'claude -p {prompt}', 'codex exec {prompt}', 'opencode run {prompt}'


@dataclass
class DiffusionConfig:
    """Text-to-image concept generation (Stage 1)."""

    backend: str = "diffusers"  # diffusers | comfyui
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # diffusers pipeline class: auto | sdxl | flux | sd
    pipeline_type: str = "auto"
    steps: int = 30
    guidance_scale: float = 6.0
    width: int = 1024
    height: int = 1024
    negative_prompt: str = (
        "busy background, scenery, multiple objects, cropped, cut off, text, watermark, "
        "logo, blurry, lowres, deformed, perspective distortion, shadows on ground"
    )
    enable_cpu_offload: bool = False
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_workflow: str = ""  # optional path to an API-format workflow JSON
    comfyui_checkpoint: str = "sd_xl_base_1.0.safetensors"
    views: List[str] = field(default_factory=lambda: ["isometric", "front", "side"])


@dataclass
class PreprocessConfig:
    rembg_model: str = "isnet-general-use"  # u2net | isnet-general-use | birefnet-general
    foreground_ratio: float = 0.85
    output_size: int = 512
    projection_size: int = 2048  # high-res copy of the reconstruction input for texture projection
    # TripoSR/InstantMesh both expect neutral grey backgrounds
    background_rgb: List[int] = field(default_factory=lambda: [127, 127, 127])


@dataclass
class ReconstructionConfig:
    """Image-to-3D (Stage 2)."""

    backend: str = "triposr"  # triposr | instantmesh | openlrm | command | primitive
    device: str = "cuda:0"
    # TripoSR
    triposr_repo: str = "third_party/TripoSR"
    triposr_model: str = "stabilityai/TripoSR"
    mc_resolution: int = 256
    chunk_size: int = 8192
    # InstantMesh / OpenLRM run in their own venvs (pinned, conflicting deps)
    instantmesh_repo: str = "third_party/InstantMesh"
    instantmesh_python: str = "python"
    instantmesh_config: str = "configs/instant-mesh-large.yaml"
    openlrm_repo: str = "third_party/OpenLRM"
    openlrm_python: str = "python"
    openlrm_model: str = "zxhezexin/openlrm-mix-large-1.1"
    # Generic backend: any CLI model. Placeholders: {python} {input} {output_dir}
    command_template: str = ""
    timeout_s: int = 1800


@dataclass
class VisionConfig:
    """Vision LLM used by the audit loop (Stage 6)."""

    provider: str = "ollama"
    model: str = "qwen2.5vl:7b"
    base_url: str = "http://localhost:11434"
    api_key_env: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_s: int = 300
    command: str = ""


@dataclass
class BlenderConfig:
    executable: str = ""  # auto-detected when empty
    timeout_s: int = 3600
    use_gpu: bool = True  # Cycles GPU baking (CUDA/OptiX)


@dataclass
class CleanupConfig:
    """Stage 3 parameters. The audit loop mutates a copy of these."""

    merge_distance: float = 0.0001  # relative to object size
    min_island_ratio: float = 0.02  # drop loose shells smaller than this fraction of faces
    remesh_mode: str = "quadriflow"  # quadriflow | voxel | voxel_shrinkwrap | decimate | none
    voxel_resolution: int = 192  # voxels along the longest axis
    target_faces: int = 0  # 0 = use the planner's per-asset face budget
    smooth_iterations: int = 1
    auto_smooth_angle: float = 40.0
    fix_intersections: bool = True
    fill_holes: bool = True
    # Undo camera-frame yaw/tilt from reconstruction (min-volume box search)
    auto_orient: bool = True
    max_tilt_correction: float = 45.0
    orient_longest_axis: bool = True  # then turn the longest horizontal extent onto X
    symmetrize_characters: bool = True  # mirror characters/creatures across X before remeshing
    # Extra rotation applied on import, for backends with an unusual axis convention
    import_rotation_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ShapeConfig:
    """Stage 3b primitive templates: rebuild hard-surface assets from clean shapes."""

    # auto: recipe/classified shapes when they fit the scan | primitives: always | mesh: never
    mode: str = "auto"
    recipe_max_error: float = 0.12  # fit error (relative to bbox diagonal) above which a recipe is rejected
    classify_max_error: float = 0.035  # stricter: single-shape guesses must clearly match
    union: bool = True  # boolean-union components into one watertight shell
    refine: bool = True  # snap component extents to the scan
    bake_normal: bool = False  # bake the scan's normals onto primitives (re-adds lumpiness)
    # Three-view blueprint (visual hull of side/top/front silhouettes) for hard-surface
    # assets without a recipe - guns, tools, machinery
    blueprint: bool = True
    blueprint_max_error: float = 0.05
    blueprint_resolution: int = 256  # silhouette cells along the longest axis


@dataclass
class TextureConfig:
    """Stage 4 parameters."""

    resolution: int = 2048
    uv_method: str = "smart"  # smart | seams | lightmap
    uv_angle_limit: float = 66.0
    island_margin: float = 0.005
    bake_margin_px: int = 16
    cage_extrusion: float = 0.01  # relative to object size
    bake_samples: int = 16
    # bake: transfer colours from raw mesh | projection: project reference views
    # | procedural: flat colours from the material spec only
    albedo_source: str = "bake"
    # Re-project the image the 3D model was built from onto every texel its camera sees
    project_reference: bool = True
    min_projection_iou: float = 0.7  # silhouette agreement required to trust the camera
    mirror_fill: bool = True  # fill unseen faces from the mirrored view on symmetric assets


@dataclass
class RenderConfig:
    resolution: int = 1024
    engine: str = "auto"  # auto | eevee | cycles | workbench
    samples: int = 32
    include_uv_checker: bool = True


@dataclass
class PipelineConfig:
    output_dir: str = "output"
    max_iterations: int = 3
    approval_score: float = 0.75
    unload_models_between_stages: bool = True
    seed: int = 42


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    blender: BlenderConfig = field(default_factory=BlenderConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    shapes: ShapeConfig = field(default_factory=ShapeConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(value: str, current: Any) -> Any:
    """Convert an env-var string into the type of the existing field value."""
    if isinstance(current, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return yaml.safe_load(value)
    return value


def _apply_mapping(cfg: Config, data: Dict[str, Any], source: str) -> None:
    for section_name, values in (data or {}).items():
        section = getattr(cfg, section_name, None)
        if section is None or not dataclasses.is_dataclass(section):
            raise ValueError(f"{source}: unknown config section '{section_name}'")
        if not isinstance(values, dict):
            raise ValueError(f"{source}: section '{section_name}' must be a mapping")
        known = {f.name for f in dataclasses.fields(section)}
        for key, value in values.items():
            if key not in known:
                raise ValueError(f"{source}: unknown key '{section_name}.{key}'")
            setattr(section, key, value)


def _apply_env(cfg: Config) -> None:
    for name, value in os.environ.items():
        if not name.startswith(ENV_PREFIX) or "__" not in name:
            continue
        section_name, key = name[len(ENV_PREFIX):].lower().split("__", 1)
        section = getattr(cfg, section_name, None)
        if section is None or not hasattr(section, key):
            continue
        setattr(section, key, _coerce(value, getattr(section, key)))


def find_blender(explicit: str = "") -> Optional[str]:
    """Locate a Blender executable (explicit > $BLENDER_PATH > PATH > common installs)."""
    candidates = [explicit, os.environ.get("BLENDER_PATH", "")]
    which = shutil.which("blender")
    if which:
        candidates.append(which)
    candidates += [
        "/usr/bin/blender",
        "/snap/bin/blender",
        "/opt/blender/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    foundation = program_files / "Blender Foundation"
    if foundation.is_dir():
        # Prefer the newest installed version
        for version_dir in sorted(foundation.iterdir(), reverse=True):
            candidates.append(str(version_dir / "blender.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


def load_config(path: Optional[str] = None) -> Config:
    cfg = Config()
    config_path = Path(path) if path else Path("config.yaml")
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as fh:
            _apply_mapping(cfg, yaml.safe_load(fh) or {}, str(config_path))
    elif path:
        raise FileNotFoundError(f"Config file not found: {path}")
    _apply_env(cfg)
    return cfg

"""Environment diagnostics (``python main.py --doctor``)."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import requests

from .config import Config, find_blender
from .llm_client import LLMClient

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def run_doctor(cfg: Config) -> int:
    rows: List[Tuple[str, str, str]] = []

    def add(name, status, detail=""):
        rows.append((name, status, detail))

    py = sys.version_info
    add("python", OK if (3, 10) <= py[:2] <= (3, 12) else WARN,
        f"{platform.python_version()} (3.10/3.11 recommended for torch + TripoSR)")

    if _has("torch"):
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1024 ** 3
            add("torch/cuda", OK if vram >= 8 else WARN,
                f"torch {torch.__version__}, CUDA {torch.version.cuda}, {props.name} {vram:.0f} GB")
        else:
            add("torch/cuda", WARN, f"torch {torch.__version__} without CUDA - diffusion/3D will be very slow")
    else:
        add("torch", FAIL, "not installed (see README: install the CUDA wheel first)")

    for module, needed_for in (("diffusers", "text-to-image"), ("transformers", "diffusion text encoders"),
                               ("rembg", "background removal"), ("trimesh", "mesh I/O"),
                               ("onnxruntime", "rembg inference")):
        add(module, OK if _has(module) else (FAIL if module in ("rembg", "trimesh") else WARN), needed_for)
    if _has("onnxruntime"):
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        add("onnxruntime providers", OK if "CUDAExecutionProvider" in providers else WARN, ", ".join(providers))

    backend = cfg.reconstruction.backend
    if backend == "triposr":
        repo = Path(cfg.reconstruction.triposr_repo)
        add("TripoSR repo", OK if (repo / "tsr").is_dir() else FAIL, str(repo.resolve()))
        add("torchmcubes", OK if _has("torchmcubes") else FAIL, "marching cubes for TripoSR")
    elif backend == "instantmesh":
        repo = Path(cfg.reconstruction.instantmesh_repo)
        add("InstantMesh repo", OK if (repo / "run.py").is_file() else FAIL, str(repo.resolve()))
    elif backend == "openlrm":
        repo = Path(cfg.reconstruction.openlrm_repo)
        add("OpenLRM repo", OK if (repo / "openlrm").is_dir() else FAIL, str(repo.resolve()))

    try:
        exe, missing = find_blender(cfg.blender.executable), "not found - set blender.executable or BLENDER_PATH"
    except FileNotFoundError as exc:
        exe, missing = None, str(exc)
    if exe:
        try:
            out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60).stdout
            version = out.splitlines()[0] if out else "unknown"
            major_minor = tuple(int(x) for x in version.split()[1].split(".")[:2])
            add("blender", OK if major_minor >= (4, 2) else WARN, f"{exe} - {version} (4.2 LTS+ recommended)")
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            add("blender", WARN, f"{exe} ({exc})")
    else:
        add("blender", FAIL, missing)

    for label, section in (("planner LLM", cfg.llm), ("vision LLM", cfg.vision)):
        try:
            client = LLMClient.from_config(section)
            add(label, OK if client.ping() else WARN, f"{section.provider}:{section.model} @ {client.base_url}")
        except ValueError as exc:
            add(label, FAIL, str(exc))

    if cfg.diffusion.backend == "comfyui":
        try:
            ok = requests.get(f"{cfg.diffusion.comfyui_url}/system_stats", timeout=5).ok
        except requests.RequestException:
            ok = False
        add("comfyui", OK if ok else FAIL, cfg.diffusion.comfyui_url)

    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"  [{status:4}] {name:<{width}}  {detail}")
    failures = sum(1 for r in rows if r[1] == FAIL)
    print(f"\n{failures} blocking problem(s)." if failures else "\nReady.")
    return 1 if failures else 0

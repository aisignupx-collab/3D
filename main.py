#!/usr/bin/env python3
"""Local AI 3D asset pipeline - command-line entry point.

Examples
--------
    # Text -> concept sheet -> TripoSR -> Blender cleanup/UV/PBR -> audited GLB
    python main.py --prompt "A functional MP5 submachine gun" --target_format GLB --enable_rigging

    # Image -> mesh, rigged character as FBX
    python main.py --image_input knight.png --prompt "armored knight" --target_format FBX --enable_rigging

    # Clean up / retopologize / texture an existing AI mesh
    python main.py --mesh_input raw.glb --prompt "medieval castle wall" --target_format OBJ

    # No GPU? Exercise Stages 3-6 with a placeholder mesh
    python main.py --prompt "wooden door" --reconstruction_backend proxy --llm_provider none

    # Verify Blender / CUDA / LLM / ComfyUI connectivity
    python main.py --check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import PipelineConfig, load_config

log = logging.getLogger("asset_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Local multi-agent 3D asset generation pipeline (LLM + diffusion + image-to-3D + Blender).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[1] if __doc__ else None,
    )
    src = ap.add_argument_group("input")
    src.add_argument("--prompt", help='Asset description, e.g. "A functional MP5 submachine gun"')
    src.add_argument("--image_input", type=Path, help="Reference image (skips text-to-image)")
    src.add_argument("--mesh_input", type=Path, help="Existing raw mesh (skips image-to-3D)")

    out = ap.add_argument_group("output")
    out.add_argument("--target_format", type=str.upper, choices=["GLB", "FBX", "OBJ"], default="GLB")
    out.add_argument("--enable_rigging", action="store_true",
                     help="Stage 5: segment moving parts (props) or build an armature (characters)")
    out.add_argument("--export_parts", action="store_true",
                     help="Also export every part as a millimetre-scale STL for 3D printing")
    out.add_argument("--output_dir", type=Path, help="Run directory (default: <output_root>/<timestamp>_<name>)")
    out.add_argument("--name", help="Asset name used for files (default: from the concept agent)")

    eng = ap.add_argument_group("engines (override the config file)")
    eng.add_argument("--config", type=Path, help="YAML config (see config.example.yaml)")
    eng.add_argument("--blender", help="Path to the Blender executable")
    eng.add_argument("--reconstruction_backend", choices=["triposr", "instantmesh", "proxy"])
    eng.add_argument("--diffusion_backend", choices=["comfyui", "diffusers", "none"])
    eng.add_argument("--llm_provider", choices=["openai_compatible", "anthropic", "none"],
                     help="Provider for both the concept and the vision LLM")
    eng.add_argument("--max_iterations", type=int, help="Max Stage 3-6 refinement iterations")
    eng.add_argument("--texture_size", type=int, choices=[1024, 2048, 4096])
    eng.add_argument("--target_faces", type=int, help="Quad budget for the remesh")
    eng.add_argument("--no_vision_audit", action="store_true", help="Geometric checks only in Stage 6")

    ap.add_argument("--check", action="store_true", help="Check the local environment and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if not args.check and not (args.prompt or args.image_input or args.mesh_input):
        ap.error("one of --prompt, --image_input or --mesh_input is required")
    for path in (args.image_input, args.mesh_input):
        if path and not path.is_file():
            ap.error(f"file not found: {path}")
    return args


def apply_cli_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    if args.blender:
        cfg.blender.executable = args.blender
    if args.reconstruction_backend:
        cfg.reconstruction.backend = args.reconstruction_backend
    if args.diffusion_backend:
        cfg.diffusion.backend = args.diffusion_backend
    if args.llm_provider:
        cfg.llm.provider = cfg.vision.provider = args.llm_provider
    if args.max_iterations is not None:
        cfg.audit.max_iterations = args.max_iterations
    if args.texture_size:
        cfg.blender.texture_size = args.texture_size
    if args.target_faces:
        cfg.blender.target_faces = args.target_faces
    if args.no_vision_audit:
        cfg.audit.use_vision_llm = False
    return cfg


def check_environment(cfg: PipelineConfig) -> bool:
    """Print a readiness report for every engine the config points at."""
    import importlib.util

    import requests

    ok = True

    def report(name: str, good: bool, detail: str, required: bool = True) -> None:
        nonlocal ok
        mark = "OK " if good else ("ERR" if required else "-- ")
        print(f"[{mark}] {name:<22} {detail}")
        ok &= good or not required

    try:
        from blender_automation import BlenderRunner

        runner = BlenderRunner(cfg.blender.executable)
        report("Blender", True, f"{runner.version()} ({runner.executable})")
    except Exception as exc:  # noqa: BLE001
        report("Blender", False, str(exc))

    if importlib.util.find_spec("torch"):
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report("PyTorch CUDA", True, f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
                                         f"{props.name} {props.total_memory / 2**30:.0f} GB")
        else:
            report("PyTorch CUDA", False, f"torch {torch.__version__} without CUDA",
                   required=cfg.reconstruction.backend != "proxy")
    else:
        report("PyTorch", False, "not installed", required=cfg.reconstruction.backend != "proxy")

    recon = cfg.reconstruction
    if recon.backend == "triposr":
        found = bool(recon.triposr_repo and (Path(recon.triposr_repo).expanduser() / "tsr").is_dir())
        report("TripoSR repo", found, recon.triposr_repo or "reconstruction.triposr_repo not set")
    elif recon.backend == "instantmesh":
        found = bool(recon.instantmesh_repo and (Path(recon.instantmesh_repo).expanduser() / "run.py").is_file())
        report("InstantMesh repo", found, recon.instantmesh_repo or "reconstruction.instantmesh_repo not set")
    else:
        report("Reconstruction", True, "proxy backend (no ML)")

    report("rembg", importlib.util.find_spec("rembg") is not None, "background removal",
           required=recon.backend != "proxy")

    if cfg.diffusion.backend == "comfyui":
        try:
            r = requests.get(cfg.diffusion.comfyui_url.rstrip("/") + "/system_stats", timeout=5)
            report("ComfyUI", r.ok, cfg.diffusion.comfyui_url, required=False)
        except requests.RequestException as exc:
            report("ComfyUI", False, f"{cfg.diffusion.comfyui_url}: {exc.__class__.__name__}", required=False)
    elif cfg.diffusion.backend == "diffusers":
        report("diffusers", importlib.util.find_spec("diffusers") is not None, cfg.diffusion.diffusers_model,
               required=False)

    for label, llm in (("Concept LLM", cfg.llm), ("Vision LLM", cfg.vision)):
        if llm.provider == "none":
            report(label, True, "disabled (heuristics / geometric checks only)", required=False)
        elif llm.provider == "anthropic":
            from llm_client import LLMClient

            report(label, bool(LLMClient(llm).api_key), f"anthropic {llm.model}", required=False)
        else:
            try:
                r = requests.get(llm.base_url.rstrip("/") + "/models", timeout=5)
                models = [m.get("id") for m in r.json().get("data", [])] if r.ok else []
                has = llm.model in models or not models
                report(label, r.ok and has, f"{llm.base_url} model={llm.model}"
                       + ("" if has else f" (not pulled; available: {', '.join(models[:5])})"), required=False)
            except (requests.RequestException, ValueError) as exc:
                report(label, False, f"{llm.base_url}: {exc.__class__.__name__}", required=False)
    print("\nReady." if ok else "\nRequired components missing (see ERR lines).")
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = apply_cli_overrides(load_config(args.config), args)
    if args.check:
        return 0 if check_environment(cfg) else 1

    from pipeline import Pipeline, RunOptions

    opts = RunOptions(
        prompt=args.prompt,
        image_input=args.image_input,
        mesh_input=args.mesh_input,
        target_format=args.target_format,
        enable_rigging=args.enable_rigging,
        export_parts=args.export_parts,
        output_dir=args.output_dir,
        name=args.name,
    )
    try:
        manifest = Pipeline(cfg, opts).run()
    except Exception:
        log.exception("Pipeline failed")
        return 1
    last = manifest["iterations"][-1] if manifest.get("iterations") else {}
    print(json.dumps({
        "output": manifest.get("output"),
        "approved": manifest.get("approved"),
        "iterations": len(manifest.get("iterations", [])),
        "dimensions_m": (last.get("metrics") or {}).get("dimensions_m"),
        "faces": (last.get("metrics") or {}).get("faces"),
        "watertight": (last.get("metrics") or {}).get("watertight"),
        "run_dir": manifest.get("run_dir"),
    }, indent=2))
    return 0 if manifest.get("approved") else 2


if __name__ == "__main__":
    sys.exit(main())

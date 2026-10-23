#!/usr/bin/env python3
"""Local 3D asset generation pipeline - command line entry point.

Examples:
  python main.py --prompt "A functional MP5 submachine gun" --target_format fbx --enable_rigging
  python main.py --image_input chair.jpg --target_format glb
  python main.py --prompt "Medieval castle wall" --texture_resolution 4096
  python main.py --resume output/mp5_20261023_101500 --target_format obj
  python main.py --doctor
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from asset_pipeline.config import load_config
from asset_pipeline.orchestrator import Pipeline, RunOptions


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prompt/image -> production-ready 3D asset (local GPU + headless Blender).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("input")
    src.add_argument("--prompt", help="text description of the asset")
    src.add_argument("--image_input", help="reference image (skips diffusion; --prompt optional)")
    src.add_argument("--resume", metavar="RUN_DIR", help="re-run Stages 3-6 of an existing run")

    out = p.add_argument_group("output")
    out.add_argument("--target_format", choices=["glb", "fbx", "obj"], type=str.lower, default="glb")
    out.add_argument("--enable_rigging", action="store_true",
                     help="Stage 5: segment moving parts (props) or rig (characters/creatures)")
    out.add_argument("--rig_method", choices=["rigify", "generated"], default="rigify")
    out.add_argument("--print_clearance_mm", type=float, default=0.0,
                     help="gap cut between segmented parts for 3D printing")
    out.add_argument("--output_dir", help="override pipeline.output_dir")

    tune = p.add_argument_group("tuning")
    tune.add_argument("--config", help="YAML config (default: ./config.yaml if present)")
    tune.add_argument("--recon_backend", choices=["triposr", "instantmesh", "openlrm", "command", "primitive"])
    tune.add_argument("--diffusion_backend", choices=["diffusers", "comfyui"])
    tune.add_argument("--texture_resolution", type=int, choices=[1024, 2048, 4096])
    tune.add_argument("--target_faces", type=int, help="face budget (default: planner's estimate)")
    tune.add_argument("--remesh_mode", choices=["quadriflow", "voxel", "voxel_shrinkwrap", "decimate", "none"])
    tune.add_argument("--shape_mode", choices=["auto", "primitives", "mesh"],
                      help="rebuild from primitive templates: when they fit / always / never")
    tune.add_argument("--max_iterations", type=int, help="Stage 6 refinement iterations")
    tune.add_argument("--blender_path", help="Blender executable")
    tune.add_argument("--seed", type=int)
    tune.add_argument("--offline", action="store_true",
                      help="no LLM/VLM calls: knowledge-base planning, metrics-only audit")
    tune.add_argument("--dry_run", action="store_true",
                      help="no GPU: offline + primitive reconstruction + no diffusion (tests Blender stages)")

    p.add_argument("--doctor", action="store_true", help="check CUDA, models, Blender and LLM endpoints")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    if args.prompt is not None and not args.prompt.strip():
        p.error("--prompt is empty")
    if not args.doctor and not (args.prompt or args.image_input or args.resume):
        p.error("one of --prompt, --image_input or --resume is required")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "PIL", "diffusers", "transformers"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))
    cfg = load_config(args.config)

    if args.output_dir:
        cfg.pipeline.output_dir = args.output_dir
    if args.recon_backend:
        cfg.reconstruction.backend = args.recon_backend
    if args.diffusion_backend:
        cfg.diffusion.backend = args.diffusion_backend
    if args.texture_resolution:
        cfg.texture.resolution = args.texture_resolution
    if args.target_faces:
        cfg.cleanup.target_faces = args.target_faces
    if args.remesh_mode:
        cfg.cleanup.remesh_mode = args.remesh_mode
    if args.shape_mode:
        cfg.shapes.mode = args.shape_mode
    if args.max_iterations:
        cfg.pipeline.max_iterations = args.max_iterations
    if args.blender_path:
        cfg.blender.executable = args.blender_path
    if args.seed is not None:
        cfg.pipeline.seed = args.seed
    if args.dry_run:
        cfg.reconstruction.backend = "primitive"
        args.offline = True

    if args.doctor:
        from asset_pipeline.doctor import run_doctor

        return run_doctor(cfg)

    options = RunOptions(
        prompt=args.prompt,
        image_input=args.image_input,
        target_format=args.target_format,
        enable_rigging=args.enable_rigging,
        offline=args.offline,
        skip_references=args.dry_run and not args.image_input,
        print_clearance_mm=args.print_clearance_mm,
        rig_method=args.rig_method,
        resume=args.resume,
    )
    if args.image_input and not Path(args.image_input).is_file():
        logging.error("image not found: %s", args.image_input)
        return 2

    try:
        ctx = Pipeline(cfg).run(options)
    except (FileNotFoundError, ValueError) as exc:  # bad paths / input: report, no traceback
        logging.error("%s", exc)
        return 2
    summary = {
        "run_dir": str(ctx.run_dir),
        "asset": ctx.export["output"] if ctx.export else None,
        "approved": ctx.export.get("approved") if ctx.export else False,
        "score": ctx.export.get("score") if ctx.export else None,
        "iterations": len(ctx.iterations),
        "timings_s": ctx.timings,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

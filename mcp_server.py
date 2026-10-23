#!/usr/bin/env python3
"""MCP server: lets Claude Code, Codex, OpenCode (or any MCP client) drive the pipeline.

Register it with your agent, e.g.
  claude mcp add asset-pipeline -- python /path/to/mcp_server.py
  codex mcp add asset-pipeline -- python /path/to/mcp_server.py
  (OpenCode: add a "local" MCP entry with the same command in opencode.json)

Generation runs synchronously and takes minutes; raise the client's tool timeout
(e.g. MCP_TOOL_TIMEOUT=1800000 for Claude Code). stdout is the MCP channel, so
all logging goes to stderr.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # MCP Python SDK 2.x renamed FastMCP to MCPServer; same decorator API
    from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: E402
except ImportError:
    from mcp.server.fastmcp import FastMCP  # noqa: E402

from asset_pipeline.config import load_config  # noqa: E402
from asset_pipeline.orchestrator import Pipeline, RunOptions  # noqa: E402
from asset_pipeline.shape_templates import PRIMITIVES, RECIPES  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
mcp = FastMCP("asset-pipeline")


def _summary(ctx) -> Dict[str, Any]:
    last = ctx.iterations[-1] if ctx.iterations else {}
    audit = last.get("audit", {})
    return {
        "run_dir": str(ctx.run_dir),
        "asset": ctx.export.get("output") if ctx.export else None,
        "files": ctx.export.get("files") if ctx.export else [],
        "previews": sorted(str(p) for p in (ctx.run_dir / "export" / "previews").glob("*.png")),
        "approved": audit.get("approved"),
        "score": audit.get("score"),
        "issues": [i.get("description") for i in audit.get("issues", [])],
        "iterations": len(ctx.iterations),
        "spec": {k: ctx.spec.to_dict()[k] for k in ("name", "category", "length_m", "width_m", "height_m")}
        if ctx.spec else None,
        "timings_s": ctx.timings,
    }


def _run(options: RunOptions, config: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_config(config or None)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        setattr(getattr(cfg, section), key, value)
    with contextlib.redirect_stdout(sys.stderr):  # keep the MCP channel clean
        ctx = Pipeline(cfg).run(options)
    return _summary(ctx)


@mcp.tool()
def generate_asset(prompt: str = "", image_path: str = "", target_format: str = "glb",
                   enable_rigging: bool = False, shape_mode: str = "auto", texture_resolution: int = 2048,
                   max_iterations: int = 3, offline: bool = False, config: str = "") -> Dict[str, Any]:
    """Generate a game-ready 3D asset from a text prompt and/or a reference image.

    Runs concept planning, diffusion references, image-to-3D, Blender cleanup, primitive
    templates / blueprint rebuild, UVs + PBR (with reference projection), segmentation or
    rigging, and the vision-audited refinement loop. Returns the exported file, previews and
    the final audit. target_format: glb | fbx | obj. shape_mode: auto | primitives | mesh.
    """
    if not prompt and not image_path:
        raise ValueError("give a prompt, an image_path, or both")
    options = RunOptions(prompt=prompt or None, image_input=image_path or None, target_format=target_format,
                         enable_rigging=enable_rigging, offline=offline)
    return _run(options, config, {"shapes.mode": shape_mode, "texture.resolution": texture_resolution,
                                  "pipeline.max_iterations": max_iterations})


@mcp.tool()
def resume_asset(run_dir: str, target_format: str = "glb", enable_rigging: bool = False,
                 max_iterations: int = 3, config: str = "") -> Dict[str, Any]:
    """Re-run the Blender stages (3-6) of an existing run with current settings - no GPU stages."""
    options = RunOptions(resume=run_dir, target_format=target_format, enable_rigging=enable_rigging)
    return _run(options, config, {"pipeline.max_iterations": max_iterations})


@mcp.tool()
def list_runs(output_dir: str = "output", limit: int = 20) -> List[Dict[str, Any]]:
    """Recent pipeline runs with their exported asset and audit result."""
    runs = []
    for manifest in sorted(Path(output_dir).glob("*/manifest.json"), key=lambda p: p.stat().st_mtime,
                           reverse=True)[:limit]:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        export = data.get("export") or {}
        runs.append({"run_dir": str(manifest.parent), "prompt": (data.get("options") or {}).get("prompt"),
                     "asset": export.get("output"), "approved": export.get("approved"),
                     "score": export.get("score")})
    return runs


@mcp.tool()
def list_shape_templates() -> Dict[str, Any]:
    """Primitive vocabulary and the built-in recipe library (keywords per recipe)."""
    return {"primitives": PRIMITIVES, "recipes": [r["keywords"] for r in RECIPES]}


@mcp.tool()
def doctor(config: str = "") -> str:
    """Check CUDA/VRAM, Python packages, TripoSR, Blender and LLM endpoints."""
    from asset_pipeline.doctor import run_doctor

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_doctor(load_config(config or None))
    return buf.getvalue()


if __name__ == "__main__":
    mcp.run()

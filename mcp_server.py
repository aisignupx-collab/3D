#!/usr/bin/env python3
"""MCP server: lets Claude Code, Codex, OpenCode (or any MCP client) drive the pipeline.

Register it with your agent, e.g.
  claude mcp add asset-pipeline -- python /path/to/mcp_server.py
  codex mcp add asset-pipeline -- python /path/to/mcp_server.py
  (OpenCode: add a "local" MCP entry with the same command in opencode.json)

A generation takes minutes. ``start_asset_job`` + ``get_job`` run it in the
background, so no client tool timeout is hit (recommended for OpenCode); the
blocking ``generate_asset`` needs a long client timeout (Claude Code:
MCP_TOOL_TIMEOUT=1800000). Jobs run one at a time so they never compete for
the GPU. stdout is the MCP channel, so all logging goes to stderr.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
import threading
import time
import traceback
import uuid
from collections import deque
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


def _pipeline(config: str, overrides: Dict[str, Any]) -> Pipeline:
    cfg = load_config(config or None)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        setattr(getattr(cfg, section), key, value)
    return Pipeline(cfg)


def _run(options: RunOptions, config: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    pipe = _pipeline(config, overrides)
    with _GPU_LOCK, contextlib.redirect_stdout(sys.stderr):  # keep the MCP channel clean
        ctx = pipe.run(options)
    return _summary(ctx)


# ----------------------------------------------------------------- background jobs

_JOBS: Dict[str, Dict[str, Any]] = {}
_THREAD_JOB: Dict[int, Dict[str, Any]] = {}
_GPU_LOCK = threading.Lock()  # one generation at a time: they would compete for VRAM


class _JobLog(logging.Handler):
    """Mirror each job thread's pipeline log into its job record (stage + recent lines)."""

    def emit(self, record):
        job = _THREAD_JOB.get(record.thread)
        if job is None or not record.name.startswith("asset_pipeline"):
            return
        msg = record.getMessage()
        job["log_tail"].append(f"{time.strftime('%H:%M:%S', time.localtime(record.created))} {msg}")
        if msg.startswith(("STAGE", "ITERATION")):
            job["stage"] = msg
        if msg.startswith("Run directory: "):
            job["run_dir"] = msg.split(": ", 1)[1]


logging.getLogger().addHandler(_JobLog())


def _job_worker(job: Dict[str, Any], options: RunOptions, config: str, overrides: Dict[str, Any]):
    _THREAD_JOB[threading.get_ident()] = job
    try:
        pipe = _pipeline(config, overrides)
        with _GPU_LOCK:
            job.update(status="running", started=time.time())
            ctx = pipe.run(options)  # pipeline logs to stderr; no stdout redirect from a thread
        job.update(status="done", result=_summary(ctx))
    except Exception as exc:  # report every failure to the client instead of losing it
        job.update(status="error", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-3000:])
    finally:
        job["finished"] = time.time()
        _THREAD_JOB.pop(threading.get_ident(), None)


def _public(job: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in job.items() if k != "log_tail"}
    out["log_tail"] = list(job["log_tail"])[-12:]
    end = job.get("finished") or time.time()
    out["elapsed_s"] = round(end - job["started"], 1) if job.get("started") else 0.0
    return out


@mcp.tool()
def start_asset_job(prompt: str = "", image_path: str = "", target_format: str = "glb",
                    enable_rigging: bool = False, shape_mode: str = "auto", texture_resolution: int = 2048,
                    max_iterations: int = 3, offline: bool = False, config: str = "") -> Dict[str, Any]:
    """Start generating a 3D asset in the background and return a job id immediately.

    Same inputs as generate_asset. Poll get_job(job_id) every 20-60 s until status is
    "done" (result holds the exported file, previews and audit) or "error". Jobs queue and
    run one at a time.
    """
    prompt, image_path = (prompt or "").strip(), (image_path or "").strip()
    if not prompt and not image_path:
        raise ValueError("give a prompt, an image_path, or both")
    job = {"job_id": uuid.uuid4().hex[:8], "status": "queued", "prompt": prompt, "image_path": image_path,
           "stage": "queued", "run_dir": None, "log_tail": deque(maxlen=40)}
    _JOBS[job["job_id"]] = job
    options = RunOptions(prompt=prompt or None, image_input=image_path or None, target_format=target_format,
                         enable_rigging=enable_rigging, offline=offline)
    overrides = {"shapes.mode": shape_mode, "texture.resolution": texture_resolution,
                 "pipeline.max_iterations": max_iterations}
    threading.Thread(target=_job_worker, args=(job, options, config, overrides), daemon=True).start()
    return _public(job)


@mcp.tool()
def get_job(job_id: str) -> Dict[str, Any]:
    """Status of a background job: queued | running | done | error, current stage, recent log, result."""
    if job_id not in _JOBS:
        raise ValueError(f"unknown job {job_id}; see list_jobs")
    return _public(_JOBS[job_id])


@mcp.tool()
def list_jobs() -> List[Dict[str, Any]]:
    """All background jobs started by this server session (newest last)."""
    return [{k: _public(j).get(k) for k in ("job_id", "status", "prompt", "stage", "elapsed_s", "run_dir")}
            for j in _JOBS.values()]


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
    prompt, image_path = (prompt or "").strip(), (image_path or "").strip()
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

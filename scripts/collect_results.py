#!/usr/bin/env python3
"""Bundle pipeline runs into one QA results archive.

Collects, for every run under ``--runs``:
  * the key metrics from manifest.json (spec, per-iteration audit, stage timings,
    template/blueprint fit, texture projection, rig landmarks and pose test, export)
    -> ``summary.json`` / ``summary.md``;
  * the concept sheet, spec and manifest;
  * the audit renders of every iteration (``renders/iter_N/*.png``, incl. posed views);
  * the exported asset with its textures (``export/``; skip with ``--no-assets``);
  * ``inspect_export.json`` from scripts/inspect_export.py when Blender is found;
  * the tail of every Blender log that reported an error.
Adds an environment snapshot (OS, Python, GPU, Blender, key package versions),
then zips everything.

  python scripts/collect_results.py --runs output --out qa_results
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGES = ("torch", "torchvision", "diffusers", "transformers", "huggingface_hub", "rembg", "onnxruntime",
            "onnxruntime-gpu", "trimesh", "numpy", "Pillow", "mcp", "torchmcubes")


def _cmd(args: List[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({exc})"


def environment() -> Dict[str, Any]:
    from asset_pipeline.config import find_blender

    env: Dict[str, Any] = {"os": platform.platform(), "python": platform.python_version(), "packages": {}}
    for name in PACKAGES:
        try:
            env["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    env["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    try:
        import torch

        env["torch_cuda"] = {"available": torch.cuda.is_available(), "cuda": torch.version.cuda,
                             "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        env["torch_cuda"] = "torch not installed"
    exe = find_blender()
    env["blender"] = _cmd([exe, "--version"]).splitlines()[0] if exe else "not found"
    return env


def summarise_run(run: Path) -> Dict[str, Any]:
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    spec = m.get("spec") or {}
    out: Dict[str, Any] = {
        "run": run.name,
        "prompt": (m.get("options") or {}).get("prompt"),
        "image_input": (m.get("options") or {}).get("image_input"),
        "options": m.get("options"),
        "backends": {k: (m.get("config") or {}).get(k, {}).get("backend") for k in ("diffusion", "reconstruction")},
        "models": {k: (m.get("config") or {}).get(k, {}).get("model") for k in ("llm", "vision")},
        "spec": {k: spec.get(k) for k in ("name", "category", "length_m", "width_m", "height_m", "fit_dimension",
                                          "rig_type", "face_budget")},
        "parts_planned": [p.get("name") for p in spec.get("parts", [])],
        "timings_s": m.get("timings"),
        "export": {k: (m.get("export") or {}).get(k) for k in ("output", "bytes", "approved", "score", "iteration")},
        "iterations": [],
    }
    for it in m.get("iterations", []):
        s3, s3b, s4, s5 = (it.get(k) or {} for k in ("stage3", "stage3b", "stage4", "stage5"))
        proj = s4.get("projection") or {}
        rec = {
            "iteration": it.get("iteration"),
            "stage3": {k: s3.get(k) for k in ("remesh", "raw_faces", "rotation_applied_deg", "front_yaw_deg",
                                              "dimension_error", "seconds")} if "reused" not in s3 else "reused",
            "templates": {k: s3b.get(k) for k in ("used", "source", "fit_error", "parts_built", "blueprint",
                                                  "candidates", "error")} if s3b else None,
            "texture": {"albedo": s4.get("albedo"), "material_ids": s4.get("material_ids"),
                        "coverage": s4.get("material_coverage"),
                        "projection": {k: proj.get(k) for k in ("silhouette_iou", "projected_fraction", "passes",
                                                                "skipped")} if proj else None},
            "stage5": {k: s5.get(k) for k in ("method", "weights", "weight_coverage", "deform_bones", "landmarks",
                                              "pose_test", "parts", "skipped")} if s5 else None,
            "audit": {k: (it.get("audit") or {}).get(k) for k in ("approved", "score", "vision_error")},
            "issues": [f"[{i.get('severity')}/{i.get('source')}] {i.get('description')} -> {i.get('fix')}"
                       for i in (it.get("audit") or {}).get("issues", [])],
            "fixes_applied": it.get("fixes"),
            "metrics": {name: {k: v.get(k) for k in ("faces", "quad_ratio", "watertight", "dimensions_m", "stretch",
                                                     "islands")}
                        for name, v in ((it.get("render") or {}).get("metrics") or {}).items()},
        }
        out["iterations"].append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["output"], help="run folders or parent folders of runs")
    ap.add_argument("--out", default="qa_results")
    ap.add_argument("--notes", help="optional file (e.g. the tester's report) to include")
    ap.add_argument("--no-assets", action="store_true", help="skip exported asset files (smaller zip)")
    args = ap.parse_args()

    from asset_pipeline.config import find_blender

    blender = find_blender()
    inspector = Path(__file__).resolve().parent / "inspect_export.py"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs: List[Path] = []
    for root in map(Path, args.runs):
        runs += [root] if (root / "manifest.json").is_file() else sorted(p.parent for p in root.glob("*/manifest.json"))

    summary = {"environment": environment(), "runs": []}
    for run in runs:
        try:
            info = summarise_run(run)
        except (OSError, ValueError, KeyError) as exc:
            info = {"run": run.name, "error": f"unreadable manifest: {exc}"}
        summary["runs"].append(info)
        dest = out / "runs" / run.name
        dest.mkdir(parents=True, exist_ok=True)
        for src in [run / "references" / "concept_sheet.png", run / "spec.json", run / "manifest.json"]:
            if src.is_file():
                shutil.copy2(src, dest / src.name)
        # Audit renders of every iteration (front/side/wireframe/shaded/uv_checker + posed/)
        for src in sorted((run / "renders").glob("iter_*/**/*.png")):
            rel = src.relative_to(run / "renders")
            (dest / "renders" / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / "renders" / rel)
        export = (info.get("export") or {}).get("output")
        if export and Path(export).is_file():
            if not args.no_assets:
                shutil.copytree(Path(export).parent, dest / "export", dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("*.blend", "*.blend1"))
            if blender:
                report = dest / "inspect_export.json"
                subprocess.run([blender, "--background", "--factory-startup", "--python", str(inspector), "--",
                                export, "--json", str(report)], capture_output=True, timeout=600)
                if report.is_file():
                    info["inspect_warnings"] = json.loads(report.read_text(encoding="utf-8")).get("warnings")
        for log in sorted((run / "blender_jobs").glob("blender_*.log")):
            text = log.read_text(encoding="utf-8", errors="replace")
            if "Traceback" in text or "Error" in text:
                (dest / f"ERROR_{log.name}").write_text("\n".join(text.splitlines()[-80:]), encoding="utf-8")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    lines = ["# QA results", "", "## Environment", "", "```", json.dumps(summary["environment"], indent=2), "```", "",
             "## Runs", "", "| run | category | path | approved | score | iters | export | total s |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary["runs"]:
        if "error" in r:
            lines.append(f"| {r['run']} | - | - | - | - | - | {r['error']} | - |")
            continue
        last = r["iterations"][-1] if r["iterations"] else {}
        tmpl = last.get("templates") or {}
        path = tmpl.get("source") if tmpl.get("used") else "scan mesh"
        total = round(sum((r.get("timings_s") or {}).values()), 1)
        exp = r["export"]
        lines.append(f"| {r['run']} | {r['spec'].get('category')} | {path} | {exp.get('approved')} | {exp.get('score')} "
                     f"| {len(r['iterations'])} | {Path(exp.get('output') or '-').name} | {total} |")
    if args.notes and Path(args.notes).is_file():
        shutil.copy2(args.notes, out / Path(args.notes).name)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"{len(runs)} runs -> {out}/summary.md, {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

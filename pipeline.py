"""Orchestrator: runs the six stages as cooperating agents and owns the
Stage 3-6 refinement loop.

    ConceptAgent ─▶ ReferenceGenerator ─▶ ReconstructionBackend ─▶
        ┌──────────── BlenderRunner.process (Stages 3, 4, 5 + renders) ◀──┐
        └─▶ VisionAuditor.audit ─▶ apply_corrective_actions ──(not approved)┘
                         │ approved / out of iterations
                         ▼
                 BlenderRunner.export (GLB / FBX / OBJ [+ STL parts])

Every run gets its own directory with all intermediate artifacts and a
``manifest.json`` describing what happened, so any stage can be inspected or
re-run by hand.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from asset_spec import AssetSpec
from blender_automation import BlenderError, BlenderRunner
from config import PipelineConfig
from llm_client import LLMClient
from mesh_reconstruction import build_backend
from reference_generator import ConceptAgent, ReferenceGenerator
from vision_auditor import AuditReport, VisionAuditor, apply_corrective_actions

log = logging.getLogger(__name__)


@dataclass
class RunOptions:
    prompt: Optional[str] = None
    image_input: Optional[Path] = None
    mesh_input: Optional[Path] = None  # skip Stages 1-2 image work, start from a mesh
    target_format: str = "GLB"
    enable_rigging: bool = False
    export_parts: bool = False
    output_dir: Optional[Path] = None
    name: Optional[str] = None


class StageTimer:
    def __init__(self, manifest: dict[str, Any]):
        self.manifest = manifest

    def __call__(self, name: str):
        manifest = self.manifest

        class _Ctx:
            def __enter__(self_inner):
                log.info("=== %s ===", name)
                self_inner.start = time.monotonic()
                manifest["stages"][name] = {"status": "running"}
                return manifest["stages"][name]

            def __exit__(self_inner, exc_type, exc, tb):
                entry = manifest["stages"][name]
                entry["seconds"] = round(time.monotonic() - self_inner.start, 2)
                entry["status"] = "failed" if exc_type else entry.get("status_override", "done")
                entry.pop("status_override", None)
                if exc_type:
                    entry["error"] = f"{exc_type.__name__}: {exc}"
                return False

        return _Ctx()


def build_blender_params(cfg: PipelineConfig, spec: AssetSpec, backend_rotation: list[float]) -> dict[str, Any]:
    """Flatten the Blender config + spec into the job parameter dict."""
    p = dataclasses.asdict(cfg.blender)
    for host_only in ("executable", "timeout_s"):
        p.pop(host_only)
    # A non-zero configured import rotation overrides the backend's default.
    if not any(p["import_rotation_deg"]):
        p["import_rotation_deg"] = list(backend_rotation)
    p.update(
        target_dimensions_m=list(spec.dimensions_m),
        primary_axis=spec.primary_axis,
        origin=spec.origin,
        extra_rotation_deg=[0.0, 0.0, 0.0],
        fit_compensation=1.0,
        cutters=[],
        seed=cfg.diffusion.seed,
        bake=True,
    )
    return p


def stage5_plan(spec: AssetSpec, enabled: bool) -> dict[str, bool]:
    """Decide between part segmentation (props) and armature rigging (characters)."""
    if not enabled:
        return {"segment": False, "rig": False}
    if spec.category in ("character", "creature") and spec.rig_type == "none":
        spec.rig_type = "humanoid" if spec.category == "character" else "quadruped"
    return {
        "segment": bool(spec.moving_parts) and spec.category not in ("character", "creature"),
        "rig": spec.rig_type != "none",
    }


class Pipeline:
    def __init__(self, cfg: PipelineConfig, opts: RunOptions):
        if not (opts.prompt or opts.image_input or opts.mesh_input):
            raise ValueError("Provide --prompt, --image_input or --mesh_input")
        self.cfg = cfg
        self.opts = opts
        self.llm = LLMClient(cfg.llm)
        self.vision = LLMClient(cfg.vision, max_image_side=cfg.vision.max_image_side)
        self.blender = BlenderRunner(cfg.blender.executable, cfg.blender.timeout_s)

    def _run_dir(self, spec_name: str) -> Path:
        if self.opts.output_dir:
            return Path(self.opts.output_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.cfg.output_root) / f"{stamp}_{spec_name}"

    def run(self) -> dict[str, Any]:
        opts, cfg = self.opts, self.cfg
        manifest: dict[str, Any] = {
            "started": datetime.now().isoformat(timespec="seconds"),
            "options": {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(opts).items()},
            "blender": self.blender.version(),
            "stages": {},
            "iterations": [],
        }
        timer = StageTimer(manifest)
        staging = Path(cfg.output_root) / "_staging"
        staging.mkdir(parents=True, exist_ok=True)

        # ---------------- Stage 1 ----------------
        with timer("1_concept_reference") as st:
            needs_images = opts.mesh_input is None and cfg.reconstruction.backend != "proxy"
            refgen = ReferenceGenerator(cfg.diffusion, cfg.reconstruction, ConceptAgent(self.llm, self.vision))
            ref = refgen.run(opts.prompt, opts.image_input, staging / "reference", need_images=needs_images)
            spec = ref.spec
            if opts.name:
                spec.name = opts.name
            run_dir = self._run_dir(spec.name)
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "reference").exists():
                shutil.rmtree(run_dir / "reference")
            shutil.move(str(staging / "reference"), run_dir / "reference")
            ref_dir = run_dir / "reference"
            st["spec_source"] = ref.spec_source
            st["artifacts"] = sorted(str(p) for p in ref_dir.iterdir())
            manifest["run_dir"] = str(run_dir)
            manifest["spec"] = spec.to_dict()
        self._attach_log(run_dir)
        primary_input = ref_dir / ref.primary_input.name if ref.primary_input else None

        # ---------------- Stage 2 ----------------
        with timer("2_reconstruction") as st:
            backend_rotation = [0.0, 0.0, 0.0]
            if opts.mesh_input:
                raw_mesh = Path(opts.mesh_input)
                st["source"] = "user mesh"
            else:
                backend = build_backend(cfg.reconstruction)
                try:
                    rec = backend.reconstruct(primary_input, run_dir / "raw_mesh", spec)
                finally:
                    backend.close()  # free VRAM before Blender bakes on the same GPU
                raw_mesh, backend_rotation = rec.mesh_path, rec.import_rotation_deg
                st["backend"] = rec.backend
            st["raw_mesh"] = str(raw_mesh)

        # ---------------- Stages 3-6 (refinement loop) ----------------
        params = build_blender_params(cfg, spec, backend_rotation)
        stage5 = stage5_plan(spec, opts.enable_rigging)
        auditor = VisionAuditor(self.vision, cfg.audit, contact_sheet=cfg.vision.contact_sheet)
        best: Optional[tuple[dict[str, Any], AuditReport]] = None
        with timer("3-6_build_and_audit") as st:
            for it in range(1, max(1, cfg.audit.max_iterations) + 1):
                work = run_dir / "blender" / f"iter_{it:02d}"
                log.info("Iteration %d: Blender stages 3-5 + renders", it)
                try:
                    result = self.blender.process(raw_mesh, spec.to_dict(), params, work, f"iter_{it:02d}", stage5)
                except BlenderError:
                    if params.get("render_engine", "CYCLES").upper() == "CYCLES":
                        raise
                    # EEVEE/Workbench abort without a GPU/EGL context in --background.
                    log.warning("Blender failed with %s renders; retrying with Cycles", params["render_engine"])
                    params["render_engine"] = "CYCLES"
                    result = self.blender.process(raw_mesh, spec.to_dict(), params, work, f"iter_{it:02d}", stage5)
                (work / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
                report = auditor.audit(it, result, spec, params, Path(work) / "renders")
                record = {
                    "iteration": it,
                    "blend": result.get("blend_path"),
                    "metrics": result.get("metrics"),
                    "uv_metrics": result.get("uv_metrics"),
                    "fit": result.get("fit"),
                    "parts": result.get("parts"),
                    "rig": result.get("rig"),
                    "renders": result.get("renders"),
                    "blender_seconds": result.get("wall_seconds"),
                    "audit": report.to_dict(),
                }
                manifest["iterations"].append(record)
                self._write_manifest(run_dir, manifest)
                log.info("Audit %d: %s (score=%s) - %s", it, "APPROVED" if report.approved else "rejected",
                         report.score, report.summary())
                if best is None or _rank(report) > _rank(best[1]):
                    best = (result, report)
                if report.approved:
                    break
                if it == cfg.audit.max_iterations:
                    break
                params, applied = apply_corrective_actions(params, report.actions)
                record["corrections_for_next"] = applied
                if not applied:
                    log.warning("Audit requested no applicable corrections; stopping refinement")
                    break
                for line in applied:
                    log.info("  correction: %s", line)
            assert best is not None
            final_result, final_report = best
            st["iterations"] = len(manifest["iterations"])
            st["approved"] = final_report.approved
            st["selected_iteration"] = final_report.iteration
            if not final_report.approved:
                st["status_override"] = "done_with_warnings"

        # ---------------- Export ----------------
        with timer("export") as st:
            if not final_report.approved and not cfg.audit.export_unapproved:
                raise RuntimeError("Asset was not approved and audit.export_unapproved is false")
            ext = opts.target_format.lower()
            out = run_dir / "export" / f"{spec.name}.{ext}"
            exp = self.blender.export(Path(final_result["blend_path"]), opts.target_format, out,
                                      opts.export_parts, run_dir / "export")
            st["output"] = exp["output_path"]
            st["part_files"] = exp.get("part_files")
            manifest["output"] = exp["output_path"]
            manifest["blend"] = final_result["blend_path"]
            manifest["approved"] = final_report.approved

        manifest["finished"] = datetime.now().isoformat(timespec="seconds")
        self._write_manifest(run_dir, manifest)
        shutil.rmtree(staging, ignore_errors=True)
        return manifest

    @staticmethod
    def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _attach_log(run_dir: Path) -> None:
        handler = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)


def _rank(report: AuditReport) -> tuple[int, int, float]:
    """Order iterations: approved first, then fewest blocking issues, then score."""
    blocking = sum(1 for i in report.issues if i.severity in ("critical", "major"))
    return (int(report.approved), -blocking, report.score or 0.0)

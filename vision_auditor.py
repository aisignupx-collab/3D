"""Stage 6 - closed-loop audit: measured geometry checks + vision LLM review.

Watertightness, self-intersections, real-world dimensions and UV distortion
are *measured* inside Blender (see ``blender_automation._geometry_metrics`` /
``_uv_metrics``); an image model cannot reliably see a non-manifold edge, so
those numbers are hard gates. The vision LLM judges what renders show well:
silhouette, proportions against the real-world reference, texture seams and
stretching, orientation. Both produce *corrective actions* from a fixed
vocabulary that :func:`apply_corrective_actions` maps onto the Stage 3/4
Blender parameters for the next iteration.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from asset_spec import AssetSpec
from config import AuditConfig
from llm_client import LLMClient, LLMUnavailable

log = logging.getLogger(__name__)

SEVERITIES = ("critical", "major", "minor")

# action -> (description shown to the LLM, parameter hint)
ACTIONS: dict[str, tuple[str, str]] = {
    "remesh_finer": ("more voxels and quads: lost detail, blobby or melted shapes", "{}"),
    "remesh_coarser": ("fewer quads: topology far denser than needed", "{}"),
    "make_watertight": ("voxel remesh + hole filling: holes, open or non-manifold surfaces", "{}"),
    "smooth_surface": ("relax lumpy / noisy surfaces", '{"iterations": 4}'),
    "preserve_detail": ("less smoothing, project back onto the source: details washed out", "{}"),
    "rotate": ("fix orientation (object lying on its side / facing backwards)",
               '{"x_deg": 0, "y_deg": 0, "z_deg": 90}'),
    "rescale": ("fix real-world size on one axis", '{"axis": "z", "target_m": 2.1}'),
    "fit_proportions": ("non-uniform scale to match all three spec dimensions", "{}"),
    "reunwrap_uv": ("redo UVs: stretched / smeared / badly scaled textures",
                    '{"method": "smart|seams|lightmap", "angle_limit_deg": 50}'),
    "increase_texture_resolution": ("texture too blurry for the asset size", "{}"),
    "rebake_textures": ("baking artifacts: black patches, projection errors, wrong colours",
                        '{"cage_extrusion_scale": 2.0}'),
    "union_intersections": ("merge intersecting / interpenetrating parts", "{}"),
    "boolean_difference": ("cut a missing opening (bore, window, slot) with a primitive",
                           '{"shape": "box|cylinder", "center_norm": [0.5, 0.1, 0.8], "size_norm": [0.1, 0.2, 0.1]}'),
}

AUDIT_SYSTEM_PROMPT = """You are the QA lead for an automated 3D asset pipeline. You receive renders of a generated asset
(front clay, side clay, wireframe overlay, final PBR shading, UV checker), the design brief and measured mesh statistics.
Judge the asset for production use. Measured statistics are ground truth - do not contradict them.
Check: does the silhouette match the brief and its real-world references? Are proportions physically plausible?
Is the object upright and facing -Y (front view shows its front)? Is topology even (wireframe)? Are textures mapped
without stretching, smearing or visible seams (UV checker squares should be square and evenly sized)?
Return ONLY JSON:
{"approved": bool, "score": 0.0-1.0,
 "issues": [{"category": "geometry|proportions|orientation|topology|uv|texture", "severity": "critical|major|minor", "description": "..."}],
 "actions": [{"action": "<name>", ...parameters}]}
Only request actions that fix a concrete issue you listed. Available actions:
"""


@dataclass
class AuditIssue:
    category: str
    severity: str
    description: str
    source: str = "geometry"  # geometry | vision


@dataclass
class AuditReport:
    iteration: int
    approved: bool
    score: Optional[float]
    issues: list[AuditIssue] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    vision_used: bool = False
    vision_raw: Optional[dict[str, Any]] = None
    contact_sheet: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if not self.issues:
            return "no issues"
        return "; ".join(f"[{i.severity}/{i.source}] {i.description}" for i in self.issues)


def make_contact_sheet(renders: dict[str, str], out_path: Path, cell: int = 512, cols: int = 3) -> Path:
    """Tile renders into one labelled image (for single-image VLMs)."""
    from PIL import Image, ImageDraw

    items = list(renders.items())
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for i, (name, path) in enumerate(items):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((cell, cell))
            x, y = (i % cols) * cell, (i // cols) * cell
            sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        draw.rectangle([x, y, x + 9 * len(name) + 12, y + 20], fill=(0, 0, 0))
        draw.text((x + 6, y + 4), name, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def geometric_review(result: dict[str, Any], spec: AssetSpec, params: dict[str, Any],
                     cfg: AuditConfig) -> tuple[list[AuditIssue], list[dict[str, Any]]]:
    """Hard checks on the metrics Blender measured."""
    issues: list[AuditIssue] = []
    actions: list[dict[str, Any]] = []
    m = result.get("metrics", {})
    uv = result.get("uv_metrics", {})

    if cfg.require_watertight and not m.get("watertight", False):
        issues.append(AuditIssue("geometry", "critical",
                                 f"mesh is not watertight ({m.get('non_manifold_edges')} non-manifold edges, "
                                 f"{m.get('boundary_edges')} boundary edges)"))
        actions.append({"action": "make_watertight"})
    if m.get("normals_inverted"):
        issues.append(AuditIssue("geometry", "major", "normals point inwards (negative volume)"))
        actions.append({"action": "make_watertight"})
    si = m.get("self_intersections")
    if si is not None and si > cfg.max_self_intersections:
        issues.append(AuditIssue("geometry", "major", f"{si} self-intersecting face pairs"))
        actions.append({"action": "union_intersections"})

    dims = m.get("dimensions_m")
    target = params.get("target_dimensions_m", spec.dimensions_m)
    if dims:
        idx = "xyz".index(params.get("primary_axis", spec.primary_axis))
        err = abs(dims[idx] - target[idx]) / target[idx]
        if err > cfg.dimension_tolerance:
            issues.append(AuditIssue("proportions", "critical",
                                     f"{'xyz'[idx]} = {dims[idx]:.3f} m, expected {target[idx]:.3f} m ({err:.0%} off)"))
            # Remeshing/shrinkwrap moved the fitted size: feed the error back.
            actions.append({"action": "compensate_scale", "factor": target[idx] / max(dims[idx], 1e-9)})
        worst = max(
            ((a, abs(dims[a] - target[a]) / target[a]) for a in range(3) if a != idx),
            key=lambda t: t[1],
        )
        if worst[1] > cfg.proportion_tolerance and not params.get("nonuniform_fit"):
            a = worst[0]
            issues.append(AuditIssue("proportions", "major",
                                     f"{'xyz'[a]} = {dims[a]:.3f} m vs expected {target[a]:.3f} m "
                                     f"({worst[1]:.0%} off) - reconstruction proportions disagree with the brief"))
            actions.append({"action": "fit_proportions"})

    if uv.get("has_uvs") is False:
        issues.append(AuditIssue("uv", "critical", "mesh has no UV map"))
        actions.append({"action": "reunwrap_uv"})
    elif uv:
        if uv.get("stretch_score", 0) > cfg.max_uv_stretch:
            issues.append(AuditIssue("uv", "major",
                                     f"UV texel-density distortion {uv['stretch_score']:.2f} > {cfg.max_uv_stretch}"))
            actions.append({"action": "reunwrap_uv"})
        zero = uv.get("zero_area_uv_faces", 0)
        if zero > 0.01 * max(m.get("faces", 1), 1):
            issues.append(AuditIssue("uv", "minor", f"{zero} faces have zero UV area"))
            actions.append({"action": "reunwrap_uv"})

    if result.get("steps", {}).get("remesh", {}).get("quadriflow_failed"):
        issues.append(AuditIssue("topology", "minor", "QuadriFlow failed; fell back to decimated voxel mesh"))
    return issues, actions


class VisionAuditor:
    def __init__(self, vision: Optional[LLMClient], cfg: AuditConfig, contact_sheet: bool = True):
        self.vision = vision
        self.cfg = cfg
        self.contact_sheet = contact_sheet

    def _vision_review(self, renders: dict[str, str], spec: AssetSpec, result: dict[str, Any],
                       sheet: Optional[Path]) -> Optional[dict[str, Any]]:
        if not (self.cfg.use_vision_llm and self.vision and self.vision.enabled and renders):
            return None
        catalog = "\n".join(f'  "{name}": {desc}; params {hint}' for name, (desc, hint) in ACTIONS.items())
        brief = {
            "prompt": spec.prompt, "category": spec.category, "description": spec.description,
            "real_world_references": spec.real_world_references,
            "functional_mechanics": spec.functional_mechanics,
            "target_dimensions_m_xyz": spec.dimensions_m, "scale_reference": spec.scale_reference,
        }
        measured = {"metrics": result.get("metrics"), "uv_metrics": result.get("uv_metrics"),
                    "parts": result.get("parts"), "rig": result.get("rig")}
        user = (f"Design brief:\n{json.dumps(brief, indent=1)}\n\nMeasured statistics:\n"
                f"{json.dumps(measured, indent=1)}\n\nImages: "
                + ("one contact sheet with labelled views." if sheet else ", ".join(renders)))
        images = [sheet] if sheet else [Path(p) for p in renders.values()]
        try:
            data = self.vision.chat_json(AUDIT_SYSTEM_PROMPT + catalog, user, images=images)
            return data if isinstance(data, dict) else None
        except (LLMUnavailable, requests.RequestException, ValueError, KeyError) as exc:
            log.warning("Vision audit unavailable (%s); using geometric checks only", exc)
            return None

    def audit(self, iteration: int, result: dict[str, Any], spec: AssetSpec,
              params: dict[str, Any], render_dir: Path) -> AuditReport:
        issues, actions = geometric_review(result, spec, params, self.cfg)
        renders = result.get("renders") or {}
        sheet = None
        if renders:
            sheet = make_contact_sheet(renders, render_dir / "contact_sheet.png")
        vision = self._vision_review(renders, spec, result, sheet if self.contact_sheet else None)

        score: Optional[float] = None
        vision_ok = True
        if vision is not None:
            for raw in vision.get("issues") or []:
                if isinstance(raw, dict):
                    sev = str(raw.get("severity", "minor")).lower()
                    issues.append(AuditIssue(str(raw.get("category", "general")),
                                             sev if sev in SEVERITIES else "minor",
                                             str(raw.get("description", "")), source="vision"))
            for act in vision.get("actions") or []:
                if isinstance(act, dict) and act.get("action") in ACTIONS:
                    actions.append(act)
            try:
                score = float(vision.get("score"))
            except (TypeError, ValueError):
                score = None
            vision_ok = bool(vision.get("approved")) and (score is None or score >= self.cfg.min_vision_score)

        blocking = any(i.severity in ("critical", "major") for i in issues)
        return AuditReport(
            iteration=iteration,
            approved=vision_ok and not blocking,
            score=score,
            issues=issues,
            actions=_dedupe(actions),
            vision_used=vision is not None,
            vision_raw=vision,
            contact_sheet=str(sheet) if sheet else None,
        )


def _dedupe(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per action name (later parameters win), except stackable cuts."""
    out: dict[str, dict[str, Any]] = {}
    cuts = []
    for act in actions:
        if act["action"] == "boolean_difference":
            cuts.append(act)
        else:
            out[act["action"]] = {**out.get(act["action"], {}), **act}
    return list(out.values()) + cuts


def apply_corrective_actions(params: dict[str, Any], actions: list[dict[str, Any]]
                             ) -> tuple[dict[str, Any], list[str]]:
    """Map audit actions onto Blender job parameters for the next iteration.

    Returns the new parameters and a human-readable list of what changed; an
    empty list means nothing could be adjusted (the loop should stop).
    """
    p = copy.deepcopy(params)
    applied: list[str] = []

    def change(key: str, value: Any, why: str) -> None:
        if p.get(key) != value:
            applied.append(f"{why}: {key} {p.get(key)!r} -> {value!r}")
            p[key] = value

    for act in actions:
        name = act.get("action")
        if name == "remesh_finer":
            change("voxel_resolution", min(int(p.get("voxel_resolution", 160) * 1.5), 512), name)
            change("target_faces", min(int(p.get("target_faces", 12000) * 1.6), 200_000), name)
        elif name == "remesh_coarser":
            change("target_faces", max(int(p.get("target_faces", 12000) * 0.6), 500), name)
        elif name == "make_watertight":
            if p.get("remesh_mode") in ("none", "decimate"):
                change("remesh_mode", "voxel", name)
            change("fill_holes", True, name)
            change("voxel_resolution", min(int(p.get("voxel_resolution", 160) * 1.25), 512), name)
        elif name == "smooth_surface":
            iters = int(act.get("iterations", 4))
            change("smooth_iterations", min(int(p.get("smooth_iterations", 2)) + iters, 30), name)
        elif name == "preserve_detail":
            change("smooth_iterations", max(int(p.get("smooth_iterations", 2)) - 2, 0), name)
            change("shrinkwrap", True, name)
            change("voxel_resolution", min(int(p.get("voxel_resolution", 160) * 1.25), 512), name)
        elif name == "rotate":
            rot = [float(act.get(k, 0.0)) for k in ("x_deg", "y_deg", "z_deg")]
            if any(rot):
                current = p.get("extra_rotation_deg") or [0.0, 0.0, 0.0]
                change("extra_rotation_deg", [(c + r) % 360 for c, r in zip(current, rot)], name)
                change("auto_orient", False, name)  # don't let auto-orient undo it
        elif name == "rescale":
            axis = str(act.get("axis", p.get("primary_axis", "z"))).lower()
            if axis in "xyz" and act.get("target_m"):
                dims = list(p["target_dimensions_m"])
                dims["xyz".index(axis)] = float(act["target_m"])
                change("target_dimensions_m", dims, name)
                change("primary_axis", axis, name)
        elif name == "compensate_scale":  # internal: emitted by geometric_review
            factor = float(p.get("fit_compensation", 1.0)) * float(act.get("factor", 1.0))
            change("fit_compensation", round(factor, 6), name)
        elif name == "fit_proportions":
            change("nonuniform_fit", True, name)
        elif name == "reunwrap_uv":
            method = act.get("method")
            if method not in ("smart", "seams", "lightmap"):
                # escalate: smart (66deg) -> smart (45deg) -> seams -> lightmap
                cur, angle = p.get("uv_method", "smart"), float(p.get("uv_angle_limit_deg", 66))
                if cur == "smart" and angle > 45:
                    method = "smart"
                    act = {**act, "angle_limit_deg": 45}
                else:
                    method = {"smart": "seams", "seams": "lightmap"}.get(cur, "lightmap")
            change("uv_method", method, name)
            if act.get("angle_limit_deg"):
                change("uv_angle_limit_deg", float(act["angle_limit_deg"]), name)
        elif name == "increase_texture_resolution":
            change("texture_size", min(int(p.get("texture_size", 2048)) * 2, 4096), name)
        elif name == "rebake_textures":
            scale = float(act.get("cage_extrusion_scale", 2.0))
            change("cage_extrusion_rel", min(float(p.get("cage_extrusion_rel", 0.01)) * scale, 0.1), name)
            change("bake_samples", min(int(p.get("bake_samples", 16)) * 2, 256), name)
        elif name == "union_intersections":
            change("boolean_union", True, name)
            if p.get("remesh_mode") == "none":
                change("remesh_mode", "voxel", name)
        elif name == "boolean_difference":
            cut = {k: act[k] for k in ("shape", "center_norm", "size_norm") if k in act}
            if "center_norm" in cut and "size_norm" in cut:
                cutters = list(p.get("cutters") or [])
                if cut not in cutters:
                    cutters.append(cut)
                    change("cutters", cutters, name)
        else:
            log.warning("Ignoring unknown corrective action %r", name)
    return p, applied

"""Stage 6 - closed-loop audit.

Two sources of truth are combined:

* **Measured metrics** from Blender (boundary / non-manifold edges, floating
  islands, dimensions vs. spec, UV stretch, face budget). A vision model
  cannot reliably judge manifoldness from pixels, so these are authoritative.
* **Vision LLM review** of the audit renders (front, side, wireframe, shaded,
  UV checker) against the spec and the concept sheet, for things metrics
  cannot see: wrong proportions, missing features, blobby artefacts, texture
  seams and stretching.

Every issue carries a ``fix`` from a closed vocabulary (:data:`FIX_ACTIONS`).
:func:`apply_fixes` turns those into new Stage 3/4 parameters, which the
orchestrator feeds back into Blender for the next iteration.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .concept_planner import AssetSpec
from .llm_client import LLMClient, LLMError

log = logging.getLogger(__name__)

FIX_ACTIONS: Dict[str, str] = {
    "make_manifold": "holes / non-manifold geometry: force voxel-based remeshing and hole filling",
    "increase_detail": "shape too blobby or lost features: raise voxel resolution and face budget",
    "reduce_noise": "lumpy, noisy or stair-stepped surface: add smoothing iterations",
    "remove_floaters": "floating fragments or debris: raise the minimum island size",
    "rescale_axis": "wrong proportions along one axis: params {\"axis\": \"X|Y|Z\", \"factor\": 0.7-1.3}",
    "reorient": "object upside down / on its side: params {\"rotation_deg\": [x, y, z]}",
    "reduce_faces": "polycount above budget: lower the face target",
    "reunwrap": "UV stretching, distorted checker squares: re-unwrap with more islands",
    "rebake": "texture smearing, black patches or baking artefacts: widen the bake cage and margin",
}

SEVERITY_PENALTY = {"low": 0.03, "medium": 0.12, "high": 0.3}


@dataclass
class Issue:
    category: str
    severity: str
    description: str
    fix: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = "metrics"  # metrics | vision


@dataclass
class AuditReport:
    iteration: int
    approved: bool
    score: float
    issues: List[Issue] = field(default_factory=list)
    vision: Optional[Dict[str, Any]] = None
    vision_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def blocking(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "high"]


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

def check_metrics(metrics: Dict[str, Dict[str, Any]], spec: AssetSpec, face_budget: int,
                  assembled_dims: Optional[Sequence[float]] = None,
                  stretch_limit: float = 0.5) -> List[Issue]:
    """Turn Blender's per-object metrics into issues.

    ``metrics`` maps object name -> mesh_metrics(); the entry named after the
    asset (else the first) is the main body. ``assembled_dims`` is the extent
    of body + parts together, which is what the spec dimensions describe.
    """
    issues: List[Issue] = []
    if not metrics:
        return [Issue("geometry", "high", "no mesh metrics were produced", "make_manifold")]
    body_name = spec.name if spec.name in metrics else next(iter(metrics))
    total_faces = sum(m.get("faces", 0) for m in metrics.values())

    for name, m in metrics.items():
        holes = m.get("boundary_edges", 0) + m.get("non_manifold_edges", 0)
        if holes:
            issues.append(Issue(
                "geometry", "high",
                f"{name}: not watertight ({m.get('boundary_edges', 0)} boundary, "
                f"{m.get('non_manifold_edges', 0)} non-manifold edges)",
                "make_manifold"))
        if m.get("stretch") is not None and m["stretch"] > stretch_limit:
            issues.append(Issue(
                "texture", "high" if m["stretch"] > 2 * stretch_limit else "medium",
                f"{name}: UV area distortion {m['stretch']:.2f} (limit {stretch_limit})", "reunwrap"))

    body = metrics[body_name]
    if body.get("small_islands", 0):
        issues.append(Issue("artifact", "medium", f"{body['small_islands']} small floating islands on the body",
                            "remove_floaters"))

    if total_faces > face_budget * 1.5:
        issues.append(Issue("topology", "low", f"{total_faces} faces exceeds budget {face_budget}", "reduce_faces"))

    dims = assembled_dims or body.get("dimensions_m")
    if dims:
        for axis, measured, target in zip("XYZ", dims, spec.dimensions):
            if target <= 0:
                continue
            err = measured / target - 1.0
            if abs(err) > 0.25:
                issues.append(Issue(
                    "proportion", "medium",
                    f"{axis} extent {measured:.3f} m vs expected {target:.3f} m ({err:+.0%})",
                    "rescale_axis", {"axis": axis, "factor": round(1.0 / (1.0 + err), 3)}))
    return issues


# ---------------------------------------------------------------------------
# Vision review
# ---------------------------------------------------------------------------

AUDIT_SYSTEM = """You are a lead technical artist doing final QA on a game-ready 3D asset.
You are strict but fair, and you only report problems you can actually see.
Reply with ONLY a JSON object."""

AUDIT_PROMPT = """Review this 3D asset.

Asset: {name} ({category})
Description: {description}
Expected real-world size: length(X) {length:.3f} m, width(Y) {width:.3f} m, height(Z) {height:.3f} m
Scale notes: {scale_notes}
Measured by Blender: {measured}

Images, in order:
{image_list}

Answer these questions:
1. Is the mesh closed and clean? Look for holes, torn faces, spikes, floating debris, lumpy noise.
   (Measured watertight flag: {watertight}.)
2. Are the proportions physically accurate for the real object? Compare the front/side silhouettes
   with what the real object looks like{reference_hint}.
3. Are textures mapped correctly - no stretching (distorted checker squares), smearing, black patches
   or visible seams on the shaded view?

Return JSON:
{{
  "mesh": {{"ok": bool, "notes": str}},
  "proportions": {{"ok": bool, "notes": str}},
  "texture": {{"ok": bool, "notes": str}},
  "issues": [{{"category": "geometry"|"proportion"|"texture"|"topology"|"artifact",
               "severity": "low"|"medium"|"high",
               "description": str,
               "fix": one of {fixes},
               "params": object}}],
  "score": number 0-1 (1 = ship it),
  "approved": bool
}}

Fix actions:
{fix_help}
Only use "rescale_axis" for clear, large proportion errors and keep factor between 0.7 and 1.3.
"""

PART_PROMPT = """This is an orthographic {view} view of a 3D model of {name}.
Image coordinates: x from 0 (left) to 1 (right), y from 0 (top) to 1 (bottom).

Locate these mechanical parts and return a tight bounding box for each one you can see:
{parts}

Return JSON: {{"parts": [{{"name": str, "visible": bool, "box": [x_min, y_min, x_max, y_max]}}]}}
"""


class VisionAuditor:
    def __init__(self, llm: Optional[LLMClient], approval_score: float = 0.75):
        self.llm = llm
        self.approval_score = approval_score

    def audit(
        self,
        iteration: int,
        renders: Dict[str, str],
        metrics: Dict[str, Dict[str, Any]],
        spec: AssetSpec,
        face_budget: int,
        reference_image: Optional[str] = None,
        assembled_dims: Optional[Sequence[float]] = None,
    ) -> AuditReport:
        issues = check_metrics(metrics, spec, face_budget, assembled_dims)
        vision, vision_error = None, None
        if self.llm is not None:
            try:
                vision = self._vision_review(renders, metrics, spec, reference_image)
                issues.extend(self._vision_issues(vision))
            except (LLMError, ValueError, FileNotFoundError, KeyError, TypeError) as exc:
                vision_error = str(exc)
                log.warning("Vision review failed (%s); auditing on metrics only", exc)

        metric_score = max(0.0, 1.0 - sum(SEVERITY_PENALTY.get(i.severity, 0.1) for i in issues))
        score = metric_score
        if vision and isinstance(vision.get("score"), (int, float)):
            score = min(metric_score, float(vision["score"]))
        approved = (
            not any(i.severity == "high" for i in issues)
            and score >= self.approval_score
            and (vision is None or bool(vision.get("approved", True)))
        )
        report = AuditReport(iteration, approved, round(score, 3), issues, vision, vision_error)
        log.info("Audit #%d: score %.2f, %s, %d issues", iteration, report.score,
                 "APPROVED" if approved else "rejected", len(issues))
        for issue in issues:
            log.info("  [%s/%s] %s -> %s", issue.severity, issue.source, issue.description, issue.fix)
        return report

    def _vision_review(self, renders, metrics, spec, reference_image) -> Dict[str, Any]:
        order = [v for v in ("front", "side", "wireframe", "shaded", "uv_checker") if v in renders]
        images = [renders[v] for v in order]
        captions = {
            "front": "orthographic front view (looking along +Y, X horizontal, Z up)",
            "side": "orthographic side view (looking along -X, Y horizontal, Z up)",
            "wireframe": "perspective view, clay material with wireframe overlay (topology)",
            "shaded": "perspective view with final PBR materials",
            "uv_checker": "perspective view with a UV checker texture (squares should be even and square)",
        }
        lines = [f"{i + 1}. {captions[v]}" for i, v in enumerate(order)]
        if reference_image and Path(reference_image).is_file():
            images.append(reference_image)
            lines.append(f"{len(images)}. concept reference sheet the model was built from")
        body = metrics.get(spec.name) or next(iter(metrics.values()))
        measured = {k: body.get(k) for k in ("dimensions_m", "faces", "quad_ratio", "watertight", "stretch")}
        prompt = AUDIT_PROMPT.format(
            name=spec.name, category=spec.category, description=spec.description or spec.prompt,
            length=spec.length_m, width=spec.width_m, height=spec.height_m, scale_notes=spec.scale_notes or "-",
            measured=json.dumps(measured), image_list="\n".join(lines), watertight=body.get("watertight"),
            reference_hint=" and with the concept reference sheet" if reference_image else "",
            fixes=list(FIX_ACTIONS), fix_help="\n".join(f"- {k}: {v}" for k, v in FIX_ACTIONS.items()),
        )
        result = self.llm.chat_json(prompt, system=AUDIT_SYSTEM, images=images)
        if not isinstance(result, dict):
            raise ValueError("vision reply is not an object")
        return result

    @staticmethod
    def _vision_issues(vision: Dict[str, Any]) -> List[Issue]:
        issues = []
        for raw in vision.get("issues") or []:
            if not isinstance(raw, dict):
                continue
            fix = raw.get("fix") if raw.get("fix") in FIX_ACTIONS else None
            severity = raw.get("severity") if raw.get("severity") in SEVERITY_PENALTY else "medium"
            params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
            issues.append(Issue(str(raw.get("category", "other")), severity,
                                str(raw.get("description", "")), fix, params, source="vision"))
        return issues

    # ------------------------------------------------------------------ parts

    def locate_parts(self, render: str, framing: Dict[str, Any], bbox_min: Sequence[float],
                     bbox_max: Sequence[float], spec: AssetSpec) -> List[Dict[str, Any]]:
        """Refine part regions by asking the VLM where each part is in the front view.

        Returns spec parts (as dicts) with X/Z region bounds replaced by the
        located boxes; Y bounds and parts the model could not see keep the
        planner's estimate.
        """
        parts = [dataclasses.asdict(p) for p in spec.parts]
        if self.llm is None or not parts:
            return parts
        listing = "\n".join(f"- {p['name']}: {p['description']}" for p in parts)
        try:
            reply = self.llm.chat_json(PART_PROMPT.format(view="front", name=spec.prompt, parts=listing),
                                       images=[render])
        except (LLMError, ValueError, FileNotFoundError) as exc:
            log.warning("Part localisation failed (%s); using planner regions", exc)
            return parts
        located = {str(p.get("name")): p for p in (reply.get("parts") or []) if isinstance(p, dict)}
        for part in parts:
            hit = located.get(part["name"])
            if not hit or not hit.get("visible", True):
                continue
            try:
                region = image_box_to_region(hit["box"], framing, bbox_min, bbox_max)
            except (KeyError, TypeError, ValueError):
                continue
            if region is None:
                continue
            (x0, z0), (x1, z1) = region
            old_lo, old_hi = list(part["region_min"]), list(part["region_max"])
            new_lo = [x0, old_lo[1], z0]
            new_hi = [x1, old_hi[1], z1]
            part["pivot"] = remap_point(part["pivot"], old_lo, old_hi, new_lo, new_hi)
            part["region_min"], part["region_max"] = new_lo, new_hi
            log.info("Located %s at X %.2f-%.2f, Z %.2f-%.2f", part["name"], x0, x1, z0, z1)
        return parts


def image_box_to_region(box: Sequence[float], framing: Dict[str, Any], bbox_min: Sequence[float],
                        bbox_max: Sequence[float]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Map a normalised image box from the front (XZ) ortho render to bbox-normalised X/Z ranges."""
    if framing.get("plane") != "XZ":
        raise ValueError("only the front (XZ) view is supported")
    u0, v0, u1, v1 = (float(b) for b in box)
    if max(u0, v0, u1, v1) > 1.5:  # some models answer in pixels
        res = float(framing.get("resolution", 1024))
        u0, v0, u1, v1 = (b / res for b in (u0, v0, u1, v1))
    u0, u1 = sorted((u0, u1))
    v0, v1 = sorted((v0, v1))
    cx, _, cz = framing["center"]
    s = float(framing["ortho_scale"])
    x0, x1 = cx + (u0 - 0.5) * s, cx + (u1 - 0.5) * s
    z1, z0 = cz + (0.5 - v0) * s, cz + (0.5 - v1) * s  # image y grows downward
    sx = float(bbox_max[0] - bbox_min[0]) or 1.0
    sz = float(bbox_max[2] - bbox_min[2]) or 1.0
    nx0, nx1 = (x0 - bbox_min[0]) / sx, (x1 - bbox_min[0]) / sx
    nz0, nz1 = (z0 - bbox_min[2]) / sz, (z1 - bbox_min[2]) / sz
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    nx0, nx1, nz0, nz1 = clamp(nx0), clamp(nx1), clamp(nz0), clamp(nz1)
    if nx1 - nx0 < 0.005 or nz1 - nz0 < 0.005:
        return None
    return (nx0, nz0), (nx1, nz1)


def remap_point(point, old_lo, old_hi, new_lo, new_hi) -> List[float]:
    out = []
    for p, a, b, c, d in zip(point, old_lo, old_hi, new_lo, new_hi):
        t = (p - a) / (b - a) if b - a > 1e-9 else 0.5
        out.append(round(c + t * (d - c), 4))
    return out


# ---------------------------------------------------------------------------
# Corrective feedback
# ---------------------------------------------------------------------------

def apply_fixes(cleanup: Dict[str, Any], texture: Dict[str, Any],
                issues: Sequence[Issue]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Map audit issues onto new Stage 3 / Stage 4 parameters.

    Returns (cleanup, texture, changes). An empty ``changes`` list means no
    actionable correction remains and the loop should stop.
    """
    c, t = copy.deepcopy(cleanup), copy.deepcopy(texture)
    changes: List[str] = []
    done = set()

    def change(key: str, what: str):
        changes.append(what)
        done.add(key)

    for issue in sorted(issues, key=lambda i: -SEVERITY_PENALTY.get(i.severity, 0.1)):
        fix = issue.fix
        if fix is None:
            continue
        if fix == "rescale_axis":
            axis = str(issue.params.get("axis", "")).upper()
            try:
                factor = float(issue.params.get("factor", 1.0))
            except (TypeError, ValueError):
                continue
            key = f"rescale_{axis}"
            if axis not in ("X", "Y", "Z") or key in done or abs(factor - 1.0) < 0.02:
                continue
            factor = min(max(factor, 0.7), 1.3)
            axis_scale = dict(c.get("axis_scale") or {})
            axis_scale[axis] = round(float(axis_scale.get(axis, 1.0)) * factor, 4)
            if not 0.7 <= axis_scale[axis] <= 1.3:
                continue
            c["axis_scale"] = axis_scale
            change(key, f"scale {axis} by {factor:.3f}")
            continue
        if fix in done:
            continue
        if fix == "make_manifold":
            if c["remesh_mode"] in ("decimate", "none"):
                c["remesh_mode"] = "voxel"
                change(fix, "remesh_mode -> voxel (watertight)")
            elif c["voxel_resolution"] < 512:
                c["voxel_resolution"] = min(512, int(c["voxel_resolution"] * 1.25))
                c["fill_holes"] = True
                change(fix, f"voxel_resolution -> {c['voxel_resolution']}")
        elif fix == "increase_detail":
            if c["voxel_resolution"] < 512 or c["target_faces"] < 200000:
                c["voxel_resolution"] = min(512, int(c["voxel_resolution"] * 1.3))
                c["target_faces"] = min(200000, int(c["target_faces"] * 1.5))
                c["smooth_iterations"] = max(0, int(c["smooth_iterations"]) - 1)
                change(fix, f"voxel_resolution -> {c['voxel_resolution']}, target_faces -> {c['target_faces']}")
        elif fix == "reduce_noise":
            if c["smooth_iterations"] < 8:
                c["smooth_iterations"] = int(c["smooth_iterations"]) + 2
                change(fix, f"smooth_iterations -> {c['smooth_iterations']}")
        elif fix == "remove_floaters":
            if c["min_island_ratio"] < 0.2:
                c["min_island_ratio"] = min(0.2, max(0.05, c["min_island_ratio"] * 2.5))
                change(fix, f"min_island_ratio -> {c['min_island_ratio']:.3f}")
        elif fix == "reorient":
            rot = issue.params.get("rotation_deg")
            if isinstance(rot, (list, tuple)) and len(rot) == 3:
                base = list(c.get("import_rotation_deg") or [0, 0, 0])
                c["import_rotation_deg"] = [(float(a) + float(b)) % 360 for a, b in zip(base, rot)]
                change(fix, f"import_rotation_deg -> {c['import_rotation_deg']}")
        elif fix == "reduce_faces":
            if c["target_faces"] > 500:
                c["target_faces"] = max(500, int(c["target_faces"] * 0.7))
                change(fix, f"target_faces -> {c['target_faces']}")
        elif fix == "reunwrap":
            if t["uv_angle_limit"] > 30:
                t["uv_angle_limit"] = max(30.0, round(t["uv_angle_limit"] * 0.75, 1))
                t["island_margin"] = min(0.03, t["island_margin"] * 1.5)
                t["uv_method"] = "smart"
                change(fix, f"uv_angle_limit -> {t['uv_angle_limit']}")
            elif t["uv_method"] != "lightmap":
                t["uv_method"] = "lightmap"
                change(fix, "uv_method -> lightmap")
        elif fix == "rebake":
            if t["cage_extrusion"] < 0.08:
                t["cage_extrusion"] = min(0.08, t["cage_extrusion"] * 1.6)
                t["bake_margin_px"] = min(64, int(t["bake_margin_px"] * 2))
                change(fix, f"cage_extrusion -> {t['cage_extrusion']:.3f}, margin -> {t['bake_margin_px']}px")
    return c, t, changes

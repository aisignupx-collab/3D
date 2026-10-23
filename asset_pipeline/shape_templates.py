"""Primitive shape templates (Stage 3b).

Feed-forward reconstructions are lumpy: a crate comes out as a wonky blob.
Hard-surface assets are better *built* from clean primitives and textured from
the reconstruction. This module holds:

* the primitive vocabulary (:data:`PRIMITIVES`),
* :class:`ShapeComponent` - one primitive placed in the asset's normalised
  bounding box (``[0,0,0]`` = min corner, ``[1,1,1]`` = max corner),
* a recipe library for common assets (chest, barrel, castle tower, ...),
* validation for recipes proposed by the planner LLM.

Blender builds and fits the components (``op_primitives`` in
``blender_automation.py``); when no recipe exists it tries each primitive as a
single-shape fit and keeps the best one if it matches the scan closely.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Canonical primitives; "axis" is the main axis (cylinder/cone axis, roof ridge,
# torus normal). Signed axes ("-Z") flip cones/pyramids/domes.
PRIMITIVES: Dict[str, str] = {
    "box": "rectangular block (crates, walls, slabs, legs, planks)",
    "cylinder": "round column along axis (towers, barrels, shafts, wheels, pipes)",
    "cone": "cone/frustum along axis, apex towards +axis; taper = top radius ratio (roofs, spear tips)",
    "sphere": "ball / ellipsoid (pommels, domes when clipped, knobs)",
    "hemisphere": "half ellipsoid, dome towards +axis (domes, round lids)",
    "half_cylinder": "half cylinder along axis, round side up +Z (arched lids, arches, vaults)",
    "wedge": "triangular prism, ridge along axis, apex up +Z (gable roofs, ramps, blades)",
    "pyramid": "square pyramid/frustum, apex towards +axis; taper = top size ratio (spires, tips)",
    "torus": "ring around axis; thickness = tube/outer radius (hoops, bands, rims, tyres)",
}
AXES = ("X", "Y", "Z", "+X", "+Y", "+Z", "-X", "-Y", "-Z")
ORGANIC = ("character", "creature")

# PBR defaults for material names used by recipes (linear base colour, metallic, roughness)
DEFAULT_MATERIALS: Dict[str, Dict[str, Any]] = {
    "wood": {"base_color": [0.40, 0.24, 0.11], "metallic": 0.0, "roughness": 0.7},
    "iron": {"base_color": [0.16, 0.16, 0.17], "metallic": 1.0, "roughness": 0.55},
    "steel": {"base_color": [0.62, 0.62, 0.64], "metallic": 1.0, "roughness": 0.3},
    "stone": {"base_color": [0.42, 0.40, 0.37], "metallic": 0.0, "roughness": 0.9},
    "roof": {"base_color": [0.35, 0.12, 0.08], "metallic": 0.0, "roughness": 0.8},
    "wall": {"base_color": [0.75, 0.70, 0.62], "metallic": 0.0, "roughness": 0.85},
    "leather": {"base_color": [0.20, 0.10, 0.05], "metallic": 0.0, "roughness": 0.75},
    "paint": {"base_color": [0.55, 0.05, 0.05], "metallic": 0.5, "roughness": 0.3},
    "glass": {"base_color": [0.10, 0.12, 0.14], "metallic": 0.0, "roughness": 0.05},
    "rubber": {"base_color": [0.02, 0.02, 0.02], "metallic": 0.0, "roughness": 0.9},
}


@dataclass
class ShapeComponent:
    name: str
    type: str = "box"
    min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    axis: str = "Z"
    taper: float = 1.0  # cone/pyramid top size ratio (0 = sharp apex); 1 for other types
    bevel: float = 0.0  # fraction of the smallest dimension (box/wedge/pyramid)
    thickness: float = 0.3  # torus tube radius / outer radius (0.05 = thin hoop hugging a cylinder)
    segments: int = 24
    operation: str = "union"  # union | subtract
    part: str = ""  # spec part this component belongs to (built as a separate object)
    material: str = ""  # spec material name -> drives metallic/roughness
    array_count: int = 1  # repeat evenly along array_axis with equal gaps (merlons, planks)
    array_axis: str = "X"
    rotation_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    refine: bool = True  # snap extents to the scan's points inside the region

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def C(name, type_, lo, hi, **kw) -> ShapeComponent:  # noqa: N802 - compact recipe syntax
    if type_ in ("cone", "pyramid"):
        kw.setdefault("taper", 0.0)
    return ShapeComponent(name, type_, list(lo), list(hi), **kw)


# ---------------------------------------------------------------------------
# Recipe library. Keywords are matched like the concept knowledge base: the
# longest matching keyword wins ("castle gate" beats "gate").
# ---------------------------------------------------------------------------

RECIPES: List[Dict[str, Any]] = [
    {"keywords": ["crate", "box", "cube", "block"], "shapes": [
        C("body", "box", (0, 0, 0), (1, 1, 1), bevel=0.03, material="wood"),
    ]},
    {"keywords": ["chest", "treasure chest", "trunk", "coffer"], "shapes": [
        C("body", "box", (0, 0, 0), (1, 1, 0.62), bevel=0.02, material="wood"),
        C("lid", "box", (0, 0, 0.62), (1, 1, 1), bevel=0.03, part="lid", material="wood"),
        C("band", "box", (0.12, -0.012, -0.005), (0.18, 1.012, 0.625), refine=False, material="iron"),
        C("band", "box", (0.82, -0.012, -0.005), (0.88, 1.012, 0.625), refine=False, material="iron"),
        C("lid_band", "box", (0.12, -0.012, 0.62), (0.18, 1.012, 1.008), part="lid", refine=False,
          material="iron"),
        C("lid_band", "box", (0.82, -0.012, 0.62), (0.88, 1.012, 1.008), part="lid", refine=False,
          material="iron"),
        C("lock", "box", (0.44, -0.03, 0.5), (0.56, 0.0, 0.7), refine=False, material="iron"),
    ]},
    {"keywords": ["barrel", "keg", "cask"], "shapes": [
        C("staves", "cylinder", (0, 0, 0), (1, 1, 1), axis="Z", segments=24, material="wood"),
        C("hoop", "torus", (-0.015, -0.015, 0.12), (1.015, 1.015, 0.18), axis="Z", thickness=0.06, refine=False,
          material="iron"),
        C("hoop", "torus", (-0.015, -0.015, 0.82), (1.015, 1.015, 0.88), axis="Z", thickness=0.06, refine=False,
          material="iron"),
    ]},
    {"keywords": ["table", "desk"], "shapes": [
        C("top", "box", (0, 0, 0.94), (1, 1, 1), bevel=0.2, material="wood"),
        C("leg", "box", (0.03, 0.05, 0), (0.08, 0.15, 0.94), material="wood"),
        C("leg", "box", (0.92, 0.05, 0), (0.97, 0.15, 0.94), material="wood"),
        C("leg", "box", (0.03, 0.85, 0), (0.08, 0.95, 0.94), material="wood"),
        C("leg", "box", (0.92, 0.85, 0), (0.97, 0.95, 0.94), material="wood"),
    ]},
    {"keywords": ["chair", "stool"], "shapes": [
        C("seat", "box", (0, 0, 0.47), (1, 1, 0.53), bevel=0.2, material="wood"),
        C("leg", "box", (0.04, 0.04, 0), (0.12, 0.12, 0.47), material="wood"),
        C("leg", "box", (0.88, 0.04, 0), (0.96, 0.12, 0.47), material="wood"),
        C("leg", "box", (0.04, 0.88, 0), (0.12, 0.96, 1.0), material="wood"),
        C("leg", "box", (0.88, 0.88, 0), (0.96, 0.96, 1.0), material="wood"),
        C("back", "box", (0.04, 0.9, 0.62), (0.96, 0.96, 0.95), material="wood"),
    ]},
    {"keywords": ["door"], "shapes": [
        C("jamb", "box", (0.0, 0.0, 0.0), (0.05, 1.0, 1.0), material="wood"),
        C("jamb", "box", (0.95, 0.0, 0.0), (1.0, 1.0, 1.0), material="wood"),
        C("header", "box", (0.0, 0.0, 0.97), (1.0, 1.0, 1.0), material="wood"),
        C("leaf", "box", (0.05, 0.2, 0.0), (0.95, 0.8, 0.97), bevel=0.15, part="door_leaf", material="wood"),
        C("handle", "sphere", (0.84, -0.4, 0.47), (0.9, 0.2, 0.5), refine=False, part="door_leaf",
          material="iron"),
    ]},
    {"keywords": ["castle wall", "rampart", "battlement", "fortification", "stone wall", "wall"], "shapes": [
        C("wall", "box", (0, 0, 0), (1, 1, 0.85), material="stone"),
        C("walkway", "box", (0, 0, 0.85), (1, 0.35, 0.87), refine=False, material="stone"),
        # Merlons: 7 teeth with equal crenel gaps on the outer face
        C("merlon", "box", (0, 0.65, 0.85), (1, 1, 1), array_count=9, array_axis="X", refine=False,
          material="stone"),
    ]},
    {"keywords": ["tower", "turret", "castle tower", "keep"], "shapes": [
        C("tower", "cylinder", (0.06, 0.06, 0), (0.94, 0.94, 0.72), axis="Z", segments=32, material="stone"),
        C("corbel", "cylinder", (0.02, 0.02, 0.68), (0.98, 0.98, 0.74), axis="Z", segments=32, refine=False,
          material="stone"),
        C("roof", "cone", (0, 0, 0.74), (1, 1, 1), axis="+Z", segments=32, material="roof"),
        C("door", "half_cylinder", (0.4, -0.05, 0.0), (0.6, 0.1, 0.16), axis="Y", operation="subtract",
          refine=False),
    ]},
    {"keywords": ["castle gate", "gatehouse", "portcullis", "gate"], "shapes": [
        C("tower_left", "cylinder", (0, 0.1, 0), (0.28, 0.9, 0.85), axis="Z", segments=32, material="stone"),
        C("tower_right", "cylinder", (0.72, 0.1, 0), (1, 0.9, 0.85), axis="Z", segments=32, material="stone"),
        C("roof_left", "cone", (0, 0.1, 0.85), (0.28, 0.9, 1.0), axis="+Z", segments=32, refine=False,
          material="roof"),
        C("roof_right", "cone", (0.72, 0.1, 0.85), (1, 0.9, 1.0), axis="+Z", segments=32, refine=False,
          material="roof"),
        C("gatehouse", "box", (0.2, 0.0, 0), (0.8, 1, 0.75), material="stone"),
        C("passage", "box", (0.38, -0.05, 0), (0.62, 1.05, 0.38), operation="subtract", refine=False),
        C("arch", "half_cylinder", (0.38, -0.05, 0.3), (0.62, 1.05, 0.46), axis="Y", operation="subtract",
          refine=False),
        C("merlon", "box", (0.2, 0.0, 0.75), (0.8, 0.2, 0.83), array_count=5, array_axis="X", refine=False,
          material="stone"),
    ]},
    {"keywords": ["house", "hut", "cottage", "cabin", "shed"], "shapes": [
        C("walls", "box", (0.02, 0.02, 0), (0.98, 0.98, 0.6), material="wall"),
        C("roof", "wedge", (0, 0, 0.6), (1, 1, 1), axis="X", material="roof"),
        C("door", "box", (0.42, -0.02, 0), (0.58, 0.05, 0.38), operation="subtract", refine=False),
    ]},
    {"keywords": ["sword", "longsword", "katana", "blade"], "shapes": [
        # Lies along X; the blade is flat in X-Z, the crossguard spans the full height (Z)
        C("blade", "box", (0.25, 0.4, 0.38), (0.92, 0.6, 0.62), bevel=0.4, material="steel"),
        C("tip", "wedge", (0.92, 0.4, 0.38), (1.0, 0.6, 0.62), axis="Y", rotation_deg=[0, 90, 0],
          material="steel"),
        C("guard", "box", (0.22, 0.0, 0.0), (0.25, 1.0, 1.0), bevel=0.2, material="steel"),
        C("grip", "cylinder", (0.05, 0.3, 0.42), (0.22, 0.7, 0.58), axis="X", material="leather"),
        C("pommel", "sphere", (0.0, 0.2, 0.38), (0.06, 0.8, 0.62), material="steel"),
    ]},
    {"keywords": ["spear", "lance", "pike", "javelin"], "shapes": [
        C("shaft", "cylinder", (0.0, 0.3, 0.3), (0.85, 0.7, 0.7), axis="X", material="wood"),
        C("tip", "cone", (0.85, 0.0, 0.0), (1.0, 1.0, 1.0), axis="+X", material="steel"),
    ]},
    {"keywords": ["hammer", "mallet", "warhammer"], "shapes": [
        C("handle", "cylinder", (0.0, 0.35, 0.35), (0.85, 0.65, 0.65), axis="X", material="wood"),
        C("head", "box", (0.78, 0.0, 0.0), (1.0, 1.0, 1.0), bevel=0.08, material="steel"),
    ]},
    {"keywords": ["car", "sedan", "truck", "jeep", "vehicle"], "shapes": [
        C("chassis", "box", (0, 0.02, 0.15), (1, 0.98, 0.55), bevel=0.25, material="paint"),
        C("cabin", "box", (0.25, 0.08, 0.55), (0.75, 0.92, 1.0), bevel=0.25, material="glass"),
        C("wheel", "cylinder", (0.72, 0.0, 0.0), (0.9, 0.15, 0.45), axis="Y", part="wheel_front_left",
          material="rubber"),
        C("wheel", "cylinder", (0.72, 0.85, 0.0), (0.9, 1.0, 0.45), axis="Y", part="wheel_front_right",
          material="rubber"),
        C("wheel", "cylinder", (0.1, 0.0, 0.0), (0.28, 0.15, 0.45), axis="Y", part="wheel_rear_left",
          material="rubber"),
        C("wheel", "cylinder", (0.1, 0.85, 0.0), (0.28, 1.0, 0.45), axis="Y", part="wheel_rear_right",
          material="rubber"),
    ]},
    {"keywords": ["pillar", "column", "post", "pole"], "shapes": [
        C("base", "box", (0, 0, 0), (1, 1, 0.08), bevel=0.1, material="stone"),
        C("shaft", "cylinder", (0.12, 0.12, 0.08), (0.88, 0.88, 0.92), axis="Z", material="stone"),
        C("capital", "box", (0, 0, 0.92), (1, 1, 1), bevel=0.1, material="stone"),
    ]},
]


def _match(prompt: str, keywords: List[str]) -> int:
    text = prompt.lower()
    return max((len(k) for k in keywords if re.search(rf"\b{re.escape(k)}\b", text)), default=0)


def recipe_for(prompt: str, category: str = "") -> List[ShapeComponent]:
    """Best-matching library recipe for a prompt (empty for organic assets / no match)."""
    if category in ORGANIC:
        return []
    best, best_len = None, 0
    for entry in RECIPES:
        n = _match(prompt, entry["keywords"])
        if n > best_len:
            best, best_len = entry, n
    return [dataclasses.replace(c) for c in best["shapes"]] if best else []


def _vec(value, default, lo=-1.0, hi=2.0) -> List[float]:
    try:
        v = [min(max(float(x), lo), hi) for x in value][:3]
    except (TypeError, ValueError):
        return list(default)
    return (v + list(default)[len(v):])[:3]


def validate_components(raw: Any, material_names: Optional[List[str]] = None,
                        part_names: Optional[List[str]] = None) -> List[ShapeComponent]:
    """Sanitise a recipe (e.g. from the planner LLM); drops anything unusable."""
    out: List[ShapeComponent] = []
    if not isinstance(raw, list):
        return out
    for i, item in enumerate(raw[:64]):
        if not isinstance(item, dict) or item.get("type") not in PRIMITIVES:
            continue
        lo = _vec(item.get("min"), [0.0, 0.0, 0.0])
        hi = _vec(item.get("max"), [1.0, 1.0, 1.0])
        lo, hi = [min(a, b) for a, b in zip(lo, hi)], [max(a, b) for a, b in zip(lo, hi)]
        if any(b - a < 1e-3 for a, b in zip(lo, hi)):
            continue
        axis = str(item.get("axis", "Z")).upper()
        material = str(item.get("material", ""))
        part = re.sub(r"[^a-z0-9]+", "_", str(item.get("part", "")).lower()).strip("_")
        try:
            comp = ShapeComponent(
                name=re.sub(r"[^a-z0-9]+", "_", str(item.get("name") or f"shape_{i}").lower()).strip("_") or
                f"shape_{i}",
                type=item["type"],
                min=lo,
                max=hi,
                axis=axis if axis in AXES else "Z",
                taper=min(max(float(item.get("taper", 0.0 if item["type"] in ("cone", "pyramid") else 1.0)),
                              0.0), 1.0),
                bevel=min(max(float(item.get("bevel", 0.0)), 0.0), 0.45),
                thickness=min(max(float(item.get("thickness", 0.3)), 0.02), 0.5),
                segments=min(max(int(item.get("segments", 24)), 4), 96),
                operation="subtract" if item.get("operation") == "subtract" else "union",
                part=part if not part_names or part in part_names else "",
                material=material if not material_names or material in material_names else "",
                array_count=min(max(int(item.get("array_count", 1)), 1), 32),
                array_axis=str(item.get("array_axis", "X")).upper()[:1] if str(
                    item.get("array_axis", "X")).upper()[:1] in "XYZ" else "X",
                rotation_deg=_vec(item.get("rotation_deg"), [0.0, 0.0, 0.0], -180.0, 180.0),
                refine=bool(item.get("refine", True)),
            )
        except (TypeError, ValueError):
            continue
        out.append(comp)
    if out and all(c.operation == "subtract" for c in out):
        return []
    return out


def expand_arrays(components: List[ShapeComponent]) -> List[ShapeComponent]:
    """Replace arrayed components by ``array_count`` evenly spaced copies with equal gaps."""
    out: List[ShapeComponent] = []
    for comp in components:
        n = comp.array_count
        if n <= 1:
            out.append(comp)
            continue
        axis = "XYZ".index(comp.array_axis)
        lo, hi = comp.min[axis], comp.max[axis]
        width = (hi - lo) / (2 * n - 1)  # n teeth, n-1 gaps of equal width
        for k in range(n):
            copy = dataclasses.replace(comp, array_count=1, name=f"{comp.name}_{k + 1}",
                                       min=list(comp.min), max=list(comp.max))
            copy.min[axis] = lo + 2 * k * width
            copy.max[axis] = copy.min[axis] + width
            out.append(copy)
    return out


RECIPE_PROMPT_HELP = "\n".join(f"  - {k}: {v}" for k, v in PRIMITIVES.items())

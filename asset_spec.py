"""AssetSpec: the structured brief every pipeline stage works from.

Axis convention (Blender, Z-up, metres):
    X = width (left/right), Y = length / depth (front/back, -Y is the front),
    Z = height.

``primary_axis`` names the dimension that is authoritative for scaling (an MP5
is scaled by its 0.68 m length on Y, a door by its height on Z). The other two
dimensions are used for the proportion check.

Moving-part regions and pivots are given in *normalized* bounding-box
coordinates (0..1 on each axis of the fitted asset), so the LLM can describe
them without knowing the final mesh.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

CATEGORIES = (
    "weapon", "vehicle", "prop", "architecture", "character", "creature",
    "furniture", "other",
)
RIG_TYPES = ("none", "humanoid", "quadruped", "chain")
AXES = ("x", "y", "z")


@dataclass
class MaterialSpec:
    name: str = "base"
    base_color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])  # linear RGB
    roughness: float = 0.5
    metallic: float = 0.0


@dataclass
class MovingPart:
    name: str
    region_min: list[float]  # normalized bbox coords
    region_max: list[float]
    pivot: list[float]  # normalized bbox coords
    motion: str = "rotate"  # rotate | translate | fixed
    axis: str = "X"
    limits: list[float] = field(default_factory=lambda: [0.0, 0.0])  # degrees or metres


@dataclass
class AssetSpec:
    name: str
    prompt: str
    category: str = "prop"
    description: str = ""
    real_world_references: list[str] = field(default_factory=list)
    functional_mechanics: list[str] = field(default_factory=list)
    dimensions_m: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    primary_axis: str = "z"
    scale_reference: str = ""
    origin: str = "bottom_center"  # bottom_center | center
    symmetry: Optional[str] = None  # "x" for bilateral symmetry, else None
    materials: list[MaterialSpec] = field(default_factory=lambda: [MaterialSpec()])
    moving_parts: list[MovingPart] = field(default_factory=list)
    rig_type: str = "none"
    negative_prompt: str = (
        "multiple objects, cropped, text, watermark, logo, busy background, "
        "shadow, floor, blurry, deformed"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], prompt: str = "") -> "AssetSpec":
        """Build a spec from (possibly sloppy) LLM output, coercing bad values."""
        def floats(value: Any, n: int, default: list[float]) -> list[float]:
            try:
                out = [float(v) for v in value][:n]
                return out if len(out) == n else default
            except (TypeError, ValueError):
                return default

        def clamp01(v: list[float]) -> list[float]:
            return [min(1.0, max(0.0, x)) for x in v]

        category = str(data.get("category", "prop")).lower()
        if category not in CATEGORIES:
            category = "other"
        primary = str(data.get("primary_axis", "z")).lower()
        if primary not in AXES:
            primary = "z"
        dims = [max(1e-3, d) for d in floats(data.get("dimensions_m"), 3, [1.0, 1.0, 1.0])]

        materials = []
        for m in data.get("materials") or []:
            if not isinstance(m, dict):
                continue
            materials.append(MaterialSpec(
                name=str(m.get("name", "material")),
                base_color=clamp01(floats(m.get("base_color"), 3, [0.5, 0.5, 0.5])),
                roughness=min(1.0, max(0.0, float(m.get("roughness", 0.5)))),
                metallic=min(1.0, max(0.0, float(m.get("metallic", 0.0)))),
            ))

        parts = []
        for p in data.get("moving_parts") or []:
            if not isinstance(p, dict) or "name" not in p:
                continue
            lo = clamp01(floats(p.get("region_min"), 3, [0.0, 0.0, 0.0]))
            hi = clamp01(floats(p.get("region_max"), 3, [1.0, 1.0, 1.0]))
            lo, hi = [min(a, b) for a, b in zip(lo, hi)], [max(a, b) for a, b in zip(lo, hi)]
            motion = str(p.get("motion", "rotate")).lower()
            parts.append(MovingPart(
                name=re.sub(r"[^A-Za-z0-9_]+", "_", str(p["name"])).strip("_") or "part",
                region_min=lo,
                region_max=hi,
                pivot=clamp01(floats(p.get("pivot"), 3, [(a + b) / 2 for a, b in zip(lo, hi)])),
                motion=motion if motion in ("rotate", "translate", "fixed") else "fixed",
                axis=str(p.get("axis", "X")).upper()[:1] if str(p.get("axis", "X")).upper()[:1] in "XYZ" else "X",
                limits=floats(p.get("limits"), 2, [0.0, 0.0]),
            ))

        rig = str(data.get("rig_type", "none")).lower()
        if rig not in RIG_TYPES:
            rig = "none"
        symmetry = data.get("symmetry")
        symmetry = str(symmetry).lower() if symmetry and str(symmetry).lower() in AXES else None

        spec = cls(
            name=slugify(str(data.get("name") or prompt or "asset")),
            prompt=prompt or str(data.get("prompt", "")),
            category=category,
            description=str(data.get("description") or prompt),
            real_world_references=[str(x) for x in data.get("real_world_references") or []],
            functional_mechanics=[str(x) for x in data.get("functional_mechanics") or []],
            dimensions_m=dims,
            primary_axis=primary,
            scale_reference=str(data.get("scale_reference", "")),
            origin="center" if data.get("origin") == "center" else "bottom_center",
            symmetry=symmetry,
            materials=materials or [MaterialSpec()],
            moving_parts=parts,
            rig_type=rig,
        )
        if data.get("negative_prompt"):
            spec.negative_prompt = str(data["negative_prompt"])
        return spec


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug[:max_len].rstrip("_") or "asset")


# ---------------------------------------------------------------------------
# Offline knowledge base: used when no LLM is reachable, and as a sanity check
# on LLM-proposed dimensions. Keyword -> partial spec.
# ---------------------------------------------------------------------------
_GUNMETAL = {"name": "gunmetal", "base_color": [0.03, 0.03, 0.035], "roughness": 0.45, "metallic": 0.9}
_POLYMER = {"name": "polymer", "base_color": [0.02, 0.02, 0.02], "roughness": 0.7, "metallic": 0.0}
_STONE = {"name": "stone", "base_color": [0.32, 0.30, 0.27], "roughness": 0.9, "metallic": 0.0}
_WOOD = {"name": "wood", "base_color": [0.25, 0.14, 0.07], "roughness": 0.75, "metallic": 0.0}
_IRON = {"name": "iron", "base_color": [0.12, 0.12, 0.12], "roughness": 0.6, "metallic": 0.85}
_PAINT = {"name": "paint", "base_color": [0.35, 0.02, 0.02], "roughness": 0.35, "metallic": 0.0}
_SKIN = {"name": "skin", "base_color": [0.6, 0.42, 0.33], "roughness": 0.55, "metallic": 0.0}
_CLOTH = {"name": "cloth", "base_color": [0.15, 0.17, 0.22], "roughness": 0.85, "metallic": 0.0}

KNOWLEDGE_BASE: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (("mp5",), {
        "category": "weapon", "dimensions_m": [0.06, 0.68, 0.26], "primary_axis": "y",
        "scale_reference": "H&K MP5A3: 680 mm overall (stock extended), 225 mm barrel, ~260 mm tall with magazine",
        "real_world_references": ["Heckler & Koch MP5A3", "roller-delayed blowback SMG"],
        "functional_mechanics": ["roller-delayed blowback bolt", "pivoting trigger", "detachable curved 30-round magazine", "retractable stock", "charging handle in forward tube"],
        "origin": "center", "symmetry": "x", "materials": [_GUNMETAL, _POLYMER],
        "moving_parts": [
            {"name": "magazine", "region_min": [0.0, 0.40, 0.0], "region_max": [1.0, 0.55, 0.45], "pivot": [0.5, 0.48, 0.45], "motion": "translate", "axis": "Z", "limits": [-0.12, 0.0]},
            {"name": "trigger", "region_min": [0.3, 0.55, 0.30], "region_max": [0.7, 0.62, 0.45], "pivot": [0.5, 0.58, 0.45], "motion": "rotate", "axis": "X", "limits": [0.0, 15.0]},
            {"name": "charging_handle", "region_min": [0.0, 0.15, 0.75], "region_max": [1.0, 0.25, 0.9], "pivot": [0.5, 0.2, 0.82], "motion": "translate", "axis": "Y", "limits": [0.0, 0.08]},
        ],
    }),
    (("rifle", "ak47", "ak-47", "m4", "carbine"), {
        "category": "weapon", "dimensions_m": [0.07, 0.88, 0.25], "primary_axis": "y",
        "scale_reference": "assault rifle ~850-1000 mm overall", "origin": "center", "symmetry": "x",
        "materials": [_GUNMETAL, _WOOD],
    }),
    (("pistol", "handgun", "glock"), {
        "category": "weapon", "dimensions_m": [0.03, 0.19, 0.14], "primary_axis": "y",
        "scale_reference": "service pistol ~190 mm long", "origin": "center", "symmetry": "x",
        "materials": [_GUNMETAL, _POLYMER],
    }),
    (("sword", "katana", "longsword"), {
        "category": "weapon", "dimensions_m": [0.2, 0.04, 1.05], "primary_axis": "z",
        "scale_reference": "longsword ~1.0-1.1 m overall", "origin": "center", "symmetry": "x",
        "materials": [{"name": "steel", "base_color": [0.6, 0.6, 0.62], "roughness": 0.25, "metallic": 1.0}, _WOOD],
    }),
    (("castle wall", "rampart", "battlement", "city wall"), {
        "category": "architecture", "dimensions_m": [10.0, 2.5, 8.0], "primary_axis": "z",
        "scale_reference": "curtain wall 6-12 m high, 2-3 m thick; crenels ~0.6 m for a 1.8 m defender",
        "functional_mechanics": ["wall walk behind crenellations", "merlons ~1.8 m tall for cover"],
        "materials": [_STONE],
    }),
    (("gate", "portcullis"), {
        "category": "architecture", "dimensions_m": [4.0, 0.6, 5.0], "primary_axis": "z",
        "scale_reference": "castle gate 3-4 m wide, 4-5 m high, passable by cart and rider",
        "materials": [_WOOD, _IRON],
        "moving_parts": [{"name": "gate_leaf", "region_min": [0.1, 0.0, 0.0], "region_max": [0.9, 1.0, 0.8], "pivot": [0.1, 0.5, 0.0], "motion": "rotate", "axis": "Z", "limits": [0.0, 95.0]}],
    }),
    (("door",), {
        "category": "architecture", "dimensions_m": [0.9, 0.05, 2.1], "primary_axis": "z",
        "scale_reference": "interior door 2.03-2.1 m x 0.8-0.9 m clears a 1.8 m character",
        "materials": [_WOOD, _IRON],
        "moving_parts": [{"name": "door_leaf", "region_min": [0.0, 0.0, 0.0], "region_max": [1.0, 1.0, 1.0], "pivot": [0.0, 0.5, 0.0], "motion": "rotate", "axis": "Z", "limits": [0.0, 100.0]}],
    }),
    (("chair",), {
        "category": "furniture", "dimensions_m": [0.48, 0.52, 0.9], "primary_axis": "z",
        "scale_reference": "dining chair: seat 0.45 m high, back 0.9 m", "symmetry": "x", "materials": [_WOOD],
    }),
    (("table", "desk"), {
        "category": "furniture", "dimensions_m": [1.6, 0.8, 0.75], "primary_axis": "z",
        "scale_reference": "table top 0.72-0.76 m high", "materials": [_WOOD],
    }),
    (("barrel",), {
        "category": "prop", "dimensions_m": [0.6, 0.6, 0.9], "primary_axis": "z",
        "scale_reference": "wine barrel ~0.9 m tall", "materials": [_WOOD, _IRON],
    }),
    (("crate", "box"), {
        "category": "prop", "dimensions_m": [0.6, 0.6, 0.6], "primary_axis": "z",
        "scale_reference": "shipping crate 0.6 m", "materials": [_WOOD],
    }),
    (("car", "sedan", "hatchback"), {
        "category": "vehicle", "dimensions_m": [1.8, 4.5, 1.45], "primary_axis": "y",
        "scale_reference": "mid-size sedan 4.5 m x 1.8 m x 1.45 m", "symmetry": "x",
        "materials": [_PAINT, {"name": "tyre", "base_color": [0.02, 0.02, 0.02], "roughness": 0.9, "metallic": 0.0}],
    }),
    (("truck", "lorry"), {
        "category": "vehicle", "dimensions_m": [2.5, 7.5, 3.2], "primary_axis": "y", "symmetry": "x",
        "materials": [_PAINT],
    }),
    (("tank",), {
        "category": "vehicle", "dimensions_m": [3.6, 9.8, 2.4], "primary_axis": "y", "symmetry": "x",
        "materials": [{"name": "armor", "base_color": [0.18, 0.2, 0.12], "roughness": 0.7, "metallic": 0.3}],
    }),
    (("dragon",), {
        "category": "creature", "dimensions_m": [6.0, 12.0, 5.0], "primary_axis": "y", "symmetry": "x",
        "rig_type": "quadruped",
        "materials": [{"name": "scales", "base_color": [0.1, 0.25, 0.08], "roughness": 0.6, "metallic": 0.0}],
    }),
    (("dog", "wolf", "horse", "cat", "quadruped"), {
        "category": "creature", "dimensions_m": [0.3, 1.0, 0.75], "primary_axis": "y", "symmetry": "x",
        "rig_type": "quadruped", "materials": [_CLOTH],
    }),
    (("snake", "tentacle", "worm"), {
        "category": "creature", "dimensions_m": [0.15, 2.0, 0.15], "primary_axis": "y", "rig_type": "chain",
        "materials": [_SKIN],
    }),
    (("knight", "soldier", "man", "woman", "person", "human", "character", "warrior", "robot"), {
        "category": "character", "dimensions_m": [0.55, 0.3, 1.8], "primary_axis": "z",
        "scale_reference": "adult human 1.75-1.8 m", "symmetry": "x", "rig_type": "humanoid",
        "materials": [_SKIN, _CLOTH],
    }),
]


def heuristic_spec(prompt: str) -> AssetSpec:
    """Best-effort spec from keywords; used when no LLM is available."""
    text = prompt.lower()
    for keywords, data in KNOWLEDGE_BASE:
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords):
            spec = AssetSpec.from_dict({**data, "name": prompt, "description": prompt}, prompt=prompt)
            break
    else:
        spec = AssetSpec.from_dict({"name": prompt, "description": prompt}, prompt=prompt)
    return spec


def knowledge_base_dimensions(prompt: str) -> Optional[list[float]]:
    """Reference dimensions from the KB, if the prompt matches an entry."""
    text = prompt.lower()
    for keywords, data in KNOWLEDGE_BASE:
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords):
            return list(data["dimensions_m"])
    return None

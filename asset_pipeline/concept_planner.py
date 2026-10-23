"""Stage 1a - concept planning agent.

Turns a free-text prompt into an :class:`AssetSpec`: category, real-world
dimensions, materials, movable parts and per-view diffusion prompts. The LLM
does the reasoning; a small built-in knowledge base is used as an offline
fallback and as a sanity check on the LLM's numbers.

Coordinate convention (Blender, Z-up, metres):
  * X = length  (longest horizontal axis; weapons/vehicles point along X)
  * Y = width / depth
  * Z = height
  * Characters face -Y (Blender "front" view looks at their face).
Part regions and pivots are normalised to the object's bounding box, so
``[0, 0, 0]`` is the min corner and ``[1, 1, 1]`` the max corner.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient, LLMError

log = logging.getLogger(__name__)

CATEGORIES = ("weapon", "vehicle", "prop", "furniture", "architecture", "character", "creature")
RIG_TYPES = ("none", "biped", "quadruped")


@dataclass
class MaterialSpec:
    name: str = "base"
    base_color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    metallic: float = 0.0
    roughness: float = 0.6
    coverage: float = 1.0  # rough fraction of surface area


@dataclass
class PartSpec:
    name: str
    description: str = ""
    # Normalised bbox region (0..1 per axis) whose faces belong to the part
    region_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    region_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    pivot: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    motion: str = "rotate"  # rotate | translate | fixed
    axis: str = "Y"
    limit_min: float = 0.0  # degrees for rotate, metres for translate
    limit_max: float = 0.0


@dataclass
class AssetSpec:
    name: str
    prompt: str
    category: str = "prop"
    description: str = ""
    length_m: float = 1.0  # X
    width_m: float = 1.0  # Y
    height_m: float = 1.0  # Z
    # Which dimension drives uniform scaling: longest | length | width | height
    fit_dimension: str = "longest"
    scale_notes: str = ""
    mechanics: str = ""
    materials: List[MaterialSpec] = field(default_factory=lambda: [MaterialSpec()])
    parts: List[PartSpec] = field(default_factory=list)
    rig_type: str = "none"
    face_budget: int = 12000
    view_prompts: Dict[str, str] = field(default_factory=dict)

    @property
    def dimensions(self) -> List[float]:
        return [self.length_m, self.width_m, self.height_m]

    @property
    def is_organic(self) -> bool:
        return self.category in ("character", "creature")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetSpec":
        data = dict(data)
        data["materials"] = [MaterialSpec(**m) for m in data.get("materials", [])] or [MaterialSpec()]
        data["parts"] = [PartSpec(**p) for p in data.get("parts", [])]
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Offline knowledge base. Dimensions are typical real-world values (metres).
# ---------------------------------------------------------------------------

_GUNMETAL = MaterialSpec("gunmetal", [0.06, 0.06, 0.07], 0.9, 0.45, 0.7)
_POLYMER = MaterialSpec("polymer", [0.03, 0.03, 0.03], 0.0, 0.7, 0.3)

KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "keywords": ["mp5", "submachine", "smg", "uzi", "mp7", "p90"],
        "category": "weapon",
        "dims": (0.68, 0.06, 0.26),
        "fit": "length",
        "notes": "HK MP5A2: 680 mm long (fixed stock), 260 mm tall with magazine, ~2.5 kg.",
        "mechanics": "Roller-delayed blowback; cocking handle on the left of the cocking tube, "
        "curved 30-round box magazine forward of the trigger guard, pivoting trigger.",
        "materials": [_GUNMETAL, _POLYMER],
        "parts": [
            PartSpec("magazine", "detachable curved box magazine", [0.50, 0.0, 0.0], [0.62, 1.0, 0.45],
                     [0.56, 0.5, 0.45], "translate", "Z", -0.25, 0.0),
            PartSpec("trigger", "pivoting trigger inside the guard", [0.36, 0.25, 0.32], [0.42, 0.75, 0.48],
                     [0.40, 0.5, 0.48], "rotate", "Y", 0.0, 15.0),
            PartSpec("cocking_handle", "charging handle on the cocking tube", [0.70, 0.0, 0.78], [0.78, 1.0, 0.92],
                     [0.74, 0.5, 0.85], "translate", "X", -0.08, 0.0),
        ],
    },
    {
        "keywords": ["rifle", "ak-47", "ak47", "m4", "ar-15", "ar15", "carbine"],
        "category": "weapon",
        "dims": (0.88, 0.07, 0.25),
        "fit": "length",
        "notes": "Assault rifles are typically 850-1000 mm long.",
        "mechanics": "Detachable magazine, trigger group, charging handle/bolt carrier.",
        "materials": [_GUNMETAL, MaterialSpec("wood", [0.35, 0.18, 0.08], 0.0, 0.6, 0.3)],
        "parts": [
            PartSpec("magazine", "detachable box magazine", [0.45, 0.0, 0.0], [0.58, 1.0, 0.45],
                     [0.52, 0.5, 0.45], "translate", "Z", -0.25, 0.0),
            PartSpec("trigger", "trigger", [0.36, 0.25, 0.30], [0.42, 0.75, 0.46],
                     [0.40, 0.5, 0.46], "rotate", "Y", 0.0, 15.0),
        ],
    },
    {
        "keywords": ["pistol", "handgun", "glock", "revolver", "1911"],
        "category": "weapon",
        "dims": (0.19, 0.03, 0.14),
        "fit": "length",
        "notes": "Full-size pistols are ~180-220 mm long and ~130-140 mm tall.",
        "mechanics": "Reciprocating slide, magazine in the grip, trigger.",
        "materials": [_GUNMETAL, _POLYMER],
        "parts": [
            PartSpec("slide", "reciprocating slide", [0.0, 0.0, 0.70], [1.0, 1.0, 1.0],
                     [0.5, 0.5, 0.85], "translate", "X", -0.03, 0.0),
            PartSpec("magazine", "magazine in grip", [0.10, 0.1, 0.0], [0.35, 0.9, 0.55],
                     [0.22, 0.5, 0.55], "translate", "Z", -0.12, 0.0),
        ],
    },
    {
        "keywords": ["sword", "katana", "longsword", "blade"],
        "category": "weapon",
        "dims": (1.0, 0.05, 0.2),
        "fit": "length",
        "notes": "Longswords 0.9-1.2 m; katana ~1.0 m overall.",
        "materials": [MaterialSpec("steel", [0.7, 0.7, 0.72], 1.0, 0.3, 0.7),
                      MaterialSpec("leather", [0.2, 0.1, 0.05], 0.0, 0.8, 0.3)],
    },
    {
        "keywords": ["castle wall", "rampart", "battlement", "fortification", "stone wall"],
        "category": "architecture",
        "dims": (6.0, 2.0, 8.0),
        "fit": "height",
        "notes": "Curtain walls are 2-3 m thick, 6-12 m tall; crenels ~0.6 m wide, merlons ~1.8 m tall "
        "so a 1.8 m defender is covered.",
        "materials": [MaterialSpec("stone", [0.45, 0.43, 0.40], 0.0, 0.9, 1.0)],
    },
    {
        "keywords": ["castle gate", "gatehouse", "portcullis", "gate"],
        "category": "architecture",
        "dims": (5.0, 3.0, 7.0),
        "fit": "height",
        "notes": "Gate passages ~3 m wide and ~4 m tall so a mounted rider passes.",
        "materials": [MaterialSpec("stone", [0.45, 0.43, 0.40], 0.0, 0.9, 0.7),
                      MaterialSpec("wood", [0.3, 0.18, 0.1], 0.0, 0.7, 0.2),
                      MaterialSpec("iron", [0.15, 0.15, 0.15], 1.0, 0.6, 0.1)],
    },
    {
        "keywords": ["door"],
        "category": "architecture",
        "dims": (0.9, 0.05, 2.1),
        "fit": "height",
        "notes": "Standard interior door 2.03-2.1 m x 0.8-0.9 m so a 1.8 m character clears it.",
        "materials": [MaterialSpec("wood", [0.4, 0.25, 0.12], 0.0, 0.6, 0.9)],
        "parts": [
            PartSpec("door_leaf", "hinged leaf", [0.02, 0.0, 0.0], [0.98, 1.0, 0.99],
                     [0.02, 0.5, 0.5], "rotate", "Z", 0.0, 100.0),
        ],
    },
    {
        "keywords": ["chair", "stool"],
        "category": "furniture",
        "dims": (0.5, 0.5, 0.9),
        "fit": "height",
        "notes": "Seat height ~0.45 m, overall ~0.8-1.0 m.",
        "materials": [MaterialSpec("wood", [0.45, 0.3, 0.15], 0.0, 0.6, 1.0)],
    },
    {
        "keywords": ["table", "desk"],
        "category": "furniture",
        "dims": (1.6, 0.8, 0.75),
        "fit": "height",
        "notes": "Work surfaces sit at ~0.72-0.76 m.",
        "materials": [MaterialSpec("wood", [0.45, 0.3, 0.15], 0.0, 0.6, 1.0)],
    },
    {
        "keywords": ["chest", "treasure chest", "trunk", "coffer"],
        "category": "prop",
        "dims": (0.9, 0.55, 0.6),
        "fit": "length",
        "notes": "Travel/treasure chests ~0.8-1.0 m wide, ~0.5-0.6 m deep and tall.",
        "mechanics": "Lid hinged along the back top edge, opens ~100 degrees.",
        "materials": [MaterialSpec("wood", [0.40, 0.24, 0.11], 0.0, 0.7, 0.75),
                      MaterialSpec("iron", [0.18, 0.18, 0.18], 1.0, 0.5, 0.25)],
        "parts": [
            PartSpec("lid", "hinged lid", [0.0, 0.0, 0.62], [1.0, 1.0, 1.0],
                     [0.5, 1.0, 0.62], "rotate", "X", -100.0, 0.0),
        ],
    },
    {
        "keywords": ["crate", "box", "barrel"],
        "category": "prop",
        "dims": (0.8, 0.6, 0.6),
        "fit": "longest",
        "notes": "Hand-movable props stay under ~1 m.",
        "materials": [MaterialSpec("wood", [0.45, 0.3, 0.15], 0.0, 0.7, 0.85),
                      MaterialSpec("iron", [0.2, 0.2, 0.2], 1.0, 0.5, 0.15)],
    },
    {
        "keywords": ["car", "sedan", "truck", "jeep", "vehicle"],
        "category": "vehicle",
        "dims": (4.5, 1.8, 1.45),
        "fit": "length",
        "notes": "Mid-size sedan ~4.5-4.9 m long, 1.8 m wide, 1.45 m tall; wheel diameter ~0.65 m.",
        "materials": [MaterialSpec("paint", [0.6, 0.05, 0.05], 0.5, 0.3, 0.7),
                      MaterialSpec("rubber", [0.02, 0.02, 0.02], 0.0, 0.9, 0.15),
                      MaterialSpec("glass", [0.1, 0.12, 0.14], 0.0, 0.05, 0.15)],
        "parts": [
            PartSpec("wheel_front_left", "wheel", [0.72, 0.0, 0.0], [0.90, 0.15, 0.45],
                     [0.81, 0.08, 0.22], "rotate", "Y", 0.0, 0.0),
            PartSpec("wheel_front_right", "wheel", [0.72, 0.85, 0.0], [0.90, 1.0, 0.45],
                     [0.81, 0.92, 0.22], "rotate", "Y", 0.0, 0.0),
            PartSpec("wheel_rear_left", "wheel", [0.10, 0.0, 0.0], [0.28, 0.15, 0.45],
                     [0.19, 0.08, 0.22], "rotate", "Y", 0.0, 0.0),
            PartSpec("wheel_rear_right", "wheel", [0.10, 0.85, 0.0], [0.28, 1.0, 0.45],
                     [0.19, 0.92, 0.22], "rotate", "Y", 0.0, 0.0),
        ],
    },
    {
        "keywords": ["dragon", "wolf", "dog", "horse", "cat", "lion", "creature", "beast", "quadruped"],
        "category": "creature",
        "dims": (1.6, 0.45, 1.0),
        "fit": "longest",
        "rig": "quadruped",
        "notes": "Large dog ~1.0-1.2 m nose-to-tail, horse ~2.4 m long and 1.6 m at the withers.",
        "materials": [MaterialSpec("skin", [0.4, 0.3, 0.2], 0.0, 0.8, 1.0)],
    },
    {
        "keywords": ["character", "human", "man", "woman", "soldier", "knight", "warrior",
                     "person", "humanoid", "robot", "orc", "elf"],
        "category": "character",
        "dims": (0.55, 0.3, 1.8),
        "fit": "height",
        "rig": "biped",
        "notes": "Adult human ~1.75-1.8 m; generate in A-pose or T-pose for rigging.",
        "materials": [MaterialSpec("skin", [0.8, 0.6, 0.5], 0.0, 0.6, 0.3),
                      MaterialSpec("cloth", [0.3, 0.3, 0.35], 0.0, 0.85, 0.7)],
    },
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "asset"


def lookup_knowledge_base(prompt: str) -> Optional[Dict[str, Any]]:
    text = prompt.lower()
    best, best_len = None, 0
    for entry in KNOWLEDGE_BASE:
        for kw in entry["keywords"]:
            # Longest keyword wins so "castle gate" beats "gate" and "castle wall" beats "wall"
            if re.search(rf"\b{re.escape(kw)}\b", text) and len(kw) > best_len:
                best, best_len = entry, len(kw)
    return best


def default_view_prompts(prompt: str, category: str) -> Dict[str, str]:
    base = (
        f"{prompt}, single object, entire object fully in frame and centered, isolated on a pure white "
        "background, studio product photography, even soft lighting, sharp focus, highly detailed, "
        "physically accurate proportions"
    )
    if category in ("character", "creature"):
        base += ", full body, neutral A-pose, arms away from body"
    return {
        "isometric": f"{base}, three-quarter isometric view from slightly above",
        "front": f"{base}, orthographic front view",
        "side": f"{base}, orthographic side profile view",
    }


def spec_from_knowledge_base(prompt: str) -> AssetSpec:
    entry = lookup_knowledge_base(prompt)
    if entry is None:
        log.warning("No knowledge-base match for '%s'; using a generic 1 m prop", prompt)
        return AssetSpec(name=_slug(prompt), prompt=prompt, view_prompts=default_view_prompts(prompt, "prop"))
    length, width, height = entry["dims"]
    return AssetSpec(
        name=_slug(prompt),
        prompt=prompt,
        category=entry["category"],
        description=prompt,
        length_m=length,
        width_m=width,
        height_m=height,
        fit_dimension=entry.get("fit", "longest"),
        scale_notes=entry.get("notes", ""),
        mechanics=entry.get("mechanics", ""),
        materials=[dataclasses.replace(m) for m in entry.get("materials", [MaterialSpec()])],
        parts=[dataclasses.replace(p) for p in entry.get("parts", [])],
        rig_type=entry.get("rig", "none"),
        face_budget=30000 if entry["category"] in ("character", "creature", "vehicle") else 12000,
        view_prompts=default_view_prompts(prompt, entry["category"]),
    )


PLANNER_SYSTEM = """You are a senior technical artist planning a production 3D asset.
You reason about the real-world object: its function, mechanics, scale and proportions.
Reply with ONLY a JSON object, no prose."""

PLANNER_PROMPT = """Plan a 3D asset for this request: "{prompt}"

Return JSON with exactly these keys:
{{
  "name": "snake_case_name",
  "category": one of {categories},
  "description": "one paragraph: what it is, real-world reference model, key visual features",
  "length_m": number (X axis, longest horizontal extent, metres),
  "width_m": number (Y axis, metres),
  "height_m": number (Z axis, metres),
  "fit_dimension": "longest" | "length" | "width" | "height"  (the most reliable real-world measurement),
  "scale_notes": "real-world measurements you relied on and human-scale checks",
  "mechanics": "how it works / moves, which parts are separate",
  "materials": [{{"name": str, "base_color": [r,g,b] 0-1 linear, "metallic": 0-1, "roughness": 0-1, "coverage": 0-1}}],
  "parts": [{{"name": str, "description": str,
              "region_min": [x,y,z], "region_max": [x,y,z],
              "pivot": [x,y,z],
              "motion": "rotate" | "translate" | "fixed",
              "axis": "X" | "Y" | "Z",
              "limit_min": number, "limit_max": number}}],
  "rig_type": one of {rig_types},
  "face_budget": integer triangle/quad budget for a game-ready asset,
  "view_prompts": {{"isometric": str, "front": str, "side": str}}
}}

Rules:
- Coordinates: X = length, Y = width, Z = height (Z up). Weapons and vehicles lie along X.
  Characters stand along Z and face -Y.
- region_min/region_max/pivot are normalised to the object's bounding box (0..1 per axis).
- List parts ONLY for mechanically separate moving pieces (magazine, trigger, bolt, wheels, doors, lids).
  Characters and creatures get no parts, they get a rig_type instead.
- rotate limits are degrees, translate limits are metres.
- view_prompts describe the SAME object for a text-to-image model: single object, pure white background,
  studio lighting, no text. "front" and "side" must be orthographic views.
"""


class ConceptPlanner:
    def __init__(self, llm: Optional[LLMClient]):
        self.llm = llm

    def plan(self, prompt: str) -> AssetSpec:
        kb_spec = spec_from_knowledge_base(prompt)
        if self.llm is None:
            log.info("Concept planner: LLM disabled, using knowledge base")
            return kb_spec
        try:
            data = self.llm.chat_json(
                PLANNER_PROMPT.format(prompt=prompt, categories=list(CATEGORIES), rig_types=list(RIG_TYPES)),
                system=PLANNER_SYSTEM,
            )
            spec = self._validate(data, prompt, kb_spec)
        except (LLMError, ValueError, TypeError, KeyError) as exc:
            log.warning("Concept planner LLM failed (%s); falling back to knowledge base", exc)
            return kb_spec
        return spec

    @staticmethod
    def _validate(data: Dict[str, Any], prompt: str, kb: AssetSpec) -> AssetSpec:
        if not isinstance(data, dict):
            raise ValueError("planner reply is not an object")
        data["prompt"] = prompt
        data["name"] = _slug(str(data.get("name") or prompt))
        if data.get("category") not in CATEGORIES:
            data["category"] = kb.category
        if data.get("rig_type") not in RIG_TYPES:
            data["rig_type"] = kb.rig_type if data["category"] in ("character", "creature") else "none"
        if data.get("fit_dimension") not in ("longest", "length", "width", "height"):
            data["fit_dimension"] = kb.fit_dimension

        for key in ("length_m", "width_m", "height_m"):
            data[key] = float(data.get(key) or getattr(kb, key))
            if data[key] <= 0:
                data[key] = getattr(kb, key)
        # Cross-check scale against the knowledge base when we have a match.
        if lookup_knowledge_base(prompt) is not None:
            llm_max = max(data["length_m"], data["width_m"], data["height_m"])
            kb_max = max(kb.dimensions)
            if not (kb_max / 3.0 <= llm_max <= kb_max * 3.0):
                log.warning("LLM scale %.2fm disagrees with reference %.2fm; using reference dims", llm_max, kb_max)
                data["length_m"], data["width_m"], data["height_m"] = kb.dimensions
                data["scale_notes"] = kb.scale_notes

        materials = []
        for m in data.get("materials") or []:
            try:
                color = [min(max(float(c), 0.0), 1.0) for c in (m.get("base_color") or [0.5] * 3)][:3]
                materials.append(MaterialSpec(
                    name=str(m.get("name", "material")),
                    base_color=color + [0.5] * (3 - len(color)),
                    metallic=min(max(float(m.get("metallic", 0.0)), 0.0), 1.0),
                    roughness=min(max(float(m.get("roughness", 0.6)), 0.0), 1.0),
                    coverage=min(max(float(m.get("coverage", 1.0)), 0.0), 1.0),
                ))
            except (TypeError, ValueError, AttributeError):
                continue
        data["materials"] = [dataclasses.asdict(m) for m in (materials or kb.materials)]

        parts = []
        for p in data.get("parts") or []:
            try:
                def vec(v, default):
                    v = [min(max(float(x), 0.0), 1.0) for x in (v or default)]
                    return (v + default[len(v):])[:3]

                lo = vec(p.get("region_min"), [0.0, 0.0, 0.0])
                hi = vec(p.get("region_max"), [1.0, 1.0, 1.0])
                lo, hi = [min(a, b) for a, b in zip(lo, hi)], [max(a, b) for a, b in zip(lo, hi)]
                parts.append(PartSpec(
                    name=_slug(str(p["name"])),
                    description=str(p.get("description", "")),
                    region_min=lo,
                    region_max=hi,
                    pivot=vec(p.get("pivot"), [0.5, 0.5, 0.5]),
                    motion=p.get("motion") if p.get("motion") in ("rotate", "translate", "fixed") else "fixed",
                    axis=str(p.get("axis", "Y")).upper()[:1] if str(p.get("axis", "Y")).upper()[:1] in "XYZ" else "Y",
                    limit_min=float(p.get("limit_min", 0.0)),
                    limit_max=float(p.get("limit_max", 0.0)),
                ))
            except (TypeError, ValueError, KeyError, AttributeError):
                continue
        if data["category"] in ("character", "creature"):
            data["parts"] = []
        else:
            data["parts"] = [dataclasses.asdict(p) for p in (parts or kb.parts)]

        try:
            data["face_budget"] = max(500, min(int(data.get("face_budget") or kb.face_budget), 200000))
        except (TypeError, ValueError):
            data["face_budget"] = kb.face_budget
        prompts = data.get("view_prompts") if isinstance(data.get("view_prompts"), dict) else {}
        defaults = default_view_prompts(prompt, data["category"])
        data["view_prompts"] = {k: str(prompts.get(k) or v) for k, v in defaults.items()}
        for key in ("description", "scale_notes", "mechanics"):
            data[key] = str(data.get(key) or getattr(kb, key))
        return AssetSpec.from_dict(data)

from asset_pipeline.concept_planner import ConceptPlanner
from asset_pipeline.config import Config
from asset_pipeline.orchestrator import Pipeline
from asset_pipeline.concept_planner import spec_from_knowledge_base
from asset_pipeline.shape_templates import PRIMITIVES, RECIPES, expand_arrays, recipe_for, validate_components

from conftest import FakeLLM


def test_recipe_lookup():
    assert {c.type for c in recipe_for("a castle tower")} >= {"cylinder", "cone"}
    assert recipe_for("a castle gate")[0].name == "tower_left"  # "castle gate" beats "gate"/"tower"
    assert recipe_for("a wooden crate")[0].type == "box"
    assert recipe_for("a knight", "character") == []
    assert recipe_for("a spaceship") == []


def test_library_is_valid():
    for entry in RECIPES:
        comps = entry["shapes"]
        assert validate_components([c.to_dict() for c in comps]) == comps, entry["keywords"]
        assert all(c.type in PRIMITIVES for c in comps)
        assert any(c.operation == "union" and not c.part for c in comps), entry["keywords"]


def test_expand_arrays_equal_gaps():
    wall = [c for c in recipe_for("castle wall") if c.name == "merlon"][0]
    teeth = expand_arrays([wall])
    assert len(teeth) == wall.array_count
    widths = {round(t.max[0] - t.min[0], 6) for t in teeth}
    assert len(widths) == 1
    width = teeth[0].max[0] - teeth[0].min[0]
    assert abs(teeth[1].min[0] - teeth[0].max[0] - width) < 1e-9  # gap == tooth width
    assert abs(teeth[-1].max[0] - wall.max[0]) < 1e-9


def test_validate_components_sanitises():
    raw = [
        {"type": "cylinder", "min": [0.9, 0, 0], "max": [0.1, 1, 1], "axis": "q", "segments": 1000,
         "material": "stone", "part": "Lid!"},
        {"type": "teapot"},
        {"type": "box", "min": [0, 0, 0], "max": [0, 1, 1]},  # degenerate
        {"type": "box", "operation": "subtract", "min": [0.4, -3, 0], "max": [0.6, 5, 0.3]},
    ]
    comps = validate_components(raw, material_names=["stone"], part_names=["lid"])
    assert [c.type for c in comps] == ["cylinder", "box"]
    cyl = comps[0]
    assert cyl.min[0] == 0.1 and cyl.max[0] == 0.9 and cyl.axis == "Z" and cyl.segments == 96
    assert cyl.part == "lid" and cyl.material == "stone"
    assert comps[1].min[1] == -1.0 and comps[1].max[1] == 2.0  # clamped
    assert validate_components([{"type": "box", "operation": "subtract"}]) == []  # nothing to cut from


def test_planner_accepts_llm_shapes():
    reply = {"name": "watchtower", "category": "architecture", "length_m": 4, "width_m": 4, "height_m": 12,
             "materials": [{"name": "stone", "base_color": [0.4, 0.4, 0.4]}],
             "shapes": [{"type": "cylinder", "min": [0, 0, 0], "max": [1, 1, 0.8], "material": "stone"},
                        {"type": "cone", "min": [0, 0, 0.8], "max": [1, 1, 1], "axis": "+Z", "material": "slate"}]}
    spec = ConceptPlanner(FakeLLM([reply])).plan("a stone watchtower")
    assert [s["type"] for s in spec.shapes] == ["cylinder", "cone"]
    assert spec.shapes[1]["material"] == ""  # unknown material dropped


def test_pipeline_shape_components_and_materials():
    pipe = Pipeline(Config())
    chest = spec_from_knowledge_base("a treasure chest")
    comps = pipe._shape_components(chest)
    assert {c["part"] for c in comps} == {"", "lid"}
    tower = spec_from_knowledge_base("castle tower")
    names = [m["name"] for m in pipe._materials(tower, pipe._shape_components(tower))]
    assert names == ["stone", "roof"]
    assert pipe._shape_components(spec_from_knowledge_base("a knight character")) is None
    cfg = Config()
    cfg.shapes.mode = "mesh"
    assert Pipeline(cfg)._shape_components(chest) is None
    assert pipe._shape_components(spec_from_knowledge_base("a spaceship")) == []  # auto-classify


def test_no_single_shape_guess_for_weapons():
    pipe = Pipeline(Config())
    assert pipe._shape_components(spec_from_knowledge_base("MP5 submachine gun")) is None
    assert pipe._shape_components(spec_from_knowledge_base("a longsword"))  # has a recipe

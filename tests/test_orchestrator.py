from pathlib import Path

from asset_pipeline.concept_planner import spec_from_knowledge_base
from asset_pipeline.config import Config
from asset_pipeline.orchestrator import Pipeline, RunContext, RunOptions
from asset_pipeline.reference_generator import ReferenceSet


def ctx_for(prompt, rigging=True):
    ctx = RunContext(Path("."), RunOptions(prompt=prompt, enable_rigging=rigging))
    ctx.spec = spec_from_knowledge_base(prompt)
    return ctx


def test_stage5_mode():
    pipe = Pipeline(Config())
    assert pipe._stage5_mode(ctx_for("MP5 submachine gun")) == "segment"
    assert pipe._stage5_mode(ctx_for("a knight character")) == "rig"
    assert pipe._stage5_mode(ctx_for("medieval castle wall")) is None  # no parts
    assert pipe._stage5_mode(ctx_for("MP5 submachine gun", rigging=False)) is None


def test_cleanup_params_use_spec_budget_unless_overridden():
    cfg = Config()
    spec = spec_from_knowledge_base("a knight character")
    assert Pipeline(cfg)._cleanup_params(spec)["target_faces"] == spec.face_budget
    cfg.cleanup.target_faces = 5000
    params = Pipeline(cfg)._cleanup_params(spec)
    assert params["target_faces"] == 5000
    assert params["dimensions"] == [0.55, 0.3, 1.8] and params["fit_dimension"] == "height"


def test_projection_planes_follow_category():
    cfg = Config()
    cfg.texture.albedo_source = "projection"
    views = {v: {"rgba": f"{v}.png", "prepared": f"{v}_p.png", "raw": ""} for v in ("isometric", "front", "side")}
    refs = ReferenceSet(primary="isometric_p.png", views=views)
    weapon = Pipeline(cfg)._texture_params(spec_from_knowledge_base("mp5 smg"), refs)["projections"]
    # A weapon's "side" diffusion view is its profile, i.e. Blender's XZ (front) plane
    assert {p["plane"]: p["image"] for p in weapon} == {"XZ": "side.png", "YZ": "front.png"}
    chair = Pipeline(cfg)._texture_params(spec_from_knowledge_base("a chair"), refs)["projections"]
    assert {p["plane"]: p["image"] for p in chair} == {"XZ": "front.png", "YZ": "side.png"}


def test_projection_without_views_falls_back_to_bake():
    cfg = Config()
    cfg.texture.albedo_source = "projection"
    refs = ReferenceSet(primary="input_p.png", views={"input": {"rgba": "i.png", "prepared": "i_p.png", "raw": ""}})
    params = Pipeline(cfg)._texture_params(spec_from_knowledge_base("a chair"), refs)
    assert params["albedo_source"] == "bake" and params["projections"] == []

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


def test_reference_projection_and_character_settings():
    from asset_pipeline.mesh_reconstruction import TripoSRBackend

    cam = TripoSRBackend.input_camera()
    # TripoSR's +X camera, after the export re-orientation, sits on glTF -Z with +Y up
    assert [round(v, 6) for v in cam["position"]] == [0.0, 0.0, -1.9]
    assert [round(v, 6) for v in cam["up"]] == [0.0, 1.0, 0.0] and cam["coords"] == "gltf"

    pipe = Pipeline(Config())
    views = {v: {"rgba": f"{v}.png", "prepared": f"{v}_p.png", "projection": f"{v}_proj.png", "raw": ""}
             for v in ("isometric", "front", "side")}
    refs = ReferenceSet(primary="isometric_p.png", views=views)
    tex = pipe._texture_params(spec_from_knowledge_base("a treasure chest"), refs, cam)
    assert tex["reference_image"] == "isometric_proj.png" and tex["reference_camera"] == cam
    assert tex["mirror_axes"] == ["X", "Y"]
    assert pipe._mirror_axes(spec_from_knowledge_base("mp5 smg")) == ["Y"]  # never swap muzzle and stock
    assert pipe._mirror_axes(spec_from_knowledge_base("a knight character")) == ["X"]

    knight = pipe._cleanup_params(spec_from_knowledge_base("a knight character"), cam)
    assert knight["symmetrize"] is True and knight["max_tilt_correction"] == 0.0
    chest = pipe._cleanup_params(spec_from_knowledge_base("a treasure chest"), cam)
    assert chest["symmetrize"] is False and chest["reference_camera"] == cam


def test_castles_keep_their_scan_layout():
    pipe = Pipeline(Config())
    spec = spec_from_knowledge_base("a medieval castle")
    assert spec.category == "architecture" and spec.fit_dimension == "height"
    params = pipe._cleanup_params(spec)
    assert params["per_axis_fit"] is False and params["match_spec_axes"] is False
    assert pipe._shape_components(spec) == []  # built from the scan in Blender, never a planner guess
    names = {m["name"] for m in pipe._materials(spec, [])}
    assert {"stone", "roof"} <= names
    assert pipe._mirror_axes(spec) == ["X"]
    crate = spec_from_knowledge_base("a wooden crate")
    assert pipe._cleanup_params(crate)["match_spec_axes"] is True

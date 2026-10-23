import dataclasses

import pytest

from asset_pipeline.concept_planner import spec_from_knowledge_base
from asset_pipeline.config import CleanupConfig, TextureConfig
from asset_pipeline.llm_client import LLMError
from asset_pipeline.vision_auditor import (Issue, VisionAuditor, apply_fixes, apply_rig_fixes, check_metrics,
                                           check_rig,
                                           image_box_to_region, remap_point)

from conftest import FakeLLM


def good_metrics(name, dims=(0.68, 0.06, 0.26)):
    return {name: {"faces": 8000, "boundary_edges": 0, "non_manifold_edges": 0, "watertight": True,
                   "small_islands": 0, "stretch": 0.1, "dimensions_m": list(dims)}}


@pytest.fixture
def spec():
    return spec_from_knowledge_base("MP5 submachine gun")


@pytest.fixture
def params():
    c = dataclasses.asdict(CleanupConfig())
    c.update(target_faces=12000, axis_scale={})
    return c, dataclasses.asdict(TextureConfig())


def test_clean_metrics_pass(spec):
    assert check_metrics(good_metrics(spec.name), spec, 12000) == []


def test_metric_issues(spec):
    m = good_metrics(spec.name, dims=(0.68, 0.06, 0.40))
    m[spec.name].update(boundary_edges=12, stretch=1.5, small_islands=3, faces=40000)
    fixes = {i.fix: i for i in check_metrics(m, spec, 12000)}
    assert fixes["make_manifold"].severity == "high"
    assert fixes["reunwrap"].severity == "high"
    assert "remove_floaters" in fixes and "reduce_faces" in fixes
    assert fixes["rescale_axis"].params["axis"] == "Z"
    assert fixes["rescale_axis"].params["factor"] == pytest.approx(0.26 / 0.40, abs=1e-3)


def test_assembled_dims_override_body(spec):
    # Body alone is short once the magazine is cut off; the assembly is what matters
    m = good_metrics(spec.name, dims=(0.68, 0.06, 0.15))
    assert check_metrics(m, spec, 12000, assembled_dims=[0.68, 0.06, 0.26]) == []


def test_apply_fixes_closed_loop(params):
    cleanup, texture = params
    cleanup["remesh_mode"] = "decimate"
    issues = [Issue("geometry", "high", "holes", "make_manifold"),
              Issue("texture", "medium", "stretch", "reunwrap"),
              Issue("proportion", "medium", "tall", "rescale_axis", {"axis": "z", "factor": 0.3}),
              Issue("artifact", "low", "bad", None)]
    c, t, changes = apply_fixes(cleanup, texture, issues)
    assert c["remesh_mode"] == "voxel"
    assert t["uv_angle_limit"] < texture["uv_angle_limit"]
    assert c["axis_scale"] == {"Z": 0.5}  # cumulative bound
    assert len(changes) == 3
    assert cleanup["remesh_mode"] == "decimate"  # inputs untouched


def test_make_manifold_changes_strategy(params):
    cleanup, texture = params
    cleanup.update(remesh_mode="quadriflow", voxel_resolution=192)
    holes = [Issue("geometry", "high", "holes", "make_manifold")]
    c, _, changes = apply_fixes(cleanup, texture, holes)
    assert c["remesh_mode"] == "voxel" and c["voxel_resolution"] == 192 and changes
    c2, _, _ = apply_fixes(c, texture, holes)  # only once the strategy is voxel does resolution grow
    assert c2["voxel_resolution"] == 240


def test_apply_fixes_exhausts(params):
    cleanup, texture = params
    cleanup.update(min_island_ratio=0.2)
    c, t, changes = apply_fixes(cleanup, texture, [Issue("artifact", "medium", "x", "remove_floaters")])
    assert changes == []


def test_duplicate_fixes_apply_once(params):
    cleanup, texture = params
    issues = [Issue("texture", "high", "a", "reunwrap"), Issue("texture", "medium", "b", "reunwrap")]
    _, t, changes = apply_fixes(cleanup, texture, issues)
    assert len(changes) == 1 and t["uv_angle_limit"] == pytest.approx(49.5)


def test_image_box_to_region():
    framing = {"plane": "XZ", "center": [0.0, 0.0, 0.13], "ortho_scale": 0.8, "resolution": 1000}
    lo, hi = [-0.34, -0.03, 0.0], [0.34, 0.03, 0.26]
    (x0, z0), (x1, z1) = image_box_to_region([0.5, 0.5, 0.6, 0.7], framing, lo, hi)
    assert x0 == pytest.approx(0.5) and x1 == pytest.approx((0.08 + 0.34) / 0.68)
    # bottom edge maps below the bbox (z = -0.03 m) and is clamped to 0
    assert z1 == pytest.approx(0.5) and z0 == 0.0
    # pixel coordinates are normalised automatically
    assert image_box_to_region([500, 500, 600, 700], framing, lo, hi) == image_box_to_region(
        [0.5, 0.5, 0.6, 0.7], framing, lo, hi)


def test_remap_point():
    assert remap_point([0.5, 0.5, 0.5], [0, 0, 0], [1, 1, 1], [0.2, 0, 0], [0.4, 1, 0.5]) == [0.3, 0.5, 0.25]


def test_audit_combines_vision_and_metrics(spec, tmp_path):
    renders = {}
    for view in ("front", "side", "wireframe", "shaded"):
        path = tmp_path / f"{view}.png"
        path.write_bytes(b"png")
        renders[view] = str(path)
    vlm = FakeLLM([{"score": 0.6, "approved": False, "issues": [
        {"category": "geometry", "severity": "medium", "description": "lumpy receiver", "fix": "reduce_noise"},
        {"category": "x", "severity": "bogus", "description": "?", "fix": "delete_everything"}]}])
    report = VisionAuditor(vlm, 0.75).audit(1, renders, good_metrics(spec.name), spec, 12000)
    assert not report.approved and report.score == 0.6
    assert [i.fix for i in report.issues] == ["reduce_noise", None]
    assert report.issues[1].severity == "medium"
    assert len(vlm.calls[0]["images"]) == 4


def test_audit_survives_vision_failure(spec):
    report = VisionAuditor(FakeLLM([LLMError("timeout")]), 0.75).audit(
        1, {}, good_metrics(spec.name), spec, 12000)
    # A configured vision model that fails makes the result unverified, never approved (QA BUG-5)
    assert not report.approved and report.unverified and report.vision_error == "timeout"
    offline = VisionAuditor(None, 0.75).audit(1, {}, good_metrics(spec.name), spec, 12000)
    assert offline.approved and not offline.unverified  # --offline: metrics-only approval is intended


def test_open_medium_metric_issue_blocks_approval(spec):
    m = good_metrics(spec.name, dims=(0.68, 0.06, 0.40))  # +54% height
    report = VisionAuditor(None, 0.75).audit(1, {}, m, spec, 12000)
    assert not report.approved and any("Z extent" in b for b in report.blocking)


def test_texture_checks():
    from asset_pipeline.vision_auditor import check_texture

    assert check_texture({"unbaked_fraction": 0.012}) == []  # normal seam misses, filled silently
    [i] = check_texture({"unbaked_fraction": 0.08})
    assert (i.severity, i.fix) == ("medium", "rebake")
    [k] = check_texture({"unbaked_fraction": 0.3})
    assert k.severity == "high"
    [j] = check_texture({"projection": {"skipped": "silhouette does not match", "silhouette_iou": 0.56}})
    assert j.severity == "low"


def test_locate_parts_updates_regions(spec, tmp_path):
    render = tmp_path / "front.png"
    render.write_bytes(b"png")
    framing = {"plane": "XZ", "center": [0.0, 0.0, 0.13], "ortho_scale": 0.8, "resolution": 1024}
    vlm = FakeLLM([{"parts": [{"name": "magazine", "visible": True, "box": [0.45, 0.55, 0.55, 0.8]},
                              {"name": "trigger", "visible": False, "box": [0, 0, 0, 0]}]}])
    parts = VisionAuditor(vlm).locate_parts(str(render), framing, [-0.34, -0.03, 0.0], [0.34, 0.03, 0.26], spec)
    by_name = {p["name"]: p for p in parts}
    mag = by_name["magazine"]
    assert mag["region_min"][0] == pytest.approx((0.34 - 0.04) / 0.68, abs=1e-6)
    assert by_name["trigger"]["region_min"] == spec.parts[1].region_min  # unchanged


def test_thin_axis_protrusions_are_tolerated():
    door = spec_from_knowledge_base("a wooden door")  # 0.9 x 0.05 x 2.1 m
    m = good_metrics(door.name, dims=(0.9, 0.07, 2.1))  # handle sticks out 2 cm
    assert check_metrics(m, door, 12000) == []


def test_rig_checks_and_reweight_escalation():
    assert check_rig(None) == []
    good = {"weight_coverage": 1.0, "pose_test": {"bad_edge_fraction": 0.004}, "landmarks": {"fallback": []}}
    assert check_rig(good) == []
    bad = {"weight_coverage": 0.9, "pose_test": {"bad_edge_fraction": 0.12}, "landmarks": {"fallback": ["arm.L"]}}
    issues = check_rig(bad)
    assert [i.severity for i in issues] == ["high", "high", "medium"]  # guessed arm = unreliable rig
    stretched = {"weight_coverage": 1.0, "pose_test": {"bad_edge_fraction": 0.01, "max_stretch": 57.8}}
    assert [i.fix for i in check_rig(stretched)] == ["reweight"]
    rig = {"weights": "auto", "weight_smooth": 1}
    rig, changes = apply_rig_fixes(rig, issues)
    assert rig["weight_smooth"] == 3 and changes
    rig, _ = apply_rig_fixes(rig, issues)
    rig, changes = apply_rig_fixes(rig, issues)
    assert rig["weights"] == "proximity"
    assert apply_rig_fixes(rig, issues)[1] == []  # exhausted
    assert apply_rig_fixes(rig, [Issue("geometry", "high", "x", "make_manifold")])[1] == []


def test_rescale_is_metric_only_and_stops_at_bound(params):
    cleanup, texture = params
    vision = Issue("proportion", "medium", "looks wide", "rescale_axis", {"axis": "Y", "factor": 9.259},
                   source="vision")
    _, _, changes = apply_fixes(cleanup, texture, [vision])
    assert changes == []  # VLM factors never rescale
    metric = Issue("proportion", "medium", "Y +329%", "rescale_axis", {"axis": "Y", "factor": 0.233})
    c, _, changes = apply_fixes(cleanup, texture, [metric])
    assert c["axis_scale"] == {"Y": 0.5} and changes
    c2, _, changes = apply_fixes(c, texture, [metric])
    assert changes == [] and c2["axis_scale"] == {"Y": 0.5}  # at the bound: no fake progress
    grow = Issue("proportion", "medium", "hammer x2.4", "rescale_axis", {"axis": "X", "factor": 2.4})
    c3, _, _ = apply_fixes(cleanup, texture, [grow])
    assert c3["axis_scale"] == {"X": 2.0}


def test_vision_proportion_issues_are_advisory():
    issues = VisionAuditor._vision_issues({"issues": [
        {"category": "proportion", "severity": "high", "description": "too tall", "fix": "rescale_axis",
         "params": {"axis": "Z", "factor": 0.026}},
        {"category": "geometry", "severity": "medium", "description": "lumpy", "fix": "reduce_noise"}]})
    assert (issues[0].fix, issues[0].severity) == (None, "low")
    assert issues[1].fix == "reduce_noise"


def test_castle_footprint_is_not_audited_against_the_plan():
    from asset_pipeline.concept_planner import spec_from_knowledge_base

    spec = spec_from_knowledge_base("a medieval castle")  # 32 x 28 x 24 m, fitted on height
    m = {spec.name: {"faces": 9000, "boundary_edges": 0, "non_manifold_edges": 0, "small_islands": 0,
                     "dimensions_m": [18.0, 40.0, 24.0]}}
    issues = check_metrics(m, spec, 12000, assembled_dims=[18.0, 40.0, 24.0])
    assert not [i for i in issues if i.fix == "rescale_axis"]
    m[spec.name]["dimensions_m"] = [18.0, 40.0, 12.0]
    issues = check_metrics(m, spec, 12000, assembled_dims=[18.0, 40.0, 12.0])
    assert [i.params["axis"] for i in issues if i.fix == "rescale_axis"] == ["Z"]

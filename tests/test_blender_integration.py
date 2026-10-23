"""End-to-end Stages 3-6 in real headless Blender (skipped when Blender is absent).

Uses the primitive reconstruction backend and offline planning, so no GPU,
models or LLM servers are needed: ``BLENDER_PATH=/path/to/blender pytest -m blender``
"""

import json

import pytest

from asset_pipeline.blender_automation import BlenderRunner
from asset_pipeline.config import find_blender
from asset_pipeline.mesh_reconstruction import PrimitiveBackend

BLENDER = find_blender()
pytestmark = [pytest.mark.blender, pytest.mark.skipif(BLENDER is None, reason="Blender not found")]


@pytest.fixture(scope="module")
def raw_mesh(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    return PrimitiveBackend(None, (0.68, 0.06, 0.26)).reconstruct("", "", out).path


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    return BlenderRunner(BLENDER, tmp_path_factory.mktemp("jobs"), timeout_s=600, use_gpu=False)


def cleanup_params(raw_mesh, out, **overrides):
    params = dict(input_mesh=raw_mesh, output_blend=str(out), name="mp5", dimensions=[0.68, 0.06, 0.26],
                  fit_dimension="length", merge_distance=1e-4, min_island_ratio=0.02, remesh_mode="quadriflow",
                  voxel_resolution=128, target_faces=4000, smooth_iterations=1, auto_smooth_angle=40,
                  fix_intersections=True, fill_holes=True, orient_longest_axis=True,
                  import_rotation_deg=[0, 0, 30])
    params.update(overrides)
    return params


def test_stage3_to_export(runner, raw_mesh, tmp_path):
    s3 = runner.run("cleanup", cleanup_params(raw_mesh, tmp_path / "s3.blend"))
    m = s3["metrics"]
    assert m["watertight"] and m["quad_ratio"] > 0.9
    rx, ry, rz = s3["rotation_applied_deg"]
    assert abs(rz + 30) < 1 and abs(rx) < 1 and abs(ry) < 1  # undid the 30 degree import yaw
    assert m["dimensions_m"][0] == pytest.approx(0.68, rel=0.01)
    assert s3["floaters_removed"] == 1

    parts = [{"name": "magazine", "region_min": [0.5, 0, 0], "region_max": [0.62, 1, 0.45],
              "pivot": [0.56, 0.5, 0.45], "motion": "translate", "axis": "Z", "limit_min": -0.25, "limit_max": 0}]
    s5 = runner.run("segment", dict(input_blend=str(tmp_path / "s3.blend"),
                                    output_blend=str(tmp_path / "s5.blend"), parts=parts, clearance_mm=0.2))
    assert s5["parts"][0]["watertight"] and s5["metrics"]["watertight"]

    s4 = runner.run("texture", dict(input_blend=str(tmp_path / "s5.blend"), output_blend=str(tmp_path / "s4.blend"),
                                    texture_dir=str(tmp_path / "tex"), resolution=256, uv_method="smart",
                                    uv_angle_limit=66, island_margin=0.005, bake_margin_px=4, cage_extrusion=0.01,
                                    bake_samples=1, albedo_source="bake",
                                    materials=[{"name": "steel", "base_color": [0.1, 0.1, 0.1], "metallic": 1.0,
                                                "roughness": 0.4, "coverage": 1.0}]))
    assert s4["albedo"].startswith("color_attribute")
    assert len(s4["metrics"]) == 2 and all(v["stretch"] < 0.5 for v in s4["metrics"].values())

    r = runner.run("render", dict(input_blend=str(tmp_path / "s4.blend"), output_dir=str(tmp_path / "renders"),
                                  resolution=64, samples=1))
    assert set(r["renders"]) == {"front", "side", "wireframe", "shaded", "uv_checker"}
    assert r["framing"]["front"]["plane"] == "XZ"

    for fmt in ("glb", "fbx", "obj"):
        e = runner.run("export", dict(input_blend=str(tmp_path / "s4.blend"), format=fmt,
                                      output_path=str(tmp_path / fmt / f"mp5.{fmt}"),
                                      output_blend=str(tmp_path / fmt / "mp5.blend")), tag=f"export_{fmt}")
        assert e["bytes"] > 10000


def test_generated_rig(runner, raw_mesh, tmp_path):
    runner.run("cleanup", cleanup_params(raw_mesh, tmp_path / "c.blend", dimensions=[0.55, 0.3, 1.8],
                                         fit_dimension="height", remesh_mode="voxel"), tag="c_char")
    rig = runner.run("rig", dict(input_blend=str(tmp_path / "c.blend"), output_blend=str(tmp_path / "r.blend"),
                                 rig_type="biped", method="generated"), tag="rig_char")
    assert rig["weight_coverage"] > 0.95 and rig["deform_bones"] == 19


def test_errors_are_reported(runner, tmp_path):
    from asset_pipeline.blender_automation import BlenderError

    with pytest.raises(BlenderError, match="no such file|not found|Cannot read|unsupported"):
        runner.run("cleanup", cleanup_params(str(tmp_path / "missing.xyz"), tmp_path / "x.blend"), tag="bad")
    result = json.loads((runner.work_dir / "result_bad.json").read_text())
    assert result["status"] == "error"


def test_tilted_reconstruction_is_uprighted(runner, raw_mesh, tmp_path):
    # Camera-elevation tilt + yaw, as a three-quarter view from above produces
    s3 = runner.run("cleanup", cleanup_params(raw_mesh, tmp_path / "t.blend", import_rotation_deg=[20, -15, 30],
                                              remesh_mode="voxel"), tag="tilted")
    dims = s3["metrics"]["dimensions_m"]
    assert dims[0] == pytest.approx(0.68, rel=0.01)
    assert dims[1] == pytest.approx(0.06, rel=0.08)
    assert dims[2] == pytest.approx(0.26, rel=0.03)


def test_primitive_templates(runner, raw_mesh, tmp_path):
    from asset_pipeline.shape_templates import expand_arrays, recipe_for

    runner.run("cleanup", cleanup_params(raw_mesh, tmp_path / "p3.blend", remesh_mode="voxel"), tag="p_clean")
    mats = [{"name": "wood", "base_color": [0.4, 0.24, 0.11], "metallic": 0.0, "roughness": 0.7, "coverage": 0.8},
            {"name": "iron", "base_color": [0.16, 0.16, 0.17], "metallic": 1.0, "roughness": 0.5, "coverage": 0.2}]
    # No recipe: the box-like blob must be classified as a box
    auto = runner.run("primitives", dict(input_blend=str(tmp_path / "p3.blend"), output_blend=str(tmp_path / "a.blend"),
                                         components=[], materials=mats, default_material="wood"), tag="p_auto")
    assert auto["used"] and auto["source"] == "classified:box"
    body = next(iter(auto["metrics"].values()))
    assert body["quad_ratio"] == 1.0 and body["watertight"]

    # Recipe with a separate part, arrays and booleans
    comps = [c.to_dict() for c in expand_arrays(recipe_for("treasure chest"))]
    parts = [{"name": "lid", "pivot": [0.5, 1.0, 0.62], "motion": "rotate", "axis": "X", "limit_min": -100,
              "limit_max": 0}]
    res = runner.run("primitives", dict(input_blend=str(tmp_path / "p3.blend"), output_blend=str(tmp_path / "c.blend"),
                                        components=comps, mode="primitives", materials=mats, parts=parts),
                     tag="p_chest")
    assert res["used"] and res["parts_built"] == ["lid"]
    assert all(m["watertight"] for m in res["metrics"].values())
    tex = runner.run("texture", dict(input_blend=str(tmp_path / "c.blend"), output_blend=str(tmp_path / "ct.blend"),
                                     texture_dir=str(tmp_path / "ctex"), resolution=128, uv_method="smart",
                                     uv_angle_limit=66, island_margin=0.01, bake_margin_px=2,
                                     cage_extrusion=res["cage_hint"], bake_samples=1, albedo_source="bake",
                                     materials=mats, bake_normal=False), tag="p_tex")
    assert tex["material_ids"] and tex["normal"] == "none"
    assert 0.05 < tex["material_coverage"]["iron"] < 0.6  # bands/lock come from template IDs, not colour guesses

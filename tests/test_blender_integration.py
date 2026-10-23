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
    assert rig["weight_coverage"] > 0.95 and rig["deform_bones"] == 21


def test_errors_are_reported(runner, tmp_path):
    from asset_pipeline.blender_automation import BlenderError

    with pytest.raises(BlenderError, match="no such file|not found|Cannot read|unsupported"):
        runner.run("cleanup", cleanup_params(str(tmp_path / "missing.xyz"), tmp_path / "x.blend"), tag="bad")
    result = json.loads((runner.work_dir / "result_bad.json").read_text())
    assert result["status"] == "error"


def test_timeout_is_reported(runner, raw_mesh, tmp_path):
    from asset_pipeline.blender_automation import BlenderTimeout

    with pytest.raises(BlenderTimeout, match="timed out after 1s"):  # Blender start-up alone takes longer
        runner.run("cleanup", cleanup_params(raw_mesh, tmp_path / "t.blend"), tag="slow", timeout_s=1)


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


@pytest.mark.parametrize("pose,method", [("A", "generated"), ("T", "generated"), ("A", "rigify")])
def test_landmark_rig_on_mannequin(runner, tmp_path, pose, method):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from make_mannequin import write_obj

    raw = write_obj(str(tmp_path / "mannequin.obj"), pose)
    runner.run("cleanup", cleanup_params(raw, tmp_path / "m.blend", dimensions=[0.6, 0.3, 1.8], fit_dimension="height",
                                         remesh_mode="voxel", voxel_resolution=160, target_faces=6000,
                                         import_rotation_deg=[0, 0, 0], max_tilt_correction=0.0, symmetrize=True),
               tag=f"m_{pose}_{method}")
    rig = runner.run("rig", dict(input_blend=str(tmp_path / "m.blend"), output_blend=str(tmp_path / "r.blend"),
                                 rig_type="biped", method=method), tag=f"rig_{pose}_{method}")
    lm = rig["landmarks"]
    assert lm["pose"] == pose and lm["fallback"] == []
    assert rig["weights"] == "bone_heat" and rig["weight_coverage"] == 1.0
    assert rig["weight_cleanup"]["max_influences"] <= 4 and rig["unweighted_vertices"] == 0
    assert rig["pose_test"]["bad_edge_fraction"] < 0.02  # arms raised, elbows/knees bent, torso twisted


def test_blueprint_rebuilds_gun(runner, tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from make_gun import write_obj

    raw = write_obj(str(tmp_path / "gun.obj"))
    s3 = runner.run("cleanup", cleanup_params(raw, tmp_path / "g.blend", dimensions=[0.83, 0.05, 0.23],
                                              fit_dimension="length", remesh_mode="voxel", voxel_resolution=256,
                                              import_rotation_deg=[0, 0, 0]), tag="gun_clean")
    res = runner.run("primitives", dict(input_blend=str(tmp_path / "g.blend"), output_blend=str(tmp_path / "gb.blend"),
                                        components=[], classify=False, blueprint=True, materials=[]), tag="gun_bp")
    assert res["used"] and res["source"] == "blueprint", res
    assert res["fit_error"] < 0.02
    assert res["blueprint"]["side"]["loops"] >= 2  # outline + the trigger-guard opening
    body = next(iter(res["metrics"].values()))
    assert body["watertight"]
    for got, want in zip(body["dimensions_m"], s3["metrics"]["dimensions_m"]):
        assert abs(got - want) < 0.02
    # Cut the magazine out of the blueprint body: both pieces must stay closed
    parts = [{"name": "magazine", "region_min": [0.42, 0.0, 0.0], "region_max": [0.52, 1.0, 0.6],
              "pivot": [0.47, 0.5, 0.6], "motion": "translate", "axis": "Z", "limit_min": -0.25, "limit_max": 0}]
    seg = runner.run("segment", dict(input_blend=str(tmp_path / "gb.blend"), output_blend=str(tmp_path / "gs.blend"),
                                     parts=parts), tag="gun_seg")
    assert seg["parts"] and seg["parts"][0]["watertight"] and seg["metrics"]["watertight"]


def test_spec_axes_and_per_axis_fit(runner, tmp_path):
    # A wall reconstructed lying on its back (height on Y) is stood up and fitted to 10 x 2.5 x 8 m
    lying = PrimitiveBackend(None, (10.0, 8.0, 2.5)).reconstruct("", "", tmp_path).path
    wall = runner.run("cleanup", cleanup_params(lying, tmp_path / "w.blend", dimensions=[10.0, 2.5, 8.0],
                                                fit_dimension="height", remesh_mode="voxel",
                                                import_rotation_deg=[0, 0, 0]), tag="wall_axes")
    assert any(wall["axis_permutation_deg"])
    for got, want in zip(wall["metrics"]["dimensions_m"], [10.0, 2.5, 8.0]):
        assert got == pytest.approx(want, rel=0.06)

    # A cube-ish chest scan is fitted per axis to the chest's real proportions
    (tmp_path / "c").mkdir()
    cube = PrimitiveBackend(None, (1.0, 0.95, 0.9)).reconstruct("", "", tmp_path / "c").path
    chest = runner.run("cleanup", cleanup_params(cube, tmp_path / "c.blend", dimensions=[0.9, 0.55, 0.6],
                                                 fit_dimension="length", remesh_mode="voxel",
                                                 import_rotation_deg=[0, 0, 0], per_axis_fit=True),
                       tag="chest_fit")
    for got, want in zip(chest["metrics"]["dimensions_m"], [0.9, 0.55, 0.6]):
        assert got == pytest.approx(want, rel=0.06)

    # Bodies are never re-proportioned or rotated onto their side
    body = runner.run("cleanup", cleanup_params(cube, tmp_path / "b.blend", dimensions=[0.55, 0.3, 1.8],
                                                fit_dimension="height", remesh_mode="voxel", organic=True,
                                                import_rotation_deg=[0, 0, 0], max_tilt_correction=0.0),
                      tag="body_fit")
    assert body.get("axis_permutation_deg") is None and body["per_axis_fit"] == [1.0, 1.0, 1.0]


def test_arms_touching_body_are_detected(runner, tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from make_mannequin import write_obj

    raw = write_obj(str(tmp_path / "i.obj"), "I")
    runner.run("cleanup", cleanup_params(raw, tmp_path / "i.blend", dimensions=[0.6, 0.3, 1.8], fit_dimension="height",
                                         remesh_mode="voxel", voxel_resolution=160, target_faces=6000, organic=True,
                                         import_rotation_deg=[0, 0, 0], max_tilt_correction=0.0, symmetrize=True),
               tag="i_clean")
    rig = runner.run("rig", dict(input_blend=str(tmp_path / "i.blend"), output_blend=str(tmp_path / "ir.blend"),
                                 rig_type="biped", method="generated"), tag="i_rig")
    lm = rig["landmarks"]
    assert not any(f.startswith("arm") for f in lm["fallback"]), lm
    assert rig["unweighted_vertices"] == 0


def test_unbaked_texels_are_filled(runner, raw_mesh, tmp_path):
    from asset_pipeline.shape_templates import expand_arrays, recipe_for

    runner.run("cleanup", cleanup_params(raw_mesh, tmp_path / "u.blend", remesh_mode="voxel"), tag="u_clean")
    mats = [{"name": "wood", "base_color": [0.4, 0.24, 0.11], "metallic": 0.0, "roughness": 0.7, "coverage": 1.0}]
    runner.run("primitives", dict(input_blend=str(tmp_path / "u.blend"), output_blend=str(tmp_path / "ut.blend"),
                                  components=[c.to_dict() for c in expand_arrays(recipe_for("treasure chest"))],
                                  mode="primitives", materials=mats), tag="u_tmpl")
    # A near-zero cage makes most selected-to-active rays miss the scan
    tex = runner.run("texture", dict(input_blend=str(tmp_path / "ut.blend"), output_blend=str(tmp_path / "ux.blend"),
                                     texture_dir=str(tmp_path / "utex"), resolution=128, uv_method="smart",
                                     uv_angle_limit=66, island_margin=0.01, bake_margin_px=2, cage_extrusion=1e-5,
                                     bake_samples=1, albedo_source="bake", materials=mats, bake_normal=False),
                     tag="u_tex")
    assert tex["unbaked_fraction"] > 0.05
    from PIL import Image
    import numpy as np

    albedo = np.asarray(Image.open(tex["textures"]["albedo"]).convert("RGB"), dtype=float) / 255
    # after filling, black is only left outside the UV islands (plus bake margin)
    coverage = sum(m.get("uv_coverage", 0) for m in tex["metrics"].values())
    assert (albedo.max(axis=-1) < 0.015).mean() <= (1 - coverage) + 0.02


def _broken_cube(path):
    """Unit cube with one face missing, one face duplicated and a fin hanging off an edge."""
    v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0.5, -0.8, 1.3)]
    faces = [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7),  # -X side left open
             (1, 2, 6, 5), (5, 6, 9)]  # duplicate face, fin on edge 5-6
    lines = [f"v {x} {y} {z}" for x, y, z in v] + ["f " + " ".join(map(str, f)) for f in faces]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize("mode", ["none", "decimate", "quadriflow"])
def test_cleanup_output_is_watertight(runner, tmp_path, mode):
    s3 = runner.run("cleanup", cleanup_params(_broken_cube(tmp_path / "broken.obj"), tmp_path / "b.blend",
                                              dimensions=[0.5, 0.5, 0.5], fit_dimension="height", remesh_mode=mode,
                                              fill_holes=False, fix_intersections=False, min_island_ratio=0.0,
                                              import_rotation_deg=[0, 0, 0]), tag=f"broken_{mode}")
    assert s3["repair"]["open_edges_before"] > 0 or mode == "quadriflow"
    assert s3["metrics"]["watertight"], s3["repair"]


def test_every_recipe_builds_watertight(runner, raw_mesh, tmp_path):
    from asset_pipeline.shape_templates import RECIPES, expand_arrays, recipe_for

    runner.run("cleanup", cleanup_params(raw_mesh, tmp_path / "r3.blend", remesh_mode="voxel"), tag="r_clean")
    for recipe in RECIPES:
        key = recipe["keywords"][0]
        comps = [c.to_dict() for c in expand_arrays(recipe_for(key))]
        res = runner.run("primitives", dict(input_blend=str(tmp_path / "r3.blend"),
                                            output_blend=str(tmp_path / "r.blend"), components=comps,
                                            mode="primitives", materials=[]), tag=f"recipe_{key.replace(' ', '_')}")
        open_ = {n: m["boundary_edges"] + m["non_manifold_edges"] for n, m in res["metrics"].items()
                 if not m["watertight"]}
        assert not open_, (key, open_, res.get("rebuilt_as_shells"))

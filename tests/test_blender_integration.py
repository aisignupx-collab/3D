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
    assert abs(s3["rotation_applied_deg"] + 30) < 2  # PCA undid the 30 degree import rotation
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

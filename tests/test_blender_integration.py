"""End-to-end Stages 2-6 with the proxy backend in real headless Blender.

Skipped unless Blender is found (BLENDER_PATH, PATH or default locations).
Uses small textures/renders so it finishes in about a minute on CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blender_automation import find_blender  # noqa: E402
from main import main  # noqa: E402

try:
    BLENDER = find_blender()
except FileNotFoundError:
    BLENDER = None

pytestmark = pytest.mark.skipif(BLENDER is None, reason="Blender not installed")


@pytest.mark.parametrize("prompt,fmt,dim_axis,expected", [
    ("A functional MP5 submachine gun", "GLB", 1, 0.68),
    ("A medieval knight character", "FBX", 2, 1.8),
])
def test_proxy_pipeline(tmp_path, prompt, fmt, dim_axis, expected):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"output_root: {tmp_path}\n"
        f"blender:\n  executable: {BLENDER}\n  texture_size: 512\n  render_resolution: 256\n"
        "  render_samples: 4\n  bake_samples: 2\n  target_faces: 4000\n  voxel_resolution: 96\n"
        "audit:\n  max_iterations: 2\n"
    )
    run_dir = tmp_path / "run"
    code = main(["--prompt", prompt, "--reconstruction_backend", "proxy", "--llm_provider", "none",
                 "--enable_rigging", "--export_parts", "--target_format", fmt,
                 "--config", str(cfg), "--output_dir", str(run_dir)])
    assert code in (0, 2)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert Path(manifest["output"]).is_file()
    last = manifest["iterations"][-1]
    m = last["metrics"]
    assert m["watertight"] and m["quad_ratio"] > 0.9 and m["self_intersections"] == 0
    assert m["dimensions_m"][dim_axis] == pytest.approx(expected, rel=0.05)
    assert last["uv_metrics"]["stretch_score"] < 0.45
    assert set(last["renders"]) >= {"front", "side", "wireframe", "shaded"}
    if "knight" in prompt:
        assert last["rig"]["weighted_vertex_fraction"] > 0.95
    else:
        assert "magazine" in last["parts"]
        assert len(manifest["stages"]["export"]["part_files"]) == len(last["parts"]) + 1

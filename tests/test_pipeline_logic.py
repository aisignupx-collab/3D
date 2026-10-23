"""Fast tests for the host-side logic (no GPU, no Blender, no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asset_spec import AssetSpec, heuristic_spec  # noqa: E402
from config import AuditConfig, LLMConfig, PipelineConfig, load_config  # noqa: E402
from llm_client import LLMClient, LLMUnavailable, extract_json  # noqa: E402
from main import apply_cli_overrides, parse_args  # noqa: E402
from pipeline import build_blender_params, stage5_plan  # noqa: E402
from reference_generator import (  # noqa: E402
    ConceptAgent, fill_workflow_placeholders, preprocess_for_reconstruction,
)
from vision_auditor import VisionAuditor, apply_corrective_actions, geometric_review  # noqa: E402


# -- asset spec ------------------------------------------------------------------

def test_heuristic_spec_mp5_scale_and_parts():
    spec = heuristic_spec("A functional MP5 submachine gun")
    assert spec.category == "weapon"
    assert spec.primary_axis == "y"
    assert spec.dimensions_m[1] == pytest.approx(0.68)
    assert {p.name for p in spec.moving_parts} >= {"magazine", "trigger"}


def test_heuristic_spec_castle_wall_human_scale():
    spec = heuristic_spec("Medieval Castle Wall")
    assert spec.category == "architecture"
    assert spec.dimensions_m[2] >= 6.0  # taller than several humans


def test_from_dict_coerces_sloppy_llm_output():
    spec = AssetSpec.from_dict({
        "category": "WEAPON", "dimensions_m": ["0.1", 0.7, 0.3], "primary_axis": "Y",
        "materials": [{"name": "steel", "base_color": [2, -1, 0.5], "roughness": 7}],
        "moving_parts": [{"name": "bolt carrier!", "region_min": [0.9, 0, 0], "region_max": [0.1, 1, 1],
                          "axis": "q", "motion": "slide"}],
        "rig_type": "dragon", "symmetry": "X",
    }, prompt="test")
    assert spec.category == "weapon" and spec.primary_axis == "y"
    assert spec.dimensions_m == [0.1, 0.7, 0.3]
    assert spec.materials[0].base_color == [1.0, 0.0, 0.5] and spec.materials[0].roughness == 1.0
    part = spec.moving_parts[0]
    assert part.name == "bolt_carrier" and part.region_min[0] == 0.1 and part.region_max[0] == 0.9
    assert part.axis == "X" and part.motion == "fixed"
    assert spec.rig_type == "none" and spec.symmetry == "x"


def test_concept_agent_falls_back_and_fixes_units():
    llm = LLMClient(LLMConfig(provider="none"))
    spec, source = ConceptAgent(llm).plan("wooden door")
    assert source == "heuristic" and spec.dimensions_m[2] == pytest.approx(2.1)

    # An LLM answering in millimetres gets corrected from the knowledge base.
    fake = mock.Mock(enabled=True)
    fake.chat_json.return_value = {"category": "architecture", "dimensions_m": [900, 50, 2100], "primary_axis": "z"}
    spec, source = ConceptAgent(fake).plan("wooden door")
    assert source == "llm" and spec.dimensions_m[2] == pytest.approx(2.1)


# -- LLM client --------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"a": 1}',
    'Sure! ```json\n{"a": 1}\n``` hope that helps',
    'The answer is {"a": 1} as requested.',
])
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_raises():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_openai_compatible_request_with_image(tmp_path):
    img = tmp_path / "r.png"
    Image.new("RGB", (64, 64), "red").save(img)
    client = LLMClient(LLMConfig(base_url="http://localhost:11434/v1", model="qwen2.5vl:7b"))
    resp = mock.Mock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": '{"approved": true}'}}]}
    with mock.patch("llm_client.requests.post", return_value=resp) as post:
        assert client.chat_json("sys", "user", images=[img]) == {"approved": True}
    url, body = post.call_args.args[0], post.call_args.kwargs["json"]
    assert url == "http://localhost:11434/v1/chat/completions"
    parts = body["messages"][1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["response_format"] == {"type": "json_object"}


def test_unreachable_llm_disables_itself():
    import requests

    client = LLMClient(LLMConfig(base_url="http://127.0.0.1:9/v1"))
    with mock.patch("llm_client.requests.post", side_effect=requests.ConnectionError("refused")) as post:
        with pytest.raises(LLMUnavailable):
            client.chat("s", "u")
        with pytest.raises(LLMUnavailable):
            client.chat("s", "u")
    assert post.call_count == 1 and not client.enabled


# -- Stage 1 image helpers ------------------------------------------------------

def test_fill_workflow_placeholders_keeps_types():
    wf = {"3": {"inputs": {"seed": "{{SEED}}", "text": "a {{PROMPT}}, white bg", "list": ["{{WIDTH}}"]}}}
    out = fill_workflow_placeholders(wf, {"SEED": 42, "PROMPT": "door", "WIDTH": 1024})
    assert out["3"]["inputs"] == {"seed": 42, "text": "a door, white bg", "list": [1024]}


def test_preprocess_centres_and_pads():
    img = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (100, 50), (255, 0, 0, 255)), (10, 20))
    rgba, rgb = preprocess_for_reconstruction(img, size=256, foreground_ratio=0.5)
    assert rgba.size == rgb.size == (256, 256)
    bbox = rgba.getchannel("A").point(lambda a: 255 if a > 127 else 0).getbbox()  # ignore resampling halo
    assert abs((bbox[0] + bbox[2]) / 2 - 128) <= 2 and abs((bbox[1] + bbox[3]) / 2 - 128) <= 2
    assert (bbox[2] - bbox[0]) == pytest.approx(128, abs=3)  # 50 % fill on the long side
    assert rgb.getpixel((0, 0)) == (128, 128, 128)


# -- Stage 6 audit & corrections -------------------------------------------------

def _result(**metrics):
    base = {"watertight": True, "non_manifold_edges": 0, "boundary_edges": 0, "self_intersections": 0,
            "normals_inverted": False, "faces": 10000, "dimensions_m": [0.06, 0.68, 0.26]}
    base.update(metrics)
    return {"metrics": base, "uv_metrics": {"has_uvs": True, "stretch_score": 0.1, "zero_area_uv_faces": 0}}


def test_geometric_review_passes_clean_asset():
    spec = heuristic_spec("MP5")
    params = build_blender_params(PipelineConfig(), spec, [0, 0, 0])
    issues, actions = geometric_review(_result(), spec, params, AuditConfig())
    assert issues == [] and actions == []


def test_geometric_review_flags_and_corrections_apply():
    spec = heuristic_spec("MP5")
    params = build_blender_params(PipelineConfig(), spec, [0, 0, 0])
    bad = _result(watertight=False, non_manifold_edges=12, self_intersections=4,
                  dimensions_m=[0.12, 0.5, 0.26])
    bad["uv_metrics"]["stretch_score"] = 0.9
    issues, actions = geometric_review(bad, spec, params, AuditConfig())
    names = {a["action"] for a in actions}
    assert {"make_watertight", "union_intersections", "compensate_scale", "fit_proportions",
            "reunwrap_uv"} <= names
    assert any(i.severity == "critical" for i in issues)

    new, applied = apply_corrective_actions(params, actions)
    assert applied
    assert new["nonuniform_fit"] is True
    assert new["fit_compensation"] == pytest.approx(0.68 / 0.5)
    assert new["uv_angle_limit_deg"] == 45  # first escalation step stays on Smart UV
    assert params["nonuniform_fit"] is False  # input not mutated


def test_uv_escalation_and_llm_actions():
    p = {"uv_method": "smart", "uv_angle_limit_deg": 45, "texture_size": 2048,
         "target_dimensions_m": [1, 1, 2], "primary_axis": "z", "extra_rotation_deg": [0, 0, 0], "cutters": []}
    p, _ = apply_corrective_actions(p, [{"action": "reunwrap_uv"}])
    assert p["uv_method"] == "seams"
    p, applied = apply_corrective_actions(p, [
        {"action": "rotate", "z_deg": 90},
        {"action": "rescale", "axis": "z", "target_m": 2.4},
        {"action": "increase_texture_resolution"},
        {"action": "boolean_difference", "shape": "cylinder", "center_norm": [0.5, 0.0, 0.8], "size_norm": [0.1, 0.2, 0.1]},
        {"action": "nonsense"},
    ])
    assert p["extra_rotation_deg"] == [0, 0, 90] and p["auto_orient"] is False
    assert p["target_dimensions_m"] == [1, 1, 2.4]
    assert p["texture_size"] == 4096 and len(p["cutters"]) == 1
    assert len(applied) >= 5
    _, applied = apply_corrective_actions(p, [{"action": "increase_texture_resolution"}])
    assert applied == []  # already at the 4K cap: no progress -> loop stops


def test_vision_auditor_merges_llm_verdict(tmp_path):
    spec = heuristic_spec("MP5")
    params = build_blender_params(PipelineConfig(), spec, [0, 0, 0])
    renders = {}
    for name in ("front", "side"):
        renders[name] = str(tmp_path / f"{name}.png")
        Image.new("RGB", (32, 32)).save(renders[name])
    result = {**_result(), "renders": renders}
    vision = mock.Mock(enabled=True)
    vision.chat_json.return_value = {
        "approved": False, "score": 0.4,
        "issues": [{"category": "orientation", "severity": "major", "description": "lying on its side"}],
        "actions": [{"action": "rotate", "x_deg": 90}, {"action": "launch_rockets"}],
    }
    report = VisionAuditor(vision, AuditConfig()).audit(1, result, spec, params, tmp_path / "renders")
    assert not report.approved and report.vision_used and report.score == 0.4
    assert report.actions == [{"action": "rotate", "x_deg": 90}]
    assert Path(report.contact_sheet).is_file()
    sent_images = vision.chat_json.call_args.kwargs["images"]
    assert len(sent_images) == 1  # contact sheet mode

    vision.chat_json.return_value = {"approved": True, "score": 0.9, "issues": [], "actions": []}
    assert VisionAuditor(vision, AuditConfig()).audit(2, result, spec, params, tmp_path / "r2").approved


# -- orchestration helpers -----------------------------------------------------------

def test_stage5_plan():
    assert stage5_plan(heuristic_spec("MP5"), True) == {"segment": True, "rig": False}
    knight = heuristic_spec("a knight character")
    assert stage5_plan(knight, True) == {"segment": False, "rig": True}
    assert stage5_plan(knight, False) == {"segment": False, "rig": False}


def test_build_blender_params_rotation_precedence():
    cfg = PipelineConfig()
    spec = heuristic_spec("MP5")
    assert build_blender_params(cfg, spec, [0, 0, -90])["import_rotation_deg"] == [0, 0, -90]
    cfg.blender.import_rotation_deg = [90, 0, 0]
    p = build_blender_params(cfg, spec, [0, 0, -90])
    assert p["import_rotation_deg"] == [90, 0, 0]
    assert "executable" not in p and p["target_dimensions_m"] == spec.dimensions_m
    json.dumps(p)  # must be serializable into the Blender job file


def test_cli_and_config(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("blender:\n  texture_size: 1024\naudit:\n  max_iterations: 5\n")
    monkeypatch.setenv("BLENDER_PATH", "/opt/blender/blender")
    args = parse_args(["--prompt", "door", "--target_format", "fbx", "--enable_rigging",
                       "--config", str(cfg_file), "--llm_provider", "none", "--texture_size", "4096"])
    assert args.target_format == "FBX" and args.enable_rigging
    cfg = apply_cli_overrides(load_config(args.config), args)
    assert cfg.blender.texture_size == 4096 and cfg.audit.max_iterations == 5
    assert cfg.blender.executable == "/opt/blender/blender"
    assert cfg.llm.provider == cfg.vision.provider == "none"
    with pytest.raises(SystemExit):
        parse_args([])
    bad = tmp_path / "bad.yaml"
    bad.write_text("blender:\n  texture_sise: 1\n")
    with pytest.raises(ValueError, match="texture_sise"):
        load_config(bad)


def test_reference_generator_text_path(tmp_path):
    from config import DiffusionConfig, ReconstructionConfig
    from reference_generator import ImageGenerator, ReferenceGenerator

    class FakeGenerator(ImageGenerator):
        prompts: list[str] = []
        released = False

        def generate(self, prompt, negative, seed):
            self.prompts.append(prompt)
            img = Image.new("RGB", (256, 256), "white")
            img.paste(Image.new("RGB", (60, 120), "red"), (40, 30))
            return img

        def release(self):
            FakeGenerator.released = True

    def fake_isolate(img):  # stand-in for rembg: white -> transparent
        rgba = img.convert("RGBA")
        rgba.putalpha(img.convert("L").point(lambda v: 0 if v == 255 else 255))
        return rgba

    gen = FakeGenerator()
    refgen = ReferenceGenerator(DiffusionConfig(), ReconstructionConfig(input_size=128),
                                ConceptAgent(LLMClient(LLMConfig(provider="none"))))
    refgen._generator = gen
    refgen.isolator = fake_isolate
    ref = refgen.run("medieval castle wall", None, tmp_path)
    assert set(ref.views) == {"front", "side", "isometric"}
    assert len(gen.prompts) == 3 and all("white background" in p for p in gen.prompts)
    assert "three-quarter" in gen.prompts[2]
    assert FakeGenerator.released
    assert Image.open(ref.primary_input).size == (128, 128)
    assert ref.concept_sheet.is_file()
    assert json.loads((tmp_path / "asset_spec.json").read_text())["category"] == "architecture"

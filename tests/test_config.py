import pytest

from asset_pipeline.config import Config, load_config


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.reconstruction.backend == "triposr"
    assert cfg.cleanup.target_faces == 0


def test_yaml_and_env_override(tmp_path, monkeypatch):
    path = tmp_path / "c.yaml"
    path.write_text("texture:\n  resolution: 4096\nblender:\n  use_gpu: false\n")
    monkeypatch.setenv("ASSET_PIPELINE_CLEANUP__VOXEL_RESOLUTION", "256")
    monkeypatch.setenv("ASSET_PIPELINE_BLENDER__USE_GPU", "true")
    cfg = load_config(str(path))
    assert cfg.texture.resolution == 4096
    assert cfg.cleanup.voxel_resolution == 256
    assert cfg.blender.use_gpu is True  # env beats YAML


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("texture:\n  resolutoin: 4096\n")
    with pytest.raises(ValueError, match="resolutoin"):
        load_config(str(path))


def test_missing_explicit_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.yaml"))


def test_example_config_matches_schema():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "config.example.yaml"
    load_config(str(example))


def test_rtx4090_profile_matches_schema():
    from pathlib import Path

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "rtx4090.yaml"))
    assert cfg.texture.resolution == 4096 and cfg.reconstruction.device == "cuda:0"

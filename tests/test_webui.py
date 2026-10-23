import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import webui
from asset_pipeline.config import Config


def test_schema_covers_every_setting_with_help():
    sections = {s["name"]: s for s in webui.schema(Config())}
    assert set(sections) == set(Config().to_dict())
    cleanup = {f["key"]: f for f in sections["cleanup"]["fields"]}
    assert cleanup["remesh_mode"]["options"][0] == "quadriflow"
    assert "voxel" in cleanup["voxel_resolution"]["help"]
    assert {f["key"]: f for f in sections["blender"]["fields"]}["use_gpu"]["type"] == "bool"


def test_validated_coerces_and_rejects(tmp_path):
    cfg = webui.validated({"texture": {"resolution": "4096", "project_reference": 0},
                           "diffusion": {"views": "[front, side]"}})
    assert cfg.texture.resolution == 4096 and cfg.texture.project_reference is False
    assert cfg.diffusion.views == ["front", "side"]
    with pytest.raises(ValueError):
        webui.validated({"texture": {"no_such_key": 1}})
    webui.write_config(cfg, tmp_path / "config.yaml")
    webui.write_config(cfg, tmp_path / "config.yaml")
    assert (tmp_path / "config.yaml.bak").is_file()
    from asset_pipeline.config import load_config
    assert load_config(str(tmp_path / "config.yaml")).texture.resolution == 4096


class _FakeLMStudio(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        data = {"/api/v0/models": {"data": [{"id": "qwen2.5-14b-instruct", "type": "llm", "state": "loaded"},
                                            {"id": "qwen2.5-vl-7b", "type": "vlm", "state": "not-loaded"},
                                            {"id": "nomic-embed", "type": "embeddings"}]}}.get(self.path)
        body = json.dumps(data).encode()
        self.send_response(200 if data else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_lm_studio_models_are_listed_with_vision_flags():
    srv = HTTPServer(("127.0.0.1", 0), _FakeLMStudio)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = webui.list_models("lmstudio", f"http://127.0.0.1:{srv.server_port}/v1")
    finally:
        srv.shutdown()
    assert r["base_url"].endswith("/v1")
    assert [(m["id"], m["vision"], m["loaded"]) for m in r["models"]] == [
        ("qwen2.5-14b-instruct", False, True), ("qwen2.5-vl-7b", True, False)]


def test_job_command_line(tmp_path):
    cfg = Config()
    cfg.pipeline.output_dir = str(tmp_path)
    jobs = webui.Jobs.__new__(webui.Jobs)  # no worker thread
    jobs.jobs, jobs.order, jobs.lock, jobs.wake = {}, [], threading.Lock(), threading.Event()
    job = jobs.submit(cfg, {"prompt": "a castle", "target_format": "fbx", "enable_rigging": True,
                            "rig_method": "generated", "dry_run": True}, {"DEEPSEEK_API_KEY": "sk-x"})
    cmd = job["cmd"]
    assert cmd[cmd.index("--prompt") + 1] == "a castle" and "--dry_run" in cmd and "--enable_rigging" in cmd
    assert cmd[cmd.index("--target_format") + 1] == "fbx"
    cfg_file = cmd[cmd.index("--config") + 1]
    assert "sk-x" not in open(cfg_file).read()  # keys go to the environment, never to disk
    assert job["_env"] == {"DEEPSEEK_API_KEY": "sk-x"}
    public = jobs.public(job)
    assert "_env" not in public and "cmd" not in public


def test_progress_parsing():
    jobs = webui.Jobs.__new__(webui.Jobs)
    job = {"progress": 0.0, "max_iterations": 2, "stage": "", "run_dir": None}
    for msg in ["Run directory: /x/run1", "STAGE 1 - concept", "STAGE 2 - image-to-3D", "ITERATION 1",
                "STAGE 3 - topology", "STAGE 6 - audit"]:
        jobs._progress(job, msg)
    assert job["run_dir"] == "/x/run1" and job["stage"].startswith("STAGE 6")
    assert 0.6 < job["progress"] < 0.7

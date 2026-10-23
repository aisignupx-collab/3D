#!/usr/bin/env python3
"""Local web UI: every setting, local/cloud model pickers and runs with live progress.

  python webui.py                    # opens http://127.0.0.1:7860 in your browser
  python webui.py --port 8080 --no-browser
  python webui.py --config configs/rtx4090.yaml
  python webui.py --host 0.0.0.0     # reachable from your LAN (no login: trusted networks only)

Standard library only. Runs are started as ``python main.py`` subprocesses, one at a
time because they share the GPU, so a crash or a cancel never takes the UI down and VRAM is
freed after every run. API keys typed into the page live in this process's memory only and
reach runs as environment variables; they are never written to disk.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from asset_pipeline import config as config_module  # noqa: E402
from asset_pipeline.config import Config, _apply_mapping, find_blender, load_config  # noqa: E402
from asset_pipeline.llm_client import PRESETS, LLMClient  # noqa: E402

# Choices for string settings (anything else is free text)
OPTIONS: Dict[str, List[str]] = {
    "llm.provider": ["lmstudio", "ollama", "openai_compatible", "vllm", "deepseek", "openrouter", "gemini", "openai",
                     "anthropic", "cli"],
    "diffusion.backend": ["diffusers", "comfyui"],
    "diffusion.pipeline_type": ["auto", "sdxl", "flux", "sd"],
    "preprocess.rembg_model": ["isnet-general-use", "u2net", "birefnet-general"],
    "reconstruction.backend": ["triposr", "instantmesh", "openlrm", "command", "primitive"],
    "reconstruction.device": ["cuda:0", "cuda:1", "cpu"],
    "cleanup.remesh_mode": ["quadriflow", "voxel", "voxel_shrinkwrap", "decimate", "none"],
    "shapes.mode": ["auto", "primitives", "mesh"],
    "texture.uv_method": ["smart", "seams", "lightmap"],
    "texture.albedo_source": ["bake", "projection", "procedural"],
    "texture.unseen_fill": ["tiles", "bake"],
    "texture.resolution": ["1024", "2048", "4096"],
    "render.engine": ["auto", "eevee", "cycles", "workbench"],
}
OPTIONS["vision.provider"] = OPTIONS["llm.provider"]
KEY_ENVS = sorted({env for _, env in PRESETS.values() if env} | {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"})
STAGE_PROGRESS = {"STAGE 1": 0.03, "STAGE 2": 0.18}
ITER_STAGES = ("STAGE 3 ", "STAGE 3b", "STAGE 5", "STAGE 4", "STAGE 6")


# ----------------------------------------------------------------- settings schema

def _field_help() -> Dict[str, Dict[str, str]]:
    """Help text per section field, read from the comments in asset_pipeline/config.py."""
    section_of = {type(getattr(Config(), f.name)).__name__: f.name for f in dataclasses.fields(Config)}
    out: Dict[str, Dict[str, str]] = {}
    current, pending = None, []
    for line in Path(config_module.__file__).read_text(encoding="utf-8").splitlines():
        m = re.match(r"class (\w+):", line)
        if m:
            current, pending = section_of.get(m.group(1)), []
            continue
        if current is None:
            continue
        comment = re.match(r"\s{4}#\s?(.*)", line)
        if comment:
            pending.append(comment.group(1).strip())
            continue
        m = re.match(r"\s{4}(\w+):[^=]+=(.*)", line)
        if m:
            trailing = m.group(2).split("  # ", 1)[1].strip() if "  # " in m.group(2) else ""
            out.setdefault(current, {})[m.group(1)] = " ".join(pending + [trailing]).strip()
        pending = [] if not comment else pending
    return out


def schema(cfg: Config) -> List[Dict[str, Any]]:
    helps = _field_help()
    defaults = Config()
    sections = []
    for f in dataclasses.fields(Config):
        section, default_section = getattr(cfg, f.name), getattr(defaults, f.name)
        fields = []
        for sf in dataclasses.fields(section):
            value, default = getattr(section, sf.name), getattr(default_section, sf.name)
            kind = ("bool" if isinstance(default, bool) else "int" if isinstance(default, int)
                    else "float" if isinstance(default, float) else "list" if isinstance(default, list) else "str")
            fields.append({"key": sf.name, "type": kind, "value": value, "default": default,
                           "help": helps.get(f.name, {}).get(sf.name, ""),
                           "options": OPTIONS.get(f"{f.name}.{sf.name}")})
        doc = (type(section).__doc__ or "").strip()
        doc = "" if doc.startswith(type(section).__name__ + "(") else doc  # dataclass auto-docstring
        sections.append({"name": f.name, "doc": doc, "fields": fields})
    return sections


def validated(values: Dict[str, Any]) -> Config:
    """A Config from UI values; raises ValueError on unknown keys or bad types."""
    cfg = Config()
    defaults = cfg.to_dict()
    clean: Dict[str, Dict[str, Any]] = {}
    for section, items in values.items():
        clean[section] = {}
        for key, value in (items or {}).items():
            default = defaults.get(section, {}).get(key)
            try:
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int):
                    value = int(float(value))
                elif isinstance(default, float):
                    value = float(value)
                elif isinstance(default, list) and isinstance(value, str):
                    value = yaml.safe_load(value) if value.strip() else []
                elif isinstance(default, str):
                    value = "" if value is None else str(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{section}.{key}: {exc}") from exc
            clean[section][key] = value
    _apply_mapping(cfg, clean, "web UI")
    return cfg


def write_config(cfg: Config, path: Path):
    if path.is_file():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    header = ("# Written by webui.py (previous version: *.bak). Every key is optional; defaults live in\n"
              "# asset_pipeline/config.py. API keys are never stored here - use environment variables.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def output_dir(cfg: Config) -> Path:
    """The run folder; relative paths are relative to the repo, where main.py runs."""
    out = Path(cfg.pipeline.output_dir)
    return out if out.is_absolute() else ROOT / out


# ----------------------------------------------------------------- model discovery

def _root(base_url: str) -> str:
    return re.sub(r"/(v1|api)/?$", "", base_url.rstrip("/"))


def list_models(provider: str, base_url: str, api_key: str = "") -> Dict[str, Any]:
    """Models a server offers, flagged ``vision`` where the server says so."""
    provider = provider.lower()
    if provider in PRESETS and not base_url:
        base_url = PRESETS[provider][0]
    if provider == "ollama":
        base = _root(base_url or "http://localhost:11434")
        tags = requests.get(f"{base}/api/tags", timeout=4).json().get("models", [])
        models = []
        for t in tags[:40]:
            caps = None
            try:
                caps = requests.post(f"{base}/api/show", json={"model": t["name"]}, timeout=4).json().get(
                    "capabilities")
            except (requests.RequestException, ValueError):
                pass
            models.append({"id": t["name"], "size_gb": round(t.get("size", 0) / 1e9, 1),
                           "vision": ("vision" in caps) if caps is not None else None})
        return {"models": models, "base_url": base}
    if provider == "lmstudio":
        base = _root(base_url or "http://localhost:1234")
        try:  # LM Studio's own API says which models are VLMs and which are loaded
            data = requests.get(f"{base}/api/v0/models", timeout=4).json().get("data", [])
            return {"base_url": f"{base}/v1", "models": [
                {"id": m["id"], "vision": m.get("type") == "vlm", "loaded": m.get("state") == "loaded",
                 "quantization": m.get("quantization")} for m in data if m.get("type") != "embeddings"]}
        except (requests.RequestException, ValueError):
            pass
        base_url = f"{base}/v1"
    if provider in ("anthropic", "cli"):
        return {"models": [], "note": "this provider does not list models; type the model name"}
    url = base_url.rstrip("/")
    url = url if url.endswith(("/v1", "/openai")) or provider in PRESETS else f"{url}/v1"
    if provider == "openai" and not base_url:
        url = "https://api.openai.com/v1"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = requests.get(f"{url}/models", headers=headers, timeout=6).json().get("data", [])
    return {"models": [{"id": m["id"], "vision": None} for m in data], "base_url": url}


# ----------------------------------------------------------------- jobs

class Jobs:
    """FIFO queue of ``main.py`` runs; one runs at a time (they share the GPU)."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.order: List[str] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, cfg: Config, options: Dict[str, Any], keys: Dict[str, str]) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:8]
        work = output_dir(cfg) / ".webui"
        work.mkdir(parents=True, exist_ok=True)
        cfg_path = work / f"job_{job_id}.yaml"
        write_config(cfg, cfg_path)
        cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--config", str(cfg_path),
               "--target_format", options.get("target_format", "glb")]
        for flag in ("prompt", "image_input", "resume", "rig_method"):
            if options.get(flag):
                cmd += [f"--{flag}", str(options[flag])]
        if float(options.get("print_clearance_mm") or 0) > 0:
            cmd += ["--print_clearance_mm", str(options["print_clearance_mm"])]
        for flag in ("enable_rigging", "offline", "dry_run"):
            if options.get(flag):
                cmd.append(f"--{flag}")
        job = {"id": job_id, "status": "queued", "options": options, "cmd": cmd, "created": time.time(),
               "log": deque(maxlen=4000), "lines": 0, "stage": "queued", "progress": 0.0, "run_dir": None,
               "summary": None, "max_iterations": cfg.pipeline.max_iterations,
               "_env": {k: v for k, v in keys.items() if v}}
        with self.lock:
            self.jobs[job_id] = job
            self.order.append(job_id)
        self.wake.set()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job["status"] == "queued":
            job["status"] = "cancelled"
            return True
        proc = job.get("_proc")
        if proc is not None and proc.poll() is None:
            job["status"] = "cancelling"
            if os.name == "nt":  # the whole tree, Blender included
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            return True
        return False

    def public(self, job: Dict[str, Any], since: int = 0) -> Dict[str, Any]:
        lines = list(job["log"])
        first = job["lines"] - len(lines)
        tail = lines[max(0, since - first):] if since else lines[-400:]
        out = {k: v for k, v in job.items() if not k.startswith("_") and k not in ("log", "cmd")}
        out.update(log=tail, lines=job["lines"])
        return out

    def _progress(self, job, msg: str):
        for key, value in STAGE_PROGRESS.items():
            if msg.startswith(key):
                job["progress"] = max(job["progress"], value)
        m = re.match(r"ITERATION (\d+)", msg)
        if m:
            job["iteration"] = int(m.group(1))
        it = job.get("iteration")
        if it:
            stage = next((i for i, s in enumerate(ITER_STAGES) if msg.startswith(s.strip()) and
                          (s != "STAGE 3 " or not msg.startswith("STAGE 3b"))), None)
            if stage is not None:
                frac = (it - 1 + (stage + 1) / len(ITER_STAGES)) / max(1, job["max_iterations"])
                job["progress"] = max(job["progress"], 0.3 + 0.65 * min(frac, 1.0))
        if msg.startswith(("STAGE", "ITERATION", "Exported", "Approved", "Resume")):
            job["stage"] = msg[:160]
        if msg.startswith("Run directory: "):
            job["run_dir"] = msg.split(": ", 1)[1].strip()

    def _run(self, job):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **job["_env"])
        popen = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        job.update(status="running", started=time.time(), stage="starting")
        proc = subprocess.Popen(job["cmd"], cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                bufsize=1, **popen)
        job["_proc"] = proc
        tail_json: List[str] = []
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            # progress bars rewrite one line with \r; keep the last state only
            line = line.split("\r")[-1]
            job["log"].append(line)
            job["lines"] += 1
            msg = re.sub(r"^\d\d:\d\d:\d\d \w+\s+[\w.]+: ", "", line)
            self._progress(job, msg)
            if line.startswith("{") or tail_json:
                tail_json.append(line)
                if line.startswith("}"):
                    try:
                        job["summary"] = json.loads("\n".join(tail_json))
                        job["run_dir"] = job["summary"].get("run_dir", job["run_dir"])
                    except ValueError:
                        pass
                    tail_json = []
        job["returncode"] = proc.wait()
        cancelled = job["status"] == "cancelling"
        job.update(status="cancelled" if cancelled else "done" if job["returncode"] == 0 else "error",
                   finished=time.time(), progress=1.0 if job["returncode"] == 0 else job["progress"])
        if job["status"] == "done":
            job["stage"] = "finished"

    def _worker(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            while True:
                with self.lock:
                    nxt = next((self.jobs[j] for j in self.order if self.jobs[j]["status"] == "queued"), None)
                if nxt is None:
                    break
                try:
                    self._run(nxt)
                except Exception as exc:  # never kill the worker thread
                    nxt.update(status="error", stage=f"could not start: {exc}", finished=time.time())


# ----------------------------------------------------------------- runs on disk

def _rel(path: Path, base: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return None


def list_runs(out_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    for manifest in sorted(out_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run = manifest.parent
        export = m.get("export") or {}
        preview = next(iter(sorted((run / "export" / "previews").glob("shaded.png"))), None) or next(
            iter(sorted(run.glob("renders/iter_*/shaded.png"))), None)
        runs.append({"name": run.name, "prompt": (m.get("options") or {}).get("prompt") or
                     (m.get("options") or {}).get("image_input"), "modified": manifest.stat().st_mtime,
                     "approved": export.get("approved"), "score": export.get("score"),
                     "iterations": len(m.get("iterations") or []), "category": (m.get("spec") or {}).get("category"),
                     "preview": _rel(preview, out_dir) if preview else None})
    return runs


def run_detail(out_dir: Path, name: str) -> Dict[str, Any]:
    run = (out_dir / name).resolve()
    if _rel(run, out_dir) is None or not (run / "manifest.json").is_file():
        raise FileNotFoundError(name)
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    iterations = []
    for it in m.get("iterations") or []:
        audit = it.get("audit") or {}
        renders = sorted(p for p in (run / "renders" / f"iter_{it.get('iteration')}").glob("**/*.png"))
        tmpl = it.get("stage3b") or {}
        iterations.append({
            "iteration": it.get("iteration"), "approved": audit.get("approved"), "score": audit.get("score"),
            "blocking": audit.get("blocking"), "unverified": audit.get("unverified"),
            "issues": [f"[{i.get('severity')}] {i.get('description')}" for i in audit.get("issues", [])],
            "fixes": it.get("fixes"), "renders": [_rel(p, out_dir) for p in renders],
            "template": {k: tmpl.get(k) for k in ("used", "source", "fit_error", "skyline")} if tmpl else None,
            "texture": {k: (it.get("stage4") or {}).get(k) for k in ("albedo", "material_tiles", "unbaked_fraction")},
        })
    export_dir = run / "export"
    files = [{"name": p.name, "path": _rel(p, out_dir), "bytes": p.stat().st_size}
             for p in sorted(export_dir.rglob("*")) if p.is_file() and p.suffix.lower() not in (".blend1",)]
    model = next((f["path"] for f in files if f["name"].lower().endswith(".glb")), None)
    return {"name": name, "dir": str(run), "spec": m.get("spec"), "options": m.get("options"),
            "export": m.get("export"), "timings": m.get("timings"), "iterations": iterations, "files": files,
            "glb": model}


# ----------------------------------------------------------------- system info

_TORCH_INFO: Dict[str, Any] = {}


def system_info(cfg: Config) -> Dict[str, Any]:
    info: Dict[str, Any] = {"gpus": [], "python": sys.version.split()[0], "platform": sys.platform}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
                              "temperature.gpu,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.strip().splitlines():
            name, used, total, util, temp, driver = [x.strip() for x in line.split(",")]
            info["gpus"].append({"name": name, "used_mb": int(used), "total_mb": int(total), "util": int(util),
                                 "temp_c": int(temp), "driver": driver})
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if not _TORCH_INFO:
        try:
            out = subprocess.run([sys.executable, "-c", "import json, torch; print(json.dumps({'torch': "
                                  "torch.__version__, 'cuda': torch.version.cuda, 'available': "
                                  "torch.cuda.is_available()}))"], capture_output=True, text=True, timeout=120)
            _TORCH_INFO.update(json.loads(out.stdout.strip().splitlines()[-1]))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            _TORCH_INFO["torch"] = None
    info["torch"] = _TORCH_INFO
    try:
        info["blender"] = find_blender(cfg.blender.executable)
    except FileNotFoundError as exc:
        info["blender"], info["blender_error"] = None, str(exc)
    servers = {}
    for name, url in (("LM Studio", "http://localhost:1234/v1/models"), ("Ollama", "http://localhost:11434/api/tags"),
                      ("ComfyUI", cfg.diffusion.comfyui_url.rstrip("/") + "/system_stats")):
        try:
            servers[name] = requests.get(url, timeout=1.5).ok
        except requests.RequestException:
            servers[name] = False
    info["servers"] = servers
    info["keys"] = {k: bool(os.environ.get(k)) for k in KEY_ENVS}
    return info


# ----------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "AssetPipelineUI/1.0"
    app: "App" = None  # set by serve()

    def log_message(self, fmt, *args):  # quiet: one line per error only
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("webui: " + fmt % args + "\n")

    # --- helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, code: int = 200):
        self._send(code, json.dumps(data, default=str).encode("utf-8"), "application/json")

    def _error(self, code: int, message: str):
        self._json({"error": message}, code)

    def _host_ok(self) -> bool:
        if self.app.public:
            return True
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1")  # blocks DNS-rebinding pages

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 80 * 1024 * 1024:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    # --- routes
    def do_GET(self):
        if not self._host_ok():
            return self._error(403, "forbidden host")
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        app = self.app
        try:
            if url.path == "/":
                return self._send(200, page().encode("utf-8"), "text/html; charset=utf-8")
            if url.path == "/api/state":
                cfg = app.config()
                return self._json({"config_path": str(app.config_path), "schema": schema(cfg),
                                   "output_dir": str(output_dir(cfg).resolve()),
                                   "presets": {k: v[0] for k, v in PRESETS.items()}, "key_envs": KEY_ENVS,
                                   "session_keys": sorted(app.keys)})
            if url.path == "/api/models":
                provider = q.get("provider", "")
                key_env = q.get("key_env") or PRESETS.get(provider, ("", ""))[1] or {
                    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "")
                key = app.keys.get(key_env) or os.environ.get(key_env, "")
                return self._json(list_models(q.get("provider", ""), q.get("base_url", ""), key))
            if url.path == "/api/system":
                return self._json(system_info(app.config()))
            if url.path == "/api/jobs":
                return self._json([app.jobs.public(app.jobs.jobs[j], 10 ** 9) for j in reversed(app.jobs.order)])
            m = re.fullmatch(r"/api/jobs/(\w+)", url.path)
            if m:
                job = app.jobs.jobs.get(m.group(1))
                return self._json(app.jobs.public(job, int(q.get("since", 0)))) if job else self._error(404, "no job")
            if url.path == "/api/runs":
                return self._json(list_runs(app.out_dir()))
            m = re.fullmatch(r"/api/runs/([^/]+)", url.path)
            if m:
                return self._json(run_detail(app.out_dir(), m.group(1)))
            if url.path == "/files":
                return self._file(q.get("p", ""), download=q.get("dl") == "1")
            return self._error(404, "not found")
        except requests.RequestException as exc:
            return self._error(502, f"server not reachable: {exc}")
        except FileNotFoundError as exc:
            return self._error(404, f"not found: {exc}")
        except Exception as exc:  # report, don't crash the server
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def _file(self, rel: str, download: bool):
        base = self.app.out_dir().resolve()
        path = (base / rel).resolve()
        if _rel(path, base) is None or not path.is_file():
            return self._error(404, "no such file")
        ctype = {".png": "image/png", ".jpg": "image/jpeg", ".glb": "model/gltf-binary", ".json": "application/json",
                 ".obj": "text/plain", ".mtl": "text/plain"}.get(path.suffix.lower(), "application/octet-stream")
        extra = {"Content-Disposition": f'attachment; filename="{path.name}"'} if download else None
        self._send(200, path.read_bytes(), ctype, extra)

    def do_POST(self):
        if not self._host_ok():
            return self._error(403, "forbidden host")
        if self.headers.get("X-Asset-UI") != "1":  # cross-site forms can't set custom headers
            return self._error(403, "missing X-Asset-UI header")
        app = self.app
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/config":
                cfg = validated(body.get("values") or {})
                write_config(cfg, app.config_path)
                return self._json({"saved": str(app.config_path)})
            if url.path == "/api/keys":  # session-only API keys
                for name, value in (body.get("keys") or {}).items():
                    if name in KEY_ENVS or re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", name or ""):
                        if value:
                            app.keys[name] = value
                        else:
                            app.keys.pop(name, None)
                return self._json({"session_keys": sorted(app.keys)})
            if url.path == "/api/ping":
                return self._json(app.ping(body.get("section", "llm"), body.get("values") or {}, body.get("chat")))
            if url.path == "/api/upload":
                return self._json(app.upload(body.get("name", "image.png"), body.get("data", "")))
            if url.path == "/api/jobs":
                cfg = validated(body.get("values") or {})
                opts = body.get("options") or {}
                if not (str(opts.get("prompt") or "").strip() or opts.get("image_input") or opts.get("resume")):
                    return self._error(400, "enter a prompt, add an image, or pick a run to resume")
                opts["prompt"] = str(opts.get("prompt") or "").strip()
                job = app.jobs.submit(cfg, opts, app.keys)
                return self._json(app.jobs.public(job))
            m = re.fullmatch(r"/api/jobs/(\w+)/cancel", url.path)
            if m:
                return self._json({"cancelled": app.jobs.cancel(m.group(1))})
            if url.path == "/api/doctor":
                cfg = validated(body.get("values") or {})
                tmp = output_dir(cfg) / ".webui" / "doctor.yaml"
                write_config(cfg, tmp)
                out = subprocess.run([sys.executable, str(ROOT / "main.py"), "--doctor", "--config", str(tmp)],
                                     cwd=str(ROOT), capture_output=True, text=True, timeout=600,
                                     env=dict(os.environ, **app.keys))
                return self._json({"output": (out.stdout + out.stderr)[-20000:], "code": out.returncode})
            return self._error(404, "not found")
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception as exc:
            return self._error(500, f"{type(exc).__name__}: {exc}")


class App:
    def __init__(self, config_path: Path, public: bool):
        self.config_path = config_path
        self.public = public
        self.keys: Dict[str, str] = {}
        self.jobs = Jobs()
        self._env_lock = threading.Lock()

    def config(self) -> Config:
        return load_config(str(self.config_path)) if self.config_path.is_file() else Config()

    def out_dir(self) -> Path:
        return output_dir(self.config())

    def upload(self, name: str, data: str) -> Dict[str, Any]:
        raw = base64.b64decode(data.split(",", 1)[-1])
        if len(raw) > 60 * 1024 * 1024:
            raise ValueError("image larger than 60 MB")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[-80:] or "image.png"
        dest = self.out_dir() / "uploads" / f"{uuid.uuid4().hex[:6]}_{safe}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        return {"path": str(dest), "rel": _rel(dest, self.out_dir())}

    def ping(self, section: str, values: Dict[str, Any], chat: bool) -> Dict[str, Any]:
        cfg = validated({section: values})
        sec = getattr(cfg, section)
        with self._env_lock:  # the client reads keys from the environment
            saved = {k: os.environ.get(k) for k in self.keys}
            os.environ.update(self.keys)
            try:
                client = LLMClient.from_config(sec)
                t0 = time.time()
                ok = client.ping()
                result: Dict[str, Any] = {"reachable": ok, "base_url": client.base_url,
                                          "seconds": round(time.time() - t0, 2)}
                if ok and chat:
                    t0 = time.time()
                    try:
                        reply = client.chat("Reply with the single word OK.")
                        result.update(reply=reply.strip()[:200], chat_seconds=round(time.time() - t0, 2))
                    except Exception as exc:  # model name wrong, not loaded, ...
                        result["chat_error"] = str(exc)[:500]
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        return result


def serve(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--config", default="config.yaml", help="settings file the page loads and saves")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else Path.cwd() / config_path
    public = args.host not in ("127.0.0.1", "localhost", "::1")
    Handler.app = App(config_path, public)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'127.0.0.1' if not public else args.host}:{args.port}/"
    print(f"Asset pipeline UI on {url}  (settings: {config_path}; Ctrl+C to stop)")
    if public:
        print("WARNING: listening beyond localhost with no login - anyone on the network can start runs.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job_id in list(Handler.app.jobs.jobs):
            Handler.app.jobs.cancel(job_id)
    return 0


def page() -> str:
    path = ROOT / "asset_pipeline" / "webui.html"
    return path.read_text(encoding="utf-8") if path.is_file() else "<h1>asset_pipeline/webui.html is missing</h1>"


if __name__ == "__main__":
    sys.exit(serve())

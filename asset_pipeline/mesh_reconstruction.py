"""Stage 2 - image-to-3D mesh extraction.

Backends
  ``triposr``     in-process PyTorch/CUDA (fast, ~6 GB VRAM, vertex colours)
  ``instantmesh`` subprocess into its own venv (Zero123++ multi-view + LRM,
                  ~16 GB VRAM, UV texture map)
  ``openlrm``     subprocess into its own venv (vertex-coloured PLY)
  ``command``     any CLI model (TRELLIS, Hunyuan3D, SF3D, ...) via a template
  ``primitive``   no-GPU placeholder blob at the spec's proportions, for
                  testing Blender stages and CI

InstantMesh and OpenLRM pin torch/xformers/diffusers versions that conflict
with each other and with the diffusion stack, so they run in separate
interpreters instead of being imported.
"""

from __future__ import annotations

import gc
import logging
import math
import random
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".ply", ".fbx", ".stl")


@dataclass
class RawMesh:
    path: str
    backend: str
    has_vertex_colors: bool = False
    has_texture: bool = False
    vertices: int = 0
    faces: int = 0
    seconds: float = 0.0
    # Input camera of the reconstruction model in the exported file's coordinates
    # ({"position", "target", "up", "fov_deg", "coords": "gltf"|"blender"}), if known
    camera: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _newest_mesh(root: Path, since: float) -> Optional[Path]:
    found = [p for p in root.rglob("*") if p.suffix.lower() in MESH_SUFFIXES and p.stat().st_mtime >= since - 1]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _mesh_stats(path: Path) -> Dict[str, int]:
    try:
        import trimesh
    except ImportError:
        return {}
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
        return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}
    except Exception as exc:  # trimesh raises many loader-specific types
        log.debug("trimesh could not read %s: %s", path, exc)
        return {}


def _run(cmd: List[str], cwd: Optional[Path], timeout: int) -> None:
    log.info("Running: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.stdout:
        log.debug(proc.stdout[-4000:])
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {proc.stderr[-4000:]}")


def _interpreter(path: str) -> str:
    """Absolute interpreter path: subprocesses run with cwd=repo, which would
    otherwise re-anchor a relative path like third_party/X/.venv/bin/python."""
    return str(Path(path).resolve()) if ("/" in path or "\\" in path) else path


def _free_cuda():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class TripoSRBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def load(self):
        if self.model is not None:
            return
        repo = Path(self.cfg.triposr_repo).resolve()
        if not (repo / "tsr").is_dir():
            raise RuntimeError(
                f"TripoSR not found at {repo}. Clone it: git clone https://github.com/VAST-AI-Research/TripoSR {repo}"
            )
        sys.path.insert(0, str(repo))
        import torch
        from tsr.system import TSR

        device = self.cfg.device if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            log.warning("CUDA not available - TripoSR will run on CPU (minutes per mesh)")
        log.info("Loading TripoSR (%s) on %s", self.cfg.triposr_model, device)
        model = TSR.from_pretrained(self.cfg.triposr_model, config_name="config.yaml", weight_name="model.ckpt")
        model.renderer.set_chunk_size(self.cfg.chunk_size)
        model.to(device)
        self.device = device
        self.model = model

    def reconstruct(self, prepared: str, rgba: str, out_dir: Path) -> RawMesh:
        import numpy as np
        import torch
        import trimesh
        from PIL import Image

        self.load()
        image = Image.open(prepared).convert("RGB")
        with torch.no_grad():
            scene_codes = self.model([image], device=self.device)
        try:
            meshes = self.model.extract_mesh(scene_codes, True, resolution=self.cfg.mc_resolution)
        except TypeError:  # older TripoSR without the has_vertex_color argument
            meshes = self.model.extract_mesh(scene_codes, resolution=self.cfg.mc_resolution)
        mesh: trimesh.Trimesh = meshes[0]
        # Same re-orientation as TripoSR's gradio demo (to_gradio_3d_orientation):
        # converts its Z-up output to glTF Y-up so Blender's importer lands it upright.
        # Residual per-model offsets are handled by cleanup.import_rotation_deg.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        out = out_dir / "raw_triposr.glb"
        mesh.export(out)
        return RawMesh(str(out), "triposr", has_vertex_colors=mesh.visual.kind == "vertex",
                       vertices=len(mesh.vertices), faces=len(mesh.faces), camera=self.input_camera())

    @staticmethod
    def input_camera() -> Dict[str, Any]:
        """TripoSR's canonical input view (camera on +X at 1.9, Z up, 40 deg fovy; see
        tsr.utils.get_spherical_cameras), re-oriented like the exported mesh."""
        import numpy as np

        rot = TripoSRBackend._export_rotation()
        return {"position": (rot @ np.array([1.9, 0.0, 0.0])).tolist(), "target": [0.0, 0.0, 0.0],
                "up": (rot @ np.array([0.0, 0.0, 1.0])).tolist(), "fov_deg": 40.0, "coords": "gltf"}

    @staticmethod
    def _export_rotation():
        import numpy as np

        def rot(axis, angle):
            c, s = np.cos(angle), np.sin(angle)
            if axis == "x":
                return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # Same as the mesh: rotate -90 about X, then +90 about Y
        return rot("y", np.pi / 2) @ rot("x", -np.pi / 2)

    def unload(self):
        self.model = None
        _free_cuda()


class InstantMeshBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    def reconstruct(self, prepared: str, rgba: str, out_dir: Path) -> RawMesh:
        repo = Path(self.cfg.instantmesh_repo).resolve()
        if not (repo / "run.py").is_file():
            raise RuntimeError(f"InstantMesh not found at {repo} (git clone https://github.com/TencentARC/InstantMesh)")
        work = out_dir / "instantmesh"
        work.mkdir(parents=True, exist_ok=True)
        start = time.time()
        # Pass the RGBA cutout: InstantMesh skips rembg for RGBA input and applies its
        # own foreground resize, which matches what Zero123++ was trained on.
        _run([_interpreter(self.cfg.instantmesh_python), "run.py", self.cfg.instantmesh_config, str(Path(rgba).resolve()),
              "--output_path", str(work.resolve()), "--export_texmap"], cwd=repo, timeout=self.cfg.timeout_s)
        found = _newest_mesh(work, start)
        if found is None:
            raise RuntimeError(f"InstantMesh produced no mesh in {work}")
        return RawMesh(str(found), "instantmesh", has_texture=found.with_suffix(".mtl").exists(),
                       has_vertex_colors=not found.with_suffix(".mtl").exists(), **_mesh_stats(found))

    def unload(self):
        pass


class OpenLRMBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    def reconstruct(self, prepared: str, rgba: str, out_dir: Path) -> RawMesh:
        repo = Path(self.cfg.openlrm_repo).resolve()
        if not (repo / "openlrm").is_dir():
            raise RuntimeError(f"OpenLRM not found at {repo} (git clone https://github.com/3DTopia/OpenLRM)")
        size = "l" if "large" in self.cfg.openlrm_model else "s" if "small" in self.cfg.openlrm_model else "b"
        start = time.time()
        _run([_interpreter(self.cfg.openlrm_python), "-m", "openlrm.launch", "infer.lrm", "--infer", f"./configs/infer-{size}.yaml",
              f"model_name={self.cfg.openlrm_model}", f"image_input={Path(rgba).resolve()}",
              "export_video=false", "export_mesh=true"], cwd=repo, timeout=self.cfg.timeout_s)
        found = _newest_mesh(repo / "dumps", start)
        if found is None:
            raise RuntimeError("OpenLRM produced no mesh under dumps/")
        dest = out_dir / f"raw_openlrm{found.suffix}"
        shutil.copy2(found, dest)
        return RawMesh(str(dest), "openlrm", has_vertex_colors=True, **_mesh_stats(dest))

    def unload(self):
        pass


class CommandBackend:
    """Run any image-to-3D CLI. Template placeholders: {python} {input} {input_rgba} {output_dir}."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reconstruct(self, prepared: str, rgba: str, out_dir: Path) -> RawMesh:
        if not self.cfg.command_template:
            raise RuntimeError("reconstruction.command_template is empty")
        work = out_dir / "command"
        work.mkdir(parents=True, exist_ok=True)
        cmd = self.cfg.command_template.format(
            python=shlex.quote(sys.executable), input=shlex.quote(str(Path(prepared).resolve())),
            input_rgba=shlex.quote(str(Path(rgba).resolve())), output_dir=shlex.quote(str(work.resolve())),
        )
        start = time.time()
        _run(shlex.split(cmd), cwd=None, timeout=self.cfg.timeout_s)
        found = _newest_mesh(work, start)
        if found is None:
            raise RuntimeError(f"command produced no mesh in {work}")
        return RawMesh(str(found), "command", **_mesh_stats(found))

    def unload(self):
        pass


class PrimitiveBackend:
    """Deterministic stand-in for a neural reconstruction.

    Emits what feed-forward models typically emit - an over-tessellated,
    unit-normalised, vertex-coloured blob plus a small floating artefact - so
    Stages 3-6 can be exercised without a GPU.
    """

    def __init__(self, cfg, dimensions=(1.0, 1.0, 1.0), seed: int = 0):
        self.cfg = cfg
        self.dimensions = dimensions
        self.seed = seed

    def reconstruct(self, prepared: str, rgba: str, out_dir: Path) -> RawMesh:
        rng = random.Random(self.seed)
        largest = max(self.dimensions)
        sx, sy, sz = (d / largest * 0.5 for d in self.dimensions)
        verts: List[tuple] = []
        faces: List[tuple] = []

        def superellipsoid(cx, cy, cz, rx, ry, rz, rings, segments, colour, noise):
            base = len(verts)
            for i in range(rings + 1):
                theta = math.pi * i / rings
                for j in range(segments):
                    phi = 2 * math.pi * j / segments
                    # exponent < 1 squares the shape off like a hard-surface object
                    ct, st = math.cos(theta), math.sin(theta)
                    cp, sp = math.cos(phi), math.sin(phi)
                    e = 0.35
                    x = math.copysign(abs(st) ** e, st) * math.copysign(abs(cp) ** e, cp)
                    y = math.copysign(abs(st) ** e, st) * math.copysign(abs(sp) ** e, sp)
                    z = math.copysign(abs(ct) ** e, ct)
                    n = 1.0 + rng.uniform(-noise, noise)
                    verts.append((cx + rx * x * n, cy + ry * y * n, cz + rz * z * n, *colour))
            for i in range(rings):
                for j in range(segments):
                    a = base + i * segments + j
                    b = base + i * segments + (j + 1) % segments
                    c = base + (i + 1) * segments + (j + 1) % segments
                    d = base + (i + 1) * segments + j
                    faces.append((a, b, c))
                    faces.append((a, c, d))

        superellipsoid(0, 0, 0, sx, sy, sz, 64, 96, (0.35, 0.33, 0.3), 0.004)
        superellipsoid(sx * 1.4, 0, sz * 1.2, 0.02, 0.02, 0.02, 6, 8, (1.0, 0.0, 1.0), 0.0)  # floater

        out = out_dir / "raw_primitive.obj"
        with out.open("w", encoding="utf-8") as fh:
            fh.write("# asset_pipeline primitive placeholder (Z-up exported as Y-up OBJ)\n")
            for x, y, z, r, g, b in verts:
                # OBJ convention is Y-up, -Z forward
                fh.write(f"v {x:.6f} {z:.6f} {-y:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            for a, b, c in faces:
                fh.write(f"f {a + 1} {b + 1} {c + 1}\n")
        return RawMesh(str(out), "primitive", has_vertex_colors=True, vertices=len(verts), faces=len(faces))

    def unload(self):
        pass


class MeshReconstructor:
    def __init__(self, cfg, dimensions=(1.0, 1.0, 1.0), seed: int = 0):
        self.cfg = cfg
        backends = {
            "triposr": lambda: TripoSRBackend(cfg),
            "instantmesh": lambda: InstantMeshBackend(cfg),
            "openlrm": lambda: OpenLRMBackend(cfg),
            "command": lambda: CommandBackend(cfg),
            "primitive": lambda: PrimitiveBackend(cfg, dimensions, seed),
        }
        if cfg.backend not in backends:
            raise ValueError(f"unknown reconstruction backend '{cfg.backend}'")
        self.backend = backends[cfg.backend]()

    def reconstruct(self, prepared: str, rgba: str, out_dir: Path) -> RawMesh:
        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        result = self.backend.reconstruct(prepared, rgba, out_dir)
        result.seconds = round(time.time() - start, 2)
        log.info("Reconstruction (%s): %s [%d verts, %d faces, %.1fs]",
                 result.backend, result.path, result.vertices, result.faces, result.seconds)
        return result

    def unload(self):
        self.backend.unload()

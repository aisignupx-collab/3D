"""Stage 2 - image-to-3D reconstruction backends.

* ``triposr``     - TripoSR (feed-forward LRM, ~6 GB VRAM, <1 s per mesh on a
                    4090). Runs in-process on PyTorch/CUDA from a cloned repo.
* ``instantmesh`` - InstantMesh (Zero123++ multi-view diffusion + sparse-view
                    LRM, ~16-24 GB VRAM). Runs as a subprocess in its own env,
                    since its pinned dependencies conflict with TripoSR's.
* ``proxy``       - no ML: builds a deliberately messy placeholder mesh from
                    the AssetSpec (overlapping boxes, floaters, noise). Lets
                    you exercise Stages 3-6 without a GPU.

Every backend writes ``raw_mesh.ply`` (or ``.obj`` with textures) and returns
the rotation Blender must apply to bring it into the Z-up / -Y-front frame.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from asset_spec import AssetSpec
from config import ReconstructionConfig

log = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    mesh_path: Path
    backend: str
    has_color: bool
    # Euler XYZ degrees mapping the backend's frame to Blender Z-up, -Y front.
    import_rotation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    extra_files: list[Path] = field(default_factory=list)


class ReconstructionBackend(ABC):
    name = "base"

    def __init__(self, cfg: ReconstructionConfig):
        self.cfg = cfg

    @abstractmethod
    def reconstruct(self, image: Optional[Path], out_dir: Path, spec: AssetSpec) -> ReconstructionResult: ...

    def close(self) -> None:
        """Release GPU memory before Blender starts baking."""


def _torch_device(requested: str) -> str:
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available - running reconstruction on CPU (slow)")
        return "cpu"
    return requested


class TripoSRBackend(ReconstructionBackend):
    """https://github.com/VAST-AI-Research/TripoSR"""

    name = "triposr"

    def __init__(self, cfg: ReconstructionConfig):
        super().__init__(cfg)
        if cfg.triposr_repo:
            sys.path.insert(0, str(Path(cfg.triposr_repo).expanduser().resolve()))
        try:
            import torch
            from tsr.system import TSR
        except ImportError as exc:
            raise RuntimeError(
                "TripoSR not importable. Clone https://github.com/VAST-AI-Research/TripoSR, "
                "install its requirements and set reconstruction.triposr_repo"
            ) from exc
        self.torch = torch
        self.device = _torch_device(cfg.device)
        log.info("Loading TripoSR weights (%s) on %s", cfg.triposr_model, self.device)
        self.model = TSR.from_pretrained(cfg.triposr_model, config_name="config.yaml", weight_name="model.ckpt")
        self.model.renderer.set_chunk_size(cfg.triposr_chunk_size)
        self.model.to(self.device)

    def reconstruct(self, image: Optional[Path], out_dir: Path, spec: AssetSpec) -> ReconstructionResult:
        from PIL import Image

        if image is None:
            raise ValueError("TripoSR needs a pre-processed input image")
        out_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(image).convert("RGB")  # already grey-composited by Stage 1
        with self.torch.no_grad():
            scene_codes = self.model([img], device=self.device)
            try:
                meshes = self.model.extract_mesh(scene_codes, True, resolution=self.cfg.triposr_mc_resolution)
            except TypeError:  # older TripoSR without the has_vertex_color argument
                meshes = self.model.extract_mesh(scene_codes, resolution=self.cfg.triposr_mc_resolution)
        mesh = meshes[0]
        ply = out_dir / "raw_mesh.ply"
        mesh.export(ply)
        glb = out_dir / "raw_mesh.glb"
        mesh.export(glb)
        log.info("TripoSR mesh: %d verts, %d faces", len(mesh.vertices), len(mesh.faces))
        # TripoSR output is Z-up with the input view looking down -X (object faces +X).
        return ReconstructionResult(ply, self.name, has_color=True,
                                    import_rotation_deg=[0.0, 0.0, -90.0], extra_files=[glb])

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class InstantMeshBackend(ReconstructionBackend):
    """https://github.com/TencentARC/InstantMesh (run.py as a subprocess)."""

    name = "instantmesh"

    def reconstruct(self, image: Optional[Path], out_dir: Path, spec: AssetSpec) -> ReconstructionResult:
        if image is None:
            raise ValueError("InstantMesh needs an input image")
        if not self.cfg.instantmesh_repo:
            raise RuntimeError("Set reconstruction.instantmesh_repo to a clone of TencentARC/InstantMesh")
        repo = Path(self.cfg.instantmesh_repo).expanduser().resolve()
        python = self.cfg.instantmesh_python or sys.executable
        out_dir.mkdir(parents=True, exist_ok=True)
        work = out_dir / "instantmesh"
        cmd = [
            python, "run.py", self.cfg.instantmesh_config, str(Path(image).resolve()),
            "--output_path", str(work.resolve()),
            "--diffusion_steps", str(self.cfg.instantmesh_diffusion_steps),
            "--export_texmap",
        ]
        env = dict(os.environ)
        if self.cfg.device.startswith("cuda:"):
            env.setdefault("CUDA_VISIBLE_DEVICES", self.cfg.device.split(":", 1)[1])
        log.info("Running InstantMesh: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True,
                              timeout=self.cfg.timeout_s)
        (out_dir / "instantmesh.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"InstantMesh failed (see {out_dir / 'instantmesh.log'}):\n{proc.stderr[-2000:]}")

        objs = sorted(work.rglob("meshes/*.obj"), key=lambda p: p.stat().st_mtime)
        if not objs:
            raise RuntimeError(f"InstantMesh produced no mesh under {work}")
        src = objs[-1]
        # Keep the OBJ next to its .mtl and texture so Blender resolves them.
        dest = out_dir / "raw_mesh.obj"
        for sibling in src.parent.glob(src.stem + "*"):
            target = out_dir / sibling.name.replace(src.stem, "raw_mesh", 1)
            shutil.copy2(sibling, target)
        mtl = dest.with_suffix(".mtl")
        if mtl.exists():
            text = mtl.read_text(encoding="utf-8").replace(src.stem, "raw_mesh")
            mtl.write_text(text, encoding="utf-8")
            obj_text = dest.read_text(encoding="utf-8").replace(f"{src.stem}.mtl", "raw_mesh.mtl")
            dest.write_text(obj_text, encoding="utf-8")
        # InstantMesh exports Y-up OBJ; Blender's importer converts it to Z-up.
        return ReconstructionResult(dest, self.name, has_color=True)


class ProxyBackend(ReconstructionBackend):
    """Placeholder "AI soup" mesh for GPU-less dry runs of Stages 3-6."""

    name = "proxy"

    def reconstruct(self, image: Optional[Path], out_dir: Path, spec: AssetSpec) -> ReconstructionResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(7)
        dims = spec.dimensions_m
        unit = max(dims)
        norm = [d / unit for d in dims]  # AI output arrives at arbitrary (unit) scale
        mats = [m.base_color for m in spec.materials] or [[0.5, 0.5, 0.5]]

        boxes = [([-n / 2 for n in norm], [n / 2 for n in norm], mats[0])]
        lo0 = boxes[0][0]
        for i, part in enumerate(spec.moving_parts):
            lo = [lo0[a] + part.region_min[a] * norm[a] for a in range(3)]
            hi = [lo0[a] + part.region_max[a] * norm[a] for a in range(3)]
            # Push parts slightly out of the body so they overlap it (intersecting soup).
            lo = [l - 0.03 * norm[a] for a, l in enumerate(lo)]
            boxes.append((lo, hi, mats[(i + 1) % len(mats)]))
        verts: list[tuple[float, float, float, tuple[int, int, int]]] = []
        faces: list[tuple[int, ...]] = []
        for lo, hi, color in boxes:
            _add_box(verts, faces, lo, hi, color, segments=12, noise=0.004, rng=rng)
        # A floater: the kind of disconnected speck marching cubes leaves behind.
        corner = [n / 2 + 0.01 for n in norm]
        _add_box(verts, faces, corner, [c + 0.01 for c in corner], mats[0], segments=1, noise=0.0, rng=rng)

        path = out_dir / "raw_mesh.ply"
        with open(path, "w", encoding="ascii") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {len(verts)}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fh.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            for x, y, z, (r, g, b) in verts:
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            for face in faces:
                fh.write(f"{len(face)} {' '.join(map(str, face))}\n")
        log.info("Proxy mesh: %d verts, %d faces (%d boxes)", len(verts), len(faces), len(boxes))
        return ReconstructionResult(path, self.name, has_color=True)


def _add_box(verts, faces, lo, hi, color, segments, noise, rng) -> None:
    """Append a subdivided, slightly noisy box (each side as an n x n grid)."""
    def srgb8(c: float) -> int:
        c = max(0.0, min(1.0, c))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return int(round(c * 255))

    rgb = tuple(srgb8(c) for c in color)
    n = segments
    # (fixed axis, fixed side, u axis, v axis) with outward winding.
    sides = [(0, 0, 2, 1), (0, 1, 1, 2), (1, 0, 0, 2), (1, 1, 2, 0), (2, 0, 1, 0), (2, 1, 0, 1)]
    shared: dict[tuple[float, float, float], int] = {}
    for axis, side, ua, va in sides:
        grid = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                p = [0.0, 0.0, 0.0]
                p[axis] = hi[axis] if side else lo[axis]
                p[ua] = lo[ua] + (hi[ua] - lo[ua]) * i / n
                p[va] = lo[va] + (hi[va] - lo[va]) * j / n
                key = tuple(round(c, 7) for c in p)
                if key not in shared:
                    jitter = [c + rng.uniform(-noise, noise) for c in p]
                    shared[key] = len(verts)
                    verts.append((jitter[0], jitter[1], jitter[2], rgb))
                grid[i][j] = shared[key]
        for i in range(n):
            for j in range(n):
                faces.append((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))


def build_backend(cfg: ReconstructionConfig) -> ReconstructionBackend:
    backends = {"triposr": TripoSRBackend, "instantmesh": InstantMeshBackend, "proxy": ProxyBackend}
    if cfg.backend not in backends:
        raise ValueError(f"Unknown reconstruction backend '{cfg.backend}' (choose {sorted(backends)})")
    return backends[cfg.backend](cfg)

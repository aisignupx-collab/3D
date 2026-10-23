"""Stages 3-6 Blender automation (headless ``bpy``).

This file has two roles:

* **Host side** (imported by the pipeline): :class:`BlenderRunner` writes a job
  JSON and launches ``blender --background --python <this file> -- --job ...``,
  then reads back a result JSON.
* **Blender side** (executed by Blender's bundled Python): the ``op_*``
  functions do the actual work: cleanup/remesh/scale, UV + PBR baking,
  segmentation, rigging, audit renders and export.

It must only import the standard library at module level, plus ``bpy`` /
``bmesh`` / ``mathutils`` / ``numpy`` (bundled with Blender) on the Blender
side. Tested against Blender 4.2 LTS; version differences (auto smooth, EEVEE
engine id, OBJ/PLY/STL operators) are handled in ``_compat_*`` helpers and by
:func:`call_op`, which drops keyword arguments an operator does not have.

Conventions: Z-up, metres, X = length, characters face -Y. After Stage 3 the
asset sits centred on the XY origin with its lowest point at Z = 0 and its
origin at the bottom centre.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import bmesh
    import bpy
    import mathutils
    import numpy as np

    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

LOG_PREFIX = "[asset_pipeline]"


# =============================================================================
# Host side
# =============================================================================

class BlenderError(RuntimeError):
    pass


class BlenderTimeout(BlenderError):
    pass


class BlenderRunner:
    """Runs operations from this file inside a headless Blender process."""

    def __init__(self, executable: str, work_dir, timeout_s: int = 3600, use_gpu: bool = True):
        if not executable or not Path(executable).is_file():
            raise BlenderError(
                f"Blender executable not found ({executable!r}). Set blender.executable in config.yaml "
                "or the BLENDER_PATH environment variable."
            )
        self.executable = executable
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.use_gpu = use_gpu
        self.script = str(Path(__file__).resolve())

    def version(self) -> str:
        out = subprocess.run([self.executable, "--version"], capture_output=True, text=True, timeout=60)
        return out.stdout.splitlines()[0] if out.stdout else "unknown"

    def run(self, operation: str, params: dict, tag: str = "", timeout_s: int = 0) -> dict:
        tag = tag or operation
        timeout = timeout_s or self.timeout_s
        job_path = self.work_dir / f"job_{tag}.json"
        result_path = self.work_dir / f"result_{tag}.json"
        log_path = self.work_dir / f"blender_{tag}.log"
        result_path.unlink(missing_ok=True)
        job_path.write_text(json.dumps({"operation": operation, "params": params, "use_gpu": self.use_gpu},
                                       indent=2), encoding="utf-8")
        cmd = [self.executable, "--background", "--factory-startup", "--python-exit-code", "1",
               "--python", self.script, "--", "--job", str(job_path), "--result", str(result_path)]
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_fh:
            try:
                proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired as exc:  # subprocess.run kills Blender before raising
                raise BlenderTimeout(f"Blender '{operation}' timed out after {timeout}s (log: {log_path})") from exc
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        if not result_path.is_file():
            raise BlenderError(f"Blender '{operation}' exited {proc.returncode} without a result.\n{tail}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            raise BlenderError(f"Blender '{operation}' failed: {result.get('error')}\n{result.get('traceback', '')}")
        result["seconds"] = round(time.time() - start, 2)
        result["log"] = str(log_path)
        return result


# =============================================================================
# Blender side - generic helpers
# =============================================================================

def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def call_op(op, **kwargs):
    """Call a bpy operator, silently dropping kwargs it does not support.

    Operator signatures drift between Blender releases (exporters especially);
    this keeps one code path working across 3.6 - 4.x.
    """
    props = op.get_rna_type().properties.keys()
    dropped = [k for k in kwargs if k not in props]
    if dropped:
        log(f"{op.idname_py()}: ignoring unsupported args {dropped}")
    return op(**{k: v for k, v in kwargs.items() if k in props})


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def set_mode(mode: str):
    if bpy.context.object and bpy.context.object.mode != mode:
        bpy.ops.object.mode_set(mode=mode)


def select_only(*objs, active=None):
    set_mode("OBJECT")
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or (objs[0] if objs else None)


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def asset_meshes():
    """Body + parts only (excludes bake sources, rig widgets, helpers)."""
    return [o for o in mesh_objects() if o.get("asset_role") in ("target", "part")]


def find_role(role: str):
    for o in bpy.context.scene.objects:
        if o.get("asset_role") == role:
            return o
    return None


def world_coords(obj) -> "np.ndarray":
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(obj.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


def bbox(objs):
    coords = np.vstack([world_coords(o) for o in objs if len(o.data.vertices)])
    return coords.min(axis=0), coords.max(axis=0)


def apply_transforms(obj):
    select_only(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def apply_modifier(obj, mod):
    select_only(obj)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def face_count(obj) -> int:
    return len(obj.data.polygons)


def save_blend(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).resolve()))


def open_blend(path: str):
    bpy.ops.wm.open_mainfile(filepath=str(Path(path).resolve()))


def enable_gpu(scene, use_gpu: bool) -> str:
    """Configure Cycles for the best available GPU backend, else CPU."""
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    if not use_gpu:
        return "CPU"
    try:
        cprefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return "CPU"
    for device_type in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            cprefs.compute_device_type = device_type
        except TypeError:
            continue
        if hasattr(cprefs, "refresh_devices"):
            cprefs.refresh_devices()
        else:
            cprefs.get_devices()
        devices = [d for d in cprefs.devices if d.type == device_type]
        if devices:
            for d in cprefs.devices:
                d.use = d.type == device_type
            scene.cycles.device = "GPU"
            log(f"Cycles using {device_type}: {[d.name for d in devices]}")
            return device_type
    return "CPU"


# ----------------------------------------------------------------- import

def import_mesh(path: str):
    path = str(Path(path).resolve())
    ext = Path(path).suffix.lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            call_op(bpy.ops.wm.obj_import, filepath=path, import_vertex_groups=False)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif ext == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"unsupported mesh format: {ext}")
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh objects imported from {path}")
    # glTF imports nest meshes under empties; bake parent transforms and drop the rest
    select_only(*meshes)
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    if len(meshes) > 1:
        select_only(*meshes, active=meshes[0])
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    apply_transforms(obj)
    return obj


# ----------------------------------------------------------------- metrics

def _islands(bm):
    """Connected face components (lists of BMFace)."""
    seen = set()
    islands = []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack, island = [f], []
        seen.add(f.index)
        while stack:
            cur = stack.pop()
            island.append(cur)
            for e in cur.edges:
                for nf in e.link_faces:
                    if nf.index not in seen:
                        seen.add(nf.index)
                        stack.append(nf)
        islands.append(island)
    return islands


def uv_stretch(bm, uv_layer):
    """Area-weighted mean |log(uv_area_share / surface_area_share)|. 0 = no distortion."""
    areas3, areas2 = [], []
    for f in bm.faces:
        uvs = [loop[uv_layer].uv for loop in f.loops]
        a2 = 0.0
        for i in range(len(uvs)):
            x1, y1 = uvs[i]
            x2, y2 = uvs[(i + 1) % len(uvs)]
            a2 += x1 * y2 - x2 * y1
        areas3.append(f.calc_area())
        areas2.append(abs(a2) * 0.5)
    a3 = np.array(areas3)
    a2 = np.array(areas2)
    if a3.sum() <= 0 or a2.sum() <= 0:
        return {"stretch": 99.0, "uv_coverage": 0.0}
    share3 = a3 / a3.sum()
    share2 = a2 / a2.sum()
    ratio = np.clip(np.where(share3 > 0, share2 / np.maximum(share3, 1e-12), 1.0), 1e-3, 1e3)
    stretch = float(np.sum(share3 * np.abs(np.log(ratio))))
    return {"stretch": round(stretch, 4), "uv_coverage": round(float(a2.sum()), 4)}


def mesh_metrics(obj) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.faces.ensure_lookup_table()
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    nm_verts = sum(1 for v in bm.verts if not v.is_manifold)
    quads = sum(1 for f in bm.faces if len(f.verts) == 4)
    tris = sum(len(f.verts) - 2 for f in bm.faces)
    islands = _islands(bm)
    total = max(1, len(bm.faces))
    small = sum(1 for i in islands if len(i) < 0.01 * total)
    degenerate = sum(1 for f in bm.faces if f.calc_area() < 1e-12)
    metrics = {
        "object": obj.name,
        "vertices": len(bm.verts),
        "faces": len(bm.faces),
        "triangles": tris,
        "quad_ratio": round(quads / total, 3),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "non_manifold_vertices": nm_verts,
        "watertight": boundary == 0 and non_manifold == 0,
        "islands": len(islands),
        "small_islands": small,
        "degenerate_faces": degenerate,
        "volume_m3": round(abs(bm.calc_volume()), 6) if boundary == 0 else None,
    }
    if bm.loops.layers.uv.active is not None:
        metrics.update(uv_stretch(bm, bm.loops.layers.uv.active))
    bm.free()
    lo, hi = bbox([obj])
    metrics["dimensions_m"] = [round(float(v), 4) for v in (hi - lo)]
    return metrics


def island_collisions(obj, max_islands: int = 40) -> int:
    """Count pairs of loose shells whose surfaces intersect (BVH overlap)."""
    from mathutils.bvhtree import BVHTree

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    islands = sorted(_islands(bm), key=len, reverse=True)[:max_islands]
    trees = []
    for island in islands:
        verts = {v for f in island for v in f.verts}
        index = {v: i for i, v in enumerate(verts)}
        trees.append(BVHTree.FromPolygons([v.co.copy() for v in verts],
                                          [[index[v] for v in f.verts] for f in island]))
    bm.free()
    hits = 0
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].overlap(trees[j]):
                hits += 1
    return hits


# ----------------------------------------------------------------- geometry ops

def bm_cleanup(obj, merge_dist: float, min_island_ratio: float, fill_holes: bool):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    before = len(bm.faces)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=merge_dist)
    loose_verts = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    loose_edges = [e for e in bm.edges if not e.link_faces]
    bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
    bm.faces.ensure_lookup_table()
    islands = _islands(bm)
    removed = 0
    if len(islands) > 1:
        largest = max(len(i) for i in islands)
        doomed = [f for i in islands if len(i) < min_island_ratio * largest for f in i]
        removed = sum(1 for i in islands if len(i) < min_island_ratio * largest)
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
    filled = 0
    if fill_holes:
        boundary = [e for e in bm.edges if e.is_boundary]
        if boundary:
            res = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
            filled = len(res["faces"])
            if res["faces"]:
                bmesh.ops.triangulate(bm, faces=res["faces"])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()
    log(f"cleanup: {before} -> {len(me.polygons)} faces, removed {removed} floaters, filled {filled} holes")
    return {"floaters_removed": removed, "holes_filled": filled}


def open_edges(obj) -> int:
    """Boundary + non-manifold edges (what keeps a mesh from being watertight)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return n


def _count_open(bm) -> int:
    return sum(1 for e in bm.edges if not e.is_manifold)


def _repair_pass(bm, tol: float, aggressive: bool):
    if not aggressive:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=tol)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=tol)
        seen, dupes = set(), []
        for f in bm.faces:
            key = frozenset(v.index for v in f.verts)
            if key in seen:
                dupes.append(f)
            seen.add(key)
        bmesh.ops.delete(bm, geom=dupes, context="FACES")
    else:
        for _ in range(3):  # fins: faces hanging off an over-shared edge, peeled layer by layer
            fins = [f for f in bm.faces if sum(1 for e in f.edges if len(e.link_faces) != 2) >= 2
                    and any(len(e.link_faces) > 2 for e in f.edges)]
            if not fins:
                break
            bmesh.ops.delete(bm, geom=fins, context="FACES")
        over = [e for e in bm.edges if len(e.link_faces) > 2]
        if over:  # remaining 3+-face edges: split so each sheet can be closed on its own
            bmesh.ops.split_edges(bm, edges=over)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        res = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
        if res["faces"]:
            bmesh.ops.triangulate(bm, faces=res["faces"])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)


def repair_manifold(obj) -> dict:
    """Close the topology without changing the shape: merge, drop duplicate and dangling faces, fill holes.

    Exact booleans, quadriflow and decimation occasionally leave duplicate faces
    (coplanar overlaps), fins on an edge shared by 3+ faces and slit holes. Each
    pass is kept only if it leaves fewer open edges.
    """
    lo, hi = bbox([obj])
    tol = 1e-5 * max(float(np.max(hi - lo)), 1e-6)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = best = _count_open(bm)
    for aggressive in (False, True):
        if not best:
            break
        work = bm.copy()
        _repair_pass(work, tol, aggressive)
        n = _count_open(work)
        if n < best:
            bm.free()
            bm, best = work, n
        else:
            work.free()
    if best < before:
        bm.to_mesh(obj.data)
        obj.data.update()
    bm.free()
    if before:
        log(f"repair: {obj.name} open edges {before} -> {best}")
    return {"open_edges_before": before, "open_edges_after": best}


def surface_area(obj) -> float:
    return sum(poly.area for poly in obj.data.polygons)


def voxel_seal(obj, resolution: int, target_faces: int = 0) -> int:
    """Last resort: rebuild the surface as a voxel shell (closed by construction), keeping the shape.

    The voxel size is chosen to land near ``target_faces`` directly; decimating the
    shell afterwards could open it again.
    """
    ref = obj.copy()
    ref.data = obj.data.copy()
    bpy.context.scene.collection.objects.link(ref)
    lo, hi = bbox([obj])
    voxel = float(np.max(hi - lo)) / max(16, resolution)
    if target_faces:
        voxel = max(voxel, math.sqrt(surface_area(obj) / target_faces))
    voxel_remesh(obj, voxel)
    shrinkwrap_to(obj, ref)
    bpy.data.objects.remove(ref, do_unlink=True)
    repair_manifold(obj)
    return open_edges(obj)


def _euler_matrices(angles_deg: "np.ndarray") -> "np.ndarray":
    """(N, 3) XYZ Euler angles in degrees -> (N, 3, 3) matrices, R = Rz @ Ry @ Rx (Blender 'XYZ')."""
    a = np.radians(angles_deg)
    cx, cy, cz = np.cos(a).T
    sx, sy, sz = np.sin(a).T
    m = np.empty((len(a), 3, 3))
    m[:, 0, 0] = cy * cz
    m[:, 0, 1] = sx * sy * cz - cx * sz
    m[:, 0, 2] = cx * sy * cz + sx * sz
    m[:, 1, 0] = cy * sz
    m[:, 1, 1] = sx * sy * sz + cx * cz
    m[:, 1, 2] = cx * sy * sz - sx * cz
    m[:, 2, 0] = -sy
    m[:, 2, 1] = sx * cy
    m[:, 2, 2] = cx * cy
    return m


def _hull_points(obj, limit: int = 4000) -> "np.ndarray":
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        res = bmesh.ops.convex_hull(bm, input=bm.verts, use_existing_faces=False)
        pts = np.array([v.co[:] for v in res["geom"] if isinstance(v, bmesh.types.BMVert)])
    except (RuntimeError, ValueError):
        pts = np.empty((0, 3))
    bm.free()
    if len(pts) < 4:
        pts = world_coords(obj)
    if len(pts) > limit:
        pts = pts[np.random.default_rng(0).choice(len(pts), limit, replace=False)]
    return pts - pts.mean(axis=0)


def _min_volume_rotation(pts, center, span, step, tilt=True):
    grid = np.arange(-span, span + 1e-9, step)
    if tilt:
        angles = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3) + center
    else:  # yaw only
        angles = np.stack([np.zeros_like(grid), np.zeros_like(grid), grid], -1) + center
    rotated = np.einsum("nij,pj->npi", _euler_matrices(angles), pts)
    volume = np.prod(rotated.max(axis=1) - rotated.min(axis=1), axis=1)
    # Tie-break towards the smallest rotation: symmetric shapes have many equal minima
    volume *= 1.0 + 0.002 * np.abs(angles).sum(axis=1) / 45.0
    return angles[int(np.argmin(volume))]


def align_to_axes(obj, max_tilt: float = 45.0, longest_to_x: bool = True):
    """Rotate so the object's tightest bounding box is axis-aligned, upright.

    Feed-forward image-to-3D models reconstruct in the input camera's frame, so
    a three-quarter view from above yields an asset that is both yawed (~45 deg)
    and tilted by the camera elevation. A coarse-to-fine search for the
    minimum-volume box over +-``max_tilt`` about each axis undoes both while
    keeping the reconstruction's overall up direction. Optionally the longest
    horizontal extent is then turned onto X (weapons, vehicles, walls).
    """
    pts = _hull_points(obj)
    # max_tilt <= 0: yaw only. Bodies in A-pose do not have an upright minimum box, so
    # characters keep the reconstruction's up axis and are only turned about Z.
    tilt = max_tilt > 0
    best = _min_volume_rotation(pts, np.zeros(3), max_tilt if tilt else 45.0, 5.0, tilt)
    best = _min_volume_rotation(pts, best, 5.0, 1.0, tilt)
    best = _min_volume_rotation(pts, best, 1.0, 0.25, tilt)
    rx, ry, rz = (float(v) for v in best)
    if longest_to_x:
        rotated = pts @ _euler_matrices(best[None])[0].T
        ext = rotated.max(axis=0) - rotated.min(axis=0)
        if ext[1] > ext[0] * 1.05:
            # Compose an extra 90 deg yaw: Rz(90) @ R
            mat = mathutils.Matrix.Rotation(math.radians(90), 3, "Z") @ mathutils.Euler(
                [math.radians(a) for a in (rx, ry, rz)], "XYZ").to_matrix()
            rx, ry, rz = (math.degrees(a) for a in mat.to_euler("XYZ"))
    if max(abs(rx), abs(ry), abs(rz)) < 0.5:
        return [0.0, 0.0, 0.0]
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = [math.radians(a) for a in (rx, ry, rz)]
    apply_transforms(obj)
    return [round(rx, 2), round(ry, 2), round(rz, 2)]


def front_facing_yaw(camera, frame, keep_long_axis: bool) -> float:
    """Yaw (multiple of 90, or of 180 to keep X as the long axis) that puts the camera at -Y."""
    pos = np.asarray(camera["position"], float)
    if camera.get("coords") == "gltf":
        pos = np.array(GLTF_TO_BLENDER) @ pos
    x, y = (frame @ np.append(pos, 1.0))[:2]
    best, best_y = 0.0, None
    for yaw in ((0.0, 180.0) if keep_long_axis else (0.0, 90.0, 180.0, 270.0)):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ry = s * x + c * y
        if best_y is None or ry < best_y - 1e-9:
            best, best_y = yaw, ry
    return best


def ground_and_center(obj, followers=()):
    """Centre on XY, rest on Z=0; ``followers`` get the same translation."""
    lo, hi = bbox([obj])
    center = (lo + hi) / 2.0
    offset = (-center[0], -center[1], -lo[2])
    for o in (obj, *followers):
        o.location = offset
        apply_transforms(o)
    return offset


def _translation(offset) -> "np.ndarray":
    m = np.eye(4)
    m[:3, 3] = offset
    return m


def _linear(mat3) -> "np.ndarray":
    m = np.eye(4)
    m[:3, :3] = mat3
    return m


AXIS_PERMUTATIONS = [(0, 0, 0), (90, 0, 0), (0, 90, 0), (0, 0, 90), (90, 0, 90), (90, 90, 0)]
PER_AXIS_LIMITS = (0.5, 2.0)


def _shape_error(dims, target):
    """Mismatch of proportions (scale-free): sum of |log ratio| of max-normalised extents."""
    d = np.maximum(np.asarray(dims, float), 1e-9)
    t = np.maximum(np.asarray(target, float), 1e-9)
    return float(np.abs(np.log((d / d.max()) / (t / t.max()))).sum())


def best_axis_permutation(obj, target_dims, min_gain: float = 0.35):
    """90-degree rotation whose axis order best matches the spec's proportions.

    Catches reconstructions lying on their side (a wall whose height ended up on Y).
    Applied only when it improves the proportion error clearly; returns the Euler (deg).
    """
    lo, hi = bbox([obj])
    dims = hi - lo
    base = _shape_error(dims, target_dims)
    t = np.asarray(target_dims, float)
    if t.max() / max(t.min(), 1e-9) < 1.5:
        # A near-cubic spec (a castle, a crate) can't tell its axes apart: an upright scan that is a
        # little wider than planned would be tipped over. Keep the reconstruction's own up axis.
        return [0.0, 0.0, 0.0], round(base, 3)
    best, best_err = (0, 0, 0), base
    for euler in AXIS_PERMUTATIONS[1:]:
        rot = np.abs(_euler_matrices(np.array([euler], float))[0])
        err = _shape_error(rot @ dims, target_dims)
        if err < best_err - 1e-9:
            best, best_err = euler, err
    if best == (0, 0, 0) or base - best_err < min_gain:
        return [0.0, 0.0, 0.0], round(base, 3)
    obj.rotation_euler = [math.radians(a) for a in best]
    apply_transforms(obj)
    log(f"axes: rotated {best} to match spec proportions (error {base:.2f} -> {best_err:.2f})")
    return [float(a) for a in best], round(best_err, 3)


def fit_scale(obj, target_dims, fit_dimension: str, axis_scale: dict, per_axis: bool = False):
    """Uniform scale on the trusted dimension, then (``per_axis``) bounded per-axis
    correction toward the spec, then the audit's cumulative ``axis_scale`` multipliers."""
    lo, hi = bbox([obj])
    dims = hi - lo
    if fit_dimension == "longest":
        current, target = float(dims.max()), float(max(target_dims))
    else:
        axis = {"length": 0, "width": 1, "height": 2}[fit_dimension]
        current, target = float(dims[axis]), float(target_dims[axis])
    factor = target / max(current, 1e-9)
    scale = [factor, factor, factor]
    per_axis_applied = [1.0, 1.0, 1.0]
    if per_axis:
        for i in range(3):
            want = float(target_dims[i]) / max(float(dims[i]) * factor, 1e-9)
            if abs(want - 1.0) > 0.08:  # leave small natural deviations alone
                per_axis_applied[i] = min(max(want, PER_AXIS_LIMITS[0]), PER_AXIS_LIMITS[1])
                scale[i] *= per_axis_applied[i]
    for axis_name, extra in (axis_scale or {}).items():
        idx = "XYZ".index(axis_name.upper())
        scale[idx] *= min(max(float(extra), PER_AXIS_LIMITS[0]), PER_AXIS_LIMITS[1])
    obj.scale = scale
    apply_transforms(obj)
    lo, hi = bbox([obj])
    dims = hi - lo
    error = [round(float(d / t - 1.0), 4) if t > 0 else 0.0 for d, t in zip(dims, target_dims)]
    log(f"scale: x{factor:.4f}, per-axis {np.round(per_axis_applied, 3).tolist()}, "
        f"dims {np.round(dims, 3).tolist()} vs target {target_dims}")
    return {"scale_factor": factor, "scale_vector": scale, "per_axis_fit": per_axis_applied,
            "dimension_error": error}


def union_islands(obj):
    """Boolean-union intersecting loose shells (used when not voxel remeshing)."""
    select_only(obj)
    set_mode("EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    set_mode("OBJECT")
    pieces = sorted(bpy.context.selected_objects, key=face_count, reverse=True)
    base = pieces[0]
    for piece in pieces[1:]:
        mod = base.modifiers.new("union", "BOOLEAN")
        mod.operation = "UNION"
        mod.solver = "EXACT"
        mod.object = piece
        apply_modifier(base, mod)
        bpy.data.objects.remove(piece, do_unlink=True)
    return base


def voxel_remesh(obj, voxel_size: float):
    obj.data.remesh_voxel_size = voxel_size
    obj.data.remesh_voxel_adaptivity = 0.0
    if hasattr(obj.data, "use_remesh_fix_poles"):
        obj.data.use_remesh_fix_poles = True
    select_only(obj)
    bpy.ops.object.voxel_remesh()


def decimate_to(obj, target_faces: int, passes: int = 3):
    # Collapse ratios act on triangles, so quad-heavy meshes overshoot; iterate.
    for _ in range(passes):
        current = face_count(obj)
        if current <= target_faces * 1.05:
            return
        mod = obj.modifiers.new("decimate", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = max(target_faces / current, 0.001)
        mod.use_collapse_triangulate = False
        apply_modifier(obj, mod)


def shrinkwrap_to(obj, target, offset: float = 0.0):
    mod = obj.modifiers.new("shrinkwrap", "SHRINKWRAP")
    mod.target = target
    mod.wrap_method = "NEAREST_SURFACEPOINT"
    mod.wrap_mode = "ON_SURFACE"
    mod.offset = offset
    apply_modifier(obj, mod)


def quadriflow(obj, target_faces: int) -> bool:
    select_only(obj)
    before = face_count(obj)
    try:
        res = call_op(bpy.ops.object.quadriflow_remesh, target_faces=target_faces, use_mesh_symmetry=False,
                      use_preserve_sharp=True, use_preserve_boundary=True, smooth_normals=False,
                      mode="FACES", seed=0)
    except RuntimeError as exc:
        log(f"quadriflow failed: {exc}")
        return False
    ok = "FINISHED" in res and face_count(obj) > 0 and face_count(obj) != before
    if not ok:
        log("quadriflow produced no result")
    return ok


def smooth(obj, iterations: int, factor: float = 0.5):
    if iterations <= 0:
        return
    mod = obj.modifiers.new("smooth", "SMOOTH")
    mod.factor = factor
    mod.iterations = iterations
    apply_modifier(obj, mod)


def auto_smooth(obj, angle_deg: float):
    select_only(obj)
    if bpy.app.version >= (4, 1, 0):
        call_op(bpy.ops.object.shade_smooth_by_angle, angle=math.radians(angle_deg), keep_sharp_edges=True)
    else:
        bpy.ops.object.shade_smooth()
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(angle_deg)


# =============================================================================
# Stage 3 - topology cleanup, remesh, scale fitting
# =============================================================================

def op_cleanup(p: dict) -> dict:
    reset_scene()
    raw = import_mesh(p["input_mesh"])
    raw.name = "raw_import"
    # ``frame`` maps imported-file coordinates to the final asset frame; Stage 4 uses it to
    # place the reconstruction model's input camera for reference-image projection.
    frame = np.eye(4)
    rot = p.get("import_rotation_deg") or [0, 0, 0]
    if any(rot):
        raw.rotation_euler = [math.radians(a) for a in rot]
        apply_transforms(raw)
        frame = _linear(_euler_matrices(np.array([rot], float))[0]) @ frame
    raw_faces = face_count(raw)

    lo, hi = bbox([raw])
    size = float((hi - lo).max())
    report = {"raw_faces": raw_faces}
    report.update(bm_cleanup(raw, p["merge_distance"] * size, p["min_island_ratio"], fill_holes=False))

    report["rotation_applied_deg"] = align_to_axes(
        raw, max_tilt=float(p.get("max_tilt_correction", 45.0)),
        longest_to_x=p.get("orient_longest_axis", True)) if p.get("auto_orient", True) else [0.0, 0.0, 0.0]
    frame = _linear(_euler_matrices(np.array([report["rotation_applied_deg"]], float))[0]) @ frame
    if p.get("match_spec_axes", True) and not p.get("organic"):
        perm, report["proportion_error"] = best_axis_permutation(raw, p["dimensions"])
        if any(perm):
            frame = _linear(_euler_matrices(np.array([perm], float))[0]) @ frame
        report["axis_permutation_deg"] = perm
    if p.get("reference_camera") and p.get("auto_orient", True):
        # The side the reference image shows is the asset's front: turn it to face -Y
        yaw = front_facing_yaw(p["reference_camera"], frame, keep_long_axis=p.get("orient_longest_axis", True))
        if yaw:
            raw.rotation_euler = (0.0, 0.0, math.radians(yaw))
            apply_transforms(raw)
            frame = _linear(_euler_matrices(np.array([[0.0, 0.0, yaw]], float))[0]) @ frame
        report["front_yaw_deg"] = yaw
    frame = _translation(ground_and_center(raw)) @ frame
    report.update(fit_scale(raw, p["dimensions"], p["fit_dimension"], p.get("axis_scale"),
                            per_axis=bool(p.get("per_axis_fit", not p.get("organic")))))
    frame = _linear(np.diag(report["scale_vector"])) @ frame
    frame = _translation(ground_and_center(raw)) @ frame
    lo, hi = bbox([raw])
    size = float((hi - lo).max())

    report["collisions_before"] = island_collisions(raw)

    # Keep the scaled, cleaned high-res mesh as the bake source for Stage 4
    high = raw.copy()
    high.data = raw.data.copy()
    high.name = "bake_source"
    high["asset_role"] = "bake_source"
    bpy.context.scene.collection.objects.link(high)
    high.hide_render = True

    obj = raw
    obj.name = p["name"]
    obj["asset_role"] = "target"
    mode = p["remesh_mode"]
    target_faces = int(p["target_faces"])
    # Remesh at unit size and scale back afterwards: remeshers have absolute tolerances, and the
    # same scan should behave identically as a 2 cm coin or a 632 m skyscraper.
    unit = 1.0 / max(size, 1e-9)
    for o in (obj, high):
        o.scale = (unit, unit, unit)
        apply_transforms(o)
    voxel_size = 1.0 / max(16, int(p["voxel_resolution"]))

    if p.get("symmetrize"):
        # Characters: mirror the better-reconstructed +X half onto -X. Clean symmetric
        # topology gives symmetric skin weights and hides one-sided reconstruction errors.
        select_only(obj)
        set_mode("EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.symmetrize(direction="POSITIVE_X", threshold=1e-4 * size)
        set_mode("OBJECT")
        report["symmetrized"] = True

    if mode in ("decimate", "none") and p.get("fix_intersections") and report["collisions_before"]:
        obj = union_islands(obj)
        obj.name = p["name"]
        obj["asset_role"] = "target"

    if mode == "quadriflow":
        # Quadriflow needs a manifold input; voxel remesh guarantees one and also
        # unions intersecting shells.
        voxel_remesh(obj, voxel_size)
        # Quadriflow time grows with input size; a dense voxel shell gains nothing past ~8x target
        decimate_to(obj, int(min(max(target_faces * 8, 20000), 100000)), passes=1)
        if quadriflow(obj, target_faces):
            shrinkwrap_to(obj, high)
            report["remesh"] = "quadriflow"
        else:
            decimate_to(obj, target_faces)
            report["remesh"] = "voxel+decimate (quadriflow fallback)"
    elif mode == "voxel":
        voxel_remesh(obj, voxel_size)
        decimate_to(obj, target_faces)
        report["remesh"] = "voxel"
    elif mode == "voxel_shrinkwrap":
        voxel_remesh(obj, voxel_size * 2.0)
        decimate_to(obj, target_faces)
        shrinkwrap_to(obj, high)
        report["remesh"] = "voxel_shrinkwrap"
    elif mode == "decimate":
        decimate_to(obj, target_faces)
        report["remesh"] = "decimate"
    else:
        report["remesh"] = "none"

    report.update({f"post_{k}": v for k, v in bm_cleanup(
        obj, 1e-6, p["min_island_ratio"], fill_holes=p.get("fill_holes", True)).items()})
    smooth(obj, int(p.get("smooth_iterations", 0)))
    report["repair"] = repair_manifold(obj)
    if report["repair"]["open_edges_after"] and mode != "none":
        left = voxel_seal(obj, int(p["voxel_resolution"]), target_faces)
        report["repair"].update(voxel_seal=True, open_edges_after=left)
        report["remesh"] += " +voxel seal"
    for o in (obj, high):
        o.scale = (size, size, size)
        apply_transforms(o)
    auto_smooth(obj, float(p.get("auto_smooth_angle", 40.0)))
    report["collisions_after"] = island_collisions(obj)
    frame = _translation(ground_and_center(obj, followers=[high])) @ frame  # bake source stays aligned
    report["frame_matrix"] = frame.round(8).tolist()
    bpy.context.scene["frame_matrix"] = [float(v) for v in frame.ravel()]
    report["metrics"] = mesh_metrics(obj)
    save_blend(p["output_blend"])
    log(f"cleanup done: {report['metrics']['faces']} faces, watertight={report['metrics']['watertight']}")
    return report


# =============================================================================
# Stage 3b - primitive templates (box / cylinder / cone / ... composites)
# =============================================================================

SQRT2 = math.sqrt(2.0)


def _bisect_keep_upper(bm):
    """Cut at z=0, keep z>=0, cap the cut, and stretch z from [0, 0.5] to [-0.5, 0.5]."""
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=(0, 0, 0), plane_no=(0, 0, 1),
                                 clear_inner=True)
    cut_edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
    if cut_edges:
        bmesh.ops.edgeloop_fill(bm, edges=cut_edges)
    for v in bm.verts:
        v.co.z = v.co.z * 2.0 - 0.5


def _torus(bm, segments, thickness=0.3, minor_segments=12):
    """Ring in XY with outer radius 0.5; the tube cross-section fills z in [-0.5, 0.5]."""
    minor = 0.5 * min(max(thickness, 0.02), 0.5)
    major = 0.5 - minor
    rings = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        ring = []
        for j in range(minor_segments):
            b = 2 * math.pi * j / minor_segments
            r = major + minor * math.cos(b)
            ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), 0.5 * math.sin(b))))
        rings.append(ring)
    for i in range(segments):
        for j in range(minor_segments):
            a, b = rings[i][j], rings[(i + 1) % segments][j]
            c, d = rings[(i + 1) % segments][(j + 1) % minor_segments], rings[i][(j + 1) % minor_segments]
            bm.faces.new((a, b, c, d))


def _wedge(bm):
    """Gable prism: ridge along X at the top, rectangle at the bottom."""
    v = [bm.verts.new(co) for co in ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
                                     (-0.5, 0.0, 0.5), (0.5, 0.0, 0.5))]
    for face in ((0, 3, 2, 1), (0, 1, 5, 4), (2, 3, 4, 5), (1, 2, 5), (3, 0, 4)):
        bm.faces.new([v[i] for i in face])


def _merlon_ring(bm, teeth, depth=0.12):
    """``teeth`` battlement blocks on a circle of outer radius 0.5, teeth and gaps equally wide."""
    teeth = max(4, int(teeth))
    r_out, r_in = 0.5, 0.5 * (1.0 - min(max(depth, 0.03), 0.6))
    half = math.pi / (2 * teeth)
    for k in range(teeth):
        a = 2 * math.pi * k / teeth
        ring = []
        for z in (-0.5, 0.5):
            for r, ang in ((r_in, a - half), (r_out, a - half), (r_out, a + half), (r_in, a + half)):
                ring.append(bm.verts.new((r * math.cos(ang), r * math.sin(ang), z)))
        b, t = ring[:4], ring[4:]
        bm.faces.new(b[::-1])
        bm.faces.new(t)
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new((b[i], b[j], t[j], t[i]))


def build_unit_primitive(kind: str, segments: int = 24, taper: float = 0.0, thickness: float = 0.3):
    """Canonical primitive filling [-0.5, 0.5]^3 with clean topology.

    Main axis is +Z, except half_cylinder / wedge whose axis is X.
    """
    bm = bmesh.new()
    if kind == "box":
        bmesh.ops.create_cube(bm, size=1.0)
    elif kind in ("cylinder", "cone"):
        top = 0.5 if kind == "cylinder" else 0.5 * taper
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=0.5,
                              radius2=max(top, 0.0), depth=1.0)
    elif kind == "pyramid":
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=4, radius1=0.5 * SQRT2,
                              radius2=0.5 * SQRT2 * taper, depth=1.0)
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=mathutils.Matrix.Rotation(math.pi / 4, 3, "Z"))
    elif kind in ("sphere", "hemisphere"):
        bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=max(6, segments // 2), radius=0.5)
        if kind == "hemisphere":
            _bisect_keep_upper(bm)
    elif kind == "half_cylinder":
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=0.5, radius2=0.5,
                              depth=1.0)
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=mathutils.Matrix.Rotation(math.pi / 2, 3, "Y"))
        _bisect_keep_upper(bm)
    elif kind == "wedge":
        _wedge(bm)
    elif kind == "torus":
        _torus(bm, segments, thickness)
    elif kind == "merlon_ring":
        _merlon_ring(bm, segments, thickness)
    else:
        bm.free()
        raise ValueError(f"unknown primitive '{kind}'")
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)  # collapses cone apexes
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


_AXIS_ROT = {  # rotate canonical +Z onto the requested axis
    "Z": None, "+Z": None, "-Z": ("X", math.pi),
    "X": ("Y", math.pi / 2), "+X": ("Y", math.pi / 2), "-X": ("Y", -math.pi / 2),
    "Y": ("X", -math.pi / 2), "+Y": ("X", -math.pi / 2), "-Y": ("X", math.pi / 2),
}


def component_object(comp, lo, size, material_slots):
    """Build one ShapeComponent (dict) as a mesh object in world space."""
    kind = comp["type"]
    bm = build_unit_primitive(kind, int(comp.get("segments", 24)), float(comp.get("taper", 0.0)),
                              float(comp.get("thickness", 0.3)))
    axis = comp.get("axis", "Z").upper()
    if kind in ("half_cylinder", "wedge"):
        if axis.lstrip("+-") == "Y":
            bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0),
                             matrix=mathutils.Matrix.Rotation(math.pi / 2, 3, "Z"))
    elif _AXIS_ROT.get(axis):
        ax, ang = _AXIS_ROT[axis]
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=mathutils.Matrix.Rotation(ang, 3, ax))
    rmin = lo + np.array(comp["min"], dtype=float) * size
    rmax = lo + np.array(comp["max"], dtype=float) * size
    dims = np.maximum(rmax - rmin, 1e-5)
    bmesh.ops.scale(bm, vec=dims.tolist(), verts=bm.verts)
    bevel = float(comp.get("bevel", 0.0))
    if bevel > 0 and kind in ("box", "wedge", "pyramid"):
        bmesh.ops.bevel(bm, geom=bm.verts[:] + bm.edges[:], offset=bevel * float(dims.min()) * 0.5,
                        offset_type="OFFSET", segments=2, profile=0.5, affect="EDGES", clamp_overlap=True)
    rot = comp.get("rotation_deg") or [0, 0, 0]
    if any(abs(a) > 1e-6 for a in rot):
        mat = mathutils.Euler([math.radians(a) for a in rot], "XYZ").to_matrix()
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=mat)
    bmesh.ops.translate(bm, vec=((rmin + rmax) / 2.0).tolist(), verts=bm.verts)
    me = bpy.data.meshes.new(comp.get("name", kind))
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(comp.get("name", kind), me)
    bpy.context.scene.collection.objects.link(obj)
    name = comp.get("material") or ""
    if name:
        mat = material_slots.get(name)
        if mat is not None:
            me.materials.append(mat)
    return obj


def sample_surface(obj, n: int, rng) -> "np.ndarray":
    """Area-weighted random points on an object's surface (world space)."""
    me = obj.data
    me.calc_loop_triangles()
    if not len(me.loop_triangles):
        return np.empty((0, 3))
    co = world_coords(obj)
    tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tris)
    tri = co[tris.reshape(-1, 3)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return np.empty((0, 3))
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _bvh(objs):
    from mathutils.bvhtree import BVHTree

    verts, polys, offset = [], [], 0
    for o in objs:
        co = world_coords(o)
        verts.extend(mathutils.Vector(c) for c in co)
        polys.extend([offset + i for i in p.vertices] for p in o.data.polygons)
        offset += len(co)
    return BVHTree.FromPolygons(verts, polys)


def fit_error(shape_objs, scan_pts, scan_bvh, diag: float, rng, n: int = 2000) -> dict:
    """Symmetric surface distance between primitives and scan, relative to the bbox diagonal."""
    shape_pts = np.vstack([sample_surface(o, max(64, n // max(1, len(shape_objs))), rng) for o in shape_objs])
    shape_bvh = _bvh(shape_objs)
    d_scan = np.array([(shape_bvh.find_nearest(mathutils.Vector(p))[3] or 0.0) for p in scan_pts])
    d_shape = np.array([(scan_bvh.find_nearest(mathutils.Vector(p))[3] or 0.0) for p in shape_pts])
    both = np.concatenate([d_scan, d_shape])
    return {"error": round(float(0.5 * (d_scan.mean() + d_shape.mean()) / diag), 5),
            "p95_m": round(float(np.percentile(both, 95)), 5)}


def refine_components(comps, scan_norm):
    """Snap free (non-shared, non-boundary) sides of each region to the scan's points.

    ``scan_norm`` are scan points normalised to the asset bbox. Sides on the
    outer bbox or touching a neighbouring component are left alone so pieces
    keep meeting cleanly; each side moves at most 25% of the component size.
    """
    unions = [c for c in comps if c.get("operation", "union") == "union"]
    for c in unions:
        if not c.get("refine", True) or c.get("array_count", 1) > 1:
            continue
        lo, hi = np.array(c["min"], float), np.array(c["max"], float)
        ext = hi - lo
        free_lo = np.ones(3, bool)
        free_hi = np.ones(3, bool)
        for k in range(3):
            free_lo[k] = lo[k] > 0.005
            free_hi[k] = hi[k] < 0.995
            for o in unions:
                if o is c:
                    continue
                olo, ohi = np.array(o["min"], float), np.array(o["max"], float)
                others = [j for j in range(3) if j != k]
                overlap = all(olo[j] < hi[j] and ohi[j] > lo[j] for j in others)
                if overlap and abs(lo[k] - ohi[k]) < 0.01:
                    free_lo[k] = False
                if overlap and abs(hi[k] - olo[k]) < 0.01:
                    free_hi[k] = False
        if not (free_lo.any() or free_hi.any()):
            continue
        grow = 0.15 * ext
        qlo = np.where(free_lo, lo - grow, lo)
        qhi = np.where(free_hi, hi + grow, hi)
        inside = np.all((scan_norm >= qlo) & (scan_norm <= qhi), axis=1)
        pts = scan_norm[inside]
        if len(pts) < 60:
            continue
        p_lo, p_hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        new_lo = np.where(free_lo, np.clip(p_lo, lo - 0.25 * ext, lo + 0.25 * ext), lo)
        new_hi = np.where(free_hi, np.clip(p_hi, hi - 0.25 * ext, hi + 0.25 * ext), hi)
        if np.all(new_hi - new_lo > 1e-3):
            c["min"], c["max"] = new_lo.round(4).tolist(), new_hi.round(4).tolist()


def _alive(obj) -> bool:
    try:
        return obj.name in bpy.data.objects
    except ReferenceError:  # removed from bpy.data
        return False


def union_objects(objs, use_boolean: bool):
    """Merge objects into the first one (EXACT boolean union, or a plain join)."""
    base = objs[0]
    rest = objs[1:]
    if use_boolean:
        for o in rest:
            mod = base.modifiers.new("union", "BOOLEAN")
            mod.operation = "UNION"
            mod.solver = "EXACT"
            mod.object = o
            if hasattr(mod, "material_mode"):
                mod.material_mode = "TRANSFER"
            before = face_count(base)
            apply_modifier(base, mod)
            if face_count(base) == 0:
                raise RuntimeError(f"boolean union with {o.name} emptied the mesh (was {before} faces)")
            bpy.data.objects.remove(o, do_unlink=True)
    elif rest:
        select_only(*objs, active=base)
        bpy.ops.object.join()
    return base


def subtract_objects(base, cutters):
    for o in cutters:
        mod = base.modifiers.new("subtract", "BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.solver = "EXACT"
        mod.object = o
        apply_modifier(base, mod)
        bpy.data.objects.remove(o, do_unlink=True)


# ----------------------------------------------------------------- skyline layout
#
# Castles, churches and other composite buildings differ in how many towers they have and
# where, so no fixed recipe fits them (one cylinder stretched over a four-tower castle). Their
# layout is read from the scan instead: a top-down height map is split at the heights where
# the roofline separates into distinct peaks. Every peak becomes a tower - round or square,
# with a cone/pyramid roof when it rises to a point, else a crenellated top - and the mass
# joining them becomes wall blocks.

def height_map(obj, lo, hi, grid: int) -> "np.ndarray":
    """(grid, grid) first-hit heights seen from above, normalised to 0..1; -1 = empty. Row = Y."""
    bvh = _bvh([obj])
    size = hi - lo
    top = float(hi[2] + 0.01 * size[2])
    down = mathutils.Vector((0.0, 0.0, -1.0))
    H = -np.ones((grid, grid))
    for i in range(grid):
        y = float(lo[1] + (i + 0.5) / grid * size[1])
        for j in range(grid):
            x = float(lo[0] + (j + 0.5) / grid * size[0])
            hit = bvh.ray_cast(mathutils.Vector((x, y, top)), down, float(size[2]) * 1.05)[0]
            if hit is not None:
                H[i, j] = (hit.z - lo[2]) / size[2]
    # 3x3 median over valid cells: scan noise must not split one roof into two peaks
    valid = H >= 0
    stack = np.stack([np.where(_shift(valid, dy, dx), _shift(H, dy, dx), np.nan)
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    return np.where(valid, np.nan_to_num(med, nan=-1.0), -1.0)


def _max_rectangle(mask) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
    """Largest all-True axis-aligned rectangle: (area, (row0, col0, row1, col1) inclusive)."""
    h, w = mask.shape
    heights = np.zeros(w, int)
    best = (0, None)
    for i in range(h):
        heights = np.where(mask[i], heights + 1, 0)
        stack: List[Tuple[int, int]] = []
        for j in range(w + 1):
            cur = int(heights[j]) if j < w else 0
            start = j
            while stack and stack[-1][1] >= cur:
                s0, hh = stack.pop()
                if hh * (j - s0) > best[0]:
                    best = (hh * (j - s0), (i - hh + 1, s0, i, j - 1))
                start = s0
            stack.append((start, cur))
    return best


def rectangle_cover(mask, max_rects: int = 6, coverage: float = 0.9, min_cells: int = 4):
    """A few rectangles covering ``mask`` (greedy largest-first)."""
    remaining = mask.copy()
    total = int(mask.sum())
    rects = []
    while len(rects) < max_rects and remaining.sum() > (1.0 - coverage) * total:
        area, rect = _max_rectangle(remaining)
        if rect is None or area < min_cells:
            break
        r0, c0, r1, c1 = rect
        remaining[r0:r1 + 1, c0:c1 + 1] = False
        if min(r1 - r0, c1 - c0) >= 1:  # at least 2 cells thick: no sliver walls
            rects.append(rect)
    return rects


def _peaks_above(H, region, t, min_cells):
    labels, sizes = _components(region & (H > t), connectivity4=False)
    subs = [labels == k for k, n in enumerate(sizes) if k and n >= min_cells]
    return [m for m in subs if np.percentile(H[m], 90) > t + 0.05]  # clearly above the split


def _skyline_elements(H, region, floor, depth=0, min_cells=6, band=0.06):
    """Split ``region`` (bool mask) at the lowest height where it separates into >= 2 peaks.

    Returns ("mass", mask, top) for the part joining the peaks (walls) and ("peak", mask, top)
    for every tower, recursively. The split level is raised by ``band`` above where the peaks
    first separate, so the noisy top of a wall stays with the wall instead of becoming a
    second, sliver-thin wall level inside a tower.
    """
    peak = float(np.percentile(H[region], 97))
    if depth < 4:
        for t in np.arange(floor + 0.03, peak - 0.03, 0.025):
            subs = _peaks_above(H, region, t, min_cells)
            if len(subs) < 2:
                continue
            higher = _peaks_above(H, region, t + band, min_cells)
            if len(higher) >= len(subs):
                subs, t = higher, t + band
            towers = np.any(subs, axis=0)
            mass = region & ~towers
            top = float(np.median(H[mass])) if mass.any() else float(t)
            out = [("mass", mass, min(top, float(t)))]
            for m in subs:
                out += _skyline_elements(H, m, float(t), depth + 1, min_cells, band)
            return out
    return [("peak", region, peak)]


def skyline_components(obj, lo, hi, p, materials) -> Tuple[List[dict], dict]:
    """Towers, wall blocks, roofs and battlements (normalised components) from the scan's roofline."""
    grid = int(p.get("skyline_grid", 64))
    size = hi - lo
    H = height_map(obj, lo, hi, grid)
    footprint = H > 0.03
    labels, sizes = _components(footprint, connectivity4=False)
    if len(sizes) > 1:  # main building only; stray blobs are scan noise
        footprint = labels == int(np.argmax(sizes[1:]) + 1)
    cell = size[:2] / grid  # metres per cell (x, y)
    min_cells = max(6, int(0.015 * footprint.sum()))
    elements = _skyline_elements(H, footprint, 0.0, min_cells=min_cells)
    stone = next((n for n in ("stone", "wall") if n in materials), p.get("default_material", ""))
    roof_mat = "roof" if "roof" in materials else stone
    merlon_h = min(1.8, 0.1 * float(size[2]))  # metres
    battlements = bool(p.get("battlements", True))
    comps, info = [], {"towers": 0, "wall_blocks": 0, "roofs": 0, "grid": grid}

    def norm_xy(col, row):
        return (col / grid, row / grid)

    def sink():
        # Every element reaches a little (and differently far) below the ground, and merlons sink
        # into what they stand on: exact booleans fail on coplanar faces. The final cut flattens it.
        return -0.01 - 0.0007 * len(comps)

    def box(name, c0, r0, c1, r1, z0, z1, material, **kw):
        z0 = sink() if z0 <= 0 else z0
        comps.append(dict(name=name, type="box", min=[c0 / grid, r0 / grid, z0], max=[c1 / grid, r1 / grid, z1],
                          axis="Z", refine=False, material=material, **kw))

    def wall_merlons(c0, r0, c1, r1, z):
        """Merlon rows on the long sides of a wall block's top."""
        mz = merlon_h / float(size[2])
        lx, ly = (c1 - c0) * cell[0], (r1 - r0) * cell[1]
        along_x = lx >= ly
        length = lx if along_x else ly
        n = int(min(24, max(3, round(length / 2.4))))
        d = 0.6 / float(cell[1] if along_x else cell[0])  # 0.6 m parapet depth, in cells
        # stand 8 cm proud of the wall face and stop short of its ends: no coplanar faces
        ex, ey = 0.08 / float(cell[0]), 0.08 / float(cell[1])
        if along_x:
            for rr0, rr1 in ((r0 - ey, r0 + d), (r1 - d, r1 + ey)):
                box("merlon", c0 + ex, rr0, c1 - ex, rr1, z - 0.3 * mz, z + mz, stone, array_count=n, array_axis="X")
        else:
            for cc0, cc1 in ((c0 - ex, c0 + d), (c1 - d, c1 + ex)):
                box("merlon", cc0, r0 + ey, cc1, r1 - ey, z - 0.3 * mz, z + mz, stone, array_count=n, array_axis="Y")

    for kind, region, top in elements[:24]:
        rows, cols = np.nonzero(region)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        ex, ey = (c1 - c0) * cell[0], (r1 - r0) * cell[1]
        fill = region.sum() / float((r1 - r0) * (c1 - c0))
        if kind == "mass":
            # walls reach one cell into the towers they join, so the union closes cleanly
            grown = region | (_dilate(region, 1) & footprint)
            rows, cols = np.nonzero(grown)
            r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
            for rr0, cc0, rr1, cc1 in rectangle_cover(grown[r0:r1, c0:c1], min_cells=min_cells):
                box("wall", c0 + cc0, r0 + rr0, c0 + cc1 + 1, r0 + rr1 + 1, 0.0, top, stone)
                info["wall_blocks"] += 1
                if battlements:
                    wall_merlons(c0 + cc0, r0 + rr0, c0 + cc1 + 1, r0 + rr1 + 1, top)
            continue
        info["towers"] += 1
        round_ = 0.55 <= fill <= 0.9 and 0.65 <= ex / max(ey, 1e-9) <= 1.55
        cy, cx = rows.mean() + 0.5, cols.mean() + 0.5
        if round_:
            radius = math.sqrt(region.sum() * cell[0] * cell[1] / math.pi)  # metres
            hx, hy = radius / cell[0], radius / cell[1]  # half extents in cells
            b0, b1 = (cx - hx, cy - hy), (cx + hx, cy + hy)
        else:
            radius = 0.5 * math.sqrt(ex * ey)
            b0, b1 = (c0, r0), (c1, r1)
        # Pointed roof when the centre clearly rises above the rim
        dist = np.hypot((cols + 0.5 - cx) * cell[0], (rows + 0.5 - cy) * cell[1])
        hs = H[rows, cols]
        centre = hs[dist <= 0.35 * radius]
        rim = hs[(dist >= 0.6 * radius)]
        centre_h = float(np.median(centre)) if len(centre) else top
        rim_h = float(np.median(rim)) if len(rim) else top
        pointed = (top - rim_h) * size[2] > 0.45 * radius and centre_h > rim_h
        name = f"tower_{info['towers']}"
        body_type = "cylinder" if round_ else "box"
        if pointed:
            eave = max(rim_h, 0.15 * top)
            comps.append(dict(name=name, type=body_type, min=[b0[0] / grid, b0[1] / grid, sink()],
                              max=[b1[0] / grid, b1[1] / grid, eave], axis="Z", segments=32, refine=False,
                              material=stone))
            o = 0.06  # eaves overhang
            wx, wy = (b1[0] - b0[0]) * o, (b1[1] - b0[1]) * o
            comps.append(dict(name=f"{name}_roof", type="cone" if round_ else "pyramid",
                              min=[(b0[0] - wx) / grid, (b0[1] - wy) / grid, eave - 0.004],
                              max=[(b1[0] + wx) / grid, (b1[1] + wy) / grid, top], axis="+Z", taper=0.0,
                              segments=32, refine=False, material=roof_mat))
            info["roofs"] += 1
            continue
        mz = merlon_h / float(size[2]) if battlements else 0.0
        comps.append(dict(name=name, type=body_type, min=[b0[0] / grid, b0[1] / grid, sink()],
                          max=[b1[0] / grid, b1[1] / grid, top - mz], axis="Z", segments=32, refine=False,
                          material=stone))
        if mz:
            if round_:
                teeth = int(min(32, max(8, 2 * round(math.pi * 2 * radius / 4.8))))
                gx, gy = 0.02 * (b1[0] - b0[0]), 0.02 * (b1[1] - b0[1])  # parapet slightly proud
                comps.append(dict(name=f"{name}_merlons", type="merlon_ring",
                                  min=[(b0[0] - gx) / grid, (b0[1] - gy) / grid, top - 1.3 * mz],
                                  max=[(b1[0] + gx) / grid, (b1[1] + gy) / grid, top],
                                  axis="Z", segments=teeth, thickness=min(0.35, 0.8 / max(radius, 1e-6)),
                                  refine=False, material=stone))
            else:
                bx0, by0, bx1, by1 = b0[0], b0[1], b1[0], b1[1]
                wall_merlons(bx0, by0, bx1, by1, top - mz)
                depth_x, depth_y = 0.6 / cell[0], 0.6 / cell[1]
                nx = int(min(24, max(3, round(ey / 2.4))))
                for cc0, cc1 in ((bx0 - 0.08 / cell[0], bx0 + depth_x),
                                 (bx1 - depth_x, bx1 + 0.08 / cell[0])):  # the other two sides
                    box("merlon", cc0, by0 + depth_y, cc1, by1 - depth_y, top - 1.3 * mz, top, stone,
                        array_count=nx, array_axis="Y")
    info["elements"] = len(comps)
    info["height_map_cells"] = int(footprint.sum())
    log(f"skyline: {info['towers']} towers ({info['roofs']} roofed), {info['wall_blocks']} wall blocks")
    return comps, info


# ----------------------------------------------------------------- three-view blueprint
#
# Hard-surface objects without a recipe (guns, tools, machinery) are rebuilt the
# way artists block them out from blueprints: trace the side, top and front
# silhouettes, extrude each through the asset and keep their intersection (a
# visual hull). Edges come out straight and crisp, holes such as a trigger-guard
# opening survive, and no primitive vocabulary is needed.

BLUEPRINT_VIEWS = {  # view: (u axis, v axis, extrusion axis)
    "side": (0, 2, 1),   # X-Z profile, extruded along Y
    "top": (0, 1, 2),    # X-Y plan, extruded along Z
    "front": (1, 2, 0),  # Y-Z section, extruded along X
}


def _components(mask, connectivity4=True):
    """Label connected True regions; returns (labels, sizes) with labels 1..n."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    sizes = [0]
    for sy, sx in np.argwhere(mask):
        if labels[sy, sx]:
            continue
        n = len(sizes)
        stack = [(sy, sx)]
        labels[sy, sx] = n
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = n
                    stack.append((ny, nx))
        sizes.append(count)
    return labels, sizes


def silhouette_mask(pts, axes, origin, cell, shape, close_r=2, min_hole_cells=12):
    """Rasterise points onto the (u, v) plane, close gaps, drop specks and pinholes."""
    u_ax, v_ax = axes
    mask = np.zeros(shape, bool)
    j = ((pts[:, u_ax] - origin[0]) / cell).astype(int)
    i = ((pts[:, v_ax] - origin[1]) / cell).astype(int)
    ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    mask[i[ok], j[ok]] = True
    mask = _erode(_dilate(mask, close_r), close_r)
    labels, sizes = _components(mask)
    if len(sizes) > 1:
        largest = max(sizes[1:])
        mask = np.isin(labels, [k for k, n in enumerate(sizes) if k and n >= 0.02 * largest])
    holes, hsizes = _components(~mask)
    border = set(np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]])).tolist())
    for k, n in enumerate(hsizes):
        if k and k not in border and n < min_hole_cells:
            mask[holes == k] = True  # sampling pinholes; real openings are larger
    return mask


def boundary_loops(mask):
    """Closed pixel-corner loops around True regions (outer CCW, holes CW)."""
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), bool)
    pad[1:-1, 1:-1] = mask
    out = {}
    for y, x in np.argwhere(mask):
        py, px = y + 1, x + 1
        if not pad[py - 1, px]:
            out.setdefault((x, y), []).append((x + 1, y))
        if not pad[py, px + 1]:
            out.setdefault((x + 1, y), []).append((x + 1, y + 1))
        if not pad[py + 1, px]:
            out.setdefault((x + 1, y + 1), []).append((x, y + 1))
        if not pad[py, px - 1]:
            out.setdefault((x, y + 1), []).append((x, y))
    loops = []
    while out:
        start = next(iter(out))
        loop, cur, prev_dir = [start], start, None
        while True:
            options = out[cur]
            if len(options) == 1 or prev_dir is None:
                nxt = options[0]
            else:
                # Diagonal contact: turn left so regions stay 4-connected and loops never cross
                def turn(o):
                    d = (o[0] - cur[0], o[1] - cur[1])
                    return prev_dir[0] * d[1] - prev_dir[1] * d[0]
                nxt = max(options, key=turn)
            options.remove(nxt)
            if not options:
                del out[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start:
                break
            loop.append(cur)
        if len(loop) >= 4:
            loops.append(np.array(loop, float))
    return loops


def _dp(points, eps):
    """Douglas-Peucker on an open polyline (inclusive endpoints)."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    norm = np.linalg.norm(ab)
    if norm < 1e-12:
        d = np.linalg.norm(points - a, axis=1)
    else:
        d = np.abs(ab[0] * (points[:, 1] - a[1]) - ab[1] * (points[:, 0] - a[0])) / norm
    k = int(np.argmax(d))
    if d[k] <= eps:
        return np.vstack([a, b])
    return np.vstack([_dp(points[:k + 1], eps)[:-1], _dp(points[k:], eps)])


def simplify_loop(loop, eps):
    far = int(np.argmax(np.linalg.norm(loop - loop[0], axis=1)))
    first = _dp(np.vstack([loop[:far + 1]]), eps)
    second = _dp(np.vstack([loop[far:], loop[:1]]), eps)
    out = np.vstack([first[:-1], second[:-1]])
    return out if len(out) >= 3 else loop


def prism_object(name, loops, axes, extrude_axis, origin, cell, depth_lo, depth_hi):
    """Fill the (simplified) loops - holes included - and extrude through the asset."""
    u_ax, v_ax = axes
    bm = bmesh.new()
    edges = []
    for loop in loops:
        verts = []
        for x, y in loop:
            co = [0.0, 0.0, 0.0]
            co[u_ax] = origin[0] + x * cell
            co[v_ax] = origin[1] + y * cell
            co[extrude_axis] = depth_lo
            verts.append(bm.verts.new(co))
        edges += [bm.edges.new((verts[i], verts[(i + 1) % len(verts)])) for i in range(len(verts))]
    normal = [0.0, 0.0, 0.0]
    normal[extrude_axis] = 1.0
    faces = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=edges, normal=normal)["geom"]
    faces = [f for f in faces if isinstance(f, bmesh.types.BMFace)]
    ext = bmesh.ops.extrude_face_region(bm, geom=faces)
    shift = [0.0, 0.0, 0.0]
    shift[extrude_axis] = depth_hi - depth_lo
    bmesh.ops.translate(bm, vec=shift, verts=[v for v in ext["geom"] if isinstance(v, bmesh.types.BMVert)])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def build_blueprint(scan, lo, hi, resolution=256, rng=None, views=("side", "top", "front")):
    """Visual hull of the scan's orthographic silhouettes. Returns (object, info)."""
    rng = rng or np.random.default_rng(0)
    pts = sample_surface(scan, 250000, rng)
    size = hi - lo
    cell = float(size.max()) / resolution
    pad = 4
    info = {}
    prisms = []
    for view in views:
        u_ax, v_ax, e_ax = BLUEPRINT_VIEWS[view]
        origin = (lo[u_ax] - pad * cell, lo[v_ax] - pad * cell)
        shape = (int(math.ceil(size[v_ax] / cell)) + 2 * pad, int(math.ceil(size[u_ax] / cell)) + 2 * pad)
        mask = silhouette_mask(pts, (u_ax, v_ax), origin, cell, shape)
        loops = [simplify_loop(lp, 0.75) for lp in boundary_loops(mask)]
        info[view] = {"loops": len(loops), "vertices": int(sum(len(lp) for lp in loops)),
                      "fill": round(float(mask.mean()), 3)}
        margin = 0.02 * float(size.max())
        prisms.append(prism_object(f"blueprint_{view}", loops, (u_ax, v_ax), e_ax, origin, cell,
                                   lo[e_ax] - margin, hi[e_ax] + margin))
    base = prisms[0]
    for other in prisms[1:]:
        mod = base.modifiers.new("hull", "BOOLEAN")
        mod.operation = "INTERSECT"
        mod.solver = "EXACT"
        mod.object = other
        apply_modifier(base, mod)
        bpy.data.objects.remove(other, do_unlink=True)
    bm = bmesh.new()
    bm.from_mesh(base.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=cell * 0.05)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(1.0), verts=bm.verts, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(base.data)
    bm.free()
    base.name = "blueprint"
    info["faces"] = face_count(base)
    return base, info


CLASSIFY_CANDIDATES = [
    {"type": "box", "bevel": 0.03}, {"type": "cylinder", "axis": "Z"}, {"type": "cylinder", "axis": "X"},
    {"type": "cylinder", "axis": "Y"}, {"type": "cone", "axis": "+Z", "taper": 0.0},
    {"type": "pyramid", "axis": "+Z", "taper": 0.0}, {"type": "sphere"}, {"type": "hemisphere", "axis": "+Z"},
    {"type": "half_cylinder", "axis": "X"}, {"type": "half_cylinder", "axis": "Y"},
    {"type": "wedge", "axis": "X"}, {"type": "wedge", "axis": "Y"},
]


def _expand_arrays(comps):
    """Blender-side twin of shape_templates.expand_arrays (n teeth, n-1 equal gaps)."""
    out = []
    for c in comps:
        n = int(c.get("array_count", 1))
        if n <= 1 or c.get("type") == "merlon_ring":
            out.append(c)
            continue
        axis = "XYZ".index(c.get("array_axis", "X"))
        lo_a, hi_a = float(c["min"][axis]), float(c["max"][axis])
        width = (hi_a - lo_a) / (2 * n - 1)
        for k in range(n):
            copy = dict(c, array_count=1, name=f"{c['name']}_{k + 1}", min=list(c["min"]), max=list(c["max"]))
            copy["min"][axis] = lo_a + 2 * k * width
            copy["max"][axis] = copy["min"][axis] + width
            out.append(copy)
    return out


def _is_inside(bvh, point) -> bool:
    """Ray parity, majority of three skewed directions (robust at edges and corners)."""
    votes = 0
    for d in ((0.577, 0.211, 0.789), (-0.302, 0.905, 0.299), (0.13, -0.42, -0.898)):
        direction = mathutils.Vector(d).normalized()
        origin, hits = mathutils.Vector(point), 0
        for _ in range(256):
            hit = bvh.ray_cast(origin, direction)
            if hit[0] is None:
                break
            hits += 1
            origin = hit[0] + direction * 1e-5
        votes += hits % 2
    return votes >= 2


def _lost_members(obj, members, lo, size) -> int:
    """Union members whose centre is not inside the result (an exact boolean silently dropped them)."""
    bvh = _bvh([obj])
    cut = [(lo + np.array(c["min"]) * size, lo + np.array(c["max"]) * size) for c in members
           if c.get("operation") == "subtract"]
    lost = 0
    for c in members:
        if c.get("operation", "union") != "union" or c.get("type") in ("torus", "merlon_ring"):
            continue  # rings have no material at their centre
        centre = lo + (np.array(c["min"]) + np.array(c["max"])) / 2.0 * size
        if any(np.all(centre >= a) and np.all(centre <= b) for a, b in cut):
            continue
        if not _is_inside(bvh, centre):
            lost += 1
    return lost


def _cut_below(obj, lo, size, z):
    ground = region_box("ground", lo - size, np.array([lo[0] + 2 * size[0], lo[1] + 2 * size[1], z]))
    boolean(obj, ground, "DIFFERENCE")
    bpy.data.objects.remove(ground, do_unlink=True)


def build_component_groups(comps, lo, size, slots, p, report, ground_z=None):
    """Build the body ("" part) and one object per part name; every group ends watertight.

    ``ground_z``: components reaching below it (skyline layouts sink them to avoid coplanar
    bottoms) are cut flat there.
    """
    groups = {}
    for c in comps:
        groups.setdefault(c.get("part") or "", []).append(c)
    built = {}
    for part_name, members in groups.items():
        unions = [component_object(c, lo, size, slots) for c in members if c.get("operation", "union") == "union"]
        cutters = [component_object(c, lo, size, slots) for c in members if c.get("operation") == "subtract"]
        if not unions:
            for o in cutters:
                bpy.data.objects.remove(o, do_unlink=True)
            continue
        use_boolean = p.get("union", True)
        try:
            obj = union_objects(unions, use_boolean)
        except RuntimeError as exc:
            log(f"primitives: {exc}; joining without boolean")
            obj = union_objects([o for o in unions if _alive(o)], False)
            use_boolean = False
        subtract_objects(obj, cutters)  # openings: doors, arches, crenels
        lost = _lost_members(obj, members, lo, size) if use_boolean else 0
        if lost:
            log(f"primitives: union dropped {lost} component(s) of '{part_name or 'body'}'")
        broken = bool(repair_manifold(obj)["open_edges_after"] or lost)
        if ground_z is not None and not broken:  # booleans need a closed mesh: repair first, then cut
            _cut_below(obj, lo, size, ground_z)
            broken = bool(repair_manifold(obj)["open_edges_after"])
        if broken and use_boolean:
            # Coplanar unions can leave open seams: rebuild as cut, separately closed shells instead
            log(f"primitives: group '{part_name or 'body'}' broken after union; rebuilding as shells")
            bpy.data.objects.remove(obj, do_unlink=True)
            shells = [component_object(c, lo, size, slots) for c in members if c.get("operation", "union") == "union"]
            cutters = [component_object(c, lo, size, slots) for c in members if c.get("operation") == "subtract"]
            for shell in shells:
                for cutter in cutters:
                    boolean(shell, cutter, "DIFFERENCE")
                if ground_z is not None:
                    _cut_below(shell, lo, size, ground_z)
                repair_manifold(shell)
            for cutter in cutters:
                bpy.data.objects.remove(cutter, do_unlink=True)
            obj = union_objects(shells, False)
            report.setdefault("rebuilt_as_shells", []).append(part_name or "body")
        built[part_name] = obj

    if "" not in built:
        for o in built.values():
            bpy.data.objects.remove(o, do_unlink=True)
        raise RuntimeError("recipe has no body components")
    return built


def op_primitives(p: dict) -> dict:
    """Replace the cleaned scan with a primitive composite when it fits.

    ``components`` from a recipe/planner are refined against the scan and built;
    with no components every single primitive is tried and the best one kept.
    The scan stays in the file as the bake source, so Stage 4 projects its
    colours onto the clean shapes.
    """
    open_blend(p["input_blend"])
    target = find_role("target")
    scan = find_role("bake_source") or target
    lo, hi = bbox([target])
    size = np.maximum(hi - lo, 1e-6)
    diag = float(np.linalg.norm(size))
    rng = np.random.default_rng(int(p.get("seed", 0)))
    scan_pts = sample_surface(scan, 2000, rng)
    scan_bvh = _bvh([scan])
    materials = {m["name"]: m for m in p.get("materials", [])}
    slots = {}
    for name, m in materials.items():
        mat = bpy.data.materials.new(f"spec:{name}")
        mat.diffuse_color = (*m["base_color"], 1.0)
        slots[name] = mat

    comps = [dict(c) for c in p.get("components") or []]
    report = {"mode": p.get("mode", "auto")}
    if p.get("layout") == "skyline":
        comps, report["skyline"] = skyline_components(target, lo, hi, p, materials)
        report["source"] = "skyline"
        limit = float(p.get("skyline_max_error", 0.1))
    elif comps:
        if p.get("refine", True):
            refine_components(comps, (world_coords(scan) - lo) / size)
        report["source"] = "recipe"
        limit = float(p.get("recipe_max_error", 0.12))
    elif not p.get("classify", True):
        comps = []
        report["source"] = "blueprint"
        limit = float(p.get("blueprint_max_error", 0.05))
    else:
        scores = {}
        for cand in CLASSIFY_CANDIDATES:
            comp = dict(cand, name="candidate", min=[0, 0, 0], max=[1, 1, 1])
            obj = component_object(comp, lo, size, {})
            key = f"{cand['type']}_{cand.get('axis', '')}".rstrip("_")
            scores[key] = fit_error([obj], scan_pts, scan_bvh, diag, rng, n=1500)["error"]
            bpy.data.objects.remove(obj, do_unlink=True)
        best_key = min(scores, key=scores.get)
        best = CLASSIFY_CANDIDATES[list(scores).index(best_key)]
        comps = [dict(best, name="body", min=[0, 0, 0], max=[1, 1, 1], material=p.get("default_material", ""))]
        report.update(source=f"classified:{best_key}", candidates=dict(sorted(scores.items(), key=lambda kv: kv[1])))
        limit = float(p.get("classify_max_error", 0.035))
        log(f"primitives: best single-shape fit {best_key} ({scores[best_key]:.4f}); ranking {report['candidates']}")
        if scores[best_key] > limit and p.get("blueprint", True):
            comps, report["source"] = [], "blueprint"  # no clean single shape: try the blueprint hull
            limit = float(p.get("blueprint_max_error", 0.05))

    def build(comps):
        if report["source"] == "blueprint":
            hull, info = build_blueprint(scan, lo, hi, int(p.get("blueprint_resolution", 256)), rng)
            report["blueprint"] = info
            if materials:
                hull.data.materials.append(slots[p.get("default_material") or next(iter(materials))])
            return {"": hull}
        return build_component_groups(_expand_arrays(comps), lo, size, slots, p, report,
                                      ground_z=float(lo[2]) if report["source"].startswith("skyline") else None)

    built = build(comps)
    shape_objs = list(built.values())
    fit = fit_error(shape_objs, scan_pts, scan_bvh, diag, rng)
    use = p.get("mode", "auto") == "primitives" or fit["error"] <= limit
    log(f"primitives: {report['source']} fit error {fit['error']:.4f} (limit {limit}) -> "
        f"{'using primitives' if use else 'rejected'}")
    if not use and report["source"] == "recipe" and p.get("skyline_fallback"):
        # A building that doesn't match its recipe (a gatehouse with three towers): read its layout instead
        report["recipe_fit_error"] = fit["error"]
        for o in shape_objs:
            bpy.data.objects.remove(o, do_unlink=True)
        comps, report["skyline"] = skyline_components(target, lo, hi, p, materials)
        report["source"], limit = "skyline (recipe rejected)", float(p.get("skyline_max_error", 0.1))
        built = build(comps)
        shape_objs = list(built.values())
        fit = fit_error(shape_objs, scan_pts, scan_bvh, diag, rng)
        use = fit["error"] <= limit
        log(f"primitives: skyline fit error {fit['error']:.4f} (limit {limit})")
    report.update(fit_error=fit["error"], fit_p95_m=fit["p95_m"],
                  cage_hint=round(min(0.15, 1.5 * fit["p95_m"] / float(size.max())), 4))
    report["used"] = bool(use)
    if not use:
        log("primitives: keeping scan mesh")
        for o in shape_objs:
            bpy.data.objects.remove(o, do_unlink=True)
        return report

    name = target.name
    bpy.data.objects.remove(target, do_unlink=True)
    body = built[""]
    body.name = name
    body.data.name = name
    body["asset_role"] = "target"
    auto_smooth(body, float(p.get("auto_smooth_angle", 40.0)))
    spec_parts = {sp["name"]: sp for sp in p.get("parts", [])}
    parts_built = []
    for part_name, obj in built.items():
        if part_name == "":
            continue
        obj.name = f"{name}_{part_name}"
        obj["asset_role"] = "part"
        auto_smooth(obj, float(p.get("auto_smooth_angle", 40.0)))
        sp = spec_parts.get(part_name)
        plo, phi = bbox([obj])
        pivot = lo + np.array(sp["pivot"], float) * size if sp else np.array(
            [(plo[0] + phi[0]) / 2, (plo[1] + phi[1]) / 2, plo[2]])
        bpy.context.scene.cursor.location = pivot.tolist()
        select_only(obj)
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        obj.parent = body
        obj.matrix_parent_inverse = body.matrix_world.inverted()
        if sp:
            for key in ("motion", "axis", "limit_min", "limit_max", "description"):
                obj[key] = sp.get(key, "")
            add_motion_limits(obj, sp)
        parts_built.append(part_name)
    report["parts_built"] = parts_built
    report["components"] = comps
    report["metrics"] = {o.name: mesh_metrics(o) for o in asset_meshes()}
    save_blend(p["output_blend"])
    return report


# =============================================================================
# Stage 4 - UV unwrap and PBR textures
# =============================================================================

def unwrap(objs, method: str, angle_limit: float, margin: float):
    """Unwrap all objects into ONE shared UV layout (one texture set per asset)."""
    select_only(*objs)
    set_mode("EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if method == "seams":
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.edges_select_sharp(sharpness=math.radians(angle_limit))
        bpy.ops.mesh.mark_seam(clear=False)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=margin)
    elif method == "lightmap":
        call_op(bpy.ops.uv.lightmap_pack, PREF_CONTEXT="ALL_FACES", PREF_MARGIN_DIV=max(margin * 100, 0.05))
    else:
        call_op(bpy.ops.uv.smart_project, angle_limit=math.radians(angle_limit), island_margin=margin,
                area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    # Uniform texel density across islands and objects before packing
    bpy.ops.uv.average_islands_scale()
    call_op(bpy.ops.uv.pack_islands, rotate=True, margin=margin)
    set_mode("OBJECT")


def new_image(name: str, res: int, non_color: bool):
    img = bpy.data.images.new(name, width=res, height=res, alpha=False, float_buffer=False)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def emission_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat, nt, emit


def color_attribute_name(mesh):
    attrs = getattr(mesh, "color_attributes", None)
    if attrs is not None and len(attrs):
        return (attrs.active_color or attrs[0]).name
    vcols = getattr(mesh, "vertex_colors", None)
    if vcols is not None and len(vcols):
        return vcols[0].name
    return None


def prepare_bake_source(high, fallback_rgb):
    """Rewire the high-res mesh's materials to emit their base colour.

    Handles texture-mapped imports (InstantMesh), vertex-coloured imports
    (TripoSR/OpenLRM) and untextured meshes. Emission baking gives the pure
    albedo independent of lighting and metallic settings.
    """
    attr = color_attribute_name(high.data)
    slots = [s for s in high.material_slots if s.material]
    linked_color = False
    for slot in slots:
        mat = slot.material.copy()
        slot.material = mat
        if not mat.use_nodes:
            mat.use_nodes = True
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new(
            "ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        if bsdf is not None and bsdf.inputs["Base Color"].is_linked:
            nt.links.new(bsdf.inputs["Base Color"].links[0].from_socket, emit.inputs["Color"])
            linked_color = True
        elif bsdf is not None:
            emit.inputs["Color"].default_value = bsdf.inputs["Base Color"].default_value
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    if linked_color:
        return "material"
    mat, nt, emit = emission_material("bake_source_emit")
    if attr:
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = attr
        nt.links.new(vc.outputs["Color"], emit.inputs["Color"])
        source = f"color_attribute:{attr}"
    else:
        emit.inputs["Color"].default_value = (*fallback_rgb, 1.0)
        source = "flat"
    high.data.materials.clear()
    high.data.materials.append(mat)
    return source


def projection_material(projections, lo, size):
    """Planar-project tight-cropped reference views onto the assembly bbox.

    Each projection: {"image": path, "plane": "XZ" | "YZ", "flip_u": bool}.
    Coordinates are world positions normalised to the whole asset's bbox
    (``lo``, ``size``), so body and parts share one projection. Views are
    blended by how squarely each face points at the projector.
    """
    mat, nt, emit = emission_material("projection_emit")
    geo_pos = nt.nodes.new("ShaderNodeNewGeometry")
    norm = nt.nodes.new("ShaderNodeMapping")
    norm.vector_type = "POINT"
    norm.inputs["Scale"].default_value = [1.0 / max(float(v), 1e-6) for v in size]
    norm.inputs["Location"].default_value = [-float(a) / max(float(v), 1e-6) for a, v in zip(lo, size)]
    nt.links.new(geo_pos.outputs["Position"], norm.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(norm.outputs["Vector"], sep.inputs["Vector"])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], nsep.inputs["Vector"])
    colors, weights = [], []
    for proj in projections:
        img = bpy.data.images.load(str(Path(proj["image"]).resolve()))
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.extension = "EXTEND"
        u_src = sep.outputs["X"] if proj["plane"] == "XZ" else sep.outputs["Y"]
        if proj.get("flip_u"):
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = "SUBTRACT"
            inv.inputs[0].default_value = 1.0
            nt.links.new(u_src, inv.inputs[1])
            u_src = inv.outputs[0]
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(u_src, comb.inputs["X"])
        nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
        nt.links.new(comb.outputs["Vector"], tex.inputs["Vector"])
        w = nt.nodes.new("ShaderNodeMath")
        w.operation = "ABSOLUTE"
        nt.links.new(nsep.outputs["Y"] if proj["plane"] == "XZ" else nsep.outputs["X"], w.inputs[0])
        colors.append(tex.outputs["Color"])
        weights.append(w.outputs[0])
    if len(colors) == 1:
        nt.links.new(colors[0], emit.inputs["Color"])
        return mat
    # Two-view blend: factor = wB / (wA + wB)
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    nt.links.new(weights[0], add.inputs[0])
    nt.links.new(weights[1], add.inputs[1])
    div = nt.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    nt.links.new(weights[1], div.inputs[0])
    nt.links.new(add.outputs[0], div.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixRGB")
    nt.links.new(div.outputs[0], mix.inputs["Fac"])
    nt.links.new(colors[0], mix.inputs["Color1"])
    nt.links.new(colors[1], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], emit.inputs["Color"])
    return mat


def bake(obj, image, bake_type: str, high=None, cage_extrusion=0.0, margin=16, clear=True):
    """Bake one pass into ``image`` on ``obj``'s active material.

    ``clear=False`` accumulates into an image another object already baked to.
    """
    added = []
    for mat in {s.material for s in obj.material_slots if s.material}:
        nt = mat.node_tree
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = image
        node.select = True
        nt.nodes.active = node
        added.append((nt, node))
    scene = bpy.context.scene
    bake_settings = scene.render.bake
    bake_settings.margin = margin
    bake_settings.use_clear = clear
    if high is not None:
        bake_settings.use_selected_to_active = True
        bake_settings.cage_extrusion = cage_extrusion
        bake_settings.max_ray_distance = cage_extrusion * 4
        high.hide_render = False
        select_only(high, obj, active=obj)
    else:
        bake_settings.use_selected_to_active = False
        select_only(obj)
    kwargs = {"type": bake_type, "margin": margin, "use_selected_to_active": high is not None,
              "cage_extrusion": cage_extrusion, "use_clear": clear}
    if bake_type == "NORMAL":
        kwargs["normal_space"] = "TANGENT"
    bpy.ops.object.bake(**kwargs)
    for nt, node in added:
        nt.nodes.remove(node)
    if high is not None:
        high.hide_render = True


def image_to_array(img) -> "np.ndarray":
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)


def array_to_image(img, arr):
    img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).ravel())
    img.update()


def srgb_from_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def bake_material_ids(objs, res, names, bake_all):
    """Bake per-face material IDs from ``spec:<name>`` slots (set by primitive templates).

    Returns an (H, W) int array of indices into ``names`` (-1 = unknown), or
    None when no object carries template materials.
    """
    tagged = any(s.material and s.material.name.startswith("spec:") for o in objs for s in o.material_slots)
    if not tagged:
        return None
    img = bpy.data.images.new("material_ids", width=res, height=res, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    n = len(names)
    cache = {}
    for o in objs:
        if not o.material_slots:
            o.data.materials.append(None)
        for slot in o.material_slots:
            key = slot.material.name[5:] if slot.material and slot.material.name.startswith("spec:") else None
            idx = names.index(key) + 1 if key in names else 0
            if idx not in cache:
                mat, _, emit = emission_material(f"id_{idx}")
                emit.inputs["Color"].default_value = (idx / (n + 1),) * 3 + (1.0,)
                cache[idx] = mat
            slot.material = cache[idx]
    bake_all(img, "EMIT")
    ids = np.rint(image_to_array(img)[..., 0] * (n + 1)).astype(np.int64) - 1
    bpy.data.images.remove(img)
    return np.clip(ids, -1, n - 1)


def _colour_features(rgb):
    """(..., 3) sRGB -> (hue * saturation as 2D vector, scaled value) for material matching."""
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    sat = np.where(mx > 1e-4, (mx - mn) / np.maximum(mx, 1e-4), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.arctan2(np.sqrt(3.0) * (g - b), 2 * r - g - b)
    return np.stack([np.cos(hue) * sat * 2.0, np.sin(hue) * sat * 2.0, mx * 0.6], axis=-1)


def derive_pbr(albedo_img, rough_img, metal_img, materials, ids=None, reliable=None):
    """Material-ID per texel (template IDs, else nearest spec colour), then roughness/metallic.

    Roughness gets a small luminance-driven variation so surfaces read as
    worn rather than flat; metallic stays binary per material, which is what
    PBR validators expect.
    """
    albedo = image_to_array(albedo_img)
    rgb = albedo[..., :3]
    palette = np.array([srgb_from_linear(m["base_color"]) for m in materials], dtype=np.float32)
    # Compare hue/saturation first and brightness only weakly: shadowed wood is dark but still
    # saturated, bare metal is unsaturated at any brightness.
    feat_px, feat_pal = _colour_features(rgb), _colour_features(palette)
    dist = ((feat_px[:, :, None, :] - feat_pal[None, None, :, :]) ** 2).sum(axis=-1)
    colour_ids = dist.argmin(axis=-1)
    if reliable is not None:
        # Unseen texels carry the reconstruction's blurry, desaturated colours, which read as
        # bare metal; give them the dominant material instead of a guess.
        dominant = int(np.argmax([m.get("coverage", 1.0) for m in materials]))
        colour_ids = np.where(reliable, colour_ids, dominant)
    # Template materials are authoritative; colour clustering only fills unknown texels
    ids = colour_ids if ids is None else np.where(ids >= 0, ids, colour_ids)
    rough_vals = np.array([m["roughness"] for m in materials], dtype=np.float32)
    metal_vals = np.array([1.0 if m["metallic"] >= 0.5 else 0.0 for m in materials], dtype=np.float32)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rough = np.clip(rough_vals[ids] + 0.15 * (lum.mean() - lum), 0.04, 1.0)
    metal = metal_vals[ids]
    ones = np.ones_like(rough)
    array_to_image(rough_img, np.stack([rough, rough, rough, ones], axis=-1))
    array_to_image(metal_img, np.stack([metal, metal, metal, ones], axis=-1))
    share = {m["name"]: round(float((ids == i).mean()), 3) for i, m in enumerate(materials)}
    return share


def assign_material(objs, mat):
    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)


def build_pbr_material(objs, name, albedo, normal, rough, metal):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def tex(img, y, label):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        node.label = label
        node.location = (-500, y)
        return node

    nt.links.new(tex(albedo, 300, "Albedo").outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex(metal, 0, "Metallic").outputs["Color"], bsdf.inputs["Metallic"])
    nt.links.new(tex(rough, -300, "Roughness").outputs["Color"], bsdf.inputs["Roughness"])
    if normal is not None:
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.location = (-200, -600)
        nt.links.new(tex(normal, -600, "Normal").outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    assign_material(objs, mat)
    return mat


def save_image(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.filepath_raw = str(path.resolve())
    img.file_format = "PNG"
    img.save()


# ----------------------------------------------------------------- reference projection
#
# The sharpest colour information is the reference image itself, not the
# reconstruction's vertex colours. We find the camera that image was taken
# from (the reconstruction model's canonical input camera, refined so the
# scan's silhouette matches the image's alpha), then re-project the image onto
# every texel that camera can actually see.

GLTF_TO_BLENDER = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]  # (x, y, z)gltf -> (x, -z, y)


def _cam_basis(cam):
    pos, target, up = (np.asarray(cam[k], float) for k in ("position", "target", "up"))
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return pos, r, np.cross(r, f), f


def project_to_image(pts, cam):
    """World points -> (u, v) with v measured from the top of the image, and depth."""
    pos, r, u_axis, f = _cam_basis(cam)
    d = pts - pos
    z = d @ f
    t = 2.0 * math.tan(math.radians(cam["fov_deg"]) / 2.0)
    z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    sx, sy = cam.get("shift", (0.0, 0.0))
    return 0.5 + (d @ r) / (z_safe * t) + sx, 0.5 - (d @ u_axis) / (z_safe * t) + sy, z


def _mask_from_uv(u, v, grid):
    m = np.zeros((grid, grid), bool)
    ok = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    m[(v[ok] * grid).astype(int), (u[ok] * grid).astype(int)] = True
    return _dilate(m, 1)


def _dilate(mask, r):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = np.zeros_like(mask)
            ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
            shifted[yd, xd] = mask[ys, xs]
            out |= shifted
    return out


def _erode(mask, r):
    return ~_dilate(~mask, r)


def load_reference(path):
    """RGBA float array, row 0 = top of the image."""
    img = bpy.data.images.load(str(Path(path).resolve()))
    img.colorspace_settings.name = "Non-Color"  # keep stored (sRGB) values untouched
    arr = image_to_array(img)[::-1].copy()
    bpy.data.images.remove(img)
    return arr


def _resample_mask(alpha, grid):
    h, w = alpha.shape
    ys = ((np.arange(grid) + 0.5) / grid * h).astype(int)
    xs = ((np.arange(grid) + 0.5) / grid * w).astype(int)
    return alpha[ys][:, xs] > 0.5


def _orbit(cam, d_az, d_el, d_scale):
    pos, r, u, f = _cam_basis(cam)
    target = np.asarray(cam["target"], float)
    offset = pos - target
    up = np.asarray(cam["up"], float)
    rot = (mathutils.Matrix.Rotation(math.radians(d_az), 3, mathutils.Vector(up)) @
           mathutils.Matrix.Rotation(math.radians(d_el), 3, mathutils.Vector(r)))
    new = dict(cam)
    new["position"] = (target + np.array(rot) @ offset * d_scale).tolist()
    return new


def calibrate_camera(pts, ref_alpha, prior, grid=96, search_deg=15.0):
    """Refine the input camera so the projected scan silhouette matches the image alpha.

    Searches azimuth/elevation/distance around ``prior`` and solves the 2D
    principal shift by centroid alignment. Returns (camera, IoU).
    """
    ref = _resample_mask(ref_alpha, grid)
    if ref.sum() == 0:
        return prior, 0.0
    ref_c = np.argwhere(ref).mean(axis=0)[::-1] / grid  # (u, v)

    def score(cam):
        cam = dict(cam, shift=(0.0, 0.0))
        u, v, z = project_to_image(pts, cam)
        front = z > 0
        if front.sum() < 10:
            return 0.0, cam
        shift = np.clip(ref_c - np.array([u[front].mean(), v[front].mean()]), -0.15, 0.15)
        cam["shift"] = shift.tolist()
        m = _mask_from_uv(u[front] + shift[0], v[front] + shift[1], grid)
        inter, union = (m & ref).sum(), (m | ref).sum()
        return (inter / union if union else 0.0), cam

    best_iou, best = score(prior)
    for step, span in ((5.0, search_deg), (2.0, 4.0), (1.0, 1.0)):
        center = best
        for d_az in np.arange(-span, span + 1e-9, step):
            for d_el in np.arange(-span, span + 1e-9, step):
                for d_s in (0.92, 0.96, 1.0, 1.04, 1.08):
                    iou, cam = score(_orbit(center, d_az, d_el, d_s))
                    if iou > best_iou:
                        best_iou, best = iou, cam
    return best, float(best_iou)


def _bake_float(objs, name, res, node_output, bake_all):
    """Bake a Geometry-node output (Position / Normal) into a float image via emission."""
    img = bpy.data.images.new(name, width=res, height=res, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    mat, nt, emit = emission_material(f"bake_{name}")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    nt.links.new(geo.outputs[node_output], emit.inputs["Color"])
    assign_material(objs, mat)
    bake_all(img, "EMIT")
    arr = image_to_array(img)[..., :3].copy()
    bpy.data.images.remove(img)
    return arr


def _bilinear(img, u, v):
    """Sample an (H, W, C) top-down image at normalised (u, v)."""
    h, w = img.shape[:2]
    x = np.clip(u * w - 0.5, 0, w - 1.001)
    y = np.clip(v * h - 0.5, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _project_pass(P, N, candidates, cam, ref, bvh, mirror_axis=None, min_facing=0.15, full_facing=0.55,
                  scan_bvh=None, scan_tol=0.0):
    """One projection pass over ``candidates`` (bool mask). Returns (rows, cols, weights, rgb).

    With ``mirror_axis`` each texel samples the image at its mirror position across that
    plane through the origin (the asset is centred on X/Y), filling faces the camera never saw
    on symmetric assets.
    """
    pos = np.asarray(cam["position"], float)
    idx = np.argwhere(candidates)
    if not len(idx):
        return idx[:, 0], idx[:, 1], np.zeros(0), np.zeros((0, 3))
    pts = P[idx[:, 0], idx[:, 1]].copy()
    nrm = N[idx[:, 0], idx[:, 1]].copy()
    if mirror_axis is not None:
        a = "XYZ".index(mirror_axis)
        pts[:, a] *= -1.0
        nrm[:, a] *= -1.0
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)[:, None]
    to_cam = pos - pts
    dist = np.linalg.norm(to_cam, axis=1)
    dirs = to_cam / np.maximum(dist, 1e-9)[:, None]
    facing = (nrm * dirs).sum(axis=1)
    u, v, z = project_to_image(pts, cam)
    ok = (facing > min_facing) & (u > 0) & (u < 1) & (v > 0) & (v < 1) & (z > 0)
    sample = np.zeros((len(idx), 4))
    sample[ok] = _bilinear(ref, u[ok], v[ok])
    ok &= sample[:, 3] > 0.5
    # Occlusion: nothing may lie between the (mirrored) texel and the camera
    eps = 1e-3 * float(np.linalg.norm(np.ptp(pts, axis=0)) + 1e-9)
    for i in np.flatnonzero(ok):
        origin = mathutils.Vector(pts[i] + nrm[i] * eps)
        if bvh.ray_cast(origin, mathutils.Vector(dirs[i]), float(dist[i]) - 2 * eps)[0] is not None:
            ok[i] = False
        elif scan_bvh is not None and dist[i] > scan_tol:
            # The photo shows whatever the *real* object has in front of this texel. If the scan
            # has surface well in front of the template (a roof the template lacks), that pixel
            # is not this face's colour.
            far = mathutils.Vector(pts[i] + dirs[i] * scan_tol)
            if scan_bvh.ray_cast(far, mathutils.Vector(dirs[i]), float(dist[i]) - scan_tol)[0] is not None:
                ok[i] = False
    w = np.clip((facing - min_facing) / (full_facing - min_facing), 0.0, 1.0) * np.clip((sample[:, 3] - 0.5) * 2, 0, 1)
    w = np.where(ok, w * w * (3 - 2 * w), 0.0)  # smoothstep into the baked colours at grazing angles
    keep = w > 0
    return idx[keep, 0], idx[keep, 1], w[keep], sample[keep, :3]


def project_reference(objs, albedo, ref, cam, res, bake_all, mirror_axes=(), P=None, N=None, scan=None,
                      scan_tol=0.0, ids=None, palette=None):
    """Overwrite albedo texels the camera sees (then mirrored fills) with the reference colours."""
    P = _bake_float(objs, "position", res, "Position", bake_all) if P is None else P
    N = _bake_float(objs, "normal", res, "Normal", bake_all) if N is None else N
    covered = np.linalg.norm(N, axis=-1) > 0.5
    bvh = _bvh(objs)
    scan_bvh = _bvh([scan]) if scan is not None and scan_tol > 0 else None
    base = image_to_array(albedo)  # baked maps and the albedo share Blender's bottom-up row order
    weight = np.zeros((res, res))
    passes = {}
    for axis in (None, *mirror_axes):
        todo = covered & (weight < 0.5)
        rows, cols, w, rgb = _project_pass(P, N, todo, cam, ref, bvh, axis, scan_bvh=scan_bvh, scan_tol=scan_tol)
        if ids is not None and palette is not None and len(palette) > 1 and len(rows):
            # Template faces know their material: a stone wall must not take a red roof pixel
            # (a turret the template doesn't have, seen in front of the wall)
            d = ((_colour_features(rgb)[:, None, :] - _colour_features(palette)[None, :, :]) ** 2).sum(-1)
            own = ids[rows, cols]
            known = own >= 0
            d_own = np.where(known, d[np.arange(len(own)), np.maximum(own, 0)], 0.0)
            d_other = np.where(np.eye(len(palette), dtype=bool)[np.maximum(own, 0)], np.inf, d).min(axis=1)
            w = np.where(known & (d_own > 0.01) & (d_other < 0.5 * d_own), 0.0, w)
        if axis is not None:
            w = w * 0.85  # mirrored detail is a guess; let a little of the baked colour through
        base[rows, cols, :3] = base[rows, cols, :3] * (1 - w[:, None]) + rgb * w[:, None]
        weight[rows, cols] = np.maximum(weight[rows, cols], w)
        passes["direct" if axis is None else f"mirror_{axis}"] = int((w > 0.5).sum())
    array_to_image(albedo, base)
    reliable = weight > 0.5
    return {"passes": passes, "covered_texels": int(covered.sum()),
            "projected_fraction": round(float(reliable.sum()) / max(1, int(covered.sum())), 3),
            "_reliable": reliable, "_weight": weight}


def _shift(a, dy, dx):
    """Shift a 2D/3D array by (dy, dx) with zero fill (no wrap-around)."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0),
                                                                    max(-dx, 0):w + min(-dx, 0)]
    return out


def fill_unbaked(objs, albedo, res, bake_all, keep=None, fallback_rgb=(0.5, 0.5, 0.5), iterations=96):
    """Find texels inside UV islands the colour bake never reached and fill them.

    Selected-to-active rays miss where the clean surface is far from the scan (lid/body
    joints, silhouette edges) and leave black. Those texels are grown in from valid
    neighbours; anything still empty takes the dominant material colour.
    Returns the fraction of island texels that needed filling.
    """
    N = _bake_float(objs, "coverage", res, "Normal", bake_all)
    covered = np.linalg.norm(N, axis=-1) > 0.5
    arr = image_to_array(albedo)
    rgb = arr[..., :3]
    bad = covered & (rgb.max(axis=-1) < 0.015)
    if keep is not None:
        bad &= ~keep
    if float(np.mean(np.asarray(fallback_rgb))) < 0.03:
        return 0.0  # genuinely black material: dark texels are expected
    total = int(covered.sum())
    fraction = float(bad.sum()) / max(1, total)
    if not bad.any():
        return 0.0
    valid = covered & ~bad
    for _ in range(iterations):
        acc = np.zeros_like(rgb)
        cnt = np.zeros(rgb.shape[:2])
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = _shift(valid, dy, dx)
            acc += _shift(rgb * valid[..., None], dy, dx)
            cnt += v
        grow = bad & (cnt > 0)
        if not grow.any():
            break
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        valid |= grow
        bad &= ~grow
    rgb[bad] = np.asarray(fallback_rgb, dtype=rgb.dtype)
    arr[..., :3] = rgb
    array_to_image(albedo, arr)
    return round(fraction, 4)


# ----------------------------------------------------------------- material tiles
#
# A single reference image only shows one side. Template faces it never saw (backs, roofs, the
# far side of a tower) used to keep the reconstruction's colour guess, which is dark and smeared.
# Instead every template material gets a tiling texture, mapped triplanar in metres so bricks
# keep a constant real-world size: a patch cut from the reference image where that material is
# visible (de-lit and made seamless), else a procedural pattern in the material's colour.

def _box_blur(img, r):
    """Separable box blur with wrap-around (tiles are periodic)."""
    out = img.astype(np.float64)
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for k in range(-r, r + 1):
            acc += np.roll(out, k, axis=axis)
        out = acc / (2 * r + 1)
    return out


def _resize(img, n):
    """Bilinear resize of an (H, W, C) array to (n, n, C)."""
    h, w = img.shape[:2]
    ys, xs = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="ij")
    return _bilinear(img, xs.ravel(), ys.ravel()).reshape(n, n, -1)


def make_seamless(tile):
    """Cross-fade the tile with its half-offset copy so opposite edges match."""
    n = tile.shape[0]
    shifted = np.roll(np.roll(tile, n // 2, axis=0), n // 2, axis=1)
    ramp = 1.0 - np.abs(np.linspace(-1.0, 1.0, n))  # 1 in the middle, 0 at the edges
    w = np.minimum.outer(ramp, ramp)[..., None] ** 0.5
    return tile * w + shifted * (1.0 - w)


def _value_noise(n, cells, rng):
    """Periodic value noise in [0, 1] on an (n, n) grid."""
    g = rng.random((cells, cells))
    ys, xs = np.meshgrid(np.arange(n) * cells / n, np.arange(n) * cells / n, indexing="ij")
    x0, y0 = xs.astype(int), ys.astype(int)
    fx, fy = xs - x0, ys - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    x1, y1 = (x0 + 1) % cells, (y0 + 1) % cells
    a, b, c, d = g[y0, x0], g[y0, x1], g[y1, x0], g[y1, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def procedural_tile(name, rgb, n=256, seed=0):
    """(tile, metres) - seamless stone blocks / roof shingles / planks / grain in colour ``rgb`` (sRGB)."""
    rng = np.random.default_rng(seed)
    key = name.lower()
    noise = 0.7 * _value_noise(n, 32, rng) + 0.3 * _value_noise(n, 8, rng)
    shade = np.ones((n, n))
    ys, xs = np.mgrid[0:n, 0:n]
    if any(k in key for k in ("stone", "brick", "wall", "rock", "masonry")):
        rows, cols, metres = 8, 4, 2.4  # 0.6 x 0.3 m ashlar blocks
        row = ys * rows // n
        off = np.where(row % 2, n // (2 * cols), 0)
        col = ((xs + off) % n) * cols // n
        block = rng.normal(1.0, 0.07, (rows, cols))[row, col]
        mortar = ((ys % (n // rows)) < 2) | (((xs + off) % (n // cols)) < 2)
        shade = np.where(mortar, 0.72, block)
    elif any(k in key for k in ("roof", "shingle", "slate", "tile")):
        rows, cols, metres = 16, 12, 1.6
        row = ys * rows // n
        off = np.where(row % 2, n // (2 * cols), 0)
        col = ((xs + off) % n) * cols // n
        within = (ys % (n // rows)) / float(n // rows)
        shade = rng.normal(1.0, 0.06, (rows, cols))[row, col] * (0.78 + 0.22 * within)
    elif any(k in key for k in ("wood", "plank", "timber")):
        planks, metres = 6, 1.2
        plank = xs * planks // n
        grain = 0.94 + 0.06 * np.sin(2 * np.pi * (ys / n * 3 + 0.4 * _value_noise(n, 6, rng)) * 4)
        shade = rng.normal(1.0, 0.06, planks)[plank] * grain * np.where((xs % (n // planks)) < 2, 0.7, 1.0)
    else:
        metres = 1.0
    shade = shade * (0.95 + 0.1 * noise)
    tile = np.clip(np.asarray(rgb, float)[None, None, :3] * shade[..., None], 0.0, 1.0)
    return tile, metres


TILE_METRES = (("stone", 2.4), ("brick", 2.4), ("wall", 2.4), ("rock", 2.4), ("roof", 1.6), ("shingle", 1.6),
               ("slate", 1.6), ("tile", 1.6), ("wood", 1.2), ("plank", 1.2))


def reference_patch(ref, cam, pts, name="", n=256, asset_size=1.0):
    """(tile, metres) - a real-world-sized, uniform square of the reference image showing one material.

    ``pts`` are world positions of texels of that material the camera saw well. Among the
    windows lying wholly on that material, the one with the least large-scale variation (no
    window, door or shadow edge) closest to the material's typical colour is cut out.
    Returns (None, 0) when the material shows too small in the image.
    """
    if len(pts) < 200:
        return None, 0.0
    G = 96
    u, v, z = project_to_image(pts, cam)
    ok = (u > 0) & (u < 1) & (v > 0) & (v < 1)
    if ok.sum() < 200:
        return None, 0.0
    H, W = ref.shape[:2]
    mask = np.zeros((G, G), bool)
    mask[(v[ok] * G).astype(int), (u[ok] * G).astype(int)] = True
    mask = _erode(_dilate(mask, 1), 1)  # close the gaps between projected texels
    depth = float(np.median(z[ok]))
    metres_per_px = depth * 2.0 * math.tan(math.radians(cam["fov_deg"]) / 2.0) / W
    want_m = next((m for k, m in TILE_METRES if k in name.lower()), 1.0)
    want_m = min(want_m, max(0.25, 0.6 * asset_size))
    size = int(round(want_m / metres_per_px * G / W))
    size = min(max(size, 6), G // 3)
    cells = _resize(ref[..., :3], G)
    integ = lambda a: np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)) + ((0, 0),) * (a.ndim - 2))  # noqa: E731
    box = lambda I: I[size:, size:] - I[:-size, size:] - I[size:, :-size] + I[:-size, :-size]  # noqa: E731
    while size >= 6:
        fill = box(integ(mask.astype(np.float64)))
        hits = np.argwhere(fill >= 0.97 * size * size)
        if len(hits):
            break
        size -= 2
    else:
        return None, 0.0
    area = float(size * size)
    mean = box(integ(cells)) / area
    var = (box(integ(cells ** 2)) / area - mean ** 2).sum(axis=-1)
    typical = np.median(cells[mask], axis=0)
    score = var[hits[:, 0], hits[:, 1]] + 0.5 * ((mean[hits[:, 0], hits[:, 1]] - typical) ** 2).sum(axis=-1)
    r0, c0 = hits[int(np.argmin(score))]
    y0, y1 = int(r0 * H / G), int((r0 + size) * H / G)
    x0, x1 = int(c0 * W / G), int((c0 + size) * W / G)
    patch = _resize(ref[y0:y1, x0:x1, :3], n)
    # De-light: remove the photo's shading gradient (one lit side of a tower), keep the detail
    low = _box_blur(patch, max(2, n // 10))
    patch = np.clip(patch / np.maximum(low, 1e-3) * low.mean(axis=(0, 1)), 0.0, 1.0)
    return make_seamless(patch), (x1 - x0) * metres_per_px


def _sample_wrap(tile, a, b):
    return _bilinear(tile, np.mod(a, 1.0), np.mod(b, 1.0))


def triplanar(tile, metres, P, N):
    """Sample a tile at world positions/normals (N, 3) with triplanar blending."""
    w = np.abs(N) ** 4
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    q = P / max(metres, 1e-6)
    out = _sample_wrap(tile, q[:, 1], -q[:, 2]) * w[:, :1]   # X-facing: (y, z), z up
    out += _sample_wrap(tile, q[:, 0], -q[:, 2]) * w[:, 1:2]  # Y-facing: (x, z)
    out += _sample_wrap(tile, q[:, 0], q[:, 1]) * w[:, 2:3]   # Z-facing: (x, y)
    return out


def apply_material_tiles(albedo, P, N, ids, materials, baked, weight, ref=None, cam=None, asset_size=1.0,
                         surface_m2=1.0, trust_baked=False):
    """Give every unseen template texel its material's tiling texture (see the section comment).

    ``trust_baked``: the reconstruction carried real colours, so its guess may set a tile's colour
    when the camera never saw that material; otherwise the spec palette does.
    """
    arr = image_to_array(albedo)
    covered = np.linalg.norm(N, axis=-1) > 0.5
    texel_m = math.sqrt(surface_m2 / max(1, int(covered.sum())))  # texture resolution in metres
    weight = np.zeros(covered.shape) if weight is None else weight
    report = {}
    for i, m in enumerate(materials):
        texels = covered & (ids == i)
        if texels.sum() < 16:
            continue
        seen = texels & (weight > 0.6)
        tile, metres, source = None, 0.0, "procedural"
        if ref is not None and cam is not None:
            tile, metres = reference_patch(ref, cam, P[seen], m["name"], asset_size=asset_size)
            source = "reference" if tile is not None else source
        if tile is None:
            # Colour: what the camera saw > the reconstruction's (non-black) guess > the spec palette
            rgb = arr[seen][:, :3] if seen.sum() > 50 else (baked[texels][:, :3] if trust_baked else arr[:0, 0, :3])
            lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
            rgb = rgb[lum > 0.08]
            colour = np.median(rgb, axis=0) if len(rgb) > 50 else srgb_from_linear(m["base_color"])
            tile, metres = procedural_tile(m["name"], colour, seed=i)
        metres = min(metres, max(0.25, 0.6 * asset_size))  # small props: several repeats, not one
        # never finer than the texture can hold: a 0.3 m block row needs a few texels or it aliases
        metres = max(metres, 48 * texel_m if source == "procedural" else 24 * texel_m)
        rows, cols = np.nonzero(texels)
        t = triplanar(tile, metres, P[rows, cols], N[rows, cols])
        keep = weight[rows, cols][:, None]
        # projected detail stays where the camera saw the surface; the tile replaces the guess elsewhere
        arr[rows, cols, :3] = arr[rows, cols, :3] + (t - baked[rows, cols, :3]) * (1.0 - keep)
        report[m["name"]] = {"source": source, "tile_m": round(float(metres), 3),
                             "unseen_fraction": round(float(1.0 - keep.mean()), 3)}
    array_to_image(albedo, arr)
    return report


def resolve_camera(p, scene):
    """Prior input camera in final asset coordinates (None when the backend gave none)."""
    cam = p.get("reference_camera")
    if not cam:
        return None
    pos, target, up = (np.asarray(cam[k], float) for k in ("position", "target", "up"))
    if cam.get("coords") == "gltf":
        conv = np.array(GLTF_TO_BLENDER)
        pos, target, up = conv @ pos, conv @ target, conv @ up
    frame = np.array(scene.get("frame_matrix", np.eye(4).ravel().tolist()), float).reshape(4, 4)
    to_final = lambda x: (frame @ np.append(x, 1.0))[:3]  # noqa: E731
    up_final = frame[:3, :3] @ up
    return {"position": to_final(pos).tolist(), "target": to_final(target).tolist(),
            "up": (up_final / np.linalg.norm(up_final)).tolist(), "fov_deg": float(cam["fov_deg"])}


def op_texture(p: dict) -> dict:
    """UV-unwrap body + parts into one layout and bake one PBR texture set."""
    open_blend(p["input_blend"])
    obj = find_role("target")
    high = find_role("bake_source")
    if obj is None:
        raise RuntimeError("no target object in blend")
    objs = [obj] + [o for o in asset_meshes() if o.get("asset_role") == "part"]
    scene = bpy.context.scene
    device = enable_gpu(scene, p.get("use_gpu", True))
    scene.cycles.samples = int(p.get("bake_samples", 16))

    unwrap(objs, p["uv_method"], p["uv_angle_limit"], p["island_margin"])
    res = int(p["resolution"])
    tex_dir = Path(p["texture_dir"])
    name = obj.name
    albedo = new_image(f"{name}_albedo", res, non_color=False)
    normal = new_image(f"{name}_normal", res, non_color=True)
    rough = new_image(f"{name}_roughness", res, non_color=True)
    metal = new_image(f"{name}_metallic", res, non_color=True)

    lo, hi = bbox(objs)
    size = float((hi - lo).max())
    cage = float(p["cage_extrusion"]) * size
    margin = int(p["bake_margin_px"])
    materials = p.get("materials") or [{"name": "base", "base_color": [0.5] * 3, "metallic": 0.0,
                                        "roughness": 0.6}]
    dominant = max(materials, key=lambda m: m.get("coverage", 1.0))
    report = {"device": device, "objects": [o.name for o in objs]}

    def bake_all(image, bake_type, from_high):
        for i, o in enumerate(objs):
            bake(o, image, bake_type, high=high if from_high else None, cage_extrusion=cage,
                 margin=margin, clear=(i == 0))

    # --- Material IDs from primitive templates (before materials are replaced)
    names = [m["name"] for m in materials]
    material_ids = bake_material_ids(objs, res, names, lambda img, t: bake_all(img, t, from_high=False))
    report["material_ids"] = material_ids is not None

    # --- Albedo
    source = p["albedo_source"]
    if source == "projection" and p.get("projections"):
        assign_material(objs, projection_material(p["projections"], lo, hi - lo))
        bake_all(albedo, "EMIT", from_high=False)
        report["albedo"] = "projection"
    elif source == "bake" and high is not None:
        report["albedo"] = prepare_bake_source(high, dominant["base_color"])
        assign_material(objs, emission_material("bake_target")[0])
        bake_all(albedo, "EMIT", from_high=True)
    else:
        col = srgb_from_linear(dominant["base_color"]).tolist()
        array_to_image(albedo, np.tile(np.array(col + [1.0], dtype=np.float32), (res, res, 1)))
        report["albedo"] = "procedural"

    # --- Sharp detail: re-project the reference image the 3D model was reconstructed from
    ref_path = p.get("reference_image")
    reliable, weight, ref, cam = None, None, None, None
    tiles = material_ids is not None and p.get("unseen_fill", "tiles") == "tiles"
    baked = image_to_array(albedo).copy() if tiles else None
    P = N = None
    if tiles or (p.get("project_reference", True) and ref_path):
        bake_plain = lambda img, t: bake_all(img, t, from_high=False)  # noqa: E731
        P = _bake_float(objs, "position", res, "Position", bake_plain)
        N = _bake_float(objs, "normal", res, "Normal", bake_plain)
    if p.get("project_reference", True) and ref_path and Path(ref_path).is_file():
        prior = resolve_camera(p, scene)
        if prior is None:
            report["projection"] = {"skipped": "reconstruction backend reports no input camera"}
        else:
            ref = load_reference(ref_path)
            scan = high if high is not None else obj
            pts = np.vstack([sample_surface(scan, 20000, np.random.default_rng(0))])
            cam, iou = calibrate_camera(pts, ref[..., 3], prior)
            proj = {"silhouette_iou": round(iou, 3), "camera": cam}
            if iou >= float(p.get("min_projection_iou", 0.7)):
                t0 = time.time()
                proj.update(project_reference(objs, albedo, ref, cam, res,
                                              lambda img, t: bake_all(img, t, from_high=False),
                                              mirror_axes=tuple(p.get("mirror_axes") or ()), P=P, N=N,
                                              # templates: only where the scan agrees there is surface
                                              scan=high if material_ids is not None else None,
                                              scan_tol=max(0.03 * size, 2.0 * cage), ids=material_ids,
                                              palette=np.array([srgb_from_linear(m["base_color"])
                                                                for m in materials])))
                proj["seconds"] = round(time.time() - t0, 1)
                weight = proj.pop("_weight", None)
                mask = proj.pop("_reliable", None)
                if mask is not None and proj["projected_fraction"] >= 0.15:
                    reliable = mask
            else:
                proj["skipped"] = "silhouette does not match the reference image"
                ref, cam = None, None  # the camera is wrong, so are patches cut through it
            report["projection"] = proj
            log(f"projection: IoU {iou:.3f}, {proj.get('projected_fraction', 0)} of texels")

    # --- Faces the camera never saw: tiling texture per template material instead of the guess
    if tiles:
        area = sum(sum(poly.area for poly in o.data.polygons) for o in objs)
        report["material_tiles"] = apply_material_tiles(
            albedo, P, N, material_ids, materials, baked, weight, ref, cam, asset_size=size, surface_m2=area,
            trust_baked=str(report.get("albedo", "")).startswith(("color_attribute", "material")))
        log(f"material tiles: {report['material_tiles']}")

    # --- Black texels the bake never reached (QA BUG-13): measure, then fill from neighbours
    report["unbaked_fraction"] = fill_unbaked(
        objs, albedo, res, lambda img, t: bake_all(img, t, from_high=False), keep=reliable,
        fallback_rgb=srgb_from_linear(dominant["base_color"]).tolist())
    if report["unbaked_fraction"]:
        log(f"texture: filled {report['unbaked_fraction']:.1%} unbaked texels")

    # --- Normal (high -> low detail transfer). Off for primitive templates by default:
    # it would re-introduce the scan's lumpiness as shading.
    if high is not None and p.get("bake_normal", True):
        assign_material(objs, emission_material("bake_normal")[0])
        bake_all(normal, "NORMAL", from_high=True)
        report["normal"] = "baked"
    else:
        normal = None
        report["normal"] = "none"

    # --- Roughness / Metallic
    report["material_coverage"] = derive_pbr(albedo, rough, metal, materials, material_ids, reliable)

    for img, suffix in ((albedo, "albedo"), (normal, "normal"), (rough, "roughness"), (metal, "metallic")):
        if img is not None:
            save_image(img, tex_dir / f"{name}_{suffix}.png")
    build_pbr_material(objs, f"M_{name}", albedo, normal, rough, metal)
    report["textures"] = {s: str((tex_dir / f"{name}_{s}.png").resolve())
                          for s in ("albedo", "normal", "roughness", "metallic")
                          if s != "normal" or normal is not None}

    if high is not None and not p.get("keep_bake_source", False):
        bpy.data.objects.remove(high, do_unlink=True)
    for m in list(bpy.data.materials):
        if m.users == 0:
            bpy.data.materials.remove(m)
    report["metrics"] = {o.name: mesh_metrics(o) for o in objs}
    save_blend(p["output_blend"])
    log(f"texture done: stretch={[m.get('stretch') for m in report['metrics'].values()]}")
    return report


# =============================================================================
# Stage 5a - segmentation of mechanical parts
# =============================================================================

def close_small_holes(obj, max_sides: int = 16) -> int:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary = [e for e in bm.edges if e.is_boundary]
    filled = 0
    if boundary:
        res = bmesh.ops.holes_fill(bm, edges=boundary, sides=max_sides)
        filled = len(res["faces"])
        if res["faces"]:
            bmesh.ops.triangulate(bm, faces=res["faces"])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
    bm.free()
    return filled


def region_box(name, rmin, rmax):
    """Axis-aligned cutter box spanning [rmin, rmax] (world space)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    size = np.maximum(rmax - rmin, 1e-6)
    center = (rmin + rmax) / 2.0
    bmesh.ops.scale(bm, vec=size.tolist(), verts=bm.verts)
    bmesh.ops.translate(bm, vec=center.tolist(), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    box = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(box)
    box.hide_render = True
    return box


def boolean(obj, cutter, operation):
    mod = obj.modifiers.new(f"bool_{operation.lower()}", "BOOLEAN")
    mod.operation = operation
    mod.solver = "EXACT"
    mod.object = cutter
    if hasattr(mod, "material_mode"):
        mod.material_mode = "INDEX"  # cut faces keep the object's own material
    apply_modifier(obj, mod)


def has_surface_inside(obj, rmin, rmax, tol) -> bool:
    """True if any face of ``obj`` lies off the box planes, i.e. came from the original surface."""
    mw = obj.matrix_world
    for poly in obj.data.polygons:
        c = np.array(mw @ poly.center)
        if np.min(np.abs(np.concatenate([c - rmin, rmax - c]))) > tol:
            return True
    return False


def separate_island_in_region(body, rmin, rmax, tol):
    """If a loose shell lies inside the region, split it off as-is (best case)."""
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.faces.ensure_lookup_table()
    islands = _islands(bm)
    if len(islands) < 2:
        bm.free()
        return None
    largest = max(islands, key=len)
    picked = []
    for island in islands:
        if island is largest:
            continue
        co = np.array([v.co[:] for f in island for v in f.verts])
        if np.all(co.min(axis=0) >= rmin - tol) and np.all(co.max(axis=0) <= rmax + tol):
            picked.extend(f.index for f in island)
    bm.free()
    if not picked:
        return None
    picked = set(picked)
    for poly in body.data.polygons:
        poly.select = poly.index in picked
    select_only(body)
    set_mode("EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    set_mode("OBJECT")
    return [o for o in bpy.context.selected_objects if o is not body][0]


def add_motion_limits(obj, part):
    """Limit constraints around the part's rest pose (local space)."""
    axis_idx = "XYZ".index(part.get("axis", "Y"))
    lo_lim, hi_lim = float(part.get("limit_min", 0.0)), float(part.get("limit_max", 0.0))
    if part.get("motion") == "rotate":
        con = obj.constraints.new("LIMIT_ROTATION")
        rest = obj.rotation_euler
        for i, a in enumerate("xyz"):
            setattr(con, f"use_limit_{a}", True)
            setattr(con, f"min_{a}", rest[i] + (math.radians(lo_lim) if i == axis_idx else 0.0))
            setattr(con, f"max_{a}", rest[i] + (math.radians(hi_lim) if i == axis_idx else 0.0))
        con.owner_space = "LOCAL"
    elif part.get("motion") == "translate":
        con = obj.constraints.new("LIMIT_LOCATION")
        rest = obj.location
        for i, a in enumerate("xyz"):
            setattr(con, f"use_min_{a}", True)
            setattr(con, f"use_max_{a}", True)
            setattr(con, f"min_{a}", rest[i] + (lo_lim if i == axis_idx else 0.0))
            setattr(con, f"max_{a}", rest[i] + (hi_lim if i == axis_idx else 0.0))
        con.owner_space = "LOCAL"


def op_segment(p: dict) -> dict:
    """Split movable parts off the body using normalised bbox regions.

    AI reconstructions fuse everything into one shell, so the default is a
    solid boolean cut (part = body AND box, body = body - box) which keeps
    both pieces watertight with planar mating faces. Parts that already are
    separate loose shells are split off untouched.
    """
    open_blend(p["input_blend"])
    body = find_role("target")
    lo, hi = bbox([body])
    size = hi - lo
    pad = 0.01 * float(size.max())
    clearance = float(p.get("clearance_mm", 0.0)) / 1000.0
    results, skipped = [], []
    for part in p.get("parts", []):
        rmin = lo + np.array(part["region_min"], dtype=float) * size
        rmax = lo + np.array(part["region_max"], dtype=float) * size
        # Push box faces that sit on the bbox boundary outward to avoid coplanar booleans
        rmin = np.where(np.array(part["region_min"]) <= 0.0, rmin - pad, rmin)
        rmax = np.where(np.array(part["region_max"]) >= 1.0, rmax + pad, rmax)
        faces_before = face_count(body)

        new = separate_island_in_region(body, rmin, rmax, pad)
        method = "loose_shell"
        if new is None:
            method = "boolean_cut"
            new = body.copy()
            new.data = body.data.copy()
            bpy.context.scene.collection.objects.link(new)
            box = region_box("cut_box", rmin, rmax)
            boolean(new, box, "INTERSECT")
            reason = None
            if face_count(new) == 0:
                reason = "region does not intersect the mesh"
            elif not has_surface_inside(new, rmin, rmax, pad * 0.1):
                # Box lies wholly inside the solid: no real part surface to extract
                reason = "region is enclosed by the body (no part surface)"
            if reason:
                bpy.data.objects.remove(new, do_unlink=True)
                bpy.data.objects.remove(box, do_unlink=True)
                skipped.append({"name": part["name"], "reason": reason})
                log(f"segment: skipping {part['name']} ({reason})")
                continue
            if clearance > 0:
                # 3D-print clearance: remove a slightly larger box from the body
                bpy.data.objects.remove(box, do_unlink=True)
                box = region_box("cut_box", rmin - clearance, rmax + clearance)
            boolean(body, box, "DIFFERENCE")
            bpy.data.objects.remove(box, do_unlink=True)
            for o in (body, new):  # exact booleans on n-gon meshes can leave slivers open
                close_small_holes(o)
                if repair_manifold(o)["open_edges_after"] and p.get("voxel_seal", True):
                    voxel_seal(o, int(p.get("voxel_resolution", 192)))
            if face_count(body) < 0.2 * faces_before:
                log(f"segment: warning - cutting {part['name']} removed most of the body")
        new.name = f"{body.name}_{part['name']}"
        new["asset_role"] = "part"
        new.modifiers.clear()

        pivot = lo + np.array(part["pivot"], dtype=float) * size
        bpy.context.scene.cursor.location = pivot.tolist()
        select_only(new)
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        new.parent = body
        new.matrix_parent_inverse = body.matrix_world.inverted()
        for key in ("motion", "axis", "limit_min", "limit_max", "description"):
            new[key] = part.get(key, "")
        add_motion_limits(new, part)
        m = mesh_metrics(new)
        results.append({"name": new.name, "method": method, "faces": m["faces"], "watertight": m["watertight"],
                        "pivot": [round(float(v), 4) for v in pivot], "motion": part.get("motion")})
        log(f"segment: {new.name} via {method}, {m['faces']} faces, pivot {np.round(pivot, 3).tolist()}")
    save_blend(p["output_blend"])
    return {"parts": results, "skipped": skipped, "metrics": mesh_metrics(body)}


# =============================================================================
# Stage 5b - rigging
# =============================================================================

def _biped_bones(lo, hi):
    """Proportional humanoid skeleton fitted to the bbox (character faces -Y)."""
    H = float(hi[2] - lo[2])
    W = float(hi[0] - lo[0])
    z0, cy = float(lo[2]), float((lo[1] + hi[1]) / 2)
    z = lambda f: z0 + f * H  # noqa: E731
    bones = {
        "hips": ((0, cy, z(0.53)), (0, cy, z(0.60)), None),
        "spine": ((0, cy, z(0.60)), (0, cy, z(0.70)), "hips"),
        "chest": ((0, cy, z(0.70)), (0, cy, z(0.80)), "spine"),
        "neck": ((0, cy, z(0.82)), (0, cy, z(0.87)), "chest"),
        "head": ((0, cy, z(0.87)), (0, cy, z(1.0)), "neck"),
    }
    shoulder_x = 0.11 * H
    reach = max(W / 2.0, shoulder_x + 0.05 * H)
    arm_len = 0.40 * H
    dx = reach - shoulder_x
    dz = -math.sqrt(max(arm_len ** 2 - dx ** 2, 0.0))  # T-pose -> 0, A-pose -> negative
    norm = math.hypot(dx, dz) or 1.0
    ux, uz = dx / norm, dz / norm
    for side, s in (("L", 1), ("R", -1)):
        sh = (s * 0.02 * H, cy, z(0.80))
        j0 = (s * shoulder_x, cy, z(0.80))
        j1 = (s * (shoulder_x + ux * 0.45 * arm_len), cy, z(0.80) + uz * 0.45 * arm_len)
        j2 = (s * (shoulder_x + ux * 0.85 * arm_len), cy, z(0.80) + uz * 0.85 * arm_len)
        j3 = (s * (shoulder_x + ux * arm_len), cy, z(0.80) + uz * arm_len)
        bones[f"shoulder.{side}"] = (sh, j0, "chest")
        bones[f"upper_arm.{side}"] = (j0, j1, f"shoulder.{side}")
        bones[f"forearm.{side}"] = (j1, j2, f"upper_arm.{side}")
        bones[f"hand.{side}"] = (j2, j3, f"forearm.{side}")
        hx = s * 0.09 * H
        bones[f"thigh.{side}"] = ((hx, cy, z(0.52)), (hx, cy - 0.01 * H, z(0.28)), "hips")
        bones[f"shin.{side}"] = ((hx, cy - 0.01 * H, z(0.28)), (hx, cy, z(0.05)), f"thigh.{side}")
        bones[f"foot.{side}"] = ((hx, cy, z(0.05)), (hx, cy - 0.08 * H, z(0.0)), f"shin.{side}")
    return bones


def _quadruped_leg_columns(co, lo, hi, head_dir):
    """k-means the points under the belly into four leg columns -> {(front|back, L|R): (x, y)}."""
    H = float(hi[2] - lo[2])
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    under = co[co[:, 2] < lo[2] + 0.35 * H][:, :2]
    if len(under) < 80:
        return {}
    keys = [(leg, side) for leg in ("front", "back") for side in ("L", "R")]
    span = np.array([hi[0] - lo[0], hi[1] - lo[1]]) * 0.25
    centers = np.array([[cx + (1 if leg == "front" else -1) * head_dir * span[0],
                         cy + (1 if side == "L" else -1) * head_dir * span[1]] for leg, side in keys])
    for _ in range(15):
        assign = np.argmin(((under[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        for k in range(4):
            if (assign == k).sum() > 0:
                centers[k] = under[assign == k].mean(axis=0)
    counts = np.bincount(assign, minlength=4)
    if counts.min() < 0.05 * len(under):
        return {}
    return {key: (float(c[0]), float(c[1])) for key, c in zip(keys, centers)}


def _quadruped_bones(lo, hi, obj):
    """Spine along X, head at the end whose silhouette is taller."""
    L = float(hi[0] - lo[0])
    H = float(hi[2] - lo[2])
    Wd = float(hi[1] - lo[1])
    co = world_coords(obj)
    front_band = co[co[:, 0] > hi[0] - 0.15 * L]
    back_band = co[co[:, 0] < lo[0] + 0.15 * L]
    head_dir = 1.0 if (front_band[:, 2].max() if len(front_band) else 0) >= (
        back_band[:, 2].max() if len(back_band) else 0) else -1.0
    cx = float((lo[0] + hi[0]) / 2)
    z0 = float(lo[2])
    x = lambda f: cx + head_dir * f * L  # noqa: E731  (f in -0.5..0.5, positive = head side)
    body_z = z0 + 0.65 * H
    bones = {
        "hips": ((x(-0.30), 0, body_z), (x(-0.10), 0, body_z), None),
        "spine": ((x(-0.10), 0, body_z), (x(0.15), 0, body_z), "hips"),
        "chest": ((x(0.15), 0, body_z), (x(0.28), 0, body_z + 0.05 * H), "spine"),
        "neck": ((x(0.28), 0, body_z + 0.05 * H), (x(0.40), 0, z0 + 0.85 * H), "chest"),
        "head": ((x(0.40), 0, z0 + 0.85 * H), (x(0.50), 0, z0 + 0.80 * H), "neck"),
        "tail": ((x(-0.30), 0, body_z), (x(-0.50), 0, body_z - 0.1 * H), "hips"),
    }
    legs = _quadruped_leg_columns(co, lo, hi, head_dir)
    for side, s in (("L", 1), ("R", -1)):
        for leg, fx, parent in (("front", 0.22, "chest"), ("back", -0.25, "hips")):
            # .L is the animal's left: +Y when it faces +X
            lx, y = legs.get((leg, side), (x(fx), s * head_dir * Wd * 0.3))
            top = (lx, y, body_z - 0.05 * H)
            mid = (lx, y, z0 + 0.35 * H)
            bot = (lx, y, z0 + 0.05 * H)
            toe = (lx + head_dir * 0.04 * L, y, z0)
            bones[f"{leg}_upper.{side}"] = (top, mid, parent)
            bones[f"{leg}_lower.{side}"] = (mid, bot, f"{leg}_upper.{side}")
            bones[f"{leg}_foot.{side}"] = (bot, toe, f"{leg}_lower.{side}")
    return bones, head_dir


def build_armature(name, bones):
    arm_data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm)
    select_only(arm)
    set_mode("EDIT")
    created = {}
    for bone_name, (head, tail, parent) in bones.items():
        eb = arm_data.edit_bones.new(bone_name)
        eb.head = head
        eb.tail = tail
        created[bone_name] = eb
    for bone_name, (_, _, parent) in bones.items():
        if parent:
            created[bone_name].parent = created[parent]
            created[bone_name].use_connect = (created[parent].tail - created[bone_name].head).length < 1e-5
    set_mode("OBJECT")
    return arm


# ----------------------------------------------------------------- landmark detection

def _x_clusters(pts, gap):
    """Split a horizontal slice into runs along X separated by gaps wider than ``gap``."""
    if not len(pts):
        return []
    order = np.argsort(pts[:, 0])
    xs = pts[order, 0]
    cuts = np.flatnonzero(np.diff(xs) > gap) + 1
    return [pts[order[a:b]] for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(xs)])]


def _centroid_near(pts, center, axis_dir, half_width, radius, min_points=12):
    """Centroid of the limb cross-section around ``center`` (slab perpendicular to axis_dir)."""
    d = pts - center
    along = d @ axis_dir
    radial = np.linalg.norm(d - np.outer(along, axis_dir), axis=1)
    sel = (np.abs(along) < half_width) & (radial < radius)
    if sel.sum() < min_points:
        return center, False
    c = pts[sel].mean(axis=0)
    # keep the joint on the limb axis line through the slab
    return center + (c - center) - ((c - center) @ axis_dir) * axis_dir, True


def _connected_to(pts, seed, cell):
    """Points in the voxel-connected (26-neighbour) component nearest to ``seed``."""
    if not len(pts):
        return pts
    keys = np.floor(pts / cell).astype(np.int64)
    occupied = {}
    for i, k in enumerate(map(tuple, keys)):
        occupied.setdefault(k, []).append(i)
    start = tuple(keys[int(np.argmin(np.linalg.norm(pts - seed, axis=1)))])
    seen, stack = {start}, [start]
    while stack:
        x, y, z = stack.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    n = (x + dx, y + dy, z + dz)
                    if n in occupied and n not in seen:
                        seen.add(n)
                        stack.append(n)
    return pts[[i for k in seen for i in occupied[k]]]


def detect_biped_landmarks(obj):
    """Joint positions for an upright biped facing -Y, from horizontal cross-sections.

    Finds the crotch (legs split), armpits (arms separate in A-pose) or a T-pose
    (very wide slices), the neck (narrowest slice above the shoulders), then
    places elbows, wrists and knees at limb cross-section centroids so bones
    run through the middle of the limbs. Anything not found falls back to
    standard human proportions and is listed under ``fallback``.
    """
    rng = np.random.default_rng(0)
    pts = np.vstack([world_coords(obj), sample_surface(obj, 60000, rng)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    H = float(hi[2] - lo[2])
    z0 = float(lo[2])
    cx = 0.0
    detected, fallback = [], []
    nb = 120
    edges = np.linspace(z0, z0 + H, nb + 1)
    bins = [pts[(pts[:, 2] >= edges[i]) & (pts[:, 2] < edges[i + 1])] for i in range(nb)]
    zc = (edges[:-1] + edges[1:]) / 2
    frac = (zc - z0) / H

    def split_at_center(sl):
        return len(sl) > 20 and not np.any(np.abs(sl[:, 0] - cx) < 0.012 * H) and \
            np.any(sl[:, 0] < cx) and np.any(sl[:, 0] > cx)

    # Crotch: top of the run of slices where the legs are separated
    crotch = None
    run = False
    for i in range(nb):
        if frac[i] < 0.08:
            continue
        if frac[i] > 0.68:
            break
        if split_at_center(bins[i]):
            run, crotch = True, zc[i]
        elif run:
            break
    if crotch is None:
        crotch = z0 + 0.47 * H
        fallback.append("crotch")
    else:
        detected.append("crotch")

    def torso_cluster(sl):
        cl = _x_clusters(sl, 0.02 * H)
        return min(cl, key=lambda c: abs(np.median(c[:, 0]) - cx)) if cl else None

    hip_sl = [torso_cluster(bins[i]) for i in range(nb) if crotch + 0.04 * H < zc[i] < crotch + 0.14 * H]
    hip_half = np.median([np.ptp(c[:, 0]) / 2 for c in hip_sl if c is not None]) if hip_sl else 0.1 * H

    # Arms: A-pose -> separate clusters beside the torso; T-pose -> very wide slices
    armpit, pose, t_heights = None, "fallback", []
    for i in range(nb):
        if not (crotch + 0.05 * H < zc[i] < z0 + 0.9 * H):
            continue
        cl = _x_clusters(bins[i], 0.02 * H)
        if len(cl) >= 3:
            meds = [np.median(c[:, 0]) for c in cl]
            if min(meds) < cx - hip_half and max(meds) > cx + hip_half:
                armpit, pose = zc[i], "A"
        if len(bins[i]) and np.ptp(bins[i][:, 0]) > 4.0 * hip_half:
            t_heights.append(zc[i])
    # T-pose arms make the body widest at shoulder height; A-pose arms are widest at the hands
    # (around half height). Decide on that, not on cluster splits, which depend on mesh density.
    high = [z for z in t_heights if z > z0 + 0.68 * H]
    if high and np.median(t_heights) > z0 + 0.68 * H:
        pose, armpit = "T", float(np.min(high))
    if armpit is None:
        armpit = z0 + 0.75 * H
        fallback.append("armpit")
    else:
        detected.append(f"armpit({pose}-pose)")

    # Neck: narrowest torso slice between the shoulders and the crown
    neck, best_w = None, None
    for i in range(nb):
        if armpit + 0.04 * H < zc[i] < z0 + 0.9 * H:
            c = torso_cluster(bins[i])
            if c is not None and len(c) > 10:
                w = np.ptp(c[:, 0])
                if best_w is None or w < best_w:
                    neck, best_w = zc[i], w
    if neck is None:
        neck = z0 + 0.85 * H
        fallback.append("neck")
    else:
        detected.append("neck")

    def slice_centroid(z, half=0.015 * H, side=0):
        sel = np.abs(pts[:, 2] - z) < half
        if side:
            sel &= side * (pts[:, 0] - cx) > 0
        sub = pts[sel]
        if side == 0 and len(sub):
            c = torso_cluster(sub)
            sub = c if c is not None else sub
        return sub.mean(axis=0) if len(sub) > 5 else np.array([cx, (lo[1] + hi[1]) / 2, z])

    L = {"pose": pose}
    hips_c = slice_centroid(crotch + 0.07 * H)
    chest_c = slice_centroid(armpit - 0.02 * H)
    neck_c = slice_centroid(neck)
    # Anatomy: the hip joints sit ~5% of height ABOVE the crotch, the pelvis root a little higher
    L["hips"] = np.array([cx, hips_c[1], crotch + 0.07 * H])
    L["spine1"] = np.array([cx, *slice_centroid(crotch + 0.07 * H + (armpit - crotch) * 0.3)[1:2],
                            crotch + 0.07 * H + (armpit - crotch) * 0.3])
    L["spine2"] = np.array([cx, chest_c[1], armpit - 0.02 * H])
    L["neck_base"] = np.array([cx, neck_c[1], neck - 0.03 * H])
    L["head_base"] = np.array([cx, neck_c[1], neck + 0.02 * H])
    L["head_top"] = np.array([cx, neck_c[1], float(hi[2])])

    chest_sl = torso_cluster(pts[np.abs(pts[:, 2] - (armpit - 0.03 * H)) < 0.015 * H])
    torso_half = float(np.ptp(chest_sl[:, 0]) / 2) if chest_sl is not None and len(chest_sl) > 5 else 0.12 * H
    if pose == "T":
        torso_half = min(torso_half, 1.6 * hip_half)
    for side, s in (("L", 1), ("R", -1)):
        shoulder_z = armpit + (0.0 if pose == "T" else 0.04 * H)
        sj = np.array([cx + s * 0.85 * torso_half, chest_c[1], shoulder_z])
        L[f"clavicle.{side}"] = np.array([cx + s * 0.02 * H, chest_c[1], shoulder_z])
        L[f"shoulder.{side}"] = sj
        outboard = pts[(s * (pts[:, 0] - cx) > torso_half * 1.05) & (pts[:, 2] < neck)]
        # Only the region connected to the shoulder is arm (outer thighs are outboard too)
        arm = _connected_to(outboard, sj, 0.02 * H)
        if len(arm) > 50 and pose != "fallback":
            tip = arm[np.argmax(np.linalg.norm(arm - sj, axis=1))]
            axis = tip - sj
            length = float(np.linalg.norm(axis))
            axis /= max(length, 1e-9)
            elbow, ok_e = _centroid_near(arm, sj + 0.42 * length * axis, axis, 0.03 * H, 0.12 * H)
            wrist, ok_w = _centroid_near(arm, sj + 0.75 * length * axis, axis, 0.03 * H, 0.10 * H)
            L[f"elbow.{side}"], L[f"wrist.{side}"], L[f"hand_tip.{side}"] = elbow, wrist, tip
            detected.append(f"arm.{side}" + ("" if ok_e and ok_w else "(axis only)"))
        else:
            # Arm touching the body (no separate region): the outermost point at arm height
            # on this side is the hand; joints go on the shoulder->hand line through the limb.
            band = pts[(s * (pts[:, 0] - cx) > 0) & (pts[:, 2] > crotch - 0.12 * H) & (pts[:, 2] < armpit)]
            tip = band[np.argmax(s * (band[:, 0] - cx))] if len(band) else None
            if tip is not None and s * (tip[0] - cx) > torso_half * 1.1 and np.linalg.norm(tip - sj) > 0.25 * H:
                axis = tip - sj
                length = float(np.linalg.norm(axis))
                axis /= length
                near = band[s * (band[:, 0] - cx) > torso_half * 0.9]
                elbow, _ = _centroid_near(near, sj + 0.42 * length * axis, axis, 0.03 * H, 0.10 * H)
                wrist, _ = _centroid_near(near, sj + 0.75 * length * axis, axis, 0.03 * H, 0.08 * H)
                L[f"elbow.{side}"], L[f"wrist.{side}"], L[f"hand_tip.{side}"] = elbow, wrist, tip
                detected.append(f"arm.{side}(touching body)")
            else:
                reach = 0.40 * H
                down = np.array([s * 0.5, 0.0, -0.866])  # 60 degrees below horizontal
                L[f"elbow.{side}"] = sj + down * reach * 0.42
                L[f"wrist.{side}"] = sj + down * reach * 0.75
                L[f"hand_tip.{side}"] = sj + down * reach
                fallback.append(f"arm.{side}")
        leg = pts[(s * (pts[:, 0] - cx) > 0) & (pts[:, 2] < crotch)]
        if len(leg) > 50:
            top = leg[leg[:, 2] > crotch - 0.08 * H]
            hip_x = float(np.median(top[:, 0])) if len(top) > 5 else cx + s * 0.09 * H
            hip = np.array([hip_x, hips_c[1], crotch + 0.05 * H])
            low = leg[(leg[:, 2] > z0 + 0.06 * H) & (leg[:, 2] < z0 + 0.10 * H)]
            ank_xy = low.mean(axis=0)[:2] if len(low) > 5 else np.array([hip_x, hips_c[1]])
            ankle = np.array([ank_xy[0], ank_xy[1], z0 + 0.05 * H])
            knee_z = (hip[2] + ankle[2]) / 2
            ks = leg[np.abs(leg[:, 2] - knee_z) < 0.02 * H]
            kxy = ks.mean(axis=0)[:2] if len(ks) > 5 else (hip[:2] + ankle[:2]) / 2
            knee = np.array([kxy[0], kxy[1] - 0.01 * H, knee_z])  # slight forward bend for IK
            foot = leg[leg[:, 2] < z0 + 0.04 * H]
            toe_y = float(foot[:, 1].min()) if len(foot) > 5 else ankle[1] - 0.08 * H
            L[f"hip.{side}"], L[f"knee.{side}"], L[f"ankle.{side}"] = hip, knee, ankle
            L[f"ball.{side}"] = np.array([ankle[0], ankle[1] + 0.7 * (toe_y - ankle[1]), z0 + 0.02 * H])
            L[f"toe.{side}"] = np.array([ankle[0], toe_y, z0 + 0.02 * H])
            detected.append(f"leg.{side}")
        else:
            hx = cx + s * 0.09 * H
            L[f"hip.{side}"] = np.array([hx, hips_c[1], crotch + 0.05 * H])
            L[f"knee.{side}"] = np.array([hx, hips_c[1] - 0.01 * H, z0 + 0.28 * H])
            L[f"ankle.{side}"] = np.array([hx, hips_c[1], z0 + 0.05 * H])
            L[f"ball.{side}"] = np.array([hx, hips_c[1] - 0.06 * H, z0 + 0.02 * H])
            L[f"toe.{side}"] = np.array([hx, hips_c[1] - 0.09 * H, z0 + 0.02 * H])
            fallback.append(f"leg.{side}")
    L["detected"], L["fallback"], L["height"] = detected, fallback, H
    return L


def _biped_bones_from_landmarks(L):
    b = {
        "hips": (L["hips"], L["spine1"], None),
        "spine": (L["spine1"], L["spine2"], "hips"),
        "chest": (L["spine2"], L["neck_base"], "spine"),
        "neck": (L["neck_base"], L["head_base"], "chest"),
        "head": (L["head_base"], L["head_top"], "neck"),
    }
    for side in ("L", "R"):
        b[f"shoulder.{side}"] = (L[f"clavicle.{side}"], L[f"shoulder.{side}"], "chest")
        b[f"upper_arm.{side}"] = (L[f"shoulder.{side}"], L[f"elbow.{side}"], f"shoulder.{side}")
        b[f"forearm.{side}"] = (L[f"elbow.{side}"], L[f"wrist.{side}"], f"upper_arm.{side}")
        b[f"hand.{side}"] = (L[f"wrist.{side}"], L[f"hand_tip.{side}"], f"forearm.{side}")
        b[f"thigh.{side}"] = (L[f"hip.{side}"], L[f"knee.{side}"], "hips")
        b[f"shin.{side}"] = (L[f"knee.{side}"], L[f"ankle.{side}"], f"thigh.{side}")
        b[f"foot.{side}"] = (L[f"ankle.{side}"], L[f"ball.{side}"], f"shin.{side}")
        b[f"toe.{side}"] = (L[f"ball.{side}"], L[f"toe.{side}"], f"foot.{side}")
    return {k: (tuple(map(float, h)), tuple(map(float, t)), par) for k, (h, t, par) in b.items()}


def fit_metarig(metarig, L):
    """Move Rigify basic-human metarig joints onto the detected landmarks."""
    H = L["height"]
    lerp = lambda a, b, t: a + (b - a) * t  # noqa: E731
    target = {
        "spine": (L["hips"], lerp(L["hips"], L["neck_base"], 0.25)),
        "spine.001": (lerp(L["hips"], L["neck_base"], 0.25), lerp(L["hips"], L["neck_base"], 0.45)),
        "spine.002": (lerp(L["hips"], L["neck_base"], 0.45), lerp(L["hips"], L["neck_base"], 0.7)),
        "spine.003": (lerp(L["hips"], L["neck_base"], 0.7), L["neck_base"]),
        "spine.004": (L["neck_base"], lerp(L["neck_base"], L["head_base"], 0.5)),
        "spine.005": (lerp(L["neck_base"], L["head_base"], 0.5), L["head_base"]),
        "spine.006": (L["head_base"], L["head_top"]),
    }
    for side, s in (("L", 1), ("R", -1)):
        sj = L[f"shoulder.{side}"]
        target.update({
            f"shoulder.{side}": (L[f"clavicle.{side}"] + np.array([0, -0.03 * H, 0]), lerp(L[f"clavicle.{side}"], sj, 0.9)),
            f"upper_arm.{side}": (sj, L[f"elbow.{side}"]),
            f"forearm.{side}": (L[f"elbow.{side}"], L[f"wrist.{side}"]),
            f"hand.{side}": (L[f"wrist.{side}"], L[f"hand_tip.{side}"]),
            f"breast.{side}": (np.array([s * 0.065 * H, L["spine2"][1], L["spine2"][2] - 0.05 * H]),
                               np.array([s * 0.065 * H, L["spine2"][1] - 0.07 * H, L["spine2"][2] - 0.05 * H])),
            f"pelvis.{side}": (L["hips"], L[f"hip.{side}"] + np.array([0, -0.05 * H, 0.08 * H])),
            f"thigh.{side}": (L[f"hip.{side}"], L[f"knee.{side}"]),
            f"shin.{side}": (L[f"knee.{side}"], L[f"ankle.{side}"]),
            f"foot.{side}": (L[f"ankle.{side}"], L[f"ball.{side}"]),
            f"toe.{side}": (L[f"ball.{side}"], L[f"toe.{side}"]),
            f"heel.02.{side}": (np.array([L[f"ankle.{side}"][0] - s * 0.02 * H, L[f"ankle.{side}"][1] + 0.025 * H, 0.0]),
                                np.array([L[f"ankle.{side}"][0] + s * 0.02 * H, L[f"ankle.{side}"][1] + 0.025 * H, 0.0])),
        })
    select_only(metarig)
    set_mode("EDIT")
    eb = metarig.data.edit_bones
    moved = 0
    for name, (head, tail) in target.items():
        bone = eb.get(name)
        if bone is None:
            continue
        connected = bone.use_connect
        bone.use_connect = False
        bone.head, bone.tail = mathutils.Vector(head), mathutils.Vector(tail)
        bone.use_connect = connected and bone.parent is not None and \
            (bone.parent.tail - bone.head).length < 1e-4
        moved += 1
    set_mode("OBJECT")
    return moved


def rigify_armature(rig_type, lo, hi, landmarks=None):
    import addon_utils

    addon_utils.enable("rigify", default_set=True)
    before = set(bpy.data.objects)
    if rig_type == "quadruped":
        op = getattr(bpy.ops.object, "armature_wolf_metarig_add", None) or bpy.ops.object.armature_basic_quadruped_metarig_add
    else:
        # Basic human: body chain without face/finger bones an AI mesh cannot support
        op = getattr(bpy.ops.object, "armature_basic_human_metarig_add", None) or bpy.ops.object.armature_human_metarig_add
    op()
    metarig = [o for o in bpy.data.objects if o not in before][0]
    if landmarks is not None and rig_type == "biped":
        fit_metarig(metarig, landmarks)
    else:
        m_lo, m_hi = np.array([1e9] * 3), np.array([-1e9] * 3)
        for b in metarig.data.bones:
            for pt in (b.head_local, b.tail_local):
                m_lo, m_hi = np.minimum(m_lo, pt), np.maximum(m_hi, pt)
        axis = 2 if rig_type == "biped" else 0
        factor = float((hi[axis] - lo[axis]) / max(m_hi[axis] - m_lo[axis], 1e-6))
        metarig.scale = (factor,) * 3
        metarig.location = ((lo[0] + hi[0]) / 2 - (m_lo[0] + m_hi[0]) / 2 * factor,
                            (lo[1] + hi[1]) / 2 - (m_lo[1] + m_hi[1]) / 2 * factor,
                            lo[2] - m_lo[2] * factor)
        apply_transforms(metarig)
    select_only(metarig)
    bpy.ops.pose.rigify_generate()
    rig = bpy.context.view_layer.objects.active
    if rig is metarig or rig.type != "ARMATURE":
        raise RuntimeError("rigify_generate did not produce a rig")
    metarig.hide_render = True
    metarig.hide_set(True)
    metarig["asset_role"] = "metarig"
    return rig


def smooth_weights(obj, repeat: int = 1, factor: float = 0.5):
    """Laplacian-smooth every vertex group over mesh edges (the bpy operator needs a UI)."""
    groups = list(obj.vertex_groups)
    if not groups or repeat <= 0:
        return
    nv = len(obj.data.vertices)
    w = np.zeros((nv, len(groups)))
    for v in obj.data.vertices:
        for g in v.groups:
            w[v.index, g.group] = g.weight
    e = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    deg = np.bincount(e.ravel(), minlength=nv).astype(float)[:, None]
    for _ in range(repeat):
        acc = np.zeros_like(w)
        np.add.at(acc, e[:, 0], w[e[:, 1]])
        np.add.at(acc, e[:, 1], w[e[:, 0]])
        w = (1 - factor) * w + factor * np.where(deg > 0, acc / np.maximum(deg, 1), w)
    for j, g in enumerate(groups):
        nz = np.flatnonzero(w[:, j] > 1e-4)
        zero = np.flatnonzero(w[:, j] <= 1e-4)
        if len(zero):
            g.remove(zero.tolist())
        for vi in nz:
            g.add([int(vi)], float(w[vi, j]), "REPLACE")


def bind_unweighted(obj, arm) -> int:
    """Give every vertex without skin weights full weight on its nearest deform bone.

    Bone heat leaves a few vertices empty (inside folds, tiny islands); engines then
    leave them frozen in place while the rest of the mesh moves.
    """
    empty = [v.index for v in obj.data.vertices if not any(g.weight > 1e-4 for g in v.groups)]
    if not empty:
        return 0
    deform = [b for b in arm.data.bones if b.use_deform]
    co = world_coords(obj)[empty]
    mw = np.array(arm.matrix_world)
    dist = np.empty((len(co), len(deform)))
    for j, b in enumerate(deform):
        h = mw[:3, :3] @ np.array(b.head_local) + mw[:3, 3]
        t = mw[:3, :3] @ np.array(b.tail_local) + mw[:3, 3]
        seg = t - h
        tt = np.clip(((co - h) @ seg) / max(seg @ seg, 1e-12), 0, 1)
        dist[:, j] = np.linalg.norm(co - (h + tt[:, None] * seg), axis=1)
    nearest = dist.argmin(axis=1)
    for vi, j in zip(empty, nearest):
        name = deform[j].name
        group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        group.add([vi], 1.0, "REPLACE")
    return len(empty)


def clean_weights(obj, max_influences: int = 4, smooth_repeat: int = 1, arm=None) -> dict:
    """Game-engine skinning rules: every vertex weighted, <= N influences, smoothed, normalised."""
    select_only(obj)
    done = {}
    if arm is not None:
        done["bound_unweighted"] = bind_unweighted(obj, arm)
    if smooth_repeat > 0:
        smooth_weights(obj, smooth_repeat)
        done["smooth"] = smooth_repeat
    for name, op, kwargs in (
        ("limit", bpy.ops.object.vertex_group_limit_total, {"group_select_mode": "ALL", "limit": max_influences}),
        ("normalize", bpy.ops.object.vertex_group_normalize_all, {"group_select_mode": "ALL", "lock_active": False}),
    ):
        try:
            done[name] = "FINISHED" in call_op(op, **kwargs)
        except RuntimeError as exc:
            done[name] = f"failed: {exc}"
    counts = [sum(1 for g in v.groups if g.weight > 1e-4) for v in obj.data.vertices]
    done["max_influences"] = max(counts) if counts else 0
    done["unweighted_vertices"] = sum(1 for c in counts if c == 0)
    return done


def _find_bone(arm, base):
    for prefix in ("DEF-", ""):
        b = arm.pose.bones.get(prefix + base)
        if b is not None:
            return b
    return None


def pose_test(obj, arm, rig_type) -> dict:
    """Bend the main joints and measure how badly the skin stretches or collapses.

    Rotations are applied in world space about each joint, so the test works for
    any skeleton naming/roll; Rigify's constraints are muted so DEF bones can be
    posed directly.
    """
    for pb in arm.pose.bones:
        for c in pb.constraints:
            c.mute = True
    bpy.context.view_layer.update()
    rest = world_coords(obj)
    eidx = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", eidx)
    eidx = eidx.reshape(-1, 2)
    rest_len = np.linalg.norm(rest[eidx[:, 0]] - rest[eidx[:, 1]], axis=1)

    if rig_type == "quadruped":
        moves = [(f"{leg}_upper.{s}", (0, 1, 0), 25) for leg in ("front", "back") for s in "LR"] + \
                [(f"{leg}_lower.{s}", (0, 1, 0), -35) for leg in ("front", "back") for s in "LR"] + \
                [("neck", (0, 1, 0), -20)]
    else:
        # Raise the arms (rotating down would drive A-pose arms through the torso)
        moves = [("upper_arm.L", (0, 1, 0), -35), ("upper_arm.R", (0, 1, 0), 35),
                 ("forearm.L", None, 60), ("forearm.R", None, 60),
                 ("thigh.L", (1, 0, 0), -30), ("thigh.R", (1, 0, 0), -30),
                 ("shin.L", (1, 0, 0), 50), ("shin.R", (1, 0, 0), 50),
                 ("chest", (0, 0, 1), 20), ("spine.003", (0, 0, 1), 20)]
    posed = []
    for base, axis, deg in moves:
        pb = _find_bone(arm, base)
        if pb is None:
            continue
        head = np.array(pb.head)
        if axis is None:  # bend forward around the axis perpendicular to the limb and -Y
            limb = np.array(pb.tail) - head
            axis = np.cross(limb / max(np.linalg.norm(limb), 1e-9), (0.0, -1.0, 0.0))
            if np.linalg.norm(axis) < 1e-6:
                axis = (1.0, 0.0, 0.0)
        rot = mathutils.Matrix.Rotation(math.radians(deg), 4, mathutils.Vector(axis).normalized())
        pivot = mathutils.Matrix.Translation(mathutils.Vector(head))
        pb.matrix = pivot @ rot @ pivot.inverted() @ pb.matrix
        bpy.context.view_layer.update()
        posed.append(pb.name)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ np.array(obj.matrix_world)[:3, :3].T + np.array(obj.matrix_world)[:3, 3]
    ev.to_mesh_clear()
    if len(co) != len(rest):
        return {"posed_bones": posed, "error": "evaluated mesh topology differs"}
    ratio = np.linalg.norm(co[eidx[:, 0]] - co[eidx[:, 1]], axis=1) / np.maximum(rest_len, 1e-9)
    bad = (ratio > 1.6) | (ratio < 0.55)
    moved = np.linalg.norm(co - rest, axis=1)
    return {"posed_bones": posed, "bad_edge_fraction": round(float(bad.mean()), 4),
            "max_stretch": round(float(ratio.max()), 3), "min_stretch": round(float(ratio.min()), 3),
            "moved_vertex_fraction": round(float((moved > 1e-4 * float(np.ptp(rest[:, 2]))).mean()), 3)}


def weight_coverage(obj) -> float:
    if not obj.vertex_groups:
        return 0.0
    weighted = sum(1 for v in obj.data.vertices if any(g.weight > 1e-4 for g in v.groups))
    return weighted / max(1, len(obj.data.vertices))


def proximity_weights(obj, arm, falloff: float = 4.0, max_influences: int = 4):
    """Fallback skinning: inverse-distance to deform bone segments, top-N normalised."""
    deform = [b for b in arm.data.bones if b.use_deform]
    co = world_coords(obj)
    mw = np.array(arm.matrix_world)
    dists = np.empty((len(co), len(deform)))
    for j, b in enumerate(deform):
        h = mw[:3, :3] @ np.array(b.head_local) + mw[:3, 3]
        t = mw[:3, :3] @ np.array(b.tail_local) + mw[:3, 3]
        seg = t - h
        tt = np.clip(((co - h) @ seg) / max(seg @ seg, 1e-12), 0, 1)
        dists[:, j] = np.linalg.norm(co - (h + tt[:, None] * seg), axis=1)
    w = 1.0 / np.maximum(dists, 1e-6) ** falloff
    keep = np.argsort(-w, axis=1)[:, :max_influences]
    obj.vertex_groups.clear()
    groups = [obj.vertex_groups.new(name=b.name) for b in deform]
    for vi in range(len(co)):
        sel = keep[vi]
        ws = w[vi, sel] / w[vi, sel].sum()
        for j, wv in zip(sel, ws):
            groups[j].add([vi], float(wv), "REPLACE")
    if not any(m.type == "ARMATURE" for m in obj.modifiers):
        mod = obj.modifiers.new("Armature", "ARMATURE")
        mod.object = arm


def op_rig(p: dict) -> dict:
    open_blend(p["input_blend"])
    obj = find_role("target")
    lo, hi = bbox([obj])
    rig_type = p.get("rig_type", "biped")
    report = {"rig_type": rig_type}
    landmarks = None
    if rig_type == "biped":
        landmarks = detect_biped_landmarks(obj)
        report["landmarks"] = {"pose": landmarks["pose"], "detected": landmarks["detected"],
                               "fallback": landmarks["fallback"]}
        log(f"rig: {landmarks['pose']}-pose, detected {landmarks['detected']}, fallback {landmarks['fallback']}")
    arm = None
    if p.get("method", "rigify") == "rigify":
        try:
            arm = rigify_armature(rig_type, lo, hi, landmarks)
            report["method"] = "rigify"
        except Exception as exc:  # rigify availability/metarig names vary per version
            log(f"rigify failed ({exc}); falling back to generated skeleton")
            for o in list(bpy.data.objects):
                if o.type == "ARMATURE":
                    bpy.data.objects.remove(o, do_unlink=True)
    if arm is None:
        if rig_type == "quadruped":
            bones, head_dir = _quadruped_bones(lo, hi, obj)
            report["head_direction"] = "+X" if head_dir > 0 else "-X"
        else:
            bones = _biped_bones_from_landmarks(landmarks)
        arm = build_armature(f"{obj.name}_rig", bones)
        report["method"] = "generated"
    arm["asset_role"] = "armature"

    weights = p.get("weights", "auto")
    if weights != "proximity":
        select_only(obj, arm, active=arm)
        try:
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        except RuntimeError as exc:
            log(f"automatic weights failed: {exc}")
    coverage = weight_coverage(obj)
    report["auto_weight_coverage"] = round(coverage, 3)
    if weights == "proximity" or coverage < 0.95:
        # Bone heat fails on non-manifold or self-intersecting meshes
        log(f"bone-heat covered {coverage:.0%} of vertices; using proximity weights")
        if obj.parent is None:
            obj.parent = arm
            obj.matrix_parent_inverse = arm.matrix_world.inverted()
        proximity_weights(obj, arm)
        report["weights"] = "proximity"
    else:
        report["weights"] = "bone_heat"
    report["weight_cleanup"] = clean_weights(obj, int(p.get("max_influences", 4)), int(p.get("weight_smooth", 1)),
                                             arm=arm)
    report["unweighted_vertices"] = report["weight_cleanup"]["unweighted_vertices"]
    report["deform_bones"] = sum(1 for b in arm.data.bones if b.use_deform)
    report["weight_coverage"] = round(weight_coverage(obj), 3)
    save_blend(p["output_blend"])  # rest pose: this is what gets exported

    if p.get("pose_test", True):
        report["pose_test"] = pose_test(obj, arm, rig_type)
        posed = str(Path(p["output_blend"]).with_name(Path(p["output_blend"]).stem + "_posed.blend"))
        save_blend(posed)
        report["posed_blend"] = posed
        log(f"rig: pose test {report['pose_test']}")
    return report


# =============================================================================
# Stage 6 - audit renders
# =============================================================================

def _set_engine(scene, engine: str, samples: int, use_gpu: bool) -> str:
    if engine in ("auto", "cycles"):
        enable_gpu(scene, use_gpu)
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        return "CYCLES"
    if engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        return "BLENDER_WORKBENCH"
    # EEVEE's identifier changed in 4.2 (BLENDER_EEVEE_NEXT) and back again later
    for ident in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = ident
            scene.eevee.taa_render_samples = samples
            return ident
        except TypeError:
            continue
    enable_gpu(scene, use_gpu)
    return "CYCLES"


def _look_at(cam, target):
    direction = mathutils.Vector(target) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _flat_material(name, rgb, emission=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if emission:
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*rgb, 1.0)
        nt.links.new(em.outputs[0], out.inputs["Surface"])
    else:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def _checker_material():
    mat = bpy.data.materials.new("uv_checker")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    chk = nt.nodes.new("ShaderNodeTexChecker")
    chk.inputs["Scale"].default_value = 24.0
    chk.inputs["Color1"].default_value = (0.9, 0.9, 0.9, 1)
    chk.inputs["Color2"].default_value = (0.05, 0.05, 0.05, 1)
    nt.links.new(tc.outputs["UV"], chk.inputs["Vector"])
    nt.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _override_materials(objs, mat):
    """Object-level material override; returns a restore callback."""
    saved = []
    for o in objs:
        if not o.material_slots:
            o.data.materials.append(None)
        for slot in o.material_slots:
            saved.append((slot, slot.link, slot.material))
            slot.link = "OBJECT"
            slot.material = mat

    def restore():
        for slot, link, material in reversed(saved):
            slot.material = None
            slot.link = link
            if link == "OBJECT":
                slot.material = material

    return restore


def op_render(p: dict) -> dict:
    open_blend(p["input_blend"])
    scene = bpy.context.scene
    for o in list(scene.objects):
        if o.type in ("CAMERA", "LIGHT") or o.get("asset_role") in ("bake_source", "metarig"):
            bpy.data.objects.remove(o, do_unlink=True)
    targets = asset_meshes()
    for o in scene.objects:
        if o.type == "ARMATURE":
            o.hide_render = True
    lo, hi = bbox(targets)
    center = (lo + hi) / 2.0
    dims = hi - lo
    radius = float(np.linalg.norm(dims)) / 2.0

    res = int(p.get("resolution", 1024))
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    engine = _set_engine(scene, p.get("engine", "auto"), int(p.get("samples", 32)), p.get("use_gpu", True))
    scene.render.film_transparent = False
    if hasattr(scene, "view_settings"):
        scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("audit_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.55, 0.57, 0.6, 1.0)
    bg.inputs["Strength"].default_value = 0.8
    scene.world = world

    for name, rot, energy in (("key", (math.radians(50), 0, math.radians(30)), 3.5),
                              ("fill", (math.radians(60), 0, math.radians(-120)), 1.2)):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.rotation_euler = rot
        scene.collection.objects.link(light)

    cam = bpy.data.objects.new("audit_cam", bpy.data.cameras.new("audit_cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.clip_end = radius * 50 + 100
    dist = radius * 4 + 1.0

    out_dir = Path(p["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    renders, framing = {}, {}
    wanted = set(p.get("views") or ("front", "side", "wireframe", "shaded", "uv_checker"))

    def shoot(view):
        path = out_dir / f"{view}.png"
        scene.render.filepath = str(path.resolve())
        bpy.ops.render.render(write_still=True)
        renders[view] = str(path.resolve())
        log(f"render: {view} -> {path}")

    def ortho(view, location, rotation, extent_h, extent_v, plane, u_axis):
        if view not in wanted:
            return
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = max(extent_h, extent_v) * 1.15
        cam.location = location
        cam.rotation_euler = rotation
        framing[view] = {"plane": plane, "u_axis": u_axis, "center": center.round(5).tolist(),
                         "ortho_scale": cam.data.ortho_scale, "resolution": res}
        shoot(view)

    def perspective():
        cam.data.type = "PERSP"
        cam.data.lens = 50
        direction = mathutils.Vector((1.0, -1.3, 0.8)).normalized()
        fov = cam.data.angle
        cam.location = mathutils.Vector(center.tolist()) + direction * (radius / math.sin(fov / 2) * 1.1)
        _look_at(cam, center.tolist())

    # 1. Front (looking +Y, sees X-Z plane) / 2. Side (looking -X, sees Y-Z plane)
    ortho("front", (center[0], center[1] - dist, center[2]), (math.radians(90), 0, 0),
          dims[0], dims[2], "XZ", "+X")
    ortho("side", (center[0] + dist, center[1], center[2]), (math.radians(90), 0, math.radians(90)),
          dims[1], dims[2], "YZ", "+Y")

    # 3. Wireframe overlay on clay
    if "wireframe" in wanted:
        _render_wireframe(scene, targets, dims, perspective, shoot)

    # 4. Shaded (final materials)
    if "shaded" in wanted:
        perspective()
        shoot("shaded")

    # 5. Optional UV checker - makes stretching visible to the vision model
    if ("uv_checker" in wanted and p.get("include_uv_checker", True)
            and all(o.data.uv_layers for o in targets)):
        perspective()
        restore = _override_materials(targets, _checker_material())
        shoot("uv_checker")
        restore()

    metrics = {o.name: mesh_metrics(o) for o in targets}
    return {"renders": renders, "framing": framing, "engine": engine, "metrics": metrics,
            "bbox_min": lo.round(5).tolist(), "bbox_max": hi.round(5).tolist()}


def _render_wireframe(scene, targets, dims, perspective, shoot):
    clay = _flat_material("audit_clay", (0.7, 0.7, 0.7))
    wire_mat = _flat_material("audit_wire", (0.02, 0.02, 0.02), emission=True)
    restore = _override_materials(targets, clay)
    wires = []
    for o in targets:
        w = o.copy()
        scene.collection.objects.link(w)
        mod = w.modifiers.new("wire", "WIREFRAME")
        mod.thickness = float(dims.max()) * 0.002
        mod.use_replace = True
        mod.use_even_offset = False
        wires.append(w)
    _override_materials(wires, wire_mat)
    perspective()
    shoot("wireframe")
    for w in wires:
        bpy.data.objects.remove(w, do_unlink=True)
    restore()


# =============================================================================
# Export
# =============================================================================

def op_export(p: dict) -> dict:
    open_blend(p["input_blend"])
    scene = bpy.context.scene
    for o in list(scene.objects):
        if o.type in ("CAMERA", "LIGHT") or o.get("asset_role") in ("bake_source", "metarig"):
            bpy.data.objects.remove(o, do_unlink=True)
    # Rigify bone-shape widgets live in an excluded collection; they are not asset geometry
    # (removing the collection itself leaves the view layer with a stale cache -> crash)
    for o in list(bpy.data.objects):
        if o.name.startswith("WGT-"):
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.context.view_layer.update()
    out = Path(p["output_path"]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = p["format"].lower()
    for o in bpy.context.view_layer.objects:
        o.hide_set(False)
        o.select_set(True)
    if fmt == "glb":
        call_op(bpy.ops.export_scene.gltf, filepath=str(out), export_format="GLB", export_extras=True,
                export_apply=True, export_yup=True, export_def_bones=True, export_animations=False,
                export_image_format="AUTO")
    elif fmt == "fbx":
        call_op(bpy.ops.export_scene.fbx, filepath=str(out), use_selection=False,
                apply_scale_options="FBX_SCALE_UNITS", path_mode="COPY", embed_textures=True,
                use_custom_props=True, add_leaf_bones=False, use_armature_deform_only=True,
                mesh_smooth_type="FACE", use_mesh_modifiers=True, bake_anim=False,
                object_types={"MESH", "ARMATURE", "EMPTY"})
    elif fmt == "obj":
        if hasattr(bpy.ops.wm, "obj_export"):
            call_op(bpy.ops.wm.obj_export, filepath=str(out), export_materials=True, path_mode="COPY",
                    export_uv=True, export_normals=True, apply_modifiers=True, export_selected_objects=False)
        else:
            bpy.ops.export_scene.obj(filepath=str(out), use_materials=True, path_mode="COPY")
    else:
        raise ValueError(f"unsupported export format {fmt}")
    save_blend(p["output_blend"])
    files = sorted(str(f) for f in out.parent.iterdir() if f.is_file())
    return {"output": str(out), "bytes": out.stat().st_size, "files": files}


OPERATIONS = {
    "cleanup": op_cleanup,
    "primitives": op_primitives,
    "texture": op_texture,
    "segment": op_segment,
    "rig": op_rig,
    "render": op_render,
    "export": op_export,
}


def blender_main(argv) -> int:
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    job_path = argv[argv.index("--job") + 1]
    result_path = argv[argv.index("--result") + 1]
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    params = dict(job["params"])
    params.setdefault("use_gpu", job.get("use_gpu", True))
    log(f"Blender {bpy.app.version_string}: {job['operation']}")
    try:
        result = OPERATIONS[job["operation"]](params)
        result["status"] = "ok"
        result["blender_version"] = bpy.app.version_string
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        log(result["traceback"])
    Path(result_path).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__" and IN_BLENDER:
    sys.exit(blender_main(sys.argv))

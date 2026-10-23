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

    def run(self, operation: str, params: dict, tag: str = "") -> dict:
        tag = tag or operation
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
                proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, timeout=self.timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise BlenderError(f"Blender '{operation}' timed out after {self.timeout_s}s (log: {log_path})") from exc
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


def _min_volume_rotation(pts, center, span, step):
    grid = np.arange(-span, span + 1e-9, step)
    angles = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3) + center
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
    best = _min_volume_rotation(pts, np.zeros(3), max_tilt, 5.0)
    best = _min_volume_rotation(pts, best, 5.0, 1.0)
    best = _min_volume_rotation(pts, best, 1.0, 0.25)
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


def ground_and_center(obj, followers=()):
    """Centre on XY, rest on Z=0; ``followers`` get the same translation."""
    lo, hi = bbox([obj])
    center = (lo + hi) / 2.0
    offset = (-center[0], -center[1], -lo[2])
    for o in (obj, *followers):
        o.location = offset
        apply_transforms(o)


def fit_scale(obj, target_dims, fit_dimension: str, axis_scale: dict):
    lo, hi = bbox([obj])
    dims = hi - lo
    if fit_dimension == "longest":
        current, target = float(dims.max()), float(max(target_dims))
    else:
        axis = {"length": 0, "width": 1, "height": 2}[fit_dimension]
        current, target = float(dims[axis]), float(target_dims[axis])
    factor = target / max(current, 1e-9)
    scale = [factor, factor, factor]
    for axis_name, extra in (axis_scale or {}).items():
        idx = "XYZ".index(axis_name.upper())
        scale[idx] *= min(max(float(extra), 0.7), 1.3)  # vision-driven proportion fixes, bounded
    obj.scale = scale
    apply_transforms(obj)
    lo, hi = bbox([obj])
    dims = hi - lo
    error = [round(float(d / t - 1.0), 4) if t > 0 else 0.0 for d, t in zip(dims, target_dims)]
    log(f"scale: x{factor:.4f}, dims {np.round(dims, 3).tolist()} vs target {target_dims}")
    return {"scale_factor": factor, "dimension_error": error}


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
    rot = p.get("import_rotation_deg") or [0, 0, 0]
    if any(rot):
        raw.rotation_euler = [math.radians(a) for a in rot]
        apply_transforms(raw)
    raw_faces = face_count(raw)

    lo, hi = bbox([raw])
    size = float((hi - lo).max())
    report = {"raw_faces": raw_faces}
    report.update(bm_cleanup(raw, p["merge_distance"] * size, p["min_island_ratio"], fill_holes=False))

    report["rotation_applied_deg"] = align_to_axes(
        raw, max_tilt=float(p.get("max_tilt_correction", 45.0)),
        longest_to_x=p.get("orient_longest_axis", True)) if p.get("auto_orient", True) else [0.0, 0.0, 0.0]
    ground_and_center(raw)
    report.update(fit_scale(raw, p["dimensions"], p["fit_dimension"], p.get("axis_scale")))
    ground_and_center(raw)
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
    voxel_size = size / max(16, int(p["voxel_resolution"]))
    target_faces = int(p["target_faces"])

    if mode in ("decimate", "none") and p.get("fix_intersections") and report["collisions_before"]:
        obj = union_islands(obj)
        obj.name = p["name"]
        obj["asset_role"] = "target"

    if mode == "quadriflow":
        # Quadriflow needs a manifold input; voxel remesh guarantees one and also
        # unions intersecting shells.
        voxel_remesh(obj, voxel_size)
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
        obj, 1e-6 * size, p["min_island_ratio"], fill_holes=p.get("fill_holes", True)).items()})
    smooth(obj, int(p.get("smooth_iterations", 0)))
    auto_smooth(obj, float(p.get("auto_smooth_angle", 40.0)))
    report["collisions_after"] = island_collisions(obj)
    ground_and_center(obj, followers=[high])  # bake source must stay in the target's frame
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


CLASSIFY_CANDIDATES = [
    {"type": "box", "bevel": 0.03}, {"type": "cylinder", "axis": "Z"}, {"type": "cylinder", "axis": "X"},
    {"type": "cylinder", "axis": "Y"}, {"type": "cone", "axis": "+Z", "taper": 0.0},
    {"type": "pyramid", "axis": "+Z", "taper": 0.0}, {"type": "sphere"}, {"type": "hemisphere", "axis": "+Z"},
    {"type": "half_cylinder", "axis": "X"}, {"type": "half_cylinder", "axis": "Y"},
    {"type": "wedge", "axis": "X"}, {"type": "wedge", "axis": "Y"},
]


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
    if comps:
        if p.get("refine", True):
            refine_components(comps, (world_coords(scan) - lo) / size)
        report["source"] = "recipe"
        limit = float(p.get("recipe_max_error", 0.12))
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

    # Build groups: body ("" part) and one group per part name
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
        try:
            obj = union_objects(unions, p.get("union", True))
        except RuntimeError as exc:
            log(f"primitives: {exc}; joining without boolean")
            obj = union_objects([o for o in unions if _alive(o)], False)
        subtract_objects(obj, cutters)  # openings: doors, arches, crenels
        built[part_name] = obj

    if "" not in built:
        for o in built.values():
            bpy.data.objects.remove(o, do_unlink=True)
        raise RuntimeError("recipe has no body components")
    shape_objs = list(built.values())
    fit = fit_error(shape_objs, scan_pts, scan_bvh, diag, rng)
    report.update(fit_error=fit["error"], fit_p95_m=fit["p95_m"],
                  cage_hint=round(min(0.15, 1.5 * fit["p95_m"] / float(size.max())), 4))
    use = p.get("mode", "auto") == "primitives" or fit["error"] <= limit
    report["used"] = bool(use)
    log(f"primitives: {report['source']} fit error {fit['error']:.4f} (limit {limit}) -> "
        f"{'using primitives' if use else 'keeping scan mesh'}")
    if not use:
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


def derive_pbr(albedo_img, rough_img, metal_img, materials, ids=None):
    """Material-ID per texel (template IDs, else nearest spec colour), then roughness/metallic.

    Roughness gets a small luminance-driven variation so surfaces read as
    worn rather than flat; metallic stays binary per material, which is what
    PBR validators expect.
    """
    albedo = image_to_array(albedo_img)
    rgb = albedo[..., :3]
    palette = np.array([srgb_from_linear(m["base_color"]) for m in materials], dtype=np.float32)
    dist = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
    colour_ids = dist.argmin(axis=-1)
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
    report["material_coverage"] = derive_pbr(albedo, rough, metal, materials, material_ids)

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
    for side, s in (("L", 1), ("R", -1)):
        y = s * Wd * 0.3
        for leg, fx, parent in (("front", 0.22, "chest"), ("back", -0.25, "hips")):
            top = (x(fx), y, body_z - 0.05 * H)
            mid = (x(fx), y, z0 + 0.35 * H)
            bot = (x(fx), y, z0 + 0.05 * H)
            toe = (x(fx + 0.04), y, z0)
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


def rigify_armature(rig_type, lo, hi):
    import addon_utils

    addon_utils.enable("rigify", default_set=True)
    before = set(bpy.data.objects)
    if rig_type == "quadruped":
        op = getattr(bpy.ops.object, "armature_wolf_metarig_add", None) or bpy.ops.object.armature_basic_quadruped_metarig_add
    else:
        op = bpy.ops.object.armature_human_metarig_add
    op()
    metarig = [o for o in bpy.data.objects if o not in before][0]
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
    arm = None
    if p.get("method", "rigify") == "rigify":
        try:
            arm = rigify_armature(rig_type, lo, hi)
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
            bones = _biped_bones(lo, hi)
        arm = build_armature(f"{obj.name}_rig", bones)
        report["method"] = "generated"
    arm["asset_role"] = "armature"

    select_only(obj, arm, active=arm)
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc:
        log(f"automatic weights failed: {exc}")
    coverage = weight_coverage(obj)
    report["auto_weight_coverage"] = round(coverage, 3)
    if coverage < 0.95:
        # Bone heat fails on non-manifold or self-intersecting meshes
        log(f"bone-heat covered {coverage:.0%} of vertices; using proximity weights")
        if obj.parent is None:
            obj.parent = arm
            obj.matrix_parent_inverse = arm.matrix_world.inverted()
        proximity_weights(obj, arm)
        report["weights"] = "proximity"
    else:
        report["weights"] = "bone_heat"
    report["deform_bones"] = sum(1 for b in arm.data.bones if b.use_deform)
    report["weight_coverage"] = round(weight_coverage(obj), 3)
    save_blend(p["output_blend"])
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

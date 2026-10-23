"""Stages 3-5 (topology, UV/PBR, segmentation/rigging), Stage 6 renders and the
final export, executed in headless Blender.

The file has two halves:

* **Host side** (any Python): :class:`BlenderRunner` writes a JSON job, runs
  ``blender --background --factory-startup --python blender_automation.py --
  --job job.json`` and reads back a JSON result (metrics, render paths, ...).
* **Blender side** (runs inside Blender's bundled Python, needs ``bpy``):
  :func:`blender_main` executes the job. It only uses ``bpy``, ``bmesh``,
  ``mathutils`` and the ``numpy`` Blender ships with.

Tested against Blender 4.2 LTS; written to tolerate 3.6 - 4.x API differences
(auto-smooth, EEVEE engine ids, importer operator names).
"""

from __future__ import annotations

import glob
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import bpy  # type: ignore[import-not-found]
    import bmesh  # type: ignore[import-not-found]
    import numpy as np
    from mathutils import Matrix, Vector  # type: ignore[import-not-found]
    from mathutils.bvhtree import BVHTree  # type: ignore[import-not-found]

    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

ASSET_COLLECTION = "ASSET"
SOURCE_COLLECTION = "_PIPELINE_SOURCE"
MAIN_OBJECT = "asset_body"

# ===========================================================================
# Host side
# ===========================================================================


class BlenderError(RuntimeError):
    pass


def find_blender(explicit: Optional[str] = None) -> str:
    """Locate a Blender executable: explicit path, $BLENDER_PATH, PATH, defaults."""
    candidates: list[Optional[str]] = [explicit, os.environ.get("BLENDER_PATH"), shutil.which("blender")]
    system = platform.system()
    if system == "Windows":
        candidates += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender*\blender.exe"), reverse=True)
    elif system == "Darwin":
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    else:
        candidates += sorted(glob.glob("/opt/blender*/blender"), reverse=True)
        candidates += ["/snap/bin/blender", os.path.expanduser("~/blender/blender")]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(cand)
    raise FileNotFoundError(
        "Blender executable not found. Set blender.executable in the config or BLENDER_PATH."
    )


class BlenderRunner:
    def __init__(self, executable: Optional[str] = None, timeout_s: float = 7200.0):
        self.executable = find_blender(executable)
        self.timeout_s = timeout_s

    def version(self) -> str:
        out = subprocess.run([self.executable, "--version"], capture_output=True, text=True, timeout=120)
        return (out.stdout.strip().splitlines() or ["unknown"])[0]

    def run_job(self, job: dict[str, Any], job_dir: Path, name: str) -> dict[str, Any]:
        job_dir.mkdir(parents=True, exist_ok=True)
        job_path = job_dir / f"{name}.job.json"
        result_path = job_dir / f"{name}.result.json"
        log_path = job_dir / f"{name}.log"
        if result_path.exists():
            result_path.unlink()
        job = {**job, "result_path": str(result_path.resolve())}
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        cmd = [
            self.executable, "--background", "--factory-startup", "--python-exit-code", "3",
            "--python", str(Path(__file__).resolve()), "--", "--job", str(job_path.resolve()),
        ]
        start = time.monotonic()
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, timeout=self.timeout_s)
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        result["log_path"] = str(log_path)
        result["wall_seconds"] = round(time.monotonic() - start, 1)
        if proc.returncode != 0 or not result.get("ok"):
            tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-40:])
            raise BlenderError(
                f"Blender job '{name}' failed (exit {proc.returncode}): {result.get('error', 'no result')}\n"
                f"--- log tail ({log_path}) ---\n{tail}"
            )
        return result

    def process(
        self,
        mesh_path: Path,
        spec: dict[str, Any],
        params: dict[str, Any],
        work_dir: Path,
        name: str,
        stage5: dict[str, bool],
        render: bool = True,
    ) -> dict[str, Any]:
        """Stages 3, 4, (5) and the Stage 6 renders for one refinement iteration."""
        job = {
            "action": "process",
            "input_mesh": str(Path(mesh_path).resolve()),
            "spec": spec,
            "params": params,
            "blend_path": str((work_dir / f"{name}.blend").resolve()),
            "texture_dir": str((work_dir / "textures").resolve()),
            "render_dir": str((work_dir / "renders").resolve()),
            "segment": bool(stage5.get("segment")),
            "rig": bool(stage5.get("rig")),
            "render": render,
        }
        return self.run_job(job, work_dir, name)

    def export(self, blend_path: Path, fmt: str, out_path: Path, export_parts: bool, work_dir: Path) -> dict[str, Any]:
        job = {
            "action": "export",
            "blend_path": str(Path(blend_path).resolve()),
            "format": fmt.upper(),
            "output_path": str(Path(out_path).resolve()),
            "export_parts": export_parts,
        }
        return self.run_job(job, work_dir, f"export_{fmt.lower()}")


# ===========================================================================
# Blender side
# ===========================================================================


def _log(msg: str) -> None:
    print(f"[asset-pipeline] {msg}", flush=True)


@contextmanager
def _step(result: dict, name: str):
    _log(f"--> {name}")
    start = time.time()
    info: dict[str, Any] = {}
    try:
        yield info
    finally:
        result["steps"][name] = {"seconds": round(time.time() - start, 2), **info}


def _op_exists(module: Any, name: str) -> bool:
    return name in dir(module)


# -- scene / selection helpers ----------------------------------------------

def _object_mode() -> None:
    obj = bpy.context.view_layer.objects.active
    if obj is not None and obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def _select(objs: Iterable[Any], active: Any = None) -> None:
    _object_mode()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    objs = list(objs)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or (objs[0] if objs else None)


def _ensure_collection(name: str) -> Any:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _move_to_collection(obj: Any, coll: Any) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _layer_collection(name: str) -> Any:
    def walk(lc):
        if lc.collection.name == name:
            return lc
        for child in lc.children:
            found = walk(child)
            if found:
                return found
        return None

    return walk(bpy.context.view_layer.layer_collection)


def _apply_modifier(obj: Any, mod: Any) -> None:
    _select([obj])
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _coords(obj: Any) -> "np.ndarray":
    me = obj.data
    arr = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", arr)
    arr = arr.reshape(-1, 3)
    if obj.matrix_world != Matrix.Identity(4):
        mw = np.array(obj.matrix_world)
        arr = arr @ mw[:3, :3].T + mw[:3, 3]
    return arr


def _bbox(objs: Iterable[Any]) -> tuple["Vector", "Vector"]:
    bpy.context.view_layer.update()  # matrix_world is stale after setting .location
    pts = np.concatenate([_coords(o) for o in objs if o.type == "MESH" and len(o.data.vertices)])
    return Vector(pts.min(axis=0)), Vector(pts.max(axis=0))


def _face_islands(bm: Any) -> list[list[Any]]:
    """Connected face groups (union-find over edges)."""
    bm.verts.index_update()
    parent = list(range(len(bm.verts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in bm.edges:
        a, b = find(e.verts[0].index), find(e.verts[1].index)
        if a != b:
            parent[a] = b
    groups: dict[int, list[Any]] = {}
    for f in bm.faces:
        groups.setdefault(find(f.verts[0].index), []).append(f)
    return sorted(groups.values(), key=len, reverse=True)


# -- Stage 3: import & topology -------------------------------------------------

def _import_mesh(path: str) -> Any:
    ext = Path(path).suffix.lower()
    before = set(bpy.data.objects)
    if ext == ".obj":
        if _op_exists(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".ply":
        if _op_exists(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif ext == ".stl":
        if _op_exists(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {path}")
    # Bake parent transforms (glTF nodes) into the meshes, drop empties.
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    _select(meshes, meshes[0])
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.name = MAIN_OBJECT
    obj.data.name = MAIN_OBJECT
    _move_to_collection(obj, _ensure_collection(ASSET_COLLECTION))
    return obj


def _cleanup(obj: Any, P: dict, info: dict) -> None:
    """Merge doubles, drop degenerate/loose geometry and floaters, fill holes."""
    lo, hi = _bbox([obj])
    diag = (hi - lo).length
    merge = diag * float(P.get("merge_distance_rel", 1e-5))
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    info["faces_in"] = len(bm.faces)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge)
    bmesh.ops.dissolve_degenerate(bm, dist=merge, edges=bm.edges)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")

    islands = _face_islands(bm)
    min_faces = max(8, int(len(bm.faces) * float(P.get("min_island_ratio", 0.002))))
    floaters = [f for isl in islands[1:] if len(isl) < min_faces for f in isl]
    if floaters:
        bmesh.ops.delete(bm, geom=floaters, context="FACES")
    info["islands"] = len(islands)
    info["floater_faces_removed"] = len(floaters)

    if P.get("fill_holes", True):
        boundary = [e for e in bm.edges if e.is_boundary]
        if boundary:
            filled = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
            info["holes_filled"] = len(filled["faces"])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    info["faces_out"] = len(obj.data.polygons)


def _rotation_from_euler_deg(deg: list[float]) -> "Matrix":
    from mathutils import Euler

    return Euler([math.radians(d) for d in deg], "XYZ").to_matrix().to_4x4()


def _auto_orient(obj: Any, target: list[float], info: dict) -> None:
    """Pick the 90-degree axis permutation whose bbox best matches the spec.

    Image-to-3D models output arbitrary orientations; matching the *ratios* of
    the bounding box to the spec fixes the common "lying on its side" case.
    Identity wins unless another permutation is clearly (>15 %) better.
    """
    import itertools

    lo, hi = _bbox([obj])
    dims = np.array(hi - lo)
    tgt = np.array(target, dtype=float)

    def score(perm: tuple[int, ...]) -> float:
        d = dims[list(perm)]
        return float(np.abs(np.log((d / d.max()) / (tgt / tgt.max()) + 1e-9)).sum())

    ident = score((0, 1, 2))
    best = min(itertools.permutations(range(3)), key=lambda p: (round(score(p), 3), p[2] != 2))
    info["orient_scores"] = {"identity": round(ident, 3), "best": round(score(best), 3), "perm": list(best)}
    if best == (0, 1, 2) or score(best) > ident * 0.85 - 1e-6:
        return
    rot = Matrix(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
    for new_axis, old_axis in enumerate(best):
        rot[new_axis][old_axis] = 1.0
    if rot.determinant() < 0:  # keep it a proper rotation (no mirroring)
        rot[0] = -rot[0]
    obj.data.transform(rot.to_4x4())
    obj.data.update()
    info["auto_rotated"] = True


def _fit_scale(obj: Any, P: dict, info: dict) -> None:
    """Scale to real-world metres on the primary axis and set the origin."""
    target = [float(x) for x in P["target_dimensions_m"]]
    fit = [t * float(P.get("fit_compensation", 1.0)) for t in target]
    lo, hi = _bbox([obj])
    dims = hi - lo
    info["raw_dimensions"] = [round(d, 5) for d in dims]
    idx = "xyz".index(P.get("primary_axis", "z"))
    if P.get("nonuniform_fit"):
        scale = Vector([t / max(d, 1e-9) for t, d in zip(fit, dims)])
    else:
        s = fit[idx] / max(dims[idx], 1e-9)
        scale = Vector((s, s, s))
    obj.data.transform(Matrix.Diagonal((*scale, 1.0)))
    lo, hi = _bbox([obj])
    center = (lo + hi) / 2
    offset = Vector((center.x, center.y, lo.z if P.get("origin") == "bottom_center" else center.z))
    obj.data.transform(Matrix.Translation(-offset))
    obj.data.update()
    lo, hi = _bbox([obj])
    final = hi - lo
    info["scale_factor"] = [round(s, 6) for s in scale]
    info["dimensions_m"] = [round(d, 4) for d in final]
    info["target_dimensions_m"] = target


def _make_source_copy(obj: Any, P: dict, info: dict) -> Any:
    """Keep the cleaned high-res mesh as the bake/shrinkwrap source."""
    src = obj.copy()
    src.data = obj.data.copy()
    src.name = "SOURCE_highres"
    _ensure_collection(SOURCE_COLLECTION).objects.link(src)
    max_faces = int(P.get("source_max_faces", 400_000))
    if len(src.data.polygons) > max_faces:
        mod = src.modifiers.new("Decimate", "DECIMATE")
        mod.ratio = max_faces / len(src.data.polygons)
        _apply_modifier(src, mod)
    src.data.polygons.foreach_set("use_smooth", [True] * len(src.data.polygons))
    info["source_faces"] = len(src.data.polygons)
    return src


def _remesh(obj: Any, src: Any, P: dict, info: dict) -> None:
    mode = P.get("remesh_mode", "quadriflow")
    target_faces = int(P.get("target_faces", 12000))
    lo, hi = _bbox([obj])
    info["mode"] = mode
    if mode in ("voxel", "quadriflow"):
        voxel = max(hi - lo) / float(P.get("voxel_resolution", 160))
        mod = obj.modifiers.new("VoxelRemesh", "REMESH")
        mod.mode = "VOXEL"
        mod.voxel_size = voxel
        mod.adaptivity = 0.0
        _apply_modifier(obj, mod)
        info["voxel_size"] = round(voxel, 6)
        info["voxel_faces"] = len(obj.data.polygons)
        if mode == "quadriflow":
            # QuadriFlow's pre-check rejects any edge whose endpoints match within
            # an *absolute* 1e-4, and the voxel mesher leaves sub-micron slivers on
            # metre-scale assets. Work in a 1000-unit frame and merge only truly
            # coincident verts (merging at voxel scale pinches thin walls).
            norm = 1000.0 / max(max(hi - lo), 1e-9)
            obj.data.transform(Matrix.Scale(norm, 4))
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-3)
            bm.to_mesh(obj.data)
            bm.free()
            before = len(obj.data.polygons)
            _select([obj])
            try:
                res = bpy.ops.object.quadriflow_remesh(
                    target_faces=target_faces, mode="FACES", use_mesh_symmetry=False,
                    use_preserve_sharp=False, use_preserve_boundary=False,
                    preserve_attributes=False, smooth_normals=False, seed=0,
                )
                # QuadriFlow reports FINISHED even when its manifold check refuses
                # the mesh, so also require that the topology actually changed.
                ok = "FINISHED" in res and 0 < len(obj.data.polygons) != before
            except RuntimeError as exc:
                _log(f"Quadriflow error: {exc}")
                ok = False
            obj.data.transform(Matrix.Scale(1.0 / norm, 4))
            if not ok:
                info["quadriflow_failed"] = True
                mode = "decimate"
    if mode == "decimate" and len(obj.data.polygons) > target_faces:
        mod = obj.modifiers.new("Decimate", "DECIMATE")
        mod.ratio = target_faces / len(obj.data.polygons)
        _apply_modifier(obj, mod)

    iterations = int(P.get("smooth_iterations", 2))
    if iterations > 0 and mode != "none":
        mod = obj.modifiers.new("Relax", "SMOOTH")
        mod.factor = 0.5
        mod.iterations = iterations
        _apply_modifier(obj, mod)
    if P.get("shrinkwrap", True) and mode != "none":
        mod = obj.modifiers.new("Shrinkwrap", "SHRINKWRAP")
        mod.target = src
        mod.wrap_method = "NEAREST_SURFACEPOINT"
        _apply_modifier(obj, mod)
    info["faces"] = len(obj.data.polygons)


def _primitive(name: str, shape: str, center: "Vector", size: "Vector") -> Any:
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    if shape == "cylinder":
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.5, radius2=0.5, depth=1.0)
    else:
        bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _boolean(obj: Any, other: Any, operation: str, solver: str = "EXACT") -> None:
    mod = obj.modifiers.new(f"Bool_{operation}", "BOOLEAN")
    mod.operation = operation
    mod.object = other
    try:
        mod.solver = solver
    except TypeError:  # Blender 4.5+ renamed FAST -> FLOAT
        mod.solver = "FLOAT" if solver == "FAST" else "EXACT"
    _apply_modifier(obj, mod)


def _volume(obj: Any) -> float:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    vol = abs(bm.calc_volume())
    bm.free()
    return vol


def _non_manifold_edges(obj: Any) -> int:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    count = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return count


def _safe_union(base: Any, piece: Any) -> bool:
    """Boolean-union ``piece`` into ``base``; revert if the result is degenerate.

    Near-coplanar noisy faces (typical of AI meshes) can make the Exact solver
    drop whole shells and the Fast solver leave non-manifold seams, so a union
    must keep at least the larger input's volume and add no non-manifold edges.
    """
    expected_min = max(_volume(base), _volume(piece)) * 0.98
    max_nm = _non_manifold_edges(base) + _non_manifold_edges(piece)
    original = base.data.copy()
    for solver in ("EXACT", "FAST"):
        _boolean(base, piece, "UNION", solver)
        if (len(base.data.polygons) and _volume(base) >= expected_min
                and _non_manifold_edges(base) <= max_nm):
            bpy.data.meshes.remove(original)
            return True
        failed = base.data
        base.data = original.copy()
        bpy.data.meshes.remove(failed)
    bpy.data.meshes.remove(original)
    return False


def _inside(tree: "BVHTree", p: "Vector", reach: float) -> bool:
    """Ray-parity point-in-mesh test, majority vote over three directions.

    More robust than nearest-face normals when a union left thin internal
    pockets (a void's boundary is crossed twice, so parity stays correct).
    """
    votes = 0
    eps = reach * 1e-6
    for d in (Vector((0.0, 0.0, 1.0)), Vector((1.0, 0.0, 0.0)), Vector((0.577, -0.577, 0.577))):
        origin, hits = p.copy(), 0
        for _ in range(256):
            loc, _, _, _ = tree.ray_cast(origin, d, reach)
            if loc is None:
                break
            hits += 1
            origin = loc + d * eps
        votes += hits % 2
    return votes >= 2


def _enclosed(base_tree: "BVHTree", piece: Any, reach: float) -> bool:
    """True if (nearly) every vertex of ``piece`` lies inside the ``base`` surface."""
    pts = _coords(piece)[:: max(1, len(piece.data.vertices) // 64)]
    inside = sum(_inside(base_tree, Vector(co), reach) for co in pts)
    return inside >= 0.95 * len(pts)


def _bvh(obj: Any) -> "BVHTree":
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree


def _booleans(obj: Any, P: dict, info: dict) -> Any:
    """Union intersecting loose parts, then apply corrective boolean cutters."""
    lo, hi = _bbox([obj])
    size = hi - lo
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n_islands = len(_face_islands(bm))
    bm.free()
    info["islands_before"] = n_islands

    if P.get("boolean_union", True) and n_islands > 1:
        _select([obj])
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")
        pieces = sorted(bpy.context.selected_objects, key=lambda o: len(o.data.polygons), reverse=True)
        pieces.sort(key=_volume, reverse=True)
        base, rest = pieces[0], pieces[1:]
        unions, failed, interior = 0, 0, 0
        changed = True
        while changed and rest:
            changed = False
            base_tree = _bvh(base)
            for piece in list(rest):
                if _enclosed(base_tree, piece, (hi - lo).length * 2):  # hidden interior shell: drop it
                    rest.remove(piece)
                    bpy.data.objects.remove(piece, do_unlink=True)
                    interior += 1
                    continue
                if base_tree.overlap(_bvh(piece)):
                    rest.remove(piece)
                    if _safe_union(base, piece):
                        bpy.data.objects.remove(piece, do_unlink=True)
                        unions += 1
                        changed = True
                        break
                    failed += 1
                    rest.append(piece)  # keep it as a separate shell
        info["interior_shells_removed"] = interior
        info["boolean_union_failed"] = failed
        if rest:
            _select([base, *rest], base)
            bpy.ops.object.join()
        base.name = MAIN_OBJECT
        info["boolean_unions"] = unions
        obj = base

    cutters = P.get("cutters") or []
    for i, cut in enumerate(cutters):
        center = lo + Vector([size[a] * float(cut.get("center_norm", [0.5] * 3)[a]) for a in range(3)])
        dims = Vector([max(size[a] * float(cut.get("size_norm", [0.1] * 3)[a]), 1e-4) for a in range(3)])
        cutter = _primitive(f"cutter_{i}", cut.get("shape", "box"), center, dims)
        _boolean(obj, cutter, "DIFFERENCE")
        bpy.data.objects.remove(cutter, do_unlink=True)
    info["cutters_applied"] = len(cutters)
    return obj


def _shade(obj: Any, P: dict) -> None:
    angle = math.radians(float(P.get("smooth_angle_deg", 35.0)))
    _select([obj])
    if _op_exists(bpy.ops.object, "shade_smooth_by_angle"):  # Blender 4.1+
        bpy.ops.object.shade_smooth_by_angle(angle=angle)
    else:
        bpy.ops.object.shade_smooth()
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = angle


# -- Stage 4: UVs, baking, PBR --------------------------------------------------

def _unwrap(obj: Any, P: dict, info: dict) -> None:
    method = P.get("uv_method", "smart")
    margin = float(P.get("island_margin", 0.004))
    _select([obj])
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if method == "seams":
        # Seams = Smart-UV island borders + sharp edges, then a conformal
        # (angle-based) unwrap inside each island: lower distortion than Smart
        # UV's planar projections. Sharp edges alone leave a smooth closed mesh
        # nearly seamless, which cannot flatten.
        bpy.ops.uv.smart_project(angle_limit=math.radians(float(P.get("uv_angle_limit_deg", 66.0))),
                                 island_margin=margin)
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.seams_from_islands()
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.edges_select_sharp(sharpness=math.radians(float(P.get("seam_angle_deg", 50.0))))
        bpy.ops.mesh.mark_seam(clear=False)
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=margin)
    elif method == "lightmap":
        bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_MARGIN_DIV=0.2)
    else:
        bpy.ops.uv.smart_project(angle_limit=math.radians(float(P.get("uv_angle_limit_deg", 66.0))),
                                 island_margin=margin)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(rotate=True, margin=margin)
    bpy.ops.object.mode_set(mode="OBJECT")
    info["method"] = method


def _setup_cycles(P: dict, samples: int) -> str:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    device = "CPU"
    if P.get("use_gpu", True):
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            requested = str(P.get("compute_device", "AUTO")).upper()
            order = [requested] if requested not in ("AUTO", "") else ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"]
            for kind in order:
                if kind == "CPU":
                    break
                try:
                    prefs.compute_device_type = kind
                except TypeError:
                    continue
                (prefs.refresh_devices if hasattr(prefs, "refresh_devices") else prefs.get_devices)()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    device = kind
                    break
        except Exception as exc:  # noqa: BLE001 - GPU setup must never kill the job
            _log(f"GPU setup failed, using CPU: {exc}")
    scene.cycles.device = "GPU" if device != "CPU" else "CPU"
    return device


def _new_image(name: str, size: int, non_color: bool) -> Any:
    img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def _source_has_color(src: Any) -> str:
    """Give the bake source a material that exposes its colour; report the kind."""
    me = src.data
    attrs = getattr(me, "color_attributes", None)
    if attrs and len(attrs):
        layer = attrs.active_color or attrs[0]
        mat = bpy.data.materials.new("SOURCE_vertex_color")
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = nodes["Principled BSDF"]
        col = nodes.new("ShaderNodeVertexColor")
        col.layer_name = layer.name
        links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
        me.materials.clear()
        me.materials.append(mat)
        return "vertex_color"
    if any(slot.material for slot in src.material_slots):
        return "material"
    return "none"


def _srgb(lin: "np.ndarray") -> "np.ndarray":
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def _fbm(size: int, rng: "np.random.Generator", octaves: int = 4, cells: int = 6) -> "np.ndarray":
    """Smooth fractal value noise in [-1, 1], used for surface variation."""
    out = np.zeros((size, size), dtype=np.float32)
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        grid = rng.random((cells + 2, cells + 2)).astype(np.float32) * 2 - 1
        coords = np.linspace(0, cells, size, endpoint=False, dtype=np.float32)
        i0 = coords.astype(np.int32)
        t = coords - i0
        t = t * t * (3 - 2 * t)
        ty, tx = t[:, None], t[None, :]
        g00, g01 = grid[np.ix_(i0, i0)], grid[np.ix_(i0, i0 + 1)]
        g10, g11 = grid[np.ix_(i0 + 1, i0)], grid[np.ix_(i0 + 1, i0 + 1)]
        out += amp * ((g00 * (1 - tx) + g01 * tx) * (1 - ty) + (g10 * (1 - tx) + g11 * tx) * ty)
        total += amp
        amp *= 0.5
        cells *= 2
    return out / total


def _set_pixels(img: Any, gray_or_rgb: "np.ndarray") -> None:
    size = img.size[0]
    if gray_or_rgb.ndim == 1:
        rgb = np.repeat(gray_or_rgb[:, None], 3, axis=1)
    else:
        rgb = gray_or_rgb
    rgba = np.empty((size * size, 4), dtype=np.float32)
    rgba[:, :3] = rgb
    rgba[:, 3] = 1.0
    img.pixels.foreach_set(rgba.ravel())
    img.update()


def _bake_textures(obj: Any, src: Any, spec: dict, P: dict, texture_dir: str, info: dict) -> dict[str, str]:
    """Bake albedo + normal from the high-res source; derive roughness/metallic.

    Roughness and metallic come from the spec materials: every albedo texel is
    assigned to the closest spec material colour, then modulated with fractal
    noise so surfaces don't read as flat CG.
    """
    size = int(P.get("texture_size", 2048))
    os.makedirs(texture_dir, exist_ok=True)
    info["device"] = _setup_cycles(P, int(P.get("bake_samples", 16)))
    lo, hi = _bbox([obj])
    extrusion = (hi - lo).length * float(P.get("cage_extrusion_rel", 0.01))
    margin = int(P.get("bake_margin_px", 16))
    name = spec.get("name", "asset")
    source_kind = _source_has_color(src)
    info["source_color"] = source_kind

    bake_mat = bpy.data.materials.new("BAKE_target")
    bake_mat.use_nodes = True
    tex_node = bake_mat.node_tree.nodes.new("ShaderNodeTexImage")
    bake_mat.node_tree.nodes.active = tex_node
    obj.data.materials.clear()
    obj.data.materials.append(bake_mat)

    def bake(kind: str, img: Any, **kw: Any) -> None:
        tex_node.image = img
        _select([src, obj], obj)
        bpy.ops.object.bake(type=kind, use_selected_to_active=True, cage_extrusion=extrusion,
                            margin=margin, use_clear=True, **kw)

    images: dict[str, Any] = {}
    albedo = _new_image(f"{name}_albedo", size, non_color=False)
    if source_kind != "none":
        bake("DIFFUSE", albedo, pass_filter={"COLOR"})
    images["albedo"] = albedo
    normal = _new_image(f"{name}_normal", size, non_color=True)
    bake("NORMAL", normal, normal_space="TANGENT")
    images["normal"] = normal

    materials = spec.get("materials") or [{"base_color": [0.5] * 3, "roughness": 0.5, "metallic": 0.0}]
    rng = np.random.default_rng(int(P.get("seed", 0)))
    noise = _fbm(size, rng).ravel()
    cols = _srgb(np.array([m["base_color"] for m in materials], dtype=np.float32))
    if source_kind == "none":
        rgb = np.tile(cols[0], (size * size, 1)) * (1.0 + 0.12 * noise[:, None])
        _set_pixels(albedo, np.clip(rgb, 0, 1))
        idx = np.zeros(size * size, dtype=np.int32)
    else:
        px = np.empty(size * size * 4, dtype=np.float32)
        albedo.pixels.foreach_get(px)
        rgb = px.reshape(-1, 4)[:, :3]
        idx = np.empty(size * size, dtype=np.int32)
        chunk = 1 << 20
        for s in range(0, len(rgb), chunk):
            d = ((rgb[s:s + chunk, None, :] - cols[None, :, :]) ** 2).sum(-1)
            idx[s:s + chunk] = d.argmin(1)
    rough = np.array([m.get("roughness", 0.5) for m in materials], dtype=np.float32)[idx]
    metal = np.array([m.get("metallic", 0.0) for m in materials], dtype=np.float32)[idx]
    roughness = _new_image(f"{name}_roughness", size, non_color=True)
    _set_pixels(roughness, np.clip(rough + 0.08 * noise, 0.02, 1.0))
    metallic = _new_image(f"{name}_metallic", size, non_color=True)
    _set_pixels(metallic, np.clip(metal + 0.03 * noise, 0.0, 1.0))
    images["roughness"], images["metallic"] = roughness, metallic
    info["material_coverage"] = {
        materials[i].get("name", str(i)): round(float((idx == i).mean()), 3) for i in range(len(materials))
    }

    paths = {}
    for kind, img in images.items():
        path = os.path.join(texture_dir, f"{name}_{kind}.png")
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        paths[kind] = path
    _build_pbr_material(obj, name, images)
    bpy.data.materials.remove(bake_mat)
    info["texture_size"] = size
    return paths


def _build_pbr_material(obj: Any, name: str, images: dict[str, Any]) -> Any:
    """Wire the maps into a standard Principled BSDF."""
    mat = bpy.data.materials.new(f"M_{name}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.location = (300, 0)

    def tex(key: str, y: int) -> Any:
        node = nodes.new("ShaderNodeTexImage")
        node.image = images[key]
        node.label = key
        node.location = (-400, y)
        return node

    links.new(tex("albedo", 300).outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex("metallic", 0).outputs["Color"], bsdf.inputs["Metallic"])
    links.new(tex("roughness", -300).outputs["Color"], bsdf.inputs["Roughness"])
    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.space = "TANGENT"
    nmap.location = (0, -600)
    links.new(tex("normal", -600).outputs["Color"], nmap.inputs["Color"])
    links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


def _flat_material(obj: Any, spec: dict) -> None:
    """Fallback when baking is disabled: a plain Principled BSDF from the spec."""
    m = (spec.get("materials") or [{}])[0]
    mat = bpy.data.materials.new(f"M_{spec.get('name', 'asset')}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*m.get("base_color", [0.5] * 3), 1.0)
    bsdf.inputs["Roughness"].default_value = m.get("roughness", 0.5)
    bsdf.inputs["Metallic"].default_value = m.get("metallic", 0.0)
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# -- Metrics -------------------------------------------------------------------

def _geometry_metrics(obj: Any, max_intersection_faces: int = 150_000) -> dict[str, Any]:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces = len(bm.faces)
    sides = [len(f.verts) for f in bm.faces]
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    watertight = non_manifold == 0 and faces > 0
    volume = bm.calc_volume(signed=True) if watertight else None
    islands = len(_face_islands(bm))

    intersections: Optional[int] = None
    if faces <= max_intersection_faces:
        bm.faces.ensure_lookup_table()
        tree = BVHTree.FromBMesh(bm)
        face_verts = [set(v.index for v in f.verts) for f in bm.faces]
        intersections = sum(
            1 for a, b in tree.overlap(tree) if a < b and not (face_verts[a] & face_verts[b])
        )
    bm.free()
    lo, hi = _bbox([obj])
    return {
        "faces": faces,
        "vertices": len(obj.data.vertices),
        "triangles": sides.count(3),
        "quads": sides.count(4),
        "ngons": sum(1 for s in sides if s > 4),
        "quad_ratio": round(sides.count(4) / max(faces, 1), 3),
        "non_manifold_edges": non_manifold,
        "boundary_edges": boundary,
        "watertight": watertight,
        "volume_m3": round(abs(volume), 6) if volume is not None else None,
        "normals_inverted": bool(volume is not None and volume < 0),
        "loose_parts": islands,
        "self_intersections": intersections,
        "dimensions_m": [round(d, 4) for d in (hi - lo)],
        "bbox_min": [round(v, 4) for v in lo],
        "bbox_max": [round(v, 4) for v in hi],
    }


def _uv_metrics(obj: Any) -> dict[str, Any]:
    """Texel-density distortion: area-weighted mean |log(uv_area/area * k)|.

    0 means every face gets the same texel density; ~0.35 is typical for
    Smart UV Project on organic shapes; >0.6 shows visible stretching.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    layer = bm.loops.layers.uv.active
    if layer is None:
        bm.free()
        return {"has_uvs": False}
    areas, uv_areas = [], []
    out_of_bounds = 0
    for f in bm.faces:
        uvs = [loop[layer].uv for loop in f.loops]
        signed = 0.0
        for i in range(len(uvs)):
            a, b = uvs[i], uvs[(i + 1) % len(uvs)]
            signed += a.x * b.y - b.x * a.y
            if not (-1e-4 <= a.x <= 1.0001 and -1e-4 <= a.y <= 1.0001):
                out_of_bounds += 1
        areas.append(f.calc_area())
        uv_areas.append(signed / 2)
    bm.free()
    area = np.array(areas)
    uv = np.array(uv_areas)
    valid = (area > 1e-12) & (np.abs(uv) > 1e-12)
    k = np.abs(uv[valid]).sum() / max(area[valid].sum(), 1e-12)
    ratio = np.abs(uv[valid]) / (area[valid] * k)
    stretch = float((area[valid] * np.abs(np.log(ratio))).sum() / max(area[valid].sum(), 1e-12))
    return {
        "has_uvs": True,
        "stretch_score": round(stretch, 4),
        "zero_area_uv_faces": int((~valid).sum()),
        "flipped_uv_faces": int((uv < -1e-12).sum()),
        "uv_coverage": round(float(np.abs(uv).sum()), 4),
        "out_of_bounds_uvs": out_of_bounds,
    }


# -- Stage 5: segmentation & rigging -------------------------------------------

def _segment(main: Any, parts: list[dict], P: dict, info: dict) -> list[str]:
    """Split moving parts into child objects with pivots at their hinge points.

    Faces are assigned by region (normalized bbox coords from the spec). If the
    mesh still has several loose islands, whole islands that are mostly inside
    a region move together, so separate components aren't cut in half.
    """
    created: list[str] = []
    asset = _ensure_collection(ASSET_COLLECTION)
    for part in parts:
        lo, hi = _bbox([main])
        size = hi - lo
        rmin = lo + Vector([size[a] * part["region_min"][a] for a in range(3)])
        rmax = lo + Vector([size[a] * part["region_max"][a] for a in range(3)])

        def inside(p: "Vector") -> bool:
            return all(rmin[a] <= p[a] <= rmax[a] for a in range(3))

        bm = bmesh.new()
        bm.from_mesh(main.data)
        bm.faces.ensure_lookup_table()
        islands = _face_islands(bm)
        selected: set[int] = set()
        for isl in islands[1:]:  # never move the main body as a whole
            frac = sum(inside(f.calc_center_median()) for f in isl) / len(isl)
            if frac >= 0.5:
                selected.update(f.index for f in isl)
        mode = "islands"
        if not selected:
            mode = "faces"
            selected = {f.index for f in bm.faces if inside(f.calc_center_median())}
        if selected and len(selected) >= len(bm.faces) * 0.9:
            # The part *is* the asset (door leaf, gate): hinge the whole body.
            bm.free()
            pivot = lo + Vector([size[a] * part["pivot"][a] for a in range(3)])
            main.data.transform(Matrix.Translation(-pivot))
            main.location = pivot
            bpy.context.view_layer.update()
            for child in main.children:
                child.matrix_parent_inverse = main.matrix_world.inverted()
            _add_motion(main, part)
            info.setdefault("parts", []).append({"name": main.name, "mode": "whole_asset",
                                                 "pivot": [round(c, 4) for c in pivot]})
            created.append(main.name)
            continue
        if not selected:
            info.setdefault("skipped", []).append(part["name"])
            bm.free()
            continue

        part_bm = bm.copy()
        part_bm.faces.ensure_lookup_table()
        bmesh.ops.delete(part_bm, geom=[f for f in part_bm.faces if f.index not in selected], context="FACES")
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in selected], context="FACES")
        for b in (bm, part_bm):  # cap the cut so each piece is printable / watertight
            boundary = [e for e in b.edges if e.is_boundary]
            if boundary:
                caps = bmesh.ops.holes_fill(b, edges=boundary, sides=0)["faces"]
                if caps:
                    bmesh.ops.triangulate(b, faces=caps)
            bmesh.ops.recalc_face_normals(b, faces=b.faces)
        bm.to_mesh(main.data)
        bm.free()

        pivot = lo + Vector([size[a] * part["pivot"][a] for a in range(3)])
        me = bpy.data.meshes.new(part["name"])
        part_bm.transform(Matrix.Translation(-pivot))
        part_bm.to_mesh(me)
        part_bm.free()
        for mat in main.data.materials:
            me.materials.append(mat)
        obj = bpy.data.objects.new(part["name"], me)
        asset.objects.link(obj)
        obj.location = pivot
        obj.parent = main
        obj.matrix_parent_inverse = main.matrix_world.inverted()
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))

        _add_motion(obj, part)
        created.append(obj.name)
        info.setdefault("parts", []).append({"name": obj.name, "faces": len(me.polygons), "mode": mode,
                                             "pivot": [round(c, 4) for c in pivot]})
    return created


def _add_motion(obj: Any, part: dict) -> None:
    """Store the motion as custom props (exported as glTF extras / FBX props)
    and limit it with local-space constraints for animators."""
    motion, axis = part.get("motion", "fixed"), part.get("axis", "X").upper()
    limits = part.get("limits", [0.0, 0.0])
    obj["motion"], obj["motion_axis"], obj["motion_limits"] = motion, axis, list(limits)
    if motion == "rotate":
        con = obj.constraints.new("LIMIT_ROTATION")
        con.owner_space = "LOCAL"
        for ax in "xyz":
            setattr(con, f"use_limit_{ax}", True)
            lo_v, hi_v = (math.radians(limits[0]), math.radians(limits[1])) if ax == axis.lower() else (0, 0)
            setattr(con, f"min_{ax}", lo_v)
            setattr(con, f"max_{ax}", hi_v)
    elif motion == "translate":
        con = obj.constraints.new("LIMIT_LOCATION")
        con.owner_space = "LOCAL"
        for i, ax in enumerate("xyz"):
            base = obj.location[i]
            d = limits if ax == axis.lower() else (0.0, 0.0)
            setattr(con, f"use_min_{ax}", True)
            setattr(con, f"use_max_{ax}", True)
            setattr(con, f"min_{ax}", base + d[0])
            setattr(con, f"max_{ax}", base + d[1])


def _fit_armature(rig: Any, mesh: Any, axis: int) -> None:
    """Uniformly scale/translate an armature so it spans the mesh along ``axis``."""
    pts = []
    for b in rig.data.bones:
        pts += [rig.matrix_world @ b.head_local, rig.matrix_world @ b.tail_local]
    rlo = Vector([min(p[a] for p in pts) for a in range(3)])
    rhi = Vector([max(p[a] for p in pts) for a in range(3)])
    mlo, mhi = _bbox([mesh])
    s = (mhi[axis] - mlo[axis]) / max(rhi[axis] - rlo[axis], 1e-9)
    rig.scale = (s, s, s)
    rcenter = (rlo + rhi) / 2 * s
    mcenter = (mlo + mhi) / 2
    rig.location = Vector((mcenter.x - rcenter.x, mcenter.y - rcenter.y, mlo.z - rlo.z * s))
    _select([rig])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _chain_rig(mesh: Any, bones: int) -> Any:
    lo, hi = _bbox([mesh])
    size = hi - lo
    axis = max(range(3), key=lambda a: size[a])
    arm = bpy.data.armatures.new("chain_rig")
    rig = bpy.data.objects.new("rig", arm)
    _ensure_collection(ASSET_COLLECTION).objects.link(rig)
    _select([rig])
    bpy.ops.object.mode_set(mode="EDIT")
    center = (lo + hi) / 2
    prev = None
    for i in range(bones):
        b = arm.edit_bones.new(f"chain_{i:02d}")
        head, tail = center.copy(), center.copy()
        head[axis] = lo[axis] + size[axis] * i / bones
        tail[axis] = lo[axis] + size[axis] * (i + 1) / bones
        b.head, b.tail = head, tail
        if prev:
            b.parent, b.use_connect = prev, True
        prev = b
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def _rig(main: Any, rig_type: str, P: dict, info: dict) -> None:
    """Rigify metarig fitted to the mesh (or a bone chain), then auto weights."""
    rig = None
    metarig_ops = {"humanoid": "armature_human_metarig_add",
                   "quadruped": "armature_basic_quadruped_metarig_add"}
    asset_lc = _layer_collection(ASSET_COLLECTION)
    if rig_type in metarig_ops:
        try:
            import addon_utils  # type: ignore[import-not-found]

            addon_utils.enable("rigify", default_set=True)
            op = metarig_ops[rig_type]
            if not _op_exists(bpy.ops.object, op):
                raise RuntimeError(f"Rigify operator {op} not available")
            bpy.context.view_layer.active_layer_collection = asset_lc
            _select([])
            getattr(bpy.ops.object, op)()
            metarig = bpy.context.view_layer.objects.active
            _fit_armature(metarig, main, axis=2 if rig_type == "humanoid" else 1)
            _select([metarig])
            bpy.ops.pose.rigify_generate()
            rig = getattr(metarig.data, "rigify_target_rig", None) or bpy.data.objects.get("rig")
            _move_to_collection(metarig, _ensure_collection(SOURCE_COLLECTION))
            # Rigify nests its widget collection under the active one; keep the
            # ASSET collection to exportable objects only.
            asset = _ensure_collection(ASSET_COLLECTION)
            for child in list(asset.children):
                asset.children.unlink(child)
                _ensure_collection(SOURCE_COLLECTION).children.link(child)
            info["rigify"] = True
        except Exception as exc:  # noqa: BLE001
            info["rigify_error"] = str(exc)
            _log(f"Rigify failed ({exc}); falling back to a bone chain")
            rig = None
    if rig is None:
        rig = _chain_rig(main, int(P.get("chain_bones", 8)))
    rig.name = "rig"
    if rig.name not in _ensure_collection(ASSET_COLLECTION).objects:
        _move_to_collection(rig, _ensure_collection(ASSET_COLLECTION))

    _select([main, rig], rig)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    weighted = _weighted_fraction(main)
    if weighted < 0.95:
        _log(f"Automatic weights covered {weighted:.0%} of vertices; using envelope weights")
        for vg in list(main.vertex_groups):
            main.vertex_groups.remove(vg)
        _select([main, rig], rig)
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
        weighted = _weighted_fraction(main)
        info["weights"] = "envelope"
    else:
        info["weights"] = "automatic"
    info["weighted_vertex_fraction"] = round(weighted, 3)
    info["bones"] = len(rig.data.bones)
    info["deform_bones"] = sum(1 for b in rig.data.bones if b.use_deform)


def _weighted_fraction(mesh: Any) -> float:
    verts = mesh.data.vertices
    if not len(verts):
        return 0.0
    return sum(1 for v in verts if any(g.weight > 1e-4 for g in v.groups)) / len(verts)


# -- Stage 6: audit renders -----------------------------------------------------

def _engine_id(name: str) -> str:
    items = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    name = name.upper()
    if name == "EEVEE":
        return "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in items else "BLENDER_EEVEE"
    if name == "WORKBENCH":
        return "BLENDER_WORKBENCH"
    return "CYCLES"


def _simple_material(name: str, color: tuple, roughness: float = 0.6, emission: bool = False) -> Any:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    if emission:
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        em = nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*color, 1.0)
        mat.node_tree.links.new(em.outputs[0], out.inputs[0])
    else:
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _checker_material() -> Any:
    mat = bpy.data.materials.new("AUDIT_uv_checker")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coord = nodes.new("ShaderNodeTexCoord")
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 24.0
    checker.inputs["Color1"].default_value = (0.9, 0.9, 0.9, 1)
    checker.inputs["Color2"].default_value = (0.05, 0.25, 0.7, 1)
    links.new(coord.outputs["UV"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    return mat


@contextmanager
def _override_materials(objs: list[Any], mat: Optional[Any]):
    saved = {o.name: list(o.data.materials) for o in objs}
    try:
        if mat is not None:
            for o in objs:
                o.data.materials.clear()
                o.data.materials.append(mat)
        yield
    finally:
        for o in objs:
            o.data.materials.clear()
            for m in saved[o.name]:
                o.data.materials.append(m)


def _render_views(meshes: list[Any], render_dir: str, P: dict, info: dict) -> dict[str, str]:
    os.makedirs(render_dir, exist_ok=True)
    scene = bpy.context.scene
    engine = _engine_id(P.get("render_engine", "CYCLES"))
    if engine == "CYCLES":
        info["device"] = _setup_cycles(P, int(P.get("render_samples", 48)))
        scene.cycles.use_denoising = True
    scene.render.engine = engine
    res = int(P.get("render_resolution", 1024))
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass

    world = bpy.data.worlds.new("AUDIT_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.32, 0.33, 0.35, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world
    sun = bpy.data.objects.new("AUDIT_sun", bpy.data.lights.new("AUDIT_sun", "SUN"))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(10)
    sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(-35))
    scene.collection.objects.link(sun)
    fill = bpy.data.objects.new("AUDIT_fill", bpy.data.lights.new("AUDIT_fill", "SUN"))
    fill.data.energy = 1.2
    fill.rotation_euler = (math.radians(65), 0.0, math.radians(120))
    scene.collection.objects.link(fill)

    cam_data = bpy.data.cameras.new("AUDIT_cam")
    cam_data.type = "ORTHO"
    cam = bpy.data.objects.new("AUDIT_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    keep = set(meshes) | {sun, fill, cam}
    hidden = [o for o in scene.objects if o not in keep and not o.hide_render]
    for o in hidden:
        o.hide_render = True

    lo, hi = _bbox(meshes)
    center, size = (lo + hi) / 2, hi - lo
    radius = size.length / 2
    cam_data.clip_start = radius * 0.01
    cam_data.clip_end = radius * 20
    iso = Vector((1.0, -1.2, 0.8)).normalized()
    views = [
        ("front", Vector((0, -1, 0)), None),
        ("side", Vector((1, 0, 0)), None),
        ("wireframe", iso, "wire"),
        ("shaded", iso, "shaded"),
    ]
    if P.get("render_uv_checker", True):
        views.append(("uv_checker", iso, "checker"))
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]

    clay = _simple_material("AUDIT_clay", (0.62, 0.62, 0.6))
    wire_mat = _simple_material("AUDIT_wire", (0.0, 0.0, 0.0), emission=True)
    checker = _checker_material()
    paths: dict[str, str] = {}
    for view, direction, style in views:
        cam.location = center + direction * radius * 5
        rot = (-direction).to_track_quat("-Z", "Y")
        cam.rotation_euler = rot.to_euler()
        # Frame the bbox as projected onto the image plane, with a small margin.
        local = [rot.inverted() @ (c - center) for c in corners]
        extent = max(max(v.x for v in local) - min(v.x for v in local),
                     max(v.y for v in local) - min(v.y for v in local))
        cam_data.ortho_scale = extent * 1.12
        override = {None: clay, "wire": clay, "shaded": None, "checker": checker}[style]
        wires = []
        if style == "wire":
            for o in meshes:
                w = bpy.data.objects.new(f"AUDIT_wire_{o.name}", o.data.copy())
                w.matrix_world = o.matrix_world.copy()
                w.data.materials.clear()
                w.data.materials.append(wire_mat)
                mod = w.modifiers.new("wire", "WIREFRAME")
                mod.thickness = cam_data.ortho_scale / res  # ~1 px wide
                mod.offset = 1.0  # sit on top of the surface, not inside it
                mod.use_replace = True
                scene.collection.objects.link(w)
                wires.append(w)
        with _override_materials(meshes, override):
            path = os.path.join(render_dir, f"{view}.png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            paths[view] = path
        for w in wires:
            me = w.data
            bpy.data.objects.remove(w, do_unlink=True)
            bpy.data.meshes.remove(me)

    for o in hidden:
        o.hide_render = False
    for o in (sun, fill, cam):
        bpy.data.objects.remove(o, do_unlink=True)
    info["engine"] = engine
    return paths


# -- job drivers -------------------------------------------------------------------

def _process(job: dict, result: dict) -> None:
    P = job["params"]
    spec = job["spec"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    with _step(result, "import") as info:
        obj = _import_mesh(job["input_mesh"])
        rot = P.get("import_rotation_deg") or [0, 0, 0]
        if any(rot):
            obj.data.transform(_rotation_from_euler_deg(rot))
        info["faces"] = len(obj.data.polygons)
        info["color_attributes"] = [a.name for a in getattr(obj.data, "color_attributes", [])]

    with _step(result, "cleanup") as info:
        _cleanup(obj, P, info)

    with _step(result, "orient_scale") as info:
        if P.get("auto_orient", True):
            _auto_orient(obj, P["target_dimensions_m"], info)
        extra = P.get("extra_rotation_deg") or [0, 0, 0]
        if any(extra):
            obj.data.transform(_rotation_from_euler_deg(extra))
        _fit_scale(obj, P, info)
        result["fit"] = dict(info)

    with _step(result, "remesh") as info:
        src = _make_source_copy(obj, P, info)
        _remesh(obj, src, P, info)

    with _step(result, "booleans") as info:
        obj = _booleans(obj, P, info)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
        bm.free()
        _shade(obj, P)

    with _step(result, "uv_unwrap") as info:
        _unwrap(obj, P, info)

    with _step(result, "pbr_textures") as info:
        if P.get("bake", True):
            result["textures"] = _bake_textures(obj, src, spec, P, job["texture_dir"], info)
        else:
            _flat_material(obj, spec)

    with _step(result, "metrics") as info:
        result["metrics"] = _geometry_metrics(obj)
        result["uv_metrics"] = _uv_metrics(obj)

    if job.get("segment") and spec.get("moving_parts"):
        with _step(result, "segmentation") as info:
            result["parts"] = _segment(obj, spec["moving_parts"], P, info)
    if job.get("rig") and spec.get("rig_type", "none") != "none":
        with _step(result, "rigging") as info:
            _rig(obj, spec["rig_type"], P, info)
            result["rig"] = dict(info)

    asset_meshes = [o for o in _ensure_collection(ASSET_COLLECTION).objects if o.type == "MESH"]
    if job.get("render", True):
        with _step(result, "render") as info:
            src_lc = _layer_collection(SOURCE_COLLECTION)
            if src_lc:
                src_lc.exclude = True
            result["renders"] = _render_views(asset_meshes, job["render_dir"], P, info)

    with _step(result, "save") as info:
        src_lc = _layer_collection(SOURCE_COLLECTION)
        if src_lc:
            src_lc.exclude = True
        os.makedirs(os.path.dirname(job["blend_path"]), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=job["blend_path"], compress=True)
        result["blend_path"] = job["blend_path"]


def _export(job: dict, result: dict) -> None:
    bpy.ops.wm.open_mainfile(filepath=job["blend_path"])
    asset = bpy.data.collections.get(ASSET_COLLECTION)
    if asset is None:
        raise RuntimeError(f"No '{ASSET_COLLECTION}' collection in {job['blend_path']}")
    objs = [o for o in asset.objects if o.type in ("MESH", "ARMATURE", "EMPTY")]
    main = bpy.data.objects.get(MAIN_OBJECT) or objs[0]
    has_rig = any(o.type == "ARMATURE" for o in objs)
    out = job["output_path"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fmt = job["format"]
    _select(objs, main)
    if fmt == "GLB":
        bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True,
                                  export_extras=True, export_yup=True, export_def_bones=has_rig)
    elif fmt == "FBX":
        bpy.ops.export_scene.fbx(filepath=out, use_selection=True, path_mode="COPY", embed_textures=True,
                                 object_types={"MESH", "ARMATURE", "EMPTY"}, add_leaf_bones=False,
                                 use_armature_deform_only=has_rig, use_custom_props=True)
    elif fmt == "OBJ":
        kwargs = dict(filepath=out, export_selected_objects=True, export_materials=True,
                      export_uv=True, export_normals=True)
        if _op_exists(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(path_mode="COPY", **kwargs)
        else:
            bpy.ops.export_scene.obj(filepath=out, use_selection=True, path_mode="COPY")
    else:
        raise ValueError(f"Unsupported export format {fmt}")
    result["output_path"] = out

    if job.get("export_parts"):
        part_dir = os.path.join(os.path.dirname(out), "parts_stl_mm")
        os.makedirs(part_dir, exist_ok=True)
        files = []
        for o in [o for o in objs if o.type == "MESH"]:
            path = os.path.join(part_dir, f"{o.name}.stl")
            _select([o])
            # Slicers assume millimetres; the scene is in metres.
            if _op_exists(bpy.ops.wm, "stl_export"):
                bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, global_scale=1000.0)
            else:
                bpy.ops.export_mesh.stl(filepath=path, use_selection=True, global_scale=1000.0)
            files.append(path)
        result["part_files"] = files


def blender_main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--job" not in argv:
        raise SystemExit("usage: blender -b -P blender_automation.py -- --job job.json")
    job = json.loads(Path(argv[argv.index("--job") + 1]).read_text(encoding="utf-8"))
    result: dict[str, Any] = {"ok": False, "blender_version": bpy.app.version_string, "steps": {}}
    try:
        if job["action"] == "process":
            _process(job, result)
        elif job["action"] == "export":
            _export(job, result)
        else:
            raise ValueError(f"Unknown action {job['action']}")
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported back to the host
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        Path(job["result_path"]).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if not result["ok"]:
        sys.exit(3)


if __name__ == "__main__" and IN_BLENDER:
    blender_main()

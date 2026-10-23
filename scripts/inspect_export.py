"""Inspect an exported asset the way a game engine would see it (run inside Blender).

  blender --background --factory-startup --python scripts/inspect_export.py -- path/to/asset.glb [--json out.json]

Prints objects, hierarchy, pivots, dimensions (m), face/triangle counts, UV maps,
materials and their texture images, armature bone counts, skin-weight sanity
(unweighted vertices, max influences) and custom properties (part motion limits),
plus a list of warnings.
"""

import json
import sys
from pathlib import Path

import bpy


def main(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    path = Path(argv[0]).resolve()
    out_json = argv[argv.index("--json") + 1] if "--json" in argv else None
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise SystemExit(f"unsupported: {ext}")

    report = {"file": str(path), "bytes": path.stat().st_size, "objects": [], "warnings": [],
              "note": "glTF stores triangles, so 'faces' of a GLB equals 'triangles' (about 2x the quad count)"}
    warn = report["warnings"].append
    # Importers add display shapes for bones (glTF: an 'Icosphere'); they are not part of the asset
    bone_shapes = {pb.custom_shape for o in bpy.context.scene.objects if o.type == "ARMATURE"
                   for pb in o.pose.bones if pb.custom_shape}
    objects = [o for o in bpy.context.scene.objects if o not in bone_shapes]
    meshes = [o for o in objects if o.type == "MESH"]
    for o in objects:
        info = {"name": o.name, "type": o.type, "parent": o.parent.name if o.parent else None,
                "location_m": [round(v, 4) for v in o.matrix_world.translation],
                "custom_props": {k: (v if isinstance(v, (int, float, str)) else str(v))
                                 for k, v in o.items() if not k.startswith("_")}}
        if o.type == "MESH":
            me = o.data
            me.calc_loop_triangles()
            ws = [o.matrix_world @ v.co for v in me.vertices]
            dims = [round(max(c[i] for c in ws) - min(c[i] for c in ws), 4) for i in range(3)] if ws else [0, 0, 0]
            mats = []
            for slot in o.material_slots:
                m = slot.material
                imgs = sorted({n.image.name + f" ({n.image.size[0]}x{n.image.size[1]})"
                               for n in (m.node_tree.nodes if m and m.use_nodes else []) if n.type == "TEX_IMAGE" and n.image})
                mats.append({"material": m.name if m else None, "textures": imgs})
            info.update(dimensions_m=dims, vertices=len(me.vertices), faces=len(me.polygons),
                        triangles=len(me.loop_triangles), uv_maps=[uv.name for uv in me.uv_layers], materials=mats)
            if not me.uv_layers:
                warn(f"{o.name}: no UV map")
            if not any(mm["textures"] for mm in mats):
                warn(f"{o.name}: no texture images on its materials")
            if o.vertex_groups:
                counts = [sum(1 for g in v.groups if g.weight > 1e-4) for v in me.vertices]
                info["skin"] = {"groups": len(o.vertex_groups), "max_influences": max(counts),
                                "unweighted_vertices": sum(1 for c in counts if c == 0)}
                if max(counts) > 4:
                    warn(f"{o.name}: {max(counts)} bone influences on some vertices (engines usually allow 4)")
                if info["skin"]["unweighted_vertices"]:
                    warn(f"{o.name}: {info['skin']['unweighted_vertices']} vertices without skin weights")
        elif o.type == "ARMATURE":
            info["bones"] = len(o.data.bones)
            info["deform_bones"] = sum(1 for b in o.data.bones if b.use_deform)
            info["root_bones"] = [b.name for b in o.data.bones if b.parent is None]
        report["objects"].append(info)
    if meshes:
        pts = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
        report["assembly_dimensions_m"] = [round(max(c[i] for c in pts) - min(c[i] for c in pts), 4) for i in range(3)]
        report["lowest_point_z"] = round(min(c[2] for c in pts), 4)
        if abs(report["lowest_point_z"]) > 0.01:
            warn(f"asset does not rest on the ground (lowest z = {report['lowest_point_z']} m)")
    else:
        warn("no meshes in file")
    text = json.dumps(report, indent=2)
    print(text)
    if out_json:
        Path(out_json).write_text(text, encoding="utf-8")


main(sys.argv)

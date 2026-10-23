"""Procedural mannequin (overlapping capsules) for rigging tests: 1.8 m, faces -Y.

``pose="A"`` arms 45 degrees down, ``pose="T"`` arms horizontal, ``pose="I"`` arms
hanging against the body. Written as a
Y-up OBJ (like real reconstruction exports); Stage 3 voxel-remeshes it into one
watertight body.
"""

import math
import sys

import numpy as np


def capsule(a, b, r, segments=16, rings=8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    length = np.linalg.norm(axis)
    z = axis / length
    x = np.cross(z, [0, 0, 1] if abs(z[2]) < 0.9 else [1, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    verts, faces = [], []
    profile = []  # (height along axis, radius)
    for i in range(rings + 1):  # bottom hemisphere
        t = -math.pi / 2 + (math.pi / 2) * i / rings
        profile.append((r * math.sin(t), r * math.cos(t)))
    for i in range(rings + 1):  # top hemisphere
        t = (math.pi / 2) * i / rings
        profile.append((length + r * math.sin(t), r * math.cos(t)))
    for h, rad in profile:
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            verts.append(a + z * h + rad * (math.cos(phi) * x + math.sin(phi) * y))
    n = len(profile)
    for i in range(n - 1):
        for j in range(segments):
            p0, p1 = i * segments + j, i * segments + (j + 1) % segments
            faces.append((p0, p1, p1 + segments, p0 + segments))
    return np.array(verts), faces


def mannequin(pose="A"):
    parts = [
        ((0, 0, 0.95), (0, 0, 1.45), 0.16),        # torso
        ((0, 0, 1.47), (0, 0, 1.55), 0.06),        # neck
        ((0, 0, 1.62), (0, 0, 1.70), 0.10),        # head
    ]
    # A: arms 45 deg down, T: horizontal, I: hanging against the body (arms touch the torso)
    drop = math.radians({"A": 45, "T": 0, "I": 84}[pose])
    for s in (1, -1):
        sh = np.array([s * (0.21 if pose == "I" else 0.19), 0, 1.42])
        d = np.array([s * math.cos(drop), 0, -math.sin(drop)])
        el = sh + d * 0.30
        wr = el + d * 0.26
        parts += [(sh, el, 0.055), (el, wr, 0.045), (wr, wr + d * 0.12, 0.04)]
        hip = np.array([s * 0.10, 0, 0.92])
        kn = np.array([s * 0.11, -0.01, 0.50])
        an = np.array([s * 0.11, 0, 0.09])
        parts += [(hip, kn, 0.075), (kn, an, 0.055), (an + [0, 0.02, -0.03], an + [0, -0.17, -0.04], 0.045)]
    verts, faces = [], []
    for a, b, r in parts:
        v, f = capsule(a, b, r)
        base = sum(len(x) for x in verts)
        verts.append(v)
        faces += [tuple(i + base for i in face) for face in f]
    return np.vstack(verts), faces


def write_obj(path, pose="A"):
    v, f = mannequin(pose)
    with open(path, "w", encoding="utf-8") as fh:
        for x, y, z in v:
            fh.write(f"v {x:.5f} {z:.5f} {-y:.5f}\n")  # Z-up -> OBJ Y-up
        for face in f:
            fh.write("f " + " ".join(str(i + 1) for i in face) + "\n")
    return path


if __name__ == "__main__":
    write_obj(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "A")

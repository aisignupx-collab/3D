"""Synthetic four-tower castle (overlapping shells) for skyline tests: ~30 x 26 x 26 m, Z-up, Y-up OBJ.

Two round towers carry cone roofs, one round tower and one square tower are flat-topped, and
four curtain walls enclose an open courtyard. Stage 3 voxel-remeshes and smooths it into one
lumpy shell, like a feed-forward reconstruction.
"""

import math
import sys

SEG = 24


def shapes():
    out = []

    def box(lo, hi):
        out.append([(lo[0] if not i & 4 else hi[0], lo[1] if not i & 2 else hi[1], lo[2] if not i & 1 else hi[2])
                    for i in range(8)])

    def frustum(cx, cy, r0, r1, z0, z1):
        ring0 = [(cx + r0 * math.cos(2 * math.pi * k / SEG), cy + r0 * math.sin(2 * math.pi * k / SEG), z0)
                 for k in range(SEG)]
        ring1 = [(cx + max(r1, 1e-3) * math.cos(2 * math.pi * k / SEG),
                  cy + max(r1, 1e-3) * math.sin(2 * math.pi * k / SEG), z1) for k in range(SEG)]
        out.append(ring0 + ring1)

    frustum(-9, -6, 4.5, 4.5, 0, 24)            # A: tall round tower, flat top
    frustum(9, -6, 4.0, 4.0, 0, 18)             # B: round tower ...
    frustum(9, -6, 4.6, 0.0, 18, 27)            # ... with a cone roof
    frustum(-9, 8, 3.5, 3.5, 0, 15)             # C: round tower ...
    frustum(-9, 8, 4.0, 0.0, 15, 22)            # ... with a cone roof
    box((5.5, 4.5, 0), (12.5, 11.5, 20))        # D: square tower, flat top
    box((-9, -8, 0), (9, -4, 11))               # curtain walls around an open courtyard
    box((-9, 6, 0), (9, 10, 11))
    box((-11, -6, 0), (-7, 8, 11))
    box((7, -6, 0), (11, 8, 11))
    return out


def write_obj(path):
    lines, base = [], 0
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    for verts in shapes():
        lines += [f"v {x:.4f} {z:.4f} {-y:.4f}" for x, y, z in verts]
        if len(verts) == 8:
            faces = quads
        else:
            n = len(verts) // 2
            faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
            faces += [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
        lines += ["f " + " ".join(str(base + i + 1) for i in f) for f in faces]
        base += len(verts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    write_obj(sys.argv[1])

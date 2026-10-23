"""Blocky synthetic SMG (overlapping boxes) for blueprint tests: 0.68 m long, Z-up, Y-up OBJ."""

import sys


def boxes():
    return [  # (min, max) in metres
        ((0.00, -0.025, 0.150), (0.50, 0.025, 0.230)),   # receiver
        ((0.50, -0.012, 0.175), (0.68, 0.012, 0.205)),   # barrel
        ((-0.15, -0.020, 0.120), (0.00, 0.020, 0.220)),  # stock
        ((0.12, -0.018, 0.000), (0.19, 0.018, 0.155)),   # grip
        ((0.28, -0.012, 0.020), (0.34, 0.012, 0.155)),   # magazine
        ((0.19, -0.006, 0.090), (0.27, 0.006, 0.100)),   # trigger-guard bottom (leaves an opening)
    ]


def write_obj(path):
    lines, base = [], 0
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    for lo, hi in boxes():
        for i in range(8):
            x = hi[0] if i & 4 else lo[0]
            y = hi[1] if i & 2 else lo[1]
            z = hi[2] if i & 1 else lo[2]
            lines.append(f"v {x:.5f} {z:.5f} {-y:.5f}")
        lines += ["f " + " ".join(str(base + i + 1) for i in q) for q in quads]
        base += 8
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    write_obj(sys.argv[1])

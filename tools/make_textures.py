import math
import os
import pickle
import struct
import subprocess
import sys
import time

import cv2
import numpy as np

TEXCONV = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\texconv.exe")
NAME = "HollowLantern"
SETS = (("8K", 0), ("4K", 1), ("2K", 2))
F = np.float32
MARGIN = 12
AO_FACTOR = 4
LANTERN_R = 2.75
LANTERN_H = 2.2
LANTERN_LOBES = 9
WRAPPED = ("hornL", "hornR", "tail", "lantern")
THREAD = (196, 92, 30)
t0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - t0, msg), flush=True)


def rgb(c):
    return np.array(c, dtype=F) / 255.0


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Ctx:
    def __init__(self, U, V, mask, dist, ao, seed, scale, key, period):
        self.h, self.w = mask.shape
        self.U, self.V = U, V
        self.X, self.Y = U * F(scale), V * F(scale)
        self.mask, self.dist, self.ao = mask, dist, ao
        self.seed, self.scale, self.key = seed, scale, key
        self.period = period


def value_noise(c, rng, cx, cy):
    if c.period:
        gw = max(3, int(round(c.period * c.scale / cx)))
        gx = c.X * F(gw / (c.period * c.scale))
    else:
        gx = (c.X - c.X.min()) / F(cx)
        gw = int(gx.max()) + 2
    gy = (c.Y - c.Y.min()) / F(cy)
    gh = int(gy.max()) + 2
    g = rng.standard_normal((gh + 1, gw + 1)).astype(F)
    xi = np.floor(gx)
    fx = gx - xi
    xi = xi.astype(np.int64)
    yi = np.floor(gy)
    fy = gy - yi
    yi = yi.astype(np.int64)
    if c.period:
        x0, x1 = xi % gw, (xi + 1) % gw
    else:
        x0, x1 = xi, xi + 1
    fx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    fy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    a = g[yi, x0] + (g[yi, x1] - g[yi, x0]) * fx
    b = g[yi + 1, x0] + (g[yi + 1, x1] - g[yi + 1, x0]) * fx
    return a + (b - a) * fy


def noise(c, seed, cell, octaves=4, gain=0.5, su=1.0, sv=1.0):
    rng = np.random.default_rng(seed)
    out = np.zeros((c.h, c.w), F)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        out += F(amp) * value_noise(c, rng, max(cell * su / 2 ** o, 1.0), max(cell * sv / 2 ** o, 1.0))
        total += amp
        amp *= gain
    return out * F(1.4 / total)


def scatter(c, cell, seed):
    rng = np.random.default_rng(seed)
    img = np.ones((c.h, c.w), np.uint8)
    if c.period:
        span = c.period * c.scale
        n = max(1, int(span * c.h / (cell * cell)))
        us = rng.uniform(0, span, n)
        rows = rng.integers(0, c.h, n)
        col0 = float(c.X[0, 0])
        for k in range(int(math.floor(col0 / span)) - 1, int(math.ceil((col0 + c.w) / span)) + 1):
            cols = np.round(us + k * span - col0).astype(np.int64)
            ok = (cols >= 0) & (cols < c.w)
            img[rows[ok], cols[ok]] = 0
    else:
        n = max(1, int(c.h * c.w / (cell * cell)))
        img[rng.integers(0, c.h, n), rng.integers(0, c.w, n)] = 0
    return img


def pebbles(c, cell, seed):
    _, lab = cv2.distanceTransformWithLabels(scatter(c, cell, seed), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    edge = np.ones((c.h, c.w), np.uint8)
    edge[:, 1:][lab[:, 1:] != lab[:, :-1]] = 0
    edge[1:, :][lab[1:, :] != lab[:-1, :]] = 0
    d = cv2.distanceTransform(edge, cv2.DIST_L2, 3)
    top = np.clip(d / (cell * 0.42), 0, 1)
    return 1.0 - (1.0 - top) ** 2


def leather(c):
    grain = 0.6 * pebbles(c, 0.1 * c.scale, c.seed) + 0.4 * pebbles(c, 0.2 * c.scale, c.seed + 1)
    mott = noise(c, c.seed + 2, 1.6 * c.scale)
    warm = noise(c, c.seed + 3, 1.2 * c.scale)
    color = rgb((37, 31, 27))[None, None] * (1 + 0.08 * mott[..., None]) * (0.84 + 0.24 * grain[..., None])
    color = color + rgb((6, 3, 1))[None, None] * np.clip(warm, 0, None)[..., None]
    height = 0.007 * grain
    spec = (0.26 + 0.14 * grain) * (1 + 0.1 * mott)
    return color, height, spec


def fabric(c):
    twill = np.sin((c.X + c.Y) * F(2 * np.pi / (0.1 * c.scale)))
    fuzz = noise(c, c.seed, 0.035 * c.scale, octaves=2)
    mott = noise(c, c.seed + 1, 2.2 * c.scale)
    tone = F(1 + 0.03 * np.random.default_rng(c.seed).standard_normal())
    color = rgb((166, 67, 19))[None, None] * tone * (1 + 0.05 * mott[..., None] + 0.05 * fuzz[..., None]
                                                     + 0.025 * twill[..., None])
    height = 0.0016 * twill + 0.0012 * fuzz
    spec = 0.05 + 0.012 * fuzz
    return color, height, spec


def pumpkin(c, lantern):
    streak = noise(c, c.seed, 0.5 * c.scale, octaves=3, su=0.15, sv=1.6)
    mott = noise(c, c.seed + 1, 1.0 * c.scale)
    dots = smoothstep(0.035 * c.scale, 0.0, cv2.distanceTransform(scatter(c, 0.45 * c.scale, c.seed + 2),
                                                                   cv2.DIST_L2, 3))
    color = rgb((204, 94, 20))[None, None] * (1 + 0.06 * mott[..., None] + 0.06 * streak[..., None])
    color = color * (1 - dots[..., None] * 0.16) + rgb((236, 168, 72))[None, None] * dots[..., None] * 0.16
    height = 0.004 * streak + 0.002 * mott
    spec = 0.3 + 0.04 * streak
    if lantern:
        theta = c.U / F(LANTERN_R) + F(math.pi)
        rib = np.cos(LANTERN_LOBES * theta / 2)
        groove = np.exp(-(rib / 0.28) ** 2)
        crest = np.abs(rib) ** 6
        color = color * (1 - 0.3 * groove[..., None] + 0.06 * crest[..., None])
        color = color + rgb((60, 12, 0))[None, None] * 0.25 * groove[..., None]
        top = smoothstep(0.55, 0.15, c.V)[..., None]
        color = color * (1 - 0.55 * top) + rgb((96, 80, 34))[None, None] * 0.55 * top
        bottom = smoothstep(math.pi * LANTERN_H - 0.5, math.pi * LANTERN_H - 0.1, c.V)[..., None]
        color = color * (1 - 0.5 * bottom) + rgb((92, 60, 28))[None, None] * 0.5 * bottom
        period = 2 * math.pi * LANTERN_R / 18
        tri = 2 * np.abs(2 * (c.U / F(period) - np.floor(c.U / F(period) + 0.5))) - 1
        off = c.V - (0.55 * LANTERN_H + 0.13 * tri)
        cut = smoothstep(0.022, 0.01, np.abs(off))
        lip = smoothstep(0.045, 0.022, off) * smoothstep(0.0, 0.022, off)
        color = color * (1 - 0.9 * cut[..., None])
        color = color * (1 - 0.5 * lip[..., None]) + rgb((232, 172, 92))[None, None] * 0.5 * lip[..., None]
        height = height - 0.03 * cut
        spec = spec * (1 - cut)
    return color, height, spec


def horn(c):
    length = float(c.V[c.mask > 0].max())
    k = smoothstep(0.5, 1.0, np.clip(c.V / F(length), 0, 1))
    streak = noise(c, c.seed, 1.2 * c.scale, octaves=3, su=0.06, sv=1.0)
    warp = noise(c, c.seed + 1, 1.5 * c.scale, octaves=2)
    fade = np.clip(0.6 + 0.5 * noise(c, c.seed + 2, 0.9 * c.scale, octaves=2), 0, 1)
    rings = (0.5 + 0.5 * np.cos(2 * np.pi * (c.V + 0.15 * warp) / 0.55)) ** 12 * fade * (1 - k)
    base = rgb((42, 33, 29))[None, None]
    tip = rgb((118, 44, 14))[None, None]
    color = base * (1 - k[..., None]) + tip * k[..., None]
    color = color * (1 + 0.1 * streak[..., None]) * (1 - 0.14 * rings[..., None])
    height = 0.004 * streak - 0.003 * rings
    spec = 0.38 + 0.06 * streak
    return color, height, spec


def bezier(pts, n=24):
    p = np.array(pts, dtype=np.float64)
    s = np.linspace(0, 1, n)[:, None]
    if len(p) == 3:
        return (1 - s) ** 2 * p[0] + 2 * (1 - s) * s * p[1] + s ** 2 * p[2]
    return (1 - s) ** 3 * p[0] + 3 * (1 - s) ** 2 * s * p[1] + 3 * (1 - s) * s ** 2 * p[2] + s ** 3 * p[3]


def stroke(x, z, pts, w0, w1, aa=0.025):
    p = bezier(pts)
    best = np.full(x.shape, np.inf, F)
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (z - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        d = np.hypot(x - (a[0] + t * ab[0]), z - (a[1] + t * ab[1]))
        s = (i + t) / (len(p) - 1)
        best = np.minimum(best, (d - (w0 + (w1 - w0) * s)).astype(F))
    return smoothstep(aa, -aa, best)


def leaf(x, z, base, tip, width, aa=0.02):
    b = np.array(base, dtype=np.float64)
    e = np.array(tip, dtype=np.float64) - b
    L = np.hypot(*e)
    u = ((x - b[0]) * e[0] + (z - b[1]) * e[1]) / L
    w = np.abs(-(x - b[0]) * e[1] + (z - b[1]) * e[0]) / L
    s = np.clip(u / L, 0, 1)
    lobe = width * np.sin(np.pi * s) ** 0.75 * (1 + 0.18 * np.cos(3 * np.pi * s))
    inside = np.where((u > 0) & (u < L), lobe - w, -1.0)
    return smoothstep(-aa, aa, inside)


def polyline(x, z, p, w0, w1, aa=0.02):
    best = np.full(x.shape, np.inf, F)
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (z - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
        d = np.hypot(x - (a[0] + t * ab[0]), z - (a[1] + t * ab[1]))
        s = (i + t) / (len(p) - 1)
        best = np.minimum(best, (d - (w0 + (w1 - w0) * s)).astype(F))
    return smoothstep(aa, -aa, best)


def spiral(center, r0, turns, start, sign, n=48):
    a = start + sign * np.linspace(0, 2 * np.pi * turns, n)
    r = r0 * np.linspace(1, 0.18, n)
    return np.stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)], 1)


def to_arc(pts, table):
    pts = np.array(pts, dtype=np.float64)
    if table is None:
        return pts
    zs, xs, S = table
    rows = np.clip(np.round((pts[:, 1] - zs[0]) / (zs[1] - zs[0])).astype(int), 0, len(zs) - 1)
    pts[:, 0] = [np.interp(px, xs, S[r]) for px, r in zip(pts[:, 0], rows)]
    return pts


def mask_inlay(x, z, table=None):
    ax = np.abs(x)

    def curve(ctrl):
        return to_arc(bezier(ctrl), table)

    def coil(*a, **k):
        return to_arc(spiral(*a, **k), table)

    def tip(p):
        return tuple(to_arc([p], table)[0])
    marks = [
        polyline(ax, z, curve([(0.44, 125.08), (1.2, 125.72), (2.3, 125.52), (3.2, 125.95)]), 0.05, 0.042),
        polyline(ax, z, curve([(3.2, 125.95), (4.0, 126.35), (4.72, 126.95), (4.8, 127.5)]), 0.042, 0.03),
        polyline(ax, z, coil((4.55, 127.48), 0.27, 1.6, 0.0, 1), 0.03, 0.012),
        polyline(ax, z, curve([(4.45, 126.6), (4.98, 125.65), (5.05, 124.45), (4.68, 123.5)]), 0.038, 0.028),
        polyline(ax, z, coil((4.42, 123.18), 0.27, 1.5, 1.2, -1), 0.028, 0.012),
        polyline(ax, z, curve([(4.5, 123.0), (3.7, 122.32), (2.85, 122.14), (2.15, 122.34)]), 0.032, 0.016),
        polyline(ax, z, coil((2.02, 122.48), 0.15, 1.3, -0.6, 1), 0.016, 0.008),
        polyline(ax, z, curve([(1.62, 125.74), (1.62, 126.0), (1.5, 126.12), (1.42, 126.1)]), 0.022, 0.016),
        polyline(ax, z, coil((1.47, 126.0), 0.11, 1.4, 1.6, 1), 0.016, 0.008),
        polyline(ax, z, curve([(3.55, 126.1), (3.62, 125.82), (3.72, 125.7), (3.82, 125.72)]), 0.022, 0.016),
        polyline(ax, z, coil((3.8, 125.6), 0.1, 1.4, 1.6, -1), 0.016, 0.008),
        leaf(ax, z, tip((2.3, 125.56)), tip((2.62, 126.18)), 0.16),
        leaf(ax, z, tip((4.05, 126.42)), tip((4.42, 126.92)), 0.13),
        leaf(ax, z, tip((5.02, 124.95)), tip((5.3, 125.42)), 0.11),
        leaf(ax, z, tip((3.25, 122.2)), tip((3.6, 121.98)), 0.09),
        leaf(x, z, (0.0, 125.42), (0.0, 126.5), 0.2),
        polyline(x, z, curve([(0.0, 124.46), (0.0, 124.0), (0.0, 123.55), (0.0, 123.1)]), 0.045, 0.02),
    ]
    vein = [polyline(ax, z, to_arc([(2.32, 125.6), (2.58, 126.12)], table), 0.01, 0.006),
            polyline(ax, z, to_arc([(4.07, 126.45), (4.38, 126.87)], table), 0.009, 0.005),
            leaf(x, z, (0.0, 125.62), (0.0, 126.2), 0.08)]
    out = np.clip(np.max(marks, axis=0), 0, 1) * (1 - 0.75 * np.clip(np.max(vein, axis=0), 0, 1))
    return out.astype(F)


def mask_leather(c):
    import piece_mask as pm
    color, height, spec = leather(c)
    x = c.U.astype(np.float64) - pm.UV_X
    z = pm.UV_Z - c.V.astype(np.float64)
    inlay = mask_inlay(x, z, MASK_ARC)
    edge = smoothstep(0.0, 0.5, inlay) * smoothstep(1.0, 0.5, inlay)
    mott = noise(c, c.seed + 9, 0.5 * c.scale, octaves=2)
    lac = rgb((206, 94, 24))[None, None] * (1 + 0.06 * mott[..., None])
    color = color * (1 - inlay[..., None]) + lac * inlay[..., None]
    color = color * (1 - 0.35 * edge[..., None])
    height = height * (1 - inlay) - 0.004 * inlay
    spec = spec * (1 - inlay) + 0.46 * inlay
    c.glow = inlay[..., None] * rgb((150, 62, 14))[None, None]
    return color, height, spec


def blade_coords(c):
    import piece_axe as pa
    x = c.U.astype(np.float64) / pa.UV_SCALE - pa.UV_X
    y = pa.UV_Y - c.V.astype(np.float64) / pa.UV_SCALE
    p = np.stack([x.ravel(), y.ravel()], 1)
    edge = pa.polyline_dist(p, pa.edge_line()).reshape(x.shape).astype(F)
    return pa, x, y, p, edge


def blade_inlay(pa, x, y, edge):
    marks = [
        stroke(x, y, [(-1.9, 36.9), (-4.2, 37.5), (-7.6, 38.6), (-10.6, 41.3)], 0.07, 0.045),
        polyline(x, y, spiral((-11.15, 42.15), 0.55, 1.5, -0.6, -1), 0.045, 0.02),
        stroke(x, y, [(-1.9, 28.5), (-4.3, 27.7), (-7.2, 25.0), (-9.3, 21.4)], 0.07, 0.045),
        polyline(x, y, spiral((-9.85, 20.55), 0.5, 1.5, 0.6, 1), 0.045, 0.02),
        leaf(x, y, (-6.0, 38.1), (-6.4, 39.0), 0.24),
        leaf(x, y, (-8.8, 39.5), (-8.7, 40.5), 0.22),
        leaf(x, y, (-5.8, 26.4), (-6.5, 25.6), 0.22),
        leaf(x, y, (-8.3, 23.4), (-9.1, 23.1), 0.2),
        smoothstep(0.03, 0.0, np.abs(edge - pa.EDGE_BAND - 0.38) - 0.035),
    ]
    vein = [polyline(x, y, np.array([(-6.02, 38.16), (-6.36, 38.92)]), 0.016, 0.01),
            polyline(x, y, np.array([(-8.8, 39.56), (-8.71, 40.42)]), 0.015, 0.01),
            polyline(x, y, np.array([(-5.84, 26.36), (-6.44, 25.66)]), 0.015, 0.01),
            polyline(x, y, np.array([(-8.34, 23.38), (-9.04, 23.12)]), 0.014, 0.01)]
    return (np.clip(np.max(marks, axis=0), 0, 1) * (1 - 0.75 * np.clip(np.max(vein, axis=0), 0, 1))).astype(F)


def iron(c):
    pa, x, y, p, edge = blade_coords(c)
    win = pa.window_field(p).reshape(x.shape).astype(F)
    hammer = pebbles(c, 0.5 * c.scale * pa.UV_SCALE, c.seed)
    mott = noise(c, c.seed + 1, 1.2 * c.scale * pa.UV_SCALE)
    grind = noise(c, c.seed + 2, 0.25 * c.scale * pa.UV_SCALE, octaves=2, su=1.6, sv=0.08)
    bevel = smoothstep(pa.EDGE_BAND + 0.9, pa.EDGE_BAND + 0.05, edge)
    forged = rgb((25, 25, 28))[None, None] * (0.9 + 0.14 * hammer[..., None]) * (1 + 0.08 * mott[..., None])
    polished = rgb((70, 72, 78))[None, None] * (1 + 0.08 * grind[..., None])
    color = forged * (1 - bevel[..., None]) + polished * bevel[..., None]
    temper = smoothstep(pa.EDGE_BAND + 0.45, pa.EDGE_BAND, edge)
    color = color * (1 - 0.5 * temper[..., None]) + rgb((96, 58, 26))[None, None] * 0.5 * temper[..., None]
    heat = smoothstep(-1.1, -0.05, win)
    color = color * (1 - 0.55 * heat[..., None]) + rgb((118, 50, 16))[None, None] * 0.55 * heat[..., None]
    inlay = blade_inlay(pa, x, y, edge)
    lac = rgb((212, 96, 22))[None, None] * (1 + 0.06 * mott[..., None])
    color = color * (1 - inlay[..., None]) + lac * inlay[..., None]
    height = 0.004 * hammer * (1 - bevel) + 0.002 * grind * bevel - 0.004 * inlay
    spec = (0.3 + 0.05 * hammer + 0.4 * bevel) * (1 - inlay) + 0.12 * inlay
    return color, height, spec


def ember(c):
    pa, x, y, p, edge = blade_coords(c)
    s = np.clip(edge / pa.EDGE_BAND, 0, 1)
    flick = noise(c, c.seed, 0.35 * c.scale * pa.UV_SCALE, octaves=3)
    grind = noise(c, c.seed + 2, 0.25 * c.scale * pa.UV_SCALE, octaves=2, su=1.6, sv=0.08)
    hot = (1 - s) ** 1.5 * (0.9 + 0.1 * flick)
    dark = rgb((34, 22, 18))[None, None]
    fire = rgb((236, 104, 22))[None, None]
    color = dark * (1 - hot[..., None]) + fire * hot[..., None]
    color = color + rgb((255, 214, 130))[None, None] * ((1 - s) ** 7)[..., None] * 0.55
    height = 0.002 * grind
    spec = 0.55 + 0.1 * grind
    c.glow = (rgb((255, 118, 28))[None, None] * (hot ** 1.25)[..., None]).astype(F)
    return color, height, spec


MATERIALS = {
    "iron": iron,
    "ember": ember,
    "leather": leather,
    "fabric": fabric,
    "pumpkin": lambda c: pumpkin(c, c.key == "lantern"),
    "horn": horn,
    "maskleather": mask_leather,
}

PATCHES = {
    "patch:lining": ((60, 42, 31), 0.03, 0.06),
    "patch:edge": ((44, 31, 24), 0.34, 0.05),
    "patch:brass": ((150, 112, 56), 0.75, 0.1),
    "patch:cord": ((196, 90, 26), 0.18, 0.05),
    "patch:sole": ((25, 22, 20), 0.2, 0.05),
    "patch:wax": ((226, 206, 168), 0.25, 0.03),
    "patch:flesh": ((238, 148, 48), 0.18, 0.06),
    "patch:stem": ((94, 80, 48), 0.08, 0.1),
    "patch:steel": ((42, 42, 47), 0.45, 0.1),
    "patch:ebony": ((26, 21, 19), 0.24, 0.12),
    "patch:ember": ((236, 112, 30), 0.4, 0.06),
}
MASK_ARC = None
GLOW = {"patch:flesh": (255, 152, 60), "patch:wax": (130, 96, 54), "patch:ember": (255, 118, 28)}
ENV = {"patch:brass": 0.62, "patch:steel": 0.5}
ENV_MATS = {"iron": 0.85}


def stitch_rows(key):
    if key.startswith("corset"):
        return [0.16, 0.7]
    if key.startswith(("briefs", "boot")):
        return [0.17]
    if key.startswith(("band", "hem")):
        return [0.15]
    if key in ("belt", "choker", "tail", "spade") or key.endswith("_sleeve"):
        return [0.12]
    if key == "mask":
        return [0.13]
    return []


def has_seam(key, mat):
    return mat in ("leather", "fabric", "maskleather") and not key.endswith("_hand")


def stitches(mask, scale, rows, length=0.2, gap=0.12, width=0.045, hole=0.032):
    h, w = mask.shape
    thread = np.zeros((h, w), np.uint8)
    holes = np.zeros((h, w), np.uint8)
    dent = np.zeros((h, w), np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    tw = max(1, int(round(width * scale)))
    hr = max(1, int(round(hole * scale * 4)))
    for d in rows:
        iso = (dist >= d * scale).astype(np.uint8)
        contours, _ = cv2.findContours(iso, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            pts = cnt[:, 0, :].astype(np.float64)
            if len(pts) < 12:
                continue
            pts = np.vstack([pts, pts[:1]])
            acc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            total = acc[-1]
            if total < 1.5 * scale:
                continue
            cv2.polylines(dent, [np.round(pts * 4).astype(np.int32)], False, 255, max(1, int(0.07 * scale)),
                          cv2.LINE_AA, shift=2)
            n = int(total / ((length + gap) * scale))
            if n < 2:
                continue
            pitch = total / n
            s = np.arange(n) * pitch
            a = np.stack([np.interp(s, acc, pts[:, k]) for k in range(2)], 1)
            b = np.stack([np.interp(s + length / (length + gap) * pitch, acc, pts[:, k]) for k in range(2)], 1)
            ia = np.round(a * 4).astype(np.int32)
            ib = np.round(b * 4).astype(np.int32)
            for p, q in zip(ia, ib):
                cv2.circle(holes, (int(p[0]), int(p[1])), hr, 255, -1, cv2.LINE_AA, shift=2)
                cv2.circle(holes, (int(q[0]), int(q[1])), hr, 255, -1, cv2.LINE_AA, shift=2)
            for p, q in zip(ia, ib):
                cv2.line(thread, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 255, tw, cv2.LINE_AA, shift=2)
    return thread.astype(F) / 255, holes.astype(F) / 255, dent.astype(F) / 255


def normals(height, scale):
    hp = np.pad(height, 1, mode="edge")
    nx = (hp[1:-1, :-2] - hp[1:-1, 2:]) * F(scale * 0.5)
    ny = (hp[:-2, 1:-1] - hp[2:, 1:-1]) * F(scale * 0.5)
    inv = 1 / np.sqrt(nx * nx + ny * ny + 1)
    return np.stack([nx * inv, ny * inv, inv], -1)


def to_u8(x):
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


def rasterize_ids(parts, keys, size):
    ids = np.full((size, size), 255, np.uint8)
    index = {k: i for i, k in enumerate(keys)}
    for p in parts:
        if p["island"] not in index:
            continue
        val = index[p["island"]]
        for tri in np.round(p["uv"][p["t"]] * 4).astype(np.int32):
            cv2.fillConvexPoly(ids, tri, val, cv2.LINE_8, 2)
    return ids


def ao_map(parts, size):
    s = size // AO_FACTOR
    acc = np.zeros((s, s), F)
    cnt = np.zeros((s, s), F)
    for p in parts:
        ao = p["attrs"].get("ao")
        if ao is None:
            continue
        tris = np.round(p["uv"][p["t"]] / AO_FACTOR * 4).astype(np.int32)
        vals = ao[p["t"]].mean(1)
        tmp = np.zeros((s, s), F)
        hit = np.zeros((s, s), F)
        for tri, val in zip(tris, vals):
            cv2.fillConvexPoly(tmp, tri, float(val), cv2.LINE_8, 2)
            cv2.fillConvexPoly(hit, tri, 1.0, cv2.LINE_8, 2)
        acc += tmp * hit
        cnt += hit
    known = (cnt > 0).astype(F)
    acc = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(F)
    wgt = cv2.GaussianBlur(known, (0, 0), 1.5)
    out = cv2.GaussianBlur(acc * known, (0, 0), 1.5) / np.maximum(wgt, 1e-4)
    have = (wgt >= 1e-3).astype(np.uint8)
    out[have == 0] = 0
    k = np.ones((3, 3), np.uint8)
    for _ in range(6):
        grown = cv2.dilate(have, k)
        ring = (grown > 0) & (have == 0)
        if not ring.any():
            break
        num = cv2.boxFilter(out * have, -1, (3, 3), normalize=False)
        den = cv2.boxFilter(have.astype(F), -1, (3, 3), normalize=False)
        out[ring] = num[ring] / den[ring]
        have = grown
    out[have == 0] = 1.0
    return np.clip(out, 0, 1)


def ao_region(ao, x0, y0, x1, y1):
    qx0, qy0 = x0 // AO_FACTOR, y0 // AO_FACTOR
    qx1, qy1 = -(-x1 // AO_FACTOR), -(-y1 // AO_FACTOR)
    q = ao[qy0:qy1, qx0:qx1]
    up = cv2.resize(q, ((qx1 - qx0) * AO_FACTOR, (qy1 - qy0) * AO_FACTOR), interpolation=cv2.INTER_LINEAR)
    return up[y0 - qy0 * AO_FACTOR:y1 - qy0 * AO_FACTOR, x0 - qx0 * AO_FACTOR:x1 - qx0 * AO_FACTOR]


def region(isl, size):
    return (max(0, isl["x"] - MARGIN), max(0, isl["y"] - MARGIN),
            min(size, isl["x"] + isl["w"] + MARGIN), min(size, isl["y"] + isl["h"] + MARGIN))


def paint_island(key, isl, mat, ids, idx, ao, size):
    x0, y0, x1, y1 = region(isl, size)
    h, w = y1 - y0, x1 - x0
    scale = isl["scale"]
    yy, xx = np.mgrid[0:h, 0:w].astype(F)
    lx, ly = xx + F(x0 - isl["x"]), yy + F(y0 - isl["y"])
    del xx, yy
    if isl["rot"]:
        lx, ly = F(isl["ext_u"]) - ly, lx
    U = F(isl["origin"][0]) + lx / F(scale)
    V = F(isl["origin"][1]) + ly / F(scale)
    del lx, ly
    mask = (ids[y0:y1, x0:x1] == idx).astype(np.uint8)
    a = ao_region(ao, x0, y0, x1, y1)
    if isl["rot"]:
        U, V, mask, a = (np.ascontiguousarray(np.rot90(x, -1)) for x in (U, V, mask, a))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) / F(scale)
    seed = 1009 + 7919 * (sum(map(ord, key)) % 997)
    period = isl["ext_u"] / scale if key in WRAPPED else None
    c = Ctx(U, V, mask, dist, a, seed, scale, key, period)
    c.glow = None
    color, height, spec = MATERIALS.get(mat, leather)(c)
    glow = c.glow
    if has_seam(key, mat):
        groove = smoothstep(0.1, 0.0, dist)
        ridge = smoothstep(0.06, 0.14, dist) * smoothstep(0.34, 0.16, dist)
        height = height - 0.016 * groove ** 2 + 0.004 * ridge
        color = color * (1 - 0.32 * groove[..., None])
        if mat in ("leather", "maskleather"):
            wear = np.clip(noise(c, seed + 11, 0.6 * scale), 0, None) * ridge
            color = color * (1 + 0.22 * wear[..., None]) + rgb((10, 5, 2))[None, None] * wear[..., None]
            spec = spec * (1 + 0.25 * ridge)
    if key.startswith("corset"):
        channel = smoothstep(0.22, 0.3, dist) * smoothstep(0.64, 0.56, dist)
        height = height + 0.012 * channel
    rows = stitch_rows(key)
    if rows:
        T, H, D = stitches(mask, scale, rows)
        height = height + 0.014 * np.sqrt(T) - 0.006 * H * (1 - T) - 0.003 * D
        color = color * (1 - 0.45 * H * (1 - T))[..., None] * (1 - 0.06 * D)[..., None]
        color = color * (1 - T[..., None]) + rgb(THREAD)[None, None] * (0.75 + 0.25 * a)[..., None] * T[..., None]
        spec = spec * (1 - T) + 0.24 * T
    lo = 0.45 if mat == "fabric" else 0.55
    color = color * (lo + (1 - lo) * a)[..., None]
    spec = spec * (0.45 + 0.55 * a)
    if glow is not None:
        glow = glow * mask[..., None]
    if isl["rot"]:
        color, height, spec = (np.ascontiguousarray(np.rot90(x, 1)) for x in (color, height, spec))
        if glow is not None:
            glow = np.ascontiguousarray(np.rot90(glow, 1))
    nrm = normals(height, scale)
    return (x0, y0, x1, y1), to_u8(color), to_u8(nrm * 0.5 + 0.5), to_u8(spec), glow


def paint_patch(key, isl, size):
    x0, y0, x1, y1 = region(isl, size)
    h, w = y1 - y0, x1 - x0
    col, spec_v, var = PATCHES[key]
    seed = 31 + sum(map(ord, key))
    yy, xx = np.mgrid[0:h, 0:w].astype(F)
    c = Ctx(xx, yy, np.ones((h, w), np.uint8), None, None, seed, 1.0, key, None)
    low = noise(c, seed, 48.0)
    fine = noise(c, seed + 1, 6.0, octaves=2)
    color = rgb(col)[None, None] * (1 + var * low[..., None] + 0.4 * var * fine[..., None])
    spec = np.full((h, w), spec_v, F) * (1 + 0.15 * low)
    if key == "patch:brass":
        patina = smoothstep(0.2, 0.9, noise(c, seed + 2, 60.0))
        color = color * (1 - 0.4 * patina[..., None]) + rgb((58, 54, 36))[None, None] * 0.4 * patina[..., None]
        spec = spec * (1 - 0.45 * patina)
    nrm = np.zeros((h, w, 3), F)
    nrm[..., 2] = 1
    return (x0, y0, x1, y1), to_u8(color), to_u8(nrm * 0.5 + 0.5), to_u8(spec)


def write_bgra_dds(path, levels):
    h, w = levels[0].shape[:2]
    header = bytearray(128)
    struct.pack_into("<4sIIIIIII", header, 0, b"DDS ", 124, 0x2100F, h, w, w * 4, 0, len(levels))
    struct.pack_into("<IIIIIIII", header, 76, 32, 0x41, 0, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    struct.pack_into("<I", header, 108, 0x401008)
    with open(path, "wb") as fh:
        fh.write(header)
        for lvl in levels:
            fh.write(np.ascontiguousarray(lvl[..., [2, 1, 0, 3]]).tobytes())


def read_bgra_dds(path):
    raw = np.fromfile(path, np.uint8)
    h, w = struct.unpack_from("<II", raw, 12)
    count = struct.unpack_from("<I", raw, 28)[0]
    levels, pos = [], 128
    for i in range(count):
        hh, ww = max(h >> i, 1), max(w >> i, 1)
        levels.append(raw[pos:pos + hh * ww * 4].reshape(hh, ww, 4)[..., [2, 1, 0, 3]])
        pos += hh * ww * 4
    return levels


def normal_levels(nrm, spec):
    levels = [np.dstack([nrm, spec])]
    cur = levels[0]
    while cur.shape[0] > 1:
        hh, ww = cur.shape[0] // 2, cur.shape[1] // 2
        nxt = np.empty((hh, ww, 4), np.uint8)
        for r in range(0, hh, 512):
            r1 = min(hh, r + 512)
            blk = cur[2 * r:2 * r1].astype(F) / 255
            blk = blk.reshape(r1 - r, 2, ww, 2, 4).mean(axis=(1, 3))
            n = blk[..., :3] * 2 - 1
            n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
            nxt[r:r1, :, :3] = to_u8(n * 0.5 + 0.5)
            nxt[r:r1, :, 3] = to_u8(blk[..., 3])
        levels.append(nxt)
        cur = nxt
    return levels


def texconv(src, out_dir):
    subprocess.run([TEXCONV, "-nologo", "-y", "-f", "BC7_UNORM", "-m", "0", "-o", out_dir, src], check=True,
                   stdout=subprocess.DEVNULL)


def save_png(path, img):
    cv2.imwrite(path, img[..., ::-1] if img.ndim == 3 else img, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def export_sets(work_dir, sets_dir):
    names = [NAME + suffix for suffix in ("_d.png", "_n.dds", "_g.png", "_m.png")]
    for label, k in SETS:
        src = os.path.join(work_dir, label) if k else work_dir
        if k:
            os.makedirs(src, exist_ok=True)
            for name in names:
                path = os.path.join(work_dir, name)
                if name.endswith(".dds"):
                    write_bgra_dds(os.path.join(src, name), read_bgra_dds(path)[k:])
                    continue
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                size = img.shape[0] >> k
                cv2.imwrite(os.path.join(src, name), cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
        target = os.path.join(sets_dir, label)
        os.makedirs(target, exist_ok=True)
        for name in names:
            texconv(os.path.join(src, name), target)
        log("%s set compressed" % label)


def main(manifest_path, out_dir, work_dir):
    global MASK_ARC
    manifest = pickle.load(open(manifest_path, "rb"))
    MASK_ARC = manifest.get("mask_arc")
    size = manifest["size"]
    layout = manifest["layout"]
    parts = manifest["parts"]
    keys = [k for k, v in layout["islands"].items() if not v["patch"]]
    mats = {}
    for p in parts:
        mats.setdefault(p["island"], p["mat"])
    ids = rasterize_ids(parts, keys, size)
    log("islands rasterized")
    ao = ao_map(parts, size)
    log("ambient occlusion map")
    color = np.zeros((size, size, 3), np.uint8)
    nrm = np.full((size, size, 3), (128, 128, 255), np.uint8)
    spec = np.zeros((size, size), np.uint8)
    q = size // AO_FACTOR
    glow = np.zeros((q, q, 3), np.uint8)
    env = np.zeros((q, q), np.uint8)
    for idx, key in enumerate(keys):
        (x0, y0, x1, y1), c, n, s, gl = paint_island(key, layout["islands"][key], mats[key], ids, idx, ao, size)
        color[y0:y1, x0:x1], nrm[y0:y1, x0:x1], spec[y0:y1, x0:x1] = c, n, s
        if gl is not None:
            g = (slice(y0 // AO_FACTOR, -(-y1 // AO_FACTOR)), slice(x0 // AO_FACTOR, -(-x1 // AO_FACTOR)))
            small = cv2.resize(gl, (g[1].stop - g[1].start, g[0].stop - g[0].start), interpolation=cv2.INTER_AREA)
            glow[g] = np.maximum(glow[g], to_u8(small))
        if mats[key] in ENV_MATS:
            g = (slice(y0 // AO_FACTOR, -(-y1 // AO_FACTOR)), slice(x0 // AO_FACTOR, -(-x1 // AO_FACTOR)))
            small = cv2.resize(s, (g[1].stop - g[1].start, g[0].stop - g[0].start), interpolation=cv2.INTER_AREA)
            env[g] = np.maximum(env[g], to_u8(small.astype(F) / 255 * F(ENV_MATS[mats[key]])))
        log("%s (%s)" % (key, mats[key]))
    del ids
    for key, isl in layout["islands"].items():
        if not isl["patch"]:
            continue
        (x0, y0, x1, y1), c, n, s = paint_patch(key, isl, size)
        color[y0:y1, x0:x1], nrm[y0:y1, x0:x1], spec[y0:y1, x0:x1] = c, n, s
        g = (slice(y0 // AO_FACTOR, -(-y1 // AO_FACTOR)), slice(x0 // AO_FACTOR, -(-x1 // AO_FACTOR)))
        if key in GLOW:
            glow[g] = GLOW[key]
        if key in ENV:
            small = cv2.resize(s, (g[1].stop - g[1].start, g[0].stop - g[0].start), interpolation=cv2.INTER_AREA)
            env[g] = to_u8(small.astype(F) / 255 * F(ENV[key] / PATCHES[key][1]))
    log("patches")
    os.makedirs(work_dir, exist_ok=True)
    d_png = os.path.join(work_dir, NAME + "_d.png")
    save_png(d_png, color)
    save_png(os.path.join(work_dir, NAME + "_d_preview.png"),
             cv2.resize(color, (2048, 2048), interpolation=cv2.INTER_AREA))
    del color
    write_bgra_dds(os.path.join(work_dir, NAME + "_n.dds"), normal_levels(nrm, spec))
    del nrm, spec
    save_png(os.path.join(work_dir, NAME + "_g.png"), glow)
    save_png(os.path.join(work_dir, NAME + "_m.png"), np.dstack([env] * 3))
    log("sources written")
    export_sets(work_dir, out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:4])

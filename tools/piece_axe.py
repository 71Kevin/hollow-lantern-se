import numpy as np

import hl_geom

BLADE = [(-1.2, 37.6), (-3.0, 38.6), (-4.6, 40.4), (-5.4, 39.7), (-7.2, 41.4), (-9.6, 43.6), (-12.2, 45.8),
         (-13.8, 43.6), (-15.6, 39.6), (-16.6, 34.4), (-16.4, 29.0), (-15.2, 23.8), (-13.0, 19.6), (-10.2, 17.4),
         (-9.6, 19.0), (-7.6, 22.4), (-5.0, 25.4), (-2.6, 27.2), (-1.2, 27.8)]
EDGE = (6, 13)
EDGE_BAND = 0.95
PIVOT = (0.0, 32.8)
WINDOWS = [(155.0, 6.3, 1.25, 3.6), (180.0, 6.7, 1.35, 4.2), (205.0, 6.3, 1.25, 3.6)]
STEP = 0.42
SPINE_T = 0.42
EDGE_T = 0.05
UV_SCALE = 0.6
UV_X = 17.0
UV_Y = 46.0
HEAD_C = 33.0
SOCKET = (1.3, 1.0, 27.0, 38.6)
HOOK = [(1.0, 33.4), (3.6, 33.9), (6.0, 35.3), (7.2, 37.4), (6.8, 39.1), (5.6, 39.5), (5.1, 38.6)]
LANTERN_R = 2.0
VARIANTS = {
    "waraxe": {"scale": 1.0, "head_y": 0.0, "bottom": -12.5, "top": 38.0, "grip": (-10.4, 3.6), "radius": 0.72,
               "lace": (17.5, 26.2), "back": "hook"},
    "battleaxe": {"scale": 1.45, "head_y": 12.5, "bottom": -41.5, "top": 55.0, "grip": (-38.0, 6.0),
                  "radius": 0.92, "lace": (21.0, 34.0), "back": "blade"},
}


def polygon_sdf(p, poly):
    poly = np.asarray(poly, dtype=np.float64)
    d = np.full(len(p), np.inf)
    inside = np.zeros(len(p), dtype=bool)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
        cross = (a[1] > p[:, 1]) != (b[1] > p[:, 1])
        xi = a[0] + (p[:, 1] - a[1]) * ab[0] / (ab[1] if ab[1] != 0 else 1e-12)
        inside ^= cross & (p[:, 0] < xi)
    return np.where(inside, -d, d)


def polyline_dist(p, line):
    d = np.full(len(p), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return d


def outline():
    return hl_geom.catmull_rom(np.array(BLADE), 12, closed=True)


def edge_line():
    return outline()[EDGE[0] * 12:EDGE[1] * 12 + 1]


def window(angle, radius, width, length, n=24):
    a = np.radians(angle)
    d = np.array([np.cos(a), np.sin(a)])
    q = np.array([-d[1], d[0]])
    c = np.array(PIVOT) + d * radius
    u0 = -length / 2 + 0.55 * length
    t = np.linspace(0, 1, n)
    side_u = np.linspace(-length / 2, u0, n)
    arch_u = u0 + (length / 2 - u0) * t
    arch_w = width / 2 * (1 - t) * (1 + 0.9 * t)
    base = np.linspace(0, np.pi, n)[1:-1]
    u = np.concatenate([side_u, arch_u[1:], arch_u[::-1][1:], side_u[::-1][1:],
                        -length / 2 - 0.22 * width * np.sin(base)])
    w = np.concatenate([np.full(n, -width / 2), -arch_w[1:], arch_w[::-1][1:], np.full(n - 1, width / 2),
                        width / 2 * np.cos(base)])
    return c + np.outer(u, d) + np.outer(w, q)


def window_field(p):
    return np.max([-polygon_sdf(p, window(*w)) for w in WINDOWS], axis=0)


def half_thickness(p):
    k = np.clip(polyline_dist(p, edge_line()) / 4.5, 0, 1)
    return 0.5 * (EDGE_T + (SPINE_T - EDGE_T) * np.sqrt(k))


def plate_uv(p):
    return np.stack([p[:, 0] + UV_X, UV_Y - p[:, 1]], 1) * UV_SCALE


def blade_plate():
    lo = np.array(BLADE).min(0) - 0.6
    hi = np.array(BLADE).max(0) + 0.6
    xs = np.arange(lo[0], hi[0] + 1e-6, STEP)
    ys = np.arange(lo[1], hi[1] + 1e-6, STEP)
    X, Y = np.meshgrid(xs, ys)
    v = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    t = hl_geom.grid_tris(len(ys), len(xs))
    full = hl_geom.iso_cut(v, t, -polygon_sdf(v[:, :2], outline()))[:2]
    cut = hl_geom.iso_cut(*full, -window_field(full[0][:, :2]))[:2]
    return full, split_band(*cut)


def split_band(v, t):
    f = EDGE_BAND - polyline_dist(v[:, :2], edge_line())
    a, ta, _ = hl_geom.iso_cut(v, t, f)
    b, tb, _ = hl_geom.iso_cut(v, t, -f)
    return (b, tb), (a, ta)


def extrude(v, t, open_edge=None):
    p = v[:, :2]
    h = half_thickness(p)
    n = len(v)
    top = np.c_[p, h]
    bot = np.c_[p, -h]
    tris = [t, t[:, ::-1] + n]
    rim_v, rim_t = [], []
    base = 2 * n
    for loop in hl_geom.boundary_loops(t):
        q = p[loop]
        tang = hl_geom.normalize(np.roll(q, -1, 0) - np.roll(q, 1, 0))
        out = np.stack([tang[:, 1], -tang[:, 0]], 1)
        mid = np.c_[q + out * (0.45 * h[loop])[:, None], np.zeros(len(loop))]
        ids = base + sum(len(r) for r in rim_v) + np.arange(len(loop))
        rim_v.append(mid)
        m = len(loop)
        for i in range(m):
            j = (i + 1) % m
            a, b = loop[i], loop[j]
            if open_edge is not None and open_edge[a] and open_edge[b]:
                continue
            rim_t += [[a, ids[i], b], [b, ids[i], ids[j]], [ids[i], a + n, ids[j]], [ids[j], a + n, b + n]]
    allv = np.concatenate([top, bot])
    plate = (allv, np.concatenate(tris), np.concatenate([plate_uv(p), plate_uv(p)]))
    if not rim_t:
        return plate, None
    rv = np.concatenate(rim_v)
    rt = np.array(rim_t)
    rim_all_v = np.concatenate([allv, rv])
    used = np.unique(rt)
    remap = -np.ones(len(rim_all_v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    rim = (rim_all_v[used], remap[rt], rim_all_v[used][:, [0, 2]])
    return plate, rim


def window_glass(full):
    v, t = full
    keep = (window_field(v[:, :2])[t] > -0.3).all(1)
    gv, gt, _, _ = hl_geom.compact(v, t[keep])
    front = np.c_[gv[:, :2], np.full(len(gv), 0.03)]
    back = np.c_[gv[:, :2], np.full(len(gv), -0.03)]
    return np.concatenate([front, back]), np.concatenate([gt, gt[:, ::-1] + len(gv)])


def superellipse(rx, rz, power=4.0):
    def profile(s, ang):
        return 1.0 / ((np.abs(np.cos(ang)) / rz) ** power + (np.abs(np.sin(ang)) / rx) ** power) ** (1 / power)
    return profile


def pumpkin(center, radius, lobes=8, squash=0.78, flip=False):
    phis = np.linspace(0, 2 * np.pi, 25)
    thetas = np.linspace(0, np.pi, 13)
    pts = []
    for th in thetas:
        for ph in phis:
            r = radius * (1 + 0.1 * np.cos(lobes * ph)) * (1 - 0.18 * np.exp(-(th / 0.35) ** 2)) * \
                (1 - 0.18 * np.exp(-((np.pi - th) / 0.35) ** 2))
            pts.append([np.sin(th) * np.cos(ph) * r, np.cos(th) * radius * squash, np.sin(th) * np.sin(ph) * r])
    v = np.array(pts) + center
    t = hl_geom.grid_tris(len(thetas), len(phis))[:, ::-1].copy()
    if flip:
        t = t[:, ::-1].copy()
    uv = np.stack([np.tile(phis, len(thetas)), np.repeat(thetas, len(phis))], 1) * radius
    return v, t, uv


def ring(y, radius, tube, segs=32):
    path = np.array([[np.cos(a) * radius, y, np.sin(a) * radius] for a in np.linspace(0, 2 * np.pi, segs + 1)])
    v, t, uv, _ = hl_geom.sweep(path, np.full(len(path), tube), 8, up=(0, 1, 0))
    return v, t, uv


def lantern(base_y, radius, lobes=8):
    squash = 0.82
    cy = base_y + radius * squash + 0.2
    core = pumpkin(np.array([0.0, cy, 0.0]), radius * 0.8, lobes, squash)
    ribs = []
    for k in range(lobes):
        a = 2 * np.pi * (k + 0.5) / lobes
        th = np.linspace(0.12, np.pi - 0.12, 26)
        r = radius * (1 + 0.04 * np.cos(lobes * a)) * np.sin(th) ** 0.9
        path = np.stack([np.cos(a) * r, cy + np.cos(th) * radius * squash, np.sin(a) * r], 1)
        rv, rt, ruv, _ = hl_geom.sweep(path, np.full(len(path), 0.085 * radius / LANTERN_R), 6,
                                       up=(0, 1, 0), cap_start=True, cap_end=True)
        ribs.append((rv, rt, ruv))
    collars = [ring(cy - radius * squash * 0.93, radius * 0.42, 0.1 * radius / LANTERN_R),
               ring(cy + radius * squash * 0.93, radius * 0.38, 0.09 * radius / LANTERN_R)]
    top = cy + radius * squash
    stem_t = np.linspace(0, 1, 40)
    ang = stem_t * 2.6 * np.pi
    rad = radius * (0.08 + 0.32 * stem_t)
    stem = np.stack([np.cos(ang) * rad * 0.6, top + 0.05 + stem_t * radius * 1.1, np.sin(ang) * rad], 1)
    sv, st, suv, _ = hl_geom.sweep(stem, radius * (0.16 * (1 - stem_t) ** 1.2 + 0.03), 10, up=(1, 0, 0),
                                   cap_start=True, cap_end=True)
    return core, ribs, collars, (sv, st, suv), top + radius * 1.1


def haft(cfg):
    ys = np.linspace(cfg["bottom"] + 0.6, cfg["top"], 48)
    r = cfg["radius"] * (1 - 0.08 * (ys - ys[0]) / (ys[-1] - ys[0]))
    path = np.stack([np.zeros_like(ys), ys, np.zeros_like(ys)], 1)
    v, t, uv, _ = hl_geom.sweep(path, r, 14, up=(0, 0, 1), cap_start=True, cap_end=True)
    return v, t, uv * [cfg["radius"], 1.0]


def wrap(cfg, pitch=0.72, width=0.31, thick=0.05):
    y0, y1 = cfg["grip"]
    r = cfg["radius"] + 0.04
    turns = (y1 - y0) / pitch
    a = np.linspace(0, 2 * np.pi * turns, int(turns * 18) + 2)
    path = np.stack([np.cos(a) * r, y0 + a / (2 * np.pi) * pitch, np.sin(a) * r], 1)

    def profile(s, ang):
        return width * thick / np.sqrt((thick * np.cos(ang)) ** 2 + (width * np.sin(ang)) ** 2)
    v, t, uv, _ = hl_geom.sweep(path, np.ones(len(path)), 6, profile=profile, up=(0, 1, 0))
    return v, t, uv


def lacing(cfg, crossings=5, radius=0.075):
    y0, y1 = cfg["lace"]
    r = cfg["radius"] * 0.95 + radius * 0.9
    out = []
    for sign in (1, -1):
        for phase in (0.0, np.pi):
            s = np.linspace(0, 1, 100)
            a = phase + sign * s * np.pi * crossings
            path = np.stack([np.cos(a) * r, y0 + s * (y1 - y0), np.sin(a) * r], 1)
            v, t, uv, _ = hl_geom.sweep(path, np.full(len(path), radius), 6, up=(0, 1, 0))
            out.append((v, t, uv))
    return out


def head_transform(v, cfg):
    s = cfg["scale"]
    out = np.array(v, dtype=np.float64)
    out[:, 0] *= s
    out[:, 2] *= s
    out[:, 1] = (out[:, 1] - HEAD_C) * s + HEAD_C + cfg["head_y"]
    return out


def mirror_back(v, t, scale):
    out = np.array(v, dtype=np.float64)
    out[:, 0] = -out[:, 0] * scale
    out[:, 1] = (out[:, 1] - HEAD_C) * scale + HEAD_C
    out[:, 2] = out[:, 2] * (0.6 + 0.4 * scale)
    return out, t[:, ::-1].copy()


def blade_parts(Part, tf, tag, mirror=None):
    full, (body, band) = blade_plate()

    def seam(v):
        return np.abs(polyline_dist(v[:, :2], edge_line()) - EDGE_BAND) < 1e-3
    (pv, pt, puv), rim = extrude(*body, open_edge=seam(body[0]))
    (bv, bt, buv), brim = extrude(*band, open_edge=seam(band[0]))
    parts = []
    sets = [("axehead" + tag, "Axe", pv, pt, puv, "axehead", "iron"),
            ("axe_edge" + tag, "AxeGlow", bv, bt, buv, "axe_edge", "ember")]
    if rim is not None:
        sets.append(("axehead_rim" + tag, "Axe", rim[0], rim[1], rim[2], "patch:steel", "steel"))
    if brim is not None:
        sets.append(("axe_edge_rim" + tag, "AxeGlow", brim[0], brim[1], brim[2], "patch:ember", "ember"))
    gv, gt = window_glass(full)
    sets.append(("axe_glass" + tag, "AxeGlow", gv, gt, gv[:, :2], "patch:flesh", "flesh"))
    for k, w in enumerate(WINDOWS):
        loop = window(*w)
        loop = hl_geom.resample_polyline(np.vstack([loop, loop[:1]]), 48, closed=True)
        h = half_thickness(loop)
        for side in (1, -1):
            path = np.c_[loop, side * (h + 0.02)]
            path = np.vstack([path, path[:1]])
            fv, ft, fuv, _ = hl_geom.sweep(path, np.full(len(path), 0.085), 6, up=(0, 0, 1))
            sets.append(("axe_frame%d%s" % (k, tag), "AxeBrass", fv, ft, fuv, "patch:brass", "brass"))
    for name, shape, v, t, uv, isl, mat in sets:
        if mirror is not None:
            v, t = mirror_back(v, t, mirror)
        parts.append(Part(name, shape, tf(v), t, uv, isl, mat))
    return parts


def build_variant(name, Part):
    cfg = VARIANTS[name]
    s = cfg["scale"]

    def tf(v):
        return head_transform(v, cfg)
    parts = blade_parts(Part, tf, "")
    y0, y1 = SOCKET[2], SOCKET[3]
    path = np.array([[0, y, 0] for y in np.linspace(y0, y1, 14)])
    sv, st, suv, _ = hl_geom.sweep(path, np.ones(len(path)), 28, profile=superellipse(SOCKET[0], SOCKET[1]),
                                   cap_start=True, cap_end=True)
    parts.append(Part("axe_socket", "Axe", tf(sv), st, suv, "patch:steel", "steel"))
    if cfg["back"] == "hook":
        tp = hl_geom.catmull_rom(np.c_[np.array(HOOK), np.zeros(len(HOOK))], 10)
        k = np.linspace(0, 1, len(tp))
        tv, tt, tuv, _ = hl_geom.sweep(tp, 1.0 * (1 - k) ** 1.15 + 0.05, 14,
                                       profile=lambda f, ang: 1.0 / np.sqrt(np.cos(ang) ** 2 +
                                                                            (np.sin(ang) / 0.55) ** 2),
                                       up=(0, 0, 1), cap_end=True)
        parts.append(Part("axe_hook", "Axe", tf(tv), tt, tuv, "patch:steel", "steel"))
    else:
        parts += blade_parts(Part, tf, "_back", mirror=0.52)
    brass = [ring(y, 1.18, 0.3) for y in (y0 - 0.25, y1 + 0.25)]
    brass = [(tf(v), t, uv) for v, t, uv in brass]
    core, ribs, collars, stem, tip = lantern(y1 + 0.3, LANTERN_R)
    parts.append(Part("axe_core", "AxeGlow", tf(core[0]), core[1], core[2], "patch:flesh", "flesh"))
    for k, (v, t, uv) in enumerate(ribs):
        parts.append(Part("axe_rib%d" % k, "Axe", tf(v), t, uv, "patch:steel", "steel"))
    for v, t, uv in collars:
        brass.append((tf(v), t, uv))
    parts.append(Part("axe_stem", "Axe", tf(stem[0]), stem[1], stem[2], "patch:steel", "steel"))
    g0, g1 = cfg["grip"]
    r = cfg["radius"]
    for y in (g0 - 0.35, g1 + 0.35):
        brass.append(ring(y, r + 0.12, 0.2 * (0.8 + 0.2 * s)))
    l0, l1 = cfg["lace"]
    for y in (l0 - 0.3, l1 + 0.3):
        brass.append(ring(y, r + 0.08, 0.16 * (0.8 + 0.2 * s)))
    brass.append(pumpkin(np.array([0, cfg["bottom"], 0]), 1.05 * r + 0.25))
    for k, (v, t, uv) in enumerate(brass):
        parts.append(Part("axe_brass%d" % k, "AxeBrass", v, t, uv, "patch:brass", "brass"))
    hv, ht, huv = haft(cfg)
    parts.append(Part("axe_haft", "AxeHaft", hv, ht, huv, "patch:ebony", "wood"))
    wv, wt, wuv = wrap(cfg)
    parts.append(Part("axe_wrap", "AxeHaft", wv, wt, wuv, "patch:edge", "edge"))
    for k, (v, t, uv) in enumerate(lacing(cfg, crossings=5 if name == "waraxe" else 7)):
        parts.append(Part("axe_lace%d" % k, "AxeHaft", v, t, uv, "patch:cord", "cord"))
    for y in list(np.linspace(g0 - 1.1, g0 - 0.55, 3)) + list(np.linspace(g1 + 0.55, g1 + 1.1, 3)):
        cv, ct, cuv = ring(y, r + 0.05, 0.09)
        parts.append(Part("axe_cord%.1f" % y, "AxeHaft", cv, ct, cuv, "patch:cord", "cord"))
    for p in parts:
        p.bones, p.w, p.D, p.sliders = [], None, None, []
    return parts


def build(body=None, proxy=None, log=print):
    from hl_parts import Part
    out = {}
    for name in VARIANTS:
        out[name] = build_variant(name, Part)
        log("%s: %d verts" % (name, sum(len(p.v) for p in out[name])))
    return out

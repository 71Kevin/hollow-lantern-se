import numpy as np

import hl_geom

STEP = 0.11
X_MAX = 6.0
Z_LO = 119.8
Z_HI = 129.6
CLEAR = 0.14
THICK = 0.16
LIFT = 0.12
HALF = [(0.0, 126.78), (0.16, 126.58), (0.42, 126.25), (0.78, 126.08), (1.5, 126.3), (2.7, 126.5), (3.7, 126.95),
        (4.5, 127.55), (5.12, 128.15), (5.32, 127.3), (5.38, 126.05), (5.22, 124.65), (4.82, 123.35), (4.38, 122.2),
        (3.6, 121.8), (2.8, 121.62), (2.0, 121.76), (1.35, 122.1), (0.9, 122.35), (0.45, 122.62), (0.0, 122.72)]
EYE = (2.38, 123.56)
EYE_LEN = 1.3
EYE_H = 0.62
EYE_SLANT = 0.2
MEDALLION = (0.0, 124.95)
MEDAL_R = 0.42
STUDS = [(4.95, 126.6), (4.42, 122.65)]
UV_X = 7.0
UV_Z = 130.0


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def blur(img, sigma, step):
    r = int(np.ceil(3 * sigma / step))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) * step / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, r, mode="edge")
    out = np.apply_along_axis(lambda a: np.convolve(a, k, mode="valid"), 0, pad)
    return np.apply_along_axis(lambda a: np.convolve(a, k, mode="valid"), 1, out)


def max_filter(img, radius, step):
    r = int(np.ceil(radius / step))
    pad = np.pad(img, r, mode="edge")
    out = np.full_like(img, -np.inf)
    for dz in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dz * dz > r * r:
                continue
            out = np.maximum(out, pad[r + dz:r + dz + img.shape[0], r + dx:r + dx + img.shape[1]])
    return out


def fill_missing(img):
    out = img.copy()
    for _ in range(400):
        bad = np.isnan(out)
        if not bad.any():
            break
        pad = np.pad(out, 1, mode="edge")
        nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        cnt = (~np.isnan(nb)).sum(0)
        avg = np.nansum(nb, 0) / np.maximum(cnt, 1)
        grow = bad & (cnt > 0)
        out[grow] = avg[grow] - 0.06
    return out


def outline_points(samples=10):
    half = np.array(HALF)
    loop = np.vstack([half, (half * [-1, 1])[-2:0:-1]])
    return hl_geom.catmull_rom(loop, samples, closed=True)


def polygon_sdf(p, poly):
    a = poly
    b = np.roll(poly, -1, axis=0)
    d = np.full(len(p), np.inf)
    inside = np.zeros(len(p), dtype=bool)
    for (ax, az), (bx, bz) in zip(a, b):
        ex, ez = bx - ax, bz - az
        px, pz = p[:, 0] - ax, p[:, 1] - az
        h = np.clip((px * ex + pz * ez) / (ex * ex + ez * ez + 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(px - h * ex, pz - h * ez))
        cross = ((az > p[:, 1]) != (bz > p[:, 1])) & (p[:, 0] < (bx - ax) * (p[:, 1] - az) / (bz - az + 1e-12) + ax)
        inside ^= cross
    return np.where(inside, d, -d)


def outline(x, z):
    p = np.stack([np.ravel(x), np.ravel(z)], 1)
    return polygon_sdf(p, outline_points()).reshape(np.shape(x))


def eye_shape(samples=64):
    s = np.linspace(-1, 1, samples)
    top = EYE_H * (1 - s ** 2) ** 0.8 * (1 - 0.18 * s) + 0.1 * np.clip(s, 0, None) ** 3
    bot = -EYE_H * 0.78 * (1 - s ** 2) ** 0.95 * (1 + 0.12 * s)
    u = np.concatenate([s, s[::-1]]) * EYE_LEN
    w = np.concatenate([top, bot[::-1]])
    flick = np.clip(u / EYE_LEN - 0.55, 0, None) ** 2 * 0.55
    w = w + flick
    pts = []
    for side in (-1, 1):
        ang = EYE_SLANT * side
        uu = u * side
        x = EYE[0] * side + uu * np.cos(ang) - w * np.sin(ang)
        z = EYE[1] + uu * np.sin(ang) + w * np.cos(ang)
        pts.append(np.stack([x, z], 1)[:-1])
    return pts


def eye_field(x, z):
    p = np.stack([np.ravel(x), np.ravel(z)], 1)
    return np.max([polygon_sdf(p, e) for e in eye_shape()], axis=0).reshape(np.shape(x))


def heads():
    import hl_face
    out = []
    for kind in ("vanilla", "highpoly"):
        v, t, m = hl_face.load_head(kind)
        env, n = hl_face.sampled_envelope(v, t, m)
        out.append((v + n * env[:, None], t))
        for race in hl_face.RACES:
            out.append((v + m[("race", race)], t))
    return out


def surface():
    import hl_blend
    variants = heads()
    trees = [hl_blend.bvh(v, t) for v, t in variants]
    xs = np.arange(-X_MAX, X_MAX + 1e-6, STEP)
    zs = np.arange(Z_LO, Z_HI + 1e-6, STEP)
    X, Z = np.meshgrid(xs, zs)
    o = np.stack([X.ravel(), np.full(X.size, 30.0), Z.ravel()], 1)
    R = np.full(X.size, np.nan)
    for tree in trees:
        loc, nrm, face, dist = hl_blend.raycast(tree, o, np.tile([0, -1.0, 0], (len(o), 1)), 60.0)
        R = np.fmax(R, np.where(np.isfinite(dist), loc[:, 1], np.nan))
    Rf = fill_missing(R.reshape(X.shape))
    E = blur(max_filter(Rf, 0.3, STEP), 0.3, STEP) + CLEAR
    for _ in range(4):
        E = blur(np.maximum(E, Rf + CLEAR), 0.3, STEP)
    for _ in range(6):
        pts = np.stack([X.ravel(), E.ravel(), Z.ravel()], 1)
        need = np.zeros(X.size)
        for tree in trees:
            loc, nrm, face, dist = hl_blend.nearest(tree, pts)
            cos = np.clip(np.abs(nrm[:, 1]), 0.35, 1.0)
            need = np.maximum(need, np.clip(CLEAR + THICK - dist, 0, None) / cos)
        need = need.reshape(E.shape)
        if need.max() < 0.01:
            break
        E = E + blur(max_filter(need, 0.25, STEP), 0.25, STEP)
    wing = LIFT * smoothstep(126.4, 128.2, Z) * smoothstep(3.6, 5.1, np.abs(X))
    Y = E + wing
    cum = np.concatenate([np.zeros((len(zs), 1)), np.cumsum(np.hypot(STEP, np.diff(Y, axis=1)), 1)], 1)
    c0 = int(np.argmin(np.abs(xs)))
    S = cum - cum[:, c0:c0 + 1] + xs[c0]
    v = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    t = hl_geom.grid_tris(len(zs), len(xs))
    if hl_geom.vertex_normals(v, t)[:, 1].mean() < 0:
        t = t[:, ::-1].copy()
    v, t, a = hl_geom.iso_cut(v, t, outline(v[:, 0], v[:, 2]), {"S": S.ravel()})
    v, t, a = hl_geom.iso_cut(v, t, -eye_field(v[:, 0], v[:, 2]), a)
    return v, t, a["S"], variants, (zs, xs, S)


def boundary_paths(v, t):
    loops = [np.asarray(l) for l in hl_geom.boundary_loops(t)]
    loops.sort(key=len, reverse=True)
    return loops


def piping(v, n, loop, radius, inset):
    path = v[loop] - n[loop] * inset
    path = np.vstack([path, path[:1]])
    path = hl_geom.resample_polyline(hl_geom.catmull_rom(path, 3), max(24, int(len(loop) * 1.5)), closed=True)
    path = np.vstack([path, path[:1]])
    pv, pt, puv, _ = hl_geom.sweep(path, np.full(len(path), radius), 8)
    return pv, pt, puv


def medallion(center, normal, up):
    import piece_corset
    n = hl_geom.normalize(normal)
    u = hl_geom.normalize(up - n * np.dot(up, n))
    r = np.cross(u, n)
    phis = np.linspace(0, 2 * np.pi, 41)
    thetas = np.linspace(0, np.pi / 2, 13)
    verts, uv = [], []
    for th in thetas:
        for ph in phis:
            rad = MEDAL_R * (1 + 0.1 * np.cos(8 * ph)) * np.sin(th)
            h = 0.2 * np.cos(th)
            verts.append(center + r * rad * np.cos(ph) + u * rad * np.sin(ph) * 1.08 + n * h)
            uv.append([ph * MEDAL_R, th * MEDAL_R])
    dv = np.array(verts)
    dt = hl_geom.grid_tris(len(thetas), len(phis))
    if np.dot(hl_geom.face_normals(dv, dt).sum(0), n) < 0:
        dt = dt[:, ::-1].copy()
    gem_phi = np.linspace(0, 2 * np.pi, 33)
    gem_th = np.linspace(0, np.pi / 2, 9)
    gv, guv = [], []
    for th in gem_th:
        for ph in gem_phi:
            rad = 0.2 * np.sin(th)
            gv.append(center + n * (0.17 + 0.13 * np.cos(th)) + r * rad * np.cos(ph) + u * rad * np.sin(ph) * 1.25)
            guv.append([ph * 0.2, th * 0.2])
    gv = np.array(gv)
    gt = hl_geom.grid_tris(len(gem_th), len(gem_phi))
    if np.dot(hl_geom.face_normals(gv, gt).sum(0), n) < 0:
        gt = gt[:, ::-1].copy()
    rv, rt, ruv = piece_corset.torus(center + n * 0.19, n, u, 0.235, 0.045, seg_major=28, seg_minor=6)
    stem_path = np.array([center + u * (MEDAL_R * 1.02), center + u * (MEDAL_R * 1.25) + n * 0.12,
                          center + u * (MEDAL_R * 1.42) + r * 0.08 + n * 0.08])
    sv, st, suv, _ = hl_geom.sweep(stem_path, np.array([0.075, 0.06, 0.04]), 8, cap_end=True)
    return [(dv, dt, np.array(uv)), (rv, rt, ruv), (sv, st, suv)], (gv, gt, np.array(guv))


def stud(center, normal, radius=0.13):
    n = hl_geom.normalize(normal)
    ref = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = hl_geom.normalize(ref - n * np.dot(ref, n))
    r = np.cross(u, n)
    phis = np.linspace(0, 2 * np.pi, 21)
    thetas = np.linspace(0, np.pi / 2, 7)
    verts, uv = [], []
    for th in thetas:
        for ph in phis:
            rad = radius * np.sin(th)
            verts.append(center + r * rad * np.cos(ph) + u * rad * np.sin(ph) + n * 0.07 * np.cos(th))
            uv.append([ph * radius, th * radius])
    v = np.array(verts)
    t = hl_geom.grid_tris(len(thetas), len(phis))
    if np.dot(hl_geom.face_normals(v, t).sum(0), n) < 0:
        t = t[:, ::-1].copy()
    return v, t, np.array(uv)


def surface_point(tree, x, z):
    import hl_blend
    loc, nrm, face, dist = hl_blend.raycast(tree, np.array([[x, 30.0, z]]), np.array([[0, -1.0, 0]]), 60.0)
    return loc[0], (nrm[0] if nrm[0, 1] > 0 else -nrm[0])


def build(body=None, proxy=None, log=print):
    import hl_blend
    from hl_parts import Part, solid_parts
    v, t, arc, variants, table = surface()
    n = hl_geom.vertex_normals(v, t)
    uv = np.stack([arc + UV_X, UV_Z - v[:, 2]], 1)
    parts = solid_parts("mask", "Mask", v, t, uv, np.ones((len(v), 1)), np.zeros((len(v), 0, 3)), THICK,
                        ("mask", "patch:lining", "patch:edge"), ("maskleather", "lining", "edge"), rim_segments=2)
    loops = boundary_paths(v, t)
    pv, pt, puv = piping(v, n, loops[0], 0.075, THICK * 0.5)
    parts.append(Part("mask_piping", "Mask", pv, pt, puv, "patch:edge", "edge"))
    for k, loop in enumerate(loops[1:3]):
        bv, bt, buv = piping(v, n, loop, 0.055, THICK * 0.5)
        parts.append(Part("mask_bezel_%d" % k, "MaskBrass", bv, bt, buv, "patch:brass", "brass"))
    tree = hl_blend.bvh(v, t)
    c, cn = surface_point(tree, *MEDALLION)
    pieces, gem = medallion(c, cn, np.array([0, 0, 1.0]))
    for k, (mv, mt, muv) in enumerate(pieces):
        parts.append(Part("mask_medal_%d" % k, "MaskBrass", mv, mt, muv, "patch:brass", "brass"))
    parts.append(Part("mask_gem", "MaskGlow", gem[0], gem[1], gem[2], "patch:flesh", "flesh"))
    for side in (-1, 1):
        for sx, sz in STUDS:
            sc, sn = surface_point(tree, side * sx, sz)
            mv, mt, muv = stud(sc, sn)
            parts.append(Part("mask_stud", "MaskBrass", mv, mt, muv, "patch:brass", "brass"))
    inner = np.concatenate([p.v for p in parts if p.name.endswith("_in")])
    worst = []
    for hv, ht in variants:
        loc, nrm, face, dist = hl_blend.nearest(hl_blend.bvh(hv, ht), inner)
        worst.append(dist.min())
    log("mask: %d verts, inner clearance per head variant %s" % (sum(len(p.v) for p in parts),
                                                                " ".join("%.2f" % w for w in worst)))
    for p in parts:
        p.bones = ["NPC Head [Head]"]
        p.w = np.ones((len(p.v), 1))
        p.D = None
        p.sliders = []
        if p.name.endswith("_in"):
            p.shape = "MaskLining"
        elif p.name == "mask":
            p.shape = "MaskGlow"
            p.arc_table = table
    return parts, None

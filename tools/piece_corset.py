import numpy as np

import hl_blend
import hl_export
import hl_fit
import hl_geom
import hl_preset
from hl_parts import Part, bind, solid_parts, sub_part

TOP = [(0, 96.0), (0.2, 97.3), (0.45, 98.6), (0.75, 98.9), (1.1, 98.2), (1.57, 97.3), (2.3, 95.6), (np.pi, 95.0)]
BOT = [(0, 74.4), (0.28, 76.2), (0.7, 77.3), (1.57, 77.0), (2.3, 77.6), (np.pi, 77.9)]
SEAMS = [0.42, 0.95, 1.55, 2.2]
GAP = 0.8
OFFSET = 0.24
CINCH = 0.16
THICK = 0.3
EYELETS = 9
COLS = 156
RIM = 2
BRIEF_OFF = 0.21
INTIMATE = ("Vagina", "Pussy", "Clitoral", "Anus", "Anal", "Labia")
BRIEF_TOP = [(0, 71.9), (0.8, 71.7), (1.57, 71.7), (2.4, 72.5), (np.pi, 73.0)]
LEG_OPEN = [(0, 65.6), (0.78, 68.6), (1.57, 69.9), (2.36, 67.4), (np.pi, 64.2)]
LEG_OPEN_INNER = [(0, 65.6), (0.78, 63.6), (1.57, 62.8), (2.36, 62.6), (np.pi, 64.2)]
HIP_L = np.array([-6.62, 0.0, 68.91])
KNEE_L = np.array([-10.12, -1.31, 33.51])


def landmarks(body):
    v = body.v
    front = v[:, 1] > 0
    nip = []
    for side in (-1, 1):
        m = front & (np.sign(v[:, 0]) == side) & (v[:, 2] > 90) & (v[:, 2] < 100)
        nip.append(v[np.argmax(np.where(m, v[:, 1], -99))])
    m = front & (np.abs(v[:, 0]) < 1.5) & (v[:, 2] > 76) & (v[:, 2] < 86)
    navel = v[np.argmin(np.where(m, v[:, 1], 99))]
    return np.array(nip), navel


def make_proxy(body):
    nip, navel = landmarks(body)
    v = body.v
    mask = np.zeros(len(v))
    for c in nip:
        d = np.linalg.norm(v - c, axis=1)
        mask = np.maximum(mask, np.clip((2.6 - d) / 1.0, 0, 1))
    d = np.linalg.norm(v - navel, axis=1)
    mask = np.maximum(mask, np.clip((2.6 - d) / 0.8, 0, 1))
    crotch = (np.abs(v[:, 0]) < 3.4) & (v[:, 2] > 60) & (v[:, 2] < 70.5) & (v[:, 1] > -6.0)
    mask = np.maximum(mask, crotch * np.clip((3.4 - np.abs(v[:, 0])) / 1.2, 0, 1))
    cleft = (np.abs(v[:, 0]) < 2.6) & (v[:, 2] > 58) & (v[:, 2] < 74.5) & (v[:, 1] < -4.0)
    mask = np.maximum(mask, cleft * np.clip((2.6 - np.abs(v[:, 0])) / 1.0, 0, 1))
    return body.smoothed(mask, iters=150)


def offset_at(z):
    return OFFSET - CINCH * np.exp(-((np.asarray(z, dtype=np.float64) - 84.0) / 4.5) ** 2)


def corset_surface(proxy):
    top = hl_fit.sym_curve(TOP)
    bot = hl_fit.sym_curve(BOT)
    zs = np.arange(72.0, 101.51, 0.36)
    cy = hl_fit.torso_axis(proxy, zs)
    p, t, theta, zz = hl_fit.ring_grid(proxy, zs, COLS, cy)
    offset = offset_at(zz)
    p = hl_fit.drape(p, t, proxy, offset, iters=120, lam=0.5)
    nip, _ = landmarks(proxy)
    soft = np.zeros(len(p))
    for c in nip:
        soft = np.maximum(soft, np.clip((3.6 - np.linalg.norm(p - c, axis=1)) / 1.6, 0, 1))
    pairs = hl_geom.neighbor_pairs(t, len(p))
    for _ in range(160):
        p = p + (0.5 * soft)[:, None] * hl_geom.laplacian(p, pairs)
    rows, cols = len(zs), COLS
    g = p.reshape(rows, cols, 3)
    ring = np.linalg.norm(np.diff(np.concatenate([g, g[:, :1]], 1), axis=1), axis=2)
    S = np.concatenate([np.zeros((rows, 1)), np.cumsum(ring, 1)[:, :-1]], 1)
    col = np.linalg.norm(np.diff(g, axis=0), axis=2)
    V = np.concatenate([np.zeros((1, cols)), np.cumsum(col, 0)], 0)
    th = np.mod(theta, 2 * np.pi)
    th = np.where(th > np.pi, th - 2 * np.pi, th)
    attrs = {"theta": th, "z0": zz, "S": S.ravel(), "V": V.ravel(), "off": offset}
    v, tt, a = hl_geom.iso_cut(p, t, top(th) - zz, attrs)
    v, tt, a = hl_geom.iso_cut(v, tt, a["z0"] - bot(a["theta"]), a)
    gap = np.where(v[:, 1] > 0, np.abs(v[:, 0]) - GAP, 1.0)
    v, tt, a = hl_geom.iso_cut(v, tt, gap, a)
    seam_S = {}
    thp = np.mod(th.reshape(rows, cols)[0], 2 * np.pi)
    for s in SEAMS + [2 * np.pi - x for x in SEAMS] + [np.pi, 0.0]:
        k = np.argmin(np.abs(np.angle(np.exp(1j * (thp - s)))))
        seam_S[round(s, 4)] = (zs, S[:, k])
    return v, tt, a, seam_S


def surface_transfer(points, tree, v, t, w, D):
    loc, nrm, face, dist = hl_blend.nearest(tree, points)
    tri = t[face]
    bc = hl_blend.barycentric(loc, v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]])
    return np.einsum("nk,nkb->nb", bc, w[tri]), np.einsum("nk,nksd->nsd", bc, np.asarray(D)[tri])


def edge_blend(points, v, t, w, D, width):
    side_of = np.sign(v[t].mean(1)[:, 0])
    res = []
    for side in (-1, 1):
        tt = t[side_of == side]
        res.append(surface_transfer(points, hl_blend.bvh(v, tt), v, tt, w, D))
    s = np.clip((points[:, 0] + width) / (2 * width), 0, 1)
    s = s * s * (3 - 2 * s)
    return (res[0][0] * (1 - s)[:, None] + res[1][0] * s[:, None],
            res[0][1] * (1 - s)[:, None, None] + res[1][1] * s[:, None, None])


def corset_panels(v, t, a, seam_S, w, D):
    bounds = [0.0] + SEAMS + [np.pi]

    def split(outer):
        thp = np.mod(outer.attrs["theta"], 2 * np.pi)
        pieces = []
        k = 0
        for side in (1, -1):
            for i in range(len(bounds) - 1):
                if side > 0:
                    lo, hi = bounds[i], bounds[i + 1]
                else:
                    lo, hi = 2 * np.pi - bounds[i + 1], 2 * np.pi - bounds[i]
                aa = dict(outer.attrs, w=outer.w, D=outer.D, thp=thp, uv=outer.uv)
                vv, tt, aa = hl_geom.iso_cut(outer.v, outer.t, aa["thp"] - lo, aa)
                vv, tt, aa = hl_geom.iso_cut(vv, tt, hi - aa["thp"], aa)
                if len(tt) == 0:
                    continue
                if side > 0 and i == 0:
                    u = aa["S"]
                else:
                    zs, sk = seam_S[round(lo, 4)]
                    u = aa["S"] - np.interp(aa["z0"], zs, sk)
                uv = np.stack([u - u.min(), aa["V"] - aa["V"].min()], 1)
                p = Part("corset_p%d" % k, outer.shape, vv, tt, uv, "corset_%d" % k, "leather", aa["w"], aa["D"])
                p.normals = hl_geom.normalize(aa["vn"])
                p.attrs = {"theta": aa["theta"], "z0": aa["z0"], "panel": np.full(len(vv), k)}
                pieces.append(p)
                k += 1
        return pieces

    return solid_parts("corset", "Corset", v, t, a["S"][:, None].repeat(2, 1), w, D, THICK,
                       ("corset", "patch:lining", "patch:edge"), ("leather", "lining", "edge"), rim_segments=RIM,
                       attrs={"theta": a["theta"], "z0": a["z0"], "S": a["S"], "V": a["V"]}, split_outer=split)


def surface_points(tree, targets, toward):
    out = []
    nrm = []
    for p in targets:
        q, n, face, dist = hl_blend.nearest(tree, np.array([p]))
        out.append(q[0])
        n0 = n[0] if np.dot(n[0], toward) > 0 else -n[0]
        nrm.append(n0)
    return np.array(out), np.array(nrm)


def torus(center, normal, up, major, minor, seg_major=14, seg_minor=6):
    n = hl_geom.normalize(normal)
    u = hl_geom.normalize(up - n * np.dot(up, n))
    w_ = np.cross(n, u)
    verts = []
    uv = []
    for i in range(seg_major):
        a = 2 * np.pi * i / seg_major
        c = np.cos(a) * u + np.sin(a) * w_
        for j in range(seg_minor):
            b = 2 * np.pi * j / seg_minor
            p = center + c * (major + minor * np.cos(b)) + n * (minor * np.sin(b))
            verts.append(p)
            uv.append([i / seg_major * 2 * np.pi * major, j / seg_minor * 2 * np.pi * minor])
    tris = []
    for i in range(seg_major):
        for j in range(seg_minor):
            a0 = i * seg_minor + j
            a1 = ((i + 1) % seg_major) * seg_minor + j
            b0 = i * seg_minor + (j + 1) % seg_minor
            b1 = ((i + 1) % seg_major) * seg_minor + (j + 1) % seg_minor
            tris.append([a0, a1, b1])
            tris.append([a0, b1, b0])
    return np.array(verts), np.array(tris), np.array(uv)


def merge(meshes):
    vs, ts, uvs = [], [], []
    off = 0
    for v, t, uv in meshes:
        vs.append(v)
        ts.append(t + off)
        uvs.append(uv)
        off += len(v)
    return np.concatenate(vs), np.concatenate(ts), np.concatenate(uvs)


def lacing(surface_v, surface_t, a, body):
    tree = hl_blend.bvh(surface_v, surface_t)
    top = hl_fit.sym_curve(TOP)(0.0)
    bot = hl_fit.sym_curve(BOT)(0.0)
    zs = np.linspace(bot + 1.25, top - 1.0, EYELETS)
    eyes = {}
    for side in (-1, 1):
        targets = np.array([[side * (GAP + 0.42), 12.0, z] for z in zs])
        guess = []
        for p in targets:
            o = np.array([side * (GAP + 0.42), 0.0, p[2]])
            loc, nrm, face, dist = hl_blend.raycast(tree, np.array([o]), np.array([[0.0, 1.0, 0.0]]), 40.0)
            guess.append(loc[0])
        pts, nrm = surface_points(tree, np.array(guess), np.array([0.0, 1.0, 0.0]))
        eyes[side] = (pts, nrm)
    rings = []
    for side in (-1, 1):
        for p, n in zip(*eyes[side]):
            rings.append(torus(p + n * 0.035, n, np.array([0, 0, 1.0]), 0.2, 0.065))
    ev, et, euv = merge(rings)
    cords = []
    radius = 0.075
    L, R = eyes[-1], eyes[1]
    paths = []
    for i in range(EYELETS - 1):
        for (A, B, lift) in ((L, R, 0.17), (R, L, 0.07)):
            p0 = A[0][i] - A[1][i] * 0.02
            p1 = B[0][i + 1] - B[1][i + 1] * 0.02
            mid = 0.5 * (p0 + p1)
            nmid = hl_geom.normalize(A[1][i] + B[1][i + 1])
            ctrl = [p0, p0 + A[1][i] * 0.12, mid + nmid * lift, p1 + B[1][i + 1] * 0.12, p1]
            paths.append(hl_geom.catmull_rom(np.array(ctrl), 6))
    p0 = L[0][0] - L[1][0] * 0.02
    p1 = R[0][0] - R[1][0] * 0.02
    nm = hl_geom.normalize(L[1][0] + R[1][0])
    paths.append(hl_geom.catmull_rom(np.array([p0, p0 + L[1][0] * 0.1, 0.5 * (p0 + p1) + nm * 0.1,
                                               p1 + R[1][0] * 0.1, p1]), 6))
    knot_c = 0.5 * (L[0][-1] + R[0][-1]) + hl_geom.normalize(L[1][-1] + R[1][-1]) * 0.32 + np.array([0, 0, 0.35])
    for A in (L, R):
        p0 = A[0][-1] - A[1][-1] * 0.02
        paths.append(hl_geom.catmull_rom(np.array([p0, p0 + A[1][-1] * 0.15, 0.5 * (p0 + knot_c) +
                                                   A[1][-1] * 0.12, knot_c]), 6))
    nk = hl_geom.normalize(L[1][-1] + R[1][-1])
    for k, side in enumerate((-1, 1)):
        start = knot_c
        ctrl = [start, start + np.array([side * 0.35, 0.05, -0.6]) + nk * 0.08,
                start + np.array([side * 0.55, 0.0, -1.6]) + nk * 0.12,
                start + np.array([side * 0.5, 0.0, -2.6]) + nk * 0.1]
        paths.append(hl_geom.catmull_rom(np.array(ctrl), 8))
    for path in paths:
        v, t, uv, _ = hl_geom.sweep(path, np.full(len(path), radius), 7)
        cords.append((v, t, uv))
    loops = []
    for side in (-1, 1):
        ang = np.linspace(0, 2 * np.pi, 13)
        lp = np.array([knot_c + side * 0.3 * np.array([1, 0, 0]) * (1 - np.cos(a)) * 0.9
                       + np.array([0, 0, 0.28]) * np.sin(a) + nk * 0.05 * np.sin(a) for a in ang])
        v, t, uv, _ = hl_geom.sweep(lp, np.full(len(lp), radius * 1.1), 7)
        loops.append((v, t, uv))
    kv, kt, kuv, _ = hl_geom.sweep(np.array([knot_c - np.array([0.16, 0, 0]), knot_c,
                                             knot_c + np.array([0.16, 0, 0])]),
                                   np.array([0.12, 0.17, 0.12]), 10, cap_start=True, cap_end=True)
    cv, ct, cuv = merge(cords + loops + [(kv, kt, kuv)])
    tips = []
    for path in paths[-2:]:
        end = path[-1]
        d = hl_geom.normalize(path[-1] - path[-2])
        tv, tt, tuv, _ = hl_geom.sweep(np.array([end - d * 0.05, end + d * 0.35]), np.array([0.085, 0.07]), 8,
                                       cap_end=True)
        tips.append((tv, tt, tuv))
    av, at, auv = merge(tips)
    return (ev, et, euv), (cv, ct, cuv), (av, at, auv), knot_c


def modesty_panel(proxy, offset_fn):
    top = hl_fit.sym_curve(TOP)(0.0) - 0.35
    bot = hl_fit.sym_curve(BOT)(0.0) + 0.9
    zs = np.arange(bot, top + 0.01, 0.25)
    xs = np.linspace(-1.55, 1.55, 15)
    pts = []
    for z in zs:
        for x in xs:
            pts.append([x, 0.0, z])
    pts = np.array(pts)
    loc, nrm, face, dist = hl_blend.raycast(proxy.tree, pts, np.tile([0, 1.0, 0], (len(pts), 1)), 30.0)
    off = offset_fn(zs.repeat(len(xs))) - 0.07
    loc = loc + nrm * off[:, None]
    t = hl_geom.grid_tris(len(zs), len(xs))[:, ::-1].copy()
    uv = np.stack([pts[:, 0] - pts[:, 0].min(), pts[:, 2] - pts[:, 2].min()], 1)
    return loc, t, uv


def leg_open(points):
    out = np.zeros(len(points))
    zo = hl_fit.sym_curve(LEG_OPEN)
    zi = hl_fit.sym_curve(LEG_OPEN_INNER)
    for side in (-1, 1):
        hip = HIP_L * np.array([side * -1.0, 1, 1])
        knee = KNEE_L * np.array([side * -1.0, 1, 1])
        ax = hl_geom.normalize(knee - hip)
        r = points - hip
        r = r - np.outer((r * ax).sum(1), ax)
        front = np.array([0, 1.0, 0])
        outer = np.array([side * 1.0, 0, 0])
        phi = np.arctan2((r * outer).sum(1), (r * front).sum(1))
        zc = np.where(phi >= 0, zo(phi), zi(phi))
        sel = np.sign(points[:, 0]) == side
        if side == 1:
            sel |= points[:, 0] == 0
        out[sel] = (points[:, 2] - zc)[sel]
    return out


def capped_proxy(proxy):
    t = proxy.inv[proxy.orig_t["3BA"]]
    v, w = proxy.v, proxy.w
    D = proxy.D
    nv, nw, nD, fans = [], [], [], []
    normals = hl_geom.vertex_normals(v, t)
    for loop in hl_geom.boundary_loops(t):
        c = v[loop].mean(0)
        if abs(c[0]) > 3.0 or not 55.0 < c[2] < 71.0:
            continue
        k = len(v) + len(nv)
        nv.append(c)
        nw.append(w[loop].mean(0))
        nD.append(D[loop].mean(0))
        f = np.stack([loop, np.roll(loop, -1), np.full(len(loop), k)], 1)
        fn = np.cross(v[f[:, 1]] - c, v[f[:, 0]] - c).sum(0)
        if np.dot(fn, normals[loop].sum(0)) > 0:
            f = f[:, [1, 0, 2]]
        fans.append(f)
    cp = object.__new__(type(proxy))
    cp.__dict__.update(proxy.__dict__)
    cp.v = np.vstack([v] + [np.array(nv)])
    cp.t = np.vstack([t] + fans)
    cp.w = np.vstack([w, np.array(nw)])
    cp.D = np.concatenate([D, np.array(nD, dtype=D.dtype)])
    cp.n = hl_geom.vertex_normals(cp.v, cp.t)
    cp.tree = hl_blend.bvh(cp.v, cp.t)
    return cp


def briefs_surface(body, proxy):
    v, t = proxy.v, proxy.t
    keep = (v[t][:, :, 2].max(1) < 76.5) & (v[t][:, :, 2].min(1) > 55.0)
    t = t[keep]
    v, t, _, used = hl_geom.compact(v, t)
    th = np.arctan2(v[:, 0], v[:, 1] + 3.5)
    f = np.minimum(hl_fit.sym_curve(BRIEF_TOP)(th) - v[:, 2], leg_open(v))
    tri_ok = (f[t] > -2.0).any(1)
    v, t, _, used = hl_geom.compact(v, t[tri_ok])
    f = f[used]
    v, t, _ = hl_geom.iso_cut(v, t, f)
    v = hl_fit.drape(v, t, proxy, BRIEF_OFF, iters=40, lam=0.4)
    v = hl_geom.smooth_boundary(v, t, iters=12, lam=0.5)
    v = hl_fit.push_out(v, proxy, BRIEF_OFF, iters=2)
    return v, t


def garment_bones(bones):
    return np.array([0.0 if any(k in b for k in INTIMATE) else 1.0 for b in bones])


def cleft_mask(v):
    return np.clip((3.5 - np.abs(v[:, 0])) / 1.5, 0, 1) * np.clip((-3.0 - v[:, 1]) / 1.5, 0, 1) * \
        np.clip((71.5 - v[:, 2]) / 1.5, 0, 1)


def bridged_proxy(proxy, iters=120):
    v = proxy.v
    cleft = np.clip((4.5 - np.abs(v[:, 0])) / 1.5, 0, 1) * np.clip((-2.0 - v[:, 1]) / 1.5, 0, 1)
    legs = np.clip((6.0 - np.abs(v[:, 0])) / 2.0, 0, 1) * np.clip((68.0 - v[:, 2]) / 2.0, 0, 1)
    k = np.maximum(cleft, legs) * np.clip((v[:, 2] - 50.0) / 2.0, 0, 1) * np.clip((77.0 - v[:, 2]) / 2.0, 0, 1)
    pairs = hl_geom.sub_pairs(hl_geom.neighbor_pairs(proxy.t, len(v)), k > 0)
    rows = pairs[3]
    D = proxy.D.reshape(len(v), -1).astype(np.float64)
    for _ in range(iters):
        D[rows] += 0.5 * k[rows, None] * hl_geom.laplacian(D, pairs)[rows]
    bp = object.__new__(type(proxy))
    bp.__dict__.update(proxy.__dict__)
    bp.D = D.reshape(proxy.D.shape).astype(np.float32)
    return bp


def briefs_layer(body, cp):
    v, t = briefs_surface(body, cp)
    v = hl_fit.push_out(hl_fit.fill_concave(v, t, cleft_mask(v)), cp, BRIEF_OFF, iters=2)
    w, D = bind(v, t, bridged_proxy(cp), smooth_iters=4, edge_body=body, bone_mask=garment_bones(cp.bones))
    w = hl_fit.beneath_weights(v, t, body, garment_bones(cp.bones))
    inner = np.clip((hl_geom.edge_distance(v, t) - 0.8) / 1.2, 0, 1)
    D, _ = hl_fit.fold_safe_diffs(v, t, D, cp.slider_names, skip=hl_export.NO_GARMENT,
                                  damp=inner * inner * (3 - 2 * inner))
    return v, t, w, hl_fit.garment_conform(v, D, cp)


def briefs_parts(v, t, w, D):
    parts = []
    for k, sgn in enumerate((1, -1)):
        f = sgn * (v[:, 1] + 3.0)
        vv, tt, aa = hl_geom.iso_cut(v, t, f, {"w": w, "D": D})
        uv = hl_blend.unwrap(vv, tt)
        parts += solid_parts("briefs_%d" % k, "Corset", vv, tt, uv, aa["w"], aa["D"], 0.22,
                             ("briefs_%d" % k, "patch:lining", "patch:edge"), ("leather", "lining", "edge"),
                             rim_segments=RIM)
    return parts


def covered_mask(body_v, body_n, cover_parts, out_dist=4.0, in_dist=1.6):
    vs, ts = [], []
    off = 0
    for p in cover_parts:
        vs.append(p.v)
        ts.append(p.t + off)
        off += len(p.v)
    tree = hl_blend.bvh(np.concatenate(vs), np.concatenate(ts))
    loc, nrm, face, dist = hl_blend.raycast(tree, body_v + body_n * 0.01, body_n, out_dist)
    outward = np.isfinite(dist) & ((nrm * body_n).sum(1) > 0)
    loc, nrm, face, dist = hl_blend.raycast(tree, body_v - body_n * 0.01, -body_n, in_dist)
    inward = np.isfinite(dist) & ((nrm * body_n).sum(1) > 0)
    return outward | inward


def bridge_gap(body_v, body_n, cover_parts, reach=4.0):
    vs, ts = [], []
    off = 0
    for p in cover_parts:
        vs.append(p.v)
        ts.append(p.t + off)
        off += len(p.v)
    tree = hl_blend.bvh(np.concatenate(vs), np.concatenate(ts))
    loc, nrm, face, dist = hl_blend.raycast(tree, body_v + body_n * 0.005, body_n, reach)
    ok = np.isfinite(dist) & ((nrm * body_n).sum(1) > 0)
    return np.where(ok, dist, 0.0)


def trim_body(body, covered, rings=2, min_island=60, gap=None, max_gap=0.4):
    pairs = hl_geom.neighbor_pairs(body.t, len(body.v))
    free = ~covered
    lab = hl_geom.components(len(body.v), pairs[0], pairs[1], free)
    sizes = np.bincount(lab[free], minlength=len(body.v))
    keep = free & (sizes[lab] >= min_island)
    for _ in range(rings):
        grow = keep.copy()
        np.logical_or.at(grow, pairs[0], keep[pairs[1]])
        keep = grow
    deep = covered & (gap > max_gap) if gap is not None else np.zeros(len(body.v), dtype=bool)
    removed = ~keep
    tri = body.inv[body.orig_t["3BA"]]
    return ~(removed[tri].all(1) | deep[tri].any(1))


def follow_skin(parts, body, skin_tris, near=0.75, far=1.5, mats=("leather", "fabric", "lining", "edge"),
                bone_mask=None):
    tree = hl_blend.bvh(body.v, skin_tris)
    bw = body.w if bone_mask is None else body.w * bone_mask[None, :]
    for p in parts:
        if p.mat not in mats:
            continue
        loc, nrm, face, dist = hl_blend.nearest(tree, p.v)
        tri = skin_tris[face]
        bc = hl_blend.barycentric(loc, body.v[tri[:, 0]], body.v[tri[:, 1]], body.v[tri[:, 2]])
        ws = np.einsum("nk,nkb->nb", bc, bw[tri])
        ws /= np.maximum(ws.sum(1, keepdims=True), 1e-9)
        k = np.clip((far - dist) / (far - near), 0, 1)
        k = k * k * (3 - 2 * k)
        p.w = p.w * (1 - k)[:, None] + ws * k[:, None]
        p.w /= np.maximum(p.w.sum(1, keepdims=True), 1e-9)


def build(body, proxy=None, log=print):
    import time
    t0 = time.time()
    proxy = proxy or make_proxy(body)
    log("proxy %.1fs" % (time.time() - t0))
    v, t, a, seam_S = corset_surface(proxy)
    log("surface %.1fs" % (time.time() - t0))
    w, D = bind(v, t, proxy, smooth_iters=3, edge_body=body)
    log("bind %.1fs" % (time.time() - t0))
    parts = corset_panels(v, t, a, seam_S, w, D)
    log("panels %.1fs" % (time.time() - t0))
    (ev, et, euv), (cv, ct, cuv), (av, at, auv), knot = lacing(v, t, a, body)
    for name, mv, mt, muv, isl, mat in (("eyelets", ev, et, euv, "patch:brass", "brass"),
                                        ("lacing", cv, ct, cuv, "patch:cord", "cord"),
                                        ("aglets", av, at, auv, "patch:brass", "brass")):
        mw, mD = edge_blend(mv, v, t, w, D, GAP + 0.4)
        parts.append(Part(name, "CorsetBrass" if mat == "brass" else "Corset", mv, mt, muv, isl, mat, mw, mD))
    mv, mt, muv = modesty_panel(proxy, offset_at)
    mw, mD = edge_blend(mv, v, t, w, D, 1.55)
    parts.append(Part("modesty", "Corset", mv, mt, muv, "modesty", "fabric", mw, mD))
    log("details %.1fs" % (time.time() - t0))
    parts += briefs_parts(*briefs_layer(body, capped_proxy(proxy)))
    log("briefs %.1fs" % (time.time() - t0))
    outer = [p for p in parts if p.mat in ("leather", "fabric") and not p.name.endswith(("_in", "_rim"))]
    covered = covered_mask(body.v, body.n, outer)
    briefs = [p for p in outer if p.name.startswith("briefs")]
    keep_tris = trim_body(body, covered, gap=bridge_gap(body.v, body.n, briefs))
    log("trim %.1fs" % (time.time() - t0))
    tri = body.inv[body.orig_t["3BA"]]
    kept = np.zeros(len(body.v), dtype=bool)
    kept[np.unique(tri[keep_tris])] = True
    under = np.nonzero(kept & covered)[0]
    uv_names = {a["name"] for a in body.slider_attrs if a.get("uv") == "true"}
    for p in outer:
        p.sliders = list(proxy.slider_names)
    D_conf, rows = hl_fit.conform_diffs(body, body.D, under, outer, uv_names)
    vs = np.concatenate([p.v for p in outer])
    ts = np.concatenate([p.t + o for p, o in zip(outer, np.cumsum([0] + [len(p.v) for p in outer[:-1]]))])
    loc, nrm, face, dist = hl_blend.nearest(hl_blend.bvh(vs, ts), body.v)
    near = np.nonzero(kept & (dist < 1.2))[0]
    samples = [vals for _, vals in hl_preset.corpus(body.slider_attrs)]
    D_conf, hist = hl_fit.conform_samples(body, D_conf, tri[keep_tris], near, outer, samples,
                                          skip=hl_export.NO_GARMENT, rounds=12)
    under = np.union1d(under, near)
    log("conform %d skin verts, pokes per preset %s %.1fs" % (len(under), " ".join("%.1f" % h for h in hist),
                                                               time.time() - t0))
    follow_skin([p for p in parts if not p.name.startswith("briefs")], body, tri[keep_tris])
    for p in parts:
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        if p.mat == "brass":
            p.shape = "CorsetBrass"
        elif p.name.startswith("briefs"):
            p.shape = "BriefsLining" if p.name.endswith("_in") else "Briefs"
        elif p.name.endswith("_in"):
            p.shape = "CorsetLining"
        else:
            p.shape = "Corset"
    return parts, keep_tris, proxy, D_conf, under

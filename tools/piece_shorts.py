import numpy as np

import hl_blend
import hl_export
import hl_fit
import hl_geom
import piece_corset
from hl_parts import Part, bind, solid_parts

BAND_TOP = [(0, 74.3), (0.8, 74.6), (1.57, 75.0), (2.4, 75.5), (np.pi, 75.7)]
BAND_H = 2.3
BAND_OFF = 0.62
BELT_H = 0.95
BUCKLE_THETA = -0.72
AXIS_Y = -3.2
OFFSET = 0.45
CROTCH_EXTRA = 0.4
MIN_REAL = 0.42
THICK = 0.14
HEM_W = 0.9
LEG_OUT = [(0, 58.4), (0.78, 58.2), (1.57, 57.8), (2.36, 57.0), (np.pi, 56.4)]
LEG_IN = [(0, 58.4), (0.78, 59.0), (1.57, 59.4), (2.36, 58.6), (np.pi, 56.4)]


def band(proxy):
    top = hl_fit.sym_curve(BAND_TOP)
    zs = np.arange(71.0, 76.21, 0.3)
    cy = hl_fit.torso_axis(proxy, zs)
    p, t, theta, zz = hl_fit.ring_grid(proxy, zs, 200, cy, theta0=np.pi)
    p = hl_fit.drape(p, t, proxy, BAND_OFF, iters=30, lam=0.5)
    rows, cols = len(zs), 200
    g = p.reshape(rows, cols, 3)
    ring = np.linalg.norm(np.diff(np.concatenate([g, g[:, :1]], 1), axis=1), axis=2)
    S = np.concatenate([np.zeros((rows, 1)), np.cumsum(ring, 1)[:, :-1]], 1).ravel()
    col = np.linalg.norm(np.diff(g, axis=0), axis=2)
    V = np.concatenate([np.zeros((1, cols)), np.cumsum(col, 0)], 0).ravel()
    th = np.arctan2(np.sin(theta), np.cos(theta))
    a = {"theta": th, "z0": zz, "S": S, "V": V}
    p, t, a = hl_geom.open_ring_seam(p, t, np.arange(rows) * cols, a, "S", np.cumsum(ring, 1)[:, -1])
    v, tt, a = hl_geom.iso_cut(p, t, top(a["theta"]) - a["z0"], a)
    v, tt, a = hl_geom.iso_cut(v, tt, a["z0"] - (top(a["theta"]) - BAND_H), a)
    back = np.mod(a["theta"], 2 * np.pi)
    v, tt, a = hl_geom.iso_cut(v, tt, np.abs(back - np.pi) - 0.004, a)
    return v, tt, a


def buckle(center, normal, up, w=2.5, h=1.95, bar=0.3):
    n = hl_geom.normalize(normal)
    u = hl_geom.normalize(up - n * np.dot(up, n))
    r = np.cross(u, n)
    path = []
    corner = 0.5
    for cx, cy, a0 in ((w / 2 - corner, h / 2 - corner, 0), (-w / 2 + corner, h / 2 - corner, 90),
                       (-w / 2 + corner, -h / 2 + corner, 180), (w / 2 - corner, -h / 2 + corner, 270)):
        for k in range(7):
            ang = np.radians(a0 + 90 * k / 6)
            path.append((cx + corner * np.cos(ang), cy + corner * np.sin(ang)))
    path = np.array(path)
    pts = center + np.outer(path[:, 0], r) + np.outer(path[:, 1], u)
    pts = np.vstack([pts, pts[:1]])
    v1, t1, uv1, _ = hl_geom.sweep(pts, np.full(len(pts), bar * 0.5), 8, profile=None, up=n)
    prong_a = center + r * (-w / 2 + 0.05) + n * 0.05
    prong_b = center + r * (w * 0.18) + n * 0.12
    v2, t2, uv2, _ = hl_geom.sweep(np.array([prong_a, 0.5 * (prong_a + prong_b) + n * 0.1, prong_b]),
                                   np.array([0.1, 0.105, 0.085]), 8, cap_start=True, cap_end=True)
    return [(v1, t1, uv1), (v2, t2, uv2)]


def belt_center(theta):
    return hl_fit.sym_curve(BAND_TOP)(theta) - BAND_H * 0.5 - 0.35 * np.cos(theta - BUCKLE_THETA) + 0.1


def belt(proxy):
    zs = np.arange(71.5, 76.01, 0.25)
    cy = hl_fit.torso_axis(proxy, zs)
    p, t, theta, zz = hl_fit.ring_grid(proxy, zs, 200, cy, theta0=BUCKLE_THETA + np.pi)
    p = hl_fit.drape(p, t, proxy, BAND_OFF + 0.22, iters=30, lam=0.5)
    th = np.arctan2(np.sin(theta), np.cos(theta))
    rows, cols = len(zs), 200
    g = p.reshape(rows, cols, 3)
    ring = np.linalg.norm(np.diff(np.concatenate([g, g[:, :1]], 1), axis=1), axis=2)
    S = np.concatenate([np.zeros((rows, 1)), np.cumsum(ring, 1)[:, :-1]], 1).ravel()
    a = {"theta": th, "z0": zz, "S": S, "V": zz}
    p, t, a = hl_geom.open_ring_seam(p, t, np.arange(rows) * cols, a, "S", np.cumsum(ring, 1)[:, -1])
    v, tt, a = hl_geom.iso_cut(p, t, belt_center(a["theta"]) + BELT_H / 2 - a["z0"], a)
    v, tt, a = hl_geom.iso_cut(v, tt, a["z0"] - (belt_center(a["theta"]) - BELT_H / 2), a)
    back = np.mod(a["theta"] - BUCKLE_THETA, 2 * np.pi)
    v, tt, a = hl_geom.iso_cut(v, tt, np.abs(back - np.pi) - 0.003, a)
    return v, tt, a


def waist_parts(body, proxy):
    bv, bt, ba = band(proxy)
    bw, bD = bind(bv, bt, proxy, smooth_iters=3, edge_body=body)
    parts = solid_parts("band", "Shorts", bv, bt, np.stack([ba["S"], ba["V"]], 1), bw, bD, 0.26,
                        ("band", "patch:lining", "patch:edge"), ("leather", "lining", "edge"), rim_segments=2)
    lv, lt, la = belt(proxy)
    lw, lD = bind(lv, lt, proxy, smooth_iters=3)
    parts += solid_parts("belt", "Shorts", lv, lt, np.stack([la["S"], la["V"]], 1), lw, lD, 0.2,
                         ("belt", "patch:lining", "patch:edge"), ("leather", "lining", "edge"), rim_segments=2)
    tree = hl_blend.bvh(lv, lt)
    zc = float(belt_center(np.array([BUCKLE_THETA]))[0])
    o = np.array([0.0, AXIS_Y, zc])
    d = np.array([np.sin(BUCKLE_THETA), np.cos(BUCKLE_THETA), 0.0])
    loc, nrm, face, dist = hl_blend.raycast(tree, o[None], d[None], 40.0)
    c = loc[0] + nrm[0] * 0.12
    for k, (mv, mt, muv) in enumerate(buckle(c, nrm[0], np.array([0, 0, 1.0]))):
        mw, mD = bind(mv, mt, proxy, rigid=True)
        parts.append(Part("buckle_%d" % k, "ShortsBrass", mv, mt, muv, "patch:brass", "brass", mw, mD))
    return parts


def subdivide(v, t):
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    mid = len(v) + inv.reshape(3, -1).T
    nv = np.vstack([v, 0.5 * (v[ue[:, 0]] + v[ue[:, 1]])])
    a, b, c = t.T
    m01, m12, m20 = mid.T
    nt = np.vstack([np.stack([a, m01, m20], 1), np.stack([m01, b, m12], 1), np.stack([m20, m12, c], 1),
                    np.stack([m01, m12, m20], 1)])
    return nv, nt


def leg_frame(points):
    phi = np.zeros(len(points))
    zc = np.zeros(len(points))
    zo = hl_fit.sym_curve(LEG_OUT)
    zi = hl_fit.sym_curve(LEG_IN)
    for side in (-1, 1):
        hip = piece_corset.HIP_L * np.array([side * -1.0, 1, 1])
        knee = piece_corset.KNEE_L * np.array([side * -1.0, 1, 1])
        ax = hl_geom.normalize(knee - hip)
        r = points - hip
        r = r - np.outer((r * ax).sum(1), ax)
        ph = np.arctan2((r * np.array([side * 1.0, 0, 0])).sum(1), r[:, 1])
        sel = np.sign(points[:, 0]) == side
        if side == 1:
            sel |= points[:, 0] == 0
        phi[sel] = ph[sel]
        zc[sel] = np.where(ph >= 0, zo(ph), zi(ph))[sel]
    return phi, points[:, 2] - zc


def cut_field(v):
    th = np.arctan2(v[:, 0], v[:, 1] - AXIS_Y)
    top = hl_fit.sym_curve(BAND_TOP)(th) - BAND_H + 0.7
    return np.minimum(top - v[:, 2], leg_frame(v)[1])


def relax(v, t, iters=10, lam=0.5):
    pairs = hl_geom.neighbor_pairs(t, len(v))
    p = np.array(v, dtype=np.float64)
    for i in range(iters):
        n = hl_geom.vertex_normals(p, t)
        dp = lam * hl_geom.laplacian(p, pairs)
        p += dp - (dp * n).sum(1)[:, None] * n
    return p


def crotch_mask(v, width=3.5, top=67.0):
    return np.clip((width - np.abs(v[:, 0])) / 1.5, 0, 1) * np.clip((top - v[:, 2]) / 2.0, 0, 1) * \
        np.clip((v[:, 1] + 6.0) / 1.5, 0, 1)


def bridge_mask(v):
    crotch = np.clip((6.0 - np.abs(v[:, 0])) / 2.0, 0, 1) * np.clip((67.5 - v[:, 2]) / 2.0, 0, 1)
    return np.maximum(crotch, piece_corset.cleft_mask(v))


def shorts_surface(proxy, real):
    v, t = proxy.v, proxy.t
    arm = proxy.mask_bones(("UpperArm", "Forearm", "Hand", "Clavicle", "Finger"))
    zt = v[t][:, :, 2]
    keep = (zt.max(1) < 78.0) & (zt.min(1) > 48.0) & (arm[t].max(1) < 0.05)
    v, t, _, _ = hl_geom.compact(v, t[keep])
    ok = (cut_field(v)[t] > -2.5).any(1)
    v, t, _, _ = hl_geom.compact(v, t[ok])
    v, t = subdivide(v, t)
    off = OFFSET + CROTCH_EXTRA * crotch_mask(v)
    v = hl_fit.drape(v, t, proxy, off, iters=60, lam=0.5)
    v = hl_fit.fill_concave(v, t, bridge_mask(v))
    for _ in range(3):
        v = hl_fit.push_out(v, proxy, off, iters=2)
        v = hl_fit.push_out(v, real, MIN_REAL, iters=2)
        v = relax(v, t, iters=10)
    v = hl_fit.push_out(hl_fit.push_out(v, proxy, off, iters=2), real, MIN_REAL, iters=2)
    v = smooth_crotch(v, t, real)
    v, t, _ = hl_geom.iso_cut(v, t, cut_field(v))
    return v, t


def smooth_crotch(v, t, real, rounds=4, iters=40):
    k = crotch_mask(v, width=4.5, top=68.0)
    sel = k > 0
    pairs = hl_geom.neighbor_pairs(t, len(v))
    v = np.array(v, dtype=np.float64)
    for _ in range(rounds):
        for _ in range(iters):
            v[sel] += (0.5 * k[:, None] * hl_geom.laplacian(v, pairs))[sel]
        sd, _, _, _ = real.signed_distance(v)
        need = np.clip(MIN_REAL - sd, 0, None)
        for _ in range(30):
            need = np.maximum(need, need + 0.5 * hl_geom.laplacian(need, pairs))
        v += hl_geom.vertex_normals(v, t) * need[:, None]
    return v


def layer_over(points, w, D, under, near=1.2, far=2.2):
    uv, ut, uw, uD = under
    tree = hl_blend.bvh(uv, ut)
    loc, nrm, face, dist = hl_blend.nearest(tree, points)
    tri = ut[face]
    bc = hl_blend.barycentric(loc, uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]])
    k = np.clip((far - dist) / (far - near), 0, 1)
    k = k * k * (3 - 2 * k)
    lw = np.einsum("nk,nkb->nb", bc, uw[tri])
    lD = np.einsum("nk,nksd->nsd", bc, uD[tri])
    w = w * (1 - k)[:, None] + lw * k[:, None]
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    return w, D * (1 - k)[:, None, None] + lD * k[:, None, None]


def split_panels(outer):
    pieces = []
    k = 0
    for sx in (1, -1):
        for sf in (1, -1):
            aa = dict(outer.attrs, w=outer.w, D=outer.D)
            vv, tt, aa = hl_geom.iso_cut(outer.v, outer.t, sx * aa["xs"], aa)
            vv, tt, aa = hl_geom.iso_cut(vv, tt, sf * aa["cphi"], aa)
            for name, sign, mat, isl in (("", 1, "fabric", "shorts_%d"), ("_hem", -1, "leather", "hem_s%d")):
                v2, t2, a2 = hl_geom.iso_cut(vv, tt, sign * (aa["legf"] - HEM_W), aa)
                if len(t2) == 0:
                    continue
                p = Part("shorts%s_%d" % (name, k), outer.shape, v2, t2, hl_blend.unwrap(v2, t2), isl % k, mat,
                         a2["w"], a2["D"])
                p.normals = hl_geom.normalize(a2["vn"])
                p.attrs = {}
                pieces.append(p)
            k += 1
    return pieces


def build(body, proxy=None, log=print):
    import time
    t0 = time.time()
    parts = waist_parts(body, proxy)
    cp = piece_corset.capped_proxy(proxy)
    briefs = piece_corset.briefs_layer(body, cp)
    v, t = shorts_surface(cp, proxy)
    log("shorts surface %d verts %.1fs" % (len(v), time.time() - t0))
    mask = piece_corset.garment_bones(proxy.bones)
    w, D = bind(v, t, piece_corset.bridged_proxy(cp), smooth_iters=3, bone_mask=mask)
    D = hl_fit.standoff_smooth(v, t, D, cp, start=0.6)
    pairs = hl_geom.neighbor_pairs(t, len(v))
    for _ in range(3):
        D = hl_fit.garment_conform(v, hl_fit.garment_conform(v, D, cp), proxy)
        D = hl_geom.smooth(D.reshape(len(v), -1), pairs, 6, 0.5).reshape(D.shape)
    _, D = layer_over(v, w, D, briefs)
    w = hl_fit.beneath_weights(v, t, body, mask)
    D, hist = hl_fit.fold_safe_diffs(v, t, D, proxy.slider_names, skip=hl_export.NO_GARMENT)
    under = hl_fit.layer_body(briefs[0], briefs[1], briefs[3])
    D = hl_fit.garment_conform(v, D, under, slack_margin=0.5, smooth_tris=t)
    for layer in (cp, proxy):
        D = hl_fit.garment_conform(v, D, layer, smooth_tris=t)
    D, hist2 = hl_fit.fold_safe_diffs(v, t, D, proxy.slider_names, skip=hl_export.NO_GARMENT, rounds=4)
    k = np.clip(crotch_mask(v, width=3.2, top=66.5) * 1.6, 0, 1)
    k = k * k * (3 - 2 * k)
    D = hl_fit.harmonic_fill(t, D, k)
    log("shorts diffs, folds per sample %s / %s" % (" ".join("%.0f" % h for h in hist),
                                                     " ".join("%.0f" % h for h in hist2)))
    phi, legf = leg_frame(v)
    shorts = solid_parts("shorts", "Shorts", v, t, np.zeros((len(v), 2)), w, D, THICK,
                         ("shorts", "patch:lining", "patch:edge"), ("fabric", "lining", "edge"), rim_segments=2,
                         attrs={"xs": v[:, 0], "cphi": np.cos(phi), "legf": legf}, split_outer=split_panels)
    piece_corset.follow_skin(parts, body, body.t, near=0.6, far=1.4, bone_mask=mask)
    parts += shorts
    log("shorts parts %.1fs" % (time.time() - t0))
    for p in parts:
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        if p.mat == "brass":
            p.shape = "ShortsBrass"
        else:
            p.shape = "ShortsLining" if p.name.endswith("_in") else "Shorts"
    return parts, None

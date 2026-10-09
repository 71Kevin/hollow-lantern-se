import numpy as np

import hl_blend
import hl_fit
import hl_geom
import piece_corset
from hl_parts import Part, bind, solid_parts

Z_LO = 112.35
Z_HI = 113.4
OFF = 0.22
OFF_TOP = 0.3


def pumpkin_charm(top, down=(0, 0, -1.0), scale=0.42, lobes=8):
    phis = np.linspace(0, 2 * np.pi, 33)
    thetas = np.linspace(0, np.pi, 17)
    verts = []
    for th in thetas:
        for ph in phis:
            r = 1.0 + 0.09 * np.cos(lobes * ph)
            x = np.sin(th) * np.cos(ph) * r
            y = np.sin(th) * np.sin(ph) * r
            z = np.cos(th) * 0.8
            verts.append([x, y, z])
    v = np.array(verts) * scale
    v[:, 2] -= scale * 0.8
    ez = -hl_geom.normalize(np.asarray(down, dtype=np.float64))
    ex = hl_geom.normalize(np.cross([0, 1.0, 0], ez)) if abs(ez[1]) < 0.9 else np.array([1.0, 0, 0])
    ey = np.cross(ez, ex)
    t = hl_geom.grid_tris(len(thetas), len(phis))
    uv = np.stack([np.tile(phis, len(thetas)), np.repeat(thetas, len(phis))], 1) * scale
    return top + np.outer(v[:, 0], ex) + np.outer(v[:, 1], ey) + np.outer(v[:, 2], ez), t, uv


def ellipse(a_major, a_minor):
    def profile(s, ang):
        return a_major * a_minor / np.sqrt((a_minor * np.cos(ang)) ** 2 + (a_major * np.sin(ang)) ** 2)
    return profile


def sagittal(n):
    n = hl_geom.normalize(np.asarray(n) * np.array([0.0, 1.0, 1.0]))
    return n, hl_geom.normalize(np.array([0, 0, 1.0]) - n * n[2])


def pendant(v, t, skin_tree, bar_r=0.06, ring_r=0.34, jr=0.15, clearance=0.2):
    band = hl_blend.bvh(v, t)
    zc = 0.5 * (Z_LO + Z_HI)
    hz = np.array([Z_HI - 0.2, zc, Z_LO + 0.03])
    loc, nrm, face, dist = hl_blend.raycast(band, np.stack([np.zeros(3), np.full(3, -2.2), hz], 1),
                                            np.tile([0, 1.0, 0], (3, 1)), 20.0)
    frames = [sagittal(k) for k in nrm]
    n_lo, u_lo = frames[2]
    lo = loc[2] - u_lo * 0.03
    anchor = int(np.argmin(np.linalg.norm(v - lo, axis=1)))
    sk, sn, _, _ = hl_blend.raycast(skin_tree, np.array([[0.0, -2.2, Z_LO - 0.7]]), np.array([[0, 1.0, 0]]), 20.0)
    n_p, u_p = sagittal(sn[0])
    x = np.array([1.0, 0, 0])
    bar = lo - u_lo * (bar_r + 0.07) + n_lo * (bar_r - 0.02)
    for alpha in np.radians(np.arange(0, 61, 3)):
        d = -np.cos(alpha) * u_p + np.sin(alpha) * n_p
        nr = np.cos(alpha) * n_p + np.sin(alpha) * u_p
        center = bar + d * ring_r
        ring_v, ring_t, ring_uv = piece_corset.torus(center, nr, -d, ring_r, bar_r, seg_major=24, seg_minor=8)
        jc = center + d * (ring_r + 0.045)
        jump_v, jump_t, jump_uv = piece_corset.torus(jc, x, -d, jr, 0.032, seg_major=16, seg_minor=6)
        tip = jc + d * jr
        charm_top = tip + d * 0.2
        charm_v, charm_t, charm_uv = pumpkin_charm(charm_top, d)
        gap = hl_blend.nearest(skin_tree, np.vstack([jump_v, charm_v]))[3].min()
        if gap >= clearance:
            break
    loop_r = bar_r + 0.055
    ang = np.linspace(np.pi / 2, -1.5 * np.pi, 19)
    loop = bar + np.outer(np.cos(ang), n_lo) * loop_r + np.outer(np.sin(ang), u_lo) * loop_r
    on_band = [loc[k] + frames[k][0] * 0.045 for k in range(3)]
    path = np.vstack(on_band + [loop[2:-1], loc[2] - n_lo * 0.09 - u_lo * 0.02])
    path = hl_geom.resample_polyline(hl_geom.catmull_rom(path, 6), 72)
    tab_v, tab_t, tab_uv, _ = hl_geom.sweep(path, np.ones(len(path)), 12, profile=ellipse(0.2, 0.035),
                                            up=tuple(x), cap_start=True, cap_end=True)
    stem_v, stem_t, stem_uv, _ = hl_geom.sweep(np.array([charm_top - d * 0.04, charm_top - d * 0.1 + nr * 0.02,
                                                         tip - d * 0.01]), np.array([0.065, 0.055, 0.04]), 8,
                                               cap_end=True)
    pieces = [("choker_tab", tab_v, tab_t, tab_uv, "Choker", "patch:edge", "edge"),
              ("choker_ring", ring_v, ring_t, ring_uv, "ChokerBrass", "patch:brass", "brass"),
              ("choker_jump", jump_v, jump_t, jump_uv, "ChokerBrass", "patch:brass", "brass"),
              ("choker_charm", charm_v, charm_t, charm_uv, "Choker", "charm", "pumpkin"),
              ("choker_stem", stem_v, stem_t, stem_uv, "Choker", "patch:edge", "edge")]
    return anchor, pieces


def skin_weights(v, proxy, w_body, D_body, head):
    hv, ht, hw, hbones = head
    bones = list(proxy.bones) + [b for b in hbones if b not in proxy.bones]
    wb = np.zeros((len(v), len(bones)))
    wb[:, :w_body.shape[1]] = w_body
    _, _, _, db = hl_blend.nearest(proxy.tree, v)
    tree = hl_blend.bvh(hv, ht)
    loc, nrm, face, dh = hl_blend.nearest(tree, v)
    tri = ht[face]
    bc = hl_blend.barycentric(loc, hv[tri[:, 0]], hv[tri[:, 1]], hv[tri[:, 2]])
    wh = np.zeros((len(v), len(bones)))
    for j, b in enumerate(hbones):
        wh[:, bones.index(b)] = (bc * hw[tri, j]).sum(1)
    s = np.clip((db - dh) / 0.3 + 0.5, 0, 1)
    s = s * s * (3 - 2 * s)
    w = wb * (1 - s)[:, None] + wh * s[:, None]
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    return w, D_body * (1 - s)[:, None, None], bones


def build(body, proxy=None, log=print):
    import piece_horns
    head = piece_horns.head_skin()
    hv, ht = head[0], head[1]
    tv = np.vstack([proxy.v, hv])
    tt = np.vstack([proxy.t, ht + len(proxy.v)])
    tree = hl_blend.bvh(tv, tt)
    zs = np.arange(Z_LO - 0.2, Z_HI + 0.21, 0.12)
    cols = 120
    th0 = np.pi + np.linspace(0, 2 * np.pi, cols, endpoint=False)
    rows = len(zs)
    d = np.stack([np.sin(th0), np.cos(th0), np.zeros(cols)], 1)
    pts = []
    for z in zs:
        c = np.array([0.0, -2.2, z])
        loc, nrm, face, dist = hl_blend.raycast(tree, c + d * 14.0, -d, 14.0)
        f = np.clip((z - Z_LO) / (Z_HI - Z_LO), 0, 1)
        pts.append(loc + nrm * (OFF + (OFF_TOP - OFF) * f))
    p = np.concatenate(pts)
    t = hl_geom.grid_tris(rows, cols, wrap=True)[:, ::-1].copy()
    theta = np.tile(th0, rows)
    zz = np.repeat(zs, cols)
    pairs = hl_geom.neighbor_pairs(t, len(p))
    p = hl_geom.taubin(p, pairs, iters=6)
    g = p.reshape(rows, cols, 3)
    ring = np.linalg.norm(np.diff(np.concatenate([g, g[:, :1]], 1), axis=1), axis=2)
    S = np.concatenate([np.zeros((rows, 1)), np.cumsum(ring, 1)[:, :-1]], 1).ravel()
    th = np.arctan2(np.sin(theta), np.cos(theta))
    a = {"S": S, "z0": zz, "theta": th}
    p, t, a = hl_geom.open_ring_seam(p, t, np.arange(rows) * cols, a, "S", np.cumsum(ring, 1)[:, -1])
    v, tt, a = hl_geom.iso_cut(p, t, Z_HI - a["z0"], a)
    v, tt, a = hl_geom.iso_cut(v, tt, a["z0"] - Z_LO, a)
    back = np.mod(a["theta"], 2 * np.pi)
    v, tt, a = hl_geom.iso_cut(v, tt, np.abs(back - np.pi) - 0.004, a)
    w, D = bind(v, tt, proxy, smooth_iters=2)
    w, D, bones = skin_weights(v, proxy, w, D, head)
    parts = solid_parts("choker", "Choker", v, tt, np.stack([a["S"], a["z0"]], 1), w, D, 0.16,
                        ("choker", "patch:lining", "patch:edge"), ("leather", "lining", "edge"), rim_segments=2)
    anchor, pieces = pendant(v, tt, tree)
    band_tree = hl_blend.bvh(v, tt)
    charm = [p[1] for p in pieces if p[0] == "choker_charm"][0].mean(0)
    cw, cD, _ = hl_fit.transfer(charm[None], proxy)
    cw, cD, _ = skin_weights(charm[None], proxy, cw, cD, head)
    for name, mv, mt, muv, shape, isl, mat in pieces:
        s = np.clip((v[anchor, 2] - mv[:, 2]) / (v[anchor, 2] - charm[2]), 0, 1)
        s = (s * s * (3 - 2 * s))[:, None]
        mw = w[anchor] * (1 - s) + cw * s
        mD = D[anchor] * (1 - s)[:, :, None] + cD * s[:, :, None]
        if name == "choker_tab":
            loc, nrm, face, dist = hl_blend.nearest(band_tree, mv)
            on = dist < 0.12
            tri = tt[face[on]]
            bc = hl_blend.barycentric(loc[on], v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]])
            mw[on] = np.einsum("nk,nkb->nb", bc, w[tri])
            mD[on] = np.einsum("nk,nksd->nsd", bc, D[tri])
        parts.append(Part(name, shape, mv, mt, muv, isl, mat, mw, mD))
    for p in parts:
        p.bones = list(bones)
        p.sliders = list(proxy.slider_names)
        if p.shape != "ChokerBrass":
            p.shape = "ChokerLining" if p.name.endswith("_in") else "Choker"
    return parts, None

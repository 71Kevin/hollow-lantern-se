import numpy as np

import hl_blend
import hl_fit
import hl_geom
from hl_parts import Part, bind, solid_parts

HAND_OFF = 0.11
SLEEVE_OFF = 0.17
LENGTH = 13.4
FLARE = 0.75
THICK = 0.14
SEGS = 48


class Arm:
    def __init__(self, side, skel):
        t = "L" if side < 0 else "R"
        self.side = side
        self.tag = t
        self.wrist = skel["NPC %s Hand [%sHnd]" % (t, t)][:3, 3]
        self.elbow = skel["NPC %s Forearm [%sLar]" % (t, t)][:3, 3]
        self.axis = hl_geom.normalize(self.elbow - self.wrist)
        up = np.array([0.0, 0.0, 1.0])
        self.u = hl_geom.normalize(np.cross(self.axis, up) * side)
        self.w = np.cross(self.axis, self.u)
        self.hand_bone = "NPC %s Hand [%sHnd]" % (t, t)


def hand_surface(hands_ref, side):
    sh = hands_ref["shapes"]["Hands"]
    v, t, uv, w = sh["v"], sh["t"], sh["uv"], sh["w"]
    keep = (np.sign(v[t][:, :, 0]).sum(1) * side) > 0
    t = t[keep]
    v2, t2, _, used = hl_geom.compact(v, t)
    uv2 = uv[used]
    w2 = w[used]
    q = np.round(v2 / 1e-4).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    wv = v2[first]
    wt = inv[t2]
    n = hl_geom.vertex_normals(wv, wt)
    pairs = hl_geom.neighbor_pairs(wt, len(wv))
    dist = hl_geom.edge_distance(wv, wt)
    k = np.clip(1.0 - dist / 3.5, 0, 1)
    k = k * k * (3 - 2 * k)
    off = HAND_OFF + (SLEEVE_OFF - HAND_OFF) * k
    p = wv + n * off[:, None]
    p = hl_geom.taubin(p, pairs, iters=6, fixed=dist < 1e-6)
    loops = hl_geom.boundary_loops(wt)
    wrist = max(loops, key=len)
    D = np.zeros((len(v2), 1, 3), dtype=np.float32)
    names = []
    for (name, tgt), (idx, d) in hands_ref["diffs"].items():
        if tgt != "Hands":
            continue
        dense = np.zeros((len(v), 3), dtype=np.float32)
        if len(idx):
            dense[idx] = d
        D = dense[used][:, None, :]
        names = [name]
    loop_w = w2[first][wrist]
    return {"v": p[inv], "t": t2, "uv": uv2, "w": w2, "bones": list(sh["bones"]), "D": D, "sliders": names,
            "weld_v": p, "weld_t": wt, "inv": inv, "wrist_loop": p[wrist], "wrist_ids": wrist, "loop_w": loop_w}


def sleeve(proxy, arm, wrist_loop):
    rel = wrist_loop - arm.wrist
    h = rel @ arm.axis
    ang = np.arctan2(rel @ arm.w, rel @ arm.u)
    order = np.argsort(ang)
    sleeve.order = order
    loop_sorted = wrist_loop[order]
    ang_sorted = ang[order]
    h0 = h.max() + 0.25
    phis = np.linspace(-np.pi, np.pi, SEGS, endpoint=False)
    hs = np.arange(h0, LENGTH + FLARE + 0.01, 0.32)
    rings = []
    for hh in hs:
        c = arm.wrist + arm.axis * hh
        d = np.outer(np.cos(phis), arm.u) + np.outer(np.sin(phis), arm.w)
        loc, nrm, face, dist = hl_blend.raycast(proxy.tree, np.tile(c, (SEGS, 1)), d, 8.0)
        r = np.where(np.isfinite(dist), dist, np.nan)
        if np.isnan(r).any():
            g = ~np.isnan(r)
            r = np.interp(phis, phis[g], r[g], period=2 * np.pi)
        rings.append(c + d * (r + SLEEVE_OFF)[:, None])
    grid = np.array(rings)
    flat = hl_fit.push_out(grid.reshape(-1, 3), proxy, SLEEVE_OFF, iters=2)
    grid = flat.reshape(grid.shape)
    v = grid.reshape(-1, 3)
    t = hl_geom.grid_tris(len(hs), SEGS, wrap=True)
    ring_len = np.linalg.norm(np.diff(np.concatenate([grid, grid[:, :1]], 1), axis=1), axis=2)
    arc = np.concatenate([np.zeros((len(hs), 1)), np.cumsum(ring_len, 1)[:, :-1]], 1)
    along = np.repeat(hs, SEGS)
    phi = np.tile(phis, len(hs))
    nl = len(loop_sorted)
    lv = loop_sorted
    base = len(v)
    zip_t = []
    i = j = 0
    first = np.arange(SEGS)
    while i < nl or j < SEGS:
        a_l = ang_sorted[i % nl] + (2 * np.pi if i >= nl else 0)
        a_r = phis[j % SEGS] + (2 * np.pi if j >= SEGS else 0)
        if (a_l <= a_r and i < nl) or j >= SEGS:
            zip_t.append([base + i % nl, base + (i + 1) % nl, first[j % SEGS]])
            i += 1
        else:
            zip_t.append([base + i % nl, first[(j + 1) % SEGS], first[j % SEGS]])
            j += 1
    v_all = np.vstack([v, lv])
    t_all = np.vstack([t, np.array(zip_t)])
    period = np.cumsum(ring_len, 1)[0, -1]
    loop_arc = np.interp(ang_sorted, np.append(phis, np.pi), np.append(arc[0], period))
    arc_all = np.concatenate([arc.ravel(), loop_arc])
    along_all = np.concatenate([along, np.full(nl, h0 - 0.25)])
    phi_all = np.concatenate([phi, ang_sorted])
    first = np.concatenate([np.arange(len(hs)) * SEGS, [base]])
    totals = np.concatenate([np.cumsum(ring_len, 1)[:, -1], [arc_all[base] + period]])
    v_all, t_all, attrs = hl_geom.open_ring_seam(v_all, t_all, first,
                                                 {"arc": arc_all, "along": along_all, "phi": phi_all}, "arc", totals)
    nr = hl_geom.vertex_normals(v_all, t_all)
    ctr = arm.wrist + np.outer((v_all - arm.wrist) @ arm.axis, arm.axis)
    if ((v_all - ctr) * nr).sum() < 0:
        t_all = t_all[:, ::-1]
    return v_all, t_all, attrs, nl, base


def build_side(body, proxy, hands_ref, skel, side, log, t0):
    import time
    arm = Arm(side, skel)
    hs = hand_surface(hands_ref, side)
    sv, st, sa, nl, base = sleeve(proxy, arm, hs["wrist_loop"])
    top = LENGTH + FLARE * np.cos(sa["phi"] - 0.0)
    sv2, st2, sa2 = hl_geom.iso_cut(sv, st, top - sa["along"], dict(sa, lid=np.arange(len(sv)).astype(float)))
    w, D = bind(sv2, st2, proxy, smooth_iters=3, edge_body=body)
    lid = np.round(sa2["lid"]).astype(np.int64)
    exact = np.abs(sa2["lid"] - lid) < 1e-6
    loop_of = -np.ones(len(sv), dtype=np.int64)
    loop_of[base:base + nl] = np.arange(nl)
    loop_of[-1] = 0
    loop_rows = exact & (loop_of[np.clip(lid, 0, len(sv) - 1)] >= 0)
    lw = hs["loop_w"][sleeve.order]
    lw_body = np.zeros((nl, len(proxy.bones)))
    for j, b in enumerate(hs["bones"]):
        if b in proxy.bones:
            lw_body[:, proxy.bones.index(b)] = lw[:, j]
    lw_body /= np.maximum(lw_body.sum(1, keepdims=True), 1e-9)
    w[loop_rows] = lw_body[loop_of[lid[loop_rows]]]
    log("%s sleeve %.1fs" % (arm.tag, time.time() - t0))
    uv = np.stack([sa2["arc"], sa2["along"]], 1)
    parts = solid_parts("glove%s_sleeve" % arm.tag, "Gloves", sv2, st2, uv, w, D, THICK,
                        ("glove%s_sleeve" % arm.tag, "patch:lining", "patch:edge"), ("leather", "lining", "edge"),
                        rim_segments=2)
    for p in parts:
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
    hp = Part("glove%s_hand" % arm.tag, "Gloves", hs["v"], hs["t"], hs["uv"], "glove%s_hand" % arm.tag, "leather",
              hs["w"], hs["D"])
    hp.bones = hs["bones"]
    hp.sliders = hs["sliders"]
    hand_n = hl_geom.vertex_normals(hs["weld_v"], hs["weld_t"])
    hp.normals = hand_n[hs["inv"]]
    loop_n = hand_n[hs["wrist_ids"]]
    for p in parts:
        if p.name.endswith("_sleeve"):
            d = np.linalg.norm(p.v[:, None, :] - hs["wrist_loop"][None, :, :], axis=2)
            j = d.argmin(1)
            hit = d[np.arange(len(p.v)), j] < 1e-4
            p.normals = p.normals.copy()
            p.normals[hit] = loop_n[j[hit]]
    hp.uv = hs["uv"] * 12.0
    parts.append(hp)
    btn_c = arm.wrist + arm.axis * 1.6
    rel_dirs = np.outer(np.cos(np.linspace(-np.pi, np.pi, SEGS, endpoint=False)), arm.u)
    q, n, f, d = hl_blend.nearest(hl_blend.bvh(sv2, st2), (btn_c + arm.w * 4.0 * side)[None])
    nn = n[0] if np.dot(n[0], q[0] - btn_c) > 0 else -n[0]
    bv, bt, buv, _ = hl_geom.sweep(np.array([q[0] - nn * 0.05, q[0] + nn * 0.1, q[0] + nn * 0.16]),
                                   np.array([0.3, 0.28, 0.16]), 14, cap_end=True)
    bw, bD = bind(bv, bt, proxy, rigid=True)
    btn = Part("glove%s_button" % arm.tag, "GlovesBrass", bv, bt, buv, "patch:brass", "brass", bw, bD)
    btn.bones = list(proxy.bones)
    btn.sliders = list(proxy.slider_names)
    parts.append(btn)
    log("%s hand %.1fs" % (arm.tag, time.time() - t0))
    return parts


def build(body, proxy=None, log=print, hands_ref=None, skel=None):
    import os
    import pickle
    import time
    t0 = time.time()
    ref_dir = r"D:\Dev\Skyrim MOD\work\hollow-lantern\ref"
    hands_ref = hands_ref or pickle.load(open(os.path.join(ref_dir, "hands.pkl"), "rb"))
    skel = skel or pickle.load(open(os.path.join(ref_dir, "skeleton.pkl"), "rb"))
    parts = build_side(body, proxy, hands_ref, skel, -1, log, t0) + build_side(body, proxy, hands_ref, skel, 1, log, t0)
    for p in parts:
        if p.mat == "brass":
            p.shape = "GlovesBrass"
        elif p.name.endswith("_in"):
            p.shape = "GlovesLining"
        else:
            p.shape = "Gloves"
    return parts, None

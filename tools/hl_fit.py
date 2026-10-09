import numpy as np

import hl_blend
import hl_geom


class Body:
    def __init__(self, ref, shape_names=("3BA", "3BA_Vagina", "3BA_Anus"), weld_tol=1e-4):
        shapes = [ref["shapes"][n] for n in shape_names]
        self.shape_names = list(shape_names)
        bones = []
        for sh in shapes:
            for b in sh["bones"]:
                if b not in bones:
                    bones.append(b)
        self.bones = bones
        s2b = {}
        for sh in shapes:
            for j, b in enumerate(sh["bones"]):
                s2b.setdefault(b, sh["s2b"][j])
        self.s2b = np.array([s2b[b] for b in bones])
        vs, ts, ws = [], [], []
        self.ranges = {}
        off = 0
        for name, sh in zip(shape_names, shapes):
            n = len(sh["v"])
            self.ranges[name] = (off, off + n)
            vs.append(sh["v"])
            ts.append(sh["t"] + off)
            w = np.zeros((n, len(bones)), dtype=np.float64)
            for j, b in enumerate(sh["bones"]):
                w[:, bones.index(b)] = sh["w"][:, j]
            ws.append(w)
            off += n
        v_all = np.concatenate(vs)
        t_all = np.concatenate(ts)
        w_all = np.concatenate(ws)
        names = []
        for attrs, data in ref["sliders"]:
            if any(tgt in shape_names for _, tgt, _ in data) and attrs["name"] not in names:
                names.append(attrs["name"])
        self.slider_names = names
        self.slider_attrs = []
        D_all = np.zeros((len(v_all), len(names), 3), dtype=np.float32)
        for attrs, data in ref["sliders"]:
            if attrs["name"] not in names:
                continue
            if attrs not in self.slider_attrs:
                self.slider_attrs.append(attrs)
            k = names.index(attrs["name"])
            for dname, tgt, _ in data:
                if tgt not in shape_names:
                    continue
                idx, d = ref["diffs"][(attrs["name"], tgt)]
                if len(idx):
                    D_all[idx + self.ranges[tgt][0], k] = d
        q = np.round(v_all / weld_tol).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv).astype(np.float64)
        self.inv = inv
        self.v = v_all[first]
        t = inv[t_all]
        good = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 2] != t[:, 0])
        self.t = t[good]
        w = np.zeros((len(first), len(bones)))
        np.add.at(w, inv, w_all)
        self.w = w / cnt[:, None]
        D = np.zeros((len(first), len(names), 3), dtype=np.float64)
        np.add.at(D, inv, D_all)
        self.D = (D / cnt[:, None, None]).astype(np.float32)
        self.orig_t = {name: sh["t"] + self.ranges[name][0] for name, sh in zip(shape_names, shapes)}
        self.n = hl_geom.vertex_normals(self.v, self.t)
        self.tree = hl_blend.bvh(self.v, self.t)

    def bone_globals(self):
        return {b: np.linalg.inv(self.s2b[j]) for j, b in enumerate(self.bones)}

    def mask_bones(self, keys):
        m = np.zeros(len(self.v))
        for j, b in enumerate(self.bones):
            if any(k in b for k in keys):
                m += self.w[:, j]
        return m

    def signed_distance(self, points):
        loc, nrm, face, dist = hl_blend.nearest(self.tree, points)
        s = np.sign(((points - loc) * nrm).sum(1))
        s[s == 0] = 1
        return s * dist, loc, nrm, face

    def project(self, points):
        loc, nrm, face, dist = hl_blend.nearest(self.tree, points)
        tri = self.t[face]
        bc = hl_blend.barycentric(loc, self.v[tri[:, 0]], self.v[tri[:, 1]], self.v[tri[:, 2]])
        return tri, bc, loc, nrm, dist

    def sample(self, tri, bc, array):
        a = np.asarray(array)
        return np.einsum("nk,nk...->n...", bc, a[tri])

    def smoothed(self, mask, iters=30, lam=0.5, mu=-0.53):
        free = np.asarray(mask, dtype=np.float64)
        pairs = hl_geom.sub_pairs(hl_geom.neighbor_pairs(self.t, len(self.v)), free > 0)
        rows = pairs[3]
        v = self.v.copy()
        D = self.D.reshape(len(self.v), -1).astype(np.float64)
        for _ in range(iters):
            for f in (lam, mu):
                k = (f * free[rows])[:, None]
                v[rows] += k * hl_geom.laplacian(v, pairs)[rows]
                D[rows] += k * hl_geom.laplacian(D, pairs)[rows]
        out = object.__new__(Body)
        out.__dict__.update(self.__dict__)
        out.v = v
        out.D = D.reshape(self.D.shape).astype(np.float32)
        out.n = hl_geom.vertex_normals(v, self.t)
        out.tree = hl_blend.bvh(v, self.t)
        return out

    def apply_preset(self, values):
        out = self.v.copy()
        for k, name in enumerate(self.slider_names):
            if name in values:
                out += self.D[:, k] * values[name]
        return out


def sym_curve(ctrl, samples=16):
    pts = np.array(ctrl, dtype=np.float64)
    ext = np.vstack([[-pts[1][0], pts[1][1]], pts, [2 * np.pi - pts[-2][0], pts[-2][1]]])
    dense = hl_geom.catmull_rom(ext, samples)
    dense = dense[(dense[:, 0] >= 0) & (dense[:, 0] <= np.pi)]
    order = np.argsort(dense[:, 0])
    a, z = dense[order, 0], dense[order, 1]

    def f(theta):
        x = np.abs(np.mod(np.asarray(theta) + np.pi, 2 * np.pi) - np.pi)
        return np.interp(x, a, z)
    return f


def push_out(points, body, offset, iters=2):
    p = np.array(points, dtype=np.float64)
    off = np.broadcast_to(np.asarray(offset, dtype=np.float64), (len(p),))
    for _ in range(iters):
        sd, loc, nrm, _ = body.signed_distance(p)
        bad = sd < off
        if not bad.any():
            break
        p[bad] = loc[bad] + nrm[bad] * off[bad, None]
    return p


def drape(points, tris, body, offset, iters=40, lam=0.5, fixed=None, pull=0.0, check_every=2):
    pairs = hl_geom.neighbor_pairs(tris, len(points))
    p = push_out(points, body, offset)
    for k in range(iters):
        dp = lam * hl_geom.laplacian(p, pairs)
        if fixed is not None:
            dp[fixed] = 0
        p = p + dp
        if pull:
            sd, loc, nrm, _ = body.signed_distance(p)
            off = np.broadcast_to(np.asarray(offset, dtype=np.float64), (len(p),))
            target = loc + nrm * off[:, None]
            p = p + pull * (target - p) * (sd > off)[:, None]
        if k % check_every == check_every - 1 or k == iters - 1:
            p = push_out(p, body, offset, iters=1)
    return push_out(p, body, offset, iters=2)


def transfer(points, body, smooth_tris=None, smooth_iters=0, bone_filter=None, weight_scale=None):
    tri, bc, loc, nrm, dist = body.project(points)
    w = body.sample(tri, bc, body.w)
    if bone_filter is not None:
        w = w * bone_filter[None, :]
    if weight_scale is not None:
        w = w * weight_scale[None, :]
    D = body.sample(tri, bc, body.D.reshape(len(body.v), -1)).reshape(len(points), -1, 3)
    if smooth_tris is not None and smooth_iters:
        pairs = hl_geom.neighbor_pairs(smooth_tris, len(points))
        w = hl_geom.smooth(w, pairs, smooth_iters, 0.5)
        D = hl_geom.smooth(D.reshape(len(points), -1), pairs, smooth_iters, 0.5).reshape(D.shape)
    s = w.sum(1, keepdims=True)
    s[s == 0] = 1
    return w / s, D, dist


def harmonic_fill(tris, X, k, iters=600):
    X = np.asarray(X, dtype=np.float64)
    flat = X.reshape(len(X), -1).copy()
    sel = k > 0.999
    rows = np.nonzero(sel)[0]
    sub = hl_geom.sub_pairs(hl_geom.neighbor_pairs(tris, len(X)), sel)
    for _ in range(iters):
        flat[rows] += 0.9 * hl_geom.laplacian(flat, sub)[rows]
    kk = k.reshape((-1,) + (1,) * (X.ndim - 1))
    return X * (1 - kk) + flat.reshape(X.shape) * kk


def beneath_weights(points, tris, body, bone_mask=None, reach=3.0, smooth_iters=4):
    n = hl_geom.vertex_normals(points, tris)
    loc, nrm, face, dist = hl_blend.raycast(body.tree, points, -n, reach)
    miss = ~np.isfinite(dist)
    if miss.any():
        loc2, _, face2, _ = hl_blend.nearest(body.tree, points[miss])
        loc[miss] = loc2
        face[miss] = face2
    tri = body.t[face]
    bc = hl_blend.barycentric(loc, body.v[tri[:, 0]], body.v[tri[:, 1]], body.v[tri[:, 2]])
    w = np.einsum("nk,nkb->nb", bc, body.w[tri])
    if bone_mask is not None:
        w = w * bone_mask[None, :]
    if smooth_iters:
        w = hl_geom.smooth(w, hl_geom.neighbor_pairs(tris, len(points)), smooth_iters, 0.5)
    return w / np.maximum(w.sum(1, keepdims=True), 1e-9)


def garment_conform(points, D, body, slack_margin=None, normals=None, smooth_tris=None, smooth_iters=60):
    tri, bc, loc, nrm, dist = body.project(points)
    Db = body.sample(tri, bc, body.D.reshape(len(body.v), -1)).reshape(D.shape)
    gap = ((Db - D) * nrm[:, None, :]).sum(2)
    if slack_margin is not None:
        gap -= np.clip(dist - slack_margin, 0, None)[:, None]
    gap = np.clip(gap, 0, None)
    if smooth_tris is not None:
        pairs = hl_geom.neighbor_pairs(smooth_tris, len(points))
        need = gap
        for _ in range(smooth_iters):
            gap = np.maximum(need, gap + 0.5 * hl_geom.laplacian(gap, pairs))
    if normals is None:
        return D + gap[:, :, None] * nrm[:, None, :]
    cos = np.clip((normals * nrm).sum(1), 0.3, 1.0)
    return D + (gap / cos[:, None])[:, :, None] * normals[:, None, :]


def standoff_smooth(points, tris, D, body, start=0.9, iters=30):
    _, _, _, _, dist = body.project(points)
    pairs = hl_geom.neighbor_pairs(tris, len(points))
    flat = D.reshape(len(points), -1)
    k = np.clip((dist - start) / 0.6, 0, 1)
    for _ in range(iters):
        flat = flat + 0.5 * k[:, None] * hl_geom.laplacian(flat, pairs)
    return flat.reshape(D.shape)


def welded(v, t, tol=1e-4):
    q = np.round(np.asarray(v) / tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    return np.asarray(v, dtype=np.float64)[first], inv[t], first, inv


def layer_body(v, t, D):
    out = object.__new__(Body)
    out.v = np.asarray(v, dtype=np.float64)
    out.t = np.asarray(t)
    out.D = np.asarray(D, dtype=np.float32)
    out.n = hl_geom.vertex_normals(out.v, out.t)
    out.tree = hl_blend.bvh(out.v, out.t)
    return out


def fill_concave(v, t, k, iters=220, lam=0.5):
    pairs = hl_geom.neighbor_pairs(t, len(v))
    p = np.array(v, dtype=np.float64)
    for i in range(iters):
        if i % 4 == 0:
            n = hl_geom.vertex_normals(p, t)
        dp = lam * hl_geom.laplacian(p, pairs)
        dn = (dp * n).sum(1)
        dp -= np.minimum(dn, 0)[:, None] * n
        p += dp * k[:, None]
    return p


def random_presets(count, n_sliders, seed=11, active=0.35, lo=-1.1, hi=1.4):
    rng = np.random.default_rng(seed)
    on = rng.random((count, n_sliders)) < active
    return rng.uniform(lo, hi, (count, n_sliders)) * on


def fold_safe_diffs(v, t, D, sliders, skip=(), damp=None, samples=160, rounds=8, iters=30, grow=3,
                    min_cos=0.0, min_area=0.2):
    wv, wt, first, inv = welded(v, t)
    pairs = hl_geom.neighbor_pairs(wt, len(wv))
    i, j = pairs[0], pairs[1]
    f0 = hl_geom.face_normals(wv, wt)
    a0 = np.linalg.norm(f0, axis=1)
    n0 = f0 / np.maximum(a0, 1e-12)[:, None]
    real = a0 > 0.25 * np.median(a0)
    out = np.array(D, dtype=np.float64)
    cols = [k for k, n in enumerate(sliders) if not any(s in n for s in skip) and np.abs(out[:, k]).max() > 1e-3]
    Dw = out[first][:, cols]
    vals = random_presets(samples, len(cols))
    scale = np.ones(len(wv))
    if damp is not None:
        np.minimum.at(scale, inv, damp)
    mask = np.zeros(len(wv))
    history = []
    for r in range(rounds + 1):
        hits = np.zeros(len(wv))
        bad_total = 0
        for s in range(samples):
            x = wv + np.einsum("c,ncd->nd", vals[s], Dw)
            f = hl_geom.face_normals(x, wt)
            a = np.linalg.norm(f, axis=1)
            bad = (((f * n0).sum(1) <= min_cos * a) | (a < min_area * a0)) & real
            bad_total += bad.sum()
            hits[wt[bad].ravel()] += 1
        history.append(bad_total / samples)
        if r == rounds or bad_total == 0:
            break
        m = (hits > 0).astype(np.float64)
        for _ in range(grow):
            g = m.copy()
            np.maximum.at(g, i, m[j])
            m = g
        mask = np.maximum(mask, m * scale)
        sub = hl_geom.sub_pairs(pairs, mask > 0)
        rows = sub[3]
        flat = Dw.reshape(len(wv), -1)
        for _ in range(iters):
            flat[rows] += 0.5 * mask[rows, None] * hl_geom.laplacian(flat, sub)[rows]
        Dw = flat.reshape(Dw.shape)
    out[:, cols] = Dw[inv]
    return out, history


def torso_axis(body, z_values):
    arm = body.mask_bones(("UpperArm", "Forearm", "Hand", "Clavicle", "Finger"))
    legs = body.mask_bones(("Thigh", "Calf", "Foot"))
    m = (arm < 0.05)
    cy = []
    for z in z_values:
        sel = m & (np.abs(body.v[:, 2] - z) < 0.8) & (np.abs(body.v[:, 0]) < 10)
        if sel.sum() < 4:
            cy.append(np.nan)
            continue
        p = body.v[sel]
        cy.append(0.5 * (p[:, 1].min() + p[:, 1].max()))
    cy = np.array(cy)
    good = ~np.isnan(cy)
    cy = np.interp(z_values, np.asarray(z_values)[good], cy[good])
    return cy


def ring_grid(body, z_values, n_theta, center_y, theta0=0.0):
    rows = len(z_values)
    th = theta0 + np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    origins = np.zeros((rows * n_theta, 3))
    dirs = np.zeros((rows * n_theta, 3))
    for r, z in enumerate(z_values):
        origins[r * n_theta:(r + 1) * n_theta] = [0.0, center_y[r], z]
        dirs[r * n_theta:(r + 1) * n_theta, 0] = np.sin(th)
        dirs[r * n_theta:(r + 1) * n_theta, 1] = np.cos(th)
    loc, nrm, face, dist = hl_blend.raycast(body.tree, origins, dirs, 60.0)
    miss = np.isnan(loc[:, 0])
    if miss.any():
        loc[miss] = origins[miss] + dirs[miss] * 10.0
    theta = np.tile(th, rows)
    zz = np.repeat(z_values, n_theta)
    t = hl_geom.grid_tris(rows, n_theta, wrap=True)[:, ::-1].copy()
    return loc, t, theta, zz


def conform_samples(body, D, tris, idx, garment_parts, samples, skip=(), rounds=4, margin=0.04, reach=1.5):
    vs, ts, ds = [], [], []
    off = 0
    for p in garment_parts:
        vs.append(p.v)
        ts.append(p.t + off)
        ds.append(np.asarray(p.D, dtype=np.float64))
        off += len(p.v)
    gv = np.concatenate(vs)
    gt = np.concatenate(ts)
    gD = np.concatenate(ds)
    names = body.slider_names
    S = np.array([[vals.get(n, 0.0) for n in names] for vals in samples])
    G = S * np.array([not any(k in n for k in skip) for n in names])
    out = np.array(D, dtype=np.float64)
    history = []
    active = np.arange(len(S))
    for r in range(rounds + 1):
        need = np.zeros(len(idx))
        dirs = np.zeros((len(idx), 3))
        pick = np.full(len(idx), -1)
        total = 0
        poked = []
        for s in active:
            bv = body.v + np.einsum("j,njd->nd", S[s], out)
            bn = hl_geom.vertex_normals(bv, tris)[idx]
            tree = hl_blend.bvh(gv + np.einsum("j,njd->nd", G[s], gD), gt)
            loc, nrm, face, dist = hl_blend.raycast(tree, bv[idx] + bn * 0.005, -bn, reach)
            poke = np.isfinite(dist) & ((nrm * bn).sum(1) > 0)
            total += poke.sum()
            if poke.any():
                poked.append(s)
            c = np.where(poke, dist + margin, 0.0)
            better = c > need
            need[better] = c[better]
            dirs[better] = bn[better]
            pick[better] = s
        history.append(total / len(S))
        if not poked:
            if len(active) == len(S):
                break
            active = np.arange(len(S))
            continue
        if r == rounds:
            break
        active = np.array(poked)
        hit = pick >= 0
        sv = S[pick[hit]]
        k = need[hit] / np.maximum((sv * sv).sum(1), 1e-9)
        out[idx[hit]] -= (k[:, None] * sv)[:, :, None] * dirs[hit][:, None, :]
    return out, history


def conform_diffs(body, D, idx, garment_parts, uv_sliders=(), reach=4.0):
    vs, ts, ds = [], [], []
    off = 0
    for p in garment_parts:
        vs.append(p.v)
        ts.append(p.t + off)
        ds.append(np.asarray(p.D, dtype=np.float64))
        off += len(p.v)
    gv = np.concatenate(vs)
    gt = np.concatenate(ts)
    gD = np.concatenate(ds)
    tree = hl_blend.bvh(gv, gt)
    n = body.n[idx]
    loc, nrm, face, dist = hl_blend.raycast(tree, body.v[idx] + n * 0.005, n, reach)
    hit = np.isfinite(dist) & ((nrm * n).sum(1) > 0)
    rows = idx[hit]
    tri = gt[face[hit]]
    bc = hl_blend.barycentric(loc[hit], gv[tri[:, 0]], gv[tri[:, 1]], gv[tri[:, 2]])
    g = np.einsum("nk,nksd->nsd", bc, gD[tri])
    nn = n[hit]
    out = np.array(D, dtype=np.float64)
    skip = np.array([s in uv_sliders for s in body.slider_names])
    rel = out[rows] - g
    r = np.einsum("nsd,nd->ns", rel, nn)
    r[:, skip] = 0
    out[rows] -= np.clip(r, 0, None)[:, :, None] * nn[:, None, :]
    return out, len(rows)

import numpy as np

import hl_fit
import hl_geom


class Part:
    def __init__(self, name, shape, v, t, uv, island, mat, w=None, D=None):
        self.name = name
        self.shape = shape
        self.v = np.asarray(v, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.int64)
        self.uv = None if uv is None else np.asarray(uv, dtype=np.float64)
        self.island = island
        self.mat = mat
        self.w = w
        self.D = D
        self.attrs = {}


def bind(v, t, body, smooth_iters=0, bone_mask=None, rigid=False, edge_body=None, edge_width=1.6):
    w, D, dist = hl_fit.transfer(v, body, t, smooth_iters, bone_filter=bone_mask)
    if edge_body is not None:
        wr, Dr, _ = hl_fit.transfer(v, edge_body, None, 0, bone_filter=bone_mask)
        d = hl_geom.edge_distance(v, t)
        k = np.clip(d / edge_width, 0, 1)
        k = k * k * (3 - 2 * k)
        w = wr * (1 - k)[:, None] + w * k[:, None]
        D = Dr * (1 - k)[:, None, None] + D * k[:, None, None]
    if rigid:
        w = np.repeat(w.mean(0, keepdims=True), len(w), 0)
        D = np.repeat(D.mean(0, keepdims=True), len(D), 0)
    return w, D


def sub_part(name, shape, v, t, uv, island, mat, w, D, sel_tris, attrs=None):
    tt = t[sel_tris]
    used = np.unique(tt)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    p = Part(name, shape, v[used], remap[tt], None if uv is None else uv[used], island, mat,
             None if w is None else w[used], None if D is None else D[used])
    p.attrs = {k: a[used] for k, a in (attrs or {}).items()}
    return p


def solid_parts(name, shape, v, t, uv, w, D, thickness, islands, mats, rim_segments=3, attrs=None,
                split_outer=None, normals=None):
    n = hl_geom.vertex_normals(v, t) if normals is None else normals
    base = dict(attrs or {})
    base["uv"] = uv
    base["w"] = w
    base["D"] = D
    vs, ts, src, layer, phase, sa, n_out = hl_geom.solidify(v, t, n, thickness, rim_segments, base)
    lay = layer[ts]
    sels = [(lay == 0).all(1), (lay == 1).all(1)]
    sels.append(~(sels[0] | sels[1]))
    rim_uv = sa["uv"].copy()
    rim_uv[:, 1] = rim_uv[:, 1] + phase * thickness * 3.0
    out = []
    extra = {k: a for k, a in sa.items() if k not in ("uv", "w", "D")}
    vn = np.concatenate([n, -n, np.zeros((len(vs) - 2 * len(v), 3))])
    for lid, suffix in enumerate(("", "_in", "_rim")):
        if not sels[lid].any():
            continue
        uvl = sa["uv"] if lid < 2 else rim_uv
        p = sub_part(name + suffix, shape, vs, ts, uvl, islands[lid], mats[lid], sa["w"], sa["D"], sels[lid],
                     dict(extra, layer=layer, phase=phase, vn=vn))
        if lid == 0 and split_outer is not None:
            out += split_outer(p)
            continue
        if lid < 2:
            p.normals = hl_geom.normalize(p.attrs["vn"])
        out.append(p)
    return out

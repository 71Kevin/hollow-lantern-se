import numpy as np


def normalize(a, axis=-1, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / np.maximum(n, eps)


def face_normals(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return np.cross(b - a, c - a)


def vertex_normals(v, t):
    fn = face_normals(v, t)
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    return normalize(n)


def edges_of(t):
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return e


def unique_edges(t):
    e = np.sort(edges_of(t), axis=1)
    return np.unique(e, axis=0)


def boundary_edges(t):
    e = edges_of(t)
    key = np.sort(e, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    return e[counts[inv] == 1]


def boundary_loops(t):
    be = boundary_edges(t)
    nxt = {int(a): int(b) for a, b in be}
    loops = []
    seen = set()
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start and cur in nxt and cur not in seen:
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        loops.append(np.array(loop))
    return loops


def neighbor_pairs(t, n):
    e = unique_edges(t)
    i = np.concatenate([e[:, 0], e[:, 1]])
    j = np.concatenate([e[:, 1], e[:, 0]])
    order = np.argsort(i, kind="stable")
    i, j = i[order], j[order]
    deg = np.bincount(i, minlength=n).astype(np.float64)
    rows = np.nonzero(deg)[0]
    starts = np.searchsorted(i, rows)
    return i, j, deg, rows, starts


def laplacian(x, pairs):
    i, j, deg, rows, starts = pairs
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    acc = np.add.reduceat(x[j], starts, axis=0)
    d = deg[rows].reshape((-1,) + (1,) * (x.ndim - 1))
    out[rows] = acc / d - x[rows]
    return out


def sub_pairs(pairs, row_mask):
    i, j, deg, rows, starts = pairs
    sel = row_mask[i]
    i2, j2 = i[sel], j[sel]
    rows2 = np.unique(i2)
    starts2 = np.searchsorted(i2, rows2)
    return i2, j2, deg, rows2, starts2


def open_ring_seam(v, t, first, attrs, key, totals):
    first = np.asarray(first)
    dup = len(v) + np.arange(len(first))
    out = {k: np.concatenate([a, np.asarray(a)[first]]) for k, a in attrs.items()}
    out[key][dup] = totals
    limit = np.zeros(len(v) + len(first))
    limit[first] = 0.5 * np.asarray(totals)
    remap = np.arange(len(v) + len(first))
    remap[first] = dup
    t = np.array(t)
    s = out[key][t]
    hit = np.isin(t, first) & (s.max(1, keepdims=True) > limit[t])
    t[hit] = remap[t[hit]]
    return np.vstack([v, v[first]]), t, out


def smooth(x, pairs, iters=10, lam=0.5, fixed=None):
    x = np.array(x, dtype=np.float64)
    for _ in range(iters):
        dx = lam * laplacian(x, pairs)
        if fixed is not None:
            dx[fixed] = 0
        x += dx
    return x


def taubin(x, pairs, iters=10, lam=0.5, mu=-0.53, fixed=None):
    x = np.array(x, dtype=np.float64)
    for _ in range(iters):
        for k in (lam, mu):
            dx = k * laplacian(x, pairs)
            if fixed is not None:
                dx[fixed] = 0
            x += dx
    return x


def compact(v, t, attrs=None):
    used = np.unique(t)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    out = {k: (a[used] if a is not None else None) for k, a in (attrs or {}).items()}
    return v[used], remap[t], out, used


def iso_cut(v, t, f, attrs=None, level=0.0):
    f = np.asarray(f, dtype=np.float64) - level
    attrs = attrs or {}
    new_v = [v]
    new_a = {k: [a] for k, a in attrs.items()}
    edge_vert = {}
    n0 = len(v)
    count = [n0]

    def cut_vertex(a, b):
        key = (a, b) if a < b else (b, a)
        if key in edge_vert:
            return edge_vert[key]
        fa, fb = f[a], f[b]
        s = fa / (fa - fb)
        p = v[a] + s * (v[b] - v[a])
        new_v.append(p[None])
        for k, arr in attrs.items():
            new_a[k].append((arr[a] + s * (arr[b] - arr[a]))[None])
        idx = count[0]
        count[0] += 1
        edge_vert[key] = idx
        return idx

    out = []
    inside = f >= 0
    for tri in t:
        ins = inside[tri]
        k = ins.sum()
        if k == 3:
            out.append(tri)
        elif k == 0:
            continue
        else:
            r = int(np.argmax(ins != (k == 2)))
            a, b, c = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
            ab = cut_vertex(a, b)
            ca = cut_vertex(c, a)
            if k == 1:
                out.append([a, ab, ca])
            else:
                out.append([ab, b, c])
                out.append([ab, c, ca])
    v2 = np.concatenate(new_v)
    a2 = {k: np.concatenate(new_a[k]) for k in attrs}
    t2 = np.array(out, dtype=np.int64)
    return compact(v2, t2, a2)[:3]


def weld(v, t, tol=1e-5, attrs=None):
    q = np.round(v / tol).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    t2 = inv[t]
    keep = (t2[:, 0] != t2[:, 1]) & (t2[:, 1] != t2[:, 2]) & (t2[:, 2] != t2[:, 0])
    out = {k: a[idx] for k, a in (attrs or {}).items()}
    return v[idx], t2[keep], out


def grid_tris(rows, cols, wrap=False):
    tris = []
    cmax = cols if wrap else cols - 1
    for r in range(rows - 1):
        for c in range(cmax):
            c1 = (c + 1) % cols
            a = r * cols + c
            b = r * cols + c1
            d = (r + 1) * cols + c
            e = (r + 1) * cols + c1
            tris.append([a, b, e])
            tris.append([a, e, d])
    return np.array(tris, dtype=np.int64)


def arc_param(points, axis):
    d = np.linalg.norm(np.diff(points, axis=axis), axis=-1)
    z = np.zeros_like(np.take(d, [0], axis=axis))
    return np.concatenate([z, np.cumsum(d, axis=axis)], axis=axis)


def solidify(v, t, n, thickness, rim_segments=3, attrs=None):
    attrs = attrs or {}
    nv = len(v)
    outer = v
    inner = v - n * thickness
    loops = boundary_loops(t)
    rim_v = []
    rim_t = []
    rim_src = []
    rim_phase = []
    base = 2 * nv
    for loop in loops:
        m = len(loop)
        p = v[loop]
        tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
        nn = n[loop]
        side = normalize(np.cross(tang, nn))
        cnt = rim_segments - 1
        ring_ids = []
        for s in range(1, rim_segments):
            phi = np.pi * s / rim_segments
            mid = p - nn * (thickness * 0.5)
            pos = mid + nn * (np.cos(phi) * thickness * 0.5) + side * (np.sin(phi) * thickness * 0.5)
            ids = base + len(rim_src) + np.arange(m)
            rim_v.append(pos)
            rim_src.extend(loop.tolist())
            rim_phase.extend([s / rim_segments] * m)
            ring_ids.append(ids)
        chain = [loop] + ring_ids + [loop + nv]
        for k in range(len(chain) - 1):
            r0, r1 = chain[k], chain[k + 1]
            for i in range(m):
                j = (i + 1) % m
                rim_t.append([r0[j], r0[i], r1[i]])
                rim_t.append([r0[j], r1[i], r1[j]])
        if cnt == 0:
            pass
    vv = np.concatenate([outer, inner] + rim_v) if rim_v else np.concatenate([outer, inner])
    tt = np.concatenate([t, t[:, ::-1] + nv] + ([np.array(rim_t)] if rim_t else []))
    src = np.concatenate([np.arange(nv), np.arange(nv), np.array(rim_src, dtype=np.int64)])
    layer = np.concatenate([np.zeros(nv), np.ones(nv), np.full(len(rim_src), 2)]).astype(np.int64)
    phase = np.concatenate([np.zeros(nv), np.ones(nv), np.array(rim_phase)])
    out_attrs = {k: a[src] for k, a in attrs.items()}
    n_out = len(t)
    return vv, tt, src, layer, phase, out_attrs, n_out


def components(n, i, j, mask=None):
    lab = np.arange(n)
    if mask is not None:
        sel = mask[i] & mask[j]
        i, j = i[sel], j[sel]
    while True:
        m = np.minimum(lab[i], lab[j])
        new = lab.copy()
        np.minimum.at(new, i, m)
        np.minimum.at(new, j, m)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def smooth_boundary(v, t, iters=10, lam=0.5):
    v = np.array(v, dtype=np.float64)
    for loop in boundary_loops(t):
        if len(loop) < 4:
            continue
        for _ in range(iters):
            p = v[loop]
            avg = 0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
            v[loop] = p + lam * (avg - p)
    return v


def edge_distance(v, t, iters=60):
    e = unique_edges(t)
    ln = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    i = np.concatenate([e[:, 0], e[:, 1]])
    j = np.concatenate([e[:, 1], e[:, 0]])
    ll = np.concatenate([ln, ln])
    d = np.full(len(v), np.inf)
    be = boundary_edges(t)
    d[np.unique(be)] = 0.0
    for _ in range(iters):
        nd = d.copy()
        np.minimum.at(nd, i, d[j] + ll)
        if np.array_equal(nd, d):
            break
        d = nd
    return d


def tri_area(v, t):
    return 0.5 * np.linalg.norm(face_normals(v, t), axis=1)


def resample_polyline(p, count, closed=False):
    p = np.asarray(p, dtype=np.float64)
    if closed:
        p = np.vstack([p, p[:1]])
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    s = np.linspace(0, d[-1], count + (1 if closed else 0))
    if closed:
        s = s[:-1]
    out = np.stack([np.interp(s, d, p[:, k]) for k in range(p.shape[1])], 1)
    return out


def catmull_rom(points, samples_per_seg=8, closed=False):
    p = np.asarray(points, dtype=np.float64)
    if closed:
        pts = np.vstack([p[-1:], p, p[:2]])
    else:
        pts = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for s in range(samples_per_seg):
            u = s / samples_per_seg
            u2, u3 = u * u, u * u * u
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    if not closed:
        out.append(p[-1])
    return np.array(out)


def frames_along(path, up=(0, 0, 1)):
    path = np.asarray(path, dtype=np.float64)
    tang = np.gradient(path, axis=0)
    tang = normalize(tang)
    upv = np.array(up, dtype=np.float64)
    nrm = np.zeros_like(path)
    ref = upv if abs(np.dot(tang[0], upv)) < 0.95 else np.array([1.0, 0, 0])
    nrm[0] = normalize(np.cross(np.cross(tang[0], ref), tang[0]))
    for i in range(1, len(path)):
        axis = np.cross(tang[i - 1], tang[i])
        s = np.linalg.norm(axis)
        if s < 1e-9:
            nrm[i] = nrm[i - 1]
            continue
        axis /= s
        ang = np.arccos(np.clip(np.dot(tang[i - 1], tang[i]), -1, 1))
        x = nrm[i - 1]
        nrm[i] = (x * np.cos(ang) + np.cross(axis, x) * np.sin(ang) + axis * np.dot(axis, x) * (1 - np.cos(ang)))
        nrm[i] = normalize(nrm[i] - tang[i] * np.dot(nrm[i], tang[i]))
    bin_ = np.cross(tang, nrm)
    return tang, nrm, bin_


def sweep(path, radii, segments=12, profile=None, up=(0, 0, 1), cap_start=False, cap_end=False, twist=None):
    tang, nrm, bnr = frames_along(path, up)
    path = np.asarray(path, dtype=np.float64)
    m = len(path)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts = []
    uv = []
    lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    for i in range(m):
        a = ang + (twist[i] if twist is not None else 0.0)
        if profile is not None:
            prof = profile(i / max(m - 1, 1), a)
        else:
            prof = np.ones_like(a)
        r = radii[i] * prof
        ring = path[i] + np.outer(np.cos(a) * r, nrm[i]) + np.outer(np.sin(a) * r, bnr[i])
        verts.append(ring)
    v = np.concatenate(verts)
    cols = segments + 1
    v_seam = []
    uv = []
    for i in range(m):
        ring = verts[i]
        v_seam.append(np.vstack([ring, ring[:1]]))
        circ = 2 * np.pi * radii[i]
        for k in range(cols):
            uv.append([k / segments, lengths[i]])
    v = np.concatenate(v_seam)
    t = grid_tris(m, cols, wrap=False)
    uv = np.array(uv)
    caps = []
    if cap_start:
        c = len(v)
        v = np.vstack([v, path[0]])
        uv = np.vstack([uv, [0.5, 0]])
        for k in range(segments):
            caps.append([c, k + 1, k])
    if cap_end:
        c = len(v)
        v = np.vstack([v, path[-1]])
        uv = np.vstack([uv, [0.5, lengths[-1]]])
        base = (m - 1) * cols
        for k in range(segments):
            caps.append([c, base + k, base + k + 1])
    if caps:
        t = np.vstack([t, np.array(caps)])
    return v, t, uv, cols


def transfer_barycentric(points, tree, src_v, src_t, src_attrs, max_dist=1e9):
    from hl_blend import barycentric, nearest
    loc, nrm, face, dist = nearest(tree, points, max_dist)
    tri = src_t[np.maximum(face, 0)]
    bc = barycentric(loc, src_v[tri[:, 0]], src_v[tri[:, 1]], src_v[tri[:, 2]])
    out = {}
    for k, a in src_attrs.items():
        a = np.asarray(a)
        if a.ndim == 1:
            out[k] = (a[tri] * bc).sum(1)
        else:
            out[k] = np.einsum("nk,nk...->n...", bc, a[tri])
    return out, loc, nrm, dist, face, bc

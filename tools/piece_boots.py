import numpy as np

import hl_blend
import hl_fit
import hl_geom
from hl_parts import Part, bind, solid_parts, sub_part

HH = 4.8
SOLE_T = 0.35
SHAFT_OFF = 0.32
LINING_FLOOR = 35.0


class Side:
    def __init__(self, side):
        m = np.array([side * -1.0, 1.0, 1.0])
        self.side = side
        self.ankle = np.array([-12.74, -5.95, 6.08]) * m
        self.fwd = hl_geom.normalize(np.array([-0.271, 0.963, 0.0]) * m)
        self.lat = np.array([-0.963, -0.271, 0.0]) * m
        self.tag = "L" if side < 0 else "R"
        self.foot = "NPC L Foot [Lft ]" if side < 0 else "NPC R Foot [Rft ]"


TOP = [(0, 47.0), (0.5, 46.2), (1.2, 44.4), (2.0, 42.4), (np.pi, 41.6)]
FOOT = [
    (-3.85, 1.4, 9.5, 1.1, 1.1),
    (-3.3, 1.05, 12.2, 1.75, 1.85),
    (-2.2, 0.95, 13.6, 2.3, 2.4),
    (-0.8, 0.85, 13.9, 2.55, 2.65),
    (0.6, 0.1, 13.8, 2.55, 2.7),
    (1.9, -1.2, 13.2, 2.45, 2.6),
    (3.0, -2.6, 11.4, 2.4, 2.6),
    (4.0, -3.5, 6.5, 2.5, 2.8),
    (4.8, -3.9, 2.6, 2.65, 3.0),
    (6.0, -4.4, -0.4, 2.85, 3.25),
    (7.2, -4.45, -1.45, 2.9, 3.25),
    (8.4, -4.4, -2.15, 2.6, 2.75),
    (9.6, -4.3, -2.7, 2.0, 2.0),
    (10.8, -4.15, -3.2, 1.2, 1.15),
    (11.8, -4.0, -3.55, 0.45, 0.4),
    (12.3, -3.85, -3.72, 0.04, 0.04),
]
HEEL = [(1.0, -3.85, -0.65, 1.75), (-0.5, -3.55, -0.95, 1.45), (-2.0, -3.05, -1.2, 1.05),
        (-3.4, -2.6, -1.3, 0.78), (-4.35, -2.55, -1.15, 0.74)]
LACE_TOP = 27.0
LACE_GAP = 0.42
EYELET_STEP = 2.15
ANKLE_SPAN = (2.0, 17.0)
ANKLE_TURN = 6.5
SOLE_HOLD = (1.5, 4.5)
SEAM_FRONT = 6.0
SEAM_BACK = 2.0
SEAM_TOP = 48.5
SEAM_SIGMA = 2.5
SEAM_KNEE = 33.5


def leg_centers(body, zs, side=-1):
    v = body.v
    legw = np.zeros(len(v))
    pre = "NPC L" if side < 0 else "NPC R"
    for j, n in enumerate(body.bones):
        if n.startswith(pre) and any(k in n for k in ("Thigh", "Calf", "Foot")):
            legw += body.w[:, j]
    m = (legw > 0.5) & (np.sign(v[:, 0]) == side)
    out = []
    for z in zs:
        s = m & (np.abs(v[:, 2] - z) < 0.6)
        if s.sum() < 6:
            out.append([np.nan, np.nan])
            continue
        p = v[s]
        out.append([0.5 * (p[:, 0].min() + p[:, 0].max()), 0.5 * (p[:, 1].min() + p[:, 1].max())])
    out = np.array(out)
    zs = np.asarray(zs, dtype=np.float64)
    order = np.argsort(zs)
    good = ~np.isnan(out[:, 0])
    go = order[good[order]]
    for k in range(2):
        out[:, k] = np.interp(zs, zs[go], out[go, k])
    srt = out[order]
    for _ in range(12):
        srt[1:-1] = 0.25 * srt[:-2] + 0.5 * srt[1:-1] + 0.25 * srt[2:]
    out[order] = srt
    return out


def shaft_tube(proxy, sd_, z_lo=8.6, z_hi=48.6, step=0.34, cols=96):
    zs = np.arange(z_lo, z_hi + 1e-6, step)
    zb = np.clip(zs, 12.4, None)
    centers = leg_centers(proxy, zb, sd_.side)
    th = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    pts = np.zeros((len(zs), cols, 3))
    ref_ring = None
    for r, z in enumerate(zs):
        c = centers[r]
        zr = max(z, 12.4)
        o = np.tile([c[0], c[1], zr], (cols, 1))
        d = np.stack([np.sin(th), np.cos(th), np.zeros(cols)], 1)
        loc, nrm, face, dist = hl_blend.raycast(proxy.tree, o, d, 12.0)
        rad = np.where(np.isfinite(dist), dist, np.nan)
        if np.isnan(rad).any():
            good = ~np.isnan(rad)
            rad = np.interp(th, th[good], rad[good], period=2 * np.pi)
        rad = rad + SHAFT_OFF
        if z < 12.4:
            k = np.clip((12.4 - z) / 3.8, 0, 1)
            rad = rad * (1 - 0.06 * k)
        pts[r, :, 0] = c[0] + rad * np.sin(th)
        pts[r, :, 1] = c[1] + rad * np.cos(th)
        pts[r, :, 2] = z
    v = pts.reshape(-1, 3)
    t = hl_geom.grid_tris(len(zs), cols, wrap=True)[:, ::-1]
    n0 = len(v)
    v = np.vstack([v, [[*centers[0], zs[0] - 0.4]], [[*centers[-1], zs[-1] + 0.4]]])
    caps = []
    last = (len(zs) - 1) * cols
    for k in range(cols):
        caps.append([n0, k, (k + 1) % cols])
        caps.append([n0 + 1, last + (k + 1) % cols, last + k])
    return v, np.vstack([t, np.array(caps)]), zs, centers


def superellipse_ring(cz, h, w_in, w_out, phis, n_bot=2.8, n_top=2.2):
    c, s = np.cos(phis), np.sin(phis)
    lat = np.where(c >= 0, w_out, w_in) * np.sign(c) * np.abs(c) ** (2 / 2.4)
    n = np.where(s < 0, n_bot, n_top)
    zz = cz + h * np.sign(s) * np.abs(s) ** (2 / n)
    return lat, zz


def foot_loft(sd_, stations=72, segs=48):
    tab = np.array(FOOT)
    u = hl_geom.catmull_rom(tab, 6)
    u = u[np.argsort(u[:, 0])]
    us = np.linspace(tab[0, 0], tab[-1, 0], stations)
    cols = [np.interp(us, u[:, 0], u[:, k]) for k in range(5)]
    _, zb, zt, win, wout = cols
    phis = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    pts = []
    for i in range(stations):
        h = 0.5 * (zt[i] - zb[i])
        lat, zz = superellipse_ring(0.5 * (zt[i] + zb[i]), h, win[i], wout[i], phis)
        lat = lat + 0.22 * np.sin(np.pi * np.clip((us[i] + 1) / 12.0, 0, 1))
        base = sd_.ankle[:2] + sd_.fwd[:2] * us[i]
        x = base[0] + sd_.lat[0] * lat
        y = base[1] + sd_.lat[1] * lat
        pts.append(np.stack([x, y, zz], 1))
    pts = np.array(pts)
    v = pts.reshape(-1, 3)
    t = hl_geom.grid_tris(stations, segs, wrap=True)
    n0 = len(v)
    a = sd_.ankle[:2] + sd_.fwd[:2] * us[0]
    b = sd_.ankle[:2] + sd_.fwd[:2] * us[-1]
    v = np.vstack([v, [[a[0], a[1], 0.5 * (zt[0] + zb[0])]], [[b[0], b[1], 0.5 * (zt[-1] + zb[-1])]]])
    caps = []
    last = (stations - 1) * segs
    for k in range(segs):
        caps.append([n0, (k + 1) % segs, k])
        caps.append([n0 + 1, last + k, last + (k + 1) % segs])
    t = np.vstack([t, np.array(caps)])
    nrm = hl_geom.vertex_normals(v, t)
    ctr = v.mean(0)
    if ((v - ctr) * nrm).sum() < 0:
        t = t[:, ::-1]
    return v, t, (us, zb, zt, win, wout)


def sole(prof, sd_, segs=40):
    us, zb, zt, win, wout = prof
    sel = us >= -3.85
    us, zb, win, wout = us[sel], zb[sel], win[sel], wout[sel]
    phis = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    rings = []
    m = len(us)
    for i in range(m):
        k = np.clip(min(i, m - 1 - i) / 2.0, 0.15, 1.0)
        lat, zz = superellipse_ring(zb[i] - SOLE_T * 0.5 + 0.08, SOLE_T * 0.62, (win[i] + 0.1) * k,
                                    (wout[i] + 0.1) * k, phis, 6.0, 6.0)
        lat = lat + 0.22 * np.sin(np.pi * np.clip((us[i] + 1) / 12.0, 0, 1))
        base = sd_.ankle[:2] + sd_.fwd[:2] * us[i]
        rings.append(np.stack([base[0] + sd_.lat[0] * lat, base[1] + sd_.lat[1] * lat, zz], 1))
    pts = np.array(rings)
    v = pts.reshape(-1, 3)
    t = hl_geom.grid_tris(m, segs, wrap=True)
    n0 = len(v)
    v = np.vstack([v, pts[0].mean(0)[None], pts[-1].mean(0)[None]])
    caps = []
    last = (m - 1) * segs
    for k in range(segs):
        caps.append([n0, (k + 1) % segs, k])
        caps.append([n0 + 1, last + k, last + (k + 1) % segs])
    t = np.vstack([t, np.array(caps)])
    nrm = hl_geom.vertex_normals(v, t)
    if ((v - v.mean(0)) * nrm).sum() < 0:
        t = t[:, ::-1]
    uv = np.stack([np.repeat(us, segs), np.tile(phis, m)], 1)
    uv = np.vstack([uv, [[us[0], 0], [us[-1], 0]]])
    return v, t, uv


def heel_block(sd_, segs=32, tip=0.42):
    tab = np.array(HEEL)
    z_top, z_bot = tab[0, 0], -HH
    zs = np.linspace(z_top, z_bot, 26)
    zz = np.interp(-zs, -tab[:, 0], tab[:, 0])
    u0 = np.interp(-zs, -tab[:, 0], tab[:, 1])
    u1 = np.interp(-zs, -tab[:, 0], tab[:, 2])
    hw = np.interp(-zs, -tab[:, 0], tab[:, 3])
    phis = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    rings = []
    for i, z in enumerate(zs):
        uc = 0.5 * (u0[i] + u1[i])
        hu = 0.5 * (u1[i] - u0[i])
        c, s = np.cos(phis), np.sin(phis)
        du = hu * np.sign(c) * np.abs(c) ** (2 / 3.0)
        dl = hw[i] * np.sign(s) * np.abs(s) ** (2 / 3.0)
        base = sd_.ankle[:2] + sd_.fwd[:2] * uc
        x = base[0] + sd_.fwd[0] * du + sd_.lat[0] * dl
        y = base[1] + sd_.fwd[1] * du + sd_.lat[1] * dl
        rings.append(np.stack([x, y, np.full(segs, z)], 1))
    pts = np.array(rings)
    v = pts.reshape(-1, 3)
    t = hl_geom.grid_tris(len(zs), segs, wrap=True)
    n0 = len(v)
    v = np.vstack([v, pts[0].mean(0)[None], pts[-1].mean(0)[None]])
    caps = []
    last = (len(zs) - 1) * segs
    for k in range(segs):
        caps.append([n0, k, (k + 1) % segs])
        caps.append([n0 + 1, last + (k + 1) % segs, last + k])
    t = np.vstack([t, np.array(caps)])
    nrm = hl_geom.vertex_normals(v, t)
    if ((v - v.mean(0)) * nrm).sum() < 0:
        t = t[:, ::-1]
    uv = np.stack([np.tile(phis, len(zs)), np.repeat(zs, segs)], 1)
    uv = np.vstack([uv, [[0, z_top], [0, z_bot]]])
    is_tip = v[:, 2] < z_bot + tip + 1e-6
    return v, t, uv, is_tip


def top_curve(theta):
    return hl_fit.sym_curve(TOP)(theta)


def lace_line(body, zs, centers, sd_, surf_tree):
    pts = []
    for z in np.arange(LACE_TOP, 11.9, -0.5):
        k = np.argmin(np.abs(zs - z))
        c = centers[k]
        o = np.array([[c[0], c[1], z]])
        d = np.array([[sd_.fwd[0] * 0.25, 1.0, 0.0]])
        d = d / np.linalg.norm(d)
        loc, nrm, face, dist = hl_blend.raycast(body.tree, o, d, 12.0)
        pts.append(loc[0] + nrm[0] * SHAFT_OFF)
    for uu, zz in ((0.9, 10.2), (1.7, 8.9), (2.4, 7.4)):
        base = sd_.ankle[:2] + sd_.fwd[:2] * uu
        pts.append(np.array([base[0], base[1], zz]))
    line = hl_geom.catmull_rom(np.array(pts), 6)
    q, n, f, d = hl_blend.nearest(surf_tree, line)
    return q


def dist_to_polyline(p, line):
    best = np.full(len(p), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        q = a + np.outer(t, ab)
        best = np.minimum(best, np.linalg.norm(p - q, axis=1))
    return best


class AnkleBlend:
    def __init__(self, feet_ref, side):
        sh = feet_ref["shapes"]["Feet"]
        pre = "NPC L" if side < 0 else "NPC R"
        legs = [j for j, b in enumerate(sh["bones"]) if b.startswith(pre)]
        calf = [j for j in legs if "Calf" in sh["bones"][j]]
        tris = sh["t"][np.sign(sh["v"][sh["t"]][:, :, 0].mean(1)) == side]
        self.v, self.t = sh["v"], tris
        self.tree = hl_blend.bvh(sh["v"], tris)
        self.share = sh["w"][:, calf].sum(1) / np.maximum(sh["w"][:, legs].sum(1), 1e-9)
        top = np.unique(tris)
        self.z_top = sh["v"][top, 2].max()
        top = top[sh["v"][top, 2] > self.z_top - 0.6]
        self.seam = self.share[top].mean()

    def __call__(self, points):
        loc, nrm, face, dist = hl_blend.nearest(self.tree, points)
        tri = self.t[face]
        bc = hl_blend.barycentric(loc, self.v[tri[:, 0]], self.v[tri[:, 1]], self.v[tri[:, 2]])
        z = np.asarray(points)[:, 2]
        hold = np.clip((z - SOLE_HOLD[0]) / (SOLE_HOLD[1] - SOLE_HOLD[0]), 0, 1)
        rise = np.clip((z - self.z_top + 0.5) / 2.0, 0, 1)
        k = np.clip((bc * self.share[tri]).sum(1) / self.seam, 0, 1) * hold * hold * (3 - 2 * hold)
        return np.maximum(k, rise * rise * (3 - 2 * rise))


def foot_blend(v, w, D, foot_bone, ankle):
    k = ankle(v)
    wf = np.zeros_like(w)
    wf[:, foot_bone] = 1.0
    return w * k[:, None] + wf * (1 - k)[:, None], D * k[:, None, None]


def section_centres(v, t, zs, axes):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    out = np.zeros((len(zs), len(axes)))
    for i, z in enumerate(zs):
        pts = []
        for p, q in ((a, b), (b, c), (c, a)):
            s = (p[:, 2] - z) * (q[:, 2] - z) < 0
            f = (z - p[s, 2]) / (q[s, 2] - p[s, 2])
            pts.append(p[s, :2] + (q[s, :2] - p[s, :2]) * f[:, None])
        x = np.concatenate(pts) @ np.array(axes).T
        out[i] = 0.5 * (x.min(0) + x.max(0))
    return out


def faired_offset(zs, c):
    z0, z1 = zs[0], zs[-1]
    slope = np.polyfit(zs[-6:], c[-6:], 1)[0]
    line = c[-1] + slope * (zs - z1)
    end = c[-1] + slope * (ANKLE_TURN - z1)
    s = np.clip((zs - z0) / (ANKLE_TURN - z0), 0, 1)
    turn = (2 * s ** 3 - 3 * s ** 2 + 1) * c[0] + (3 * s ** 2 - 2 * s ** 3) * end + (s ** 3 - s ** 2) * (
        ANKLE_TURN - z0) * slope
    delta = np.where(zs < ANKLE_TURN, turn, line) - c
    for _ in range(4):
        delta[1:-1] = 0.25 * delta[:-2] + 0.5 * delta[1:-1] + 0.25 * delta[2:]
    return delta


def smooth_line(z, y, sigma):
    h = z[1] - z[0]
    n = int(4 * sigma / h)
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * h / sigma) ** 2)
    lo, hi = np.polyfit(z[:8], y[:8], 1), np.polyfit(z[-8:], y[-8:], 1)
    ext = np.concatenate([np.polyval(lo, z[0] - h * np.arange(n, 0, -1)), y,
                          np.polyval(hi, z[-1] + h * np.arange(1, n + 1))])
    return np.convolve(ext, k / k.sum(), mode="valid")


def straight_seam(z, x):
    A = np.stack([np.ones_like(z), z, np.maximum(z - SEAM_KNEE, 0)], 1)
    return smooth_line(z, A @ np.linalg.lstsq(A, x, rcond=None)[0], SEAM_SIGMA)


def column_lat(grid, col, zs, lat):
    c = grid[:, col]
    order = np.argsort(c[:, 2])
    return np.interp(zs, c[order, 2], c[order, :2] @ lat)


def ankle_shift(v, t, grid, sd_):
    zs = np.linspace(ANKLE_SPAN[0], ANKLE_SPAN[1], 53)
    lat = np.array([sd_.lat[0], sd_.lat[1]])
    fwd = np.array([sd_.fwd[0], sd_.fwd[1]])
    centre = section_centres(v, t, zs, (lat, fwd))
    d_side = faired_offset(zs, centre[:, 0])
    d_front = faired_offset(zs, column_lat(grid, 0, zs, lat))
    lat3 = np.array([lat[0], lat[1], 0.0])

    def shift(p):
        p = np.array(p, dtype=np.float64)
        z = p[:, 2]
        dl = p[:, :2] @ lat - np.interp(z, zs, centre[:, 0])
        df = p[:, :2] @ fwd - np.interp(z, zs, centre[:, 1])
        k = np.clip(df / np.maximum(np.hypot(dl, df), 1e-9), 0, 1) ** 2
        d = np.interp(z, zs, d_side) * (1 - k) + np.interp(z, zs, d_front) * k
        return p + d[:, None] * lat3
    return shift


def seam_shift(grid, sd_, proxy, ankle):
    fwd = np.array([sd_.fwd[0], sd_.fwd[1]])
    curves = []
    for col, z0 in ((0, SEAM_FRONT), (grid.shape[1] // 2, SEAM_BACK)):
        c = ankle(grid[:, col])
        order = np.argsort(c[:, 2])
        zz = np.linspace(z0, SEAM_TOP, int(round((SEAM_TOP - z0) / 0.25)) + 1)
        x = np.interp(zz, c[order, 2], c[order, 0])
        ramp = np.clip((zz - z0) / 3.0, 0, 1)
        curves.append((zz, (straight_seam(zz, x) - x) * ramp * ramp * (3 - 2 * ramp)))
    zl = np.linspace(SEAM_BACK, SEAM_TOP, int(round((SEAM_TOP - SEAM_BACK) / 0.25)) + 1)
    centres = leg_centers(proxy, zl, sd_.side)

    def shift(p):
        p = np.array(p, dtype=np.float64)
        z = p[:, 2]
        rel = p[:, :2] - np.stack([np.interp(z, zl, centres[:, 0]), np.interp(z, zl, centres[:, 1])], 1)
        c = rel @ fwd / np.maximum(np.linalg.norm(rel, axis=1), 1e-9)
        p[:, 0] += (np.clip(c, 0, 1) ** 4 * np.interp(z, *curves[0]) +
                    np.clip(-c, 0, 1) ** 4 * np.interp(z, *curves[1]))
        return p
    return shift


BACK_RAIL = [(-2.75, 14.0), (-3.25, 11.0), (-3.75, 7.2), (-3.85, 4.2), (-3.65, 2.2), (-3.1, 1.15), (-1.8, 0.92),
             (-0.6, 0.85), (0.8, -0.15), (2.2, -1.5), (3.6, -3.0), (4.8, -3.92), (6.0, -4.4), (7.2, -4.45),
             (8.4, -4.4), (9.6, -4.3), (10.8, -4.15), (11.8, -4.0), (12.3, -3.8)]
FRONT_RAIL = [(3.25, 14.0), (3.45, 11.4), (3.85, 8.4), (4.3, 5.4), (4.85, 2.6), (5.5, 0.6), (6.3, -0.75),
              (7.3, -1.5), (8.4, -2.15), (9.6, -2.7), (10.8, -3.2), (11.8, -3.55), (12.3, -3.8)]
SHAFT_BOTTOM = 14.0


def rail_points(sd_, rail, start=None, lat_shift=True):
    pts = []
    if start is not None:
        pts.append(start)
        pts.append(start + np.array([0.0, 0.0, -1.6]))
        rail = [r for r in rail if r[1] <= start[2] - 2.6]
    for uu, zz in rail:
        base = sd_.ankle[:2] + sd_.fwd[:2] * uu
        lat = 0.22 * np.sin(np.pi * np.clip((uu + 1) / 12.0, 0, 1)) if lat_shift else 0.0
        pts.append([base[0] + sd_.lat[0] * lat, base[1] + sd_.lat[1] * lat, zz])
    dense = hl_geom.catmull_rom(np.array(pts), 12)
    return dense


def shaft_rings(proxy, sd_, segs, z_lo=SHAFT_BOTTOM, z_hi=48.6, step=0.34):
    zs = np.arange(z_hi, z_lo - 1e-6, -step)
    centers = leg_centers(proxy, zs, sd_.side)
    phis = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    fwd = np.array([sd_.fwd[0], sd_.fwd[1], 0.0])
    latd = np.array([sd_.lat[0], sd_.lat[1], 0.0])
    rings = []
    for c, z in zip(centers, zs):
        d = np.outer(np.cos(phis), fwd) + np.outer(np.sin(phis), latd)
        o = np.tile([c[0], c[1], z], (segs, 1))
        loc, nrm, face, dist = hl_blend.raycast(proxy.tree, o, d, 12.0)
        rad = np.where(np.isfinite(dist), dist, np.nan)
        if np.isnan(rad).any():
            good = ~np.isnan(rad)
            rad = np.interp(phis, phis[good], rad[good], period=2 * np.pi)
        rings.append(o + d * (rad + SHAFT_OFF)[:, None])
    return np.array(rings), zs, centers


def foot_rings(sd_, segs, bottom_ring, count=90):
    tab = np.array(FOOT)
    dense = hl_geom.catmull_rom(tab, 6)
    dense = dense[np.argsort(dense[:, 0])]
    back = rail_points(sd_, BACK_RAIL, bottom_ring[segs // 2])
    front = rail_points(sd_, FRONT_RAIL, bottom_ring[0])
    tb = np.linspace(0, 1, count + 1)[1:]
    bl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(back, axis=0), axis=1))])
    fl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(front, axis=0), axis=1))])
    phis = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    latd = np.array([sd_.lat[0], sd_.lat[1], 0.0])
    c0 = 0.5 * (bottom_ring[0] + bottom_ring[segs // 2])
    a0 = 0.5 * (bottom_ring[0] - bottom_ring[segs // 2])
    rel0 = bottom_ring - c0
    lat0 = rel0 @ latd
    w0_out = max(lat0.max(), 0.5)
    w0_in = max(-lat0.min(), 0.5)
    rings = []
    for t in tb:
        B = np.array([np.interp(t * bl[-1], bl, back[:, k]) for k in range(3)])
        F = np.array([np.interp(t * fl[-1], fl, front[:, k]) for k in range(3)])
        C = 0.5 * (B + F)
        A = 0.5 * (F - B)
        uc = (C[:2] - sd_.ankle[:2]) @ sd_.fwd[:2]
        kw = np.clip(t / 0.3, 0, 1)
        kw = kw * kw * (3 - 2 * kw)
        win = w0_in * (1 - kw) + np.interp(uc, dense[:, 0], dense[:, 3]) * kw
        wout = w0_out * (1 - kw) + np.interp(uc, dense[:, 0], dense[:, 4]) * kw
        taper = np.clip((1.0 - t) / 0.06, 0, 1) ** 0.6
        c, s_ = np.cos(phis), np.sin(phis)
        bottomness = np.clip((t - 0.15) / 0.35, 0, 1)
        n_exp = np.where(c < 0, 2.2 + 3.0 * bottomness, 2.2)
        cc = np.sign(c) * np.abs(c) ** (2 / n_exp)
        ss = np.sign(s_) * np.abs(s_) ** (2 / n_exp)
        w = np.where(s_ >= 0, wout, win) * taper
        ring_new = C + np.outer(cc, A) + np.outer(ss * w, latd)
        k = np.clip(t / 0.22, 0, 1)
        k = k * k * (3 - 2 * k)
        ring_old = C + np.outer(rel0 @ (a0 / max(np.dot(a0, a0), 1e-9)), A) + np.outer(lat0, latd)
        rings.append(ring_old * (1 - k) + ring_new * k)
    return np.array(rings), (dense[:, 0], dense[:, 1], dense[:, 2], dense[:, 3], dense[:, 4]), front


def boot_surface(proxy, sd_, segs=72):
    shaft, zs, centers = shaft_rings(proxy, sd_, segs)
    foot, prof, front = foot_rings(sd_, segs, shaft[-1])
    grid = np.concatenate([shaft, foot], 0)
    m = len(grid)
    j = len(shaft)
    lo, hi = max(j - 22, 1), min(j + 26, m - 2)
    wts = np.zeros(m)
    wts[lo:hi] = np.sin(np.linspace(0, np.pi, hi - lo)) ** 0.5
    for _ in range(40):
        lap = np.zeros_like(grid)
        lap[1:-1] = 0.5 * (grid[:-2] + grid[2:]) - grid[1:-1]
        grid = grid + 0.5 * wts[:, None, None] * lap
    flat = grid.reshape(-1, 3)
    up = flat[:, 2] > 12.6
    flat[up] = hl_fit.push_out(flat[up], proxy, SHAFT_OFF, iters=2)
    grid = flat.reshape(grid.shape)
    tip = grid[-1].mean(0)
    v = np.vstack([grid.reshape(-1, 3), tip[None]])
    tris = hl_geom.grid_tris(m, segs, wrap=True)
    last = (m - 1) * segs
    cap = [[m * segs, last + (k + 1) % segs, last + k] for k in range(segs)]
    t_all = np.vstack([tris, np.array(cap)])
    nr = hl_geom.vertex_normals(v, t_all)
    ctr = np.repeat(grid.mean(1), segs, 0)
    if ((v[:-1] - ctr) * nr[:-1]).sum() < 0:
        t_all = t_all[:, ::-1]
    phis = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    ring = np.linalg.norm(np.diff(np.concatenate([grid, grid[:, :1]], 1), axis=1), axis=2)
    arc = np.concatenate([np.zeros((m, 1)), np.cumsum(ring, 1)[:, :-1]], 1)
    circ = arc[:, -1] + ring[:, -1]
    arc_signed = np.where(arc > circ[:, None] / 2, arc - circ[:, None], arc)
    col = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    S = np.concatenate([np.zeros((1, segs)), np.cumsum(col, 0)], 0)
    along = S[:, 0]
    attrs = {"phi": np.append(np.tile(phis, m), 0.0), "arc": np.append(arc_signed.ravel(), 0.0),
             "s": np.append(S.ravel(), along[-1]), "row": np.append(np.repeat(np.arange(m), segs), m)}
    path = grid.mean(1)
    return v, t_all, attrs, path, along, prof, grid


def build_side(body, proxy, feet_ref, side, log, t0):
    import time
    sd_ = Side(side)
    v, t, a, path, along, prof, grid = boot_surface(proxy, sd_)
    log("%s surface %d verts %.1fs" % (sd_.tag, len(v), time.time() - t0))
    zc = v[:, 2]
    v, t, a = hl_geom.iso_cut(v, t, top_curve(a["phi"]) - zc, a)
    shift = ankle_shift(v, t, grid, sd_)
    seam = seam_shift(grid, sd_, proxy, shift)
    zk = along[np.argmin(np.abs(grid[:, 0, 2] - 27.0))]
    s_end = along[np.argmin(np.abs(grid[:, 0, 2] - 6.0))]
    ds = np.clip(a["s"], zk, s_end) - a["s"]
    d_lace = np.sqrt(ds ** 2 + a["arc"] ** 2)
    tv, tt, ta = hl_geom.iso_cut(v, t, 1.2 - d_lace, {})
    tn = hl_geom.vertex_normals(tv, tt)
    tv = tv - tn * 0.13
    tuv = hl_blend.unwrap(tv, tt)
    def settle(p):
        before = shift(p)
        out = seam(before)
        s = (out[:, 2] > 12.6) & (np.linalg.norm(out - before, axis=1) > 1e-9)
        d0 = proxy.signed_distance(before[s])[0]
        d1, loc, nrm, _ = proxy.signed_distance(out[s])
        out[s] += hl_geom.normalize(out[s] - loc) * (d0 - d1)[:, None]
        return out

    tv = settle(tv)
    sel = (along >= zk) & (along <= s_end)
    line = settle(grid[sel, 0, :])
    v, t, a = hl_geom.iso_cut(v, t, d_lace - LACE_GAP, dict(a, dl=d_lace))
    v = settle(v)
    log("%s cuts %.1fs" % (sd_.tag, time.time() - t0))
    foot_bone = proxy.bones.index(sd_.foot)
    ankle = AnkleBlend(feet_ref, side)
    w, D = bind(v, t, proxy, smooth_iters=3, edge_body=body)
    w, D = foot_blend(v, w, D, foot_bone, ankle)
    tag = sd_.tag

    def halves(outer):
        out = []
        for side_name, sgn in (("a", 1), ("b", -1)):
            aa = dict(outer.attrs, w=outer.w, D=outer.D)
            vv, tt2, aa = hl_geom.iso_cut(outer.v, outer.t, sgn * np.sin(aa["phi"]), aa)
            uv = np.stack([np.abs(aa["arc"]), aa["s"]], 1)
            key = "boot%s_%s" % (tag, side_name)
            p = Part(key, "Boots", vv, tt2, uv, key, "leather", aa["w"], aa["D"])
            p.normals = hl_geom.normalize(aa["vn"])
            out.append(p)
        return out

    parts = solid_parts("boot" + tag, "Boots", v, t, np.stack([a["arc"], a["s"]], 1), w, D, 0.28,
                        ("boot" + tag, "patch:lining", "patch:edge"), ("leather", "lining", "edge"), rim_segments=2,
                        attrs={"phi": a["phi"], "arc": a["arc"], "s": a["s"]}, split_outer=halves)
    tw, tD = bind(tv, tt, proxy, smooth_iters=2)
    tw, tD = foot_blend(tv, tw, tD, foot_bone, ankle)
    parts.append(Part("tongue" + sd_.tag, "Boots", tv, tt, tuv, "tongue" + sd_.tag, "fabric", tw, tD))
    log("%s upper %.1fs" % (sd_.tag, time.time() - t0))
    ov, ot, ouv = sole(prof, sd_)
    hv, ht, huv, is_tip = heel_block(sd_)
    rig = np.zeros((1, len(proxy.bones)))
    rig[0, foot_bone] = 1.0
    nz = len(proxy.slider_names)
    parts.append(Part("sole" + sd_.tag, "Boots", ov, ot, ouv, "patch:sole", "sole", np.repeat(rig, len(ov), 0),
                      np.zeros((len(ov), nz, 3), np.float32)))
    tip_tris = is_tip[ht].all(1)
    zh = np.zeros((len(hv), nz, 3), np.float32)
    parts.append(sub_part("heel" + sd_.tag, "Boots", hv, ht, huv, "patch:sole", "sole",
                          np.repeat(rig, len(hv), 0), zh, ~tip_tris))
    parts.append(sub_part("heeltip" + sd_.tag, "BootsBrass", hv, ht, huv, "patch:brass", "brass",
                          np.repeat(rig, len(hv), 0), zh, tip_tris))
    parts += lacing(line, v, t, proxy, foot_bone, sd_.tag, ankle)
    log("%s details %.1fs" % (sd_.tag, time.time() - t0))
    for p in parts:
        p.side = sd_.tag
    return parts


def trim_lining(p, floor):
    keep = p.v[p.t][:, :, 2].max(1) >= floor
    q = sub_part(p.name, p.shape, p.v, p.t, p.uv, p.island, p.mat, p.w, p.D, keep, p.attrs)
    q.normals = hl_geom.normalize(q.attrs["vn"])
    q.side = p.side
    return q


def build(body, proxy=None, log=print, feet_ref=None):
    import os
    import pickle
    import time
    t0 = time.time()
    feet_ref = feet_ref or pickle.load(open(os.path.join(r"D:\Dev\Skyrim MOD\work\hollow-lantern\ref", "feet.pkl"),
                                            "rb"))
    parts = build_side(body, proxy, feet_ref, -1, log, t0) + build_side(body, proxy, feet_ref, 1, log, t0)
    parts = [trim_lining(p, LINING_FLOOR) if p.name.endswith("_in") else p for p in parts]
    for p in parts:
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        if p.mat == "brass":
            p.shape = "BootsBrass"
        elif p.name.endswith("_in"):
            p.shape = "BootsLining" + p.side
        else:
            p.shape = "Boots" + p.side
    return parts, None


def lacing(line, v, t, proxy, foot_bone, tag, ankle):
    import piece_corset
    tree = hl_blend.bvh(v, t)
    length = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    stations = np.arange(0.9, length[-1] - 0.6, EYELET_STEP)
    eyes = {-1: [], 1: []}
    for s in stations:
        p = np.array([np.interp(s, length, line[:, k]) for k in range(3)])
        k = np.searchsorted(length, s)
        k = np.clip(k, 1, len(line) - 1)
        tang = hl_geom.normalize(line[k] - line[k - 1])
        q, n, face, dist = hl_blend.nearest(tree, p[None])
        n0 = n[0]
        side_dir = hl_geom.normalize(np.cross(tang, n0))
        for sgn in (-1, 1):
            target = p + side_dir * sgn * (LACE_GAP + 0.38)
            q2, n2, f2, d2 = hl_blend.nearest(tree, target[None])
            nn = n2[0] if np.dot(n2[0], n0) > 0 else -n2[0]
            eyes[sgn].append((q2[0], nn, tang))
    rings = []
    for sgn in (-1, 1):
        for p, n, tang in eyes[sgn]:
            rings.append(piece_corset.torus(p + n * 0.03, n, tang, 0.17, 0.055))
    ev, et, euv = piece_corset.merge(rings)
    paths = []
    L, R = eyes[-1], eyes[1]
    for i in range(len(L) - 1):
        for A, B, lift in ((L, R, 0.15), (R, L, 0.06)):
            p0 = A[i][0] - A[i][1] * 0.02
            p1 = B[i + 1][0] - B[i + 1][1] * 0.02
            nm = hl_geom.normalize(A[i][1] + B[i + 1][1])
            ctrl = [p0, p0 + A[i][1] * 0.1, 0.5 * (p0 + p1) + nm * lift, p1 + B[i + 1][1] * 0.1, p1]
            paths.append(hl_geom.catmull_rom(np.array(ctrl), 6))
    top_l, top_r = L[0], R[0]
    nk = hl_geom.normalize(top_l[1] + top_r[1])
    knot = 0.5 * (top_l[0] + top_r[0]) + nk * 0.28
    for A in (top_l, top_r):
        p0 = A[0] - A[1] * 0.02
        paths.append(hl_geom.catmull_rom(np.array([p0, p0 + A[1] * 0.12, knot]), 6))
    down = -hl_geom.normalize(top_l[2])
    for sgn in (-1, 1):
        side = hl_geom.normalize(np.cross(top_l[2], nk)) * sgn
        ctrl = [knot, knot + side * 0.3 + down * 0.5 + nk * 0.08, knot + side * 0.45 + down * 1.4 + nk * 0.1,
                knot + side * 0.4 + down * 2.2 + nk * 0.08]
        paths.append(hl_geom.catmull_rom(np.array(ctrl), 8))
    cords = []
    for path in paths:
        cv, ct, cuv, _ = hl_geom.sweep(path, np.full(len(path), 0.065), 7)
        cords.append((cv, ct, cuv))
    kv, kt, kuv, _ = hl_geom.sweep(np.array([knot - nk * 0.05 - top_l[2] * 0.12, knot, knot + top_l[2] * 0.12]),
                                   np.array([0.1, 0.15, 0.1]), 10, cap_start=True, cap_end=True)
    cords.append((kv, kt, kuv))
    cv, ct, cuv = piece_corset.merge(cords)
    tips = []
    for path in paths[-2:]:
        end = path[-1]
        d = hl_geom.normalize(path[-1] - path[-2])
        tv, tt, tuv, _ = hl_geom.sweep(np.array([end - d * 0.04, end + d * 0.3]), np.array([0.075, 0.06]), 8,
                                       cap_end=True)
        tips.append((tv, tt, tuv))
    av, at, auv = piece_corset.merge(tips)
    out = []
    for name, mv, mt, muv, isl, mat in (("boot_eyelets", ev, et, euv, "patch:brass", "brass"),
                                        ("boot_lacing", cv, ct, cuv, "patch:cord", "cord"),
                                        ("boot_aglets", av, at, auv, "patch:brass", "brass")):
        mw, mD = bind(mv, mt, proxy, smooth_iters=4)
        mw, mD = foot_blend(mv, mw, mD, foot_bone, ankle)
        out.append(Part(name + tag, "Boots", mv, mt, muv, isl, mat, mw, mD))
    return out

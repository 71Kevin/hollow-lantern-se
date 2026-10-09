import math

import bmesh
import bpy
import numpy as np

import hl_blend
import hl_geom
import piece_corset
from hl_parts import Part

SCALE = 2.0
PLACE = np.array([0.1, -4.1, 1.0])
HANDLE = (-3.3, 3.0)
HANDLE_R = 1.3
C = np.array([0.0, -5.3, 0.0])
R = 2.75
H = 2.2
WALL = 0.26
LOBES = 9
BAIL_Y = -4.45
FACE = [
    [(-1.38, 0.52), (-0.42, 0.34), (-0.96, 1.28)],
    [(1.38, 0.52), (0.96, 1.28), (0.42, 0.34)],
    [(-0.23, -0.06), (0.23, -0.06), (0.0, 0.4)],
    [(-1.58, -0.28), (-1.22, -0.93), (-0.56, -1.2), (-0.05, -1.24), (0.24, -0.94), (0.55, -1.2), (1.22, -0.93),
     (1.58, -0.28), (0.96, -0.6), (0.7, -0.38), (0.45, -0.66), (-0.45, -0.66), (-0.7, -0.38), (-0.96, -0.6)],
]


def shell_point(theta, phi, inner=False):
    lobe = 1.0 + 0.085 * (np.abs(np.cos(LOBES * theta / 2.0)) ** 0.7 - 0.55) * np.sin(phi) ** 1.4
    r = R * np.sin(phi) ** 0.92 * lobe
    dimple = 0.42 * np.exp(-(phi / 0.38) ** 2) - 0.3 * np.exp(-((np.pi - phi) / 0.4) ** 2)
    y = C[1] + H * np.cos(phi) * (1 - 0.1 * np.sin(phi) ** 4) - dimple * H * 0.5
    x = r * np.cos(theta)
    p = np.stack([x, np.broadcast_to(y, x.shape), r * np.sin(theta)], -1)
    if inner:
        q = C + (p - C) * np.array([1 - WALL / R, 1 - WALL / H, 1 - WALL / R])
        return q
    return p


def shell_mesh(rings=40, segs=72):
    phis = np.linspace(0.05, np.pi - 0.05, rings)
    thetas = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    out = []
    for inner in (False, True):
        P = shell_point(thetas[None, :], phis[:, None], inner).reshape(-1, 3)
        top = shell_point(np.array([0.0]), np.array([0.0]), inner)[0]
        bot = shell_point(np.array([0.0]), np.array([np.pi]), inner)[0]
        top[0] = top[2] = bot[0] = bot[2] = 0.0
        v = np.vstack([P, top, bot])
        t = hl_geom.grid_tris(rings, segs, wrap=True)
        n0 = len(P)
        caps = []
        last = (rings - 1) * segs
        for k in range(segs):
            caps.append([n0, (k + 1) % segs, k])
            caps.append([n0 + 1, last + k, last + (k + 1) % segs])
        t = np.vstack([t, np.array(caps)])
        nr = hl_geom.vertex_normals(v, t)
        sgn = ((v - C) * nr).sum()
        if (sgn < 0) != inner:
            t = t[:, ::-1]
        uv = np.vstack([np.stack([np.tile(np.mod(thetas - np.pi, 2 * np.pi), rings) * R,
                                  np.repeat(phis, segs) * H], 1),
                        [[0, 0], [0, np.pi * H]]])
        out.append((v, t, uv))
    return out


def prism(poly, axis, depth0, depth1, up=np.array([0.0, 1.0, 0.0]), right=None):
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    a = np.asarray(axis, dtype=np.float64)
    if right is None:
        right = np.cross(up, a)
    pts = []
    for d in (depth0, depth1):
        for u, vv in poly:
            pts.append(C + a * d + right * u + up * vv)
    v = np.array(pts)
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        faces.append([k, k1, n + k1, n + k])
    faces.append(list(range(n))[::-1])
    faces.append(list(range(n, 2 * n)))
    return v, faces


def crescent(r=0.85, ri=0.75, d=0.42, steps=36, rot=-25.0, centre=(0.1, -0.62)):
    xi = (r * r - ri * ri + d * d) / (2 * d)
    yi = math.sqrt(max(r * r - xi * xi, 0.0))
    a0 = math.atan2(yi, xi)
    b0 = math.atan2(yi, xi - d)
    pts = []
    for k in range(steps + 1):
        a = a0 + (2 * math.pi - 2 * a0) * k / steps
        pts.append((r * math.cos(a), r * math.sin(a)))
    for k in range(1, steps):
        b = (2 * math.pi - b0) - (2 * math.pi - 2 * b0) * k / steps
        pts.append((d + ri * math.cos(b), ri * math.sin(b)))
    area = 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    if area < 0:
        pts = pts[::-1]
    c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    return [(x * c - y * s_ + centre[0], x * s_ + y * c + centre[1]) for x, y in pts]


def carve(shell):
    (ov, ot, ouv), (iv, it, iuv) = shell
    me = bpy.data.meshes.new("pumpkin")
    v = np.vstack([ov, iv])
    t = np.vstack([ot, it + len(ov)])
    me.from_pydata([tuple(p) for p in v], [], [tuple(int(i) for i in f) for f in t])
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    uv = np.vstack([ouv, iuv * 0.97])
    loops = np.zeros(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loops)
    luv = uv[loops].reshape(-1, 3, 2)
    corner = loops.reshape(-1, 3)
    poles = np.isin(corner, [len(ov) - 2, len(ov) - 1, len(v) - 2, len(v) - 1])
    for faces, period in ((slice(0, len(ot)), 2 * np.pi * R), (slice(len(ot), None), 2 * np.pi * R * 0.97)):
        u = luv[faces, :, 0]
        pole = poles[faces]
        ring = np.where(pole, np.nan, u)
        wrap = (np.nanmax(ring, 1) - np.nanmin(ring, 1)) > period / 2
        u[wrap] = np.where(u[wrap] < period / 2, u[wrap] + period, u[wrap])
        ring = np.where(pole, np.nan, u)
        u[pole] = np.nanmean(ring, 1)[np.nonzero(pole)[0]]
        luv[faces, :, 0] = u
    uvl.data.foreach_set("uv", luv.ravel().astype(np.float32))
    layer = me.attributes.new("part", "INT", "FACE")
    vals = np.concatenate([np.zeros(len(ot), np.int32), np.ones(len(it), np.int32)])
    layer.data.foreach_set("value", vals)
    ob = bpy.data.objects.new("pumpkin", me)
    bpy.context.scene.collection.objects.link(ob)
    cut_v, cut_f = [], []
    for poly in FACE:
        pv, pf = prism(poly, np.array([1.0, 0.0, 0.0]), 0.3, R + 1.5, right=np.array([0.0, 0.0, -1.0]))
        cut_f += [[i + len(cut_v) for i in f] for f in pf]
        cut_v += pv.tolist()
    pv, pf = prism(crescent(), np.array([0.0, 0.0, 1.0]), 0.3, R + 1.5, right=np.array([1.0, 0.0, 0.0]))
    cut_f += [[i + len(cut_v) for i in f] for f in pf]
    cut_v += pv.tolist()
    cme = bpy.data.meshes.new("cutter")
    cme.from_pydata(cut_v, [], cut_f)
    cme.update()
    cl = cme.attributes.new("part", "INT", "FACE")
    cl.data.foreach_set("value", np.full(len(cme.polygons), 2, np.int32))
    cme.uv_layers.new(name="UVMap")
    cob = bpy.data.objects.new("cutter", cme)
    bpy.context.scene.collection.objects.link(cob)
    mod = ob.modifiers.new("carve", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.object = cob
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    em = ob.evaluated_get(dg).to_mesh()
    bm = bmesh.new()
    bm.from_mesh(em)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    uv_layer = bm.loops.layers.uv.active
    part_layer = bm.faces.layers.int.get("part")
    groups = {0: [], 1: [], 2: []}
    for f in bm.faces:
        pid = f[part_layer] if part_layer is not None else 0
        groups.setdefault(pid, []).append([(tuple(l.vert.co), tuple(l[uv_layer].uv)) for l in f.loops])
    bm.free()
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.objects.remove(cob)
    out = {}
    for pid, tris in groups.items():
        verts = {}
        vv, uu, tt = [], [], []
        for tri in tris:
            idx = []
            for co, uvc in tri:
                key = (round(co[0], 5), round(co[1], 5), round(co[2], 5), round(uvc[0], 5), round(uvc[1], 5))
                if key not in verts:
                    verts[key] = len(vv)
                    vv.append(co)
                    uu.append(uvc)
                idx.append(verts[key])
            tt.append(idx)
        if tt:
            out[pid] = (np.array(vv), np.array(tt), np.array(uu))
    return out


def stem():
    base = np.array([0.0, C[1] + H * 0.79, 0.0])
    path = np.array([base, base + [0.05, 0.55, 0.02], base + [0.2, 1.05, 0.08], base + [0.5, 1.35, 0.15],
                     base + [0.82, 1.42, 0.2]])
    path = hl_geom.catmull_rom(path, 6)
    n = len(path)
    radii = np.linspace(0.42, 0.24, n)
    radii[0] = 0.62

    def prof(f, ang):
        return 1.0 + 0.12 * np.cos(6 * ang) * (1 - f)

    v, t, uv, _ = hl_geom.sweep(path, radii, 18, profile=prof, cap_end=True)
    return v, t, uv


def bail():
    ang = np.linspace(-np.pi * 0.5, np.pi * 0.5, 41)
    hz = R + 0.16
    hy = -BAIL_Y
    path = np.stack([np.zeros_like(ang), BAIL_Y + hy * np.cos(ang), hz * np.sin(ang)], 1)
    v, t, uv, _ = hl_geom.sweep(path, np.full(len(path), 0.11), 10)
    parts = [(v, t, uv)]
    for sgn in (-1, 1):
        c = np.array([0.0, BAIL_Y, sgn * (R * 0.985)])
        dv, dt, duv, _ = hl_geom.sweep(np.array([c - [0, 0, sgn * 0.08], c + [0, 0, sgn * 0.12], c + [0, 0, sgn * 0.2]]),
                                       np.array([0.3, 0.27, 0.14]), 14, cap_end=True)
        parts.append((dv, dt, duv))
    return parts


def handle():
    x, z = PLACE[0], PLACE[2]
    y0, y1 = HANDLE
    axis = np.array([[x, y, z] for y in np.linspace(y0, y1, 10)])
    wood = hl_geom.sweep(axis, np.full(len(axis), HANDLE_R), 20, cap_start=True, cap_end=True)[:3]
    brass = [hl_geom.sweep(np.array([[x, ya, z], [x, yb, z]]), np.full(2, HANDLE_R + 0.1), 20, cap_start=True,
                           cap_end=True)[:3] for ya, yb in ((y0 - 0.1, y0 + 0.4), (y1 - 0.4, y1 + 0.1))]
    brass.append(piece_corset.torus(np.array([x, PLACE[1] + 0.2, z]), np.array([0.0, 0.0, 1.0]),
                                    np.array([0.0, 1.0, 0.0]), 0.55, 0.13, seg_major=20, seg_minor=8))
    turns = 7
    n = turns * 16 + 1
    w = np.linspace(0, 2 * np.pi * turns, n)
    r = HANDLE_R + 0.03
    hel = np.stack([x + np.cos(w) * r, np.linspace(y0 + 0.6, y1 - 0.6, n), z + np.sin(w) * r], 1)
    cord = hl_geom.sweep(hel, np.full(n, 0.09), 6)[:3]

    def unplace(mesh):
        v, t, uv = mesh
        return (np.asarray(v) - PLACE) / SCALE, t, uv

    return unplace(wood), [unplace(m) for m in brass], unplace(cord)


def candle():
    y0 = C[1] - H * 0.88 + WALL
    path = np.array([[0.0, y0, 0.0], [0.0, y0 + 0.6, 0.0], [0.0, y0 + 1.15, 0.0]])
    v, t, uv, _ = hl_geom.sweep(path, np.array([0.44, 0.42, 0.4]), 16, cap_start=True, cap_end=True)
    wick_v, wick_t, wick_uv, _ = hl_geom.sweep(np.array([[0.0, y0 + 1.14, 0.0], [0.02, y0 + 1.36, 0.0]]),
                                               np.array([0.04, 0.03]), 6, cap_end=True)
    return (v, t, uv), (wick_v, wick_t, wick_uv), np.array([0.0, y0 + 1.75, 0.0])


def flame_cards(base, length=0.95, width=0.5, stations=6):
    verts, tris, uvs, normals, colors = [], [], [], [], []
    for k in range(3):
        th = math.pi * k / 3
        wdir = np.array([math.cos(th), 0.0, math.sin(th)])
        nrm = np.cross(np.array([0.0, 1.0, 0.0]), wdir)
        start = len(verts)
        for i in range(stations + 1):
            s = i / stations
            q = base + np.array([0.0, length * s, 0.0])
            half = 0.5 * width * (1 - s ** 1.7) * (0.75 + 0.25 * math.sin(math.pi * s)) + 0.015
            for x in (-1.0, 0.0, 1.0):
                verts.append(q + wdir * x * half)
                uvs.append((0.5 + 0.5 * x, 1.0 - s))
                normals.append(nrm)
                colors.append((1.0, 1.0, 1.0, (1 - abs(x)) ** 0.7 * min(1.0, 0.35 + s / 0.15) * (1 - s) ** 0.6))
        for i in range(stations):
            for c in range(2):
                a = start + i * 3 + c
                tris.append([a, a + 1, a + 4])
                tris.append([a, a + 4, a + 3])
    return np.array(verts), np.array(tris), np.array(uvs), np.array(normals), np.array(colors)


def halo(center, radius=1.1, segs=16):
    verts, tris, uvs, normals, colors = [], [], [], [], []
    for k in range(3):
        th = math.pi * k / 3
        a = np.array([math.cos(th), 0.0, math.sin(th)])
        b = np.array([0.0, 1.0, 0.0])
        n = np.cross(a, b)
        start = len(verts)
        verts.append(center)
        uvs.append((0.5, 0.5))
        normals.append(n)
        colors.append((1, 1, 1, 1))
        for i in range(segs):
            ang = 2 * math.pi * i / segs
            verts.append(center + radius * (math.cos(ang) * a + math.sin(ang) * b))
            uvs.append((0.5 + 0.5 * math.cos(ang), 0.5 + 0.5 * math.sin(ang)))
            normals.append(n)
            colors.append((1, 1, 1, 0))
        for i in range(segs):
            tris.append([start, start + 1 + i, start + 1 + (i + 1) % segs])
    return np.array(verts), np.array(tris), np.array(uvs), np.array(normals), np.array(colors)


def build(body=None, proxy=None, log=print):
    shell = shell_mesh()
    carved = carve(shell)
    parts = []
    names = {0: ("lantern_skin", "Lantern", "lantern", "pumpkin"), 1: ("lantern_inner", "LanternGlow", "patch:flesh",
                                                                    "flesh"),
             2: ("lantern_walls", "LanternGlow", "patch:flesh", "flesh")}
    for pid, (v, t, uv) in carved.items():
        name, shape, isl, mat = names[pid]
        if pid == 2:
            uv = np.stack([v[:, 1] + v[:, 0], v[:, 2] + v[:, 0]], 1)
        parts.append(Part(name, shape, v, t, uv, isl, mat))
    sv, st, suv = stem()
    parts.append(Part("lantern_stem", "Lantern", sv, st, suv, "patch:stem", "stem"))
    for k, (v, t, uv) in enumerate(bail()):
        parts.append(Part("lantern_bail%d" % k, "LanternBrass", v, t, uv, "patch:brass", "brass"))
    wood, fittings, cord = handle()
    parts.append(Part("lantern_handle", "Lantern", *wood, "patch:ebony", "wood"))
    for k, (v, t, uv) in enumerate(fittings):
        parts.append(Part("lantern_fitting%d" % k, "LanternBrass", v, t, uv, "patch:brass", "brass"))
    parts.append(Part("lantern_grip", "Lantern", *cord, "patch:cord", "cord"))
    (cv, ct, cuv), (wv, wt, wuv), flame_base = candle()
    parts.append(Part("lantern_candle", "LanternGlow", cv, ct, cuv, "patch:wax", "wax"))
    parts.append(Part("lantern_wick", "Lantern", wv, wt, wuv, "patch:edge", "edge"))
    for p in parts:
        p.static = True
        p.bones = []
        p.sliders = []
    fx = {"flame": flame_cards(flame_base - np.array([0.0, 0.42, 0.0])), "halo": halo(flame_base, 1.15),
          "light": flame_base + np.array([0.0, 0.1, 0.0]), "scale": SCALE, "offset": PLACE}
    return parts, fx

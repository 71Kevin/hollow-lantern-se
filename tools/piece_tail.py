import math

import bmesh
import bpy
import numpy as np

import hl_blend
import hl_export
import hl_geom
from hl_parts import Part, bind

PATH = [(0.0, -9.35, 76.3), (0.0, -12.5, 75.9), (0.3, -15.4, 73.6), (0.7, -16.9, 68.6), (0.9, -17.2, 62.4),
        (0.5, -16.4, 56.0), (-0.3, -15.1, 50.4), (-0.9, -14.1, 46.4), (-1.1, -13.6, 44.0)]
BONES = 10
PREFIX = "HL Tail %02d"
R_ROOT = 0.86
R_NECK = 0.25
SPADE_LEN = 3.7
SPADE_W = 2.9
SPADE_T = 0.32
PARENT = "NPC Pelvis [Pelv]"
CHAIN = ["NPC", "NPC Root [Root]", "NPC COM [COM ]", "CME Body [Body]", "CME LBody [LBody]", "NPC Pelvis [Pelv]"]


def tail_path(samples=140):
    raw = hl_geom.catmull_rom(np.array(PATH), 16)
    return hl_geom.resample_polyline(raw, samples)


def bone_frames(path):
    lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    total = lengths[-1] - SPADE_LEN * 0.5
    stations = np.linspace(0, total, BONES)
    pts = np.stack([np.interp(stations, lengths, path[:, k]) for k in range(3)], 1)
    mats = []
    for i, p in enumerate(pts):
        a = pts[min(i + 1, BONES - 1)] - pts[max(i - 1, 0)]
        x = hl_geom.normalize(a)
        side = np.array([1.0, 0.0, 0.0])
        z = hl_geom.normalize(np.cross(x, side))
        y = np.cross(z, x)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, p
        mats.append(m)
    return stations, mats


def spade(base, direction, side):
    d = hl_geom.normalize(direction)
    s = hl_geom.normalize(side - d * np.dot(side, d))
    n = np.cross(d, s)
    outline = []
    steps = 24
    for k in range(steps + 1):
        u = k / steps
        if u < 0.35:
            w = 0.18 + (SPADE_W * 0.5 - 0.18) * math.sin(u / 0.35 * math.pi * 0.5) ** 1.3
        else:
            w = SPADE_W * 0.5 * (1 - (u - 0.35) / 0.65) ** 0.85
        barb = 0.22 * math.exp(-((u - 0.3) / 0.07) ** 2)
        outline.append((u * SPADE_LEN - barb * 0.4, w + barb))
    top, bottom = [], []
    for along, w in outline:
        for sgn, dst in ((1, top), (-1, bottom)):
            dst.append((along, w * sgn))
    ring = top + bottom[::-1][1:-1]
    v = []
    for layer, off in ((0, SPADE_T * 0.5), (1, -SPADE_T * 0.5)):
        for along, lat in ring:
            taper = 1 - 0.5 * (abs(lat) / (SPADE_W * 0.5)) ** 2
            v.append(base + d * along + s * lat + n * off * taper)
    v = np.array(v)
    m = len(ring)
    center_top = base + d * SPADE_LEN * 0.42 + n * SPADE_T * 0.55
    center_bot = base + d * SPADE_LEN * 0.42 - n * SPADE_T * 0.55
    v = np.vstack([v, center_top, center_bot])
    tris = []
    for k in range(m):
        k1 = (k + 1) % m
        tris.append([2 * m, k, k1])
        tris.append([2 * m + 1, m + k1, m + k])
        tris.append([k, m + k, m + k1])
        tris.append([k, m + k1, k1])
    t = np.array(tris)
    nr = hl_geom.vertex_normals(v, t)
    if ((v - v.mean(0)) * nr).sum() < 0:
        t = t[:, ::-1]
    uv = np.stack([(v - base) @ d, (v - base) @ s], 1)
    return v, t, uv


def collision_proxy(proxy, ratio=0.06):
    v, t = proxy.v, proxy.t
    keep = (v[t][:, :, 2].min(1) > 30.0) & (v[t][:, :, 2].max(1) < 84.0)
    arm = proxy.mask_bones(("UpperArm", "Forearm", "Hand", "Clavicle"))
    keep &= arm[t].max(1) < 0.05
    tt = t[keep]
    used = np.unique(tt)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    sv, st = v[used], remap[tt]
    ob = hl_blend.mesh_object("colproxy", sv, st, smooth=False)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.ratio = ratio
    dg = bpy.context.evaluated_depsgraph_get()
    em = ob.evaluated_get(dg).to_mesh()
    bm = bmesh.new()
    bm.from_mesh(em)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.index_update()
    cv = np.array([p.co[:] for p in bm.verts])
    ct = np.array([[p.index for p in f.verts] for f in bm.faces])
    bm.free()
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob)
    return cv, ct


def build(body, proxy=None, log=print, skel=None):
    import os
    import pickle
    skel = skel or pickle.load(open(os.path.join(r"D:\Dev\Skyrim MOD\work\hollow-lantern\ref", "skeleton.pkl"), "rb"))
    path = tail_path()
    lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    total = lengths[-1]
    stations, mats = bone_frames(path)
    body_len = total
    s = lengths / total
    radii = R_NECK + (R_ROOT - R_NECK) * (1 - s) ** 1.15
    radii[-6:] = np.linspace(radii[-6], R_NECK * 0.9, 6)

    def prof(f, ang):
        return 1.0 + 0.03 * np.cos(2 * ang)

    tv, tt, tuv, cols = hl_geom.sweep(path, radii, 14, profile=prof, cap_start=True, up=(1, 0, 0))
    tuv[:len(path) * cols, 0] *= 2 * np.pi * R_ROOT
    tang = hl_geom.normalize(path[-1] - path[-4])
    sv, st, suv = spade(path[-1] - tang * 0.15, tang, np.array([1.0, 0.0, 0.0]))
    ring_c = path[-1] - tang * 0.05
    rv, rt, ruv, _ = hl_geom.sweep(np.array([ring_c - tang * 0.18, ring_c, ring_c + tang * 0.18]),
                                   np.array([R_NECK * 1.7, R_NECK * 1.95, R_NECK * 1.7]), 16, up=(1, 0, 0))
    bone_names = [PREFIX % (i + 1) for i in range(BONES)]

    def chain_weights(pts):
        rel = np.array([np.argmin(np.linalg.norm(path - p, axis=1)) for p in pts])
        sp = lengths[rel]
        w = np.zeros((len(pts), BONES))
        k = np.clip(np.searchsorted(stations, sp) - 1, 0, BONES - 2)
        f = np.clip((sp - stations[k]) / (stations[k + 1] - stations[k]), 0, 1)
        w[np.arange(len(pts)), k] = 1 - f
        w[np.arange(len(pts)), k + 1] = f
        w[sp >= stations[-1], :] = 0
        w[sp >= stations[-1], BONES - 1] = 1
        return w

    parts = []
    for name, v, t, uv, isl, mat, shape in (("tail", tv, tt, tuv, "tail", "leather", "Tail"),
                                             ("tail_spade", sv, st, suv, "spade", "leather", "Tail"),
                                             ("tail_ring", rv, rt, ruv, "patch:brass", "brass", "TailBrass")):
        p = Part(name, shape, v, t, uv, isl, mat, chain_weights(v), None)
        p.bones = list(bone_names)
        p.sliders = []
        parts.append(p)
    cv, ct = collision_proxy(proxy)
    cw, cD = bind(cv, ct, proxy)
    cw, cbones = hl_export.collapse_to_smp_bones(cw, proxy.bones)
    cp = Part("tail_collision", "TailBody", cv, ct, np.zeros((len(cv), 2)), "patch:edge", "collision", cw, cD)
    cp.bones = cbones
    cp.sliders = list(proxy.slider_names)
    cp.uv = np.stack([cv[:, 0], cv[:, 2]], 1)
    parts.append(cp)
    globals_ = {}
    for name in CHAIN:
        globals_[name] = skel[name]
    for name, m in zip(bone_names, mats):
        globals_[name] = m
    hierarchy = [(CHAIN[i], CHAIN[i - 1] if i else None) for i in range(len(CHAIN))]
    hierarchy += [(bone_names[i], bone_names[i - 1] if i else PARENT) for i in range(BONES)]
    fx = {"bones": bone_names, "globals": globals_, "hierarchy": hierarchy}
    log("tail %d verts, proxy %d tris" % (len(tv) + len(sv), len(ct)))
    return parts, fx

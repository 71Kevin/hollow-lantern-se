import os
import pickle

import numpy as np

import hl_blend
import hl_geom
import hl_nif
from hl_parts import Part
from hl_preset import MODS

HEAD_NIF = os.path.join(MODS, r"[BODY] Expressive Facegen Morphs\meshes\actors\character\character assets"
                              r"\femalehead.nif")
CTRL = [(0.0, 0.0, 0.0), (-0.6, -0.3, 2.2), (-1.5, -1.6, 4.3), (-2.8, -3.9, 5.6), (-3.8, -6.5, 5.6),
        (-4.2, -8.4, 6.7)]
BASE_X = 3.6
BASE_Y = 3.8
R0 = 1.15
RIDGES = 12
CUFF_S = 0.3


def head_mesh():
    f = hl_nif.NifFile(HEAD_NIF)
    sh = f.shapes[0]
    xf = hl_nif.xf_to_mat(sh.transform)
    v = np.array(sh.verts)
    v = v @ xf[:3, :3].T + xf[:3, 3]
    return v, np.array(sh.tris)


def head_skin():
    f = hl_nif.NifFile(HEAD_NIF)
    d = hl_nif.read_shape(f.shapes[0])
    v, t = head_mesh()
    return v, t, d["w"], list(d["bones"])


def horn(side, tree):
    o = np.array([BASE_X * side, BASE_Y, 140.0])
    loc, nrm, face, dist = hl_blend.raycast(tree, o[None], np.array([[0, 0, -1.0]]), 40.0)
    base = loc[0] - np.array([0, 0, 0.55])
    ctrl = np.array([base + np.array([-c[0] * side, c[1], c[2]]) for c in CTRL])
    path = hl_geom.catmull_rom(ctrl, 14)
    path = hl_geom.resample_polyline(path, 110)
    n = len(path)
    s = np.linspace(0, 1, n)
    radii = R0 * (1 - s) ** 0.8 + 0.05
    seg = 20

    def profile(f, ang):
        ridge = 1.0 + 0.07 * (1 - f) ** 1.5 * np.sin(np.pi * f * RIDGES * 1.6) ** 2 * (f < 0.75)
        return ridge * np.sqrt(1.0 / (np.cos(ang) ** 2 + (np.sin(ang) / 0.82) ** 2))

    v, t, uv, cols = hl_geom.sweep(path, radii, seg, profile=profile, up=(0, 1, 0), cap_start=True, cap_end=True,
                                   twist=np.linspace(0, 0.9 * side, n))
    lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    uv[:len(path) * cols, 0] *= 2 * np.pi * R0 * 0.9
    k = int(CUFF_S * (n - 1))
    tang, nr, bn = hl_geom.frames_along(path, up=(0, 1, 0))
    c = path[k]
    rr = radii[k] * 1.12
    cuffs = []
    for off, rad, wid in ((-0.32, rr + 0.05, 0.07), (0.0, rr, 0.32), (0.32, rr + 0.05, 0.07)):
        cc = c + tang[k] * off
        ring = []
        for a in np.linspace(0, 2 * np.pi, 25):
            ring.append(cc + (np.cos(a) * nr[k] + np.sin(a) / 0.82 * bn[k] * 0.82) * rad)
        cv, ct, cuv, _ = hl_geom.sweep(np.array(ring), np.full(25, wid), 8)
        cuffs.append((cv, ct, cuv))
    band = []
    for off in np.linspace(-0.3, 0.3, 7):
        cc = c + tang[k] * off
        band.append(cc)
    bv, bt, buv, _ = hl_geom.sweep(np.array(band), np.full(7, rr), 28,
                                   profile=lambda f, ang: np.sqrt(1.0 / (np.cos(ang) ** 2 + (np.sin(ang) / 0.82) ** 2)),
                                   up=tuple(nr[k]))
    cuffs.append((bv, bt, buv))
    return (v, t, uv), cuffs


def build(body, proxy=None, log=print):
    hv, ht = head_mesh()
    tree = hl_blend.bvh(hv, ht)
    parts = []
    head_bone = "NPC Head [Head]"
    for side, tag in ((-1, "L"), (1, "R")):
        (v, t, uv), cuffs = horn(side, tree)
        p = Part("horn" + tag, "Horns", v, t, uv, "horn" + tag, "horn")
        parts.append(p)
        for k, (cv, ct, cuv) in enumerate(cuffs):
            parts.append(Part("horncuff%s%d" % (tag, k), "HornsBrass", cv, ct, cuv, "patch:brass", "brass"))
    for p in parts:
        p.bones = [head_bone]
        p.w = np.ones((len(p.v), 1))
        p.D = None
        p.sliders = []
    return parts, None

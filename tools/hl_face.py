import os
import re

import numpy as np

import hl_geom
import hl_nif
import hl_tri
from hl_preset import MODS

HEADS = {
    "vanilla": (r"[BODY] Expressive Facegen Morphs\meshes\actors\character\character assets\femalehead.nif",
                r"[BODY] Expressive Facegen Morphs\meshes\actors\character\character assets\femaleheadraces.tri",
                r"[BODY] Expressive Facegen Morphs\meshes\actors\character\character assets\femaleheadchargen.tri",
                r"[ANIMATION] Expressive Facial Animation (Female)\meshes\actors\character\character assets"
                r"\femalehead.tri"),
    "highpoly": (r"[PIERCING] Satisfactory Facial Piercings\meshes\KL\High Poly Head\FemaleHead.nif",
                 r"[PATCH] Closed Mouths for Orcs\meshes\KL\High Poly Head\FemaleHeadRaces.tri",
                 r"[BODY] High Poly Head SE\meshes\KL\High Poly Head\FemaleHeadCharGen.tri",
                 r"[BODY] High Poly Head SE\meshes\KL\High Poly Head\FemaleHead.tri"),
}
RACES = ("NordRace", "OrcRace", "HighElfRace")
SKIP_EXPR = ("Aah", "BigAah", "CombatShout", "Oh", "OohQ", "Eh", "I", "K", "N", "DST", "Th", "W", "R", "FV", "ChJSh",
             "BMP", "Eee", "SkinnyMorph", "Blink")


def load_head(kind="vanilla"):
    nif, races, chargen, expr = (os.path.join(MODS, p) for p in HEADS[kind])
    f = hl_nif.NifFile(nif)
    sh = [s for s in f.shapes if len(s.verts) > 500][0]
    xf = hl_nif.xf_to_mat(sh.transform)
    v = np.array(sh.verts) @ xf[:3, :3].T + xf[:3, 3]
    t = np.array(sh.tris)
    rot = xf[:3, :3]
    morphs = {}
    for tag, path in (("race", races), ("chargen", chargen), ("expr", expr)):
        if os.path.exists(path):
            _, _, m = hl_tri.read_tri(path)
            for k, d in m.items():
                if len(d) == len(v):
                    morphs[(tag, k)] = d @ rot.T
    return v, t, morphs


def groups(morphs):
    out = {}
    for (tag, name), d in morphs.items():
        if tag == "race":
            key = "race"
        elif tag == "expr":
            if any(name.startswith(s) for s in SKIP_EXPR):
                continue
            key = "expr"
        else:
            m = re.match(r"(NoseType|LipType|EyesType)\d+", name)
            if m:
                key = m.group(1)
            else:
                key = re.sub(r"(Up|Down|In|Out|Wide|Narrow|Forward|Back|Long|Short|Thin|Left|Right)$", "", name)
        out.setdefault(key, []).append(d)
    return out


def sampled_envelope(v, t, morphs, races=RACES, count=3000, active=0.6, expr_active=0.5, expr_max=0.8,
                     pct=97.0, seed=5):
    rng = np.random.default_rng(seed)
    n = hl_geom.vertex_normals(v, t)
    g = groups(morphs)
    proj = {k: np.array([(d * n).sum(1) for d in ds]) for k, ds in g.items() if k != "race"}
    race = np.array([(morphs[("race", r)] * n).sum(1) for r in races if ("race", r) in morphs])
    out = np.zeros((count, len(v)))
    for s in range(count):
        acc = race[rng.integers(len(race))].copy()
        for k, P in proj.items():
            on, top = (expr_active, expr_max) if k == "expr" else (active, 1.0)
            if k == "expr":
                for row in P:
                    if rng.random() < on / len(P) * 4:
                        acc += rng.uniform(0, top) * row
            elif rng.random() < on:
                acc += rng.uniform(0, top) * P[rng.integers(len(P))]
        out[s] = acc
    return np.clip(np.percentile(out, pct, axis=0), 0, None), n


def envelope(v, t, morphs, races=RACES, expr_scale=1.0):
    n = hl_geom.vertex_normals(v, t)
    g = groups(morphs)
    total = np.zeros(len(v))
    for key, ds in g.items():
        if key == "race":
            ds = [morphs[("race", r)] for r in races if ("race", r) in morphs]
        out = np.max([(d * n).sum(1) for d in ds], axis=0)
        total += np.clip(out, 0, None) * (expr_scale if key == "expr" else 1.0)
    return total, n

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import hl_fit  # noqa: E402
import hl_preset  # noqa: E402

SET = "CBBE 3BBB Body Amazing"
GROUPS = ("CBBE", "3BA", "3BBB", "CBBE Bodies", "Hollow Lantern")
BASIS = (
    "Breasts", "BreastsSmall", "BreastsNewSH", "BreastGravity2", "BreastPerkiness", "BreastHeight", "BreastWidth",
    "BreastCleavage", "BreastsTogether", "BreastTopSlope", "BreastFlatness", "BreastSideShape", "PushUp",
    "BigTorso", "ChestDepth", "ChestWidth", "Waist", "WaistHeight", "WideWaistLine", "ChubbyWaist", "Belly",
    "BigBelly", "TummyTuck", "BackArch", "Back", "MuscleAbs", "ShoulderWidth", "ShoulderSmooth", "ShoulderTweak",
    "Arms", "ChubbyArms", "MuscleArms", "MuscleMoreAbs_v2", "MuscleMoreArms_v2", "MuscleMoreLegs_v2", "MusclePecs",
    "RibsProminance", "Hips", "HipBone", "HipForward", "HipUpperWidth", "HipCarved",
    "HipNarrow_v2", "Butt", "BigButt", "ButtClassic", "ButtShape2", "RoundAss", "AppleCheeks", "ButtSmall",
    "ChubbyButt", "ButtSaggy_v2", "MuscleButt", "ButtCrack", "Groin", "Thighs", "SlimThighs",
    "ThighOutsideThicc_v2", "ThighInsideThicc_v2", "ThighFBThicc_v2", "ChubbyLegs", "LegsThin", "KneeHeight",
    "KneeShape", "CalfSize", "CalfSmooth", "MuscleLegs", "LegSpread_v2", "CrotchGap",
)
AUX = ("Groin", "LegSpread_v2", "ThighInsideThicc_v2", "CrotchGap", "HipForward", "ButtCrack", "BreastsSmall",
       "BreastTopSlope", "BreastHeight", "BreastWidth", "BreastFlatness", "Arms", "ChestDepth", "Back", "HipBone",
       "HipCarved", "HipNarrow_v2", "LegsThin", "SlimThighs")
MUSCLE = ("MuscleAbs", "MuscleMoreAbs_v2", "MuscleArms", "MuscleMoreArms_v2", "MuscleLegs", "MuscleMoreLegs_v2",
          "MuscleButt", "MusclePecs", "RibsProminance")
VOLUME = ("Breasts", "BreastsNewSH", "BigTorso", "Belly", "BigBelly", "Hips", "Butt", "BigButt", "ButtClassic",
          "RoundAss", "ChubbyButt", "Thighs", "ThighOutsideThicc_v2", "ThighFBThicc_v2", "ChubbyLegs", "ChubbyArms",
          "CalfSize")
PRESETS = [
    ("Hollow Lantern - Harvest Queen", "BONOBODY SE ver.7.xml", None, 0.82),
    ("Hollow Lantern - Tavern Witch", "!WenchMaiden.xml", None, None),
    ("Hollow Lantern - Night Huntress", "0_Rachel (Ninja Gaiden).xml", None, 0.85),
    ("Hollow Lantern - Wisp Dancer", "0_Yui_Horizon_walker.xml", None, 0.85),
    ("Hollow Lantern - Ember Warden", "Metis_Preset.xml", None, None),
]


def find_preset(mods, file_name):
    found = [os.path.join(root, file_name) for root, dirs, files in os.walk(mods)
             if os.path.basename(root).lower() == "sliderpresets" and file_name in files]
    if len(found) != 1:
        raise FileNotFoundError("%s: %d copies under %s" % (file_name, len(found), mods))
    return found[0]


def regions(v):
    w = np.ones(len(v))
    crotch = (np.abs(v[:, 0]) < 2.6) & (v[:, 2] > 58) & (v[:, 2] < 70)
    w[crotch] = 0.15
    nip = (v[:, 2] > 90) & (v[:, 2] < 100) & (v[:, 1] > 9)
    w[nip] = 0.4
    arms = np.abs(v[:, 0]) > 16
    w[arms] = 0.6
    return w


def crossing_rows(v, vals, D, B):
    rows, rhs = [], []
    zones = [((v[:, 0] < -0.05) & (v[:, 0] > -3.2) & (v[:, 2] > 88) & (v[:, 2] < 99) & (v[:, 1] > 3), 0.18),
             ((v[:, 0] < -0.05) & (v[:, 0] > -4.5) & (v[:, 2] > 48) & (v[:, 2] < 66) & (np.abs(v[:, 1]) < 4.5),
              0.25)]
    for sel, gap in zones:
        idx = np.nonzero(sel)[0]
        x = v[idx, 0] + D[idx][:, B, 0] @ vals
        bad = idx[x > -gap]
        for i in bad:
            rows.append(D[i, B, 0])
            rhs.append(-gap - v[i, 0])
    return np.array(rows), np.array(rhs)


def bounded_solve(M, y, lo, hi, iters=30):
    free = np.ones(len(y), dtype=bool)
    x = np.zeros(len(y))
    for _ in range(iters):
        rhs = y - M[:, ~free] @ x[~free]
        x[free] = np.linalg.solve(M[np.ix_(free, free)], rhs[free])
        out = free & ((x < lo) | (x > hi))
        if not out.any():
            break
        x[out] = np.clip(x, lo, hi)[out]
        free &= ~out
    return x


def fit(body, target, W, B, lam=0.004, mu=80.0, rounds=12):
    D = body.D.astype(np.float64)
    names = [body.slider_names[j] for j in B]
    aux = np.array([n in AUX for n in names])
    lo = np.where(aux, -0.5, -1.0)
    hi = np.where(aux, 0.8, np.where([n in MUSCLE for n in names], 1.0, 2.0))
    A = (np.sqrt(W)[:, None, None] * D[:, B, :]).transpose(0, 2, 1).reshape(-1, len(B))
    b = (np.sqrt(W)[:, None] * target).reshape(-1)
    AtA = A.T @ A + lam * np.diag(np.where(aux, 12.0, 1.0)) * len(body.v)
    Atb = A.T @ b
    x = bounded_solve(AtA, Atb, lo, hi)
    extra_A = np.zeros((0, len(B)))
    extra_b = np.zeros(0)
    for _ in range(rounds):
        R, r = crossing_rows(body.v, x, D, B)
        if len(R) == 0:
            break
        extra_A = np.vstack([extra_A, R])
        extra_b = np.concatenate([extra_b, r])
        x = bounded_solve(AtA + mu * extra_A.T @ extra_A, Atb + mu * extra_A.T @ extra_b, lo, hi)
    return x


def to_preset_value(name, applied, attrs):
    a = attrs[name]
    return 1.0 - applied if a.get("invert") == "true" else applied


def main(mods, ref, out_path, report_path):
    body = hl_fit.Body(pickle.load(open(ref, "rb")))
    names = body.slider_names
    attrs = {a["name"]: a for a in body.slider_attrs}
    B = [names.index(n) for n in BASIS if n in names]
    W = regions(body.v)
    D = body.D.astype(np.float64)
    out = []
    report = []
    for new_name, file_name, pname, slim in PRESETS:
        _, vals = hl_preset.read_preset(find_preset(mods, file_name), pname)
        toned = any(abs(v) > 0.1 for n in MUSCLE for v in vals.get(n, {}).values())
        PB = [j for j in B if toned or names[j] not in MUSCLE]
        sizes = {}
        for size, wgt in (("small", 0.0), ("big", 1.0)):
            sv = hl_preset.slider_values(body.slider_attrs, vals, wgt)
            s = np.array([sv.get(n, 0.0) for n in names])
            target = np.einsum("j,njd->nd", s, D)
            x = fit(body, target, W, PB)
            keep = [j for k, j in enumerate(PB) if abs(x[k]) >= 0.06]
            x = np.zeros(len(B))
            x[[B.index(j) for j in keep]] = fit(body, target, W, keep)
            sizes[size] = (x, s, target)
        if slim:
            x = sizes["big"][0].copy()
            for k, j in enumerate(B):
                if names[j] in VOLUME:
                    x[k] *= slim
            small_target = np.einsum("k,nkd->nd", x, D[:, B, :])
            sizes["small"] = (x, sizes["big"][1], small_target)
        lines = []
        for size in ("small", "big"):
            x, s, target = sizes[size]
            got = np.einsum("k,nkd->nd", x, D[:, B, :])
            err = np.linalg.norm(got - target, axis=1)
            report.append("%s %s: rms dev %.3f, p95 %.3f, max %.3f" % (new_name, size, np.sqrt((err ** 2 * W).sum() / W.sum()),
                                                                     np.percentile(err, 95), err.max()))
            for k, j in enumerate(B):
                pv = int(round(to_preset_value(names[j], x[k], attrs) * 100))
                default = int(round(float(attrs[names[j]].get(size, 0))))
                if pv != default:
                    lines.append((names[j], size, pv))
        lines.sort(key=lambda r: (r[0].lower(), r[1]))
        body_xml = ['    <Preset name="%s" set="%s">' % (new_name, SET)]
        body_xml += ['        <Group name="%s"/>' % g for g in GROUPS]
        body_xml += ['        <SetSlider name="%s" size="%s" value="%d"/>' % r for r in lines]
        body_xml.append("    </Preset>")
        out.append("\n".join(body_xml))
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<SliderPresets>\n' + "\n".join(out) + "\n</SliderPresets>\n"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(xml)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    main(*argv)

import os
import re

import numpy as np

MODS = os.environ.get("HL_MODS", os.path.expandvars(r"%LOCALAPPDATA%\ModOrganizer\Skyrim Special Edition\mods"))


def read_preset(path, name=None):
    s = open(path, encoding="utf-8-sig", errors="replace").read()
    presets = re.findall(r'<Preset name="([^"]+)"[^>]*>(.*?)</Preset>', s, re.S)
    for pname, body in presets:
        if name is None or pname == name:
            vals = {}
            for m in re.finditer(r'<SetSlider name="([^"]+)" size="(small|big)" value="(-?[0-9.]+)"', body):
                vals.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3)) / 100.0
            return pname, vals
    raise KeyError(name)


def slider_values(slider_attrs, preset_vals, weight=1.0):
    out = {}
    for a in slider_attrs:
        n = a["name"]
        small = float(a.get("small", 0)) / 100.0
        big = float(a.get("big", 0)) / 100.0
        pv = preset_vals.get(n, {})
        small = pv.get("small", small)
        big = pv.get("big", big)
        val = small + (big - small) * weight
        if a.get("invert") == "true":
            val = 1.0 - val
        if a.get("uv") == "true":
            continue
        out[n] = val
    return out


def corpus(slider_attrs, mods_dir=MODS, weights=(0.0, 1.0)):
    out = []
    for root, dirs, files in os.walk(mods_dir):
        if os.path.basename(root).lower() != "sliderpresets":
            continue
        for fn in sorted(files):
            if not fn.lower().endswith(".xml"):
                continue
            path = os.path.join(root, fn)
            s = open(path, encoding="utf-8-sig", errors="replace").read()
            for pname, _ in re.findall(r'<Preset name="([^"]+)"[^>]*>(.*?)</Preset>', s, re.S):
                _, vals = read_preset(path, pname)
                for wgt in weights:
                    out.append((pname, slider_values(slider_attrs, vals, wgt)))
    return out


def apply(v, D, sliders, values):
    out = np.array(v, dtype=np.float64)
    for j, n in enumerate(sliders):
        if n in values and values[n] != 0:
            out += D[:, j] * values[n]
    return out

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hl_bs  # noqa: E402
import hl_nif  # noqa: E402
from hl_preset import MODS  # noqa: E402

BS_3BA = os.path.join(MODS, r"[BODY] CBBE 3BA (3BBB)\CalienteTools\BodySlide")
BS_CBBE = os.path.join(MODS, r"[BODY] CBBE (Base)\CalienteTools\BodySlide")
SKELETON = os.path.join(MODS, r"[FRAMEWORK] XP32 Maximum Skeleton Special Extended (XPMSSE)\meshes\actors\character"
                              r"\character assets female\skeleton_female.nif")
OSP = os.path.join(BS_3BA, r"SliderSets\SE 3BBB Amazing.osp")

SETS = {
    "body": ("CBBE 3BBB Body Amazing", os.path.join(BS_3BA, r"ShapeData\SE 3BBB Amazing")),
    "feet": ("CBBE 3BBB Feet", os.path.join(BS_CBBE, r"ShapeData\CBBE")),
    "hands": ("CBBE 3BBB Hands", os.path.join(BS_CBBE, r"ShapeData\CBBE")),
}
SOURCES = {"body": "SE 3BBB Body Amazing v2.nif", "feet": "CBBE Feet.nif", "hands": "CBBE Hands.nif"}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    skel = hl_nif.skeleton_globals(SKELETON)
    with open(os.path.join(out_dir, "skeleton.pkl"), "wb") as f:
        pickle.dump(skel, f)
    print("skeleton nodes", len(skel))
    for key, (set_name, folder) in SETS.items():
        sliders = hl_bs.read_slider_set(OSP, set_name)
        nif = hl_nif.NifFile(os.path.join(folder, SOURCES[key]))
        shapes = {s.name: hl_nif.read_shape(s) for s in nif.shapes}
        osd_cache = {}
        diffs = {}
        for attrs, data in sliders:
            for dname, target, dfile in data:
                fname, entry = dfile.split("\\", 1)
                if fname not in osd_cache:
                    osd_cache[fname] = hl_bs.read_osd(os.path.join(folder, fname))
                if entry in osd_cache[fname]:
                    diffs[(attrs["name"], target)] = osd_cache[fname][entry]
                else:
                    diffs[(attrs["name"], target)] = (np.zeros(0, np.int64), np.zeros((0, 3)))
        for name, sh in shapes.items():
            dev = []
            for j, b in enumerate(sh["bones"]):
                g = np.linalg.inv(sh["s2b"][j])
                if b in skel:
                    dev.append(np.abs(g - skel[b]).max())
            print(key, name, len(sh["v"]), "verts", len(sh["bones"]), "bones",
                  "max bind deviation from skeleton %.4f" % (max(dev) if dev else -1))
        with open(os.path.join(out_dir, key + ".pkl"), "wb") as f:
            pickle.dump({"set": set_name, "sliders": sliders, "shapes": shapes, "diffs": diffs}, f)
        print(key, set_name, len(sliders), "sliders", len(diffs), "diff sets")


if __name__ == "__main__":
    main(sys.argv[1])

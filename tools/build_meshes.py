import json
import os
import pickle
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "lib"))

import hl_atlas  # noqa: E402
import hl_blend  # noqa: E402
import hl_bs  # noqa: E402
import hl_export  # noqa: E402
import hl_fit  # noqa: E402
import hl_geom  # noqa: E402
import hl_nif  # noqa: E402
import nif_rigid_lib as rigid  # noqa: E402
import piece_axe  # noqa: E402
import piece_boots  # noqa: E402
import piece_choker  # noqa: E402
import piece_corset  # noqa: E402
import piece_gloves  # noqa: E402
import piece_horns  # noqa: E402
import piece_lantern  # noqa: E402
import piece_mask  # noqa: E402
import piece_shorts  # noqa: E402
import piece_tail  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
WORK = argv[0] if argv else r"D:\Dev\Skyrim MOD\work\hollow-lantern"
OUT = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(TOOLS), "out")
REF = os.path.join(WORK, "ref")
VANILLA = os.path.join(WORK, "refs", "vanilla", "meshes")
MESH_REL = "Tinesh\\HollowLantern"
MESH_DIR = os.path.join(OUT, "data", "meshes", "Tinesh", "HollowLantern")
BS_DIR = os.path.join(OUT, "data", "CalienteTools", "BodySlide")
SHAPEDATA = os.path.join(BS_DIR, "ShapeData", "Hollow Lantern")
ATLAS = 8192
HH = piece_boots.HH

ITEMS = [
    ("corset", "Corset", "corset", 32),
    ("shorts", "Shorts", "shorts", 49),
    ("boots", "Boots", "boots", 37),
    ("gloves", "Gloves", "gloves", 33),
    ("choker", "Choker", "choker", 45),
    ("tail", "Tail", "tail", 40),
]

log_t0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - log_t0, msg), flush=True)


def load_body():
    ref = pickle.load(open(os.path.join(REF, "body.pkl"), "rb"))
    body = hl_fit.Body(ref)
    path = os.path.join(REF, "proxy.pkl")
    if os.path.exists(path):
        proxy = pickle.load(open(path, "rb"))
        proxy.tree = hl_blend.bvh(proxy.v, proxy.t)
    else:
        proxy = piece_corset.make_proxy(body)
        tree, proxy.tree = proxy.tree, None
        pickle.dump(proxy, open(path, "wb"))
        proxy.tree = tree
    return ref, body, proxy


def bone_globals(body, hands_ref, skel, extra=None):
    g = {}
    for name, m in skel.items():
        g[name] = m
    sh = hands_ref["shapes"]["Hands"]
    for j, b in enumerate(sh["bones"]):
        g[b] = np.linalg.inv(sh["s2b"][j])
    for j, b in enumerate(body.bones):
        g[b] = np.linalg.inv(body.s2b[j])
    if extra:
        g.update(extra)
    return g


def shader_for(group):
    shape = group[0].shape
    mats = {p.mat for p in group}
    if "collision" in mats:
        return "invisible"
    if shape.endswith("Brass"):
        return "brass"
    if shape.endswith("Glow"):
        return "glow"
    if shape == "Axe":
        return "iron"
    if shape.startswith("Horns"):
        return "horn"
    return "cloth"


def group_by_shape(parts):
    order = []
    for p in parts:
        if p.shape not in order:
            order.append(p.shape)
    return [(name, [p for p in parts if p.shape == name]) for name in order]


def smp_tail_xml(fx):
    bones = fx["bones"]
    kin = hl_export.SMP_BODY_BONES
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<system xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="description.xsd">']
    for b in kin:
        lines.append('\t<bone name="%s"/>' % b)
    lines.append('\t<bone name="%s"/>' % bones[0])
    n = len(bones) - 1
    for i, b in enumerate(bones[1:]):
        f = i / max(n - 1, 1)
        lines += ['\t<bone-default>',
                  '\t\t<mass>%.2f</mass>' % (0.9 - 0.45 * f),
                  '\t\t<inertia x="%.1f" y="%.1f" z="%.1f"/>' % ((6 - 3 * f,) * 3),
                  '\t\t<centerOfMassTransform>', '\t\t\t<basis x="0" y="0" z="0" w="1"/>',
                  '\t\t\t<origin x="0" y="0" z="0"/>', '\t\t</centerOfMassTransform>',
                  '\t\t<linearDamping>%.2f</linearDamping>' % (0.55 - 0.15 * f),
                  '\t\t<angularDamping>%.2f</angularDamping>' % (0.6 - 0.2 * f),
                  '\t\t<friction>0.3</friction>', '\t\t<rollingFriction>0.1</rollingFriction>',
                  '\t\t<restitution>0.05</restitution>', '\t\t<margin-multiplier>1</margin-multiplier>',
                  '\t\t<gravity-factor>%.2f</gravity-factor>' % (0.55 + 0.35 * f), '\t</bone-default>',
                  '\t<bone name="%s"/>' % b]
    lines += ['\t<bone-default>', '\t\t<mass>0</mass>', '\t</bone-default>']
    lines += ['\t<generic-constraint-default>', '\t\t<frameInB>', '\t\t\t<basis x="0" y="0" z="0" w="1"/>',
              '\t\t\t<origin x="0" y="0" z="0"/>', '\t\t</frameInB>',
              '\t\t<useLinearReferenceFrameA>false</useLinearReferenceFrameA>',
              '\t\t<linearLowerLimit x="0" y="0" z="0"/>', '\t\t<linearUpperLimit x="0" y="0" z="0"/>',
              '\t\t<angularLowerLimit x="-0.12" y="-0.42" z="-0.42"/>',
              '\t\t<angularUpperLimit x="0.12" y="0.42" z="0.42"/>',
              '\t\t<linearStiffness x="0" y="0" z="0"/>', '\t\t<angularStiffness x="22" y="22" z="22"/>',
              '\t\t<linearDamping x="0" y="0" z="0"/>', '\t\t<angularDamping x="0.4" y="0.4" z="0.4"/>',
              '\t\t<linearEquilibrium x="0" y="0" z="0"/>', '\t\t<angularEquilibrium x="0" y="0" z="0"/>',
              '\t\t<linearBounce x="0" y="0" z="0"/>', '\t\t<angularBounce x="0" y="0" z="0"/>',
              '\t</generic-constraint-default>', '\t<constraint-group>']
    for a, b in zip(bones[1:], bones[:-1]):
        lines.append('\t\t<generic-constraint bodyA="%s" bodyB="%s"/>' % (a, b))
    lines += ['\t</constraint-group>',
              '\t<per-triangle-shape name="TailBody">', '\t\t<margin>0.9</margin>',
              '\t\t<prenetration>0.4</prenetration>', '\t\t<shared>private</shared>', '\t\t<tag>hl_body</tag>',
              '\t\t<no-collide-with-tag>hl_body</no-collide-with-tag>', '\t</per-triangle-shape>',
              '\t<per-vertex-shape name="Tail">', '\t\t<margin>0.45</margin>', '\t\t<tag>hl_tail</tag>',
              '\t\t<can-collide-with-tag>hl_body</can-collide-with-tag>']
    for b in bones[:3]:
        lines.append('\t\t<weight-threshold bone="%s">1</weight-threshold>' % b)
    lines += ['\t</per-vertex-shape>', '</system>']
    return "\n".join(lines) + "\n"


def lay_down(v):
    r = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    p = v @ r.T
    c = 0.5 * (p.min(0) + p.max(0))
    p[:, 0] -= c[0]
    p[:, 1] -= c[1]
    p[:, 2] -= p[:, 2].min()
    return p, r, c


def write_gnd(path, parts, template, mass, root_name):
    from io_scene_nifly.pyn.pynifly import NifFile
    base = NifFile(template)
    keep = [p for p in parts if p.mat not in ("collision",)]
    allv = np.concatenate([p.v for p in keep])
    _, r, _ = lay_down(allv)
    lv = allv @ r.T
    c = 0.5 * (lv.min(0) + lv.max(0))
    zmin = lv[:, 2].min()
    nif = NifFile()
    nif.initialize("SKYRIMSE", path, root_type="BSFadeNode", root_name=root_name)
    root = nif.root
    rigid.copy_root_extra_data(base.root, nif, root)
    groups = []
    for name, group in group_by_shape(keep):
        data = hl_export.merge_parts(group)
        p = data["v"] @ r.T
        p[:, 0] -= c[0]
        p[:, 1] -= c[1]
        p[:, 2] -= zmin
        n = data["n"] @ r.T
        key = shader_for(group)
        groups.append((name, {"v": p, "t": data["t"], "uv": data["uv"], "n": n}, key, None))
    hl_export.static_shapes(nif, groups, parent=root)
    pts = np.concatenate([g[1]["v"] for g in groups])
    cal = rigid.inertia_calibration(base.root.collision_object.body)
    info = rigid.add_convex_collision(root, pts, base.root.collision_object, mass, cal, 48)
    nif.save()
    b = rigid.bounds(pts)
    return {"collision": info, "bounds": b}


FLICKER = [(0.0, 1.0), (0.17, 0.86), (0.31, 0.97), (0.5, 0.82), (0.66, 1.0), (0.83, 0.88), (1.0, 0.95),
           (1.21, 0.84), (1.4, 1.0)]


def write_weapon(path, parts, template, collision_template, root_name, mass):
    from io_scene_nifly.pyn.pynifly import NifFile
    base = NifFile(template)
    coll = NifFile(collision_template)
    bsx = [int(ed.properties.integerData) for ed in base.root.extra_data() if ed.blockname == "BSXFlags"][0]
    nif = NifFile()
    nif.initialize("SKYRIMSE", path, root_type="BSFadeNode", root_name=root_name)
    root = nif.root
    rigid.copy_root_extra_data(base.root, nif, root, bsx_flags=bsx | 1)
    groups = []
    for name, group in group_by_shape(parts):
        data = hl_export.merge_parts(group)
        groups.append((name, {"v": data["v"], "t": data["t"], "uv": data["uv"], "n": data["n"]}, shader_for(group),
                       None))
    shapes = hl_export.static_shapes(nif, groups, parent=root)
    glow_shape = [s for s, g in zip(shapes, groups) if g[2] == "glow"][0]
    rigid.add_emissive_loop(nif, glow_shape, [(t, (1.0 * f, 0.5 * f, 0.14 * f)) for t, f in FLICKER])
    pts = np.concatenate([g[1]["v"] for g in groups])
    cal = rigid.inertia_calibration(coll.root.collision_object.body)
    info = rigid.add_convex_collision(root, pts, coll.root.collision_object, mass, cal, 48)
    nif.save()
    return {"collision": info, "bounds": rigid.bounds(pts)}


def write_lantern(path, parts, fx, template_torch, template_gourd):
    from io_scene_nifly.pyn.pynifly import NifFile
    torch = NifFile(template_torch)
    gourd = NifFile(template_gourd)
    nif = NifFile()
    nif.initialize("SKYRIMSE", path, root_type="BSFadeNode", root_name="HollowLantern")
    root = nif.root
    rigid.copy_root_extra_data(torch.root, nif, root, bsx_flags=195)

    def place(v):
        return np.asarray(v) * fx["scale"] + fx["offset"]

    groups = []
    for name, group in group_by_shape(parts):
        data = hl_export.merge_parts(group)
        key = "brass" if name.endswith("Brass") else ("glow" if name.endswith("Glow") else "cloth")
        groups.append((name, {"v": place(data["v"]), "t": data["t"], "uv": data["uv"], "n": data["n"]}, key, None))
    shapes = hl_export.static_shapes(nif, groups, parent=root)
    flicker = FLICKER
    glow_shape = [s for s, g in zip(shapes, groups) if g[0].endswith("Glow")][0]
    rigid.add_emissive_loop(nif, glow_shape, [(t, (1.0 * f, 0.55 * f, 0.16 * f)) for t, f in flicker])
    base_fx = dict(Shader_Flags_2=0x30, falloffStartAngle=0.866, falloffStopAngle=0.174, falloffStartOpacity=1.0,
                   falloffStopOpacity=0.0, textureClampMode=3, UV_Scale_U=1.0, UV_Scale_V=1.0, UV_Offset_U=0.0,
                   UV_Offset_V=0.0, LightingInfluence=255)
    fv, ft, fuv, fn, fc = fx["flame"]
    flame = rigid.add_effect_shape(nif, root, "LanternFlame", place(fv), ft, fuv, fn, fc,
                                   dict(base_fx, Shader_Flags_1=0xC0000048, Emissive_Color=(1.0, 0.62, 0.22, 1.0),
                                        Emissive_Mult=2.4, softFalloffDepth=1.0),
                                   {"Diffuse": r"textures\effects\candleflame01.dds"})
    rigid.add_effect_float_controllers(nif, flame, [(0, [(t, 2.4 * f) for t, f in flicker])])
    hv, ht, huv, hn, hc = fx["halo"]
    halo = rigid.add_effect_shape(nif, root, "LanternHalo", place(hv), ht, huv, hn, hc,
                                  dict(base_fx, Shader_Flags_1=0xC0000048, Emissive_Color=(1.0, 0.5, 0.14, 1.0),
                                       Emissive_Mult=0.9, softFalloffDepth=1.5, textureClampMode=0),
                                  {"Diffuse": r"textures\effects\GlowSoft01.dds"})
    rigid.add_effect_float_controllers(nif, halo, [(0, [(t, 0.9 * f) for t, f in flicker])])
    light = np.eye(4)
    light[:3, 3] = place(fx["light"])
    nif.add_node("AttachLight", hl_nif.mat_to_xf(light), root)
    pts = np.concatenate([g[1]["v"] for g in groups])
    cal = rigid.inertia_calibration(gourd.root.collision_object.body)
    info = rigid.add_convex_collision(root, pts, gourd.root.collision_object, 1.2, cal, 48)
    nif.save()
    return {"collision": info, "bounds": rigid.bounds(pts)}


def hemisphere(count, seed=7):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(count * 3, 3))
    d = d[d[:, 2] > 0.15][:count]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def compute_ao(parts, occluders, rays=14, reach=1.8):
    from mathutils import Vector
    vs, ts = [], []
    off = 0
    for v, t in occluders:
        vs.append(v)
        ts.append(t + off)
        off += len(v)
    tree = hl_blend.bvh(np.concatenate(vs), np.concatenate(ts))
    dirs = hemisphere(rays)
    for p in parts:
        n = p.normals if getattr(p, "normals", None) is not None else hl_geom.vertex_normals(p.v, p.t)
        a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
        tx = hl_geom.normalize(np.cross(n, a))
        ty = np.cross(n, tx)
        occ = np.zeros(len(p.v))
        for d in dirs:
            w = tx * d[0] + ty * d[1] + n * d[2]
            o = p.v + n * 0.03
            for i in range(len(o)):
                h = tree.ray_cast(Vector(o[i]), Vector(w[i]), reach)
                if h[0] is not None:
                    occ[i] += (1.0 - h[3] / reach) ** 0.5
        p.attrs["ao"] = 1.0 - occ / len(dirs)


def manifest_entry(p):
    return {"name": p.name, "mat": p.mat, "island": p.island, "v": p.v.astype(np.float32), "t": p.t.astype(np.int32),
            "uv": p.uv_px.astype(np.float32),
            "n": (p.normals if getattr(p, "normals", None) is not None
                  else hl_geom.vertex_normals(p.v, p.t)).astype(np.float32),
            "attrs": {k: np.asarray(a, dtype=np.float32) for k, a in p.attrs.items()
                      if np.asarray(a).ndim == 1 and len(a) == len(p.v)}}


def main():
    ref, body, proxy = load_body()
    hands_ref = pickle.load(open(os.path.join(REF, "hands.pkl"), "rb"))
    skel = pickle.load(open(os.path.join(REF, "skeleton.pkl"), "rb"))
    log("references")
    built = {}
    corset = piece_corset.build(body, proxy, log)
    built["corset"] = corset[0]
    keep_tris, D_conf, under = corset[1], corset[3], corset[4]
    built["shorts"] = piece_shorts.build(body, proxy, log)[0]
    built["boots"] = piece_boots.build(body, proxy, log)[0]
    built["gloves"] = piece_gloves.build(body, proxy, log, hands_ref=hands_ref, skel=skel)[0]
    built["choker"] = piece_choker.build(body, proxy, log)[0]
    tail_parts, tail_fx = piece_tail.build(body, proxy, log, skel=skel)
    built["tail"] = tail_parts
    built["horns"] = piece_horns.build(body, proxy, log)[0]
    built["mask"] = piece_mask.build(body, proxy, log)[0]
    lantern_parts, lantern_fx = piece_lantern.build(body, proxy, log)
    built["lantern"] = lantern_parts
    built.update(piece_axe.build(log=log))
    log("pieces built")
    every = [p for parts in built.values() for p in parts]
    layout = hl_atlas.pack(every, ATLAS)
    hl_atlas.apply(every, layout)
    log("atlas %.1f px/unit, %d islands" % (layout["scale"], len(layout["islands"])))
    globals_ = bone_globals(body, hands_ref, skel, tail_fx["globals"])
    os.makedirs(MESH_DIR, exist_ok=True)
    os.makedirs(SHAPEDATA, exist_ok=True)
    report = {"atlas": {"size": ATLAS, "px_per_unit": layout["scale"]}, "items": {}}
    sets = []
    for key, stem, out_name, partition in ITEMS:
        ns = hl_export.NifSet(stem, out_name, partition, globals_)
        if key == "corset":
            bdata, bsh = hl_export.body_shape_data(ref, body, keep_tris, D_weld=D_conf, changed=under)
            ns.add_shape("3BA", bdata, shader=bsh["shader"], textures=bsh["textures"], keep_all=True)
        if key == "tail":
            ns.nodes = [(name, parent, tail_fx["globals"][name]) for name, parent in tail_fx["hierarchy"]]
            ns.root_extras.append(("HDT Skinned Mesh Physics Object", "Meshes\\%s\\tail.xml" % MESH_REL))
        for name, group in group_by_shape(built[key]):
            data = hl_export.merge_parts(group, body.bones)
            extras = []
            if key == "boots" and name == "BootsL":
                extras.append(("SDTA", '[{"name":"NPC","pos":[0,0,%.1f]}]' % HH))
            ns.add_shape(name, data, shader_for(group), extras=extras)
        ns.write_shapedata(SHAPEDATA)
        ns.write_meshes(MESH_DIR, "%s\\%s.tri" % (MESH_REL, out_name), body.slider_attrs)
        sets.append(ns.slider_set("Hollow Lantern - %s" % stem, "Hollow Lantern", "meshes\\" + MESH_REL,
                                  body.slider_attrs))
        allv = np.concatenate([s["data"]["v"] for s in ns.shapes if s["name"] != "3BA"])
        report["items"][key] = {"shapes": {s["name"]: [len(s["data"]["v"]), len(s["data"]["t"])] for s in ns.shapes},
                                "sliders": {s["name"]: len(s["diffs"]) for s in ns.shapes},
                                "bounds": rigid.bounds(allv)}
        report["items"][key]["gnd"] = write_gnd(os.path.join(MESH_DIR, "%s_gnd.nif" % out_name), built[key],
                                                os.path.join(VANILLA, "armor", "studded", "male", "body_go.nif"),
                                                2.0, "HollowLantern%sGnd" % stem)
        log("%s exported" % key)
    hl_bs.write_osp(os.path.join(BS_DIR, "SliderSets", "Hollow Lantern.osp"), sets)
    names = ["Hollow Lantern - %s" % stem for _, stem, _, _ in ITEMS]
    hl_bs.write_groups(os.path.join(BS_DIR, "SliderGroups", "Hollow Lantern.xml"),
                       {"CBBE": names, "3BA": names, "3BBB": names, "CBBE Bodies": names})
    hns = hl_export.NifSet("Horns", "horns", 42, globals_)
    for name, group in group_by_shape(built["horns"]):
        hns.add_shape(name, hl_export.merge_parts(group), shader_for(group))
    hns.write_single(os.path.join(MESH_DIR, "horns.nif"))
    report["items"]["horns"] = {"gnd": write_gnd(os.path.join(MESH_DIR, "horns_gnd.nif"), built["horns"],
                                                 os.path.join(VANILLA, "armor", "studded", "male", "body_go.nif"),
                                                 1.0, "HollowLanternHornsGnd")}
    mns = hl_export.NifSet("Mask", "mask", 44, globals_)
    for name, group in group_by_shape(built["mask"]):
        mns.add_shape(name, hl_export.merge_parts(group), shader_for(group))
    mns.write_single(os.path.join(MESH_DIR, "mask.nif"))
    report["items"]["mask"] = {"gnd": write_gnd(os.path.join(MESH_DIR, "mask_gnd.nif"), built["mask"],
                                                os.path.join(VANILLA, "armor", "studded", "male", "body_go.nif"),
                                                1.0, "HollowLanternMaskGnd")}
    steel = os.path.join(VANILLA, "weapons", "steel")
    for key, template, root_name, mass in (("waraxe", "1stpersonsteelwaraxe.nif", "HollowLanternWarAxe", 9.0),
                                           ("battleaxe", "1stpersonsteelbattleaxe.nif", "HollowLanternBattleaxe",
                                            13.0)):
        report["items"][key] = write_weapon(os.path.join(MESH_DIR, "%s.nif" % key), built[key],
                                            os.path.join(steel, template),
                                            os.path.join(steel, "1stpersonsteelwaraxe.nif"), root_name, mass)
    report["items"]["lantern"] = write_lantern(os.path.join(MESH_DIR, "lantern.nif"), lantern_parts, lantern_fx,
                                               os.path.join(VANILLA, "weapons", "torch", "torch.nif"),
                                               os.path.join(VANILLA, "plants", "gourd01.nif"))
    with open(os.path.join(MESH_DIR, "tail.xml"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(smp_tail_xml(tail_fx))
    log("horns, mask, weapons, lantern, physics written")
    worn = [p for p in every if not p.name.startswith(("lantern", "axe")) and p.mat != "collision"]
    hv, ht = piece_horns.head_mesh()
    for targets, exclude in ((built["shorts"], []), (None, built["shorts"])):
        skip = {id(p) for p in exclude}
        sel = [p for p in worn if id(p) not in skip]
        occ = [(p.v, p.t) for p in sel if not p.island.startswith("patch:")]
        occ += [(p.v, p.t) for p in sel if p.mat in ("lining", "edge", "brass", "cord", "sole")]
        occ += [(body.v, body.t), (hv, ht)]
        compute_ao([p for p in (targets if targets is not None else sel) if not p.island.startswith("patch:")], occ)
    for parts in (lantern_parts, built["waraxe"], built["battleaxe"]):
        compute_ao([p for p in parts if not p.island.startswith("patch:")], [(p.v, p.t) for p in parts])
    log("ambient occlusion")
    manifest = {"size": ATLAS, "layout": layout, "parts": [manifest_entry(p) for p in every],
                "mask_arc": next(p.arc_table for p in built["mask"] if hasattr(p, "arc_table"))}
    with open(os.path.join(WORK, "texture_manifest.pkl"), "wb") as f:
        pickle.dump(manifest, f)
    with open(os.path.join(OUT, "meshes_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("done")


main()

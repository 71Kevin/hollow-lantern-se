"""Helpers for writing rigid Skyrim SE meshes with PyNifly (run inside Blender: bmesh builds the convex hulls)."""
import math

import bmesh
import numpy as np
from mathutils import Vector

from io_scene_nifly.pyn.nifconstants import CycleType
from io_scene_nifly.pyn.nifdefs import (BSEffectShaderPropertyFloatControllerBuf,
                                        BSLightingShaderPropertyColorControllerBuf, BSTriShapeBuf,
                                        LightingShaderControlledColor, NiAnimKeyQuadTransBuf, NiFloatDataBuf,
                                        NiFloatInterpolatorBuf, NiKeyType, NiPoint3InterpolatorBuf, PynBufferTypes,
                                        TimeControllerFlags, bhkConvexVerticesShapeProps)
from io_scene_nifly.pyn.pynifly import (BSInvMarker, BSXFlags, LinearScalarKey, NiFloatData, NiPosData,
                                        NiStringExtraData)

HSF = 69.99125
FLT_MAX = 3.4028234663852886e38
SHADER_FIELDS = [
    "bBSLightingShaderProperty", "bslspShaderType", "shaderFlags", "Shader_Type", "Shader_Flags_1", "Shader_Flags_2",
    "Env_Map_Scale", "UV_Offset_U", "UV_Offset_V", "UV_Scale_U", "UV_Scale_V", "Emissive_Color", "Emissive_Mult",
    "textureClampMode", "Alpha", "Refraction_Str", "Glossiness", "Spec_Color", "Spec_Str", "Soft_Lighting",
    "Rim_Light_Power",
]


def relative_texture(path):
    lower = path.lower().replace("/", "\\")
    cut = lower.find("textures\\")
    return path[cut:] if cut > 0 else path


def copy_shader(src, dst):
    for field in SHADER_FIELDS:
        value = getattr(src, field)
        if hasattr(value, "_length_"):
            for k in range(value._length_):
                getattr(dst, field)[k] = value[k]
        else:
            setattr(dst, field, value)


def copy_root_extra_data(src_root, nif, root, bsx_flags=None):
    for ed in src_root.extra_data():
        if ed.blockname == "BSInvMarker":
            p = ed.properties
            BSInvMarker.New(nif, name=ed.name, rotation=(p.rot0, p.rot1, p.rot2), zoom=p.zoom, parent=root)
        elif ed.blockname == "BSXFlags":
            flags = int(ed.properties.integerData) if bsx_flags is None else bsx_flags
            BSXFlags.New(nif, name=ed.name, flags=flags, parent=root)
        elif ed.blockname == "NiStringExtraData":
            NiStringExtraData.New(nif, name=ed.name, string_value=ed.string_data, parent=root)
        else:
            print("NIFLIB: skipped root extra data", ed.blockname, ed.name)


def add_shape(nif, parent, name, verts, tris, uvs, normals, flags, shader_src, textures):
    shape = nif.createShapeFromData(
        name, [tuple(v) for v in np.asarray(verts).tolist()], [tuple(t) for t in np.asarray(tris).tolist()],
        [tuple(u) for u in np.asarray(uvs).tolist()], [tuple(n) for n in np.asarray(normals).tolist()],
        props=BSTriShapeBuf(), parent=parent)
    shape.flags = flags
    copy_shader(shader_src, shape.shader.properties)
    for slot, path in textures.items():
        shape.set_texture(slot, path)
    shape.save_shader_attributes()
    return shape


def add_effect_shape(nif, parent, name, verts, tris, uvs, normals, colors, settings, textures,
                     alpha_flags=0x100D, alpha_threshold=128, flags=0xC000E):
    """Shape with a BSEffectShaderProperty (settings: NiShaderBuf field -> value), vertex colours and an
    NiAlphaProperty (default additive, SRC_ALPHA -> ONE), as vanilla fire and glow cards are built. nifly writes an
    effect shader's emissive colour and multiple from baseColor/baseColorScale, so those mirror Emissive_*."""
    settings = dict(settings)
    if "Emissive_Color" in settings:
        settings["baseColor"] = settings["Emissive_Color"]
    if "Emissive_Mult" in settings:
        settings["baseColorScale"] = settings["Emissive_Mult"]
    shape = nif.createShapeFromData(
        name, [tuple(v) for v in np.asarray(verts).tolist()], [tuple(t) for t in np.asarray(tris).tolist()],
        [tuple(u) for u in np.asarray(uvs).tolist()], [tuple(n) for n in np.asarray(normals).tolist()],
        props=BSTriShapeBuf(), parent=parent)
    shape.flags = flags
    p = shape.shader.properties
    p.bufType = PynBufferTypes.BSEffectShaderPropertyBufType
    p.bBSLightingShaderProperty = 0
    for field, value in settings.items():
        if hasattr(getattr(p, field), "_length_"):
            getattr(p, field)[:] = value
        else:
            setattr(p, field, value)
    for slot, path in textures.items():
        shape.set_texture(slot, path)
    shape.set_colors([tuple(c) for c in np.asarray(colors).tolist()])
    shape.save_shader_attributes()
    shape.has_alpha_property = True
    shape.alpha_property.properties.flags = alpha_flags
    shape.alpha_property.properties.threshold = alpha_threshold
    shape.save_alpha_property()
    return shape


def add_effect_float_controllers(nif, shape, channels):
    """Loop effect-shader variables of the shape: channels = [(variable, [(time, value), ...]), ...] with linear keys
    (variables as in EffectShaderControlledVariable: 0 emissive multiple, 8 V offset, ...). The controllers are chained
    on the shader, like the flame cards of the vanilla Dwemer green flame."""
    shader = shape.shader
    next_id = 0xFFFFFFFF
    head = None
    for variable, keys in reversed(channels):
        dp = NiFloatDataBuf()
        dp.keys.interpolation = NiKeyType.LINEAR_KEY
        linear = []
        for time, value in keys:
            k = LinearScalarKey()
            k.time = time
            k.value = value
            linear.append(k)
        data = NiFloatData(file=nif, properties=dp, keys=linear)
        ip = NiFloatInterpolatorBuf()
        ip.value = -FLT_MAX
        ip.dataID = data.id
        interp = nif.add_block(None, ip)
        cp = BSEffectShaderPropertyFloatControllerBuf()
        cp.flags = TimeControllerFlags(cycle_type=CycleType.LOOP).flags
        cp.frequency = 1.0
        cp.phase = 0.0
        cp.startTime = keys[0][0]
        cp.stopTime = keys[-1][0]
        cp.targetID = shader.id
        cp.interpolatorID = interp.id
        cp.controlledVariable = variable
        cp.nextControllerID = next_id
        head = nif.add_block(None, cp)
        next_id = head.id
    shader.controller = head
    return head


def add_emissive_loop(nif, shape, keys):
    """Loop the shape's emissive colour through keys [(time, (r, g, b)), ...], as vanilla glowing weapons do
    (BSLightingShaderPropertyColorController on the emissive colour; the last key should repeat the first)."""
    data = NiPosData.New(nif, interpolation=NiKeyType.QUADRATIC_KEY)
    for time, rgb in keys:
        k = NiAnimKeyQuadTransBuf()
        k.time = time
        for c in range(3):
            k.value[c] = rgb[c]
            k.forward[c] = 0.0
            k.backward[c] = 0.0
        data.add_key(k)
    ip = NiPoint3InterpolatorBuf()
    ip.value[:] = (-FLT_MAX, -FLT_MAX, -FLT_MAX)
    ip.dataID = data.id
    interp = nif.add_block(None, ip)
    shader = shape.shader
    cp = BSLightingShaderPropertyColorControllerBuf()
    cp.flags = TimeControllerFlags(cycle_type=CycleType.LOOP).flags
    cp.frequency = 1.0
    cp.phase = 0.0
    cp.startTime = keys[0][0]
    cp.stopTime = keys[-1][0]
    cp.targetID = shader.id
    cp.interpolatorID = interp.id
    cp.controlledVariable = LightingShaderControlledColor.EMISSIVE
    ctl = nif.add_block(None, cp, shader)
    shader.controller = ctl
    return ctl


def convex_hull(points, max_verts):
    bm = bmesh.new()
    for p in points:
        bm.verts.new(Vector(p))
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)
    drop = list({g for g in res["geom_interior"] + res["geom_unused"] if isinstance(g, bmesh.types.BMVert)})
    bmesh.ops.delete(bm, geom=drop, context="VERTS")
    hv = np.array([v.co[:] for v in bm.verts])
    bm.free()
    if len(hv) > max_verts:
        chosen = [int(np.argmax(np.linalg.norm(hv - hv.mean(0), axis=1)))]
        dist = np.linalg.norm(hv - hv[chosen[0]], axis=1)
        while len(chosen) < max_verts:
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(hv - hv[nxt], axis=1))
        return convex_hull(hv[chosen], max_verts)
    bm = bmesh.new()
    for p in hv:
        bm.verts.new(Vector(p))
    bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.index_update()
    verts = np.array([v.co[:] for v in bm.verts])
    tris = np.array([[v.index for v in f.verts] for f in bm.faces])
    planes = []
    for f in bm.faces:
        n = np.array(f.normal[:])
        d = -float(np.dot(n, np.array(f.verts[0].co[:])))
        if not any(np.dot(n, q[:3]) > 0.9999 and abs(d - q[3]) < 0.01 for q in planes):
            planes.append(np.array([n[0], n[1], n[2], d]))
    bm.free()
    return verts, tris, np.array(planes)


def mass_properties(verts, tris):
    """Volume, centroid and inertia tensor (unit density, about the centroid) of a closed triangle mesh."""
    def sub(w0, w1, w2):
        t0 = w0 + w1
        f1 = t0 + w2
        t1 = w0 * w0
        t2 = t1 + w1 * t0
        f2 = t2 + w2 * f1
        f3 = w0 * t1 + w1 * t2 + w2 * f2
        return f1, f2, f3, f2 + w0 * (f1 + w0), f2 + w1 * (f1 + w1), f2 + w2 * (f1 + w2)
    integ = np.zeros(10)
    for i0, i1, i2 in tris:
        x0, y0, z0 = verts[i0]
        x1, y1, z1 = verts[i1]
        x2, y2, z2 = verts[i2]
        a1, b1, c1 = x1 - x0, y1 - y0, z1 - z0
        a2, b2, c2 = x2 - x0, y2 - y0, z2 - z0
        d0, d1, d2 = b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - a2 * b1
        f1x, f2x, f3x, g0x, g1x, g2x = sub(x0, x1, x2)
        f1y, f2y, f3y, g0y, g1y, g2y = sub(y0, y1, y2)
        f1z, f2z, f3z, g0z, g1z, g2z = sub(z0, z1, z2)
        integ += [d0 * f1x, d0 * f2x, d1 * f2y, d2 * f2z, d0 * f3x, d1 * f3y, d2 * f3z,
                  d0 * (y0 * g0x + y1 * g1x + y2 * g2x), d1 * (z0 * g0y + z1 * g1y + z2 * g2y),
                  d2 * (x0 * g0z + x1 * g1z + x2 * g2z)]
    integ *= [1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120]
    vol = integ[0]
    cm = integ[1:4] / vol
    ixx = integ[5] + integ[6] - vol * (cm[1] ** 2 + cm[2] ** 2)
    iyy = integ[4] + integ[6] - vol * (cm[2] ** 2 + cm[0] ** 2)
    izz = integ[4] + integ[5] - vol * (cm[0] ** 2 + cm[1] ** 2)
    ixy = -(integ[7] - vol * cm[0] * cm[1])
    iyz = -(integ[8] - vol * cm[1] * cm[2])
    ixz = -(integ[9] - vol * cm[2] * cm[0])
    return vol, cm, np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def inertia_calibration(body_src):
    """Ratio between the inertia stored on a source body and the solid inertia of its convex shape."""
    shape = body_src.shape
    if shape is None or shape.blockname != "bhkConvexVerticesShape":
        return 1.0
    hv, ht, _ = convex_hull(np.array([v[:3] for v in shape.vertices]) * HSF, 256)
    vol, cm, inert = mass_properties(hv / HSF, ht)
    computed = np.diag(inert) * body_src.properties.mass / vol
    stored = np.array([body_src.properties.inertiaMatrix[k] for k in (0, 5, 10)])
    return float(np.mean(stored / computed)) if np.all(computed > 0) and np.all(stored > 0) else 1.0


def add_convex_collision(root, points, coll_src, mass, calibration, hull_max):
    """Convex hull collision around points, rigid body settings copied from coll_src (a bhkCollisionObject)."""
    body_src = coll_src.body
    hull_v, hull_t, planes = convex_hull(np.unique(np.round(points, 4), axis=0), hull_max)
    hvh = hull_v / HSF
    vol, cm, inert = mass_properties(hvh, hull_t)
    inert_mass = inert * mass / vol * calibration
    coll = root.add_collision(None, flags=coll_src.flags)
    props = body_src.properties.copy()
    props.shapeID = 0xFFFFFFFF
    props.translation[:] = (0, 0, 0, 0)
    props.rotation[:] = (0, 0, 0, 1)
    for r in range(3):
        for c in range(3):
            props.inertiaMatrix[r * 4 + c] = float(inert_mass[r, c])
        props.inertiaMatrix[r * 4 + 3] = 0.0
    props.center[:] = (float(cm[0]), float(cm[1]), float(cm[2]), 0.0)
    props.mass = mass
    body = coll.add_body(props)
    cvs = bhkConvexVerticesShapeProps(game="SKYRIMSE")
    cvs.bhkMaterial = body_src.shape.properties.bhkMaterial
    cvs.bhkRadius = body_src.shape.properties.bhkRadius
    body.add_shape(cvs, vertices=hvh.tolist(),
                   normals=[(float(q[0]), float(q[1]), float(q[2]), float(q[3] / HSF)) for q in planes])
    return {"hull_verts": len(hull_v), "planes": len(planes), "mass": mass, "inertia_calibration": round(calibration, 3)}


def bounds(points):
    bmin, bmax = np.asarray(points).min(0), np.asarray(points).max(0)
    return {"min": bmin.round(2).tolist(), "max": bmax.round(2).tolist(),
            "obnd": [int(math.floor(x)) for x in bmin] + [int(math.ceil(x)) for x in bmax]}

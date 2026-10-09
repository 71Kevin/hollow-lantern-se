import os
import sys
import types

import numpy as np

ADDON = os.environ.get("HL_PYNIFLY", os.path.expandvars(r"%APPDATA%\Blender Foundation\Blender\4.3\scripts\addons"
                                                         r"\io_scene_nifly"))
if "io_scene_nifly" not in sys.modules:
    pkg = types.ModuleType("io_scene_nifly")
    pkg.__path__ = [ADDON]
    sys.modules["io_scene_nifly"] = pkg

from io_scene_nifly.pyn.nifdefs import BSTriShapeBuf, PynBufferTypes  # noqa: E402
from io_scene_nifly.pyn.pynifly import (BSXFlags, NifFile, NiStringExtraData,  # noqa: E402
                                        SkyPartition)
from io_scene_nifly.pyn.structs import TransformBuf  # noqa: E402

SHADER_FIELDS = [
    "Shader_Type", "Shader_Flags_1", "Shader_Flags_2", "Env_Map_Scale", "UV_Offset_U", "UV_Offset_V", "UV_Scale_U",
    "UV_Scale_V", "Emissive_Color", "Emissive_Mult", "textureClampMode", "Alpha", "Refraction_Str", "Glossiness",
    "Spec_Color", "Spec_Str", "Soft_Lighting", "Rim_Light_Power", "Skin_Tint_Color", "Skin_Tint_Alpha",
]


def xf_to_mat(buf):
    m = np.eye(4)
    rot = np.array([[buf.rotation[r][c] for c in range(3)] for r in range(3)])
    m[:3, :3] = rot * buf.scale
    m[:3, 3] = [buf.translation[k] for k in range(3)]
    return m


def mat_to_xf(m):
    m = np.asarray(m, dtype=float)
    scale = float(np.cbrt(np.linalg.det(m[:3, :3])))
    rot = m[:3, :3] / scale
    buf = TransformBuf()
    buf.store(m[:3, 3].tolist(), rot.tolist(), [scale] * 3)
    return buf


SKIP_FIELDS = ("bufSize", "bufType", "nameID", "controllerID", "extraDataCount", "textureSetID", "bBSLightingShaderProperty")


def shader_settings(shape):
    p = shape.shader.properties
    out = {}
    for field, _ in p._fields_:
        if field in SKIP_FIELDS or field.endswith("ID") or field.endswith("Id"):
            continue
        value = getattr(p, field)
        if isinstance(value, bytes):
            continue
        out[field] = list(value) if hasattr(value, "_length_") else value
    return out


def apply_shader(shape, settings, textures):
    p = shape.shader.properties
    for field, value in settings.items():
        if isinstance(value, (list, tuple)):
            getattr(p, field)[:] = value
        else:
            setattr(p, field, value)
    for slot, path in textures.items():
        shape.set_texture(slot, path)
    shape.save_shader_attributes()


def read_shape(shape):
    bones = list(shape.bone_names)
    unique = []
    for b in bones:
        if b not in unique:
            unique.append(b)
    nv = len(shape.verts)
    w = np.zeros((nv, len(unique)), dtype=np.float32)
    for name, pairs in shape.bone_weights.items():
        j = unique.index(name)
        for vi, wt in pairs:
            w[vi, j] += wt
    s2b = []
    for b in unique:
        buf = shape.get_shape_skin_to_bone(b)
        s2b.append(xf_to_mat(buf) if buf is not None else np.eye(4))
    textures = {k: v for k, v in shape.textures.items() if v}
    extras = [(ed.name, ed.string_data) for ed in shape.extra_data() if ed.blockname == "NiStringExtraData"]
    alpha = None
    if shape.has_alpha_property:
        alpha = (int(shape.alpha_property.properties.flags), int(shape.alpha_property.properties.threshold))
    return {
        "name": shape.name,
        "v": np.array(shape.verts, dtype=np.float64),
        "t": np.array(shape.tris, dtype=np.int64),
        "uv": np.array(shape.uvs, dtype=np.float64),
        "n": np.array(shape.normals, dtype=np.float64) if shape.normals else None,
        "bones": unique,
        "w": w,
        "s2b": np.array(s2b),
        "xf": xf_to_mat(shape.transform),
        "partitions": [p.id for p in shape.partitions],
        "partition_tris": list(shape.partition_tris) if shape.partitions else [],
        "shader": shader_settings(shape),
        "shader_block": shape.shader.blockname,
        "textures": textures,
        "alpha": alpha,
        "extras": extras,
        "flags": shape.flags,
    }


def limit_weights(w, max_bones=4, floor=1e-4):
    w = np.array(w, dtype=np.float64)
    if w.shape[1] > max_bones:
        order = np.argsort(-w, axis=1)
        keep = np.zeros_like(w, dtype=bool)
        rows = np.arange(w.shape[0])[:, None]
        keep[rows, order[:, :max_bones]] = True
        w = np.where(keep, w, 0.0)
    w[w < floor] = 0.0
    s = w.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return w / s


def new_nif(path, root_name="Scene Root"):
    nif = NifFile()
    nif.initialize("SKYRIMSE", path, "NiNode", root_name)
    return nif


def add_root_string(nif, name, value):
    NiStringExtraData.New(nif, name=name, string_value=value, parent=nif.rootNode)


def add_bsx(nif, flags):
    BSXFlags.New(nif, name="BSX", flags=flags, parent=nif.rootNode)


def add_skinned_shape(nif, name, v, t, uv, n, bones, w, bone_globals, partition_id, shader, textures,
                      alpha=None, extras=(), shader_block="BSLightingShaderProperty", flags=None, colors=None):
    props = BSTriShapeBuf()
    shape = nif.createShapeFromData(
        name, [tuple(x) for x in np.asarray(v, dtype=float).tolist()],
        [tuple(int(i) for i in x) for x in np.asarray(t).tolist()],
        [tuple(x) for x in np.asarray(uv, dtype=float).tolist()],
        [tuple(x) for x in np.asarray(n, dtype=float).tolist()], props=props, parent=nif.rootNode)
    if flags is not None:
        shape.flags = flags
    if shader_block == "BSEffectShaderProperty":
        p = shape.shader.properties
        p.bufType = PynBufferTypes.BSEffectShaderPropertyBufType
        p.bBSLightingShaderProperty = 0
    apply_shader(shape, shader, textures)
    if colors is not None:
        shape.set_colors([tuple(c) for c in np.asarray(colors, dtype=float).tolist()])
    w = limit_weights(w)
    used = [j for j in range(len(bones)) if w[:, j].max() > 0]
    shape.skin()
    shape.set_global_to_skin(TransformBuf())
    for j in used:
        shape.add_bone(bones[j], mat_to_xf(bone_globals[bones[j]]))
    for j in used:
        shape.set_skin_to_bone_xform(bones[j], mat_to_xf(np.linalg.inv(bone_globals[bones[j]])))
    for j in used:
        idx = np.nonzero(w[:, j])[0]
        shape.setShapeWeights(bones[j], [(int(i), float(w[i, j])) for i in idx])
    ids = partition_id if isinstance(partition_id, (list, tuple)) else [partition_id]
    parts = [SkyPartition(part_id=pid, namedict=nif.dict) for pid in ids]
    tri_parts = [0] * len(t) if len(ids) == 1 else None
    shape.set_partitions(parts, tri_parts)
    if alpha is not None:
        shape.has_alpha_property = True
        shape.alpha_property.properties.flags = alpha[0]
        shape.alpha_property.properties.threshold = alpha[1]
        shape.save_alpha_property()
    for ename, evalue in extras:
        NiStringExtraData.New(nif, name=ename, string_value=evalue, parent=shape)
    return shape


def add_static_shape(nif, name, v, t, uv, n, shader, textures, alpha=None, parent=None, flags=None,
                     shader_block="BSLightingShaderProperty", colors=None):
    props = BSTriShapeBuf()
    shape = nif.createShapeFromData(
        name, [tuple(x) for x in np.asarray(v, dtype=float).tolist()],
        [tuple(int(i) for i in x) for x in np.asarray(t).tolist()],
        [tuple(x) for x in np.asarray(uv, dtype=float).tolist()],
        [tuple(x) for x in np.asarray(n, dtype=float).tolist()], props=props,
        parent=parent if parent is not None else nif.rootNode)
    if flags is not None:
        shape.flags = flags
    if shader_block == "BSEffectShaderProperty":
        p = shape.shader.properties
        p.bufType = PynBufferTypes.BSEffectShaderPropertyBufType
        p.bBSLightingShaderProperty = 0
    apply_shader(shape, shader, textures)
    if colors is not None:
        shape.set_colors([tuple(c) for c in np.asarray(colors, dtype=float).tolist()])
    if alpha is not None:
        shape.has_alpha_property = True
        shape.alpha_property.properties.flags = alpha[0]
        shape.alpha_property.properties.threshold = alpha[1]
        shape.save_alpha_property()
    return shape


def skeleton_globals(path):
    f = NifFile(path)
    return {name: xf_to_mat(node.global_transform) for name, node in f.nodes.items()}

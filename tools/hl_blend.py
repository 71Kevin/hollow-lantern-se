import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mesh_object(name, v, t, uv=None, color=(0.6, 0.6, 0.6, 1.0), smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in np.asarray(v)], [], [tuple(map(int, f)) for f in np.asarray(t)])
    me.update()
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        loops = np.zeros(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", loops)
        coords = np.asarray(uv, dtype=np.float64)[loops]
        coords[:, 1] = 1.0 - coords[:, 1]
        layer.data.foreach_set("uv", coords.ravel().astype(np.float32))
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.color = color
    return ob


def bvh(v, t):
    return BVHTree.FromPolygons([tuple(map(float, p)) for p in np.asarray(v)],
                                [tuple(map(int, f)) for f in np.asarray(t)], all_triangles=True)


def nearest(tree, points, max_dist=1e9):
    loc = np.zeros((len(points), 3))
    nrm = np.zeros((len(points), 3))
    face = np.full(len(points), -1, dtype=np.int64)
    dist = np.full(len(points), np.inf)
    for i, p in enumerate(np.asarray(points)):
        hit = tree.find_nearest(Vector(p), max_dist)
        if hit[0] is not None:
            loc[i] = hit[0]
            nrm[i] = hit[1]
            face[i] = hit[2]
            dist[i] = hit[3]
    return loc, nrm, face, dist


def raycast(tree, origins, dirs, max_dist=1e9):
    loc = np.full((len(origins), 3), np.nan)
    nrm = np.zeros((len(origins), 3))
    face = np.full(len(origins), -1, dtype=np.int64)
    dist = np.full(len(origins), np.inf)
    for i in range(len(origins)):
        hit = tree.ray_cast(Vector(origins[i]), Vector(dirs[i]), max_dist)
        if hit[0] is not None:
            loc[i] = hit[0]
            nrm[i] = hit[1]
            face[i] = hit[2]
            dist[i] = hit[3]
    return loc, nrm, face, dist


def barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den[den == 0] = 1e-12
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    bc = np.stack([u, v, w], 1)
    bc = np.clip(bc, 0, None)
    return bc / bc.sum(1, keepdims=True)


def setup_render(res=1024, engine="BLENDER_WORKBENCH", color_type="MATERIAL"):
    sc = bpy.context.scene
    sc.render.engine = engine
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.film_transparent = False
    if engine == "BLENDER_WORKBENCH":
        sh = sc.display.shading
        sh.light = "STUDIO"
        sh.color_type = color_type
        sh.show_cavity = True
        sh.cavity_type = "BOTH"
        sh.show_specular_highlight = True
        sc.display.shading.background_type = "VIEWPORT"
        sc.display.shading.background_color = (0.82, 0.82, 0.82)
    return sc


def camera(center, direction, ortho_scale, distance=200.0, up=(0, 0, 1), lens_ortho=True):
    cam_data = bpy.data.cameras.new("cam")
    if lens_ortho:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho_scale
    cam_data.clip_end = 2000
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    d = Vector(direction).normalized()
    cam.location = Vector(center) - d * distance
    rot = d.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot.to_euler()
    bpy.context.scene.camera = cam
    return cam


VIEWS = {
    "front": (0, -1, 0),
    "back": (0, 1, 0),
    "left": (1, 0, 0),
    "right": (-1, 0, 0),
    "q34": (0.7, -0.7, -0.12),
    "q34b": (-0.7, 0.7, -0.12),
    "top": (0, 0, -1),
    "below": (0, 0.2, 1),
}


def render_views(out_prefix, center, scale, views=("front", "left", "back"), res=1024):
    setup_render(res)
    for name in views:
        d = VIEWS[name] if isinstance(name, str) else name
        cam = camera(center, d, scale)
        bpy.context.scene.render.filepath = "%s_%s.png" % (out_prefix, name if isinstance(name, str) else "v")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)


def unwrap(v, t, method="CONFORMAL", margin=0.0):
    me = bpy.data.meshes.new("unwrap_tmp")
    me.from_pydata([tuple(map(float, p)) for p in np.asarray(v)], [], [tuple(map(int, f)) for f in np.asarray(t)])
    me.update()
    ob = bpy.data.objects.new("unwrap_tmp", me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    for o in bpy.context.view_layer.objects:
        o.select_set(o == ob)
    me.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method=method, margin=margin)
    bpy.ops.object.mode_set(mode="OBJECT")
    loops = np.zeros(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loops)
    luv = np.zeros(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers[0].data.foreach_get("uv", luv)
    luv = luv.reshape(-1, 2).astype(np.float64)
    uv = np.zeros((len(v), 2))
    cnt = np.zeros(len(v))
    np.add.at(uv, loops, luv)
    np.add.at(cnt, loops, 1)
    uv /= np.maximum(cnt, 1)[:, None]
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    tri = np.asarray(t)
    a3 = 0.5 * np.linalg.norm(np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]), axis=1).sum()
    e1 = uv[tri[:, 1]] - uv[tri[:, 0]]
    e2 = uv[tri[:, 2]] - uv[tri[:, 0]]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if signed.sum() < 0:
        uv[:, 0] = -uv[:, 0]
    a2 = np.abs(signed).sum()
    uv *= np.sqrt(a3 / max(a2, 1e-12))
    uv -= uv.min(0)
    return uv


def voxel_union(meshes, voxel=0.15, smooth_shading=False):
    obs = []
    for k, (v, t) in enumerate(meshes):
        obs.append(mesh_object("vu_%d" % k, v, t, smooth=False))
    for o in bpy.context.view_layer.objects:
        o.select_set(o in obs)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.data.remesh_voxel_size = voxel
    ob.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.index_update()
    v = np.array([p.co[:] for p in bm.verts])
    t = np.array([[p.index for p in f.verts] for f in bm.faces])
    bm.free()
    bpy.data.objects.remove(ob)
    return v, t

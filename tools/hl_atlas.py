import numpy as np

import hl_geom

PATCHES = {
    "patch:lining": 448,
    "patch:edge": 320,
    "patch:brass": 512,
    "patch:cord": 320,
    "patch:sole": 384,
    "patch:wax": 192,
    "patch:flesh": 256,
    "patch:stem": 192,
    "patch:steel": 320,
    "patch:ebony": 256,
    "patch:ember": 160,
}


def snap_fragments(parts, share=0.05):
    for p in parts:
        if p.island is None or p.island.startswith("patch:") or p.uv is None or len(p.t) == 0:
            continue
        e = hl_geom.unique_edges(p.t)
        lab = hl_geom.components(len(p.v), e[:, 0], e[:, 1])
        tri_lab = lab[p.t[:, 0]]
        roots, counts = np.unique(tri_lab, return_counts=True)
        if len(roots) < 2:
            continue
        main = roots[np.argmax(counts)]
        main_v = np.unique(p.t[tri_lab == main])
        uv = np.array(p.uv, dtype=np.float64)
        for r, c in zip(roots, counts):
            if r == main or c > share * len(p.t):
                continue
            frag_v = np.unique(p.t[tri_lab == r])
            probe = frag_v[::max(1, len(frag_v) // 64)]
            d = np.linalg.norm(p.v[probe][:, None] - p.v[main_v][None], axis=2)
            fi, mi = np.unravel_index(np.argmin(d), d.shape)
            uv[frag_v] += uv[main_v[mi]] - uv[probe[fi]]
        p.uv = uv


def island_boxes(parts):
    boxes = {}
    for p in parts:
        if p.island is None or p.island.startswith("patch:") or p.uv is None:
            continue
        lo, hi = p.uv.min(0), p.uv.max(0)
        if p.island in boxes:
            a, b = boxes[p.island]
            boxes[p.island] = (np.minimum(a, lo), np.maximum(b, hi))
        else:
            boxes[p.island] = (lo, hi)
    return boxes


def shelf_pack(items, size, pad):
    order = sorted(items, key=lambda k: (-items[k][1], -items[k][0]))
    x = y = pad
    row_h = 0
    place = {}
    for key in order:
        w, h = items[key]
        if w + 2 * pad > size:
            return None
        if x + w + pad > size:
            x = pad
            y += row_h + pad
            row_h = 0
        if y + h + pad > size:
            return None
        place[key] = (x, y)
        x += w + pad
        row_h = max(row_h, h)
    return place


def pack(parts, size=8192, pad=24, patch_scale=1.0):
    snap_fragments(parts)
    boxes = island_boxes(parts)
    rot = {k: (b[1][1] - b[0][1]) > (b[1][0] - b[0][0]) for k, b in boxes.items()}
    patches = {k: int(v * patch_scale) for k, v in PATCHES.items()
               if any(p.island == k for p in parts)}
    lo, hi = 1.0, 2000.0
    best = None
    for _ in range(40):
        s = 0.5 * (lo + hi)
        items = {}
        for k, b in boxes.items():
            ext = (int(np.ceil((b[1][0] - b[0][0]) * s)) + 1, int(np.ceil((b[1][1] - b[0][1]) * s)) + 1)
            items[k] = ext[::-1] if rot[k] else ext
        items.update({k: (v, v) for k, v in patches.items()})
        place = shelf_pack(items, size, pad)
        if place is None:
            hi = s
        else:
            lo = s
            best = (s, place, items)
    s, place, items = best
    layout = {"size": size, "scale": s, "islands": {}}
    for k, (x, y) in place.items():
        if k in boxes:
            b = boxes[k]
            layout["islands"][k] = {"x": x, "y": y, "w": items[k][0], "h": items[k][1], "origin": b[0],
                                    "scale": s, "patch": False, "rot": rot[k],
                                    "ext_u": float((b[1][0] - b[0][0]) * s)}
        else:
            layout["islands"][k] = {"x": x, "y": y, "w": items[k][0], "h": items[k][1], "origin": None,
                                    "scale": None, "patch": True, "rot": False, "ext_u": 0.0}
    return layout


def to_pixels(uv, isl):
    local = (uv - isl["origin"]) * isl["scale"]
    if isl["rot"]:
        local = np.stack([local[:, 1], isl["ext_u"] - local[:, 0]], 1)
    return np.array([isl["x"], isl["y"]]) + local


def apply(parts, layout, margin=6):
    size = layout["size"]
    for p in parts:
        isl = layout["islands"][p.island]
        if isl["patch"]:
            lo, hi = p.uv.min(0), p.uv.max(0)
            span = np.maximum(hi - lo, 1e-6)
            k = (isl["w"] - 2 * margin) / span.max()
            px = np.array([isl["x"], isl["y"]]) + margin + (p.uv - lo) * k
        else:
            px = to_pixels(p.uv, isl)
        p.uv_px = px
        p.uv_final = px / size
    return parts

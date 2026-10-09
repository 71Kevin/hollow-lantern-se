import struct

import numpy as np


def read_tri(path):
    data = open(path, "rb").read()
    if data[:8] != b"FRTRI003":
        raise ValueError("not a FRTRI003 file: %s" % path)
    (nv, nt, nq, nlv, nls, ntc, ext, ndm, nsm, nsv) = struct.unpack_from("<10i", data, 8)
    pos = 8 + 40 + 16
    verts = np.frombuffer(data, "<f4", (nv + nsv) * 3, pos).reshape(-1, 3).astype(np.float64)
    pos += (nv + nsv) * 12
    tris = np.frombuffer(data, "<i4", nt * 3, pos).reshape(-1, 3)
    pos += nt * 12
    pos += nq * 16
    pos += nlv * 4
    pos += nls * 20
    if ext & 1:
        pos += ntc * 8
        pos += (nt * 3 + nq * 4) * 4
    morphs = {}
    for _ in range(ndm):
        n, = struct.unpack_from("<i", data, pos)
        pos += 4
        name = data[pos:pos + n].split(b"\0")[0].decode("latin-1")
        pos += n
        scale, = struct.unpack_from("<f", data, pos)
        pos += 4
        d = np.frombuffer(data, "<i2", nv * 3, pos).reshape(-1, 3).astype(np.float64) * scale
        pos += nv * 6
        morphs[name] = d
    base = verts[:nv]
    stat = verts[nv:]
    k = 0
    for _ in range(nsm):
        n, = struct.unpack_from("<i", data, pos)
        pos += 4
        name = data[pos:pos + n].split(b"\0")[0].decode("latin-1")
        pos += n
        cnt, = struct.unpack_from("<i", data, pos)
        pos += 4
        idx = np.frombuffer(data, "<i4", cnt, pos)
        pos += cnt * 4
        d = np.zeros_like(base)
        d[idx] = stat[k:k + cnt] - base[idx]
        k += cnt
        morphs[name] = d
    return base, tris, morphs

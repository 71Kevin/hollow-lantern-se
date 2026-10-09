"""Extract single files from a Skyrim SE BSA (v105), or list its contents.

bsa_get.py <archive> <out_dir> <path> [<path>...]
bsa_get.py <archive> --list [substring ...]      (paths containing every substring, case-insensitive)
"""
import os
import struct
import sys

import lz4.frame

BS = chr(92)


def norm(p):
    return p.lower().replace('/', BS)


def read_entries(f):
    magic, ver, off, flags, nfold, nfile, lfold, lfile, fflags = struct.unpack('<4sIIIIIIII', f.read(36))
    assert magic == b'BSA\0' and ver == 105, (magic, ver)
    folders = [struct.unpack('<QIIQ', f.read(24)) for _ in range(nfold)]
    entries = []
    for h, count, pad, foff in folders:
        nlen = f.read(1)[0]
        fname = f.read(nlen)[:-1].decode('cp1252')
        for _ in range(count):
            fh, size, doff = struct.unpack('<QII', f.read(16))
            entries.append([fname, fh, size, doff])
    names = f.read(lfile).split(b'\0')
    for e, n in zip(entries, names):
        e.append(n.decode('cp1252'))
    return flags, entries


def read_bsa(path, wanted):
    wanted = {norm(w): w for w in wanted}
    f = open(path, 'rb')
    flags, entries = read_entries(f)
    compressed_default = bool(flags & 0x4)
    embed_names = bool(flags & 0x100)
    out = {}
    for folder, fh, size, doff, name in entries:
        full = norm(folder + BS + name)
        if full not in wanted:
            continue
        comp = compressed_default ^ bool(size & 0x40000000)
        size &= 0x3FFFFFFF
        f.seek(doff)
        if embed_names:
            ln = f.read(1)[0]
            f.read(ln)
            size -= ln + 1
        if comp:
            orig = struct.unpack('<I', f.read(4))[0]
            data = lz4.frame.decompress(f.read(size - 4))
            assert len(data) == orig
        else:
            data = f.read(size)
        out[wanted[full]] = data
    return out


if __name__ == '__main__':
    arc = sys.argv[1]
    if sys.argv[2] == '--list':
        needles = [norm(n) for n in sys.argv[3:]]
        with open(arc, 'rb') as f:
            _, entries = read_entries(f)
        for folder, fh, size, doff, name in entries:
            full = folder + BS + name
            if all(n in full.lower() for n in needles):
                print(full)
        sys.exit(0)
    outdir, paths = sys.argv[2], sys.argv[3:]
    got = read_bsa(arc, paths)
    for p in paths:
        if p not in got:
            print('NOT FOUND', p)
            continue
        dst = os.path.join(outdir, *norm(p).split(BS))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as fh:
            fh.write(got[p])
        print('extracted', p, len(got[p]))

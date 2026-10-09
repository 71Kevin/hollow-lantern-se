import re
import struct
from xml.sax.saxutils import escape

import numpy as np

OSD_MAGIC = 0x4F534400


def read_osd(path):
    b = open(path, "rb").read()
    magic, version, count = struct.unpack_from("<III", b, 0)
    if magic != OSD_MAGIC or version != 1:
        raise ValueError("not an OSD file: %s" % path)
    off = 12
    out = {}
    for _ in range(count):
        n = b[off]
        off += 1
        name = b[off:off + n].decode("latin-1")
        off += n
        k, = struct.unpack_from("<H", b, off)
        off += 2
        rec = np.frombuffer(b, dtype=np.dtype([("i", "<u2"), ("d", "<f4", 3)]), count=k, offset=off)
        off += 14 * k
        out[name] = (rec["i"].astype(np.int64), rec["d"].astype(np.float64))
    return out


def write_osd(path, diffs):
    parts = [struct.pack("<III", OSD_MAGIC, 1, len(diffs))]
    for name, (idx, d) in diffs.items():
        raw = name.encode("latin-1")
        idx = np.asarray(idx, dtype=np.int64)
        d = np.asarray(d, dtype=np.float64)
        if idx.max(initial=0) > 0xFFFF or len(idx) > 0xFFFF:
            raise ValueError("OSD index overflow in %s" % name)
        rec = np.zeros(len(idx), dtype=np.dtype([("i", "<u2"), ("d", "<f4", 3)]))
        rec["i"] = idx
        rec["d"] = d
        parts.append(struct.pack("<B", len(raw)) + raw + struct.pack("<H", len(idx)) + rec.tobytes())
    with open(path, "wb") as f:
        f.write(b"".join(parts))


def write_tri(path, shapes):
    out = [b"PIRT", struct.pack("<H", len(shapes))]
    for shape_name, morphs in shapes:
        raw = shape_name.encode("latin-1")
        out.append(struct.pack("<B", len(raw)) + raw + struct.pack("<H", len(morphs)))
        for morph_name, (idx, d) in morphs:
            mraw = morph_name.encode("latin-1")
            d = np.asarray(d, dtype=np.float64)
            mult = float(np.abs(d).max()) / 32767.0 if len(d) else 0.0
            q = np.zeros((len(idx), 3), dtype=np.int16) if mult == 0 else np.round(d / mult).astype(np.int16)
            rec = np.zeros(len(idx), dtype=np.dtype([("i", "<u2"), ("q", "<i2", 3)]))
            rec["i"] = idx
            rec["q"] = q
            out.append(struct.pack("<B", len(mraw)) + mraw + struct.pack("<fH", mult, len(idx)) + rec.tobytes())
    out.append(struct.pack("<H", 0))
    with open(path, "wb") as f:
        f.write(b"".join(out))


def read_tri(path):
    b = open(path, "rb").read()
    assert b[:4] == b"PIRT"
    off = 4
    ns, = struct.unpack_from("<H", b, off)
    off += 2
    shapes = []
    for _ in range(ns):
        n = b[off]
        off += 1
        sname = b[off:off + n].decode("latin-1")
        off += n
        nm, = struct.unpack_from("<H", b, off)
        off += 2
        morphs = []
        for _ in range(nm):
            n = b[off]
            off += 1
            mname = b[off:off + n].decode("latin-1")
            off += n
            mult, k = struct.unpack_from("<fH", b, off)
            off += 6
            rec = np.frombuffer(b, dtype=np.dtype([("i", "<u2"), ("q", "<i2", 3)]), count=k, offset=off)
            off += 8 * k
            morphs.append((mname, (rec["i"].astype(np.int64), rec["q"].astype(np.float64) * mult)))
        shapes.append((sname, morphs))
    return shapes


def read_slider_set(osp_path, set_name):
    s = open(osp_path, encoding="utf-8-sig").read()
    m = re.search(r'<SliderSet name="%s">(.*?)</SliderSet>' % re.escape(set_name), s, re.S)
    body = m.group(1)
    sliders = []
    for sm in re.finditer(r"<Slider ([^>]*)>(.*?)</Slider>", body, re.S):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', sm.group(1)))
        data = []
        for dm in re.finditer(r"<Data ([^>]*)>([^<]*)</Data>", sm.group(2)):
            da = dict(re.findall(r'(\w+)="([^"]*)"', dm.group(1)))
            data.append((da.get("name"), da.get("target"), dm.group(2)))
        sliders.append((attrs, data))
    return sliders


def slider_set_xml(name, folder, source, output_path, output_file, shapes, sliders, gen_weights=True):
    lines = ['    <SliderSet name="%s">' % escape(name),
             "        <DataFolder>%s</DataFolder>" % escape(folder),
             "        <SourceFile>%s</SourceFile>" % escape(source),
             "        <OutputPath>%s</OutputPath>" % escape(output_path),
             '        <OutputFile GenWeights="%s">%s</OutputFile>' % ("true" if gen_weights else "false",
                                                                     escape(output_file))]
    for shape in shapes:
        lines.append('        <Shape target="%s">%s</Shape>' % (escape(shape), escape(shape)))
    for attrs, data in sliders:
        attr_txt = " ".join('%s="%s"' % (k, escape(str(v))) for k, v in attrs.items())
        lines.append("        <Slider %s>" % attr_txt)
        for dname, target, dfile in data:
            lines.append('            <Data name="%s" target="%s" local="true">%s</Data>' % (
                escape(dname), escape(target), escape(dfile)))
        lines.append("        </Slider>")
    lines.append("    </SliderSet>")
    return "\n".join(lines)


def write_osp(path, set_blocks):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = '<?xml version="1.0" encoding="UTF-8"?>\n<SliderSetInfo version="1">\n%s\n</SliderSetInfo>\n' % "\n".join(
        set_blocks)
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text)


def write_groups(path, groups):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<SliderGroups>"]
    for group, members in groups.items():
        lines.append('    <Group name="%s">' % escape(group))
        for m in members:
            lines.append('        <Member name="%s"/>' % escape(m))
        lines.append("    </Group>")
    lines.append("</SliderGroups>")
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")

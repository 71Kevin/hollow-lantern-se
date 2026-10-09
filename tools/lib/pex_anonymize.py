"""Blank the user and computer names that the Papyrus compiler writes into a .pex header.

python pex_anonymize.py <file.pex> [...]
"""
import struct
import sys

for path in sys.argv[1:]:
    data = open(path, "rb").read()
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic != 0xFA57C0DE:
        raise SystemExit("%s: not a Skyrim .pex file" % path)
    pos = 16
    header = []
    for _ in range(3):
        n = struct.unpack_from(">H", data, pos)[0]
        header.append(data[pos + 2:pos + 2 + n])
        pos += 2 + n
    source = header[0]
    rebuilt = data[:16] + struct.pack(">H", len(source)) + source + struct.pack(">H", 0) + struct.pack(">H", 0) + data[pos:]
    with open(path, "wb") as fh:
        fh.write(rebuilt)
    print("%s: removed %r / %r" % (path, header[1].decode("latin1"), header[2].decode("latin1")))

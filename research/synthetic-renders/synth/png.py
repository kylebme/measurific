"""Streaming lossless PNG writers used by Blender's render worker."""

import os
import struct
import zlib


_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(stream, name, data):
    stream.write(struct.pack(">I", len(data)))
    stream.write(name)
    stream.write(data)
    stream.write(struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF))


def write_png(path, width, height, bit_depth, color_type, rows):
    """Write one PNG row at a time; rows are already packed into PNG byte order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(_SIGNATURE)
        _chunk(stream, b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0))
        encoder = zlib.compressobj(level=6)
        buffer = bytearray()
        count = 0
        for row in rows:
            buffer.extend(encoder.compress(b"\x00" + row))
            count += 1
            if len(buffer) >= 1 << 20:
                _chunk(stream, b"IDAT", bytes(buffer))
                buffer.clear()
        if count != height:
            raise ValueError(f"expected {height} rows, received {count}")
        buffer.extend(encoder.flush())
        if buffer:
            _chunk(stream, b"IDAT", bytes(buffer))
        _chunk(stream, b"IEND", b"")


def write_gray16(path, pixels):
    """pixels: 2-D NumPy uint16 array, top row first."""
    height, width = pixels.shape
    def rows():
        for row in pixels:
            yield row.astype(">u2", copy=False).tobytes()
    write_png(path, width, height, 16, 0, rows())


def write_binary(path, pixels):
    """pixels: 2-D boolean or 0/1 NumPy array, top row first."""
    import numpy as np
    height, width = pixels.shape
    def rows():
        for row in pixels:
            yield np.packbits(row, bitorder="big").tobytes()
    write_png(path, width, height, 1, 0, rows())

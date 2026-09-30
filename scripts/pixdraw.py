"""Tiny numpy drawing helpers (boxes, dashed boxes, 5x7 bitmap text).

Why this exists: this machine's ffmpeg has no drawtext/libass and we have no OpenCV/Pillow,
and adding a dependency just to put digits on a video is not worth it. Frames are decoded
by ffmpeg, drawn on here, and piped back to ffmpeg for encoding.
"""
from __future__ import annotations

import colorsys

import numpy as np

_G = {
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    "A": "01110 10001 10001 11111 10001 10001 10001", "C": "01110 10001 10000 10000 10000 10001 01110",
    "D": "11110 10001 10001 10001 10001 10001 11110", "E": "11111 10000 10000 11110 10000 10000 11111",
    "F": "11111 10000 10000 11110 10000 10000 10000", "G": "01110 10001 10000 10111 10001 10001 01111",
    "H": "10001 10001 10001 11111 10001 10001 10001", "I": "01110 00100 00100 00100 00100 00100 01110",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "R": "11110 10001 10001 11110 10100 10010 10001", "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100", "U": "10001 10001 10001 10001 10001 10001 01110",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "-": "00000 00000 00000 11111 00000 00000 00000",
    " ": "00000 00000 00000 00000 00000 00000 00000",
}
FONT = {k: np.array([[c == "1" for c in row] for row in v.split()], dtype=bool) for k, v in _G.items()}


def id_color(i: int) -> tuple[int, int, int]:
    """Stable, well separated colour per track id (golden-ratio hue walk)."""
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def text_size(s: str, scale: int) -> tuple[int, int]:
    return len(s) * 6 * scale - scale, 7 * scale


def draw_text(img: np.ndarray, s: str, x: int, y: int, scale: int = 2,
              fg=(255, 255, 255), bg=(0, 0, 0)) -> None:
    w, h = text_size(s, scale)
    pad = scale
    H, W = img.shape[:2]
    x, y = int(np.clip(x, pad, max(pad, W - w - pad))), int(np.clip(y, pad, max(pad, H - h - pad)))
    img[y - pad : y + h + pad, x - pad : x + w + pad] = bg
    for k, ch in enumerate(s.upper()):
        glyph = FONT.get(ch, FONT[" "])
        big = np.kron(glyph, np.ones((scale, scale), dtype=bool))
        x0 = x + k * 6 * scale
        region = img[y : y + h, x0 : x0 + 5 * scale]
        region[big[: region.shape[0], : region.shape[1]]] = fg


def draw_box(img: np.ndarray, box, color, thick: int = 2, dashed: bool = False) -> None:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = np.clip([x1, x2], 0, W - 1)
    y1, y2 = np.clip([y1, y2], 0, H - 1)
    if x2 <= x1 or y2 <= y1:
        return
    if not dashed:
        img[y1 : y1 + thick, x1:x2] = color
        img[max(y2 - thick, y1) : y2, x1:x2] = color
        img[y1:y2, x1 : x1 + thick] = color
        img[y1:y2, max(x2 - thick, x1) : x2] = color
        return
    on = np.arange(max(x2 - x1, y2 - y1) + 1) % 12 < 7  # 7 px on, 5 px off
    img[y1 : y1 + thick, x1:x2][:, on[: x2 - x1]] = color
    img[max(y2 - thick, y1) : y2, x1:x2][:, on[: x2 - x1]] = color
    img[y1:y2, x1 : x1 + thick][on[: y2 - y1], :] = color
    img[y1:y2, max(x2 - thick, x1) : x2][on[: y2 - y1], :] = color

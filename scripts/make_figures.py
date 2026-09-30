"""Qualitative figure: crops around long-gap (>= 40 frames) reappearances for the full tracker.

Selection rule (fixed, not cherry-picked): among all >= 40-frame gaps of `ours` on the real
detections, show the longest successful re-identification and the 3 longest failures.
Each row: last MATCHED frame before the gap | mid-gap (object hidden, GT position) | first
MATCHED frame after (the two frames recovery is judged on) | 10 frames later. Border colour = predicted track id; the number is that id ('-' = no box).
"""
from __future__ import annotations

import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np

from eval.clear import clear_match
from eval.gaps import gap_recovery
from eval.mot_io import hidden_frames, load_mot_gt
from eval.common import iou_matrix
from scripts.pixdraw import draw_box, draw_text, id_color
from scripts.runlib import load_config, prepare, run_method

CROP_W, CROP_H = 120, 260


def write_png(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(t: bytes, d: bytes) -> bytes:
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def read_jpg(seq_dir: Path, frame0: int) -> np.ndarray:
    kv = dict(l.split("=") for l in (seq_dir / "seqinfo.ini").read_text().split() if "=" in l)
    w, h = int(kv["imWidth"]), int(kv["imHeight"])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(seq_dir / "img1" / f"{frame0 + 1:06d}.jpg"),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def crop(img: np.ndarray, box, pid, label: str, hidden: bool) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hh = max(y2 - y1, 40) * 0.75
    ww = hh * CROP_W / CROP_H
    xa, ya = int(np.clip(cx - ww, 0, W - 2)), int(np.clip(cy - hh, 0, H - 2))
    xb, yb = int(np.clip(cx + ww, xa + 2, W)), int(np.clip(cy + hh, ya + 2, H))
    sub = img[ya:yb, xa:xb]
    ys = (np.arange(CROP_H) * sub.shape[0] / CROP_H).astype(int)
    xs = (np.arange(CROP_W) * sub.shape[1] / CROP_W).astype(int)
    out = sub[ys][:, xs].copy()
    col = (255, 255, 255) if pid is None else id_color(int(pid))
    draw_box(out, (0, 0, CROP_W, CROP_H), col, 4, dashed=hidden)
    draw_text(out, label, 6, 8, 2)
    draw_text(out, "ID -" if pid is None else f"ID {pid}", 6, CROP_H - 22, 2, (0, 0, 0), col)
    return out


def main() -> None:
    cfg = load_config()
    root = Path(cfg["data"]["root"])
    events = []
    for s in cfg["data"]["sequences"]:
        gt, vis = load_mot_gt(root / s)
        hid = hidden_frames(vis, cfg["data"]["min_visibility"])
        dets = prepare(cfg, s)[0]
        pred, _ = run_method("ours", cfg, dets)
        clr = clear_match(gt, pred, cfg["eval"]["iou_match"])
        for e in gap_recovery(gt, clr, hid).events:
            if e.gap_len >= 40:
                events.append((s, gt, hid, clr, e))
    ok = sorted([x for x in events if x[4].recovered], key=lambda x: -x[4].gap_len)[:1]
    bad = sorted([x for x in events if not x[4].recovered], key=lambda x: -x[4].gap_len)[:3]
    rows = []
    for tag, (s, gt, hid, clr, e) in [("OK", x) for x in ok] + [("FAIL", x) for x in bad]:
        owner = [dict(p) for p in clr.matches]
        end = e.gap_start + e.gap_len                       # first frame the object is visible again
        seg_end = end
        while seg_end + 1 < len(gt) and e.gt_id in set(gt[seg_end + 1].ids) and (e.gt_id, seg_end + 1) not in hid:
            seg_end += 1
        # the two frames the metric actually compares: last match before, first match after
        before = max((f for f in range(e.gap_start) if e.gt_id in owner[f]), default=e.gap_start - 1)
        after = next((f for f in range(end, seg_end + 1) if e.gt_id in owner[f]), end)
        frames = [before, e.gap_start + e.gap_len // 2, after, min(after + 10, len(gt) - 1)]
        img_dir = root / s
        crops, last = [], None
        for k, f in enumerate(frames):
            pid = owner[f].get(e.gt_id)
            if pid is not None:
                last = pid
            i = list(gt[f].ids).index(e.gt_id)
            is_hidden = (e.gt_id, f) in hid
            label = f"{s[-2:]} F{f}" + (" HID" if is_hidden else "")
            crops.append(crop(read_jpg(img_dir, f), gt[f].boxes[i], pid, label, is_hidden))
        head = np.zeros((26, (CROP_W + 4) * 4, 3), np.uint8)
        draw_text(head, f"{tag} {s} GT{e.gt_id} HIDDEN {e.gap_len} FRAMES", 6, 6, 2,
                  (255, 255, 255), (0, 110, 0) if tag == "OK" else (150, 0, 0))
        strip = np.concatenate([np.pad(c, ((0, 0), (0, 4), (0, 0))) for c in crops], axis=1)
        rows.append(np.concatenate([head, np.pad(strip, ((0, 6), (0, 0), (0, 0)))], axis=0))
        print(f"{tag}: {s} gt{e.gt_id} gap {e.gap_start}..{e.gap_start + e.gap_len - 1}"
              f" recovered={e.recovered} never_reacquired={e.never_reacquired}")
    w = max(r.shape[1] for r in rows)
    fig = np.concatenate([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows], axis=0)
    Path("figures").mkdir(exist_ok=True)
    write_png(Path("figures/qualitative.png"), fig)
    print("wrote figures/qualitative.png", fig.shape)


if __name__ == "__main__":
    main()

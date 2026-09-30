"""Render side-by-side clips of the trackers on real MOT17 frames (oracle detections).

    python -m scripts.render_video                 # picks a success clip and a failure clip
    python -m scripts.render_video MOT17-02        # other sequence

Reading the picture: every box is a tracker output, coloured by its track ID (same ID = same
colour). The WHITE frame is the object we follow: solid when visible, dashed while it is
hidden behind something (the tracker gets no box for it then). If the colour of the box on
the white frame changes after it reappears, the tracker gave it a new identity.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from baselines.greedy_iou import run_greedy_iou
from baselines.kalman_iou import run_kalman_iou
from eval.clear import clear_match
from eval.gaps import gap_recovery
from eval.mot_io import hidden_frames, load_mot_gt
from eval.oracle import oracle_detections
from scripts.pixdraw import draw_box, draw_text, id_color

DATA, OUT = Path("data/MOT17"), Path("figures")
MIN_VIS, FPS, PANEL_W = 0.25, 15, 960


def seq_size(d: Path) -> tuple[int, int]:
    kv = dict(l.split("=") for l in (d / "seqinfo.ini").read_text().split() if "=" in l)
    return int(kv["imWidth"]), int(kv["imHeight"])


def read_frames(d: Path, start: int, n: int, w: int, h: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-start_number", str(start + 1), "-i", str(d / "img1/%06d.jpg"),
           "-frames:v", str(n), "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy()


def panel(img, name, pred_frame, gt_frame, focus, hidden, t, focus_pid, sc, caption):
    for pid, box in zip(pred_frame.ids, pred_frame.boxes):
        c = id_color(int(pid))
        draw_box(img, box * sc, c, 2)
        draw_text(img, str(int(pid)), int(box[0] * sc), int(box[1] * sc) - 16, 2, (0, 0, 0), c)
    fb = gt_frame.boxes[list(gt_frame.ids).index(focus)] * sc
    is_hidden = (focus, t) in hidden
    draw_box(img, fb + np.array([-5, -5, 5, 5]), (255, 255, 255), 2, dashed=is_hidden)
    draw_text(img, name, 8, 8, 3)
    draw_text(img, f"FRAME {t}", 8, 40, 2)
    pid_s = "-" if focus_pid is None else str(focus_pid)
    draw_text(img, f"FOCUS ID {pid_s}" + ("  HIDDEN" if is_hidden else ""), 8, 62, 2,
              (255, 255, 255), (150, 0, 0) if is_hidden else (0, 0, 0))
    draw_text(img, caption, 8, img.shape[0] - 24, 2)
    return img


def render(name, seq_dir, gt, hidden, preds, clears, focus, a, b, caption):
    w0, h0 = seq_size(seq_dir)
    pw, ph = PANEL_W, int(PANEL_W * h0 / w0)
    sc = pw / w0
    frames = read_frames(seq_dir, a, b - a + 1, pw, ph)
    out = OUT / f"{name}.mp4"
    OUT.mkdir(exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * pw}x{ph}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        stdin=subprocess.PIPE)
    last = {m: None for m in preds}
    for k, t in enumerate(range(a, b + 1)):
        panels = []
        for m in preds:
            owner = dict((g, p) for g, p in clears[m].matches[t])
            if focus in owner:
                last[m] = owner[focus]
            panels.append(panel(frames[k].copy(), m, preds[m][t], gt[t], focus, hidden, t,
                                last[m], sc, caption))
        enc.stdin.write(np.hstack(panels).tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote", out, f"({b - a + 1} frames)")


def main(seq: str) -> None:
    d = DATA / seq
    gt, vis = load_mot_gt(d)
    hidden = hidden_frames(vis, MIN_VIS)
    dets = oracle_detections(gt, hidden)
    preds = {"GREEDY IOU": run_greedy_iou(dets), "KALMAN IOU": run_kalman_iou(dets)}
    clears = {m: clear_match(gt, p) for m, p in preds.items()}
    ev = {m: {(e.gt_id, e.gap_start): e for e in gap_recovery(gt, clears[m], hidden).events}
          for m in preds}
    g, k = ev["GREEDY IOU"], ev["KALMAN IOU"]
    long_ = [key for key in k if k[key].gap_len >= 40 and key in g]
    win = lambda key: (max(0, key[1] - 30), min(len(gt) - 1, key[1] + k[key].gap_len + 40))
    short = [key for key in long_ if win(key)[1] - win(key)[0] <= 260]
    success = [key for key in short if k[key].recovered and not g[key].recovered]
    failure = [key for key in short if not k[key].recovered]
    cap = "ORACLE DETECTIONS"
    for label, pool in (("success", success), ("failure", failure)):
        if not pool:
            print(f"no {label} clip available in {seq}")
            continue
        key = max(pool, key=lambda q: k[q].gap_len)
        a, b = win(key)
        print(f"{label}: gt_id={key[0]} hidden {k[key].gap_len} frames from {key[1]}; clip {a}-{b}")
        render(f"{seq}_{label}_gt{key[0]}", d, gt, hidden, preds, clears, key[0], a, b, cap)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MOT17-09")

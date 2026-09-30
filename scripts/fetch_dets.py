"""Fetch MOT17's public detections (det.txt) for our sequences. These are the output of a
pretrained detector run by the benchmark authors; we treat it as a frozen black box."""
from __future__ import annotations

from pathlib import Path

from scripts.fetch_data import OUT, SEQUENCES, URL
from scripts.remote_zip import open_remote_zip


def main() -> None:
    z = open_remote_zip(URL)
    for s in SEQUENCES:
        dest = OUT / s / "det" / "det.txt"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(f"MOT17/train/{s}-SDP/det/det.txt"))
        print("wrote", dest, dest.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()

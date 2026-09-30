"""Download only the MOT17 sequences we evaluate on (Range requests, no full-zip download).

Images and GT are identical across MOT17's DPM/FRCNN/SDP variants; only det.txt differs. We
pull the SDP folder and ignore det.txt: the detector is our own frozen model, not MOT17's.
"""
from __future__ import annotations

import sys
from pathlib import Path

from scripts.remote_zip import open_remote_zip

URL = "https://motchallenge.net/data/MOT17.zip"
SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-09"]  # static camera: we do no camera-motion compensation
OUT = Path("data/MOT17")


def main() -> None:
    z = open_remote_zip(URL)
    wanted = {f"MOT17/train/{s}-SDP/": s for s in SEQUENCES}
    members = [
        i for i in z.infolist()
        if not i.is_dir() and any(i.filename.startswith(p) for p in wanted)
        and ("/img1/" in i.filename or i.filename.endswith(("gt/gt.txt", "seqinfo.ini")))
    ]
    members.sort(key=lambda i: i.header_offset)  # sequential reads -> the 1 MB cache gets reused
    for n, info in enumerate(members, 1):
        seq = wanted[next(p for p in wanted if info.filename.startswith(p))]
        rel = info.filename.split("-SDP/", 1)[1]
        dest = OUT / seq / rel
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(info))
        if n % 200 == 0:
            print(f"{n}/{len(members)}", flush=True)
    print("done", len(members), "files", flush=True)


if __name__ == "__main__":
    sys.exit(main())

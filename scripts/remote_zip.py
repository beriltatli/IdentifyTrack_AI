"""Read individual files out of a huge remote zip with HTTP Range requests (stdlib + system curl).

MOT17.zip is 5.9 GB and the disk is nearly full, so we never download the whole archive:
zipfile only needs to seek to the central directory (end of file) and then to the members
we ask for, and each seek becomes one Range request.
"""
from __future__ import annotations

import io
import subprocess
import zipfile


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20) -> None:
        self.url, self.pos, self.block = url, 0, block
        head = subprocess.run(["curl", "-sIL", "-m", "60", url], capture_output=True, text=True,
                              check=True).stdout
        lengths = [l.split(":")[1] for l in head.splitlines() if l.lower().startswith("content-length")]
        self.size = int(lengths[-1])
        self._cache_start, self._cache = -1, b""

    def seekable(self) -> bool: return True
    def readable(self) -> bool: return True
    def tell(self) -> int: return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        # curl (not urllib) because python.org builds on macOS ship without a CA bundle.
        return subprocess.run(["curl", "-sfL", "-m", "300", "-r", f"{start}-{end}", self.url],
                              capture_output=True, check=True).stdout

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n == 0:
            return b""
        s, e = self.pos, self.pos + n
        cs, ce = self._cache_start, self._cache_start + len(self._cache)
        if not (cs <= s and e <= ce):  # small reads are batched into 1 MB blocks
            want = max(n, self.block)
            self._cache = self._fetch(s, min(s + want, self.size) - 1)
            self._cache_start = s
            cs = s
        out = self._cache[s - cs : e - cs]
        self.pos = e
        return out


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(HttpRangeFile(url))

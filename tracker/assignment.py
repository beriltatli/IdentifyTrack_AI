"""Hungarian algorithm (shortest augmenting path with row/column potentials).

O(n^2 m) for an n x m problem with n <= m. Written from the textbook description;
scipy is used only in tests to check that total cost matches.

Forbidden pairs are passed as +inf (or -inf when maximising). They are swapped for a
finite BIG that exceeds any achievable finite total, so the solver first avoids
forbidden pairs, and any assignment that still lands on one is dropped from the output.
"""
from __future__ import annotations

import numpy as np


def _solve_min(c: np.ndarray) -> np.ndarray:
    """Min-cost assignment of every row of finite `c` (n x m, n <= m). Returns col per row."""
    n, m = c.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row (1-based) currently assigned to column j
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = c[i0 - 1] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            masked = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:  # walk the augmenting path back to the root
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col_of_row = np.empty(n, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j]:
            col_of_row[p[j] - 1] = j - 1
    return col_of_row


def linear_sum_assignment(
    cost: np.ndarray, maximize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one assignment. Returns (rows, cols) sorted by row.

    Rectangular inputs match min(n, m) pairs. Non-finite entries are forbidden pairs and
    never appear in the output; if that leaves fewer than min(n, m) pairs, the result is
    the cardinality-first, then min-cost, partial matching.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be 2-D")
    if cost.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    c = -cost if maximize else cost
    forbidden = ~np.isfinite(c)
    finite = c[~forbidden]
    if finite.size:
        c = np.where(forbidden, 0.0, c)
        shift = finite.min()
        c = c - shift  # all finite costs now >= 0 ; forbidden ones set to BIG below
        big = (c.max() + 1.0) * (min(c.shape) + 1)
    else:
        c = np.zeros_like(c)
        big = 1.0
    c = np.where(forbidden, big, c)
    transposed = c.shape[0] > c.shape[1]
    if transposed:
        c, forbidden = c.T, forbidden.T
    cols = _solve_min(c)
    rows = np.arange(c.shape[0])
    keep = ~forbidden[rows, cols]
    rows, cols = rows[keep], cols[keep]
    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    return rows, cols

"""Hungarian solver vs scipy: equal total cost on 500 random matrices.

scipy is used here and nowhere else in the repo. Coverage: square + rectangular shapes,
integer costs (many ties), negative costs, maximise mode, and forbidden (inf) pairs.
Forbidden cases are built so a full matching always exists, because scipy raises on
infeasible input; the infeasible case is checked against brute force instead.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment as scipy_lsa

from tracker.assignment import linear_sum_assignment


def _random_matrix(rng: np.random.Generator, kind: int) -> np.ndarray:
    n, m = int(rng.integers(1, 13)), int(rng.integers(1, 13))
    if kind == 0:  # plain floats
        return rng.random((n, m)) * 10
    if kind == 1:  # small integers -> lots of ties
        return rng.integers(0, 4, size=(n, m)).astype(float)
    if kind == 2:  # negative and positive
        return rng.normal(size=(n, m)) * 5
    # kind 3: forbidden pairs, but a full matching is planted so it stays feasible
    c = rng.random((n, m)) * 10
    forbid = rng.random((n, m)) < 0.5
    k = min(n, m)
    rows, cols = rng.permutation(n)[:k], rng.permutation(m)[:k]
    forbid[rows, cols] = False
    c[forbid] = np.inf
    return c


def _check_valid(c: np.ndarray, r: np.ndarray, col: np.ndarray) -> None:
    assert len(r) == len(col) == min(c.shape)
    assert len(set(r)) == len(r) and len(set(col)) == len(col)
    assert np.isfinite(c[r, col]).all(), "solver used a forbidden pair"


def test_500_random_matrices_match_scipy():
    rng = np.random.default_rng(12345)
    counts = {"rect": 0, "inf": 0, "max": 0}
    for i in range(500):
        c = _random_matrix(rng, i % 4)
        maximize = i % 5 == 0 and not np.isinf(c).any()
        r, col = linear_sum_assignment(c, maximize=maximize)
        sr, scol = scipy_lsa(c, maximize=maximize)
        _check_valid(c, r, col)
        assert np.isclose(c[r, col].sum(), c[sr, scol].sum()), (i, c.shape)
        counts["rect"] += c.shape[0] != c.shape[1]
        counts["inf"] += bool(np.isinf(c).any())
        counts["max"] += maximize
    # make sure the run really exercised what the brief asks for
    assert counts["rect"] > 100 and counts["inf"] > 100 and counts["max"] > 50
    print(f"\n500 matrices OK  (rectangular={counts['rect']}, with inf={counts['inf']}, "
          f"maximise={counts['max']})")


def _brute_force(c: np.ndarray) -> tuple[int, float]:
    """Max cardinality over allowed pairs, then min cost. Only for tiny matrices."""
    n, m = c.shape
    best = (0, 0.0)

    def rec(i: int, used: frozenset[int], size: int, cost: float) -> None:
        nonlocal best
        if i == n:
            if size > best[0] or (size == best[0] and cost < best[1]):
                best = (size, cost)
            return
        rec(i + 1, used, size, cost)  # leave row i unmatched
        for j in range(m):
            if j not in used and np.isfinite(c[i, j]):
                rec(i + 1, used | {j}, size + 1, cost + c[i, j])

    rec(0, frozenset(), 0, 0.0)
    return best


def test_infeasible_forbidden_pairs_match_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(100):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        c = rng.random((n, m)) * 10
        c[rng.random((n, m)) < 0.6] = np.inf  # often impossible to fill every row
        r, col = linear_sum_assignment(c)
        size, cost = _brute_force(c)
        assert len(r) == size
        assert np.isfinite(c[r, col]).all()
        assert np.isclose(c[r, col].sum(), cost)


def test_all_forbidden_and_empty():
    r, col = linear_sum_assignment(np.full((3, 3), np.inf))
    assert len(r) == 0 and len(col) == 0
    r, col = linear_sum_assignment(np.empty((0, 4)))
    assert len(r) == 0 and len(col) == 0


def test_output_is_sorted_by_row_even_when_transposed():
    c = np.array([[5.0, 1.0], [1.0, 5.0], [9.0, 9.0]])  # 3 rows > 2 cols -> internal transpose
    r, col = linear_sum_assignment(c)
    assert list(r) == sorted(r) and c[r, col].sum() == 2.0

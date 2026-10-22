"""Core estimators for map accuracy assessment and area estimation.

Pure Python + numpy, no ArcPy, so every formula can be unit tested outside
ArcGIS Pro. ``AccuracyAssessment.pyt`` imports this module and only handles
I/O (rasters, cursors, tables, messages).

Notation follows Olofsson et al. (2013, 2014):

* ``n[i, j]``  sample counts, rows = map class i, columns = reference class j
* ``W[i]``     mapped area proportion of class i (pixel count / total)
* ``p[i, j]``  estimated area proportion, ``W[i] * n[i, j] / n[i, .]``

References
----------
Olofsson, P., Foody, G. M., Stehman, S. V., & Woodcock, C. E. (2013). Making
    better use of accuracy data in land change studies: Estimating accuracy and
    area and quantifying uncertainty using stratified estimation. Remote
    Sensing of Environment, 129, 122-131.
Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., &
    Wulder, M. A. (2014). Good practices for estimating area and assessing
    accuracy of land change. Remote Sensing of Environment, 148, 42-57.
Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "z_value",
    "wilson_interval",
    "sample_size_overall",
    "allocate_sample",
    "error_matrix",
    "AccuracyResult",
    "area_adjusted_accuracy",
    "StratifiedPixelSelector",
    "cell_index",
    "block_index",
    "assign_folds",
    "buffer_split",
    "render_markdown",
    "render_html",
    "build_report",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def z_value(confidence: float = 0.95) -> float:
    """Two-sided standard normal quantile, e.g. 0.95 -> 1.959964."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1 (e.g. 0.95)")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(successes, n, confidence: float = 0.95):
    """Wilson score interval for a binomial proportion (Wilson 1927).

    Unlike the Wald interval it never has zero width when the observed
    proportion is 0 or 1, and it stays inside [0, 1]. Returns (low, high)
    arrays; NaN where n == 0.
    """
    x = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = z_value(confidence)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = x / n
        den = 1.0 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / den
        half = z / den * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
        low = np.where(n > 0, np.clip(centre - half, 0.0, 1.0), np.nan)
        high = np.where(n > 0, np.clip(centre + half, 0.0, 1.0), np.nan)
    return low, high


def _as_weights(mapped: Sequence[float]) -> np.ndarray:
    a = np.asarray(mapped, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("mapped areas must be a non-empty 1-D sequence")
    if np.any(a < 0) or a.sum() <= 0:
        raise ValueError("mapped areas must be non-negative with a positive sum")
    return a / a.sum()


# ---------------------------------------------------------------------------
# 1. Sample design
# ---------------------------------------------------------------------------

def sample_size_overall(
    weights: Sequence[float],
    users_accuracy: Sequence[float],
    target_se: float = 0.01,
    population: Optional[float] = None,
) -> int:
    """Total sample size for stratified random sampling (Olofsson et al. 2014, eq. 13).

    ``n = (sum W_i S_i)^2 / (S(O)^2 + (1/N) sum W_i S_i^2)`` with
    ``S_i = sqrt(U_i (1 - U_i))`` (Cochran 1977). ``U_i`` are the
    anticipated user's accuracies and ``S(O)`` the target standard error of
    overall accuracy. With ``population=None`` the finite population term is
    dropped (N is effectively infinite for pixel populations).
    """
    w = _as_weights(weights)
    u = np.asarray(users_accuracy, dtype=float)
    if u.shape != w.shape:
        raise ValueError("weights and users_accuracy must have the same length")
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("anticipated user's accuracies must be strictly between 0 and 1")
    if not target_se > 0:
        raise ValueError("target standard error must be > 0")
    s = np.sqrt(u * (1.0 - u))
    num = float(np.sum(w * s)) ** 2
    den = target_se ** 2
    if population:
        den += float(np.sum(w * s ** 2)) / float(population)
    return int(math.ceil(num / den))


def allocate_sample(
    n_total: int,
    weights: Sequence[float],
    min_per_class: int = 50,
    available: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Allocate ``n_total`` points to strata: proportional to area, with a floor.

    Classes whose proportional share is below ``min_per_class`` receive the
    floor; the remaining sample is shared proportionally among the other
    classes (iterated until stable), which is the compromise allocation
    recommended by Olofsson et al. (2014). If ``n_total`` is
    smaller than ``k * min_per_class`` every class gets the floor and the
    total grows. ``available`` caps each class at its pixel count.
    """
    w = _as_weights(weights)
    k = w.size
    n_total = int(n_total)
    min_per_class = int(min_per_class)
    if n_total < 1:
        raise ValueError("total sample size must be >= 1")
    if min_per_class < 0:
        raise ValueError("minimum per class must be >= 0")

    cap = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    if available is not None:
        cap = np.asarray(available, dtype=np.int64)
        if cap.shape != w.shape:
            raise ValueError("available must have one entry per class")
    floor = np.minimum(np.full(k, min_per_class, dtype=np.int64), cap)

    if n_total <= floor.sum():
        return floor.copy()

    fixed = np.zeros(k, dtype=bool)
    alloc = np.zeros(k, dtype=np.int64)
    for _ in range(k + 1):
        free = ~fixed
        remaining = n_total - alloc[fixed].sum()
        share = w[free] / w[free].sum() if w[free].sum() > 0 else np.full(free.sum(), 1.0 / free.sum())
        raw = share * remaining
        low = raw < floor[free]
        high = raw > cap[free]
        if not low.any() and not high.any():
            alloc[free] = _round_preserving_sum(raw, int(remaining))
            break
        idx = np.flatnonzero(free)
        alloc[idx[low]] = floor[idx[low]]
        alloc[idx[high]] = cap[idx[high]]
        fixed[idx[low | high]] = True
        if fixed.all():
            break
    return np.minimum(np.maximum(alloc, floor), cap)


def _round_preserving_sum(values: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder rounding so the integers add up to ``total``."""
    base = np.floor(values).astype(np.int64)
    short = total - int(base.sum())
    if short > 0:
        order = np.argsort(-(values - base), kind="stable")
        base[order[:short]] += 1
    return base


class StratifiedPixelSelector:
    """Seedable stratified random selection of pixels, read block by block.

    Pass 1 (done by the caller) counts pixels per class. This object draws,
    for each class, ``n_h`` distinct ranks out of ``N_h`` (simple random
    sampling without replacement within the stratum). Blocks of the raster are
    then fed in row-major order with :meth:`feed`, and the selected pixels are
    returned as (row, col, class) without ever holding the whole raster in
    memory.
    """

    def __init__(self, counts: Mapping[int, int], allocation: Mapping[int, int], seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self._ranks: Dict[int, np.ndarray] = {}
        self._seen: Dict[int, int] = {}
        for cls in sorted(allocation):
            n_h = int(allocation[cls])
            N_h = int(counts.get(cls, 0))
            if n_h > N_h:
                raise ValueError(f"class {cls}: requested {n_h} points but only {N_h} pixels exist")
            ranks = np.sort(self.rng.choice(N_h, size=n_h, replace=False)) if n_h else np.empty(0, np.int64)
            self._ranks[cls] = ranks
            self._seen[cls] = 0

    def feed(self, block: np.ndarray, row_offset: int = 0, valid: Optional[np.ndarray] = None
             ) -> List[Tuple[int, int, int]]:
        """Return selected (row, col, class) cells inside this block.

        ``valid`` is an optional boolean mask (False = NoData) of the block's shape.
        """
        out: List[Tuple[int, int, int]] = []
        flat = block.ravel()
        ok = None if valid is None else np.asarray(valid, bool).ravel()
        ncols = block.shape[1]
        for cls, ranks in self._ranks.items():
            if ranks.size == 0:
                continue
            hit = flat == cls
            if ok is not None:
                hit &= ok
            pos = np.flatnonzero(hit)
            start = self._seen[cls]
            stop = start + pos.size
            lo, hi = np.searchsorted(ranks, [start, stop])
            for r in ranks[lo:hi]:
                p = int(pos[r - start])
                out.append((row_offset + p // ncols, p % ncols, int(cls)))
            self._seen[cls] = stop
        out.sort()
        return out


def cell_index(x, y, xmin: float, ymax: float, cell_w: float, cell_h: float):
    """Row/column of the raster cell containing map coordinate (x, y)."""
    col = np.floor((np.asarray(x, float) - xmin) / cell_w).astype(np.int64)
    row = np.floor((ymax - np.asarray(y, float)) / cell_h).astype(np.int64)
    return row, col


# ---------------------------------------------------------------------------
# 2. Error matrix and area-adjusted accuracy
# ---------------------------------------------------------------------------

def error_matrix(map_labels: Iterable, ref_labels: Iterable, classes: Optional[Sequence] = None):
    """Sample-count error matrix. Rows = map class, columns = reference class."""
    m = list(map_labels)
    r = list(ref_labels)
    if len(m) != len(r):
        raise ValueError("map and reference label lists differ in length")
    if classes is None:
        classes = sorted(set(m) | set(r))
    classes = list(classes)
    index = {c: i for i, c in enumerate(classes)}
    n = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for a, b in zip(m, r):
        if a not in index or b not in index:
            raise ValueError(f"label pair ({a}, {b}) not among classes {classes}")
        n[index[a], index[b]] += 1
    return classes, n


@dataclass
class AccuracyResult:
    """Outputs of :func:`area_adjusted_accuracy` (all proportions in 0-1)."""

    classes: List
    counts: np.ndarray            # n_ij sample counts (map rows x reference cols)
    weights: np.ndarray           # W_i mapped area proportions
    mapped_area: np.ndarray       # mapped area per class (input units)
    total_area: float
    proportions: np.ndarray       # p_ij estimated area proportions
    overall: float
    overall_se: float
    users: np.ndarray
    users_se: np.ndarray
    producers: np.ndarray
    producers_se: np.ndarray
    area_prop: np.ndarray         # p_.j
    area_prop_se: np.ndarray
    confidence: float
    z: float
    users_interval: str = "wald"
    users_low: Optional[np.ndarray] = None          # user's accuracy bounds (method above)
    users_high: Optional[np.ndarray] = None
    users_wilson_low: Optional[np.ndarray] = None   # Wilson bounds, always computed for reference
    users_wilson_high: Optional[np.ndarray] = None
    labels: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def area(self) -> np.ndarray:
        return self.area_prop * self.total_area

    @property
    def area_ci(self) -> np.ndarray:
        return self.z * self.area_prop_se * self.total_area

    @property
    def overall_ci(self) -> float:
        return self.z * self.overall_se

    @property
    def users_ci(self) -> np.ndarray:
        return self.z * self.users_se

    @property
    def producers_ci(self) -> np.ndarray:
        return self.z * self.producers_se

    @property
    def naive_overall(self) -> float:
        """Unweighted sample overall accuracy (what a plain confusion matrix gives)."""
        return float(np.trace(self.counts) / self.counts.sum())

    def kappa(self) -> float:
        """Cohen's kappa of the raw sample counts, shown only for comparison."""
        n = self.counts.astype(float)
        tot = n.sum()
        po = np.trace(n) / tot
        pe = float(np.sum(n.sum(1) * n.sum(0))) / tot ** 2
        return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def area_adjusted_accuracy(
    counts: np.ndarray,
    mapped_area: Sequence[float],
    confidence: float = 0.95,
    classes: Optional[Sequence] = None,
    users_interval: str = "wald",
    labels: Optional[Mapping] = None,
) -> AccuracyResult:
    """Stratified estimators of accuracy and area (Olofsson et al. 2013, 2014).

    Parameters
    ----------
    counts : (k, k) array
        Sample counts, rows = map class (stratum), columns = reference class.
    mapped_area : length-k sequence
        Mapped area of each class (pixel counts or pixel counts x cell area).
    confidence : float
        Confidence level for intervals (0.95 -> +/- 1.96 SE).
    users_interval : {"wald", "wilson"}
        Interval for user's accuracy. "wald" (default) is U +/- z SE as in
        Olofsson et al. (2014); "wilson" is the Wilson score interval within the
        map stratum, which behaves better when U is near 0 or 1 or n is small.
        Overall accuracy, producer's accuracy and area always use Wald intervals.
    labels : mapping, optional
        Class names used in warning messages.

    Formulas (Olofsson et al. 2013, 2014)
    -------------------------------------
    p_ij  = W_i n_ij / n_i.
    O     = sum_j p_jj                    V(O)  = sum_i W_i^2 U_i (1-U_i) / (n_i.-1)
    U_i   = p_ii / p_i.                   V(U_i)= U_i (1-U_i) / (n_i.-1)
    P_j   = p_jj / p_.j                   V(P_j)= see below
    p_.k  = sum_i p_ik                    S(p_.k)= sqrt(sum_i (W_i p_ik - p_ik^2)/(n_i.-1))
    A_k   = A_tot p_.k                    CI    = A_tot * z * S(p_.k)

    V(P_j) = 1/N_.j^2 [ N_j.^2 (1-P_j)^2 U_j (1-U_j)/(n_j.-1)
                         + P_j^2 sum_{i!=j} N_i.^2 (n_ij/n_i.)(1-n_ij/n_i.)/(n_i.-1) ]
    with N_i. the mapped size of stratum i and N_.j = sum_i N_i. n_ij / n_i.
    """
    n = np.asarray(counts, dtype=float)
    if n.ndim != 2 or n.shape[0] != n.shape[1]:
        raise ValueError("counts must be a square matrix")
    k = n.shape[0]
    a = np.asarray(mapped_area, dtype=float)
    if a.shape != (k,):
        raise ValueError("mapped_area must have one value per class")
    w = _as_weights(a)
    classes = list(classes) if classes is not None else list(range(1, k + 1))
    z = z_value(confidence)
    users_interval = str(users_interval).lower()
    if users_interval not in ("wald", "wilson"):
        raise ValueError("users_interval must be 'wald' or 'wilson'")
    labels = dict(labels or {})
    warn: List[str] = []

    def nm(c):
        lab = labels.get(c)
        return f"{lab} ({c})" if lab not in (None, "", str(c)) else str(c)

    n_i = n.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n_i[:, None] > 0, n / n_i[:, None], 0.0)    # n_ij / n_i.
        denom = np.where(n_i > 1, n_i - 1.0, np.nan)               # n_i. - 1

    for i, c in enumerate(classes):
        if w[i] > 0 and n_i[i] == 0:
            warn.append(f"Map class {nm(c)} has mapped area but no sample points; its area "
                        "contribution cannot be estimated (treated as 0).")
        elif w[i] > 0 and n_i[i] == 1:
            warn.append(f"Map class {nm(c)} has only one sample point; variances need n >= 2.")
        if w[i] == 0 and n_i[i] > 0:
            warn.append(f"Map class {nm(c)} has sample points but zero mapped area; "
                        "these points carry no weight.")

    w_lo, w_hi = wilson_interval(np.diag(n), n_i, confidence)
    pct = int(round(confidence * 100))
    for i, c in enumerate(classes):
        if n_i[i] > 1 and frac[i, i] in (0.0, 1.0):
            what = "agree" if frac[i, i] == 1 else "disagree"
            msg = (f"Map class {nm(c)}: all {int(n_i[i])} samples {what} with the reference. Read this as "
                   "'no errors observed', not 'no uncertainty'.")
            if users_interval == "wald":
                msg += (" The Wald interval for its user's accuracy (and its share of the area intervals) has "
                        f"zero width; the Wilson {pct}% interval is {w_lo[i]:.3f} to {w_hi[i]:.3f}.")
            else:
                msg += (f" Its Wilson {pct}% interval is {w_lo[i]:.3f} to {w_hi[i]:.3f}; its share of the "
                        "area intervals (Wald) still has zero width.")
            warn.append(msg)

    p = w[:, None] * frac
    p_row = p.sum(axis=1)
    p_col = p.sum(axis=0)
    diag = np.diag(p)

    with np.errstate(divide="ignore", invalid="ignore"):
        users = np.where(n_i > 0, np.diag(frac), np.nan)
        producers = np.where(p_col > 0, diag / p_col, np.nan)
        users_var = users * (1 - users) / denom

        overall = float(diag.sum())
        ov_terms = np.where(w > 0, w ** 2 * users * (1 - users) / denom, 0.0)
        overall_se = float(np.sqrt(np.nansum(ov_terms)))

        # area proportion SE
        terms = np.where((w > 0)[:, None], (w[:, None] * p - p ** 2) / denom[:, None], 0.0)
        area_se = np.sqrt(np.nansum(terms, axis=0))

        # producer's accuracy SE (stratified ratio estimator, Olofsson et al. 2014)
        N_i = a                              # mapped size of each stratum
        N_col = (N_i[:, None] * frac).sum(axis=0)
        prod_var = np.full(k, np.nan)
        for j in range(k):
            if N_col[j] <= 0 or not np.isfinite(producers[j]):
                continue
            P, U = producers[j], users[j]
            t1 = 0.0
            if n_i[j] > 1:
                t1 = N_i[j] ** 2 * (1 - P) ** 2 * U * (1 - U) / (n_i[j] - 1)
            t2 = 0.0
            for i in range(k):
                if i == j or n_i[i] < 2:
                    continue
                t2 += N_i[i] ** 2 * frac[i, j] * (1 - frac[i, j]) / (n_i[i] - 1)
            prod_var[j] = (t1 + P ** 2 * t2) / N_col[j] ** 2

    return AccuracyResult(
        classes=classes,
        counts=n.astype(np.int64),
        weights=w,
        mapped_area=a,
        total_area=float(a.sum()),
        proportions=p,
        overall=overall,
        overall_se=overall_se,
        users=users,
        users_se=np.sqrt(users_var),
        users_interval=users_interval,
        users_low=(w_lo if users_interval == "wilson" else users - z * np.sqrt(users_var)),
        users_high=(w_hi if users_interval == "wilson" else users + z * np.sqrt(users_var)),
        users_wilson_low=w_lo,
        users_wilson_high=w_hi,
        labels=labels,
        producers=producers,
        producers_se=np.sqrt(prod_var),
        area_prop=p_col,
        area_prop_se=area_se,
        confidence=confidence,
        z=z,
        warnings=warn,
    )


# ---------------------------------------------------------------------------
# 3. Spatial blocks and folds
# ---------------------------------------------------------------------------

def block_index(x, y, size: float, origin: Optional[Tuple[float, float]] = None):
    """Square-block row/col for each point. Default origin snaps the grid to
    multiples of ``size`` so the same points always fall in the same blocks."""
    if not size > 0:
        raise ValueError("block size must be > 0")
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if origin is None:
        origin = (math.floor(x.min() / size) * size, math.floor(y.min() / size) * size)
    col = np.floor((x - origin[0]) / size).astype(np.int64)
    row = np.floor((y - origin[1]) / size).astype(np.int64)
    return row, col, origin


def assign_folds(block_ids: Sequence[int], n_folds: int, seed: Optional[int] = None) -> Dict[int, int]:
    """Assign whole blocks to folds so every fold gets a similar point count.

    Blocks are shuffled (seeded), then taken largest-first and given to the
    fold that currently holds the fewest points (greedy balancing). All points
    of a block share one fold, which keeps spatially autocorrelated neighbours
    out of each other's test sets (Roberts et al. 2017).
    """
    if n_folds < 2:
        raise ValueError("need at least 2 folds")
    ids, sizes = np.unique(np.asarray(block_ids), return_counts=True)
    if ids.size < n_folds:
        raise ValueError(f"only {ids.size} occupied blocks for {n_folds} folds; use smaller blocks or fewer folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(ids.size)
    order = order[np.argsort(-sizes[order], kind="stable")]
    load = np.zeros(n_folds, dtype=np.int64)
    out: Dict[int, int] = {}
    for idx in order:
        f = int(np.argmin(load))
        out[int(ids[idx])] = f + 1
        load[f] += sizes[idx]
    return out


def buffer_split(x, y, block_row, block_col, fold, size: float, origin: Tuple[float, float],
                 buffer: float, n_folds: int) -> np.ndarray:
    """Per-fold roles for k-fold block cross-validation with a buffer.

    Returns an (n_points, n_folds) int8 array. For fold f (column f-1) each point
    is 1 = test (its block is in fold f), 0 = train, or 2 = excluded from
    training because it lies within ``buffer`` of a fold-f block (distance from
    the point to the nearest edge of any occupied test block). This removes
    training points that sit just across a block edge from test points, as in
    buffered block cross-validation (Roberts et al. 2017).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    br = np.asarray(block_row, np.int64)
    bc = np.asarray(block_col, np.int64)
    fold = np.asarray(fold, np.int64)
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    if not size > 0:
        raise ValueError("block size must be > 0")
    roles = np.zeros((x.size, n_folds), dtype=np.int8)
    roles[np.arange(x.size), fold - 1] = 1
    if buffer == 0 or x.size == 0:
        return roles
    block_fold = {}
    for r, c, f in zip(br.tolist(), bc.tolist(), fold.tolist()):
        block_fold[(r, c)] = f
    reach = int(math.ceil(buffer / size))
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = br + dr, bc + dc
            x0 = origin[0] + nc * size
            y0 = origin[1] + nr * size
            dx = np.maximum(np.maximum(x0 - x, 0.0), x - (x0 + size))
            dy = np.maximum(np.maximum(y0 - y, 0.0), y - (y0 + size))
            near = np.flatnonzero(np.hypot(dx, dy) <= buffer)
            for i in near:
                f = block_fold.get((int(nr[i]), int(nc[i])))
                if f is not None and roles[i, f - 1] == 0:
                    roles[i, f - 1] = 2
    return roles


# ---------------------------------------------------------------------------
# Report rendering (Markdown / HTML from one structure)
# ---------------------------------------------------------------------------

def _f(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _area(v):
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


def build_report(
    res: AccuracyResult,
    labels: Optional[Mapping] = None,
    area_unit: str = "ha",
    area_factor: float = 1.0,
    title: str = "Area-adjusted accuracy assessment",
    source: str = "",
) -> List[Tuple[str, object]]:
    """Return report blocks: ("h1"|"h2"|"p"|"table"|"list", content)."""
    labels = dict(labels or {})
    names = [str(labels.get(c, c)) for c in res.classes]
    pct = int(round(res.confidence * 100))
    blocks: List[Tuple[str, object]] = [("h1", title)]
    if source:
        blocks.append(("p", source))
    n_tot = int(res.counts.sum())
    blocks.append(("p", (
        f"Stratified estimators of Olofsson et al. (2013, 2014). {n_tot} reference samples, "
        f"{len(res.classes)} classes, total mapped area {_area(res.total_area * area_factor)} {area_unit}. "
        f"Intervals are {pct}% confidence intervals (estimate +/- {res.z:.2f} SE"
        + ("; user's accuracy uses the Wilson score interval)." if res.users_interval == "wilson" else ")."))))

    blocks.append(("h2", "Summary"))
    blocks.append(("table", (["Measure", "Value"], [
        ["Overall accuracy (area-weighted)", f"{_f(res.overall)} +/- {_f(res.overall_ci)}"],
        ["Overall accuracy (unweighted sample, as in Compute Confusion Matrix)", _f(res.naive_overall)],
        ["Kappa (sample counts, for comparison only)", _f(res.kappa())],
    ])))

    blocks.append(("h2", "Error matrix of sample counts"))
    hdr = ["Map \\ Reference"] + names + ["Total", "W_i"]
    rows = []
    for i, nm in enumerate(names):
        rows.append([nm] + [str(int(v)) for v in res.counts[i]] + [str(int(res.counts[i].sum())), _f(res.weights[i], 4)])
    rows.append(["Total"] + [str(int(v)) for v in res.counts.sum(0)] + [str(n_tot), "1.0000"])
    blocks.append(("table", (hdr, rows)))

    blocks.append(("h2", "Error matrix of estimated area proportions"))
    hdr = ["Map \\ Reference"] + names + ["Total"]
    rows = [[nm] + [_f(v, 4) for v in res.proportions[i]] + [_f(res.proportions[i].sum(), 4)]
            for i, nm in enumerate(names)]
    rows.append(["Total"] + [_f(v, 4) for v in res.area_prop] + ["1.0000"])
    blocks.append(("table", (hdr, rows)))

    blocks.append(("h2", "Accuracy by class"))
    if res.users_interval == "wilson":
        ua_hdr = f"User's accuracy (Wilson {pct}% CI)"
        ua_txt = [f"{_f(res.users[i])} ({_f(res.users_low[i])} to {_f(res.users_high[i])})"
                  for i in range(len(names))]
    else:
        ua_hdr = f"User's accuracy +/- {pct}% CI"
        ua_txt = [f"{_f(res.users[i])} +/- {_f(res.users_ci[i])}" for i in range(len(names))]
    hdr = ["Class", "n (map stratum)", ua_hdr, f"Producer's accuracy +/- {pct}% CI"]
    rows = [[nm, str(int(res.counts[i].sum())),
             ua_txt[i],
             f"{_f(res.producers[i])} +/- {_f(res.producers_ci[i])}"] for i, nm in enumerate(names)]
    blocks.append(("table", (hdr, rows)))

    blocks.append(("h2", "Area estimates"))
    hdr = ["Class", f"Mapped area ({area_unit})", f"Estimated area ({area_unit})",
           f"{pct}% CI ({area_unit})", "Estimate - mapped"]
    rows = []
    for i, nm in enumerate(names):
        mapped = res.mapped_area[i] * area_factor
        est = res.area[i] * area_factor
        ci = res.area_ci[i] * area_factor
        rows.append([nm, _area(mapped), _area(est), f"+/- {_area(ci)}", _area(est - mapped)])
    blocks.append(("table", (hdr, rows)))

    if res.warnings:
        blocks.append(("h2", "Warnings"))
        blocks.append(("list", list(res.warnings)))

    blocks.append(("h2", "Method"))
    blocks.append(("list", [
        "Rows of the error matrix are map classes (strata); p_ij = W_i n_ij / n_i.",
        "User's accuracy U_i = p_ii / p_i.; producer's accuracy P_j = p_jj / p_.j; overall accuracy O = sum p_jj.",
        "Area of class k = total area x p_.k, with SE(p_.k) = sqrt(sum_i (W_i p_ik - p_ik^2) / (n_i. - 1)).",
        "Variances follow Olofsson et al. (2013, 2014); producer's accuracy uses the variance of the stratified ratio estimator.",
        "User's accuracy intervals: Wald (U +/- z SE, as in Olofsson et al. 2014) by default. The Wilson score "
        "interval (Wilson 1927) is available because the Wald interval has zero width when U is 0 or 1 and "
        "tends to under-cover when U is near 0 or 1 with small n.",
        "A user's accuracy of exactly 0 or 1 gives a zero-width interval; the true uncertainty is not zero, so treat those intervals with caution.",
    ]))
    blocks.append(("h2", "References"))
    blocks.append(("list", [
        "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal "
        "of the American Statistical Association, 22, 209-212.",
        "Olofsson, P., Foody, G. M., Stehman, S. V., & Woodcock, C. E. (2013). Making better use of accuracy data "
        "in land change studies: Estimating accuracy and area and quantifying uncertainty using stratified "
        "estimation. Remote Sensing of Environment, 129, 122-131.",
        "Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). Good "
        "practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment, "
        "148, 42-57.",
    ]))
    return blocks


def render_markdown(blocks) -> str:
    out: List[str] = []
    for kind, content in blocks:
        if kind == "h1":
            out.append(f"# {content}\n")
        elif kind == "h2":
            out.append(f"## {content}\n")
        elif kind == "p":
            out.append(f"{content}\n")
        elif kind == "list":
            out.extend(f"- {item}" for item in content)
            out.append("")
        elif kind == "table":
            hdr, rows = content
            out.append("| " + " | ".join(hdr) + " |")
            out.append("|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|")
            out.extend("| " + " | ".join(r) + " |" for r in rows)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1f2933;background:#fff}
h1{font-size:1.6rem;border-bottom:3px solid #2b6cb0;padding-bottom:.3rem}
h2{font-size:1.15rem;margin-top:1.8rem;color:#2b6cb0}
table{border-collapse:collapse;margin:.6rem 0;font-size:.92rem}
th,td{border:1px solid #cbd2d9;padding:.35rem .6rem;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#edf2f7}
tr:last-child td{font-weight:600}
li{margin:.2rem 0}
"""


def render_html(blocks) -> str:
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"]
    title = next((c for k, c in blocks if k == "h1"), "Accuracy report")
    out.append(f"<title>{html.escape(str(title))}</title><style>{_CSS}</style></head><body>")
    for kind, content in blocks:
        if kind in ("h1", "h2"):
            out.append(f"<{kind}>{html.escape(str(content))}</{kind}>")
        elif kind == "p":
            out.append(f"<p>{html.escape(str(content))}</p>")
        elif kind == "list":
            out.append("<ul>" + "".join(f"<li>{html.escape(str(i))}</li>" for i in content) + "</ul>")
        elif kind == "table":
            hdr, rows = content
            out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in hdr) + "</tr></thead><tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
    out.append("</body></html>")
    return "\n".join(out)

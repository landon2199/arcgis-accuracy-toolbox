"""Unit tests for accuracy_core (no ArcPy needed).

Run with:  python -m pytest tests
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import accuracy_core as ac  # noqa: E402

# ---------------------------------------------------------------------------
# Olofsson et al. (2014), Section 5 worked example (Table 8).
# Rows = map class, columns = reference class.
# Classes: deforestation, forest gain, stable forest, stable non-forest.
# Mapped areas: 200,000 / 150,000 / 3,200,000 / 6,450,000 Landsat pixels (30 m).
# ---------------------------------------------------------------------------
OLOFSSON_COUNTS = np.array([
    [66, 0, 5, 4],
    [0, 55, 8, 12],
    [1, 0, 153, 11],
    [2, 1, 9, 313],
])
OLOFSSON_PIXELS = np.array([200_000, 150_000, 3_200_000, 6_450_000])
HA_PER_PIXEL = 30 * 30 / 10_000


@pytest.fixture(scope="module")
def olofsson():
    return ac.area_adjusted_accuracy(OLOFSSON_COUNTS, OLOFSSON_PIXELS, confidence=0.95)


def test_olofsson_weights(olofsson):
    assert np.allclose(olofsson.weights, [0.020, 0.015, 0.320, 0.645])


def test_olofsson_area_estimates_ha(olofsson):
    # Published (Olofsson et al. 2014, p. 54): 21,158 +/- 6,158; 11,686 +/- 3,756;
    # 285,770 +/- 15,510; 581,386 +/- 16,282 ha (95% CI).
    total_ha = OLOFSSON_PIXELS.sum() * HA_PER_PIXEL
    area = olofsson.area_prop * total_ha
    ci = olofsson.area_ci / olofsson.total_area * total_ha
    assert np.allclose(np.round(area), [21_158, 11_686, 285_770, 581_386], atol=1)
    assert np.allclose(np.round(ci), [6_158, 3_756, 15_510, 16_282], atol=1)


def test_olofsson_accuracies(olofsson):
    # Published: UA 0.88, 0.73, 0.93, 0.96; PA 0.75, 0.85, 0.93, 0.96; OA 0.95 (2 d.p.)
    assert np.allclose(np.round(olofsson.users, 2), [0.88, 0.73, 0.93, 0.96])
    assert np.allclose(np.round(olofsson.producers, 2), [0.75, 0.85, 0.93, 0.96])
    assert round(olofsson.overall, 2) == 0.95
    # UA CIs 0.07, 0.10, 0.04, 0.02; OA CI 0.02
    assert np.allclose(np.round(olofsson.users_ci, 2), [0.07, 0.10, 0.04, 0.02])
    assert round(olofsson.overall_ci, 2) == 0.02
    # PA CIs for deforestation, stable forest, stable non-forest: 0.21, 0.03, 0.02
    assert np.allclose(np.round(olofsson.producers_ci[[0, 2, 3]], 2), [0.21, 0.03, 0.02])


def test_olofsson_exact_values(olofsson):
    """Full-precision regression values (hand-checkable from the formulas)."""
    p_def = 0.02 * 66 / 75 + 0.32 * 1 / 165 + 0.645 * 2 / 325
    assert olofsson.area_prop[0] == pytest.approx(p_def, rel=1e-12)
    assert olofsson.overall == pytest.approx(0.946512, abs=1e-6)
    assert olofsson.producers[0] == pytest.approx(0.0176 / p_def, rel=1e-12)


def test_olofsson_proportion_matrix_sums(olofsson):
    p = olofsson.proportions
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p.sum(axis=1), olofsson.weights)


def test_sample_size_eq13_matches_paper():
    # Olofsson et al. (2014) worked example: U = 0.70, 0.60, 0.90, 0.95, S(O) = 0.01 -> n = 641
    assert ac.sample_size_overall(OLOFSSON_PIXELS, [0.70, 0.60, 0.90, 0.95], 0.01) == 641
    # finite-population correction can only lower n
    assert ac.sample_size_overall(OLOFSSON_PIXELS, [0.70, 0.60, 0.90, 0.95], 0.01,
                                  population=OLOFSSON_PIXELS.sum()) <= 641


def test_sample_size_validation():
    with pytest.raises(ValueError):
        ac.sample_size_overall([1, 1], [0.8, 1.0], 0.01)
    with pytest.raises(ValueError):
        ac.sample_size_overall([1, 1], [0.8, 0.9], 0)


def test_allocation_floor_and_total():
    a = ac.allocate_sample(641, OLOFSSON_PIXELS, 50)
    assert a.sum() == 641
    assert a[0] == 50 and a[1] == 50
    # remaining 541 split proportional to 0.320 : 0.645
    assert abs(a[2] - 541 * 0.32 / 0.965) <= 1
    a = ac.allocate_sample(641, OLOFSSON_PIXELS, 100)
    assert list(a) == [100, 100, 146, 295]


def test_allocation_small_total_and_caps():
    a = ac.allocate_sample(100, [1, 1, 1, 97], 50)
    assert list(a) == [50, 50, 50, 50]          # floor wins, total grows
    a = ac.allocate_sample(300, [10, 1000, 1000], 50, available=[10, 1000, 1000])
    assert a[0] == 10 and a.sum() == 300        # capped by available pixels


def test_error_matrix_orientation():
    classes, n = ac.error_matrix([1, 1, 2, 3], [1, 2, 2, 3])
    assert classes == [1, 2, 3]
    assert n[0, 1] == 1   # map 1, reference 2
    assert np.trace(n) == 3


def test_class_without_samples_warns():
    n = np.array([[10, 0], [0, 0]])
    res = ac.area_adjusted_accuracy(n, [50, 50])
    assert any("no sample points" in w for w in res.warnings)


def test_standard_errors_monte_carlo():
    """Analytic SEs should match the spread of estimates over repeated samples,
    and 95% intervals for area should cover the true area ~95% of the time.

    (User's accuracy near 1 with n ~ 150 uses the Wald interval, which is known
    to under-cover slightly, so coverage is only checked for area here.)
    """
    rng = np.random.default_rng(7)
    truth = rng.choice(3, size=200_000, p=[0.15, 0.30, 0.55])
    mapped = truth.copy()
    flip = rng.random(truth.size)
    mapped[(truth == 0) & (flip < 0.20)] = 1
    mapped[(truth == 1) & (flip < 0.10)] = 2
    mapped[(truth == 2) & (flip < 0.05)] = 1
    mapped[(truth == 1) & (flip > 0.95)] = 0
    mapped[(truth == 2) & (flip > 0.97)] = 0
    N = np.bincount(mapped, minlength=3)
    true_area = np.bincount(truth, minlength=3) / truth.size
    idx = [np.flatnonzero(mapped == i) for i in range(3)]
    alloc = [100, 150, 150]
    reps = 1500
    est, se = [], []
    cover_area = np.zeros(3)
    for _ in range(reps):
        m, r = [], []
        for i in range(3):
            s = rng.choice(idx[i], alloc[i], replace=False)
            m.extend(mapped[s])
            r.extend(truth[s])
        _, n = ac.error_matrix(m, r, [0, 1, 2])
        res = ac.area_adjusted_accuracy(n, N)
        cover_area += np.abs(res.area_prop - true_area) <= res.z * res.area_prop_se
        est.append(np.r_[res.area_prop, res.users, res.producers, res.overall])
        se.append(np.r_[res.area_prop_se, res.users_se, res.producers_se, res.overall_se])
    ratio = np.mean(se, axis=0) / np.std(est, axis=0)
    assert np.all((ratio > 0.9) & (ratio < 1.1)), ratio
    cov = cover_area / reps
    assert np.all((cov > 0.92) & (cov < 0.98)), cov
    # stratified area estimator is unbiased
    assert np.allclose(np.mean(est, axis=0)[:3], true_area, atol=0.002)


def test_pixel_selector_blocks_match_whole_array():
    rng = np.random.default_rng(1)
    arr = rng.choice([1, 2, 3, 255], size=(97, 61), p=[0.5, 0.3, 0.15, 0.05])
    counts = {c: int((arr == c).sum()) for c in (1, 2, 3)}
    alloc = {1: 20, 2: 15, 3: 10}
    whole = ac.StratifiedPixelSelector(counts, alloc, seed=3).feed(arr, 0, valid=arr != 255)
    sel = ac.StratifiedPixelSelector(counts, alloc, seed=3)
    blocked = []
    for r0 in range(0, arr.shape[0], 10):
        blocked += sel.feed(arr[r0:r0 + 10], r0, valid=arr[r0:r0 + 10] != 255)
    assert sorted(whole) == sorted(blocked)
    assert len(set(whole)) == 45
    for row, col, cls in whole:
        assert arr[row, col] == cls
    for c, k in alloc.items():
        assert sum(1 for _, _, v in whole if v == c) == k
    other = ac.StratifiedPixelSelector(counts, alloc, seed=4).feed(arr, 0, valid=arr != 255)
    assert sorted(other) != sorted(whole)


def test_pixel_selector_too_many():
    with pytest.raises(ValueError):
        ac.StratifiedPixelSelector({1: 5}, {1: 6}, seed=0)


def test_cell_index():
    r, c = ac.cell_index([0.5, 29.9, 30.0], [99.0, 70.1, 70.0], 0.0, 100.0, 30.0, 30.0)
    assert list(r) == [0, 0, 1]
    assert list(c) == [0, 0, 1]


def test_blocks_and_folds():
    rng = np.random.default_rng(0)
    x = rng.uniform(1000, 11000, 2000)
    y = rng.uniform(5000, 9000, 2000)
    row, col, origin = ac.block_index(x, y, 1000)
    assert origin == (1000.0, 5000.0)
    assert row.max() <= 4 and col.max() <= 10
    bid = row * 100 + col
    folds = ac.assign_folds(bid, 5, seed=1)
    assert set(folds.values()) == {1, 2, 3, 4, 5}
    per_fold = np.bincount([folds[b] for b in bid])[1:]
    assert per_fold.max() - per_fold.min() <= 0.1 * per_fold.mean()
    assert folds == ac.assign_folds(bid, 5, seed=1)       # reproducible
    with pytest.raises(ValueError):
        ac.assign_folds([1, 1, 2], 3)


def test_report_rendering(olofsson):
    labels = {1: "Deforestation", 2: "Forest gain", 3: "Stable forest", 4: "Stable non-forest"}
    blocks = ac.build_report(olofsson, labels, area_factor=HA_PER_PIXEL)
    md = ac.render_markdown(blocks)
    assert "| Deforestation |" in md and "21,158" in md and "+/- 6,158" in md
    html = ac.render_html(blocks)
    assert html.startswith("<!doctype html>") and "<table>" in html and "Forest gain" in html


def test_z_value():
    assert ac.z_value(0.95) == pytest.approx(1.959964, abs=1e-6)
    with pytest.raises(ValueError):
        ac.z_value(1.5)


def test_wilson_interval_known_values():
    # 0 of 10 at 95%: Wilson upper bound = z^2 / (n + z^2) = 0.2775
    lo, hi = ac.wilson_interval([0, 10, 5], [10, 10, 10], 0.95)
    assert lo[0] == pytest.approx(0.0) and hi[0] == pytest.approx(0.27753, abs=1e-4)
    assert lo[1] == pytest.approx(0.72247, abs=1e-4) and hi[1] == pytest.approx(1.0)
    assert lo[2] == pytest.approx(0.23659, abs=1e-4) and hi[2] == pytest.approx(0.76341, abs=1e-4)


def test_users_interval_option_and_labelled_warnings():
    n = np.array([[100, 0], [5, 95]])
    labels = {1: "Water", 2: "Forest"}
    wald = ac.area_adjusted_accuracy(n, [10, 90], classes=[1, 2], labels=labels)
    assert wald.users_low[0] == pytest.approx(1.0) and wald.users_high[0] == pytest.approx(1.0)
    msg = " ".join(wald.warnings)
    assert "Water (1)" in msg and "Wilson 95% interval is 0.963 to 1.000" in msg
    wil = ac.area_adjusted_accuracy(n, [10, 90], classes=[1, 2], users_interval="wilson", labels=labels)
    assert wil.users_low[0] == pytest.approx(0.96301, abs=1e-4) and wil.users_high[0] == 1.0
    # point estimates and area results do not depend on the interval choice
    assert np.allclose(wil.users, wald.users) and np.allclose(wil.area_prop_se, wald.area_prop_se)
    md = ac.render_markdown(ac.build_report(wil, labels))
    assert "Wilson 95% CI" in md and "(0.963 to 1.000)" in md
    with pytest.raises(ValueError):
        ac.area_adjusted_accuracy(n, [10, 90], users_interval="exact")


def test_buffer_split_roles():
    # 1 km blocks in a row: block columns 0, 1, 2 with folds 1, 2, 1
    x = np.array([500.0, 1050.0, 1500.0, 1950.0, 2500.0])
    y = np.full(5, 500.0)
    row, col, origin = ac.block_index(x, y, 1000.0, origin=(0.0, 0.0))
    fold = np.array([1, 2, 2, 2, 1])
    roles = ac.buffer_split(x, y, row, col, fold, 1000.0, origin, buffer=100.0, n_folds=2)
    # fold 1 is test: points 0 and 4. Points 1 (50 m from block 0) and 3 (50 m from block 2) are excluded.
    assert roles[:, 0].tolist() == [1, 2, 0, 2, 1]
    # fold 2 is test: points 1-3. Point 0 is 500 m away and point 4 is 500 m away -> both train.
    assert roles[:, 1].tolist() == [0, 1, 1, 1, 0]
    # no buffer: plain block split
    plain = ac.buffer_split(x, y, row, col, fold, 1000.0, origin, buffer=0.0, n_folds=2)
    assert set(plain[:, 0].tolist()) == {0, 1}


def test_buffer_split_brute_force():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 5000, 400)
    y = rng.uniform(0, 5000, 400)
    size, buf = 1000.0, 300.0
    row, col, origin = ac.block_index(x, y, size)
    bid = row * 100 + col
    fmap = ac.assign_folds(bid, 4, seed=0)
    fold = np.array([fmap[b] for b in bid])
    roles = ac.buffer_split(x, y, row, col, fold, size, origin, buf, 4)
    for f in range(1, 5):
        test_blocks = {(r, c) for r, c, ff in zip(row, col, fold) if ff == f}
        for i in range(x.size):
            if fold[i] == f:
                assert roles[i, f - 1] == 1
                continue
            d = min(np.hypot(max(origin[0] + c * size - x[i], 0, x[i] - origin[0] - (c + 1) * size),
                             max(origin[1] + r * size - y[i], 0, y[i] - origin[1] - (r + 1) * size))
                    for r, c in test_blocks)
            assert roles[i, f - 1] == (2 if d <= buf else 0)

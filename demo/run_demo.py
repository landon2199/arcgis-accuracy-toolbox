"""End-to-end demo of the Accuracy Assessment toolbox on synthetic data.

Run with the ArcGIS Pro Python environment:

    "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe" demo\\run_demo.py

What it does
1. Builds a synthetic 30 m "true" land cover raster (5 classes) and a "map" made from
   it with known per-pixel confusion rates, so the true class areas are known.
2. Runs Stratified Random Sample Design on the map.
3. Labels each sample point from the true raster (a perfect interpreter).
4. Runs Area-Adjusted Accuracy -> docs/example_report.md.
5. Runs Spatial Block Assignment (2 km blocks, 5 folds).
6. Draws docs/hero.png: true area vs. pixel counting vs. the stratified estimate.

All outputs except the docs/ files go to scratch/ (git-ignored).
"""

import json
import os
import shutil
import sys

import arcpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import accuracy_core as core  # noqa: E402

SCRATCH = os.path.join(ROOT, "scratch")
DOCS = os.path.join(ROOT, "docs")
GDB = os.path.join(SCRATCH, "demo.gdb")
PYT = os.path.join(ROOT, "AccuracyAssessment.pyt")

SEED = 2026
NROWS, NCOLS, CELL = 800, 1000, 30.0          # 24 km x 30 km at 30 m
X0, Y0 = 500_000.0, 1_200_000.0                 # lower-left, CONUS Albers metres
SR = arcpy.SpatialReference(5070)               # NAD83 / Conus Albers (equal area)
LABELS = {1: "Water", 2: "Forest", 3: "Grass/shrub", 4: "Cropland", 5: "Developed"}

# Map errors: true class -> {mapped-as class: probability}
CONFUSION = {
    1: {2: 0.02},
    2: {3: 0.12, 4: 0.02},
    3: {2: 0.06, 4: 0.15, 5: 0.02},
    4: {3: 0.08, 5: 0.02},
    5: {3: 0.10, 4: 0.04},
}


def smooth_noise(rng, shape, scale):
    """Spatially correlated noise via a Gaussian filter in the Fourier domain."""
    white = rng.standard_normal(shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    kernel = np.exp(-2 * (np.pi * scale) ** 2 * (fx ** 2 + fy ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(white) * kernel))
    return (out - out.mean()) / out.std()


def make_rasters():
    rng = np.random.default_rng(SEED)
    elev = smooth_noise(rng, (NROWS, NCOLS), 40)
    moist = smooth_noise(rng, (NROWS, NCOLS), 25)
    urban = smooth_noise(rng, (NROWS, NCOLS), 12)
    truth = np.full((NROWS, NCOLS), 3, dtype=np.uint8)          # grass/shrub
    truth[moist > 0.35] = 2                                      # forest
    truth[(elev < -0.2) & (moist < 0.35) & (moist > -1.2)] = 4  # cropland on low flats
    truth[urban > 1.6] = 5                                       # developed
    truth[elev < -1.75] = 1                                      # water in the lowest spots

    mapped = truth.copy()
    u = rng.random(truth.shape)
    for t, dest in CONFUSION.items():
        lo = 0.0
        for m, prob in dest.items():
            sel = (truth == t) & (u >= lo) & (u < lo + prob)
            mapped[sel] = m
            lo += prob
    # NoData collar on the right edge (e.g. outside the study area)
    truth[:, -40:] = 0
    mapped[:, -40:] = 0
    return truth, mapped


def save_raster(arr, path):
    if arcpy.Exists(path):
        arcpy.management.Delete(path)
    ras = arcpy.NumPyArrayToRaster(arr, arcpy.Point(X0, Y0), CELL, CELL, value_to_nodata=0)
    ras.save(path)
    arcpy.management.DefineProjection(path, SR)
    return path


def check_block_reader(map_ras, mapped):
    """The toolbox reads rasters in row blocks; check that against numpy with tiny blocks."""
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader("acc_pyt", PYT)
    spec = importlib.util.spec_from_loader("acc_pyt", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    mod.BLOCK_ROWS = 97
    info = mod._raster_info(map_ras)
    got = mod._class_counts(info)
    want = {int(c): int(n) for c, n in zip(*np.unique(mapped[mapped > 0], return_counts=True))}
    assert got == want, (got, want)
    sel = core.StratifiedPixelSelector(got, {c: 5 for c in got}, seed=1)
    picks = [p for r0, b, v in mod._iter_blocks(info) for p in sel.feed(b, r0, v)]
    assert all(mapped[r, c] == k for r, c, k in picks) and len(picks) == 5 * len(got)
    print("Block reader check passed (97-row blocks, NoData collar excluded):", got)


def main():
    arcpy.env.overwriteOutput = True
    if os.path.isdir(GDB):
        shutil.rmtree(GDB)
    os.makedirs(SCRATCH, exist_ok=True)
    arcpy.management.CreateFileGDB(SCRATCH, "demo.gdb")

    print("ArcGIS", arcpy.GetInstallInfo()["Version"], "| license", arcpy.ProductInfo())
    truth, mapped = make_rasters()
    map_ras = save_raster(mapped, os.path.join(GDB, "landcover_map"))
    truth_ras = save_raster(truth, os.path.join(GDB, "landcover_truth"))

    check_block_reader(map_ras, mapped)
    tb = arcpy.ImportToolbox(PYT)
    print("Imported toolbox tools:", [t for t in dir(arcpy) if t.endswith("_accuracy")])

    # ---- 1. sample design ---------------------------------------------------
    pts = os.path.join(GDB, "sample_points")
    r1 = arcpy.accuracy.StratifiedSampleDesign(
        in_raster=map_ras, out_points=pts, target_se=0.01, default_users_accuracy=0.80,
        class_users_accuracy=[[1, 0.95], [2, 0.85]], min_per_class=100, seed=42,
        out_design_table=os.path.join(GDB, "sample_design"))
    print(r1.getMessages())

    # ---- 2. simulate the interpreter: reference label = true class -----------
    with arcpy.da.UpdateCursor(pts, ["SHAPE@XY", "REF_CLASS"]) as cur:
        for (x, y), _ in cur:
            row, col = core.cell_index(x, y, X0, Y0 + NROWS * CELL, CELL, CELL)
            cur.updateRow([(x, y), int(truth[int(row), int(col)])])

    # ---- 3. area-adjusted accuracy -----------------------------------------
    labels = [[k, v] for k, v in LABELS.items()]
    report_md = os.path.join(DOCS, "example_report.md")
    r2 = arcpy.accuracy.AreaAdjustedAccuracy(
        in_points=pts, reference_field="REF_CLASS", map_field="MAP_CLASS", in_raster=map_ras,
        out_table=os.path.join(GDB, "accuracy"), out_report=report_md, confidence=95,
        class_labels=labels, out_matrix_table=os.path.join(GDB, "area_matrix"))
    print(r2.getMessages())
    # same run, map class read from the raster at each point, HTML report
    r2b = arcpy.accuracy.AreaAdjustedAccuracy(
        in_points=pts, reference_field="REF_CLASS", map_field=None, in_raster=map_ras,
        out_table=os.path.join(GDB, "accuracy_from_raster"),
        out_report=os.path.join(SCRATCH, "example_report.html"), confidence=95, class_labels=labels)

    acc = {r[0]: r[1:] for r in arcpy.da.SearchCursor(
        os.path.join(GDB, "accuracy"), ["CLASS", "MAP_HA", "AREA_HA", "AREA_CI_HA", "UA", "PA", "OA", "OA_CI"])}
    acc_b = {r[0]: r[1:] for r in arcpy.da.SearchCursor(
        os.path.join(GDB, "accuracy_from_raster"), ["CLASS", "MAP_HA", "AREA_HA", "AREA_CI_HA", "UA", "PA", "OA", "OA_CI"])}
    assert all(np.allclose(acc[c], acc_b[c]) for c in acc), "map-field and raster-lookup runs differ"

    # ---- 4. spatial blocks ---------------------------------------------------
    r3 = arcpy.accuracy.SpatialBlockAssignment(
        in_points=pts, out_features=os.path.join(GDB, "sample_points_blocked"), block_size="2000 Meters",
        n_folds=5, seed=42, class_field="REF_CLASS", out_blocks=os.path.join(GDB, "blocks_2km"),
        buffer_distance="250 Meters")
    print(r3.getMessages())
    folds = {}
    split = np.array([r for r in arcpy.da.SearchCursor(os.path.join(GDB, "sample_points_blocked"),
                                                        [f"SPLIT_F{f}" for f in range(1, 6)])])
    for f, in arcpy.da.SearchCursor(os.path.join(GDB, "sample_points_blocked"), ["FOLD"]):
        folds[f] = folds.get(f, 0) + 1
    buffer_excluded = {f + 1: int((split[:, f] == 2).sum()) for f in range(5)}

    # ---- truth vs estimates ---------------------------------------------------
    ha = CELL * CELL / 10_000
    true_ha = {c: float((truth == c).sum() * ha) for c in LABELS}
    summary = {"arcgis_version": arcpy.GetInstallInfo()["Version"], "n_points": int(arcpy.management.GetCount(pts)[0]),
               "overall_accuracy": acc[1][5], "overall_ci": acc[1][6], "fold_counts": folds,
               "buffer_250m_excluded": buffer_excluded, "classes": {}}
    print("\nClass         true ha   mapped ha   estimate ha (95% CI)     covered")
    for c, name in LABELS.items():
        mapped_ha, est, ci, ua, pa = acc[c][:5]
        covered = abs(est - true_ha[c]) <= ci
        summary["classes"][name] = {"true_ha": true_ha[c], "mapped_ha": mapped_ha, "estimate_ha": est,
                                    "ci_ha": ci, "users": ua, "producers": pa, "ci_covers_truth": bool(covered)}
        print(f"{name:12s} {true_ha[c]:9,.0f} {mapped_ha:11,.0f} {est:11,.0f} +/- {ci:7,.0f}   {covered}")
    print(f"OA = {acc[1][5]:.4f} +/- {acc[1][6]:.4f}; fold sizes {dict(sorted(folds.items()))}")
    print(f"250 m buffer: training points excluded per fold {buffer_excluded}")
    alloc = {r[0]: r[1] for r in arcpy.da.SearchCursor(os.path.join(GDB, "sample_design"), ["CLASS", "N_POINTS"])}
    summary["repeat"] = repeat_sampling(truth, mapped, alloc, reps=1000)
    print("Repeated sampling (1000 draws, same design): CI coverage",
          {k: v["coverage"] for k, v in summary["repeat"].items()})
    with open(os.path.join(SCRATCH, "demo_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    make_hero(summary, mapped, os.path.join(DOCS, "hero.png"))
    print("Wrote", os.path.join(DOCS, "hero.png"), "and", report_md)


def repeat_sampling(truth, mapped, alloc, reps=1000, seed=99):
    """Redraw the same stratified design many times (core only) to show the
    estimator's spread and the coverage of its 95% intervals."""
    rng = np.random.default_rng(seed)
    classes = sorted(LABELS)
    m_flat, t_flat = mapped.ravel(), truth.ravel()
    idx = [np.flatnonzero(m_flat == c) for c in classes]
    N = np.array([i.size for i in idx], dtype=float)
    true = np.array([(t_flat == c).sum() for c in classes], dtype=float)
    err, cover = [], np.zeros(len(classes))
    for _ in range(reps):
        m, r = [], []
        for c, ix in zip(classes, idx):
            s = rng.choice(ix, alloc[c], replace=False)
            m.append(m_flat[s])
            r.append(t_flat[s])
        _, n = core.error_matrix(np.concatenate(m).tolist(), np.concatenate(r).tolist(), classes)
        res = core.area_adjusted_accuracy(n, N)
        est = res.area_prop * N.sum()
        cover += np.abs(est - true) <= res.z * res.area_prop_se * N.sum()
        err.append((est - true) / true * 100)
    err = np.array(err)
    return {LABELS[c]: {"coverage": round(float(cover[i] / reps), 3),
                        "err_p2_5": float(np.percentile(err[:, i], 2.5)),
                        "err_p97_5": float(np.percentile(err[:, i], 97.5)),
                        "mean_err_pct": float(err[:, i].mean())} for i, c in enumerate(classes)}


def make_hero(summary, mapped, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ink, ink2, muted, grid = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 13, "axes.edgecolor": muted,
                         "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#fcfcfb")
    fig.text(0.04, 0.93, "Accuracy Assessment Toolbox for ArcGIS Pro", fontsize=26, weight="bold", color=ink)
    fig.text(0.04, 0.885, "Pixel counting inherits the map's errors. Stratified estimation (Olofsson et al. 2014) "
             "corrects them and reports how uncertain the corrected area is.", fontsize=14, color=ink2)

    # --- left: error relative to the true area ------------------------------
    ax = fig.add_axes([0.13, 0.20, 0.47, 0.61])
    names = list(summary["classes"])
    y = np.arange(len(names))[::-1]
    light = "#b9d3f2"
    for yi, nm in zip(y, names):
        d = summary["classes"][nm]
        rp = summary["repeat"][nm]
        t = d["true_ha"]
        est, lo, hi = ((d["estimate_ha"] - t) / t * 100, (d["estimate_ha"] - d["ci_ha"] - t) / t * 100,
                       (d["estimate_ha"] + d["ci_ha"] - t) / t * 100)
        px = (d["mapped_ha"] - t) / t * 100
        ax.plot([rp["err_p2_5"], rp["err_p97_5"]], [yi + 0.16] * 2, color=light, lw=9, solid_capstyle="round")
        ax.plot([lo, hi], [yi + 0.16] * 2, color=blue, lw=2.5, solid_capstyle="round")
        ax.plot(est, yi + 0.16, "o", ms=10, color=blue, mec="#fcfcfb", mew=2)
        ax.plot(px, yi - 0.18, "s", ms=10, color=orange, mec="#fcfcfb", mew=2)
        ax.text(px + (1.2 if px >= 0 else -1.2), yi - 0.18, f"{px:+.0f}%", va="center",
                ha="left" if px >= 0 else "right", fontsize=11, color=ink2)
    ax.axvline(0, color=ink, lw=1.5)
    ax.text(0.4, y[0] + 0.55, "true area", fontsize=11, color=ink2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=14, color=ink)
    ax.set_ylim(-0.7, len(names) - 0.2)
    ax.set_xlabel("Error relative to the true class area (%)")
    ax.grid(axis="x", color=grid, lw=1)
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor("#fcfcfb")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="s", ls="", ms=10, color=orange, label="Pixel counting (mapped area)"),
               Line2D([], [], marker="o", color=blue, lw=2.5, ms=9, label="Stratified estimate, 95% CI (this sample)"),
               Line2D([], [], color=light, lw=9, label="Middle 95% of estimates over 1,000 repeat samples")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=11,
              handlelength=1.6, columnspacing=1.2)
    oa, oaci = summary["overall_accuracy"], summary["overall_ci"]
    ax.set_title(f"{summary['n_points']:,} stratified random points  |  overall accuracy {oa:.3f} +/- {oaci:.3f}",
                 loc="left", fontsize=13, color=ink2, pad=10)

    rp = summary["repeat"]
    cov = ", ".join(f"{k} {v['coverage']:.2f}" for k, v in rp.items())
    misses = [k for k, d in summary["classes"].items() if not d["ci_covers_truth"]]
    note = (f"95% CI coverage over 1,000 repeat samples: {cov}. Water's rare omission errors are usually absent "
            "from a sample, so its interval often collapses to zero width.")
    if misses:
        note += (f"\nThis particular draw (seed 42) misses the truth for {', '.join(misses)}; "
                 "a 95% interval is expected to miss about 1 draw in 20.")
    fig.text(0.04, 0.025, note, fontsize=10.5, color=muted, linespacing=1.4)

    # --- right top: the map ---------------------------------------------------
    axm = fig.add_axes([0.66, 0.47, 0.30, 0.36])
    map_colors = ["#fcfcfb", "#2a78d6", "#008300", "#e6c36a", "#e87ba4", "#52514e"]
    cmap = ListedColormap(map_colors)
    axm.imshow(mapped, cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
    axm.set_xticks([])
    axm.set_yticks([])
    for s in axm.spines.values():
        s.set_visible(False)
    axm.set_title("Synthetic 30 m classified raster (demo)", loc="left", fontsize=12, color=ink2)
    from matplotlib.patches import Patch
    axm.legend(handles=[Patch(color=map_colors[c], label=LABELS[c]) for c in sorted(LABELS)],
               loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False, fontsize=10,
               handlelength=1.0, columnspacing=0.9, handletextpad=0.4)

    # --- right bottom: tool list ---------------------------------------------
    tools = [("1  Stratified Random Sample Design", "Sample size (eq. 13), allocation with a per-class floor,\nseeded stratified random points"),
             ("2  Area-Adjusted Accuracy", "Area-weighted error matrix, OA / UA / PA and class areas\nwith 95% CIs; table + Markdown/HTML report"),
             ("3  Spatial Block Assignment", "Fishnet blocks, balanced folds and an optional buffer\nfor spatially blocked train/test splits")]
    fig.text(0.66, 0.37, "Python toolbox  AccuracyAssessment.pyt", fontsize=13, weight="bold", color=ink)
    for i, (t, d) in enumerate(tools):
        yy = 0.315 - i * 0.09
        fig.text(0.665, yy, t, fontsize=13, color=blue, weight="bold")
        fig.text(0.68, yy - 0.055, d, fontsize=11, color=ink2, linespacing=1.3)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()

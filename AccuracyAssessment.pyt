# -*- coding: utf-8 -*-
"""Accuracy Assessment Toolbox for ArcGIS Pro.

Good-practice accuracy assessment and area estimation for classified rasters
(Olofsson et al. 2013, 2014), plus spatial blocking of reference points for
spatially blocked validation (Roberts et al. 2017).

All statistics live in accuracy_core.py (numpy only, unit tested). This file
handles parameters, validation, raster/table I/O and geoprocessing messages.
"""

import importlib
import os
import sys

import arcpy
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import accuracy_core as core  # noqa: E402

importlib.reload(core)  # pick up edits without restarting ArcGIS Pro

BLOCK_ROWS = 2048       # raster rows read per block
INT_FIELD_TYPES = ["Short", "Long", "BigInteger", "Text"]

# Linear unit -> metres, for GPLinearUnit parameters.
_UNIT_M = {
    "meters": 1.0, "kilometers": 1000.0, "centimeters": 0.01, "millimeters": 0.001,
    "decimeters": 0.1, "feet": 0.3048, "ussurveyfeet": 1200.0 / 3937.0, "yards": 0.9144,
    "miles": 1609.344, "nauticalmiles": 1852.0, "inches": 0.0254,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _param(name, display, datatype, direction="Input", ptype="Required", multi=False):
    return arcpy.Parameter(name=name, displayName=display, datatype=datatype,
                           parameterType=ptype, direction=direction, multiValue=multi)


def _raster_info(raster_path):
    """Describe a classified raster; raises ValueError with a user-facing message."""
    r = arcpy.Raster(raster_path)
    if r.bandCount != 1:
        raise ValueError("The classified raster must have a single band.")
    if not r.isInteger:
        raise ValueError("The classified raster must be an integer raster (class codes).")
    sr = r.spatialReference
    if sr is None or sr.type != "Projected":
        raise ValueError("The classified raster must be in a projected coordinate system so cell "
                         "areas are equal. Project it first (equal-area projection recommended).")
    return {
        "raster": r,
        "xmin": r.extent.XMin, "ymax": r.extent.YMax, "ymin": r.extent.YMin,
        "cw": r.meanCellWidth, "ch": r.meanCellHeight,
        "nrows": r.height, "ncols": r.width,
        "sentinels": _sentinels(r.pixelType),
        "sr": sr,
        "m_per_unit": sr.metersPerUnit or 1.0,
    }


def _sentinels(pixel_type):
    """Two fill values for NoData, taken from the top of the pixel type's range.

    Raster.noDataValue is not always populated (e.g. some geodatabase rasters),
    so NoData is detected by reading with one fill value and, only where that
    value appears, re-reading with a second one: a cell is NoData only if it
    takes both fill values.
    """
    pt = str(pixel_type).upper()
    bits = int("".join(ch for ch in pt if ch.isdigit()) or 32)
    signed = pt.startswith("S")
    container = 8 if bits <= 8 else bits
    top = (2 ** (container - 1) - 1) if signed else (2 ** container - 1)
    return top, top - 1


def _iter_blocks(info):
    """Yield (row_offset, 2-D class array, boolean valid mask) blocks, top to bottom."""
    r = info["raster"]
    nrows, ncols, ch, cw = info["nrows"], info["ncols"], info["ch"], info["cw"]
    s1, s2 = info["sentinels"]
    for r0 in range(0, nrows, BLOCK_ROWS):
        nr = min(BLOCK_ROWS, nrows - r0)
        y_low = info["ymax"] - (r0 + nr) * ch
        corner = arcpy.Point(info["xmin"] + cw * 1e-6, y_low + ch * 1e-6)
        kw = {"lower_left_corner": corner, "ncols": ncols, "nrows": nr}
        block = arcpy.RasterToNumPyArray(r, nodata_to_value=s1, **kw)
        maybe = block == s1
        if maybe.any():
            block2 = arcpy.RasterToNumPyArray(r, nodata_to_value=s2, **kw)
            valid = ~(maybe & (block2 == s2))
        else:
            valid = np.ones(block.shape, dtype=bool)
        yield r0, block, valid


def _class_counts(info):
    """Pixel count per class value (NoData excluded), block by block."""
    counts = {}
    for _, block, valid in _iter_blocks(info):
        vals, cnt = np.unique(block[valid], return_counts=True)
        for v, c in zip(vals.tolist(), cnt.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
    if not counts:
        raise ValueError("The classified raster has no valid (non-NoData) cells.")
    return dict(sorted(counts.items()))


def _area_units(info):
    """(label, factor) converting map-unit^2 to hectares."""
    return "ha", info["m_per_unit"] ** 2 / 10000.0


def _linear_unit_to_map(linear_unit_text, sr):
    """'1000 Meters' -> distance in the units of ``sr``."""
    parts = str(linear_unit_text).split()
    value = float(parts[0].replace(",", "."))
    unit = parts[1].lower() if len(parts) > 1 else "unknown"
    if unit in ("unknown", "") or sr.type != "Projected":
        return value
    if unit not in _UNIT_M:
        raise ValueError(f"Unsupported linear unit '{parts[1]}'.")
    return value * _UNIT_M[unit] / (sr.metersPerUnit or 1.0)


def _to_int(v, what):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} value '{v}' is not an integer class code.")
    if f != int(f):
        raise ValueError(f"{what} value '{v}' is not an integer class code.")
    return int(f)


def _parse_value_table(param_value, cast=float):
    """GPValueTable -> {class: value}."""
    out = {}
    if param_value:
        for row in param_value:
            if row[0] in (None, "") or row[1] in (None, ""):
                continue
            out[int(row[0])] = cast(row[1])
    return out


def _write_table(path, records, dtype):
    if arcpy.Exists(path):
        arcpy.management.Delete(path)
    arr = np.array(records, dtype=dtype)
    arcpy.da.NumPyArrayToTable(arr, path)


def _check_out_ext(param, exts):
    if param.valueAsText:
        ext = os.path.splitext(param.valueAsText)[1].lower()
        if ext not in exts:
            param.setErrorMessage("Use one of these extensions: " + ", ".join(exts))


def _msg_table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = "  ".join(str(h).rjust(w) for h, w in zip(header, widths))
    arcpy.AddMessage(line)
    arcpy.AddMessage("  ".join("-" * w for w in widths))
    for r in rows:
        arcpy.AddMessage("  ".join(str(c).rjust(w) for c, w in zip(r, widths)))


# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------

class Toolbox(object):
    def __init__(self):
        self.label = "Accuracy Assessment"
        self.alias = "accuracy"
        self.description = ("Good-practice sample design, area-adjusted accuracy and area estimation "
                            "(Olofsson et al. 2014), and spatial blocking for classified rasters.")
        self.tools = [StratifiedSampleDesign, AreaAdjustedAccuracy, SpatialBlockAssignment]


# ---------------------------------------------------------------------------
# Tool 1: Stratified Random Sample Design
# ---------------------------------------------------------------------------

class StratifiedSampleDesign(object):
    def __init__(self):
        self.label = "Stratified Random Sample Design"
        self.description = (
            "Computes the total reference sample size needed to reach a target standard error of overall "
            "accuracy (Olofsson et al. 2014, eq. 13), allocates it to the map classes (proportional to "
            "mapped area with a minimum per class), and creates a seeded stratified random sample of "
            "pixel-centre points with the map class attached and an empty reference field to fill in.")
        self.category = "1 Sample design"
        self.canRunInBackground = True

    def getParameterInfo(self):
        p0 = _param("in_raster", "Classified Raster", ["DERasterDataset", "GPRasterLayer"])
        p1 = _param("out_points", "Output Sample Points", "DEFeatureClass", "Output")
        p2 = _param("target_se", "Target Standard Error of Overall Accuracy", "GPDouble")
        p2.value = 0.01
        p3 = _param("default_users_accuracy", "Anticipated User's Accuracy (default for all classes)", "GPDouble")
        p3.value = 0.80
        p4 = _param("class_users_accuracy", "Anticipated User's Accuracy by Class", "GPValueTable", ptype="Optional")
        p4.columns = [["GPLong", "Class Value"], ["GPDouble", "Anticipated User's Accuracy"]]
        p5 = _param("min_per_class", "Minimum Points per Class", "GPLong")
        p5.value = 50
        p6 = _param("total_sample_size", "Total Sample Size Override", "GPLong", ptype="Optional")
        p7 = _param("seed", "Random Seed", "GPLong")
        p7.value = 42
        p8 = _param("out_design_table", "Output Sample Design Table", "DETable", "Output", "Optional")
        return [p0, p1, p2, p3, p4, p5, p6, p7, p8]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        p = {x.name: x for x in parameters}
        if p["in_raster"].altered and p["in_raster"].value:
            try:
                d = arcpy.Describe(p["in_raster"].valueAsText)
                sr = getattr(d, "spatialReference", None)
                if sr is not None and sr.type != "Projected":
                    p["in_raster"].setErrorMessage("Raster must be in a projected coordinate system "
                                                   "(equal-area recommended) so pixel counts are areas.")
                if getattr(d, "bandCount", 1) != 1:
                    p["in_raster"].setErrorMessage("Raster must be single-band.")
            except Exception:
                pass
        se = p["target_se"].value
        if se is not None and not 0 < se < 0.5:
            p["target_se"].setErrorMessage("Target SE must be between 0 and 0.5 (e.g. 0.01).")
        ua = p["default_users_accuracy"].value
        if ua is not None and not 0 < ua < 1:
            p["default_users_accuracy"].setErrorMessage("Anticipated accuracy must be strictly between 0 and 1.")
        if p["class_users_accuracy"].value:
            for row in p["class_users_accuracy"].value:
                if row[1] not in (None, "") and not 0 < float(row[1]) < 1:
                    p["class_users_accuracy"].setErrorMessage(
                        f"Class {row[0]}: anticipated accuracy must be strictly between 0 and 1.")
        mn = p["min_per_class"].value
        if mn is not None and mn < 0:
            p["min_per_class"].setErrorMessage("Minimum points per class must be >= 0.")
        elif mn is not None and 0 < mn < 20:
            p["min_per_class"].setWarningMessage("Fewer than ~20-50 points per class gives very wide "
                                                 "user's-accuracy intervals.")
        tot = p["total_sample_size"].value
        if tot is not None and tot < 1:
            p["total_sample_size"].setErrorMessage("Total sample size must be >= 1.")
        return

    def execute(self, parameters, messages):
        p = {x.name: x for x in parameters}
        info = _raster_info(p["in_raster"].valueAsText)
        out_fc = p["out_points"].valueAsText
        seed = int(p["seed"].value)

        arcpy.SetProgressor("default", "Counting pixels per class...")
        counts = _class_counts(info)
        classes = list(counts)
        N = np.array([counts[c] for c in classes], dtype=float)
        W = N / N.sum()
        area_label, f_ha = _area_units(info)
        cell_area = info["cw"] * info["ch"]

        ua_map = _parse_value_table(p["class_users_accuracy"].value)
        unknown = sorted(set(ua_map) - set(classes))
        if unknown:
            arcpy.AddWarning(f"Anticipated accuracies given for classes not in the raster: {unknown}")
        ua = np.array([ua_map.get(c, float(p["default_users_accuracy"].value)) for c in classes])

        target_se = float(p["target_se"].value)
        n_eq13 = core.sample_size_overall(W, ua, target_se, population=N.sum())
        n_total = int(p["total_sample_size"].value) if p["total_sample_size"].value else n_eq13
        alloc = core.allocate_sample(n_total, W, int(p["min_per_class"].value), available=N.astype(int))

        arcpy.AddMessage(f"Classes: {len(classes)}   valid pixels: {int(N.sum()):,}   "
                         f"cell: {info['cw']:g} x {info['ch']:g} map units")
        arcpy.AddMessage(f"Sample size from Olofsson et al. (2014) eq. 13 for SE(OA) = {target_se:g}: n = {n_eq13}")
        if p["total_sample_size"].value:
            arcpy.AddMessage(f"Total sample size overridden by user: n = {n_total}")
        if alloc.sum() != n_total:
            arcpy.AddWarning(f"Allocation total is {int(alloc.sum())} (minimum per class and/or class "
                             f"sizes changed it from {n_total}).")
        rows = [[c, f"{int(N[i]):,}", f"{W[i]:.4f}", f"{ua[i]:.2f}", int(alloc[i])] for i, c in enumerate(classes)]
        _msg_table(["Class", "Pixels", "W_i", "Exp. UA", "n_i"], rows)
        arcpy.AddMessage(f"Total points: {int(alloc.sum())}")
        # expected SE of OA under this allocation (for the anticipated accuracies)
        with np.errstate(divide="ignore", invalid="ignore"):
            exp_se = float(np.sqrt(np.nansum(W ** 2 * ua * (1 - ua) / np.maximum(alloc - 1, 1))))
        arcpy.AddMessage(f"Anticipated SE(OA) with this allocation: {exp_se:.4f}")

        arcpy.SetProgressor("default", "Drawing stratified random pixels...")
        selector = core.StratifiedPixelSelector(counts, dict(zip(classes, alloc.tolist())), seed=seed)
        picks = []
        for r0, block, valid in _iter_blocks(info):
            picks.extend(selector.feed(block, r0, valid))

        arcpy.SetProgressor("default", "Writing sample points...")
        out_dir, out_name = os.path.split(out_fc)
        if arcpy.Exists(out_fc):
            arcpy.management.Delete(out_fc)
        arcpy.management.CreateFeatureclass(out_dir, out_name, "POINT", spatial_reference=info["sr"])
        arcpy.management.AddField(out_fc, "SAMPLE_ID", "LONG")
        arcpy.management.AddField(out_fc, "MAP_CLASS", "LONG")
        arcpy.management.AddField(out_fc, "REF_CLASS", "LONG", field_is_nullable="NULLABLE")
        arcpy.management.AddField(out_fc, "STRAT_W", "DOUBLE")
        w_of = dict(zip(classes, W.tolist()))
        rng = np.random.default_rng(seed + 1)
        order = rng.permutation(len(picks))      # random IDs so interpreters see no class order
        with arcpy.da.InsertCursor(out_fc, ["SHAPE@XY", "SAMPLE_ID", "MAP_CLASS", "STRAT_W"]) as cur:
            for sid, k in enumerate(order, start=1):
                row, col, cls = picks[k]
                x = info["xmin"] + (col + 0.5) * info["cw"]
                y = info["ymax"] - (row + 0.5) * info["ch"]
                cur.insertRow(((x, y), sid, cls, w_of[cls]))
        arcpy.AddMessage(f"Wrote {len(picks)} points to {out_fc} (seed {seed}). Fill REF_CLASS from "
                         "reference data, then run Area-Adjusted Accuracy.")

        if p["out_design_table"].valueAsText:
            dtype = [("CLASS", "i4"), ("PIXELS", "f8"), ("AREA_HA", "f8"), ("W_I", "f8"),
                     ("EXP_UA", "f8"), ("N_POINTS", "i4")]
            recs = [(c, N[i], N[i] * cell_area * f_ha, W[i], ua[i], int(alloc[i])) for i, c in enumerate(classes)]
            _write_table(p["out_design_table"].valueAsText, recs, dtype)
            arcpy.AddMessage(f"Wrote design table {p['out_design_table'].valueAsText}")
        return


# ---------------------------------------------------------------------------
# Tool 2: Area-Adjusted Accuracy
# ---------------------------------------------------------------------------

class AreaAdjustedAccuracy(object):
    def __init__(self):
        self.label = "Area-Adjusted Accuracy (Olofsson 2014)"
        self.description = (
            "Estimates overall, user's and producer's accuracy and the area of each class with confidence "
            "intervals using the stratified estimators of Olofsson et al. (2013, 2014). Mapped class areas "
            "are computed from the classified raster (pixel count x cell area). Writes a per-class table, "
            "an optional error matrix table of estimated area proportions, and a Markdown or HTML report.")
        self.category = "2 Accuracy and area"
        self.canRunInBackground = True

    def getParameterInfo(self):
        p0 = _param("in_points", "Reference Points or Table", ["GPFeatureLayer", "GPTableView"])
        p1 = _param("reference_field", "Reference Class Field", "Field")
        p1.parameterDependencies = [p0.name]
        p1.filter.list = INT_FIELD_TYPES
        p2 = _param("map_field", "Map Class Field (empty = read from raster)", "Field", ptype="Optional")
        p2.parameterDependencies = [p0.name]
        p2.filter.list = INT_FIELD_TYPES
        p3 = _param("in_raster", "Classified Raster", ["DERasterDataset", "GPRasterLayer"])
        p4 = _param("out_table", "Output Accuracy Table", "DETable", "Output")
        p5 = _param("out_report", "Output Report (.md or .html)", "DEFile", "Output", "Optional")
        p5.filter.list = ["md", "html"]
        p6 = _param("confidence", "Confidence Level (%)", "GPDouble")
        p6.value = 95
        p7 = _param("class_labels", "Class Labels", "GPValueTable", ptype="Optional")
        p7.columns = [["GPLong", "Class Value"], ["GPString", "Label"]]
        p8 = _param("out_matrix_table", "Output Area-Proportion Error Matrix Table", "DETable", "Output", "Optional")
        p9 = _param("users_interval", "User's Accuracy Interval", "GPString", ptype="Optional")
        p9.filter.type = "ValueList"
        p9.filter.list = ["Wald", "Wilson"]
        p9.value = "Wald"
        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        p = {x.name: x for x in parameters}
        c = p["confidence"].value
        if c is not None and not 50 <= c < 100:
            p["confidence"].setErrorMessage("Confidence level must be between 50 and 99.9 percent.")
        if (p["map_field"].valueAsText and p["reference_field"].valueAsText
                and p["map_field"].valueAsText == p["reference_field"].valueAsText):
            p["map_field"].setErrorMessage("Map and reference fields must be different.")
        if p["in_points"].value and not p["map_field"].valueAsText:
            try:
                d = arcpy.Describe(p["in_points"].valueAsText)
                if getattr(d, "shapeType", None) != "Point":
                    p["map_field"].setErrorMessage("A map class field is required when the input is a table "
                                                   "(only point features can be read from the raster).")
            except Exception:
                pass
        _check_out_ext(p["out_report"], [".md", ".html", ".htm"])
        return

    def execute(self, parameters, messages):
        p = {x.name: x for x in parameters}
        info = _raster_info(p["in_raster"].valueAsText)
        conf = float(p["confidence"].value) / 100.0
        ref_f = p["reference_field"].valueAsText
        map_f = p["map_field"].valueAsText
        labels = _parse_value_table(p["class_labels"].value, cast=str)

        arcpy.SetProgressor("default", "Counting mapped pixels per class...")
        counts = _class_counts(info)
        cell_area = info["cw"] * info["ch"]
        area_label, f_ha = _area_units(info)

        arcpy.SetProgressor("default", "Reading reference points...")
        m_lab, r_lab = [], []
        skipped_ref = skipped_map = 0
        if map_f:
            with arcpy.da.SearchCursor(p["in_points"].valueAsText, [map_f, ref_f]) as cur:
                for mv, rv in cur:
                    mv, rv = _to_int(mv, "Map"), _to_int(rv, "Reference")
                    if rv is None:
                        skipped_ref += 1
                        continue
                    if mv is None:
                        skipped_map += 1
                        continue
                    m_lab.append(mv)
                    r_lab.append(rv)
        else:
            m_lab, r_lab, skipped_ref, skipped_map = self._sample_raster(p["in_points"].valueAsText, ref_f, info)
        if skipped_ref:
            arcpy.AddWarning(f"{skipped_ref} points have no reference class and were skipped.")
        if skipped_map:
            arcpy.AddWarning(f"{skipped_map} points have no map class (NoData or outside raster) and were skipped.")
        if len(m_lab) < 2:
            raise arcpy.ExecuteError("Fewer than two usable reference points.")

        not_mapped = sorted(set(m_lab) - set(counts))
        if not_mapped:
            arcpy.AddWarning(f"Map classes {not_mapped} occur at points but not in the raster; "
                             "check that the map field matches this raster.")
        classes = sorted(set(counts) | set(m_lab) | set(r_lab))
        _, n = core.error_matrix(m_lab, r_lab, classes)
        mapped_px = np.array([counts.get(c, 0) for c in classes], dtype=float)
        ua_method = (p["users_interval"].valueAsText or "Wald").lower()
        res = core.area_adjusted_accuracy(n, mapped_px * cell_area, confidence=conf, classes=classes,
                                          users_interval=ua_method, labels=labels)
        for w in res.warnings:
            arcpy.AddWarning(w)

        pct = int(round(conf * 100))
        arcpy.AddMessage(f"Reference points used: {int(n.sum())}   classes: {len(classes)}")
        arcpy.AddMessage(f"Overall accuracy (area-weighted): {res.overall:.4f} +/- {res.overall_ci:.4f} ({pct}% CI)")
        arcpy.AddMessage(f"Overall accuracy (unweighted sample): {res.naive_overall:.4f}   kappa: {res.kappa():.4f}")
        rows = []
        for i, c in enumerate(classes):
            rows.append([labels.get(c, c), int(n[i].sum()),
                         (f"{res.users[i]:.3f} ({res.users_low[i]:.3f}-{res.users_high[i]:.3f})"
                          if ua_method == "wilson" else f"{res.users[i]:.3f}+/-{res.users_ci[i]:.3f}"),
                         f"{res.producers[i]:.3f}+/-{res.producers_ci[i]:.3f}",
                         f"{res.mapped_area[i] * f_ha:,.1f}",
                         f"{res.area[i] * f_ha:,.1f}+/-{res.area_ci[i] * f_ha:,.1f}"])
        _msg_table(["Class", "n", "User's (Wilson)" if ua_method == "wilson" else "User's", "Producer's",
                    f"Mapped {area_label}", f"Estimated {area_label}"], rows)

        dtype = [("CLASS", "i4"), ("LABEL", "U64"), ("N_SAMPLE", "i4"), ("MAP_PIXELS", "f8"),
                 ("MAP_HA", "f8"), ("W_I", "f8"), ("UA", "f8"), ("UA_CI", "f8"), ("UA_LO", "f8"), ("UA_HI", "f8"), ("PA", "f8"),
                 ("PA_CI", "f8"), ("AREA_PROP", "f8"), ("AREA_HA", "f8"), ("AREA_CI_HA", "f8"),
                 ("OA", "f8"), ("OA_CI", "f8")]
        recs = []
        for i, c in enumerate(classes):
            recs.append((c, str(labels.get(c, c)), int(n[i].sum()), mapped_px[i], res.mapped_area[i] * f_ha,
                         res.weights[i], res.users[i], res.users_ci[i], res.users_low[i], res.users_high[i],
                         res.producers[i], res.producers_ci[i],
                         res.area_prop[i], res.area[i] * f_ha, res.area_ci[i] * f_ha, res.overall, res.overall_ci))
        _write_table(p["out_table"].valueAsText, recs, dtype)
        arcpy.AddMessage(f"Wrote accuracy table {p['out_table'].valueAsText}")

        if p["out_matrix_table"].valueAsText:
            mdt = [("MAP_CLASS", "i4"), ("LABEL", "U64")] + [(f"REF_{c}", "f8") for c in classes] + [("ROW_TOTAL", "f8")]
            mrecs = [tuple([c, str(labels.get(c, c))] + res.proportions[i].tolist() + [res.proportions[i].sum()])
                     for i, c in enumerate(classes)]
            _write_table(p["out_matrix_table"].valueAsText, mrecs, mdt)
            arcpy.AddMessage(f"Wrote error matrix table {p['out_matrix_table'].valueAsText}")

        if p["out_report"].valueAsText:
            path = p["out_report"].valueAsText
            src = (f"Map: {os.path.basename(p['in_raster'].valueAsText)}; reference: "
                   f"{os.path.basename(p['in_points'].valueAsText)} (field {ref_f}).")
            blocks = core.build_report(res, labels, area_unit=area_label, area_factor=f_ha, source=src)
            text = core.render_html(blocks) if path.lower().endswith((".html", ".htm")) else core.render_markdown(blocks)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            arcpy.AddMessage(f"Wrote report {path}")
        return

    @staticmethod
    def _sample_raster(points, ref_f, info):
        """Map class at each point by direct cell lookup (no Spatial Analyst needed)."""
        xs, ys, refs = [], [], []
        skipped_ref = 0
        d = arcpy.Describe(points)
        src_sr = d.spatialReference
        with arcpy.da.SearchCursor(points, ["SHAPE@XY", ref_f], spatial_reference=info["sr"]) as cur:
            for (x, y), rv in cur:
                rv = _to_int(rv, "Reference")
                if rv is None:
                    skipped_ref += 1
                    continue
                xs.append(x)
                ys.append(y)
                refs.append(rv)
        if src_sr and src_sr.name != info["sr"].name:
            arcpy.AddMessage(f"Points projected on the fly from {src_sr.name} to {info['sr'].name}.")
        rows, cols = core.cell_index(xs, ys, info["xmin"], info["ymax"], info["cw"], info["ch"])
        vals = np.full(len(xs), -1, dtype=np.int64)
        valid = np.zeros(len(xs), dtype=bool)
        for r0, block, valid_block in _iter_blocks(info):
            sel = (rows >= r0) & (rows < r0 + block.shape[0]) & (cols >= 0) & (cols < info["ncols"])
            if sel.any():
                v = block[rows[sel] - r0, cols[sel]]
                ok = valid_block[rows[sel] - r0, cols[sel]]
                idx = np.flatnonzero(sel)
                vals[idx[ok]] = v[ok]
                valid[idx[ok]] = True
        m_lab = vals[valid].tolist()
        r_lab = [r for r, ok in zip(refs, valid) if ok]
        return m_lab, r_lab, skipped_ref, int((~valid).sum())


# ---------------------------------------------------------------------------
# Tool 3: Spatial Block Assignment
# ---------------------------------------------------------------------------

class SpatialBlockAssignment(object):
    def __init__(self):
        self.label = "Spatial Block Assignment"
        self.description = (
            "Assigns reference points to square spatial blocks (a fishnet of user-set size) and assigns "
            "whole blocks to k folds, balanced by point count, for spatially blocked train/test splits "
            "(block cross-validation, Roberts et al. 2017). Adds BLOCK_ID, BLOCK_ROW, BLOCK_COL and FOLD "
            "fields and reports counts per fold (and per class if a class field is given). An optional "
            "buffer adds SPLIT_F1..SPLIT_Fk fields that drop training points lying within the buffer of "
            "each fold's test blocks.")
        self.category = "3 Validation design"
        self.canRunInBackground = True

    def getParameterInfo(self):
        p0 = _param("in_points", "Reference Points", "GPFeatureLayer")
        p0.filter.list = ["Point"]
        p1 = _param("out_features", "Output Points", "DEFeatureClass", "Output")
        p2 = _param("block_size", "Block Size", "GPLinearUnit")
        p2.value = "1000 Meters"
        p3 = _param("n_folds", "Number of Folds", "GPLong")
        p3.value = 5
        p4 = _param("seed", "Random Seed", "GPLong")
        p4.value = 42
        p5 = _param("class_field", "Class Field (for per-fold counts)", "Field", ptype="Optional")
        p5.parameterDependencies = [p0.name]
        p6 = _param("out_blocks", "Output Block Polygons", "DEFeatureClass", "Output", "Optional")
        p7 = _param("buffer_distance", "Buffer Distance (exclude training points near test blocks)",
                    "GPLinearUnit", ptype="Optional")
        return [p0, p1, p2, p3, p4, p5, p6, p7]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        p = {x.name: x for x in parameters}
        k = p["n_folds"].value
        if k is not None and k < 2:
            p["n_folds"].setErrorMessage("Use at least 2 folds.")
        if p["block_size"].valueAsText:
            try:
                if float(p["block_size"].valueAsText.split()[0].replace(",", ".")) <= 0:
                    p["block_size"].setErrorMessage("Block size must be greater than 0.")
            except ValueError:
                pass
        if p["buffer_distance"].valueAsText:
            try:
                bval = float(p["buffer_distance"].valueAsText.split()[0].replace(",", "."))
                if bval < 0:
                    p["buffer_distance"].setErrorMessage("Buffer distance must be 0 or greater.")
                elif p["block_size"].valueAsText:
                    bsz = p["block_size"].valueAsText.split()
                    bf = p["buffer_distance"].valueAsText.split()
                    if (len(bsz) > 1 and len(bf) > 1 and bsz[1].lower() == bf[1].lower()
                            and bval >= float(bsz[0].replace(",", "."))):
                        p["buffer_distance"].setWarningMessage(
                            "Buffer is at least one block wide; most training points near test blocks will "
                            "be dropped. Consider larger blocks instead.")
            except ValueError:
                pass
        if p["in_points"].value:
            try:
                sr = arcpy.Describe(p["in_points"].valueAsText).spatialReference
                if sr.type != "Projected":
                    p["in_points"].setWarningMessage("Points are in a geographic coordinate system; blocks will "
                                                     "be in degrees. Project the points for metric blocks.")
            except Exception:
                pass
        return

    def execute(self, parameters, messages):
        p = {x.name: x for x in parameters}
        src = p["in_points"].valueAsText
        out_fc = p["out_features"].valueAsText
        k = int(p["n_folds"].value)
        seed = int(p["seed"].value)
        class_f = p["class_field"].valueAsText
        sr = arcpy.Describe(src).spatialReference
        size = _linear_unit_to_map(p["block_size"].valueAsText, sr)

        if arcpy.Exists(out_fc):
            arcpy.management.Delete(out_fc)
        arcpy.management.CopyFeatures(src, out_fc)
        oid = arcpy.Describe(out_fc).OIDFieldName
        fields = [oid, "SHAPE@XY"] + ([class_f] if class_f else [])
        data = [r for r in arcpy.da.SearchCursor(out_fc, fields) if r[1] is not None and r[1][0] is not None]
        if len(data) < k:
            raise arcpy.ExecuteError(f"Only {len(data)} points with geometry for {k} folds.")
        xs = np.array([r[1][0] for r in data])
        ys = np.array([r[1][1] for r in data])
        brow, bcol, origin = core.block_index(xs, ys, size)
        ncol = int(bcol.max()) + 1
        bid = brow * ncol + bcol + 1
        try:
            fold_of = core.assign_folds(bid, k, seed)
        except ValueError as e:
            raise arcpy.ExecuteError(str(e))

        for name, ftype in (("BLOCK_ID", "LONG"), ("BLOCK_ROW", "LONG"), ("BLOCK_COL", "LONG"), ("FOLD", "SHORT")):
            if name in [f.name for f in arcpy.ListFields(out_fc)]:
                arcpy.management.DeleteField(out_fc, name)
            arcpy.management.AddField(out_fc, name, ftype)
        lookup = {r[0]: (int(bid[i]), int(brow[i]), int(bcol[i]), fold_of[int(bid[i])]) for i, r in enumerate(data)}
        with arcpy.da.UpdateCursor(out_fc, [oid, "BLOCK_ID", "BLOCK_ROW", "BLOCK_COL", "FOLD"]) as cur:
            for row in cur:
                if row[0] in lookup:
                    cur.updateRow([row[0], *lookup[row[0]]])

        folds = np.array([fold_of[int(b)] for b in bid])
        n_blocks = len(fold_of)
        arcpy.AddMessage(f"Block size: {size:g} map units; origin ({origin[0]:g}, {origin[1]:g}); "
                         f"{n_blocks} occupied blocks; {len(data)} points; {k} folds (seed {seed}).")
        per_block = np.unique(bid, return_counts=True)[1]
        arcpy.AddMessage(f"Points per occupied block: min {per_block.min()}, median {int(np.median(per_block))}, "
                         f"max {per_block.max()}")
        header = ["Fold", "Blocks", "Points"]
        cls_vals = []
        if class_f:
            cls_vals = sorted({r[2] for r in data if r[2] is not None}, key=lambda v: (str(type(v)), v))
            header += [f"cls {c}" for c in cls_vals]
        rows = []
        for f in range(1, k + 1):
            sel = folds == f
            row = [f, sum(1 for v in fold_of.values() if v == f), int(sel.sum())]
            if class_f:
                cl = [data[i][2] for i in np.flatnonzero(sel)]
                row += [cl.count(c) for c in cls_vals]
            rows.append(row)
        _msg_table(header, rows)
        if class_f:
            for j, c in enumerate(cls_vals):
                missing = [r[0] for r in rows if r[3 + j] == 0]
                if missing:
                    arcpy.AddWarning(f"Class {c} is absent from fold(s) {missing}; consider smaller "
                                     "blocks or fewer folds.")

        if p["buffer_distance"].valueAsText:
            buf = _linear_unit_to_map(p["buffer_distance"].valueAsText, sr)
            if buf > 0:
                self._apply_buffer(out_fc, oid, data, xs, ys, brow, bcol, folds, size, origin, buf, k)

        if p["out_blocks"].valueAsText:
            self._write_blocks(p["out_blocks"].valueAsText, sr, origin, size, bid, brow, bcol, fold_of)
        return

    @staticmethod
    def _apply_buffer(out_fc, oid, data, xs, ys, brow, bcol, folds, size, origin, buf, k):
        """Add SPLIT_F<f> fields: 1 = test, 0 = train, 2 = excluded (within buffer of a test block)."""
        roles = core.buffer_split(xs, ys, brow, bcol, folds, size, origin, buf, k)
        names = [f"SPLIT_F{f}" for f in range(1, k + 1)]
        existing = [f.name for f in arcpy.ListFields(out_fc)]
        for name in names:
            if name in existing:
                arcpy.management.DeleteField(out_fc, name)
            arcpy.management.AddField(out_fc, name, "SHORT", field_alias=f"Fold {name[7:]} role (0 train, 1 test, 2 excluded)")
        lookup = {r[0]: roles[i].tolist() for i, r in enumerate(data)}
        with arcpy.da.UpdateCursor(out_fc, [oid] + names) as cur:
            for row in cur:
                if row[0] in lookup:
                    cur.updateRow([row[0], *lookup[row[0]]])
        arcpy.AddMessage(f"Buffer {buf:g} map units: for each fold, training points within the buffer of "
                         "that fold's test blocks are marked 2 (excluded) in SPLIT_F<fold>.")
        rows = []
        for f in range(k):
            col = roles[:, f]
            n_test, n_train, n_excl = int((col == 1).sum()), int((col == 0).sum()), int((col == 2).sum())
            share = n_excl / max(n_train + n_excl, 1)
            rows.append([f + 1, n_test, n_train, n_excl, f"{share:.0%}"])
            if share > 0.5:
                arcpy.AddWarning(f"Fold {f + 1}: the buffer removes {share:.0%} of the training points; "
                                 "use larger blocks or a smaller buffer.")
        _msg_table(["Fold", "Test", "Train", "Excluded", "Excl. share"], rows)

    @staticmethod
    def _write_blocks(path, sr, origin, size, bid, brow, bcol, fold_of):
        out_dir, name = os.path.split(path)
        if arcpy.Exists(path):
            arcpy.management.Delete(path)
        arcpy.management.CreateFeatureclass(out_dir, name, "POLYGON", spatial_reference=sr)
        for f, t in (("BLOCK_ID", "LONG"), ("FOLD", "SHORT"), ("N_POINTS", "LONG")):
            arcpy.management.AddField(path, f, t)
        ids, cnt = np.unique(bid, return_counts=True)
        first = {int(b): i for i, b in reversed(list(enumerate(bid.tolist())))}
        with arcpy.da.InsertCursor(path, ["SHAPE@", "BLOCK_ID", "FOLD", "N_POINTS"]) as cur:
            for b, c in zip(ids.tolist(), cnt.tolist()):
                i = first[b]
                x0 = origin[0] + bcol[i] * size
                y0 = origin[1] + brow[i] * size
                ring = arcpy.Array([arcpy.Point(x0, y0), arcpy.Point(x0, y0 + size), arcpy.Point(x0 + size, y0 + size),
                                    arcpy.Point(x0 + size, y0), arcpy.Point(x0, y0)])
                cur.insertRow((arcpy.Polygon(ring, sr), b, fold_of[b], c))
        arcpy.AddMessage(f"Wrote {len(ids)} block polygons to {path}")

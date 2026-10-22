"""Write the tool help metadata (*.pyt.xml) that ArcGIS Pro shows in tool dialogs.

ArcGIS Pro reads ``<Toolbox>.<Tool>.pyt.xml`` next to the .pyt for the parameter
hover help, tool summary and usage notes. This script keeps that help text in
one place and regenerates the XML from the toolbox's own parameter list:

    "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe" tools\\build_metadata.py
"""

import datetime as dt
import importlib.machinery
import importlib.util
import os
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYT = os.path.join(ROOT, "AccuracyAssessment.pyt")

HELP = {
    "StratifiedSampleDesign": {
        "summary": "Designs a stratified random reference sample for a classified raster: total sample size from "
                   "a target standard error of overall accuracy (Olofsson et al. 2014, eq. 13), allocation to "
                   "classes in proportion to mapped area with a minimum per class, and seeded random points at "
                   "pixel centres.",
        "usage": [
            "The raster must be a single-band integer raster in a projected (ideally equal-area) coordinate system.",
            "Each class is a stratum; points are a simple random sample of pixels within each class.",
            "Output points carry MAP_CLASS (the stratum), STRAT_W (mapped area proportion) and an empty "
            "REF_CLASS field for the reference label.",
            "The same seed and inputs always produce the same points.",
        ],
        "params": {
            "in_raster": "Single-band integer classified raster. Class values are the strata.",
            "out_points": "Point feature class of sample locations (pixel centres).",
            "target_se": "Target standard error of overall accuracy, S(O). 0.01 is a common choice; smaller "
                         "values need many more points.",
            "default_users_accuracy": "Anticipated user's accuracy used for every class not listed in the "
                                      "by-class table (between 0 and 1).",
            "class_users_accuracy": "Optional anticipated user's accuracy per class value, e.g. higher for "
                                    "stable classes and lower for rare change classes.",
            "min_per_class": "Minimum number of points per class (e.g. 50 or 100). Rare classes get this "
                             "floor; the rest is shared in proportion to area.",
            "total_sample_size": "Optional total that replaces the eq. 13 sample size (e.g. a budget).",
            "seed": "Random seed for reproducible samples.",
            "out_design_table": "Optional table with pixels, area (ha), weight, anticipated accuracy and "
                                "points per class.",
        },
    },
    "AreaAdjustedAccuracy": {
        "summary": "Estimates overall, user's and producer's accuracy and class areas with confidence intervals "
                   "from a stratified random sample, using the stratified estimators of Olofsson et al. "
                   "(2013, 2014). Mapped areas come from the classified raster (pixel count x cell area).",
        "usage": [
            "Use the raster that defined the strata (the map the sample was drawn from).",
            "Rows with an empty reference class are skipped and counted in a warning.",
            "Leave the map class field empty to read the map class from the raster at each point.",
            "The report is Markdown (.md) or HTML (.html) depending on the file extension.",
            "When a class's user's accuracy is exactly 0 or 1, the report and messages give its Wilson "
            "interval, because the Wald interval has zero width there.",
            "The estimators assume the sample was stratified by the classes of this raster (or is a simple "
            "random sample).",
        ],
        "params": {
            "in_points": "Reference points (feature layer) or a table with map and reference class fields.",
            "reference_field": "Field holding the reference (ground truth) class code.",
            "map_field": "Field holding the map class code. Leave empty to sample the raster at each point "
                         "(point features only).",
            "in_raster": "Classified raster used to compute the mapped area of each class.",
            "out_table": "Table with one row per class: sample size, mapped area, weight, user's and "
                         "producer's accuracy with CI half-widths, estimated area and its CI, and overall "
                         "accuracy.",
            "out_report": "Optional Markdown or HTML report with both error matrices and all estimates.",
            "confidence": "Confidence level for intervals in percent (95 gives +/- 1.96 SE).",
            "class_labels": "Optional class names for the table and report.",
            "out_matrix_table": "Optional table of the error matrix in estimated area proportions "
                                "(rows = map class, columns = reference class).",
            "users_interval": "Interval for user's accuracy. Wald (default) is U +/- z SE, as in Olofsson "
                              "et al. (2014). Wilson is the Wilson score interval: it stays inside 0-1 and "
                              "does not collapse to zero width when every sample of a class agrees (U = 1) "
                              "or disagrees (U = 0). Overall accuracy, producer's accuracy and areas always "
                              "use Wald intervals.",
        },
    },
    "SpatialBlockAssignment": {
        "summary": "Assigns points to square spatial blocks and assigns whole blocks to k folds balanced by "
                   "point count, for spatially blocked train/test splits (Roberts et al. 2017). An optional "
                   "buffer drops training points that lie close to each fold's test blocks.",
        "usage": [
            "Choose a block size at least as large as the range of spatial autocorrelation in the data.",
            "All points in a block share a fold, so near neighbours never sit on both sides of a split.",
            "The block grid is aligned to multiples of the block size, so reruns give the same blocks.",
            "Give a class field to see per-fold class counts and a warning when a class is missing from a fold.",
            "Without a buffer, points on either side of a block edge can still be close. With a buffer, "
            "SPLIT_F1..SPLIT_Fk fields give each point's role when fold f is the test set: 1 = test, "
            "0 = train, 2 = excluded from training (within the buffer of a test block). Train on SPLIT_Ff = 0, "
            "test on SPLIT_Ff = 1.",
        ],
        "params": {
            "in_points": "Point features to split (projected coordinates recommended).",
            "out_features": "Copy of the points with BLOCK_ID, BLOCK_ROW, BLOCK_COL and FOLD fields.",
            "block_size": "Width of the square blocks (linear unit).",
            "n_folds": "Number of folds (k >= 2).",
            "seed": "Random seed for the order in which blocks are dealt to folds.",
            "class_field": "Optional class field for per-fold class counts.",
            "out_blocks": "Optional polygon feature class of occupied blocks with their fold and point count.",
            "buffer_distance": "Optional distance. For each fold, training points closer than this to any of "
                               "that fold's test blocks are marked as excluded (SPLIT_F<fold> = 2). Use roughly "
                               "the range of spatial autocorrelation; leave empty for a plain block split.",
        },
    },
}


def _div(text):
    return escape(f'<DIV STYLE="text-align:Left;"><P><SPAN>{escape(text)}</SPAN></P></DIV>')


def _load_pyt():
    loader = importlib.machinery.SourceFileLoader("acc_pyt", PYT)
    spec = importlib.util.spec_from_loader("acc_pyt", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def write_tool(mod, tool_cls, now):
    tool = tool_cls()
    h = HELP[tool_cls.__name__]
    params = tool.getParameterInfo()
    lines = ['<?xml version="1.0"?>', '<metadata xml:lang="en">',
             f"<Esri><CreaDate>{now:%Y%m%d}</CreaDate><CreaTime>{now:%H%M%S}00</CreaTime>"
             "<ArcGISFormat>1.0</ArcGISFormat><SyncOnce>TRUE</SyncOnce>"
             f"<ModDate>{now:%Y%m%d}</ModDate><ModTime>{now:%H%M%S}00</ModTime></Esri>",
             f'<tool name="{tool_cls.__name__}" displayname="{escape(tool.label)}" '
             'toolboxalias="accuracy" xmlns="">',
             "<parameters>"]
    for p in params:
        ptype = p.parameterType
        lines.append(f'<param name="{p.name}" displayname="{escape(p.displayName)}" type="{ptype}" '
                     f'direction="{p.direction}" datatype="{escape(str(p.datatype))}" expression="{p.name}">'
                     f"<dialogReference>{_div(h['params'][p.name])}</dialogReference>"
                     f"<pythonReference>{_div(h['params'][p.name])}</pythonReference></param>")
    lines.append("</parameters>")
    lines.append(f"<summary>{_div(h['summary'])}</summary>")
    usage = "".join(f"<P><SPAN>{escape(u)}</SPAN></P>" for u in h["usage"])
    lines.append(f"<usage>{escape('<DIV STYLE=\"text-align:Left;\">' + usage + '</DIV>')}</usage>")
    lines.append("</tool>")
    lines.append(f"<dataIdInfo><idCitation><resTitle>{escape(tool.label)}</resTitle></idCitation>"
                 f"<idAbs>{_div(h['summary'])}</idAbs>"
                 "<searchKeys><keyword>accuracy assessment</keyword><keyword>area estimation</keyword>"
                 "<keyword>remote sensing</keyword></searchKeys></dataIdInfo>")
    lines.append("<distInfo><distributor><distorFormat><formatName>ArcToolbox Tool</formatName>"
                 "</distorFormat></distributor></distInfo>")
    lines.append('<mdHrLv><ScopeCd value="005"/></mdHrLv>')
    lines.append("</metadata>")
    path = os.path.join(ROOT, f"AccuracyAssessment.{tool_cls.__name__}.pyt.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_toolbox(mod, now):
    tb = mod.Toolbox()
    text = ('<?xml version="1.0"?>\n<metadata xml:lang="en">'
            f"<Esri><CreaDate>{now:%Y%m%d}</CreaDate><ArcGISFormat>1.0</ArcGISFormat><SyncOnce>TRUE</SyncOnce></Esri>"
            f"<toolbox name=\"AccuracyAssessment\" alias=\"accuracy\"><arcToolboxHelpPath>"
            "</arcToolboxHelpPath><toolsets/></toolbox>"
            f"<dataIdInfo><idCitation><resTitle>{escape(tb.label)}</resTitle></idCitation>"
            f"<idAbs>{_div(tb.description)}</idAbs></dataIdInfo>"
            '<mdHrLv><ScopeCd value="005"/></mdHrLv></metadata>\n')
    path = os.path.join(ROOT, "AccuracyAssessment.pyt.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def main():
    mod = _load_pyt()
    now = dt.datetime.now()
    for cls in mod.Toolbox().tools:
        print("wrote", write_tool(mod, cls, now))
    print("wrote", write_toolbox(mod, now))


if __name__ == "__main__":
    main()

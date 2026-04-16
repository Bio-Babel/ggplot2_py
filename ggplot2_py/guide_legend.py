"""
Legend guide building functions — faithful port of R's GuideLegend.

Each legend is built as an independent :class:`~gtable_py.Gtable` with
its own viewport-based layout.  Multiple legends are combined via
:func:`package_legend_box` into a composite guide-box gtable.

R references
------------
* ``ggplot2/R/guide-legend.R`` — GuideLegend class
* ``ggplot2/R/guides-.R``      — Guides$package_box
* ``ggplot2/R/guide-.R``       — Guide$add_title
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grid_py import (
    GList,
    GTree,
    Gpar,
    Unit,
    Viewport,
    null_grob,
    rect_grob,
    text_grob,
    unit_c,
)
from grid_py._grob import grob_tree

from gtable_py import (
    Gtable,
    gtable_add_cols,
    gtable_add_grob,
    gtable_add_padding,
    gtable_add_row_space,
    gtable_add_rows,
    gtable_col,
    gtable_height,
    gtable_width,
)

__all__ = [
    "build_legend_decor",
    "build_legend_labels",
    "measure_legend_grobs",
    "arrange_legend_layout",
    "assemble_legend",
    "add_legend_title",
    "package_legend_box",
]

# ---------------------------------------------------------------------------
# Constants  (R defaults from ggplot2 theme)
# ---------------------------------------------------------------------------

_DEFAULT_KEY_WIDTH_CM: float = 0.5     # legend.key.width  default ~1.2 lines
_DEFAULT_KEY_HEIGHT_CM: float = 0.5    # legend.key.height default ~1.2 lines
_DEFAULT_SPACING_X_CM: float = 0.15    # legend.key.spacing.x
_DEFAULT_SPACING_Y_CM: float = 0.0     # legend.key.spacing.y (vertical: 0)
_DEFAULT_PADDING_CM: float = 0.15      # legend.margin
_DEFAULT_LABEL_SIZE: float = 6.0       # legend.text size (pt)
_DEFAULT_TITLE_SIZE: float = 7.0       # legend.title size (pt)


def _text_width_cm(text: str, fontsize: float = 10.0) -> float:
    """Measure text width in cm using Cairo font metrics.

    Replaces the old ``len(text) * 0.18`` character-count heuristic with
    actual font measurement, matching R's ``width_cm(label_grob)`` pattern
    (utilities-grid.R:67-77).
    """
    from grid_py._size import calc_string_metric
    m = calc_string_metric(text, Gpar(fontsize=fontsize))
    return m["width"] * 2.54  # inches → cm


def _text_height_cm(text: str, fontsize: float = 10.0) -> float:
    """Measure text height in cm using Cairo font metrics.

    Matches R's ``height_cm(label_grob)`` pattern.
    """
    from grid_py._size import calc_string_metric
    m = calc_string_metric(text, Gpar(fontsize=fontsize))
    return (m["ascent"] + m["descent"]) * 2.54  # inches → cm


# ---------------------------------------------------------------------------
# build_legend_decor
# ---------------------------------------------------------------------------

def build_legend_decor(
    entry: Dict[str, Any],
    draw_key_fn: Callable,
    layers: Any,
    key_width_cm: float = _DEFAULT_KEY_WIDTH_CM,
    key_height_cm: float = _DEFAULT_KEY_HEIGHT_CM,
) -> List[Any]:
    """Build legend key glyphs for one merged entry.

    For each break index, calls *draw_key_fn* with the aesthetic data for
    that break, then wraps the resulting grob in a ``GTree`` whose viewport
    is sized ``key_width_cm x key_height_cm``.

    Mirrors ``GuideLegend$build_decor`` in R (guide-legend.R:396-431).

    Parameters
    ----------
    entry : dict
        Merged legend entry with keys ``aes_mapped``, ``breaks``, ``labels``.
    draw_key_fn : callable
        The geom's ``draw_key`` function (e.g. ``draw_key_point``).
    layers : list
        Plot layers (used to detect geom params).
    key_width_cm, key_height_cm : float
        Key glyph dimensions in cm.

    Returns
    -------
    list of grob
        One glyph GTree per break.
    """
    from ggplot2_py.plot import _safe_colour

    aes_mapped = entry["aes_mapped"]
    n_breaks = len(entry["breaks"])

    # Resolve layer params (first layer only, like R)
    layer_params: Dict[str, Any] = {}
    if layers:
        for layer in layers:
            geom = getattr(layer, "geom", None)
            if geom is not None:
                layer_params = getattr(layer, "computed_geom_params", {})
                if not layer_params:
                    layer_params = getattr(geom, "default_params", {})
                    if callable(layer_params):
                        layer_params = layer_params()
                break

    # Key size passed to draw_key as mm (R multiplies by 10 from cm)
    key_size = (key_width_cm * 10, key_height_cm * 10)

    # Key background grob (R: element_grob(elements$key))
    key_bg = rect_grob(
        gp=Gpar(fill="white", col="grey90", lwd=0.5),
        name="legend.key.bg",
    )

    key_glyphs = []
    for i in range(n_breaks):
        # Build the aesthetic data dict for this break
        data: Dict[str, Any] = {}
        for aes_name, mapped_vals in aes_mapped.items():
            val = mapped_vals[i] if i < len(mapped_vals) else None
            if aes_name in ("colour", "color"):
                data["colour"] = _safe_colour(val)
            elif aes_name == "fill":
                data["fill"] = _safe_colour(val)
            elif aes_name == "shape":
                data["shape"] = int(val) if val is not None else 19
            elif aes_name == "size":
                try:
                    data["size"] = float(val) if val is not None else 1.5
                    if np.isnan(data["size"]):
                        data["size"] = 1.5
                except (TypeError, ValueError):
                    data["size"] = 1.5
            elif aes_name == "linetype":
                data["linetype"] = val
            elif aes_name == "linewidth":
                data["linewidth"] = val
            elif aes_name == "alpha":
                data["alpha"] = val
            else:
                data[aes_name] = val

        # Fill in defaults that draw_key functions expect
        data.setdefault("colour", "black")
        data.setdefault("fill", "black")
        data.setdefault("size", 1.5)
        data.setdefault("alpha", None)
        data.setdefault("stroke", 0.5)
        data.setdefault("shape", 19)
        data.setdefault("linetype", 1)
        data.setdefault("linewidth", 0.5)

        # Call the draw_key function.
        # draw_key_fn may be a bound method (from ggproto) or a plain function.
        # Try plain call first; if TypeError (too many args from bound self),
        # extract the underlying function.
        try:
            glyph = draw_key_fn(data, layer_params, key_size)
        except TypeError:
            # Likely a bound method — get the underlying function
            fn = getattr(draw_key_fn, "__func__", draw_key_fn)
            glyph = fn(data, layer_params, key_size)

        # --- set_key_size (R: guide-legend.R:626-641) ---
        # Compute glyph physical size from aesthetics: (size + linewidth) / 10
        # This converts mm to cm, matching R's set_key_size().
        glyph_w = getattr(glyph, "_width", None)
        glyph_h = getattr(glyph, "_height", None)
        if glyph_w is None or glyph_h is None:
            _size = data.get("size", 0) or 0
            _lwd = data.get("linewidth", 0) or 0
            _stroke = data.get("stroke", 0) or 0
            try:
                _size = float(_size) if not (isinstance(_size, float) and np.isnan(_size)) else 0
            except (TypeError, ValueError):
                _size = 0
            try:
                _lwd = float(_lwd) if not (isinstance(_lwd, float) and np.isnan(_lwd)) else 0
            except (TypeError, ValueError):
                _lwd = 0
            try:
                _stroke = float(_stroke) if not (isinstance(_stroke, float) and np.isnan(_stroke)) else 0
            except (TypeError, ValueError):
                _stroke = 0
            measured_cm = (_size + _lwd + _stroke) / 10.0
            if glyph_w is None:
                glyph_w = measured_cm
            if glyph_h is None:
                glyph_h = measured_cm

        # Effective key size = max(default, measured glyph size)
        eff_w = max(key_width_cm, glyph_w, 0)
        eff_h = max(key_height_cm, glyph_h, 0)

        # Wrap in a GTree with a justified viewport (R: build_decor lines 417-428)
        vp = Viewport(
            x=0.5, y=0.5, just="centre",
            width=Unit(eff_w, "cm"),
            height=Unit(eff_h, "cm"),
        )
        key_grob = GTree(
            children=GList(key_bg, glyph),
            vp=vp,
            name=f"key-{i}",
        )
        # Store measured size on grob (R: attr(grob, "width") <- width)
        key_grob._width = eff_w
        key_grob._height = eff_h
        key_glyphs.append(key_grob)

    return key_glyphs


# ---------------------------------------------------------------------------
# build_legend_labels
# ---------------------------------------------------------------------------

def build_legend_labels(
    entry: Dict[str, Any],
    label_size: float = _DEFAULT_LABEL_SIZE,
    label_colour: str = "grey20",
) -> List[Any]:
    """Build text grobs for legend labels.

    Mirrors ``GuideLegend$build_labels`` (guide-legend.R:433-450).

    Parameters
    ----------
    entry : dict
        Merged legend entry.
    label_size : float
        Font size in points.
    label_colour : str
        Font colour.

    Returns
    -------
    list of grob
        One text grob per label.
    """
    labels = entry.get("labels", [])
    if not labels:
        return [null_grob()]

    grobs = []
    for lab in labels:
        grobs.append(text_grob(
            label=str(lab),
            x=0.0,
            y=0.5,
            just=("left", "centre"),
            gp=Gpar(fontsize=label_size, col=label_colour),
            name=f"guide.label.{lab}",
        ))
    return grobs


# ---------------------------------------------------------------------------
# measure_legend_grobs
# ---------------------------------------------------------------------------

def measure_legend_grobs(
    decor: List[Any],
    labels: List[Any],
    n_breaks: int,
    nrow: int,
    ncol: int,
    key_width_cm: float = _DEFAULT_KEY_WIDTH_CM,
    key_height_cm: float = _DEFAULT_KEY_HEIGHT_CM,
    spacing_x: float = _DEFAULT_SPACING_X_CM,
    spacing_y: float = _DEFAULT_SPACING_Y_CM,
    text_position: str = "right",
    byrow: bool = False,
    label_size: float = _DEFAULT_LABEL_SIZE,
) -> Dict[str, List[float]]:
    """Measure keys and labels, compute gtable widths/heights with spacing.

    Mirrors ``GuideLegend$measure_grobs`` (guide-legend.R:452-501).

    The returned widths/heights include interleaved spacing columns/rows
    between key and label cells.  For ``text_position="right"`` (default)
    the column pattern is: [key_w, label_w, gap, key_w, label_w, gap, ...]
    (last gap stripped).

    Parameters
    ----------
    decor : list of grob
        Legend key glyphs.
    labels : list of grob
        Legend label grobs.
    n_breaks : int
        Number of legend breaks.
    nrow, ncol : int
        Legend grid dimensions.
    key_width_cm, key_height_cm : float
        Default key dimensions in cm.
    spacing_x, spacing_y : float
        Gap between columns / rows in cm.
    text_position : str
        Where labels go relative to keys: "right", "left", "top", "bottom".
    byrow : bool
        Fill matrix by row?

    Returns
    -------
    dict
        ``{"widths": [...], "heights": [...]}`` in cm.
    """
    # Pad to fill the nrow x ncol matrix
    pad = nrow * ncol - n_breaks

    # Key sizes: read dynamic _width/_height from decor grobs (set by set_key_size
    # in build_legend_decor), then take column-max / row-max.
    # Mirrors R's measure_legend_keys / get_key_size (guide-legend.R:595-624).
    key_w_per_entry = []
    key_h_per_entry = []
    for i in range(n_breaks):
        if i < len(decor):
            kw = getattr(decor[i], "_width", key_width_cm) or key_width_cm
            kh = getattr(decor[i], "_height", key_height_cm) or key_height_cm
        else:
            kw, kh = key_width_cm, key_height_cm
        key_w_per_entry.append(max(kw, key_width_cm))
        key_h_per_entry.append(max(kh, key_height_cm))
    # Pad with zeros
    key_w_per_entry.extend([0.0] * pad)
    key_h_per_entry.extend([0.0] * pad)

    # Arrange into matrix and take column-max / row-max
    if byrow:
        kw_matrix = _fill_matrix(key_w_per_entry, nrow, ncol, byrow=True)
        kh_matrix = _fill_matrix(key_h_per_entry, nrow, ncol, byrow=True)
    else:
        kw_matrix = _fill_matrix(key_w_per_entry, nrow, ncol, byrow=False)
        kh_matrix = _fill_matrix(key_h_per_entry, nrow, ncol, byrow=False)

    key_widths = [max(kw_matrix[r][c] for r in range(nrow)) for c in range(ncol)]
    key_heights = [max(kh_matrix[r][c] for c in range(ncol)) for r in range(nrow)]

    # Label sizes: measure with font metrics (R: width_cm(grobs$labels))
    label_w_per_entry = []
    for i, lab_grob in enumerate(labels):
        label_text = ""
        if hasattr(lab_grob, "label"):
            label_text = str(lab_grob.label)
        elif hasattr(lab_grob, "_label"):
            label_text = str(lab_grob._label)
        label_w_per_entry.append(_text_width_cm(label_text, fontsize=label_size)
                                 if label_text else 0.3)

    # Pad label widths
    label_w_per_entry.extend([0.0] * pad)

    label_h_per_entry = [key_height_cm] * n_breaks + [0.0] * pad

    # Arrange into matrix and take column-max / row-max
    if byrow:
        # Fill by row
        label_w_matrix = _fill_matrix(label_w_per_entry, nrow, ncol, byrow=True)
        label_h_matrix = _fill_matrix(label_h_per_entry, nrow, ncol, byrow=True)
    else:
        label_w_matrix = _fill_matrix(label_w_per_entry, nrow, ncol, byrow=False)
        label_h_matrix = _fill_matrix(label_h_per_entry, nrow, ncol, byrow=False)

    label_widths = [max(label_w_matrix[r][c] for r in range(nrow))
                    for c in range(ncol)]
    label_heights = [max(label_h_matrix[r][c] for c in range(ncol))
                     for r in range(nrow)]

    # Interleave widths: [key_w, label_w, hgap] per column, strip last hgap
    if text_position == "right":
        width_lists = _interleave(key_widths, label_widths, spacing_x)
    elif text_position == "left":
        width_lists = _interleave(label_widths, key_widths, spacing_x)
    else:
        # top/bottom: labels and keys share same column
        width_lists = _interleave(
            [max(kw, lw) for kw, lw in zip(key_widths, label_widths)],
            None, spacing_x)

    # Interleave heights: [key_h, vgap] per row, strip last vgap
    if text_position == "top":
        height_lists = _interleave(label_heights, key_heights, spacing_y)
    elif text_position == "bottom":
        height_lists = _interleave(key_heights, label_heights, spacing_y)
    else:
        # left/right: labels and keys share same row
        height_lists = _interleave(
            [max(kh, lh) for kh, lh in zip(key_heights, label_heights)],
            None, spacing_y)

    return {"widths": width_lists, "heights": height_lists}


# ---------------------------------------------------------------------------
# arrange_legend_layout
# ---------------------------------------------------------------------------

def arrange_legend_layout(
    n_breaks: int,
    nrow: int,
    ncol: int,
    text_position: str = "right",
    byrow: bool = False,
) -> Dict[str, List[int]]:
    """Compute cell positions for keys and labels in the legend gtable.

    Mirrors ``GuideLegend$arrange_layout`` (guide-legend.R:503-531).

    Parameters
    ----------
    n_breaks, nrow, ncol : int
        Number of breaks and legend grid dimensions.
    text_position : str
        "right", "left", "top", or "bottom".
    byrow : bool
        Fill by row?

    Returns
    -------
    dict
        ``{"key_row": [...], "key_col": [...],
           "label_row": [...], "label_col": [...]}``
        1-based indices into the gtable.
    """
    break_seq = list(range(1, n_breaks + 1))

    if byrow:
        row = [math.ceil(b / ncol) for b in break_seq]
        col = [((b - 1) % ncol) + 1 for b in break_seq]
    else:
        row = [((b - 1) % nrow) + 1 for b in break_seq]
        col = [math.ceil(b / nrow) for b in break_seq]

    # Account for spacing rows/cols in between keys (every other row/col is a gap)
    key_row = [r * 2 - 1 for r in row]
    key_col = [c * 2 - 1 for c in col]

    # Offset for key-label gaps depending on text_position
    if text_position == "right":
        key_col = [kc + (c - 1) for kc, c in zip(key_col, col)]
        lab_col = [kc + 1 for kc in key_col]
        lab_row = list(key_row)
    elif text_position == "left":
        key_col = [kc + c for kc, c in zip(key_col, col)]
        lab_col = [kc - 1 for kc in key_col]
        lab_row = list(key_row)
    elif text_position == "top":
        key_row = [kr + r for kr, r in zip(key_row, row)]
        lab_row = [kr - 1 for kr in key_row]
        lab_col = list(key_col)
    elif text_position == "bottom":
        key_row = [kr + (r - 1) for kr, r in zip(key_row, row)]
        lab_row = [kr + 1 for kr in key_row]
        lab_col = list(key_col)
    else:
        # Default to "right"
        key_col = [kc + (c - 1) for kc, c in zip(key_col, col)]
        lab_col = [kc + 1 for kc in key_col]
        lab_row = list(key_row)

    return {
        "key_row": key_row,
        "key_col": key_col,
        "label_row": lab_row,
        "label_col": lab_col,
    }


# ---------------------------------------------------------------------------
# assemble_legend
# ---------------------------------------------------------------------------

def assemble_legend(
    decor: List[Any],
    labels: List[Any],
    title_grob: Any,
    layout: Dict[str, List[int]],
    sizes: Dict[str, List[float]],
    title_position: str = "top",
    padding_cm: float = _DEFAULT_PADDING_CM,
    bg_colour: Optional[str] = "white",
) -> Gtable:
    """Assemble a complete legend as a Gtable.

    Mirrors ``GuideLegend$assemble_drawing`` (guide-legend.R:533-591)
    plus ``Guide$add_title`` (guide-.R:924-951).

    Parameters
    ----------
    decor : list of grob
        Key glyphs from :func:`build_legend_decor`.
    labels : list of grob
        Label grobs from :func:`build_legend_labels`.
    title_grob : grob
        Legend title grob.
    layout : dict
        Cell positions from :func:`arrange_legend_layout`.
    sizes : dict
        Widths/heights from :func:`measure_legend_grobs`.
    title_position : str
        Where to place the title: "top", "right", "bottom", "left".
    padding_cm : float
        Padding around the legend in cm.
    bg_colour : str or None
        Background fill colour.

    Returns
    -------
    Gtable
        Self-contained legend gtable.
    """
    widths = Unit(sizes["widths"], "cm")
    heights = Unit(sizes["heights"], "cm")

    gt = Gtable(widths=widths, heights=heights, name="legend")

    # Add key glyphs
    if decor:
        for idx, grob in enumerate(decor):
            kr = layout["key_row"][idx]
            kc = layout["key_col"][idx]
            gt = gtable_add_grob(
                gt, grob,
                t=kr, l=kc, b=kr, r=kc,
                clip="off",
                name=f"key-{kr}-{kc}",
            )

    # Add labels
    if labels:
        for idx, grob in enumerate(labels):
            lr = layout["label_row"][idx]
            lc = layout["label_col"][idx]
            gt = gtable_add_grob(
                gt, grob,
                t=lr, l=lc, b=lr, r=lc,
                clip="off",
                name=f"label-{lr}-{lc}",
            )

    # Add title  (mirrors Guide$add_title, guide-.R:924-951)
    gt = add_legend_title(gt, title_grob, title_position)

    # Add padding  (mirrors gtable_add_padding)
    pad = Unit([padding_cm] * 4, "cm")
    gt = gtable_add_padding(gt, pad)

    # Add background
    if bg_colour is not None:
        bg = rect_grob(
            gp=Gpar(fill=bg_colour, col="grey85", lwd=0.5),
            name="legend.background",
        )
        nrow_gt = gt.nrow
        ncol_gt = gt.ncol
        gt = gtable_add_grob(
            gt, bg,
            t=1, l=1, b=nrow_gt, r=ncol_gt,
            z=-math.inf,
            clip="off",
            name="background",
        )

    return gt


# ---------------------------------------------------------------------------
# add_legend_title
# ---------------------------------------------------------------------------

def add_legend_title(
    gt: Gtable,
    title_grob: Any,
    position: str = "top",
) -> Gtable:
    """Add a title to a legend gtable.

    Mirrors ``Guide$add_title`` (guide-.R:924-951).

    Parameters
    ----------
    gt : Gtable
        Legend gtable under construction.
    title_grob : grob
        Title grob.
    position : str
        "top", "right", "bottom", or "left".

    Returns
    -------
    Gtable
        With title added.
    """
    if title_grob is None:
        return gt

    # R (guide-.R:924-951): cell size = ``grobHeight(title)`` /
    # ``grobWidth(title)``.  When ``title_grob`` is a ``_TitleGrob``
    # (produced by ``element_render(legend.title)``), ``grob_height``
    # returns a lazy unit covering text + margin, which grid_py
    # resolves correctly at draw time.  For a bare text_grob without
    # margins, ``grob_height`` still gives the glyph box.
    from grid_py import grob_height as _gh, grob_width as _gw

    if position == "top":
        gt = gtable_add_rows(gt, _gh(title_grob), pos=0)
        gt = gtable_add_grob(
            gt, title_grob,
            t=1, l=1, r=gt.ncol, b=1,
            z=-math.inf, clip="off", name="title",
        )
    elif position == "bottom":
        gt = gtable_add_rows(gt, _gh(title_grob), pos=-1)
        gt = gtable_add_grob(
            gt, title_grob,
            t=gt.nrow, l=1, r=gt.ncol, b=gt.nrow,
            z=-math.inf, clip="off", name="title",
        )
    elif position == "left":
        gt = gtable_add_cols(gt, _gw(title_grob), pos=0)
        gt = gtable_add_grob(
            gt, title_grob,
            t=1, l=1, r=1, b=gt.nrow,
            z=-math.inf, clip="off", name="title",
        )
    elif position == "right":
        gt = gtable_add_cols(gt, _gw(title_grob), pos=-1)
        gt = gtable_add_grob(
            gt, title_grob,
            t=1, l=gt.ncol, r=gt.ncol, b=gt.nrow,
            z=-math.inf, clip="off", name="title",
        )

    return gt


# ---------------------------------------------------------------------------
# package_legend_box
# ---------------------------------------------------------------------------

def package_legend_box(
    legends: List[Gtable],
    position: str = "right",
    spacing_cm: float = 0.2,
) -> Gtable:
    """Combine multiple legends into a single guide-box Gtable.

    Mirrors ``Guides$package_box`` (guides-.R:592-757).

    For ``position="right"`` or ``"left"`` (vertical), legends are stacked
    in a ``gtable_col``.  For ``"top"`` / ``"bottom"`` (horizontal), they
    are placed side by side in a ``gtable_row``.

    Parameters
    ----------
    legends : list of Gtable
        Individual legend gtables.
    position : str
        Legend box position relative to the plot.
    spacing_cm : float
        Spacing between legends in cm.

    Returns
    -------
    Gtable
        Combined guide-box.
    """
    if not legends:
        return Gtable(name="guide-box")

    if len(legends) == 1:
        legends[0].name = "guide-box"
        return legends[0]

    direction = "horizontal" if position in ("top", "bottom") else "vertical"

    if direction == "vertical":
        # Stack vertically
        # Compute common width = max of all legends
        max_width_cm = 0.0
        heights_cm = []
        for lg in legends:
            w = _gtable_total_cm(lg.widths)
            h = _gtable_total_cm(lg.heights)
            max_width_cm = max(max_width_cm, w)
            heights_cm.append(h)

        guides = gtable_col(
            name="guides",
            grobs=legends,
            width=Unit(max_width_cm, "cm"),
            heights=Unit(heights_cm, "cm"),
        )
        guides = gtable_add_row_space(guides, Unit(spacing_cm, "cm"))
    else:
        # Place side by side
        max_height_cm = 0.0
        widths_cm = []
        for lg in legends:
            w = _gtable_total_cm(lg.widths)
            h = _gtable_total_cm(lg.heights)
            max_height_cm = max(max_height_cm, h)
            widths_cm.append(w)

        from gtable_py import gtable_row, gtable_add_col_space
        guides = gtable_row(
            name="guides",
            grobs=legends,
            height=Unit(max_height_cm, "cm"),
            widths=Unit(widths_cm, "cm"),
        )
        guides = gtable_add_col_space(guides, Unit(spacing_cm, "cm"))

    guides.name = "guide-box"
    return guides


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_matrix(
    values: List[float], nrow: int, ncol: int, byrow: bool = False
) -> List[List[float]]:
    """Fill a flat list into a nrow x ncol matrix.

    Parameters
    ----------
    values : list of float
        Flat values (length >= nrow * ncol).
    nrow, ncol : int
        Matrix dimensions.
    byrow : bool
        Fill by row if True, else by column (R default).

    Returns
    -------
    list of list of float
        ``matrix[row][col]``.
    """
    matrix = [[0.0] * ncol for _ in range(nrow)]
    for idx, val in enumerate(values[: nrow * ncol]):
        if byrow:
            r = idx // ncol
            c = idx % ncol
        else:
            r = idx % nrow
            c = idx // nrow
        matrix[r][c] = val
    return matrix


def _interleave(
    a: List[float],
    b: Optional[List[float]],
    gap: float,
) -> List[float]:
    """Interleave two lists with a gap, stripping the trailing gap.

    If *b* is ``None``, just interleave *a* with *gap*.

    Examples
    --------
    >>> _interleave([1, 2], [3, 4], 0.1)
    [1, 3, 0.1, 2, 4, 0.1]  # then strip last → [1, 3, 0.1, 2, 4]
    """
    result: List[float] = []
    if b is not None:
        for i in range(len(a)):
            result.append(a[i])
            result.append(b[i] if i < len(b) else 0.0)
            result.append(gap)
    else:
        for i in range(len(a)):
            result.append(a[i])
            result.append(gap)

    # Strip trailing gap
    if result and result[-1] == gap:
        result = result[:-1]
    return result


def _gtable_total_cm(unit: Optional[Unit]) -> float:
    """Sum a Unit vector, returning cm as a float.

    Falls back to simple sum of values for "cm" units; returns a
    reasonable estimate for mixed/null units.
    """
    if unit is None or len(unit) == 0:
        return 0.0
    total = 0.0
    for i in range(len(unit)):
        part = unit[i: i + 1]
        vals = part.values if hasattr(part, "values") else [0.0]
        units = part.units if hasattr(part, "units") else ["cm"]
        v = vals[0] if vals else 0.0
        u = units[0] if units else "cm"
        if u == "cm":
            total += v
        elif u == "mm":
            total += v / 10.0
        elif u == "inches":
            total += v * 2.54
        elif u == "pt" or u == "points":
            total += v / 72.27 * 2.54
        else:
            # null, npc, etc. — use the numeric value as a rough estimate
            total += v
    return total

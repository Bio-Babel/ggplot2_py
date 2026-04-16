"""
Axis guide rendering — faithful port of R's GuideAxis.

Builds axis grobs (line, ticks, labels) as a **gtable** so that
``gtable_width()`` / ``gtable_height()`` return the correct measured
dimensions, eliminating hardcoded cm values and manual arithmetic.

R references
------------
* ``ggplot2/R/guide-axis.R``  — GuideAxis class + draw_axis helper
* ``ggplot2/R/guide-.R``      — Guide$build_ticks base method
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grid_py import (
    GList,
    GTree,
    Gpar,
    Unit,
    Viewport,
    null_grob,
    segments_grob,
    text_grob,
    unit_c,
    grob_height,
    grob_width,
)
from grid_py._grob import grob_tree

from gtable_py import (
    Gtable,
    gtable_add_cols,
    gtable_add_grob,
    gtable_add_rows,
    gtable_height,
    gtable_width,
)

__all__ = ["draw_axis"]


# ---------------------------------------------------------------------------
# Unit conversion constant (R: .pt = 72.27 / 25.4, mm → points)
# ---------------------------------------------------------------------------
_PT: float = 72.27 / 25.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_to_cm(u: Unit) -> float:
    """Convert a Unit (possibly compound/sum) to cm.

    ``convert_height/width`` can't handle compound ``"sum"`` units
    without a viewport context.  This helper decomposes them into
    leaf components, converts each individually, and sums the results.

    Sum unit structure (from ``Unit.__add__``)::

        _units  = ['sum']
        _values = [1.0]
        _data   = [Unit([2.75, 0.42], ['points', 'cm'])]

    The ``data`` element is itself a multi-element Unit with the
    individual operands.
    """
    from grid_py import convert_height

    units = getattr(u, "_units", None)
    values = getattr(u, "_values", None)
    data = getattr(u, "_data", None)
    if units is None or values is None:
        return 0.0

    total_cm = 0.0
    n = len(u)
    for i in range(n):
        unit_type = units[i] if i < len(units) else "cm"

        if unit_type == "sum":
            # The operands are stored as a multi-element Unit in data[i]
            inner = data[i] if data and i < len(data) else None
            if inner is not None and isinstance(inner, Unit):
                total_cm += _unit_to_cm(inner)
            continue

        # Skip context-dependent units we can't resolve statically
        if unit_type in ("npc", "native", "null", "grobwidth", "grobheight",
                         "strwidth", "strheight"):
            continue

        val = float(values[i]) if i < len(values) else 0.0
        leaf = Unit(val, unit_type)
        try:
            cm = convert_height(leaf, "cm", valueOnly=True)
            total_cm += float(np.sum(cm))
        except Exception:
            pass

    return total_cm


def _has_sum_unit(u: Unit) -> bool:
    """Check if a Unit contains any ``"sum"`` type components."""
    units = getattr(u, "_units", None)
    return units is not None and "sum" in units


def _width_cm(x: Any) -> float:
    """Measure a grob or unit width in cm (R: utilities-grid.R:67-76)."""
    from grid_py import convert_width
    if hasattr(x, "width_details") and callable(x.width_details):
        u = x.width_details()
    elif isinstance(x, Unit):
        u = x
    else:
        return 0.0
    # For compound (sum) units, convert_width returns bogus results;
    # decompose and convert leaf-by-leaf instead.
    if _has_sum_unit(u):
        return _unit_to_cm(u)
    try:
        result = convert_width(u, "cm", valueOnly=True)
        return float(np.sum(result))
    except Exception:
        return _unit_to_cm(u)


def _height_cm(x: Any) -> float:
    """Measure a grob or unit height in cm (R: utilities-grid.R:78-88)."""
    from grid_py import convert_height
    if hasattr(x, "height_details") and callable(x.height_details):
        u = x.height_details()
    elif isinstance(x, Unit):
        u = x
    else:
        return 0.0
    if _has_sum_unit(u):
        return _unit_to_cm(u)
    try:
        result = convert_height(u, "cm", valueOnly=True)
        return float(np.sum(result))
    except Exception:
        return _unit_to_cm(u)


# ---------------------------------------------------------------------------
# draw_axis — main entry point (mirrors R's draw_axis, guide-axis.R:508-529)
# ---------------------------------------------------------------------------

def draw_axis(
    break_positions: Any,
    break_labels: List[str],
    axis_position: str,
    theme: Any,
    check_overlap: bool = False,
    angle: Optional[float] = None,
    n_dodge: int = 1,
    minor_ticks: bool = False,
    minor_positions: Optional[Any] = None,
    cap: str = "none",
) -> Any:
    """Build a complete axis grob as a **gtable**.

    Mirrors R's ``draw_axis()`` (guide-axis.R:508-529) and the
    ``GuideAxis$assemble_drawing()`` method (guide-axis.R:420-474)
    which constructs a properly-measured gtable with tick, label,
    and title components.

    Parameters
    ----------
    break_positions : array-like
        Major break positions in [0, 1] NPC.
    break_labels : list of str
        Labels for each major break.
    axis_position : str
        One of ``"top"``, ``"bottom"``, ``"left"``, ``"right"``.
    theme : Theme
        Plot theme.
    check_overlap : bool
        Silently remove overlapping labels.
    angle : float or None
        Label rotation angle in degrees.
    n_dodge : int
        Number of rows/columns for dodging labels.
    minor_ticks : bool
        Whether to draw minor tick marks.
    minor_positions : array-like or None
        Minor break positions in [0, 1] NPC.
    cap : str
        Axis line cap style: ``"none"``, ``"both"``, ``"upper"``, ``"lower"``.

    Returns
    -------
    Gtable
        An axis gtable containing line, ticks, and labels, with
        proper widths/heights for measurement via ``gtable_width()``
        / ``gtable_height()``.
    """
    from ggplot2_py.theme_elements import calc_element, element_render, _PT

    breaks = np.asarray(break_positions, dtype=float) if break_positions is not None else np.array([])
    if len(breaks) == 0:
        return null_grob()

    if len(break_labels) != len(breaks):
        break_labels = [str(round(b, 2)) for b in breaks]

    # --- Setup params (R: GuideAxis$setup_params, lines 275-306) ----------
    is_horizontal = axis_position in ("top", "bottom")
    is_vertical = not is_horizontal
    aes = "x" if is_horizontal else "y"
    orth_aes = "y" if is_horizontal else "x"
    is_secondary = axis_position in ("top", "right")
    opposite = {"top": "bottom", "bottom": "top",
                "left": "right", "right": "left"}[axis_position]
    orth_side = 0.0 if is_secondary else 1.0
    lab_first = axis_position in ("top", "left")

    # --- Resolve theme elements ----------------------------------------
    # Use calc_element for proper inheritance resolution.
    line_el = _resolve_el(f"axis.line.{aes}", theme,
                          fallback={"colour": "grey20", "linewidth": 0.5, "linetype": 1})
    tick_el = _resolve_el(f"axis.ticks.{aes}", theme,
                          fallback={"colour": "grey20", "linewidth": 0.5})
    text_el = _resolve_el(f"axis.text.{aes}", theme,
                          fallback={"colour": "grey30", "size": 8, "angle": 0,
                                    "hjust": None, "vjust": None})

    # Tick length from theme (R: elements$major_length / minor_length)
    # R theme default: axis.ticks.length = unit(2.75, "pt")
    tick_length = _resolve_tick_length(theme, aes)
    minor_tick_length = tick_length * 0.5  # R: axis.minor.ticks.length = rel(0.75)

    # --- Build axis line (R: GuideAxis$build_decor, lines 313-322) -----
    if cap == "none" or len(breaks) == 0:
        line_start, line_end = 0.0, 1.0
    else:
        line_start = min(breaks) if cap in ("both", "lower") else 0.0
        line_end = max(breaks) if cap in ("both", "upper") else 1.0

    line_lwd = float(line_el.get("linewidth", 0.5)) * _PT
    if is_horizontal:
        axis_line = segments_grob(
            x0=[line_start], y0=[orth_side],
            x1=[line_end], y1=[orth_side],
            gp=Gpar(col=line_el.get("colour", "grey20"), lwd=line_lwd),
            name="axis.line",
        )
    else:
        axis_line = segments_grob(
            x0=[orth_side], y0=[line_start],
            x1=[orth_side], y1=[line_end],
            gp=Gpar(col=line_el.get("colour", "grey20"), lwd=line_lwd),
            name="axis.line",
        )

    # --- Build tick marks (R: GuideAxis$build_ticks, lines 324-342) ----
    tick_sign = -1.0 if axis_position in ("bottom", "left") else 1.0
    tick_lwd = float(tick_el.get("linewidth", 0.5)) * _PT
    tick_col = tick_el.get("colour", "grey20")

    if is_horizontal:
        major_ticks = segments_grob(
            x0=breaks.tolist(),
            y0=[orth_side] * len(breaks),
            x1=breaks.tolist(),
            y1=[orth_side + tick_sign * tick_length] * len(breaks),
            gp=Gpar(col=tick_col, lwd=tick_lwd),
            name="axis.ticks.major",
        )
    else:
        major_ticks = segments_grob(
            x0=[orth_side] * len(breaks),
            y0=breaks.tolist(),
            x1=[orth_side + tick_sign * tick_length] * len(breaks),
            y1=breaks.tolist(),
            gp=Gpar(col=tick_col, lwd=tick_lwd),
            name="axis.ticks.major",
        )

    ticks_grob = major_ticks

    # Minor ticks (R: lines 332-341)
    if minor_ticks and minor_positions is not None:
        minor_pos = np.asarray(minor_positions, dtype=float)
        minor_pos = np.array([p for p in minor_pos if p not in breaks])
        if len(minor_pos) > 0:
            if is_horizontal:
                minor_grob = segments_grob(
                    x0=minor_pos.tolist(),
                    y0=[orth_side] * len(minor_pos),
                    x1=minor_pos.tolist(),
                    y1=[orth_side + tick_sign * minor_tick_length] * len(minor_pos),
                    gp=Gpar(col=tick_col, lwd=tick_lwd * 0.5),
                    name="axis.ticks.minor",
                )
            else:
                minor_grob = segments_grob(
                    x0=[orth_side] * len(minor_pos),
                    y0=minor_pos.tolist(),
                    x1=[orth_side + tick_sign * minor_tick_length] * len(minor_pos),
                    y1=minor_pos.tolist(),
                    gp=Gpar(col=tick_col, lwd=tick_lwd * 0.5),
                    name="axis.ticks.minor",
                )
            ticks_grob = grob_tree(major_ticks, minor_grob, name="axis.ticks")

    # --- Build labels (R: GuideAxis$build_labels + draw_axis_labels) ---
    # Route through element_render so hjust/vjust/margin/angle from the
    # axis.text element drive positioning — matching R guide-axis.R:531-553
    #   element_grob(element_text, <pos_dim>=breaks, margin_x/y=TRUE, label=...)
    # This ensures:
    #   * default x (vertical) / y (horizontal) come from ``rotate_just``
    #     (left-axis labels: x=1npc hjust=1 → right-aligned against tick)
    #   * the titleGrob margin offset is applied so labels sit slightly
    #     inside the cell edge (no clipping on the outside)
    fontsize = float(text_el.get("size", 8))

    # Override angle (R: override_elements, guide-axis.R:263-265)
    if angle is not None:
        rot = float(angle)
    elif text_el.get("angle") is not None and float(text_el["angle"]) != 0:
        rot = float(text_el["angle"])
    else:
        rot = 0.0

    # N-dodge: split labels across groups (R: guide-axis.R:359-371)
    label_grobs = []
    dodge_groups = [[] for _ in range(n_dodge)]
    for i in range(len(breaks)):
        dodge_groups[i % n_dodge].append(i)

    el_name = f"axis.text.{aes}.{axis_position}"
    for dodge_idx, indices in enumerate(dodge_groups):
        if not indices:
            continue

        dodge_breaks = [float(breaks[i]) for i in indices]
        dodge_labels = [str(break_labels[i]) for i in indices]

        render_kwargs: Dict[str, Any] = {
            "label": dodge_labels,
            "size": fontsize,
        }
        if angle is not None:
            render_kwargs["angle"] = rot

        if is_horizontal:
            render_kwargs["x"] = Unit(dodge_breaks, "npc")
            render_kwargs["margin_y"] = True
        else:
            render_kwargs["y"] = Unit(dodge_breaks, "npc")
            render_kwargs["margin_x"] = True

        grob = element_render(theme, el_name, **render_kwargs)
        # Wrap in a GTree so make_content can place it via the gtable layout
        label_grobs.append(GTree(
            children=GList(grob),
            name=f"axis.labels.{dodge_idx}",
        ))

    # --- Measure components (R: GuideAxis$measure_grobs, lines 373-402) -
    # R: labels <- unit(measure(grobs$labels), "cm")
    # R: measure = height_cm for horizontal, width_cm for vertical
    # R: the labels are titleGrobs with margin, so grobHeight includes margin.
    #
    # We measure using calc_string_metric + axis text margin from theme.
    from grid_py._size import calc_string_metric
    from ggplot2_py.theme_elements import calc_element as _calc_el

    label_gp = Gpar(fontsize=fontsize)
    max_label_w_in = 0.0
    max_label_h_in = 0.0
    for lbl in break_labels:
        m = calc_string_metric(str(lbl), label_gp)
        max_label_w_in = max(max_label_w_in, m["width"])
        # R grobHeight for text = ascent + descent (whole glyph box)
        max_label_h_in = max(max_label_h_in, m["ascent"] + m["descent"])

    # Font descent for titleGrob height adjustment (R: margins.R:115-132)
    descent_in = 0.0
    for lbl in break_labels:
        m = calc_string_metric(str(lbl), label_gp)
        descent_in = max(descent_in, m["descent"])

    # Account for rotation (R: margins.R:126-132)
    rad = math.radians(abs(rot) % 360) if rot != 0 else 0.0
    y_descent = abs(math.cos(rad)) * descent_in if rad != 0 else descent_in

    # Projected height/width for rotated text
    if is_horizontal and rot != 0:
        proj_h = (max_label_w_in * abs(math.sin(rad))
                  + max_label_h_in * abs(math.cos(rad)))
    else:
        proj_h = max_label_h_in

    # Add font descent (R: titleGrob adds this)
    proj_h += y_descent
    proj_h *= n_dodge  # multiple dodge rows

    # Add axis text margin from theme (R: axis.text.x/y has margin)
    text_margin_cm = 0.0
    text_theme_el = _calc_el(f"axis.text.{aes}", theme)
    if text_theme_el is not None:
        margin_obj = getattr(text_theme_el, "margin", None)
        if margin_obj is not None:
            from ggplot2_py.theme_elements import Margin
            if isinstance(margin_obj, Margin):
                # For horizontal: margin_y = TRUE, so add top + bottom margin
                # For vertical: margin_x = TRUE, so add left + right margin
                if is_horizontal:
                    # Convert pt to cm: 1pt = 1/72.27 inch = 1/72.27*2.54 cm
                    text_margin_cm = (margin_obj.t + margin_obj.b) / 72.27 * 2.54
                else:
                    text_margin_cm = (margin_obj.l + margin_obj.r) / 72.27 * 2.54

    # Convert measurements to Units for gtable construction
    tick_size = _resolve_tick_length_unit(theme, aes)

    if is_horizontal:
        label_size_cm = proj_h * 2.54 + text_margin_cm  # inches → cm + margin
    else:
        label_size_cm = max_label_w_in * 2.54 + text_margin_cm
    label_size = Unit(label_size_cm, "cm")

    # --- Assemble gtable (R: GuideAxis$assemble_drawing, lines 420-474) -
    # Build the orthogonal dimension sizes.
    # R: sizes = unit.c(tick_length, spacer, labels, title)
    # We omit the title (standalone axis has no title in this path).
    # Order: [ticks, labels] or [labels, ticks] depending on lab_first.
    if lab_first:
        sizes = unit_c(label_size, tick_size)
        tick_pos = 2  # 1-indexed: tick is in column/row 2
        label_pos = 1
    else:
        sizes = unit_c(tick_size, label_size)
        tick_pos = 1
        label_pos = 2

    # Create the gtable with proper dimensions
    if is_horizontal:
        # Horizontal axis: widths = 1npc (full panel), heights = sizes
        gt = Gtable(
            widths=Unit(1, "npc"),
            heights=sizes,
            name=f"axis-{axis_position}",
        )
        # Add ticks
        gt = gtable_add_grob(gt, ticks_grob,
                             t=tick_pos, l=1, clip="off",
                             name="axis.ticks")
        # Add label grobs (one per dodge level)
        for lg in label_grobs:
            gt = gtable_add_grob(gt, lg,
                                 t=label_pos, l=1, clip="off",
                                 name="axis.labels")
    else:
        # Vertical axis: widths = sizes, heights = 1npc (full panel)
        gt = Gtable(
            widths=sizes,
            heights=Unit(1, "npc"),
            name=f"axis-{axis_position}",
        )
        # Add ticks
        gt = gtable_add_grob(gt, ticks_grob,
                             t=1, l=tick_pos, clip="off",
                             name="axis.ticks")
        # Add label grobs
        for lg in label_grobs:
            gt = gtable_add_grob(gt, lg,
                                 t=1, l=label_pos, clip="off",
                                 name="axis.labels")

    # --- Create justification viewport (R: guide-axis.R:444-450) --------
    # The viewport positions the axis+line pair at the correct panel edge.
    # R attaches the vp to the ``absoluteGrob`` *wrapper* (not to the
    # inner gtable), so both the axis_line and the gtable share a single
    # transform.  Attaching to the inner gt here caused the axis to
    # overflow its cell and occlude xlab/ylab (observed during gallery
    # validation).
    if is_horizontal:
        vp = Viewport(
            y=Unit(orth_side, "npc"),
            height=gtable_height(gt),
            just=opposite,
        )
    else:
        vp = Viewport(
            x=Unit(orth_side, "npc"),
            width=gtable_width(gt),
            just=opposite,
        )

    # --- Wrap with axis line (R: absoluteGrob pattern, lines 468-473) ---
    # The axis line sits on top of the gtable in an absoluteGrob-like
    # wrapper that reports the gtable's dimensions and carries the vp.
    result = _AbsoluteAxisGrob(
        children=GList(axis_line, gt),
        width=gtable_width(gt),
        height=gtable_height(gt),
        name=f"axis-{axis_position}",
    )
    result.vp = vp

    # Backward compatibility: store _width_cm / _height_cm for callers
    # that haven't been updated yet.
    if is_horizontal:
        result._height_cm = _height_cm(gtable_height(gt))
        result._width_cm = None
    else:
        result._width_cm = _width_cm(gtable_width(gt))
        result._height_cm = None

    return result


# ---------------------------------------------------------------------------
# AbsoluteAxisGrob — equivalent of R's absoluteGrob
# ---------------------------------------------------------------------------

class _AbsoluteAxisGrob(GTree):
    """A GTree wrapper that reports fixed width/height for measurement.

    Mirrors R's ``absoluteGrob()`` (grid/R/grob.R) used by GuideAxis
    to wrap the axis line + gtable with known dimensions.
    """

    def __init__(self, children: GList, width: Unit, height: Unit,
                 name: str = "absolute") -> None:
        super().__init__(children=children, name=name)
        self._abs_width = width
        self._abs_height = height

    def width_details(self) -> Unit:
        return self._abs_width

    def height_details(self) -> Unit:
        return self._abs_height


# ---------------------------------------------------------------------------
# Theme element resolution helpers
# ---------------------------------------------------------------------------

def _resolve_el(element_name: str, theme: Any,
                fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a theme element to a dict of properties.

    Tries ``calc_element`` first; falls back to a static dict.
    """
    from ggplot2_py.theme_elements import calc_element
    el = calc_element(element_name, theme)
    if el is not None and not _is_blank(el):
        result = {}
        for key in fallback:
            val = getattr(el, key, None)
            if val is not None:
                result[key] = val
            else:
                result[key] = fallback[key]
        return result
    return dict(fallback)


def _is_blank(el: Any) -> bool:
    """Check if an element is ElementBlank."""
    return getattr(el, "__class__", None).__name__ == "ElementBlank"


def _resolve_tick_length(theme: Any, aes: str) -> float:
    """Get tick length in NPC units from theme.

    R: ``axis.ticks.length`` defaults to ``unit(2.75, "pt")``.
    Returns a float in NPC for positioning (approximate).
    """
    from ggplot2_py.theme_elements import calc_element
    # Try to get the theme's tick length
    el = None
    for name in [f"axis.ticks.length.{aes}", "axis.ticks.length"]:
        el = calc_element(name, theme)
        if el is not None:
            break
    if el is not None and isinstance(el, Unit):
        # Convert to approximate NPC (assuming ~400pt panel = ~14cm)
        try:
            from grid_py import convert_height
            cm_val = convert_height(el, "cm", valueOnly=True)
            return float(np.sum(cm_val)) / 14.0  # rough NPC
        except Exception:
            pass
    # Default: 2.75pt ≈ 0.097cm → ~0.007 of a 14cm panel
    # Use a sensible NPC default
    return 0.03


def _resolve_tick_length_unit(theme: Any, aes: str) -> Unit:
    """Get tick length as a proper Unit from theme.

    R: ``axis.ticks.length`` defaults to ``unit(2.75, "pt")``.
    """
    from ggplot2_py.theme_elements import calc_element
    for name in [f"axis.ticks.length.{aes}", "axis.ticks.length"]:
        el = calc_element(name, theme)
        if el is not None and isinstance(el, Unit):
            return el
    # Default: 2.75 pt (R's default)
    return Unit(2.75, "points")

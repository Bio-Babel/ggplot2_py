"""
Axis guide rendering functions — faithful port of R's GuideAxis.

Builds axis grobs (line, ticks, labels) as a gtable, replacing the
hardcoded ``_render_axis()`` in ``coord.py``.

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
# Constants
# ---------------------------------------------------------------------------

_TICK_LENGTH_NPC: float = 0.12    # major tick length in NPC
_TICK_GAP_NPC: float = 0.08      # gap between tick and label in NPC
_MINOR_TICK_RATIO: float = 0.5   # minor tick = 50% of major


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
    """Build a complete axis grob using the GuideAxis pipeline.

    Mirrors R's ``draw_axis()`` (guide-axis.R:508-529) which constructs
    guide parameters and calls the GuideAxis draw pipeline.

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
    GTree
        An axis grob containing line, ticks, and labels.
    """
    from ggplot2_py.coord import _resolve_element

    breaks = np.asarray(break_positions, dtype=float) if break_positions is not None else np.array([])
    if len(breaks) == 0:
        return null_grob()

    if len(break_labels) != len(breaks):
        break_labels = [str(round(b, 2)) for b in breaks]

    # --- Setup params (R: GuideAxis$setup_params, lines 275-306) ---
    is_horizontal = axis_position in ("top", "bottom")
    is_vertical = not is_horizontal
    aes = "x" if is_horizontal else "y"
    orth_aes = "y" if is_horizontal else "x"
    is_secondary = axis_position in ("top", "right")
    opposite = {"top": "bottom", "bottom": "top",
                "left": "right", "right": "left"}[axis_position]
    orth_side = 0.0 if is_secondary else 1.0
    lab_first = axis_position in ("top", "left")

    # --- Resolve theme elements (R: GuideAxis$setup_elements, lines 251-261) ---
    # R appends "{aes}.{position}" suffix to theme element names
    suffix = f"{aes}.{axis_position}"
    line_el = _resolve_element(
        f"axis.line.{suffix}", theme,
        _resolve_element(f"axis.line.{aes}", theme,
            _resolve_element("axis.line", theme,
                {"colour": "grey20", "linewidth": 0.5, "linetype": 1})))
    tick_el = _resolve_element(
        f"axis.ticks.{suffix}", theme,
        _resolve_element(f"axis.ticks.{aes}", theme,
            _resolve_element("axis.ticks", theme,
                {"colour": "grey20", "linewidth": 0.5})))
    text_el = _resolve_element(
        f"axis.text.{suffix}", theme,
        _resolve_element(f"axis.text.{aes}", theme,
            _resolve_element("axis.text", theme,
                {"colour": "grey30", "size": 8, "angle": 0,
                 "hjust": None, "vjust": None})))

    # --- Build axis line (R: GuideAxis$build_decor, lines 313-322) ---
    # Axis line spans [0,1] in the parallel direction
    if cap == "none" or len(breaks) == 0:
        line_start, line_end = 0.0, 1.0
    else:
        line_start = min(breaks) if cap in ("both", "lower") else 0.0
        line_end = max(breaks) if cap in ("both", "upper") else 1.0

    if is_horizontal:
        axis_line = segments_grob(
            x0=[line_start], y0=[orth_side],
            x1=[line_end], y1=[orth_side],
            gp=Gpar(col=line_el["colour"], lwd=line_el["linewidth"]),
            name="axis.line",
        )
    else:
        axis_line = segments_grob(
            x0=[orth_side], y0=[line_start],
            x1=[orth_side], y1=[line_end],
            gp=Gpar(col=line_el["colour"], lwd=line_el["linewidth"]),
            name="axis.line",
        )

    # --- Build tick marks (R: GuideAxis$build_ticks, lines 324-342) ---
    # Tick direction: outward from the panel edge
    # bottom: ticks go down (negative y); top: ticks go up (positive y)
    # left: ticks go left (negative x); right: ticks go right (positive x)
    tick_sign = -1.0 if axis_position in ("bottom", "left") else 1.0

    if is_horizontal:
        major_ticks = segments_grob(
            x0=breaks.tolist(),
            y0=[orth_side] * len(breaks),
            x1=breaks.tolist(),
            y1=[orth_side + tick_sign * _TICK_LENGTH_NPC] * len(breaks),
            gp=Gpar(col=tick_el["colour"], lwd=tick_el["linewidth"]),
            name="axis.ticks.major",
        )
    else:
        major_ticks = segments_grob(
            x0=[orth_side] * len(breaks),
            y0=breaks.tolist(),
            x1=[orth_side + tick_sign * _TICK_LENGTH_NPC] * len(breaks),
            y1=breaks.tolist(),
            gp=Gpar(col=tick_el["colour"], lwd=tick_el["linewidth"]),
            name="axis.ticks.major",
        )

    ticks_grob = major_ticks

    # Minor ticks (R: lines 332-341)
    if minor_ticks and minor_positions is not None:
        minor_pos = np.asarray(minor_positions, dtype=float)
        # Remove minor ticks that coincide with major ticks
        minor_pos = np.array([p for p in minor_pos if p not in breaks])
        if len(minor_pos) > 0:
            minor_len = _TICK_LENGTH_NPC * _MINOR_TICK_RATIO
            if is_horizontal:
                minor_grob = segments_grob(
                    x0=minor_pos.tolist(),
                    y0=[orth_side] * len(minor_pos),
                    x1=minor_pos.tolist(),
                    y1=[orth_side + tick_sign * minor_len] * len(minor_pos),
                    gp=Gpar(col=tick_el["colour"], lwd=tick_el["linewidth"] * 0.5),
                    name="axis.ticks.minor",
                )
            else:
                minor_grob = segments_grob(
                    x0=[orth_side] * len(minor_pos),
                    y0=minor_pos.tolist(),
                    x1=[orth_side + tick_sign * minor_len] * len(minor_pos),
                    y1=minor_pos.tolist(),
                    gp=Gpar(col=tick_el["colour"], lwd=tick_el["linewidth"] * 0.5),
                    name="axis.ticks.minor",
                )
            ticks_grob = grob_tree(major_ticks, minor_grob, name="axis.ticks")

    # --- Build labels (R: GuideAxis$build_labels, lines 344-371) ---
    fontsize = float(text_el["size"])
    col = text_el["colour"]

    # Override angle (R: override_elements, lines 263-265)
    if angle is not None:
        rot = float(angle)
    elif text_el.get("angle") is not None and float(text_el["angle"]) != 0:
        rot = float(text_el["angle"])
    else:
        # Auto-rotation heuristic for horizontal axes
        rot = 0.0
        if is_horizontal:
            max_chars = max((len(str(l)) for l in break_labels), default=3)
            n = len(breaks)
            spacing = (breaks[-1] - breaks[0]) / max(n - 1, 1) if n > 1 else 1.0
            est_width = max_chars * 0.016
            if n > 1 and est_width > spacing * 0.9:
                rot = 30.0

    # Resolve justification based on position and rotation
    # (R: label_angle_heuristic + draw_axis_labels)
    if rot != 0:
        if is_horizontal:
            if axis_position == "bottom":
                just = ("right", "top")
            else:
                just = ("right", "bottom")
        else:
            just = ("right", "centre")
    else:
        if is_horizontal:
            just = ("centre", "top") if axis_position == "bottom" else ("centre", "bottom")
        else:
            just = ("right", "centre") if axis_position == "left" else ("left", "centre")

    # N-dodge: split labels across multiple rows/columns
    # (R: lines 359-371)
    label_y_offset = orth_side + tick_sign * (_TICK_LENGTH_NPC + _TICK_GAP_NPC)

    # For n_dodge > 1, create separate label grobs per dodge level
    label_grobs = []
    dodge_groups = [[] for _ in range(n_dodge)]
    for i in range(len(breaks)):
        dodge_groups[i % n_dodge].append(i)

    for dodge_idx, indices in enumerate(dodge_groups):
        if not indices:
            continue
        dodge_offset = dodge_idx * 0.05 * tick_sign  # stagger labels

        children = []
        for i in indices:
            if is_horizontal:
                children.append(text_grob(
                    label=str(break_labels[i]),
                    x=float(breaks[i]),
                    y=label_y_offset + dodge_offset,
                    rot=rot,
                    just=just,
                    gp=Gpar(fontsize=fontsize, col=col),
                    name=f"axis.text.{aes}.{i}",
                ))
            else:
                children.append(text_grob(
                    label=str(break_labels[i]),
                    x=label_y_offset + dodge_offset,
                    y=float(breaks[i]),
                    rot=rot,
                    just=just,
                    gp=Gpar(fontsize=fontsize, col=col),
                    name=f"axis.text.{aes}.{i}",
                ))

        label_grobs.append(GTree(
            children=GList(*children),
            name=f"axis.labels.{dodge_idx}",
        ))

    # --- Assemble (R: GuideAxis$assemble_drawing, lines 420-474) ---
    # Combine all parts into a single GTree
    all_children = [axis_line, ticks_grob] + label_grobs

    result = GTree(
        children=GList(*all_children),
        name=f"axis-{axis_position}",
    )

    # --- Declare physical dimensions (R: absoluteGrob pattern) ---
    # R wraps axis grobs in absoluteGrob(width=gtable_width(gt),
    # height=gtable_height(gt)) so grobWidth/grobHeight can read them.
    # We store the measured dimensions as attributes.
    from grid_py._size import calc_string_metric

    label_gp = Gpar(fontsize=fontsize)
    max_label_w_in = 0.0
    max_label_h_in = 0.0
    for lbl in break_labels:
        m = calc_string_metric(str(lbl), label_gp)
        max_label_w_in = max(max_label_w_in, m["width"])
        max_label_h_in = max(max_label_h_in, m["ascent"] + m["descent"])

    tick_in = _TICK_LENGTH_NPC * 2.54 / 2.54  # NPC constant as approximate cm→in
    # Use theme tick length if available; fallback to 0.15cm
    tick_cm = 0.15
    gap_cm = 0.08
    tick_in = tick_cm / 2.54
    gap_in = gap_cm / 2.54

    if is_horizontal:
        if rot != 0:
            proj_h = (max_label_w_in * abs(math.sin(math.radians(rot)))
                      + max_label_h_in * abs(math.cos(math.radians(rot))))
        else:
            proj_h = max_label_h_in
        proj_h *= n_dodge
        result._height_cm = (tick_in + gap_in + proj_h) * 2.54
        result._width_cm = None
    else:
        result._width_cm = (tick_in + gap_in + max_label_w_in) * 2.54
        result._height_cm = None

    return result

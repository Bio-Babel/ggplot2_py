"""
Plot rendering functions — conversion from built plot to gtable.

Extracted from plot.py to match R's separation of
plot-build.R (build pipeline) from plot-render.R (rendering).

Contains:
- ggplot_gtable() — convert built plot to gtable
- _table_add_legends() — build legends from scales
- _table_add_titles() — add title/subtitle/caption
- ggplotGrob() — build + render convenience
- find_panel() / panel_rows() / panel_cols() — panel location
- print_plot() — render to device

R references
------------
* ggplot2/R/plot-render.R
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver

__all__ = [
    "ggplot_gtable",
    "ggplotGrob",
    "_safe_colour",
    "_table_add_legends",
    "_table_add_titles",
    "find_panel",
    "panel_rows",
    "panel_cols",
    "print_plot",
]


def _legend_label_width_cm(labels: List[Any], fontsize: float = 6.0) -> float:
    """Measure max label width in cm using Cairo font metrics.

    Replaces ``max(len(str(l)) for l in labels) * 0.18`` with actual
    text measurement, matching R's ``width_cm(grobs$labels)`` pattern.
    """
    from grid_py._size import calc_string_metric
    from grid_py import Gpar
    max_w = 0.0
    for l in labels:
        m = calc_string_metric(str(l), Gpar(fontsize=fontsize))
        max_w = max(max_w, m["width"] * 2.54)  # inches → cm
    return max(max_w, 0.3)  # minimum width 0.3 cm


@singledispatch
def ggplot_gtable(data: Any) -> Any:
    """Convert a built ggplot to a gtable for rendering.

    This is a :func:`functools.singledispatch` generic (R ref:
    ``plot-render.R:22``, ``UseMethod("ggplot_gtable")``).  Extension
    packages can register custom built-plot types::

        @ggplot_gtable.register(MyBuiltPlot)
        def _gtable_my_plot(data):
            ...

    Parameters
    ----------
    data : BuiltGGPlot
        Output from :func:`ggplot_build`.

    Returns
    -------
    gtable
        A gtable suitable for drawing with ``grid_draw()``.
    """
    raise TypeError(
        f"Cannot render object of type {type(data).__name__}. "
        "Expected a BuiltGGPlot instance."
    )


def _ggplot_gtable_impl(data):
    """Core ggplot_gtable implementation for BuiltGGPlot objects."""
    from gtable_py import (
        Gtable,
        gtable_add_grob,
        gtable_add_rows,
        gtable_add_cols,
        gtable_width,
        gtable_height,
    )
    from grid_py import null_grob

    plot = data.plot
    layout = data.layout
    layer_data = data.data
    theme = plot.theme
    labels = plot.labels

    # Draw geom grobs for each layer
    geom_grobs: List[Any] = []
    for i, layer in enumerate(plot.layers):
        if hasattr(layer, "draw_geom"):
            geom_grobs.append(layer.draw_geom(layer_data[i], layout))
        else:
            geom_grobs.append(null_grob())

    # Render panels via layout
    plot_table = layout.render(geom_grobs, layer_data, theme, labels)

    # Legends — build directly from trained non-position scales.
    plot_table = _table_add_legends(
        plot_table, plot.scales, labels, theme, layers=plot.layers,
    )

    # Title / subtitle / caption / tag annotations
    plot_table = _table_add_titles(plot_table, labels, theme)

    # Add plot margin (R: table_add_background, plot-render.R:342-345)
    # R: margin <- calc_element("plot.margin", theme) %||% margin()
    #    table  <- gtable_add_padding(table, margin)
    # Margin.unit preserves the original unit type (default "pt").
    if hasattr(plot_table, "_widths"):
        from gtable_py import gtable_add_padding
        from grid_py import Unit
        from ggplot2_py.theme_elements import Margin, ElementBlank
        try:
            from ggplot2_py.theme_elements import calc_element as _calc_el
            margin = _calc_el("plot.margin", theme)
            if margin is None or isinstance(margin, ElementBlank):
                margin = Margin(5.5, 5.5, 5.5, 5.5, unit="pt")
            elif not isinstance(margin, Margin):
                margin = Margin(5.5, 5.5, 5.5, 5.5, unit="pt")
            plot_table = gtable_add_padding(plot_table, margin.unit)
        except Exception:
            plot_table = gtable_add_padding(
                plot_table, Unit([0.2, 0.2, 0.2, 0.2], "cm"),
            )

    # Add alt-text attribute
    if hasattr(plot_table, "__dict__"):
        plot_table._alt_label = labels.get("alt", "")

    return plot_table


def _safe_colour(colour: Any) -> str:
    """Validate a colour value, returning 'grey50' for invalid inputs."""
    if colour is None:
        return "grey50"
    s = str(colour)
    if s.startswith("#") and len(s) in (7, 9):
        return s
    # Use matplotlib to validate named colours
    try:
        from matplotlib.colors import is_color_like
        if is_color_like(s):
            return s
    except (ImportError, ValueError):
        pass
    return "grey50"


def _table_add_legends(
    table: Any, scales_list: Any, labels: Dict[str, Any], theme: Any,
    layers: Any = None,
) -> Any:
    """Build legends from trained non-position scales and add to the gtable.

    Each legend is built as an independent :class:`~gtable_py.Gtable` with
    its own viewport-based cell layout, faithfully mirroring R's
    ``GuideLegend`` pipeline.  Scales sharing the same title and breaks
    are merged into a single legend (R's guide-merge semantics).

    Mirrors R's ``table_add_legends`` in ``plot-render.R`` and the
    ``GuideLegend`` class in ``guide-legend.R``.

    Parameters
    ----------
    table : gtable
    scales_list : ScalesList
    labels : dict
    theme : Theme
    layers : list of Layer, optional
        Plot layers — used to determine the ``draw_key`` function for each
        aesthetic.

    Returns
    -------
    gtable
    """
    if not hasattr(table, "_widths"):
        return table

    import math
    from gtable_py import gtable_add_grob, gtable_add_cols, gtable_width, gtable_height
    from grid_py import Unit as unit, text_grob, Gpar

    from ggplot2_py.guide_legend import (
        build_legend_decor,
        build_legend_labels,
        measure_legend_grobs,
        arrange_legend_layout,
        assemble_legend,
        package_legend_box,
    )

    # ------------------------------------------------------------------
    # 1. Collect raw legend entries from non-position scales
    # ------------------------------------------------------------------
    raw_entries: List[Dict[str, Any]] = []
    np_scales = (
        scales_list.non_position_scales()
        if hasattr(scales_list, "non_position_scales")
        else None
    )
    if np_scales is None or np_scales.n() == 0:
        return table

    for sc in np_scales.scales:
        aes_name = sc.aesthetics[0] if sc.aesthetics else "unknown"

        breaks = getattr(sc, "get_breaks", lambda: None)()
        if breaks is None or (hasattr(breaks, "__len__") and len(breaks) == 0):
            continue

        mapped = breaks
        if hasattr(sc, "map"):
            try:
                mapped = sc.map(breaks)
            except (TypeError, ValueError):
                pass

        if hasattr(sc, "get_labels"):
            try:
                labs = sc.get_labels(breaks)
            except (TypeError, ValueError, AttributeError):
                labs = [str(b) for b in breaks]
        else:
            labs = [str(b) for b in breaks]

        # Drop NA/NaN-mapped entries
        keep: List[int] = []
        mapped_arr = np.asarray(mapped) if not isinstance(mapped, np.ndarray) else mapped
        for j in range(len(breaks)):
            val = mapped_arr[j] if j < len(mapped_arr) else None
            try:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    keep.append(j)
            except (TypeError, ValueError):
                keep.append(j)
        if not keep:
            continue
        breaks = [breaks[j] for j in keep]
        mapped = [mapped_arr[j] for j in keep]
        labs = [labs[j] for j in keep if j < len(labs)]

        title = labels.get(aes_name, aes_name)
        if hasattr(title, "__class__") and title.__class__.__name__ == "Waiver":
            title = aes_name

        raw_entries.append({
            "aesthetic": aes_name,
            "breaks": breaks,
            "mapped": mapped,
            "labels": labs,
            "title": str(title),
            "scale": sc,
            "is_continuous": not getattr(sc, "is_discrete", lambda: True)(),
            "is_binned": sc.__class__.__name__.startswith("ScaleBinned") or
                         getattr(sc, "guide", None) in ("bins", "coloursteps"),
        })

    if not raw_entries:
        return table

    # ------------------------------------------------------------------
    # 2. Merge entries that share the same title + number of breaks
    #    (R merges guides whose hash — based on title and breaks — match)
    # ------------------------------------------------------------------
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in raw_entries:
        key = entry["title"]
        if key in merged and len(merged[key]["breaks"]) == len(entry["breaks"]):
            merged[key]["aes_mapped"][entry["aesthetic"]] = entry["mapped"]
        else:
            merged[key] = {
                "title": entry["title"],
                "breaks": entry["breaks"],
                "labels": entry["labels"],
                "aes_mapped": {entry["aesthetic"]: entry["mapped"]},
                "scale": entry.get("scale"),
                "is_continuous": entry.get("is_continuous", False),
                "is_binned": entry.get("is_binned", False),
            }
    entries = list(merged.values())

    # ------------------------------------------------------------------
    # 3. Resolve theme elements (R: calc_element for proper inheritance)
    #    R always has a complete theme.  If Python's theme is None or
    #    incomplete, reset the element tree and use theme_grey().
    # ------------------------------------------------------------------
    from ggplot2_py.theme_elements import calc_element as _calc_theme_el

    if theme is None:
        from ggplot2_py.theme_defaults import theme_grey
        theme = theme_grey()

    _ltitle_raw = _calc_theme_el("legend.title", theme)
    if _ltitle_raw is None:
        from ggplot2_py.theme_elements import reset_theme_settings
        reset_theme_settings()
        from ggplot2_py.theme_defaults import theme_grey as _tg
        theme = _tg()
        _ltitle_raw = _calc_theme_el("legend.title", theme)
    _ltext_raw = _calc_theme_el("legend.text", theme)

    title_size = float(_ltitle_raw.size)
    label_size = float(_ltext_raw.size)
    _ltitle_colour = _ltitle_raw.colour
    _ltext_colour = _ltext_raw.colour

    # Resolve legend key dimensions from theme
    # (R: GuideLegend$override_elements → width_cm/height_cm of theme units)
    from ggplot2_py.theme_elements import calc_element as _calc_el
    from grid_py import Unit as _Unit, convert_width, convert_height

    def _unit_to_cm(u, axis="height"):
        """Convert a theme Unit to cm using grid's **device-default** gp.

        Mirrors R's ``convertUnit(u, "cm", valueOnly=TRUE)`` called at
        gtable-construction time (pre-draw, no viewport active): R's
        grid falls back to the device default gp (``fontsize=12``,
        ``lineheight=1.2``).  R's ggplot2 uses this device default —
        **not** the theme's ``text`` element — when computing static
        layout sizes such as ``legend.key.width``.  grid_py's
        ``convert_*`` with no active viewport reproduces the same
        behaviour, so we just call it directly.
        """
        if u is None or not isinstance(u, _Unit):
            return None
        fn = convert_height if axis == "height" else convert_width
        cm = fn(u, "cm", valueOnly=True)
        val = float(np.sum(cm))
        return val if val > 0 else None

    key_size = _unit_to_cm(_calc_el("legend.key.size", theme))
    key_w = _unit_to_cm(_calc_el("legend.key.width", theme), "width")
    key_h = _unit_to_cm(_calc_el("legend.key.height", theme))
    spacing_x = _unit_to_cm(_calc_el("legend.key.spacing.x", theme), "width")
    spacing_y = _unit_to_cm(_calc_el("legend.key.spacing.y", theme))
    legend_spacing = _unit_to_cm(_calc_el("legend.spacing", theme))

    KEY_W_CM = key_w or key_size
    KEY_H_CM = key_h or key_size
    SPACING_X_CM = spacing_x
    SPACING_Y_CM = spacing_y
    PADDING_CM = 0.15  # R: legend.margin default padding

    # ------------------------------------------------------------------
    # 4. Determine draw_key function from layers
    # ------------------------------------------------------------------
    from ggplot2_py.draw_key import draw_key_point as _draw_key_point
    draw_key_fn = _draw_key_point
    if layers:
        for layer in layers:
            geom = getattr(layer, "geom", None)
            if geom is not None and hasattr(geom, "draw_key"):
                draw_key_fn = geom.draw_key
                break

    # ------------------------------------------------------------------
    # 5. Build each guide as an independent Gtable
    #    Dispatch: continuous colour/fill → colourbar; else → legend
    # ------------------------------------------------------------------
    from ggplot2_py.guide_colourbar import (
        extract_colourbar_decor,
        extract_coloursteps_decor,
        build_colourbar_decor,
        build_coloursteps_decor,
        build_colourbar_labels,
        build_colourbar_ticks,
        assemble_colourbar,
    )

    legend_gtables = []

    for entry in entries:
        n_breaks = len(entry["breaks"])
        if n_breaks == 0:
            continue

        aes_names = list(entry["aes_mapped"].keys())
        is_colour_fill = any(a in ("colour", "color", "fill") for a in aes_names)
        is_continuous = entry.get("is_continuous", False)
        is_binned = entry.get("is_binned", False)
        sc = entry.get("scale")

        # --- Coloursteps path: binned colour/fill scale ---
        if is_colour_fill and is_binned and sc is not None:
            title_grob = text_grob(
                label=entry["title"],
                x=0.0, y=0.5,
                just=("left", "centre"),
                gp=Gpar(
                    fontsize=title_size,
                    col=_ltitle_colour,
                    fontface="bold",
                ),
                name=f"coloursteps.title.{entry['title']}",
            )

            # Extract stepped colour bins
            decor = extract_coloursteps_decor(
                sc, entry["breaks"], even_steps=True,
            )

            # Build stepped rectangle bar
            bar_parts = build_coloursteps_decor(decor, direction="vertical")

            # Labels and ticks (same as colourbar)
            limits = sc.get_limits()
            cb_labels = build_colourbar_labels(
                entry["breaks"], entry["labels"], limits,
                direction="vertical",
                label_size=label_size, label_colour=_ltext_colour,
            )
            ticks = build_colourbar_ticks(
                entry["breaks"], limits, direction="vertical",
            )

            label_w_cm = _legend_label_width_cm(entry["labels"], label_size)

            legend_gt = assemble_colourbar(
                bar_grob=bar_parts["bar"],
                frame_grob=bar_parts["frame"],
                ticks_grob=ticks,
                label_grobs=cb_labels,
                title_grob=title_grob,
                direction="vertical",
                bar_width_cm=KEY_W_CM,
                bar_height_cm=KEY_H_CM * 5,
                label_width_cm=label_w_cm,
                padding_cm=PADDING_CM,
                bg_colour="white",
            )
            legend_gtables.append(legend_gt)
            continue

        # --- Colourbar path: continuous colour/fill scale ---
        if is_colour_fill and is_continuous and sc is not None:
            # Title grob
            title_grob = text_grob(
                label=entry["title"],
                x=0.0, y=0.5,
                just=("left", "centre"),
                gp=Gpar(
                    fontsize=title_size,
                    col=_ltitle_colour,
                    fontface="bold",
                ),
                name=f"colourbar.title.{entry['title']}",
            )

            # Extract dense colour sequence
            decor = extract_colourbar_decor(sc, nbin=300)

            # Build bar grob (raster mode)
            bar_parts = build_colourbar_decor(decor, direction="vertical",
                                              display="raster")

            # Build tick labels
            limits = sc.get_limits()
            cb_labels = build_colourbar_labels(
                entry["breaks"], entry["labels"], limits,
                direction="vertical",
                label_size=label_size, label_colour=_ltext_colour,
            )

            # Build tick marks
            ticks = build_colourbar_ticks(
                entry["breaks"], limits, direction="vertical",
            )

            # Estimate label width
            label_w_cm = _legend_label_width_cm(entry["labels"], label_size)

            # Assemble
            legend_gt = assemble_colourbar(
                bar_grob=bar_parts["bar"],
                frame_grob=bar_parts["frame"],
                ticks_grob=ticks,
                label_grobs=cb_labels,
                title_grob=title_grob,
                direction="vertical",
                bar_width_cm=KEY_W_CM,
                bar_height_cm=KEY_H_CM * 5,
                label_width_cm=label_w_cm,
                padding_cm=PADDING_CM,
                bg_colour="white",
            )
            legend_gtables.append(legend_gt)
            continue

        # --- Legend path: discrete scales ---
        nrow = min(n_breaks, 20)
        ncol = 1

        decor = build_legend_decor(
            entry, draw_key_fn, layers,
            key_width_cm=KEY_W_CM, key_height_cm=KEY_H_CM,
        )

        label_grobs = build_legend_labels(
            entry, label_size=label_size, label_colour=_ltext_colour,
        )

        sizes = measure_legend_grobs(
            decor, label_grobs, n_breaks,
            nrow=nrow, ncol=ncol,
            key_width_cm=KEY_W_CM, key_height_cm=KEY_H_CM,
            spacing_x=SPACING_X_CM, spacing_y=SPACING_Y_CM,
            text_position="right",
            label_size=label_size,
        )

        layout = arrange_legend_layout(
            n_breaks, nrow=nrow, ncol=ncol,
            text_position="right",
        )

        # R (guide-legend.R): title is rendered via element_render so
        # grobHeight(title) includes the legend.title element's margin.
        # Using a bare text_grob here shrinks the cell below the
        # glyph box and clips the top of the text (e.g. "drv" cap).
        from ggplot2_py.theme_elements import element_render as _el_render
        title_grob = _el_render(
            theme, "legend.title",
            label=str(entry["title"]),
            margin_x=True, margin_y=True,
        )

        legend_gt = assemble_legend(
            decor, label_grobs, title_grob,
            layout, sizes,
            title_position="top",
            padding_cm=PADDING_CM,
            bg_colour="white",
        )
        legend_gtables.append(legend_gt)

    if not legend_gtables:
        return table

    # ------------------------------------------------------------------
    # 6. Package multiple legends into a guide-box
    # ------------------------------------------------------------------
    guide_box = package_legend_box(
        legend_gtables, position="right",
        spacing_cm=legend_spacing,
    )

    # ------------------------------------------------------------------
    # 7. Place guide-box in the plot table
    #    Mirrors R's table_add_legends (plot-render.R:98-105)
    # ------------------------------------------------------------------
    from ggplot2_py.guide_legend import _gtable_total_cm

    guide_w_cm = _gtable_total_cm(guide_box.widths)
    guide_w_cm = max(guide_w_cm, 1.0)

    # R: place <- find_panel(table); t=place$t, b=place$b  (plot-render.R:96-104)
    place = find_panel(table)

    table = gtable_add_cols(table, unit([legend_spacing], "cm"), pos=-1)
    table = gtable_add_cols(table, unit([guide_w_cm], "cm"), pos=-1)
    ncol_t = len(table._widths)
    table = gtable_add_grob(
        table, guide_box, t=place["t"], b=place["b"], l=ncol_t,
        clip="off", name="guide-box-right",
    )

    return table


def _table_add_titles(table: Any, labels: Dict[str, Any], theme: Any) -> Any:
    """Add title, subtitle, caption annotations to the plot table.

    Mirrors R's ``table_add_titles()`` / ``table_add_caption()`` in
    ``plot-render.R`` (lines 147-224):
      1. Render the text via ``element_render(theme, element_name, label, ...)``
      2. Measure actual rendered height via ``grob_height(grob)``
      3. Add a row of that measured height to the gtable

    Parameters
    ----------
    table : gtable
        The plot gtable.
    labels : dict
        Plot labels (``title``, ``subtitle``, ``caption``).
    theme : Theme
        Complete theme.

    Returns
    -------
    gtable
        Modified table.
    """
    from gtable_py import gtable_add_grob, gtable_add_rows
    from grid_py import grob_height
    from ggplot2_py.theme_elements import element_render, calc_element

    if not hasattr(table, "_widths"):
        return table

    ncol = len(table._widths)

    # --- Caption (bottom) --- (R: plot-render.R:193-224)
    caption = labels.get("caption")
    if caption:
        caption_grob = element_render(
            theme, "plot.caption", label=str(caption),
            margin_y=True, margin_x=True,
        )
        caption_height = grob_height(caption_grob)
        table = gtable_add_rows(table, caption_height, pos=-1)
        nrow = len(table._heights)
        table = gtable_add_grob(
            table, caption_grob,
            t=nrow, l=1, r=ncol, clip="off", name="caption",
        )

    # --- Subtitle (top, added first so title goes above) ---
    # (R: plot-render.R:157-161, 182-184)
    subtitle = labels.get("subtitle")
    if subtitle:
        subtitle_grob = element_render(
            theme, "plot.subtitle", label=str(subtitle),
            margin_y=True, margin_x=True,
        )
        subtitle_height = grob_height(subtitle_grob)
        table = gtable_add_rows(table, subtitle_height, pos=0)
        table = gtable_add_grob(
            table, subtitle_grob,
            t=1, l=1, r=ncol, clip="off", name="subtitle",
        )

    # --- Title (top) --- (R: plot-render.R:150-154, 186-188)
    title = labels.get("title")
    if title:
        title_grob = element_render(
            theme, "plot.title", label=str(title),
            margin_y=True, margin_x=True,
        )
        title_height = grob_height(title_grob)
        table = gtable_add_rows(table, title_height, pos=0)
        table = gtable_add_grob(
            table, title_grob,
            t=1, l=1, r=ncol, clip="off", name="title",
        )

    return table


# ---------------------------------------------------------------------------
# ggplotGrob
# ---------------------------------------------------------------------------

def ggplotGrob(plot: "GGPlot") -> Any:
    """Build and convert a ggplot to a gtable grob.

    Parameters
    ----------
    plot : GGPlot
        A ggplot object.

    Returns
    -------
    gtable
    """
    from ggplot2_py.plot import ggplot_build
    return ggplot_gtable(ggplot_build(plot))


def find_panel(table: Any) -> Dict[str, Any]:
    """Find the panel area in a gtable.

    Mirrors R's ``find_panel()`` in ``layout.R``.  Supports gtable layouts
    stored as either a ``pd.DataFrame`` or a plain dict-of-lists.

    Parameters
    ----------
    table : gtable
        A gtable object.

    Returns
    -------
    dict
        ``{"t": int, "l": int, "b": int, "r": int}`` panel bounds.
    """
    layout = getattr(table, "layout", None)
    if layout is None:
        return {"t": 1, "l": 1, "b": 1, "r": 1}

    # --- DataFrame path ---
    if isinstance(layout, pd.DataFrame):
        panel_rows = layout.loc[
            layout["name"].str.contains("panel", case=False, na=False)
        ]
        if not panel_rows.empty:
            return {
                "t": int(panel_rows["t"].min()),
                "l": int(panel_rows["l"].min()),
                "b": int(panel_rows["b"].max()),
                "r": int(panel_rows["r"].max()),
            }

    # --- dict-of-lists path (gtable_py stores layout this way) ---
    elif isinstance(layout, dict) and "name" in layout:
        names = layout["name"]
        indices = [i for i, n in enumerate(names)
                   if isinstance(n, str) and "panel" in n.lower()]
        if indices:
            return {
                "t": min(layout["t"][i] for i in indices),
                "l": min(layout["l"][i] for i in indices),
                "b": max(layout["b"][i] for i in indices),
                "r": max(layout["r"][i] for i in indices),
            }

    return {"t": 1, "l": 1, "b": 1, "r": 1}


def panel_rows(table: Any) -> Dict[str, int]:
    """Return the row range of panels in a gtable.

    Parameters
    ----------
    table : gtable

    Returns
    -------
    dict
        ``{"t": int, "b": int}``
    """
    p = find_panel(table)
    return {"t": p["t"], "b": p["b"]}


def panel_cols(table: Any) -> Dict[str, int]:
    """Return the column range of panels in a gtable.

    Parameters
    ----------
    table : gtable

    Returns
    -------
    dict
        ``{"l": int, "r": int}``
    """
    p = find_panel(table)
    return {"l": p["l"], "r": p["r"]}


# ---------------------------------------------------------------------------
# Matplotlib label helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# print_plot
# ---------------------------------------------------------------------------

def print_plot(
    plot: "GGPlot",
    newpage: bool = True,
    vp: Any = None,
) -> "GGPlot":
    """Render a ggplot to the current device.

    Parameters
    ----------
    plot : GGPlot
        The plot to display.
    newpage : bool, optional
        If ``True``, create a new page / figure first.
    vp : Viewport, optional
        Viewport to draw in.

    Returns
    -------
    GGPlot
        The original plot (invisibly).
    """
    from grid_py import grid_draw, grid_newpage
    from ggplot2_py.plot import ggplot_build, set_last_plot

    set_last_plot(plot)

    if newpage and vp is None:
        grid_newpage()

    built = ggplot_build(plot)
    gtable = ggplot_gtable(built)

    if vp is None:
        grid_draw(gtable)
    else:
        from grid_py import push_viewport, up_viewport
        push_viewport(vp)
        grid_draw(gtable)
        up_viewport()

    return plot


# ---------------------------------------------------------------------------
# Deferred singledispatch registration for ggplot_gtable
# ---------------------------------------------------------------------------

def _register_ggplot_gtable_types():
    """Register BuiltGGPlot for ggplot_gtable dispatch.

    Called from plot.py after BuiltGGPlot is defined.
    """
    from ggplot2_py.plot import BuiltGGPlot
    ggplot_gtable.register(BuiltGGPlot)(_ggplot_gtable_impl)


_register_ggplot_gtable_types()

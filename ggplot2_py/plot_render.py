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


def ggplot_gtable(data: "BuiltGGPlot") -> Any:
    """Convert a built ggplot to a gtable for rendering.

    Parameters
    ----------
    data : BuiltGGPlot
        Output from :func:`ggplot_build`.

    Returns
    -------
    gtable
        A gtable suitable for drawing with ``grid_draw()``.
    """
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
    # 3. Resolve theme elements
    # ------------------------------------------------------------------
    from ggplot2_py.coord import _resolve_element

    ltitle_el = _resolve_element("legend.title", theme,
        {"colour": "grey10", "size": 7})
    ltext_el = _resolve_element("legend.text", theme,
        {"colour": "grey20", "size": 6})

    title_size = float(ltitle_el["size"])
    label_size = float(ltext_el["size"])

    KEY_W_CM = 0.5
    KEY_H_CM = 0.5
    SPACING_X_CM = 0.15
    SPACING_Y_CM = 0.0
    PADDING_CM = 0.15

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
                    col=ltitle_el["colour"],
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
                label_size=label_size, label_colour=ltext_el["colour"],
            )
            ticks = build_colourbar_ticks(
                entry["breaks"], limits, direction="vertical",
            )

            max_lab_len = max((len(str(l)) for l in entry["labels"]), default=3)
            label_w_cm = max(max_lab_len * 0.18, 0.5)

            legend_gt = assemble_colourbar(
                bar_grob=bar_parts["bar"],
                frame_grob=bar_parts["frame"],
                ticks_grob=ticks,
                label_grobs=cb_labels,
                title_grob=title_grob,
                direction="vertical",
                bar_width_cm=0.5,
                bar_height_cm=3.0,
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
                    col=ltitle_el["colour"],
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
                label_size=label_size, label_colour=ltext_el["colour"],
            )

            # Build tick marks
            ticks = build_colourbar_ticks(
                entry["breaks"], limits, direction="vertical",
            )

            # Estimate label width
            max_lab_len = max((len(str(l)) for l in entry["labels"]), default=3)
            label_w_cm = max(max_lab_len * 0.18, 0.5)

            # Assemble
            legend_gt = assemble_colourbar(
                bar_grob=bar_parts["bar"],
                frame_grob=bar_parts["frame"],
                ticks_grob=ticks,
                label_grobs=cb_labels,
                title_grob=title_grob,
                direction="vertical",
                bar_width_cm=0.5,
                bar_height_cm=3.0,
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
            entry, label_size=label_size, label_colour=ltext_el["colour"],
        )

        sizes = measure_legend_grobs(
            decor, label_grobs, n_breaks,
            nrow=nrow, ncol=ncol,
            key_width_cm=KEY_W_CM, key_height_cm=KEY_H_CM,
            spacing_x=SPACING_X_CM, spacing_y=SPACING_Y_CM,
            text_position="right",
        )

        layout = arrange_legend_layout(
            n_breaks, nrow=nrow, ncol=ncol,
            text_position="right",
        )

        title_grob = text_grob(
            label=entry["title"],
            x=0.0, y=0.5,
            just=("left", "centre"),
            gp=Gpar(
                fontsize=title_size,
                col=ltitle_el["colour"],
                fontface="bold",
            ),
            name=f"legend.title.{entry['title']}",
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
        legend_gtables, position="right", spacing_cm=0.2,
    )

    # ------------------------------------------------------------------
    # 7. Place guide-box in the plot table
    #    Mirrors R's table_add_legends (plot-render.R:98-105)
    # ------------------------------------------------------------------
    from ggplot2_py.guide_legend import _gtable_total_cm

    guide_w_cm = _gtable_total_cm(guide_box.widths)
    guide_w_cm = max(guide_w_cm, 1.0)

    LEGEND_SPACING_CM = 0.2
    table = gtable_add_cols(table, unit([LEGEND_SPACING_CM], "cm"), pos=-1)
    table = gtable_add_cols(table, unit([guide_w_cm], "cm"), pos=-1)
    ncol_t = len(table._widths)
    nrow_t = len(table._heights)
    table = gtable_add_grob(
        table, guide_box, t=1, b=nrow_t, l=ncol_t,
        clip="off", name="guide-box-right",
    )

    return table


def _table_add_titles(table: Any, labels: Dict[str, Any], theme: Any) -> Any:
    """Add title, subtitle, caption annotations to the plot table.

    Mirrors R's plot assembly in ``ggplot_gtable.R``: title and subtitle
    are added as rows at the top, caption at the bottom.

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
    from grid_py import Unit as unit, text_grob, Gpar
    from ggplot2_py.coord import _resolve_element

    if not hasattr(table, "_widths"):
        return table

    ncol = len(table._widths)

    # --- Caption (bottom) ---
    caption = labels.get("caption")
    if caption:
        el = _resolve_element("plot.caption", theme,
            {"colour": "grey30", "size": 7, "hjust": 1.0})
        table = gtable_add_rows(table, unit([0.35], "cm"), pos=-1)
        nrow = len(table._heights)
        table = gtable_add_grob(
            table,
            text_grob(
                label=str(caption), x=float(el.get("hjust", 0.95)), y=0.5,
                just="right",
                gp=Gpar(fontsize=float(el["size"]), col=el["colour"]),
                name="caption",
            ),
            t=nrow, l=1, r=ncol, clip="off", name="caption",
        )

    # --- Subtitle (top, added first so title goes above) ---
    subtitle = labels.get("subtitle")
    if subtitle:
        el = _resolve_element("plot.subtitle", theme,
            {"colour": "grey30", "size": 8, "hjust": 0.5})
        table = gtable_add_rows(table, unit([0.35], "cm"), pos=0)
        table = gtable_add_grob(
            table,
            text_grob(
                label=str(subtitle), x=float(el.get("hjust", 0.5)), y=0.5,
                just="centre",
                gp=Gpar(fontsize=float(el["size"]), col=el["colour"]),
                name="subtitle",
            ),
            t=1, l=1, r=ncol, clip="off", name="subtitle",
        )

    # --- Title (top) ---
    title = labels.get("title")
    if title:
        el = _resolve_element("plot.title", theme,
            {"colour": "black", "size": 11, "hjust": 0.5})
        table = gtable_add_rows(table, unit([0.5], "cm"), pos=0)
        table = gtable_add_grob(
            table,
            text_grob(
                label=str(title), x=float(el.get("hjust", 0.5)), y=0.5,
                just="centre",
                gp=Gpar(fontsize=float(el["size"]), col=el["colour"]),
                name="title",
            ),
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

    Parameters
    ----------
    table : gtable
        A gtable object.

    Returns
    -------
    dict
        ``{"t": int, "l": int, "b": int, "r": int}`` panel bounds.
    """
    if hasattr(table, "layout") and isinstance(table.layout, pd.DataFrame):
        panel_rows = table.layout.loc[
            table.layout["name"].str.contains("panel", case=False, na=False)
        ]
        if not panel_rows.empty:
            return {
                "t": int(panel_rows["t"].min()),
                "l": int(panel_rows["l"].min()),
                "b": int(panel_rows["b"].max()),
                "r": int(panel_rows["r"].max()),
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

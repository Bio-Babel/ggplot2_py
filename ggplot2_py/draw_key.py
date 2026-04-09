"""
Legend key drawing functions for ggplot2.

Each ``draw_key_*`` function draws a single legend glyph for a given geom.
They accept the same three arguments: *data* (a single-row dict-like of
scaled aesthetics), *params* (extra layer parameters), and *size* (key
dimensions in mm).

Notes
-----
All functions return a grid_py grob.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from grid_py import (
    points_grob,
    rect_grob,
    segments_grob,
    lines_grob,
    polygon_grob,
    text_grob,
    circle_grob,
    null_grob,
    Gpar,
    Unit,
    grob_tree,
    GTree,
    Viewport,
    roundrect_grob,
    GList,
)

from scales import alpha as _scales_alpha

__all__ = [
    "draw_key_point",
    "draw_key_path",
    "draw_key_rect",
    "draw_key_polygon",
    "draw_key_blank",
    "draw_key_boxplot",
    "draw_key_crossbar",
    "draw_key_dotplot",
    "draw_key_label",
    "draw_key_linerange",
    "draw_key_pointrange",
    "draw_key_smooth",
    "draw_key_text",
    "draw_key_abline",
    "draw_key_vline",
    "draw_key_timeseries",
    "draw_key_vpath",
]

# ---------------------------------------------------------------------------
# Graphic-unit constants (same as R's ggplot2)
# ---------------------------------------------------------------------------

_PT: float = 72.27 / 25.4
_STROKE: float = 96 / 25.4


def _alpha(colour: Any, alpha_val: Any) -> Any:
    """Apply alpha to a colour, tolerating ``None``/``np.nan``."""
    try:
        return _scales_alpha(colour, alpha_val)
    except Exception:
        return colour


def _fill_alpha(fill: Any, alpha_val: Any) -> Any:
    """Like ``fill_alpha`` in R -- apply alpha only for non-NA fills."""
    if fill is None:
        return None
    try:
        return _scales_alpha(fill, alpha_val)
    except Exception:
        return fill


def _get(data: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict-like *data*."""
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


# ---------------------------------------------------------------------------
# Key glyph functions
# ---------------------------------------------------------------------------


def draw_key_point(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for point geoms.

    Parameters
    ----------
    data : dict
        Scaled aesthetics for a single legend entry.
    params : dict
        Extra layer parameters.
    size : optional
        Key dimensions.

    Returns
    -------
    grob
        A grid_py ``points_grob``.
    """
    shape = _get(data, "shape", 19)
    from ggplot2_py.geom import translate_shape_string
    shape = translate_shape_string(shape)

    return points_grob(
        x=0.5,
        y=0.5,
        pch=shape,
        gp=Gpar(
            col=_alpha(_get(data, "colour", "black"), _get(data, "alpha")),
            fill=_fill_alpha(_get(data, "fill", "black"), _get(data, "alpha")),
            fontsize=(_get(data, "size", 1.5) * _PT) + (_get(data, "stroke", 0.5) * _STROKE),
            lwd=_get(data, "stroke", 0.5) * _STROKE,
        ),
    )


def draw_key_abline(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for abline geoms (diagonal segment)."""
    return segments_grob(
        x0=0, y0=0, x1=1, y1=1,
        gp=Gpar(
            col=_alpha(
                _get(data, "colour", _get(data, "fill", "black")),
                _get(data, "alpha"),
            ),
            lwd=_get(data, "linewidth", 0.5) * _PT,
            lty=_get(data, "linetype", 1),
            lineend=_get(params, "lineend", "butt"),
        ),
    )


def draw_key_rect(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for rect/tile geoms (filled rectangle)."""
    fill = _get(data, "fill")
    colour = fill if fill is not None else _get(data, "colour")
    return rect_grob(
        gp=Gpar(
            col=None,
            fill=_fill_alpha(colour if colour is not None else "grey20", _get(data, "alpha")),
            lty=_get(data, "linetype", 1),
        ),
    )


def draw_key_polygon(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for polygon/bar geoms (outlined filled rectangle)."""
    lwd = _get(data, "linewidth", 0)
    return rect_grob(
        gp=Gpar(
            col=_get(data, "colour"),
            fill=_fill_alpha(_get(data, "fill", "grey20"), _get(data, "alpha")),
            lty=_get(data, "linetype", 1),
            lwd=lwd * _PT if lwd else 0,
            linejoin=_get(params, "linejoin", "mitre"),
            lineend=_get(params, "lineend", "butt"),
        ),
    )


def draw_key_blank(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw nothing (blank legend key)."""
    return null_grob()


def draw_key_boxplot(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for boxplot geoms."""
    gp = Gpar(
        col=_get(data, "colour", "grey20"),
        fill=_fill_alpha(_get(data, "fill", "white"), _get(data, "alpha")),
        lwd=(_get(data, "linewidth", 0.5)) * _PT,
        lty=_get(data, "linetype", 1),
        lineend=_get(params, "lineend", "butt"),
        linejoin=_get(params, "linejoin", "mitre"),
    )
    return grob_tree(
        lines_grob(x=[0.5, 0.5], y=[0.1, 0.25], gp=gp),
        lines_grob(x=[0.5, 0.5], y=[0.75, 0.9], gp=gp),
        rect_grob(height=Unit(0.5, "npc"), width=Unit(0.75, "npc"), gp=gp),
        segments_grob(x0=0.125, y0=0.5, x1=0.875, y1=0.5, gp=gp),
    )


def draw_key_crossbar(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for crossbar geoms."""
    gp = Gpar(
        col=_get(data, "colour", "grey20"),
        fill=_fill_alpha(_get(data, "fill", "white"), _get(data, "alpha")),
        lwd=(_get(data, "linewidth", 0.5)) * _PT,
        lty=_get(data, "linetype", 1),
        lineend=_get(params, "lineend", "butt"),
        linejoin=_get(params, "linejoin", "mitre"),
    )
    return grob_tree(
        rect_grob(height=Unit(0.5, "npc"), width=Unit(0.75, "npc"), gp=gp),
        segments_grob(x0=0.125, y0=0.5, x1=0.875, y1=0.5, gp=gp),
    )


def draw_key_dotplot(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for dotplot geoms."""
    return points_grob(
        x=0.5,
        y=0.5,
        pch=21,
        size=Unit(0.5, "npc"),
        gp=Gpar(
            col=_alpha(_get(data, "colour", "black"), _get(data, "alpha")),
            fill=_fill_alpha(_get(data, "fill", "black"), _get(data, "alpha")),
            lty=_get(data, "linetype", 1),
        ),
    )


def draw_key_label(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for label geoms (text with background)."""
    label = _get(data, "label", "a")
    lwd = _get(data, "linewidth", 0.25)
    return grob_tree(
        roundrect_grob(
            gp=Gpar(
                col=_get(data, "colour", "black") if lwd > 0 else None,
                fill=_fill_alpha(_get(data, "fill", "white"), _get(data, "alpha")),
                lwd=lwd * _PT,
                lty=_get(data, "linetype", 1),
            ),
        ),
        text_grob(
            label=label,
            x=0.5,
            y=0.5,
            gp=Gpar(
                col=_alpha(_get(data, "colour", "black"), _get(data, "alpha")),
                fontsize=(_get(data, "size", 3.88)) * _PT,
                fontfamily=_get(data, "family", ""),
                fontface=_get(data, "fontface", 1),
            ),
        ),
    )


def draw_key_linerange(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for linerange geoms (vertical segment)."""
    if _get(params, "flipped_aes", False):
        return draw_key_path(data, params, size)
    return draw_key_vpath(data, params, size)


def draw_key_pointrange(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for pointrange geoms."""
    line_grob = draw_key_linerange(data, params, size)
    pt_data = dict(data) if isinstance(data, dict) else {k: getattr(data, k, None) for k in dir(data)}
    pt_data["size"] = (_get(data, "size", 1.5)) * 4
    point_grob = draw_key_point(pt_data, params, size)
    return grob_tree(line_grob, point_grob)


def draw_key_smooth(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for smooth geoms (ribbon + line)."""
    fill_colour = _fill_alpha(
        _get(data, "fill", "grey60"),
        _get(data, "alpha"),
    )
    path_grob = draw_key_path(data, params, size)
    se = _get(params, "se", False)
    children = []
    if se:
        children.append(rect_grob(gp=Gpar(col=None, fill=fill_colour)))
    children.append(path_grob)
    return grob_tree(*children)


def draw_key_text(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for text geoms."""
    label = _get(data, "label", "a")
    return text_grob(
        label=label,
        x=0.5,
        y=0.5,
        rot=_get(data, "angle", 0),
        gp=Gpar(
            col=_alpha(
                _get(data, "colour", _get(data, "fill", "black")),
                _get(data, "alpha"),
            ),
            fontfamily=_get(data, "family", ""),
            fontface=_get(data, "fontface", 1),
            fontsize=(_get(data, "size", 3.88)) * _PT,
        ),
    )


def draw_key_path(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for path/line geoms (horizontal segment)."""
    linetype = _get(data, "linetype")
    if linetype is None:
        linetype = 0
    return segments_grob(
        x0=0.1,
        y0=0.5,
        x1=0.9,
        y1=0.5,
        gp=Gpar(
            col=_alpha(
                _get(data, "colour", _get(data, "fill", "black")),
                _get(data, "alpha"),
            ),
            lwd=(_get(data, "linewidth", 0.5)) * _PT,
            lty=linetype if linetype else 1,
            lineend=_get(params, "lineend", "butt"),
        ),
    )


def draw_key_vpath(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key as a vertical segment."""
    return segments_grob(
        x0=0.5,
        y0=0.1,
        x1=0.5,
        y1=0.9,
        gp=Gpar(
            col=_alpha(
                _get(data, "colour", _get(data, "fill", "black")),
                _get(data, "alpha"),
            ),
            lwd=(_get(data, "linewidth", 0.5)) * _PT,
            lty=_get(data, "linetype", 1),
            lineend=_get(params, "lineend", "butt"),
        ),
    )


def draw_key_vline(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for vline geoms (full-height vertical segment)."""
    return segments_grob(
        x0=0.5,
        y0=0.0,
        x1=0.5,
        y1=1.0,
        gp=Gpar(
            col=_alpha(
                _get(data, "colour", _get(data, "fill", "black")),
                _get(data, "alpha"),
            ),
            lwd=(_get(data, "linewidth", 0.5)) * _PT,
            lty=_get(data, "linetype", 1),
            lineend=_get(params, "lineend", "butt"),
        ),
    )


def draw_key_timeseries(
    data: Dict[str, Any],
    params: Dict[str, Any],
    size: Any = None,
) -> Any:
    """Draw a legend key for time-series geoms (wiggle line)."""
    linetype = _get(data, "linetype")
    if linetype is None:
        linetype = 0
    return lines_grob(
        x=[0.0, 0.4, 0.6, 1.0],
        y=[0.1, 0.6, 0.4, 0.9],
        gp=Gpar(
            col=_alpha(
                _get(data, "colour", _get(data, "fill", "black")),
                _get(data, "alpha"),
            ),
            lwd=(_get(data, "linewidth", 0.5)) * _PT,
            lty=linetype if linetype else 1,
            lineend=_get(params, "lineend", "butt"),
            linejoin=_get(params, "linejoin", "round"),
        ),
    )

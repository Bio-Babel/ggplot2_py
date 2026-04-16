"""
Colourbar guide building functions — faithful port of R's GuideColourbar.

Produces a continuous colour gradient bar as an independent
:class:`~gtable_py.Gtable`, using ``raster_grob`` for the colour strip
and text/segment grobs for tick labels and tick marks.

R references
------------
* ``ggplot2/R/guide-colorbar.R`` — GuideColourbar class
* ``ggplot2/R/guide-legend.R``   — inherited assemble_drawing
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grid_py import (
    Gpar,
    Unit,
    Viewport,
    null_grob,
    raster_grob,
    rect_grob,
    segments_grob,
    text_grob,
    unit_c,
)
from grid_py._grob import GList, GTree, grob_tree

from gtable_py import (
    Gtable,
    gtable_add_cols,
    gtable_add_grob,
    gtable_add_padding,
    gtable_add_rows,
)

__all__ = [
    "extract_colourbar_decor",
    "extract_coloursteps_decor",
    "build_colourbar_decor",
    "build_coloursteps_decor",
    "build_colourbar_labels",
    "build_colourbar_ticks",
    "assemble_colourbar",
]

# ---------------------------------------------------------------------------
# Constants  (R defaults from GuideColourbar / theme)
# ---------------------------------------------------------------------------

_DEFAULT_NBIN: int = 300
_DEFAULT_BAR_WIDTH_CM: float = 0.5      # legend.key.width
_DEFAULT_BAR_HEIGHT_CM: float = 3.0     # legend.key.height * 5 (R multiplies)
_DEFAULT_LABEL_SIZE: float = 6.0
_DEFAULT_TITLE_SIZE: float = 7.0
_DEFAULT_PADDING_CM: float = 0.15
_DEFAULT_TICK_LENGTH_CM: float = 0.1


# ---------------------------------------------------------------------------
# extract_colourbar_decor
# ---------------------------------------------------------------------------

def extract_colourbar_decor(
    scale: Any,
    nbin: int = _DEFAULT_NBIN,
    alpha: Optional[float] = None,
    reverse: bool = False,
) -> Dict[str, Any]:
    """Generate a dense colour sequence across the scale's limits.

    Mirrors ``GuideColourbar$extract_decor`` (guide-colorbar.R:244-260).

    Parameters
    ----------
    scale : Scale
        A trained continuous colour/fill scale.
    nbin : int
        Number of colour bins.
    alpha : float or None
        Optional alpha override.
    reverse : bool
        Reverse the colour order.

    Returns
    -------
    dict
        ``{"colour": list[str], "value": ndarray}``
    """
    limits = scale.get_limits()
    bar_values = np.linspace(limits[0], limits[1], nbin)
    if len(bar_values) == 0:
        bar_values = np.unique(limits)

    # Map values through the scale to get colours
    colours = scale.map(bar_values)
    if isinstance(colours, np.ndarray):
        colours = colours.tolist()

    # Apply alpha if specified
    if alpha is not None and not (isinstance(alpha, float) and np.isnan(alpha)):
        try:
            from scales import alpha as _scales_alpha
            colours = [_scales_alpha(c, alpha) for c in colours]
        except Exception:
            pass

    if reverse:
        colours = list(reversed(colours))
        bar_values = bar_values[::-1]

    return {"colour": colours, "value": bar_values}


# ---------------------------------------------------------------------------
# extract_coloursteps_decor
# ---------------------------------------------------------------------------

def extract_coloursteps_decor(
    scale: Any,
    breaks: List[Any],
    alpha: Optional[float] = None,
    reverse: bool = False,
    even_steps: bool = True,
) -> Dict[str, Any]:
    """Generate discrete colour bins from a binned scale's breaks.

    Mirrors ``GuideColoursteps$extract_decor`` (guide-colorsteps.R:137-159).

    Instead of a dense colour sequence, produces one colour per bin
    between consecutive breaks.

    Parameters
    ----------
    scale : Scale
        A trained binned colour/fill scale.
    breaks : list
        Scale breaks (bin boundaries).
    alpha : float or None
        Optional alpha override.
    reverse : bool
        Reverse the colour order.
    even_steps : bool
        If ``True``, bins have equal visual width.

    Returns
    -------
    dict
        ``{"colour": list[str], "min": list[float], "max": list[float]}``
    """
    limits = scale.get_limits()
    # Combine limits and breaks into sorted unique boundary set
    boundaries = sorted(set(list(limits) + [float(b) for b in breaks
                                             if b is not None and not
                                             (isinstance(b, float) and np.isnan(b))]))
    n = len(boundaries)
    if n < 2:
        return {"colour": [], "min": [], "max": []}

    # Bin midpoints: map each mid value to get the bin colour
    bin_mids = [(boundaries[i] + boundaries[i + 1]) / 2.0 for i in range(n - 1)]
    colours = scale.map(np.array(bin_mids))
    if isinstance(colours, np.ndarray):
        colours = colours.tolist()

    # Apply alpha
    if alpha is not None and not (isinstance(alpha, float) and np.isnan(alpha)):
        try:
            from scales import alpha as _scales_alpha
            colours = [_scales_alpha(c, alpha) for c in colours]
        except Exception:
            pass

    # Even steps: use integer indices instead of actual break values
    if even_steps:
        mins = list(range(n - 1))
        maxs = list(range(1, n))
    else:
        mins = boundaries[:-1]
        maxs = boundaries[1:]

    if reverse:
        colours = list(reversed(colours))
        mins = list(reversed(mins))
        maxs = list(reversed(maxs))

    return {"colour": colours, "min": mins, "max": maxs}


def build_coloursteps_decor(
    decor: Dict[str, Any],
    direction: str = "vertical",
) -> Dict[str, Any]:
    """Build stepped rectangle grobs for coloursteps guide.

    Mirrors ``GuideColoursteps$build_decor`` (guide-colorsteps.R:208-229).

    Parameters
    ----------
    decor : dict
        From :func:`extract_coloursteps_decor`.
    direction : str
        ``"vertical"`` or ``"horizontal"``.

    Returns
    -------
    dict
        ``{"bar": grob, "frame": grob}``
    """
    colours = decor["colour"]
    mins = decor["min"]
    maxs = decor["max"]
    n = len(colours)

    if n == 0:
        return {"bar": null_grob(), "frame": null_grob()}

    # Normalise positions to [0, 1]
    all_vals = mins + maxs
    lo = min(all_vals)
    hi = max(all_vals)
    rng = hi - lo if hi != lo else 1.0

    norm_mins = [(v - lo) / rng for v in mins]
    norm_maxs = [(v - lo) / rng for v in maxs]
    sizes = [mx - mn for mn, mx in zip(norm_mins, norm_maxs)]

    if direction == "vertical":
        bar = rect_grob(
            x=[0.0] * n, y=norm_mins,
            width=[1.0] * n, height=sizes,
            just=("left", "bottom"),
            default_units="npc",
            gp=Gpar(col=None, fill=colours),
            name="coloursteps.bar",
        )
    else:
        bar = rect_grob(
            x=norm_mins, y=[0.0] * n,
            width=sizes, height=[1.0] * n,
            just=("left", "bottom"),
            default_units="npc",
            gp=Gpar(col=None, fill=colours),
            name="coloursteps.bar",
        )

    frame = rect_grob(
        gp=Gpar(col="grey50", fill=None, lwd=0.5),
        name="coloursteps.frame",
    )

    return {"bar": bar, "frame": frame}


# ---------------------------------------------------------------------------
# build_colourbar_decor
# ---------------------------------------------------------------------------

def build_colourbar_decor(
    decor: Dict[str, Any],
    direction: str = "vertical",
    display: str = "raster",
) -> Dict[str, Any]:
    """Build the colour bar grob.

    Mirrors ``GuideColourbar$build_decor`` (guide-colorbar.R:360-413).

    Supports two display modes:
    - ``"raster"``: a single ``raster_grob`` with interpolated colours
      (default, matches R's default)
    - ``"rectangles"``: individual ``rect_grob`` for each colour bin

    The ``"gradient"`` mode (using ``linearGradient``) is deferred pending
    grid_py gradient support.

    Parameters
    ----------
    decor : dict
        From :func:`extract_colourbar_decor`.
    direction : str
        ``"vertical"`` or ``"horizontal"``.
    display : str
        ``"raster"`` or ``"rectangles"``.

    Returns
    -------
    dict
        ``{"bar": grob, "frame": grob}``
    """
    colours = decor["colour"]
    n = len(colours)

    if display == "raster":
        # Build a raster image from the colour array
        # R: rasterGrob(image, width=1, height=1, default.units="npc",
        #               interpolate=TRUE)
        if direction == "horizontal":
            # 1-row image, colours left to right
            image = np.array([colours], dtype=object)  # shape (1, n)
        else:
            # n-row image (reversed for bottom-to-top), 1 column
            image = np.array([[c] for c in reversed(colours)], dtype=object)

        bar = raster_grob(
            image=image,
            x=0.5, y=0.5,
            width=1, height=1,
            default_units="npc",
            interpolate=True,
            gp=Gpar(col=None),
            name="colourbar.bar",
        )

    elif display == "rectangles":
        # Individual rectangles for each colour bin
        # R: rectGrob(x, y, width, height, vjust=0, hjust=0, ...)
        if direction == "horizontal":
            w = 1.0 / n
            xs = [(i * w) for i in range(n)]
            bar = rect_grob(
                x=xs, y=0,
                width=w, height=1,
                just=("left", "bottom"),
                default_units="npc",
                gp=Gpar(col=None, fill=colours),
                name="colourbar.bar",
            )
        else:
            h = 1.0 / n
            ys = [(i * h) for i in range(n)]
            bar = rect_grob(
                x=0, y=ys,
                width=1, height=h,
                just=("left", "bottom"),
                default_units="npc",
                gp=Gpar(col=None, fill=colours),
                name="colourbar.bar",
            )
    else:
        # Fallback to raster
        return build_colourbar_decor(decor, direction, display="raster")

    # Frame around the bar
    frame = rect_grob(
        gp=Gpar(col="grey50", fill=None, lwd=0.5),
        name="colourbar.frame",
    )

    return {"bar": bar, "frame": frame}


# ---------------------------------------------------------------------------
# build_colourbar_labels
# ---------------------------------------------------------------------------

def build_colourbar_labels(
    breaks: List[Any],
    break_labels: List[str],
    limits: Tuple[float, float],
    direction: str = "vertical",
    label_size: float = _DEFAULT_LABEL_SIZE,
    label_colour: str = "grey20",
) -> List[Any]:
    """Build tick labels positioned at break NPC positions along the bar.

    Mirrors ``GuideColourbar$build_labels`` (guide-colorbar.R:327-341).

    Parameters
    ----------
    breaks : list
        Numeric break values.
    break_labels : list of str
        Formatted break labels.
    limits : tuple of float
        Scale limits (min, max).
    direction : str
        ``"vertical"`` or ``"horizontal"``.
    label_size : float
        Font size in points.
    label_colour : str
        Font colour.

    Returns
    -------
    list of grob
        One text grob per break.
    """
    lo, hi = limits
    rng = hi - lo
    if rng == 0:
        rng = 1.0

    # R (guide-colorbar.R: build_labels filters breaks to the visible
    # bar via GuideColourbar$extract_key / limit clip).  A break at
    # e.g. 0 when limits=[1,14] lies at NPC = -0.077 and would render
    # OUTSIDE the bar extent — both visually (overflowing the legend
    # frame) and semantically (labelling a colour that isn't in the
    # bar).  Skip those breaks, matching R.
    _EPS = 1e-9
    grobs = []
    for i, (brk, lab) in enumerate(zip(breaks, break_labels)):
        try:
            brk_val = float(brk)
        except (TypeError, ValueError):
            continue
        npc_pos = (brk_val - lo) / rng
        if npc_pos < -_EPS or npc_pos > 1 + _EPS:
            continue

        if direction == "vertical":
            grobs.append(text_grob(
                label=str(lab),
                x=0.0, y=npc_pos,
                just=("left", "centre"),
                gp=Gpar(fontsize=label_size, col=label_colour),
                name=f"colourbar.label.{i}",
            ))
        else:
            grobs.append(text_grob(
                label=str(lab),
                x=npc_pos, y=0.0,
                just=("centre", "top"),
                gp=Gpar(fontsize=label_size, col=label_colour),
                name=f"colourbar.label.{i}",
            ))
    return grobs


# ---------------------------------------------------------------------------
# build_colourbar_ticks
# ---------------------------------------------------------------------------

def build_colourbar_ticks(
    breaks: List[Any],
    limits: Tuple[float, float],
    direction: str = "vertical",
    draw_lim: Tuple[bool, bool] = (True, True),
    tick_length_npc: float = 0.2,
    tick_colour: str = "white",
    tick_linewidth_mm: float = 0.5 / (72.27 / 25.4),
) -> Any:
    """Build tick marks at break positions.

    Mirrors ``GuideColourbar$build_ticks`` + ``Guide$build_ticks``
    (guide-colorbar.R:343-358, guide-.R:698-741).  In R each tick
    starts at the bar edge and extends **inward** by
    ``length = rel(0.2) * legend.key.size`` — for a vertical bar
    whose width matches the key size this is 0.2 npc of the bar's
    parallel dimension.  The default colour is **white** (from
    ``default_ticks = element_line(colour="white", linewidth=0.5/.pt)``)
    so the ticks are visible on any dark/light bar colour.

    Parameters
    ----------
    breaks : list
        Numeric break values.
    limits : tuple of float
        Scale limits.
    direction : str
        ``"vertical"`` or ``"horizontal"``.
    draw_lim : tuple of bool
        Whether to draw ticks at lower/upper limits.
    tick_length_npc : float
        Tick length as a fraction of the bar's orthogonal dimension
        (0.2 = R default ``legend.ticks.length = rel(0.2)``).
    tick_colour : str
        Tick line colour (default ``"white"`` per R ``default_ticks``).
    tick_linewidth : float
        Tick line width in grid lwd units.
    """
    lo, hi = limits
    rng = hi - lo
    if rng == 0:
        rng = 1.0

    # Filter breaks to the visible bar range (R guide-colorbar.R:
    # extract_key discards breaks outside [lo, hi]).
    _EPS = 1e-9
    positions = []
    for brk in breaks:
        try:
            npc_pos = (float(brk) - lo) / rng
        except (TypeError, ValueError):
            continue
        if npc_pos < -_EPS or npc_pos > 1 + _EPS:
            continue
        positions.append(npc_pos)

    if not draw_lim[0] and positions:
        positions = positions[1:]
    if not draw_lim[1] and positions:
        positions = positions[:-1]

    if not positions:
        return null_grob()

    n = len(positions)
    # R (utilities-grid.R:32-33 gg_par):
    #   args$lwd <- args$lwd * .pt
    # element_line linewidth is in mm; grid lwd is in points.  Multiply
    # by .pt = 72.27/25.4 to convert.  default_ticks linewidth=0.5/.pt
    # mm → lwd = 0.5 pt (matches R's visible tick stroke).
    _PT = 72.27 / 25.4
    gp = Gpar(col=tick_colour, lwd=tick_linewidth_mm * _PT)

    if direction == "vertical":
        # Ticks on both sides, extending INWARD (R guide-.R:728-732):
        #   right side (position=1): dir = -length; segment = [1, 1-length]
        #   left  side (position=0): dir = +length; segment = [0, 0+length]
        x0_r = [1.0] * n
        x1_r = [1.0 - tick_length_npc] * n
        x0_l = [0.0] * n
        x1_l = [0.0 + tick_length_npc] * n
        return grob_tree(
            segments_grob(x0=x0_l, y0=positions, x1=x1_l, y1=positions,
                          gp=gp, name="ticks.left"),
            segments_grob(x0=x0_r, y0=positions, x1=x1_r, y1=positions,
                          gp=gp, name="ticks.right"),
            name="colourbar.ticks",
        )
    else:
        # Horizontal: ticks top & bottom, extending inward
        y0_t = [1.0] * n
        y1_t = [1.0 - tick_length_npc] * n
        y0_b = [0.0] * n
        y1_b = [0.0 + tick_length_npc] * n
        return grob_tree(
            segments_grob(x0=positions, y0=y0_b, x1=positions, y1=y1_b,
                          gp=gp, name="ticks.bottom"),
            segments_grob(x0=positions, y0=y0_t, x1=positions, y1=y1_t,
                          gp=gp, name="ticks.top"),
            name="colourbar.ticks",
        )


# ---------------------------------------------------------------------------
# assemble_colourbar
# ---------------------------------------------------------------------------

def assemble_colourbar(
    bar_grob: Any,
    frame_grob: Any,
    ticks_grob: Any,
    label_grobs: List[Any],
    title_grob: Any,
    direction: str = "vertical",
    bar_width_cm: float = _DEFAULT_BAR_WIDTH_CM,
    bar_height_cm: float = _DEFAULT_BAR_HEIGHT_CM,
    label_width_cm: float = 0.8,
    padding_cm: float = _DEFAULT_PADDING_CM,
    bg_colour: Optional[str] = "white",
) -> Gtable:
    """Assemble all colourbar parts into a Gtable.

    Mirrors ``GuideColourbar`` using the inherited
    ``GuideLegend$assemble_drawing`` pattern (guide-legend.R:533-591).

    The layout for a vertical colourbar:
    ```
    [bar_col] [gap] [labels_col]
    ```
    The bar occupies a single tall cell; labels are stacked in the
    adjacent column.

    Parameters
    ----------
    bar_grob, frame_grob, ticks_grob : grob
        Colour bar components.
    label_grobs : list of grob
        Tick label grobs.
    title_grob : grob
        Title grob.
    direction : str
        ``"vertical"`` or ``"horizontal"``.
    bar_width_cm, bar_height_cm : float
        Bar dimensions.
    label_width_cm : float
        Width for label column.
    padding_cm : float
        Padding.
    bg_colour : str or None
        Background fill.

    Returns
    -------
    Gtable
        Self-contained colourbar gtable.
    """
    gap_cm = 0.1

    if direction == "vertical":
        # Layout: [bar] [gap] [labels]  — 1 row, 3 columns
        widths = Unit([bar_width_cm, gap_cm, label_width_cm], "cm")
        heights = Unit([bar_height_cm], "cm")
        gt = Gtable(widths=widths, heights=heights, name="colourbar")

        # Bar + frame + ticks in cell (1, 1)
        bar_tree = GTree(
            children=GList(bar_grob, frame_grob, ticks_grob),
            name="colourbar.bar.tree",
        )
        gt = gtable_add_grob(gt, bar_tree, t=1, l=1, clip="off", name="bar")

        # Labels in cell (1, 3)
        if label_grobs:
            label_tree = GTree(
                children=GList(*label_grobs),
                name="colourbar.labels",
            )
            gt = gtable_add_grob(gt, label_tree, t=1, l=3, clip="off", name="labels")

    else:
        # Horizontal: [labels] above [gap] above [bar]
        #   — 3 rows, 1 column  (bar at bottom, labels on top)
        widths = Unit([bar_height_cm], "cm")  # bar length is "height" param
        heights = Unit([label_width_cm, gap_cm, bar_width_cm], "cm")
        gt = Gtable(widths=widths, heights=heights, name="colourbar")

        bar_tree = GTree(
            children=GList(bar_grob, frame_grob, ticks_grob),
            name="colourbar.bar.tree",
        )
        gt = gtable_add_grob(gt, bar_tree, t=3, l=1, clip="off", name="bar")

        if label_grobs:
            label_tree = GTree(
                children=GList(*label_grobs),
                name="colourbar.labels",
            )
            gt = gtable_add_grob(gt, label_tree, t=1, l=1, clip="off", name="labels")

    # Add title
    from ggplot2_py.guide_legend import add_legend_title
    gt = add_legend_title(gt, title_grob, position="top")

    # Add padding
    pad = Unit([padding_cm] * 4, "cm")
    gt = gtable_add_padding(gt, pad)

    # Add background
    if bg_colour is not None:
        bg = rect_grob(
            gp=Gpar(fill=bg_colour, col="grey85", lwd=0.5),
            name="colourbar.background",
        )
        gt = gtable_add_grob(
            gt, bg,
            t=1, l=1, b=gt.nrow, r=gt.ncol,
            z=-math.inf, clip="off", name="background",
        )

    return gt

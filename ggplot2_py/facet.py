"""
Faceting system for ggplot2.

Facets control how data is split into subsets and displayed as a matrix
of panels. The base :class:`Facet` class defines the interface; concrete
implementations include :class:`FacetNull` (no faceting),
:class:`FacetGrid` (rows x columns grid), and :class:`FacetWrap`
(1-d ribbon wrapped into 2-d).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn
from ggplot2_py.ggproto import GGProto, ggproto
from ggplot2_py._utils import snake_class, compact, modify_list, empty


def _is_null_grob(grob: Any) -> bool:
    """Check if a grob is a null grob (R semantics: zeroGrob / nullGrob)."""
    if grob is None:
        return True
    cls = getattr(grob, "_grid_class", "")
    name = getattr(grob, "_name", getattr(grob, "name", ""))
    return cls == "null" or "null" in str(name).lower() or "zero" in str(name).lower()


def _axis_width_cm(ax: Any) -> float:
    """Measure axis grob width in cm.

    R measures axis width via ``gtable_width(gt)`` + ``convertUnit(..., "cm")``.
    No fallback — if measurement fails, let it surface.
    """
    from gtable_py import Gtable, gtable_width
    from grid_py import convert_width
    if isinstance(ax, Gtable):
        w = gtable_width(ax)
        result = convert_width(w, "cm", valueOnly=True)
        return float(np.sum(result))
    # _AbsoluteAxisGrob path
    val = getattr(ax, "_width_cm", None)
    if val is not None:
        return val
    # width_details path
    if hasattr(ax, "width_details"):
        from ggplot2_py.guide_axis import _width_cm
        return _width_cm(ax)
    raise ValueError(f"Cannot measure width of {type(ax).__name__}")


def _axis_height_cm(ax: Any) -> float:
    """Measure axis grob height in cm.

    R measures axis height via ``gtable_height(gt)`` + ``convertUnit(..., "cm")``.
    No fallback — if measurement fails, let it surface.
    """
    from gtable_py import Gtable, gtable_height
    from grid_py import convert_height
    if isinstance(ax, Gtable):
        h = gtable_height(ax)
        result = convert_height(h, "cm", valueOnly=True)
        return float(np.sum(result))
    val = getattr(ax, "_height_cm", None)
    if val is not None:
        return val
    if hasattr(ax, "height_details"):
        from ggplot2_py.guide_axis import _height_cm
        return _height_cm(ax)
    raise ValueError(f"Cannot measure height of {type(ax).__name__}")

__all__ = [
    "Facet",
    "FacetNull",
    "FacetGrid",
    "FacetWrap",
    "facet_null",
    "facet_grid",
    "facet_wrap",
    "is_facet",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layout_null() -> pd.DataFrame:
    """Return a single-panel layout.

    Returns
    -------
    pd.DataFrame
        One-row layout with columns ``PANEL``, ``ROW``, ``COL``,
        ``SCALE_X``, ``SCALE_Y``.
    """
    return pd.DataFrame({
        "PANEL": pd.Categorical([1]),
        "ROW": [1],
        "COL": [1],
        "SCALE_X": [1],
        "SCALE_Y": [1],
    })


def _n2mfrow(n: int) -> Tuple[int, int]:
    """Port of R's ``grDevices::n2mfrow`` (aspect=1).

    Returns ``(rows, cols)``.  R uses this to compute panel grid
    shapes for ``par(mfrow=...)``; ``facet_wrap`` reuses it and
    *swaps* the result so that ``nrow<-rc[2]; ncol<-rc[1]`` —
    giving a wide-preferring layout (1×n for n ≤ 3).
    """
    if n <= 3:
        return (n, 1)
    if n <= 6:
        return ((n + 1) // 2, 2)
    if n <= 12:
        return ((n + 2) // 3, 3)
    asp = 1
    return (math.ceil(math.sqrt(n / asp)), math.ceil(math.sqrt(n * asp)))


def wrap_dims(n: int, nrow: Optional[int] = None, ncol: Optional[int] = None) -> Tuple[int, int]:
    """Compute grid dimensions for *n* panels.

    Mirrors R's ``wrap_dims()`` (facet-wrap.R:478-493): when both
    nrow and ncol are ``NULL``, uses ``n2mfrow`` and swaps the axes
    so n=3 → 1×3 (not 2×2). R exports ``wrap_dims`` publicly
    (NAMESPACE:757); this Python port matches that visibility.

    Parameters
    ----------
    n : int
        Number of panels.
    nrow, ncol : int or None

    Returns
    -------
    tuple of (nrow, ncol)

    Raises
    ------
    ValueError
        If the grid is too small for *n* panels.
    """
    if nrow is None and ncol is None:
        # R: rc <- n2mfrow(n); nrow <- rc[2]; ncol <- rc[1]
        rc = _n2mfrow(n)
        nrow = rc[1]
        ncol = rc[0]
    elif ncol is None:
        ncol = math.ceil(n / nrow)
    elif nrow is None:
        nrow = math.ceil(n / ncol)

    if nrow * ncol < n:
        cli_abort(
            f"Need {n} panels, but nrow*ncol = {nrow * ncol}. "
            "Increase nrow and/or ncol."
        )
    return nrow, ncol


def max_height(grobs: Any, value_only: bool = False) -> Any:
    """Largest height over a list of grobs / units, returned in cm.

    Port of R ``facet-.R:1237-1241``:

    .. code-block:: R

        max_height <- function(grobs, value_only = FALSE) {
          height <- max(unlist(lapply(grobs, height_cm)))
          if (!value_only) height <- unit(height, "cm")
          height
        }

    R NAMESPACE exports this (``export(max_height)``); Python port
    matches that visibility. Used by patchwork's
    ``R/collect_axes.R:173`` and other ggplot2-extension packages.

    Parameters
    ----------
    grobs : iterable of Grob / Unit / numeric
        Elements whose heights are compared.
    value_only : bool, default False
        If True, return a bare float in cm; otherwise return a
        ``grid_py.Unit`` carrying cm.

    Returns
    -------
    grid_py.Unit or float
    """
    from ._utils import height_cm
    from grid_py import Unit

    heights = [np.atleast_1d(height_cm(g)) for g in grobs]
    if len(heights) == 0:
        h = float("-inf")
    else:
        h = float(np.max(np.concatenate(heights)))
    if value_only:
        return h
    return Unit(h, "cm")


def max_width(grobs: Any, value_only: bool = False) -> Any:
    """Largest width over a list of grobs / units, returned in cm.

    Port of R ``facet-.R:1244-1248``:

    .. code-block:: R

        max_width <- function(grobs, value_only = FALSE) {
          width <- max(unlist(lapply(grobs, width_cm)))
          if (!value_only) width <- unit(width, "cm")
          width
        }

    R NAMESPACE exports this (``export(max_width)``).

    Parameters
    ----------
    grobs : iterable of Grob / Unit / numeric
    value_only : bool, default False

    Returns
    -------
    grid_py.Unit or float
    """
    from ._utils import width_cm
    from grid_py import Unit

    widths = [np.atleast_1d(width_cm(g)) for g in grobs]
    if len(widths) == 0:
        w = float("-inf")
    else:
        w = float(np.max(np.concatenate(widths)))
    if value_only:
        return w
    return Unit(w, "cm")


def _seam_axes(
    gt: Any,
    grobs: List[Any],
    side: str,
    name: str,
    z: int = 1,
    spacing: Any = None,
    panel_col_to_gtable: Optional[Dict[int, int]] = None,
    panel_row_to_gtable: Optional[Dict[int, int]] = None,
) -> Tuple[Any, Dict[int, int], Dict[int, int]]:
    """Port of R ``seam_table`` (facet-grid-.R:436-478) for axis grobs.

    Adds **one** row (top/bottom) or **one** column (left/right) at the
    outer edge of the panel area, with each panel-col / panel-row's grob
    placed in its corresponding gtable slot. Updates the panel→gtable
    maps in-place so the caller can chain further seam_axes / strip
    operations.

    *grobs*: list whose length equals the number of panel cols
    (top/bottom) or panel rows (left/right). Blank entries become
    ``null_grob`` of zero size.
    """
    from grid_py import Unit as unit, null_grob
    from gtable_py import gtable_add_grob, gtable_add_rows, gtable_add_cols

    if grobs is None or len(grobs) == 0:
        return gt, panel_col_to_gtable, panel_row_to_gtable

    def _blank(g: Any) -> bool:
        return g is None or _is_null_grob(g)

    pcm = dict(panel_col_to_gtable or {})
    prm = dict(panel_row_to_gtable or {})

    if side in ("top", "bottom"):
        panel_rows = sorted(set(prm.values()))
        if side == "bottom":
            row_pos = panel_rows[-1]                # max(panel_row$b) + shift - 1, shift=1
        else:                                       # top
            row_pos = panel_rows[0] - 1             # min(panel_row$t) - shift, shift=1
        if spacing is not None:
            gt = gtable_add_rows(gt, spacing, row_pos)
            for k in prm:
                if prm[k] > row_pos:
                    prm[k] += 1
            if side == "bottom":
                row_pos += 1
        # R: ``max_height(grobs)`` returns 0 cm when every entry is a
        # zeroGrob — the row is still added and grobs placed (so axis-t-1..ncol
        # exist as zero-sized layout entries). Mirror that.
        live_heights = [
            _axis_height_cm(g) for g in grobs if not _blank(g)
        ]
        height_cm = max(live_heights) if live_heights else 0.0
        gt = gtable_add_rows(gt, unit([height_cm], "cm"), row_pos)
        for k in prm:
            if prm[k] > row_pos:
                prm[k] += 1
        new_row = row_pos + 1
        for i, g in enumerate(grobs):
            panel_c = i + 1
            col = pcm.get(panel_c, panel_c)
            gt = gtable_add_grob(
                gt, g if not _blank(g) else null_grob(),
                t=new_row, l=col, clip="off",
                name=f"{name}-{panel_c}", z=z,
            )
        return gt, pcm, prm

    # left / right
    panel_cols_pos = sorted(set(pcm.values()))
    if side == "right":
        col_pos = panel_cols_pos[-1]
    else:
        col_pos = panel_cols_pos[0] - 1
    if spacing is not None:
        gt = gtable_add_cols(gt, spacing, col_pos)
        for k in pcm:
            if pcm[k] > col_pos:
                pcm[k] += 1
        if side == "right":
            col_pos += 1
    live_widths = [_axis_width_cm(g) for g in grobs if not _blank(g)]
    width_cm = max(live_widths) if live_widths else 0.0
    gt = gtable_add_cols(gt, unit([width_cm], "cm"), col_pos)
    for k in pcm:
        if pcm[k] > col_pos:
            pcm[k] += 1
    new_col = col_pos + 1
    for i, g in enumerate(grobs):
        panel_r = i + 1
        row = prm.get(panel_r, panel_r)
        gt = gtable_add_grob(
            gt, g if not _blank(g) else null_grob(),
            t=row, l=new_col, clip="off",
            name=f"{name}-{panel_r}", z=z,
        )
    return gt, pcm, prm


def _resolve_facet_vars(facets: Any) -> List[str]:
    """Resolve *facets* specification to a list of column-name strings.

    Parameters
    ----------
    facets : str, list, tuple, or None
        Faceting variable specification.

    Returns
    -------
    list of str
    """
    if facets is None:
        return []
    if isinstance(facets, str):
        # Could be formula-like "a + b" or simple name
        parts = [s.strip() for s in facets.replace("~", " ").replace("+", " ").split()]
        return [p for p in parts if p and p != "."]
    if isinstance(facets, (list, tuple)):
        result = []
        for f in facets:
            if isinstance(f, str):
                result.append(f)
            else:
                result.append(str(f))
        return result
    if isinstance(facets, dict):
        return list(facets.keys())
    return []


def _combine_vars(
    data_list: List[pd.DataFrame],
    vars_: List[str],
    drop: bool = True,
) -> pd.DataFrame:
    """Combine the unique values of *vars_* across all datasets.

    Parameters
    ----------
    data_list : list of DataFrame
    vars_ : list of str
    drop : bool

    Returns
    -------
    pd.DataFrame
        Unique combinations of the faceting variables.
    """
    if not vars_:
        return pd.DataFrame()

    frames = []
    for df in data_list:
        if df is None or (isinstance(df, pd.DataFrame) and len(df) == 0):
            continue
        cols = [c for c in vars_ if c in df.columns]
        if cols:
            frames.append(df[cols].drop_duplicates())

    if not frames:
        return pd.DataFrame({v: pd.Series(dtype=object) for v in vars_})

    combined = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    # Fill missing columns
    for v in vars_:
        if v not in combined.columns:
            combined[v] = "(all)"
    combined = combined[vars_].reset_index(drop=True)
    # R (facet-.R: combine_vars calls df_layout which runs unique +
    # sort via reorder/id on the faceting vars): for non-factor
    # inputs, panel order follows ``sort(unique(x))``.  Factor inputs
    # keep level order.  Mirrors the same alphabetical rule we fixed
    # for discrete scales in scales_py/range.py.
    sort_cols = [c for c in vars_
                 if c in combined.columns
                 and not hasattr(combined[c], "cat")]
    if sort_cols:
        try:
            combined = combined.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        except TypeError:
            # Mixed / unsortable types — fall back to insertion order
            pass
    return combined


def _map_facet_data(
    data: pd.DataFrame,
    layout: pd.DataFrame,
    params: Dict[str, Any],
    facet_vars: List[str],
) -> pd.DataFrame:
    """Map data rows to panels.

    Parameters
    ----------
    data : pd.DataFrame
        Layer data.
    layout : pd.DataFrame
        Layout with faceting variable columns and ``PANEL``.
    params : dict
    facet_vars : list of str

    Returns
    -------
    pd.DataFrame
        Data with a ``PANEL`` column.
    """
    if data is None or (isinstance(data, pd.DataFrame) and len(data) == 0):
        return pd.DataFrame({"PANEL": pd.Categorical([])})

    if is_waiver(data):
        return pd.DataFrame({"PANEL": pd.Categorical([])})

    data = data.copy()
    if not facet_vars:
        data["PANEL"] = pd.Categorical([1] * len(data))
        return data

    # Match data to layout on facet vars
    present = [v for v in facet_vars if v in data.columns and v in layout.columns]
    if not present:
        # No matching vars: repeat across all panels
        data["PANEL"] = pd.Categorical([1] * len(data))
        return data

    # Merge to get PANEL assignment
    merged = data.merge(
        layout[present + ["PANEL"]],
        on=present,
        how="left",
    )
    # Rows that didn't match any panel get dropped
    merged = merged.dropna(subset=["PANEL"]).reset_index(drop=True)
    merged["PANEL"] = pd.Categorical(merged["PANEL"])
    return merged


# ---------------------------------------------------------------------------
# Base Facet
# ---------------------------------------------------------------------------

class Facet(GGProto):
    """Base facet class.

    Attributes
    ----------
    shrink : bool
        Whether to shrink scales to fit stat output.
    params : dict
        Faceting parameters (populated by the constructor).
    """

    # --- Auto-registration registry (Python-exclusive) -------------------
    _registry: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if name.startswith("Facet") and len(name) > 5:
            key = name[5:]
            Facet._registry[key] = cls
            Facet._registry[key.lower()] = cls

    shrink: bool = False
    params: Dict[str, Any] = {}

    def setup_params(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and modify faceting parameters.

        Parameters
        ----------
        data : list of DataFrame
            Global + layer data.
        params : dict

        Returns
        -------
        dict
        """
        all_cols: List[str] = []
        for df in data:
            if isinstance(df, pd.DataFrame):
                all_cols.extend(df.columns.tolist())
        params["_possible_columns"] = list(set(all_cols))
        return params

    def setup_data(
        self, data: List[pd.DataFrame], params: Dict[str, Any]
    ) -> List[pd.DataFrame]:
        """Modify data before processing.

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        list of DataFrame
        """
        return data

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Create the panel layout table.

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
            Must have ``PANEL``, ``ROW``, ``COL``, ``SCALE_X``, ``SCALE_Y``.

        Raises
        ------
        NotImplementedError
            In the base class.
        """
        cli_abort("compute_layout() is not implemented in the base Facet class.")

    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Assign data rows to panels via the ``PANEL`` column.

        Parameters
        ----------
        data : pd.DataFrame
        layout : pd.DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        cli_abort("map_data() is not implemented in the base Facet class.")

    def init_scales(
        self,
        layout: pd.DataFrame,
        x_scale: Any = None,
        y_scale: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, list]:
        """Initialise per-panel scales.

        Parameters
        ----------
        layout : pd.DataFrame
        x_scale, y_scale : Scale or None
            Prototype scales.
        params : dict

        Returns
        -------
        dict
            ``{"x": [scales...], "y": [scales...]}``.
        """
        # R parity (facet-.R:225-234): clone the prototype scale once per
        # panel so free_x / free_y can train each independently. A bare
        # ``[x_scale] * n`` produces N aliases of the same object, which
        # collapses per-panel training into a single union range.
        scales: Dict[str, list] = {}
        if x_scale is not None:
            n_x = int(layout["SCALE_X"].max())
            scales["x"] = [x_scale.clone() for _ in range(n_x)]
        if y_scale is not None:
            n_y = int(layout["SCALE_Y"].max())
            scales["y"] = [y_scale.clone() for _ in range(n_y)]
        return scales

    def train_scales(
        self,
        x_scales: list,
        y_scales: list,
        layout: pd.DataFrame,
        data: List[pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train per-panel scales on data.

        Parameters
        ----------
        x_scales, y_scales : list
        layout : pd.DataFrame
        data : list of DataFrame
        params : dict
        """
        for layer_data in data:
            if layer_data is None or (hasattr(layer_data, "empty") and layer_data.empty):
                continue
            if "PANEL" not in layer_data.columns:
                continue
            for _, row in layout.iterrows():
                panel_id = row["PANEL"]
                sx_idx = int(row["SCALE_X"]) - 1
                sy_idx = int(row["SCALE_Y"]) - 1
                mask = layer_data["PANEL"] == panel_id
                panel_data = layer_data.loc[mask]
                if panel_data.empty:
                    continue
                if x_scales and sx_idx < len(x_scales):
                    x_scales[sx_idx].train_df(panel_data)
                if y_scales and sy_idx < len(y_scales):
                    y_scales[sy_idx].train_df(panel_data)

    def finish_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Final data adjustments.

        Parameters
        ----------
        data : pd.DataFrame
        layout : pd.DataFrame
        x_scales, y_scales : list
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        return data

    def draw_panels(
        self,
        panels: list,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        ranges: list,
        coord: Any,
        data: Any,
        theme: Any,
        params: Dict[str, Any],
    ) -> Any:
        """Assemble panels into a gtable with background, axes, and labels.

        Mirrors R's ``Facet$draw_panels`` → ``init_gtable`` → ``attach_axes``
        pipeline (facet-.R:501-532):

        1. Create panel-only gtable with null units (init_gtable)
        2. Decorate each panel with coord background + foreground
        3. Render axis grobs, measure them, and attach as new rows/columns

        Parameters
        ----------
        panels : list of grobs (per-layer, each containing per-panel grobs)
        layout : pd.DataFrame
        x_scales, y_scales : list
        ranges : list of panel_params dicts
        coord : Coord
        data : list
        theme : Theme
        params : dict

        Returns
        -------
        gtable
        """
        from grid_py import GTree, GList, null_grob, Viewport
        from gtable_py import (
            Gtable, gtable_add_grob, gtable_add_rows, gtable_add_cols,
            gtable_add_col_space, gtable_add_row_space,
        )
        from grid_py import Unit as unit
        from ggplot2_py.theme_elements import calc_element as _calc_el

        nrow = int(layout["ROW"].max()) if len(layout) > 0 else 1
        ncol = int(layout["COL"].max()) if len(layout) > 0 else 1

        # ── Step 1: init_gtable — panel-only matrix (R: facet-.R:562-612)
        # Panel sizes use "null" units (flexible, fill available space).
        # Aspect ratio from coord is encoded in the null-unit ratio.
        aspect_ratio = None
        if hasattr(coord, "aspect") and ranges:
            aspect_ratio = coord.aspect(ranges[0])

        panel_h = abs(aspect_ratio) if aspect_ratio is not None else 1.0
        widths = unit([1] * ncol, "null")
        heights = unit([panel_h] * nrow, "null")
        gt = Gtable(widths=widths, heights=heights, name="layout")

        # Mark respect flag for aspect ratio (R: facet-.R:592)
        if aspect_ratio is not None:
            gt._respect = True

        # ── Step 2: Place decorated panels into the gtable.
        # R ``init_gtable`` (facet-.R:566-602) starts from a full
        # ``matrix(list(zeroGrob()), nrow, ncol)`` and *overwrites* the
        # active cells with real panels. Empty cells therefore still get
        # a ``panel-{c}-{r}`` layout entry (as a zeroGrob). Mirror that
        # by walking the full nrow×ncol grid first.
        active = {(int(row["ROW"]), int(row["COL"])): row
                  for _, row in layout.iterrows()}
        for r in range(1, nrow + 1):
            for c in range(1, ncol + 1):
                key = (r, c)
                if key in active:
                    row_info = active[key]
                    panel_id = int(row_info["PANEL"])
                    panel_idx = panel_id - 1
                    pp = ranges[panel_idx] if panel_idx < len(ranges) else {}
                    panel_grobs = []
                    for layer_grobs in panels:
                        if isinstance(layer_grobs, list) and panel_idx < len(layer_grobs):
                            panel_grobs.append(layer_grobs[panel_idx])
                        elif not isinstance(layer_grobs, list) and layer_grobs is not None:
                            panel_grobs.append(layer_grobs)
                    if hasattr(coord, "draw_panel"):
                        decorated = coord.draw_panel(panel_grobs, pp, theme)
                    else:
                        decorated = GTree(
                            children=GList(*panel_grobs),
                            name=f"panel-{panel_id}",
                        )
                else:
                    decorated = null_grob()
                # R: panel-{COL}-{ROW} (facet-.R:597-601).
                gt = gtable_add_grob(
                    gt, decorated, t=r, l=c, name=f"panel-{c}-{r}",
                    clip=getattr(coord, "clip", "on"),
                )

        # ── Step 2b: panel spacing — R: facet-.R:604-611.
        # Insert (ncol-1) cols and (nrow-1) rows of theme$panel.spacing.x/y
        # between the panels so per-facet axes have somewhere to sit.
        spacing_x = _calc_el("panel.spacing.x", theme) if theme is not None else None
        if spacing_x is not None:
            gt = gtable_add_col_space(gt, spacing_x)
        spacing_y = _calc_el("panel.spacing.y", theme) if theme is not None else None
        if spacing_y is not None:
            gt = gtable_add_row_space(gt, spacing_y)

        # After spacing insertion the panel grobs sit at the odd gtable
        # cols/rows (1, 3, 5, …); refresh our panel→gtable maps from
        # the layout instead of recomputing manually.
        def _refresh_panel_maps() -> Tuple[Dict[int, int], Dict[int, int]]:
            # Panel names are R-style ``panel-{COL}-{ROW}`` (facet-.R:597-601).
            cmap: Dict[int, int] = {}
            rmap: Dict[int, int] = {}
            names = list(gt.layout["name"])
            for idx, nm in enumerate(names):
                if not isinstance(nm, str) or not nm.startswith("panel-"):
                    continue
                parts = nm.split("-")
                if len(parts) != 3:
                    continue
                if not (parts[1].isdigit() and parts[2].isdigit()):
                    continue
                cc = int(parts[1])
                rr = int(parts[2])
                cmap.setdefault(cc, int(gt.layout["l"][idx]))
                rmap.setdefault(rr, int(gt.layout["t"][idx]))
            return cmap, rmap

        panel_col_to_gtable, panel_row_to_gtable = _refresh_panel_maps()
        # Fall back to identity if layout didn't carry the panel labels.
        for c_ in range(1, ncol + 1):
            panel_col_to_gtable.setdefault(c_, c_)
        for r_ in range(1, nrow + 1):
            panel_row_to_gtable.setdefault(r_, r_)

        # ── Step 3: Render axes and weave them between panels.
        # R: ``facet-wrap.R:267-383`` + ``weave_axes``. Axes are rendered
        # per panel, interior axes are blanked unless ``free_x`` /
        # ``free_y`` (or ``draw_axes = "all"``), then the surviving axes
        # are woven into the gtable one column (or row) per panel column
        # (or row). R loops every panel column / row unconditionally so
        # the gtable shape stays predictable; even an interior-blanked
        # axis still occupies a 0-cm column.

        free = params.get("free", {"x": False, "y": False})
        draw_axes = params.get("draw_axes", {"x": False, "y": False})
        show_all_x = bool(free.get("x", False)) or bool(draw_axes.get("x", False))
        show_all_y = bool(free.get("y", False)) or bool(draw_axes.get("y", False))

        def _grid() -> List[List[Any]]:
            return [[None] * ncol for _ in range(nrow)]

        left_grid = _grid(); right_grid = _grid()
        top_grid = _grid();  bottom_grid = _grid()

        for _, row_info in layout.iterrows():
            panel_idx = int(row_info["PANEL"]) - 1
            r = int(row_info["ROW"]); c = int(row_info["COL"])
            pp = ranges[panel_idx] if panel_idx < len(ranges) else {}
            if hasattr(coord, "render_axis_v"):
                v = coord.render_axis_v(pp, theme)
                left_grid[r - 1][c - 1] = v.get("left")
                right_grid[r - 1][c - 1] = v.get("right")
            if hasattr(coord, "render_axis_h"):
                h = coord.render_axis_h(pp, theme)
                top_grid[r - 1][c - 1] = h.get("top")
                bottom_grid[r - 1][c - 1] = h.get("bottom")

        def _blank(grob: Any) -> bool:
            return grob is None or _is_null_grob(grob)

        # Mirror R facet-wrap.R:297-304 — suppress interior axes unless free.
        if not show_all_y:
            for r in range(nrow):
                for c in range(1, ncol):          # cols 2..ncol: blank left
                    left_grid[r][c] = None
                for c in range(0, ncol - 1):      # cols 1..ncol-1: blank right
                    right_grid[r][c] = None
        if not show_all_x:
            for c in range(ncol):
                for r in range(1, nrow):          # rows 2..nrow: blank top
                    top_grid[r][c] = None
                for r in range(0, nrow - 1):      # rows 1..nrow-1: blank bottom
                    bottom_grid[r][c] = None

        # Delegate axis attachment to a subclass-overridable method —
        # mirror R's ``Facet$attach_axes`` / ``FacetWrap$attach_axes`` /
        # ``FacetGrid$attach_axes`` polymorphism (facet-.R:643-645,
        # facet-wrap.R:267-378, facet-grid-.R:299-341).
        gt, panel_col_to_gtable, panel_row_to_gtable = self._attach_axes(
            gt, nrow, ncol,
            left_grid, right_grid, top_grid, bottom_grid,
            panel_col_to_gtable, panel_row_to_gtable,
            theme,
        )

        # Strip labels receive the panel→gtable offset maps so they align
        # with panels whichever interleaving pattern the axes used.
        col_offset_for_first_panel = panel_col_to_gtable[1] - 1
        gt = self._add_strip_labels(
            gt, layout, nrow, ncol, params, theme,
            col_offset=col_offset_for_first_panel,
            panel_col_to_gtable=panel_col_to_gtable,
            panel_row_to_gtable=panel_row_to_gtable,
        )

        return gt

    # ------------------------------------------------------------------
    # attach_axes — overridden by FacetWrap (weave_axes) / FacetGrid
    # (seam_table). The base default is the wrap-style emission so that
    # FacetWrap inherits it without restating, matching R's S3 dispatch
    # where ``Facet$attach_axes`` is a no-op but the wrap-style behaviour
    # is the most common downstream expectation.
    # ------------------------------------------------------------------
    def _attach_axes(
        self,
        gt: Any,
        nrow: int,
        ncol: int,
        left_grid: List[List[Any]],
        right_grid: List[List[Any]],
        top_grid: List[List[Any]],
        bottom_grid: List[List[Any]],
        panel_col_to_gtable: Dict[int, int],
        panel_row_to_gtable: Dict[int, int],
        theme: Any,
    ) -> Tuple[Any, Dict[int, int], Dict[int, int]]:
        """R: ``Facet$attach_axes`` (facet-.R:643-645) — default no-op."""
        return gt, panel_col_to_gtable, panel_row_to_gtable

    def _add_strip_labels(
        self,
        gt: Any,
        layout: pd.DataFrame,
        nrow: int,
        ncol: int,
        params: Dict[str, Any],
        theme: Any = None,
        col_offset: int = 0,
        panel_col_to_gtable: Optional[Dict[int, int]] = None,
        panel_row_to_gtable: Optional[Dict[int, int]] = None,
    ) -> Any:
        """Add facet strip text labels to the gtable.

        All visual properties resolved from theme via ``calc_element()``
        for strip.background.x/y and strip.text.x/y.
        """
        from gtable_py import gtable_add_grob, gtable_add_rows, gtable_add_cols
        from grid_py import Unit as unit, text_grob, Gpar, rect_grob
        from grid_py._grob import grob_tree, GList
        from ggplot2_py.theme_elements import calc_element as _calc_el

        # R always has a complete theme at this point.  If Python's theme
        # is None, fall back to theme_grey() to surface real bugs rather
        # than None-attribute errors.
        if theme is None:
            from ggplot2_py.theme_defaults import theme_grey
            theme = theme_grey()

        meta_cols = {"PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y", "COORD"}
        facet_vars = [c for c in layout.columns if c not in meta_cols]
        if not facet_vars:
            return gt

        col_vars = _resolve_facet_vars(params.get("cols"))
        row_vars = _resolve_facet_vars(params.get("rows"))
        wrap_vars = _resolve_facet_vars(params.get("facets"))

        # Resolve labeller function
        from ggplot2_py.labeller import as_labeller, label_value
        labeller_spec = params.get("labeller", "label_value")
        try:
            labeller_fn = as_labeller(labeller_spec)
        except (ValueError, TypeError):
            labeller_fn = label_value

        # Resolve strip theme elements via calc_element (proper inheritance).
        # R always has a complete theme with strip elements defined.
        # If calc_element returns None, the element tree is incomplete
        # — reset it and retry with a guaranteed-complete theme.
        from ggplot2_py.theme_elements import ElementBlank as _EB
        strip_txt_x_el = _calc_el("strip.text.x", theme)
        if strip_txt_x_el is None:
            from ggplot2_py.theme_elements import reset_theme_settings
            reset_theme_settings()
            from ggplot2_py.theme_defaults import theme_grey
            theme = theme_grey()
            strip_txt_x_el = _calc_el("strip.text.x", theme)
        strip_bg_x_el = _calc_el("strip.background.x", theme)
        strip_txt_y_el = _calc_el("strip.text.y", theme)
        strip_bg_y_el = _calc_el("strip.background.y", theme)

        def _props(el, attrs):
            """Extract attrs from an element, returning None for ElementBlank.

            R: when a strip element is ``element_blank()``, the
            corresponding grob is simply a ``zeroGrob()``.  We surface
            this by mapping each attr to ``None``.
            """
            if el is None or isinstance(el, _EB):
                return {k: None for k in attrs}
            return {k: getattr(el, k, None) for k in attrs}

        strip_txt_x = _props(strip_txt_x_el, ["colour", "size", "angle"])
        strip_bg_x  = _props(strip_bg_x_el,  ["fill", "colour"])
        strip_txt_y = _props(strip_txt_y_el, ["colour", "size", "angle"])
        strip_bg_y  = _props(strip_bg_y_el,  ["fill", "colour"])

        _txt_blank_x = isinstance(strip_txt_x_el, _EB)
        _bg_blank_x  = isinstance(strip_bg_x_el,  _EB)
        _txt_blank_y = isinstance(strip_txt_y_el, _EB)
        _bg_blank_y  = isinstance(strip_bg_y_el,  _EB)

        def _make_strip(label_text, bg_el, txt_el, rot, name,
                         bg_blank=False, txt_blank=False):
            """Compose a strip: optional rect bg + optional text.

            R: ElementBlank → zeroGrob, omitted from the output.
            """
            from grid_py import null_grob as _null
            bg = (
                _null()
                if bg_blank or bg_el.get("fill") is None and bg_el.get("colour") is None
                else rect_grob(
                    x=0.5, y=0.5, width=1, height=1,
                    gp=Gpar(fill=bg_el.get("fill"), col=bg_el.get("colour")),
                    name=f"strip.bg.{name}",
                )
            )
            txt = (
                _null()
                if txt_blank or txt_el.get("size") is None
                else text_grob(
                    label=label_text, x=0.5, y=0.5, rot=rot, just="centre",
                    gp=Gpar(fontsize=float(txt_el["size"]),
                            col=txt_el.get("colour")),
                    name=f"strip.text.{name}",
                )
            )
            return grob_tree(bg, txt, name=f"strip-{name}")

        def _get_strip_text(vars_list, row_info):
            """Get formatted strip label text using the labeller."""
            lab_dict = {v: [str(row_info.get(v, ""))] for v in vars_list}
            result = labeller_fn(lab_dict)
            return result[0] if result else ""

        # Helper: measure strip text height/width in cm
        # R: assemble_strips → max_height(grobs) / max_width(grobs)
        from grid_py._size import calc_string_metric

        def _strip_height_cm(labels, txt_el):
            """Max height of strip labels (R: max_height(strip_grobs))."""
            fs = float(txt_el.get("size") or 8)
            max_h = 0.0
            for lbl in labels:
                m = calc_string_metric(str(lbl), Gpar(fontsize=fs))
                max_h = max(max_h, (m["ascent"] + m["descent"]) * 2.54)
            # Add small padding for strip background
            return max(max_h + 0.1, 0.2)

        def _strip_width_cm(labels, txt_el):
            """Max width of strip labels (R: max_width(strip_grobs))."""
            fs = float(txt_el.get("size") or 8)
            max_w = 0.0
            for lbl in labels:
                m = calc_string_metric(str(lbl), Gpar(fontsize=fs))
                max_w = max(max_w, (m["ascent"] + m["descent"]) * 2.54)
            return max(max_w + 0.1, 0.2)

        # --- facet_wrap ---
        if wrap_vars and not col_vars and not row_vars:
            # R wrap strips honour params$strip.position; we mirror that
            # so patchwork's strip-collecting code (which keys off the
            # R-style ``strip-{t,b,l,r}-{col}-{row}`` prefix) sees them.
            position = str(params.get("strip.position", "top"))
            pos_letter = position[0]            # "t" | "b" | "l" | "r"

            # Map wrap-strip cell coordinates → gtable coordinates so the
            # post-spacing layout (panels at odd rows/cols) stays aligned.
            pcol_to_gtable = panel_col_to_gtable or {c: c + col_offset
                                                     for c in range(1, ncol + 1)}
            prow_to_gtable = panel_row_to_gtable or {r: r for r in range(1, nrow + 1)}

            # Collect all wrap labels to measure max height/width.
            all_wrap_labels = []
            for _, row_info in layout.iterrows():
                all_wrap_labels.append(_get_strip_text(wrap_vars, row_info))

            # R ``weave_tables_row/col`` (facet-wrap.R:501-524) places a
            # strip in *every* (row, col) panel cell of the full grid —
            # for empty cells (``layout`` doesn't include them) the
            # strip is a zeroGrob entry so the layout-name coverage
            # stays stable. Build a {(row,col): row_info} dict so the
            # placement loop can emit a null_grob for missing cells.
            from grid_py import null_grob as _null
            active_rc = {(int(r["ROW"]), int(r["COL"])): r
                         for _, r in layout.iterrows()}

            if position in ("top", "bottom"):
                strip_size = _strip_height_cm(all_wrap_labels, strip_txt_x)
                row_shift = -1 if position == "top" else 0   # R: shift = c(-1, 0)
                for panel_r in range(nrow, 0, -1):
                    cur_row = prow_to_gtable[panel_r]
                    gt = gtable_add_rows(gt, unit([strip_size], "cm"),
                                          pos=cur_row + row_shift)
                    new_row = cur_row + row_shift + 1
                    for rr in range(panel_r if row_shift == -1 else panel_r + 1,
                                    nrow + 1):
                        prow_to_gtable[rr] += 1
                    for c in range(1, ncol + 1):
                        row_info = active_rc.get((panel_r, c))
                        if row_info is not None:
                            label_text = _get_strip_text(wrap_vars, row_info)
                            strip = _make_strip(
                                label_text, strip_bg_x, strip_txt_x, 0,
                                f"{pos_letter}-{c}-{panel_r}",
                                bg_blank=_bg_blank_x, txt_blank=_txt_blank_x,
                            )
                        else:
                            strip = _null()
                        gt = gtable_add_grob(
                            gt, strip,
                            t=new_row, l=pcol_to_gtable[c],
                            clip="off",
                            name=f"strip-{pos_letter}-{c}-{panel_r}",
                        )
            else:  # "left" or "right"
                strip_size = _strip_width_cm(all_wrap_labels, strip_txt_y)
                col_shift = -1 if position == "left" else 0
                rot = float((strip_txt_y or {}).get("angle") or 0)
                for panel_c in range(ncol, 0, -1):
                    cur_col = pcol_to_gtable[panel_c]
                    gt = gtable_add_cols(gt, unit([strip_size], "cm"),
                                          pos=cur_col + col_shift)
                    new_col = cur_col + col_shift + 1
                    for cc in range(panel_c if col_shift == -1 else panel_c + 1,
                                    ncol + 1):
                        pcol_to_gtable[cc] += 1
                    for r in range(1, nrow + 1):
                        row_info = active_rc.get((r, panel_c))
                        if row_info is not None:
                            label_text = _get_strip_text(wrap_vars, row_info)
                            strip = _make_strip(
                                label_text, strip_bg_y, strip_txt_y, rot,
                                f"{pos_letter}-{panel_c}-{r}",
                                bg_blank=_bg_blank_y, txt_blank=_txt_blank_y,
                            )
                        else:
                            strip = _null()
                        gt = gtable_add_grob(
                            gt, strip,
                            t=prow_to_gtable[r], l=new_col,
                            clip="off",
                            name=f"strip-{pos_letter}-{panel_c}-{r}",
                        )
            return gt

        # --- facet_grid strips (R: ``FacetGrid$attach_strips``,
        # facet-grid-.R:343-386). ``switch`` selects the side:
        #   "x" / "both" → bottom strips (strip-b)
        #   "y" / "both" → left strips   (strip-l)
        # otherwise top + right.
        switch = params.get("switch")
        switch_x = switch is not None and switch in ("both", "x")
        switch_y = switch is not None and switch in ("both", "y")
        pcm = dict(panel_col_to_gtable or {})
        prm = dict(panel_row_to_gtable or {})

        if col_vars:
            col_labels: List[str] = []
            for c in range(1, ncol + 1):
                panel_row = layout[layout["COL"] == c].iloc[0]
                col_labels.append(_get_strip_text(col_vars, panel_row))
            strip_h = (_strip_height_cm(col_labels, strip_txt_x)
                       if not _txt_blank_x else 0.2)
            if switch_x:
                # strip-b: insert below bottom panel row.
                bottom_panel_row = max(prm.values())
                gt = gtable_add_rows(gt, unit([strip_h], "cm"), pos=bottom_panel_row)
                for k in prm:
                    if prm[k] > bottom_panel_row:
                        prm[k] += 1
                new_row = bottom_panel_row + 1
                pos_letter = "b"
            else:
                # strip-t: insert above top panel row.
                top_panel_row = min(prm.values())
                gt = gtable_add_rows(gt, unit([strip_h], "cm"), pos=top_panel_row - 1)
                for k in prm:
                    if prm[k] >= top_panel_row:
                        prm[k] += 1
                new_row = top_panel_row
                pos_letter = "t"
            for c in range(1, ncol + 1):
                panel_row = layout[layout["COL"] == c].iloc[0]
                label_text = _get_strip_text(col_vars, panel_row)
                strip = _make_strip(
                    label_text, strip_bg_x, strip_txt_x, 0, f"{pos_letter}-{c}",
                    bg_blank=_bg_blank_x, txt_blank=_txt_blank_x,
                )
                gt = gtable_add_grob(
                    gt, strip,
                    t=new_row, l=pcm.get(c, c),
                    clip="off", name=f"strip-{pos_letter}-{c}",
                )

        if row_vars:
            row_labels: List[str] = []
            for r in range(1, nrow + 1):
                panel_row = layout[layout["ROW"] == r].iloc[0]
                row_labels.append(_get_strip_text(row_vars, panel_row))
            strip_w = (_strip_width_cm(row_labels, strip_txt_y)
                       if not _txt_blank_y else 0.2)
            rot = float((strip_txt_y or {}).get("angle") or 0)
            if switch_y:
                # strip-l: insert immediately left of leftmost panel col.
                left_panel_col = min(pcm.values())
                gt = gtable_add_cols(gt, unit([strip_w], "cm"), pos=left_panel_col - 1)
                for k in pcm:
                    if pcm[k] >= left_panel_col:
                        pcm[k] += 1
                new_col = left_panel_col
                pos_letter = "l"
            else:
                right_panel_col = max(pcm.values())
                gt = gtable_add_cols(gt, unit([strip_w], "cm"), pos=right_panel_col)
                for k in pcm:
                    if pcm[k] > right_panel_col:
                        pcm[k] += 1
                new_col = right_panel_col + 1
                pos_letter = "r"
            for r in range(1, nrow + 1):
                panel_row = layout[layout["ROW"] == r].iloc[0]
                label_text = _get_strip_text(row_vars, panel_row)
                strip = _make_strip(
                    label_text, strip_bg_y, strip_txt_y, rot, f"{pos_letter}-{r}",
                    bg_blank=_bg_blank_y, txt_blank=_txt_blank_y,
                )
                gt = gtable_add_grob(
                    gt, strip,
                    t=prm.get(r, r), l=new_col,
                    clip="off", name=f"strip-{pos_letter}-{r}",
                )

        return gt

    def draw_labels(
        self,
        panels: Any,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        ranges: list,
        coord: Any,
        data: Any,
        theme: Any,
        labels: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Any:
        """Attach x/y axis titles to the panels table.

        Ports R's ``Facet$draw_labels`` (ggplot2/R/facet-.R:824-850)
        verbatim: unconditionally adds a top row + bottom row for the
        x-axis titles and a left col + right col for the y-axis titles,
        then places each label grob with the directional name R uses
        (``xlab-t`` / ``xlab-b`` / ``ylab-l`` / ``ylab-r``). Downstream
        consumers such as patchwork's ``collect_axis_titles`` key off
        those names. Empty (null) grobs still take up a zero-sized
        row/column so the layout shape is stable.

        Note on ordering: R's ``labels$x`` is ``list(top, bottom)``
        (R's ``c("top", "bottom")[i]`` in ``Layout$render_labels``).
        ``ggplot2_py``'s ``render_labels`` currently returns the pair
        as ``[primary, secondary]`` — primary is bottom, secondary is
        top — so we swap when dispatching: ``pair[1]`` → top,
        ``pair[0]`` → bottom. Same mapping for y: ``pair[0]`` → left
        (primary), ``pair[1]`` → right (secondary).

        Parameters
        ----------
        panels : gtable
        labels : dict
            Rendered label grobs keyed by ``"x"`` / ``"y"``, each a
            two-element list ``[primary, secondary]``.
        """
        from gtable_py import gtable_add_grob, gtable_add_rows, gtable_add_cols
        from grid_py import grob_height, grob_width, null_grob

        gt = panels

        def _pair_for(axis: str, index: int) -> Any:
            pair = labels.get(axis)
            if isinstance(pair, list) and len(pair) > index:
                v = pair[index]
                if v is not None:
                    return v
            return null_grob()

        # x-axis: top = secondary (pair[1]), bottom = primary (pair[0]).
        x_top    = _pair_for("x", 1)
        x_bottom = _pair_for("x", 0)

        # Add top row and place xlab-t.
        gt = gtable_add_rows(gt, grob_height(x_top), pos=0)
        # Panel column span uses find_panel; after the row insertion the
        # panel row has shifted down by 1, so find_panel on the new gt
        # is reliable for column coordinates.
        from .plot_render import find_panel as _find_panel
        pdim = _find_panel(gt)
        gt = gtable_add_grob(
            gt, x_top,
            t=1, l=pdim["l"], b=1, r=pdim["r"],
            clip="off", name="xlab-t",
        )

        # Add bottom row and place xlab-b.
        gt = gtable_add_rows(gt, grob_height(x_bottom), pos=-1)
        nrow = len(gt._heights)
        pdim = _find_panel(gt)
        gt = gtable_add_grob(
            gt, x_bottom,
            t=nrow, l=pdim["l"], b=nrow, r=pdim["r"],
            clip="off", name="xlab-b",
        )

        # y-axis: left = primary (pair[0]), right = secondary (pair[1]).
        y_left  = _pair_for("y", 0)
        y_right = _pair_for("y", 1)

        # Left col.
        gt = gtable_add_cols(gt, grob_width(y_left), pos=0)
        pdim = _find_panel(gt)
        gt = gtable_add_grob(
            gt, y_left,
            t=pdim["t"], l=1, b=pdim["b"], r=1,
            clip="off", name="ylab-l",
        )

        # Right col.
        gt = gtable_add_cols(gt, grob_width(y_right), pos=-1)
        ncol = len(gt._widths)
        pdim = _find_panel(gt)
        gt = gtable_add_grob(
            gt, y_right,
            t=pdim["t"], l=ncol, b=pdim["b"], r=ncol,
            clip="off", name="ylab-r",
        )

        return gt

    def vars(self) -> List[str]:
        """Return the faceting variable names.

        Returns
        -------
        list of str
        """
        return []

    # ------------------------------------------------------------------
    # set_panel_size — R facet-.R:725-770 — enforce theme panel.widths /
    # panel.heights on the assembled gtable. ``layout.py`` calls this
    # immediately after ``draw_panels`` so the size override works for
    # every Facet subclass without each having to call it explicitly.
    # ------------------------------------------------------------------
    def set_panel_size(self, table: Any, theme: Any) -> Any:
        """Resize panel rows / columns to match theme settings.

        Looks up ``panel.widths`` and ``panel.heights`` via
        :func:`theme_elements.calc_element`. If either is set, finds the
        gtable rows / columns whose grobs are panels (name starts with
        ``"panel"``) and overwrites their sizes with the theme values.

        Matches the common case in R; the aspect-ratio reconstruction
        single-panel special case (facet-.R:737-750) is intentionally
        not ported yet.
        """
        if table is None or theme is None:
            return table
        try:
            from ggplot2_py.theme_elements import calc_element as _calc
        except Exception:
            return table

        new_widths = _calc("panel.widths", theme)
        new_heights = _calc("panel.heights", theme)
        if new_widths is None and new_heights is None:
            return table

        # Scan the gtable layout for panel cells. Names matching "panel*"
        # are the R convention (``draw_panels`` uses ``name=f"panel-{r}-{c}"``).
        layout = getattr(table, "layout", None)
        if not isinstance(layout, dict):
            return table
        names = layout.get("name") or []
        t_pos = layout.get("t") or []
        l_pos = layout.get("l") or []

        panel_rows: List[int] = []
        panel_cols: List[int] = []
        for i, nm in enumerate(names):
            if not isinstance(nm, str) or not nm.startswith("panel"):
                continue
            if i < len(t_pos):
                panel_rows.append(int(t_pos[i]))
            if i < len(l_pos):
                panel_cols.append(int(l_pos[i]))
        panel_rows = sorted(set(panel_rows))
        panel_cols = sorted(set(panel_cols))

        def _broadcast(value: Any, n: int) -> List[Any]:
            # R: if length-1, recycle; if length matches n, use as-is.
            try:
                length = len(value)
            except TypeError:
                return [value] * n
            if length == 1:
                return [value[0]] * n
            if length == n:
                return list(value)
            # Mismatched length — bail out rather than silently mis-align.
            return []

        if new_widths is not None and panel_cols:
            widths = _broadcast(new_widths, len(panel_cols))
            tbl_widths = getattr(table, "widths", None)
            if widths and tbl_widths is not None:
                for c, w in zip(panel_cols, widths):
                    try:
                        tbl_widths[c] = w
                    except (TypeError, IndexError):
                        pass

        if new_heights is not None and panel_rows:
            heights = _broadcast(new_heights, len(panel_rows))
            tbl_heights = getattr(table, "heights", None)
            if heights and tbl_heights is not None:
                for r, h in zip(panel_rows, heights):
                    try:
                        tbl_heights[r] = h
                    except (TypeError, IndexError):
                        pass

        return table


# ---------------------------------------------------------------------------
# FacetNull
# ---------------------------------------------------------------------------

class FacetNull(Facet):
    """Single-panel facet (no faceting).

    This is the default when no faceting is specified.
    """

    shrink: bool = True

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        return _layout_null()

    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        if is_waiver(data):
            return pd.DataFrame({"PANEL": pd.Categorical([])})
        if isinstance(data, pd.DataFrame) and len(data) == 0:
            df = data.copy()
            df["PANEL"] = pd.Categorical([])
            return df
        data = data.copy()
        data["PANEL"] = pd.Categorical([1] * len(data))
        return data

    def draw_panels(
        self,
        panels: list,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        ranges: list,
        coord: Any,
        data: Any,
        theme: Any,
        params: Dict[str, Any],
    ) -> Any:
        """Single-panel layout — port of R ``FacetNull$draw_panels``
        (facet-null.R:45-78).

        Builds a 3×3 ``gtable_matrix`` directly with corner spacers,
        edge axes, and the panel in the centre — bypassing the
        ``init_gtable → attach_axes → attach_strips`` pipeline that the
        faceted variants need.
        """
        from grid_py import (
            GTree, GList, null_grob, Unit as unit, grob_height, grob_width,
            unit_c,
        )
        from gtable_py import gtable_matrix

        rng = ranges[0] if ranges else {}
        aspect_ratio = theme.get("aspect.ratio") if hasattr(theme, "get") else None
        if aspect_ratio is None and hasattr(coord, "aspect") and ranges:
            aspect_ratio = coord.aspect(rng)
        respect = aspect_ratio is not None
        if aspect_ratio is None:
            aspect_ratio = 1.0

        axis_h = (coord.render_axis_h(rng, theme)
                  if hasattr(coord, "render_axis_h") else
                  {"top": null_grob(), "bottom": null_grob()})
        axis_v = (coord.render_axis_v(rng, theme)
                  if hasattr(coord, "render_axis_v") else
                  {"left": null_grob(), "right": null_grob()})

        # Compose the panel grob (geom layers + coord background/foreground)
        panel_grobs: list = []
        for layer_grobs in panels:
            if isinstance(layer_grobs, list) and len(layer_grobs) > 0:
                panel_grobs.append(layer_grobs[0])
            elif not isinstance(layer_grobs, list) and layer_grobs is not None:
                panel_grobs.append(layer_grobs)
        if hasattr(coord, "draw_panel"):
            panel = coord.draw_panel(panel_grobs, rng, theme)
        else:
            panel = GTree(children=GList(*panel_grobs), name="panel-1")

        # 3×3 grob matrix mirroring R facet-null.R:60-64.
        zero = null_grob
        all_grobs = [
            [zero(),       axis_h["top"],   zero()],
            [axis_v["left"], panel,         axis_v["right"]],
            [zero(),       axis_h["bottom"], zero()],
        ]
        z_matrix = [
            [5, 6, 4],
            [7, 1, 8],
            [3, 9, 2],
        ]
        # R: ``unit.c(grobWidth(axis_v$left), unit(1, "null"),
        #             grobWidth(axis_v$right))`` (facet-null.R:66) —
        # axis side cells carry **lazy** grobwidth/grobheight units so
        # the resolution to cm happens at draw time off the actual grob,
        # not via a stale snapshot value extracted now.
        widths = unit_c(
            grob_width(axis_v["left"]),
            unit([1.0], ["null"]),
            grob_width(axis_v["right"]),
        )
        heights = unit_c(
            grob_height(axis_h["top"]),
            unit([abs(aspect_ratio)], ["null"]),
            grob_height(axis_h["bottom"]),
        )
        gt = gtable_matrix(
            "layout", all_grobs,
            widths=widths, heights=heights,
            respect=respect, clip="off", z=z_matrix,
        )
        # R: ``layout$layout$name <- grob_names`` (facet-null.R:75) —
        # column-major flatten of the 3×3 matrix.
        gt.layout["name"] = [
            "spacer", "axis-l", "spacer",
            "axis-t", "panel",  "axis-b",
            "spacer", "axis-r", "spacer",
        ]
        return gt


# ---------------------------------------------------------------------------
# FacetGrid
# ---------------------------------------------------------------------------

class FacetGrid(Facet):
    """Grid facet: panels arranged in a row x column matrix.

    Attributes
    ----------
    shrink : bool
    params : dict
        Contains ``rows``, ``cols``, ``scales``, ``space``, ``labeller``,
        ``as_table``, ``switch``, ``drop``, ``margins``, ``free``,
        ``space_free``, ``draw_axes``, ``axis_labels``.
    """

    shrink: bool = True

    def _attach_axes(
        self,
        gt: Any,
        nrow: int,
        ncol: int,
        left_grid: List[List[Any]],
        right_grid: List[List[Any]],
        top_grid: List[List[Any]],
        bottom_grid: List[List[Any]],
        panel_col_to_gtable: Dict[int, int],
        panel_row_to_gtable: Dict[int, int],
        theme: Any,
    ) -> Tuple[Any, Dict[int, int], Dict[int, int]]:
        """R: ``FacetGrid$attach_axes`` (facet-grid-.R:299-341) →
        ``seam_table`` (facet-grid-.R:436-478).

        For default ``draw_axes={x:FALSE, y:FALSE}``, attaches **one**
        axis row above/below and **one** axis col left/right of the
        whole panel grid (vs FacetWrap's per-panel weaving).
        """
        # Outer-edge per-col axis grobs (one entry per panel col / row).
        top_axes    = list(top_grid[0]) if nrow > 0 else []
        bottom_axes = list(bottom_grid[nrow - 1]) if nrow > 0 else []
        left_axes   = [left_grid[r][0] for r in range(nrow)]
        right_axes  = [right_grid[r][ncol - 1] for r in range(nrow)]

        gt, panel_col_to_gtable, panel_row_to_gtable = _seam_axes(
            gt, top_axes, "top", "axis-t", z=3,
            panel_col_to_gtable=panel_col_to_gtable,
            panel_row_to_gtable=panel_row_to_gtable,
        )
        gt, panel_col_to_gtable, panel_row_to_gtable = _seam_axes(
            gt, bottom_axes, "bottom", "axis-b", z=3,
            panel_col_to_gtable=panel_col_to_gtable,
            panel_row_to_gtable=panel_row_to_gtable,
        )
        gt, panel_col_to_gtable, panel_row_to_gtable = _seam_axes(
            gt, left_axes, "left", "axis-l", z=3,
            panel_col_to_gtable=panel_col_to_gtable,
            panel_row_to_gtable=panel_row_to_gtable,
        )
        gt, panel_col_to_gtable, panel_row_to_gtable = _seam_axes(
            gt, right_axes, "right", "axis-r", z=3,
            panel_col_to_gtable=panel_col_to_gtable,
            panel_row_to_gtable=panel_row_to_gtable,
        )
        return gt, panel_col_to_gtable, panel_row_to_gtable

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Compute a grid layout from data and parameters.

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        row_vars = _resolve_facet_vars(params.get("rows"))
        col_vars = _resolve_facet_vars(params.get("cols"))
        drop = params.get("drop", True)
        free = params.get("free", {"x": False, "y": False})

        base_rows = _combine_vars(data, row_vars, drop=drop) if row_vars else pd.DataFrame()
        base_cols = _combine_vars(data, col_vars, drop=drop) if col_vars else pd.DataFrame()

        # Cross-product
        if len(base_rows) > 0 and len(base_cols) > 0:
            base_rows["_key_"] = 1
            base_cols["_key_"] = 1
            base = base_rows.merge(base_cols, on="_key_").drop("_key_", axis=1)
        elif len(base_rows) > 0:
            base = base_rows.copy()
        elif len(base_cols) > 0:
            base = base_cols.copy()
        else:
            return _layout_null()

        if len(base) == 0:
            return _layout_null()

        base = base.drop_duplicates().reset_index(drop=True)

        # Assign PANEL
        n = len(base)
        base["PANEL"] = pd.Categorical(range(1, n + 1))

        # ROW / COL identifiers
        if row_vars and any(v in base.columns for v in row_vars):
            present_rows = [v for v in row_vars if v in base.columns]
            row_ids = base[present_rows].apply(
                lambda r: "|".join(str(v) for v in r), axis=1
            )
            base["ROW"] = pd.Categorical(row_ids).codes + 1
        else:
            base["ROW"] = 1

        if col_vars and any(v in base.columns for v in col_vars):
            present_cols = [v for v in col_vars if v in base.columns]
            col_ids = base[present_cols].apply(
                lambda r: "|".join(str(v) for v in r), axis=1
            )
            base["COL"] = pd.Categorical(col_ids).codes + 1
        else:
            base["COL"] = 1

        # Scale identifiers
        base["SCALE_X"] = base["COL"] if free.get("x", False) else 1
        base["SCALE_Y"] = base["ROW"] if free.get("y", False) else 1

        base = base.sort_values("PANEL").reset_index(drop=True)
        return base

    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        row_vars = _resolve_facet_vars(params.get("rows"))
        col_vars = _resolve_facet_vars(params.get("cols"))
        all_vars = row_vars + col_vars
        return _map_facet_data(data, layout, params, all_vars)

    def vars(self) -> List[str]:
        row_vars = _resolve_facet_vars(self.params.get("rows"))
        col_vars = _resolve_facet_vars(self.params.get("cols"))
        return row_vars + col_vars


# ---------------------------------------------------------------------------
# FacetWrap
# ---------------------------------------------------------------------------

class FacetWrap(Facet):
    """Wrap facet: 1-d ribbon of panels wrapped into 2-d.

    Attributes
    ----------
    shrink : bool
    params : dict
        Contains ``facets``, ``nrow``, ``ncol``, ``scales``, ``free``,
        ``space_free``, ``labeller``, ``strip_position``, ``dir``,
        ``drop``, ``draw_axes``, ``axis_labels``.
    """

    shrink: bool = True

    def _attach_axes(
        self,
        gt: Any,
        nrow: int,
        ncol: int,
        left_grid: List[List[Any]],
        right_grid: List[List[Any]],
        top_grid: List[List[Any]],
        bottom_grid: List[List[Any]],
        panel_col_to_gtable: Dict[int, int],
        panel_row_to_gtable: Dict[int, int],
        theme: Any,
    ) -> Tuple[Any, Dict[int, int], Dict[int, int]]:
        """R: ``FacetWrap$attach_axes`` (facet-wrap.R:267-378) →
        ``weave_axes`` (lines 526-542) → ``weave_tables_col/row`` (501-524).

        Adds one axis row / col **per panel** even when the per-panel
        axis grob is blank — blanked entries become ``null_grob`` of zero
        size so layout-name coverage stays stable for downstream consumers.
        """
        from grid_py import Unit as unit, null_grob
        from gtable_py import gtable_add_grob, gtable_add_rows, gtable_add_cols

        def _blank(grob: Any) -> bool:
            return grob is None or _is_null_grob(grob)

        def _emit(g: Any) -> Any:
            return g if not _blank(g) else null_grob()

        # LEFT axes
        for panel_c0 in reversed(range(ncol)):
            panel_c = panel_c0 + 1
            widths_cm = [
                _axis_width_cm(left_grid[r][panel_c0])
                for r in range(nrow) if not _blank(left_grid[r][panel_c0])
            ]
            w = max(widths_cm) if widths_cm else 0
            cur = panel_col_to_gtable[panel_c]
            gt = gtable_add_cols(gt, unit([w], "cm"), pos=cur - 1)
            ax_col = cur
            for cc in range(panel_c, ncol + 1):
                panel_col_to_gtable[cc] += 1
            for r in range(nrow):
                gt = gtable_add_grob(
                    gt, _emit(left_grid[r][panel_c0]),
                    t=panel_row_to_gtable[r + 1], l=ax_col,
                    clip="off", name=f"axis-l-{r + 1}-{panel_c}",
                )

        # RIGHT axes
        for panel_c0 in reversed(range(ncol)):
            panel_c = panel_c0 + 1
            widths_cm = [
                _axis_width_cm(right_grid[r][panel_c0])
                for r in range(nrow) if not _blank(right_grid[r][panel_c0])
            ]
            w = max(widths_cm) if widths_cm else 0
            cur = panel_col_to_gtable[panel_c]
            gt = gtable_add_cols(gt, unit([w], "cm"), pos=cur)
            ax_col = cur + 1
            for cc in range(panel_c + 1, ncol + 1):
                panel_col_to_gtable[cc] += 1
            for r in range(nrow):
                gt = gtable_add_grob(
                    gt, _emit(right_grid[r][panel_c0]),
                    t=panel_row_to_gtable[r + 1], l=ax_col,
                    clip="off", name=f"axis-r-{r + 1}-{panel_c}",
                )

        # BOTTOM axes
        for panel_r0 in reversed(range(nrow)):
            panel_r = panel_r0 + 1
            heights_cm = [
                _axis_height_cm(bottom_grid[panel_r0][c])
                for c in range(ncol) if not _blank(bottom_grid[panel_r0][c])
            ]
            h = max(heights_cm) if heights_cm else 0
            cur = panel_row_to_gtable[panel_r]
            gt = gtable_add_rows(gt, unit([h], "cm"), pos=cur)
            ax_row = cur + 1
            for rr in range(panel_r + 1, nrow + 1):
                panel_row_to_gtable[rr] += 1
            for c in range(ncol):
                gt = gtable_add_grob(
                    gt, _emit(bottom_grid[panel_r0][c]),
                    t=ax_row, l=panel_col_to_gtable[c + 1],
                    clip="off", name=f"axis-b-{panel_r}-{c + 1}",
                )

        # TOP axes
        for panel_r0 in reversed(range(nrow)):
            panel_r = panel_r0 + 1
            heights_cm = [
                _axis_height_cm(top_grid[panel_r0][c])
                for c in range(ncol) if not _blank(top_grid[panel_r0][c])
            ]
            h = max(heights_cm) if heights_cm else 0
            cur = panel_row_to_gtable[panel_r]
            gt = gtable_add_rows(gt, unit([h], "cm"), pos=cur - 1)
            ax_row = cur
            for rr in range(panel_r, nrow + 1):
                panel_row_to_gtable[rr] += 1
            for c in range(ncol):
                gt = gtable_add_grob(
                    gt, _emit(top_grid[panel_r0][c]),
                    t=ax_row, l=panel_col_to_gtable[c + 1],
                    clip="off", name=f"axis-t-{panel_r}-{c + 1}",
                )

        return gt, panel_col_to_gtable, panel_row_to_gtable

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Compute a wrapped layout.

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        facet_vars = _resolve_facet_vars(params.get("facets"))
        drop = params.get("drop", True)
        free = params.get("free", {"x": False, "y": False})
        nrow = params.get("nrow")
        ncol = params.get("ncol")
        dir_ = params.get("dir", "lt")

        if not facet_vars:
            return _layout_null()

        base = _combine_vars(data, facet_vars, drop=drop)
        if len(base) == 0:
            return _layout_null()

        base = base.drop_duplicates().reset_index(drop=True)
        n = len(base)
        dims = wrap_dims(n, nrow, ncol)

        # Assign PANEL, ROW, COL
        ids = np.arange(1, n + 1)
        base["PANEL"] = pd.Categorical(ids)

        # Determine layout direction
        if len(dir_) == 2:
            row_vals, col_vals = _wrap_layout(ids, dims, dir_)
        else:
            # Fallback
            row_vals = (ids - 1) // dims[1] + 1
            col_vals = (ids - 1) % dims[1] + 1

        base["ROW"] = row_vals.astype(int)
        base["COL"] = col_vals.astype(int)

        # Scale identifiers
        base["SCALE_X"] = ids if free.get("x", False) else 1
        base["SCALE_Y"] = ids if free.get("y", False) else 1

        base = base.sort_values("PANEL").reset_index(drop=True)
        return base

    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        facet_vars = _resolve_facet_vars(params.get("facets"))
        return _map_facet_data(data, layout, params, facet_vars)

    def vars(self) -> List[str]:
        return _resolve_facet_vars(self.params.get("facets"))


def _wrap_layout(
    ids: np.ndarray,
    dims: Tuple[int, int],
    dir_: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ROW and COL for wrapped layout.

    Parameters
    ----------
    ids : np.ndarray
        1-based panel IDs.
    dims : tuple of (nrow, ncol)
    dir_ : str
        Two-letter direction code.

    Returns
    -------
    tuple of (ROW, COL) arrays
    """
    nrow, ncol = dims
    ids0 = ids - 1  # 0-based

    if dir_ in ("lt", "lb"):
        row = ids0 // ncol
        col = ids0 % ncol
    elif dir_ in ("tl", "bl"):
        row = ids0 % nrow
        col = ids0 // nrow
    elif dir_ in ("rt", "rb"):
        row = ids0 // ncol
        col = ncol - 1 - ids0 % ncol
    elif dir_ in ("tr", "br"):
        row = ids0 % nrow
        col = ncol - 1 - ids0 // nrow
    else:
        row = ids0 // ncol
        col = ids0 % ncol

    # Handle bottom-start directions
    if dir_ in ("lb", "bl", "rb", "br"):
        row = nrow - 1 - row

    return row + 1, col + 1


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------

def facet_null(shrink: bool = True) -> FacetNull:
    """Create a null facet (single panel).

    Parameters
    ----------
    shrink : bool

    Returns
    -------
    FacetNull
    """
    obj = FacetNull()
    obj.shrink = shrink
    return obj


def _grid_as_facets_list(rows: Any, cols: Any) -> Tuple[Any, Any]:
    """Port of R ``grid_as_facets_list`` (facet-grid-.R:202-235).

    When *rows* is a formula-style string ``"a + b ~ c + d"`` and *cols*
    is ``None``, split on ``~``: LHS becomes the row dimension, RHS the
    column dimension. ``.`` and empty sides collapse to ``None``.

    Mirrors R's ``as_facets_list`` (facet-.R:1010-1053): a length-3
    formula yields ``(rows=LHS, cols=RHS)``; a length-2 formula
    (``~ x``) yields ``(rows=NULL, cols=x)``.
    """
    if cols is None and isinstance(rows, str) and "~" in rows:
        lhs, _, rhs = rows.partition("~")
        def _split(side: str) -> Optional[List[str]]:
            parts = [v.strip() for v in side.split("+")]
            parts = [v for v in parts if v and v != "."]
            return parts or None
        return _split(lhs), _split(rhs)
    return rows, cols


def facet_grid(
    rows: Any = None,
    cols: Any = None,
    scales: str = "fixed",
    space: str = "fixed",
    shrink: bool = True,
    labeller: Any = "label_value",
    as_table: bool = True,
    switch: Optional[str] = None,
    drop: bool = True,
    margins: Union[bool, List[str]] = False,
    axes: str = "margins",
    axis_labels: str = "all",
) -> FacetGrid:
    """Create a grid facet.

    Parameters
    ----------
    rows, cols : str, list, or None
        Faceting variables for rows and columns.
    scales : str
        ``"fixed"``, ``"free_x"``, ``"free_y"``, or ``"free"``.
    space : str
        ``"fixed"``, ``"free_x"``, ``"free_y"``, or ``"free"``.
    shrink : bool
    labeller : callable or str
    as_table : bool
    switch : str or None
        ``"x"``, ``"y"``, ``"both"``, or None.
    drop : bool
    margins : bool or list of str
    axes : str
        ``"margins"``, ``"all_x"``, ``"all_y"``, or ``"all"``.
    axis_labels : str
        ``"margins"``, ``"all_x"``, ``"all_y"``, or ``"all"``.

    Returns
    -------
    FacetGrid
    """
    free = {
        "x": scales in ("free_x", "free"),
        "y": scales in ("free_y", "free"),
    }
    space_free = {
        "x": space in ("free_x", "free"),
        "y": space in ("free_y", "free"),
    }
    draw_axes_ = {
        "x": axes in ("all_x", "all"),
        "y": axes in ("all_y", "all"),
    }
    axis_labels_ = {
        "x": not draw_axes_["x"] or axis_labels in ("all_x", "all"),
        "y": not draw_axes_["y"] or axis_labels in ("all_y", "all"),
    }

    # R: grid_as_facets_list (facet-grid-.R:187) splits a formula spec
    # like ``drv ~ class`` into rows / cols before storing in params.
    rows, cols = _grid_as_facets_list(rows, cols)

    obj = FacetGrid()
    obj.shrink = shrink
    obj.params = {
        "rows": rows,
        "cols": cols,
        "margins": margins,
        "free": free,
        "space_free": space_free,
        "labeller": labeller,
        "as_table": as_table,
        "switch": switch,
        "drop": drop,
        "draw_axes": draw_axes_,
        "axis_labels": axis_labels_,
    }
    return obj


def facet_wrap(
    facets: Any,
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    scales: str = "fixed",
    space: str = "fixed",
    shrink: bool = True,
    labeller: Any = "label_value",
    as_table: bool = True,
    drop: bool = True,
    dir: str = "h",
    strip_position: str = "top",
    axes: str = "margins",
    axis_labels: str = "all",
) -> FacetWrap:
    """Create a wrap facet.

    Parameters
    ----------
    facets : str, list, or dict
        Faceting variables.
    nrow, ncol : int or None
    scales : str
        ``"fixed"``, ``"free_x"``, ``"free_y"``, or ``"free"``.
    space : str
    shrink : bool
    labeller : callable or str
    as_table : bool
    drop : bool
    dir : str
        Direction: ``"h"`` or ``"v"``, or a two-letter code.
    strip_position : str
        ``"top"``, ``"bottom"``, ``"left"``, or ``"right"``.
    axes : str
    axis_labels : str

    Returns
    -------
    FacetWrap
    """
    free = {
        "x": scales in ("free_x", "free"),
        "y": scales in ("free_y", "free"),
    }
    space_free = {
        "x": space == "free_x",
        "y": space == "free_y",
    }
    draw_axes_ = {
        "x": free["x"] or axes in ("all_x", "all"),
        "y": free["y"] or axes in ("all_y", "all"),
    }
    axis_labels_ = {
        "x": free["x"] or not draw_axes_["x"] or axis_labels in ("all_x", "all"),
        "y": free["y"] or not draw_axes_["y"] or axis_labels in ("all_y", "all"),
    }

    # Resolve direction
    if len(dir) == 1:
        if dir == "h":
            dir = "lt" if as_table else "lb"
        elif dir == "v":
            dir = "tl" if as_table else "tr"

    if strip_position not in ("top", "bottom", "left", "right"):
        cli_abort("strip_position must be 'top', 'bottom', 'left', or 'right'.")

    obj = FacetWrap()
    obj.shrink = shrink
    obj.params = {
        "facets": facets,
        "nrow": nrow,
        "ncol": ncol,
        "free": free,
        "space_free": space_free,
        "labeller": labeller,
        "dir": dir,
        "strip_position": strip_position,
        "drop": drop,
        "draw_axes": draw_axes_,
        "axis_labels": axis_labels_,
    }
    return obj


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def is_facet(x: Any) -> bool:
    """Test whether *x* is a Facet.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
    """
    return isinstance(x, Facet)

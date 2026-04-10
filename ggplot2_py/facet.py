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


def _wrap_dims(n: int, nrow: Optional[int] = None, ncol: Optional[int] = None) -> Tuple[int, int]:
    """Compute grid dimensions for *n* panels.

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
        ncol = math.ceil(math.sqrt(n))
        nrow = math.ceil(n / ncol)
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
    return combined[vars_].reset_index(drop=True)


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
        scales: Dict[str, list] = {}
        if x_scale is not None:
            n_x = int(layout["SCALE_X"].max())
            scales["x"] = [x_scale] * n_x
        if y_scale is not None:
            n_y = int(layout["SCALE_Y"].max())
            scales["y"] = [y_scale] * n_y
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

        Mirrors R's ``Facet$draw_panels``:
        1. Decorate each panel with coord background + foreground
        2. Render axis grobs via coord
        3. Assemble into a gtable with axis rows/columns

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
        from gtable_py import Gtable, gtable_add_grob
        from grid_py import Unit as unit

        nrow = int(layout["ROW"].max()) if len(layout) > 0 else 1
        ncol = int(layout["COL"].max()) if len(layout) > 0 else 1

        # Axis cell sizing — mirrors R's ggplot_gtable() which uses
        # absolute units (cm) for axis cells and "null" for panels.
        # In R, grobWidth()/grobHeight() compute the exact text size;
        # we approximate using character count and a standard font size.
        #
        # R defaults: axis.ticks.length = unit(0.15, "cm"),
        #   font ~11pt ≈ 0.39cm per character height,
        #   character width ~0.6 × height ≈ 0.23cm
        y_labels = ranges[0].get("y_labels", []) if ranges else []
        x_labels = ranges[0].get("x_labels", []) if ranges else []

        # Left axis width (cm): tick + gap + label text width
        # R reference: tick=0.15cm, gap≈0.1cm, char≈0.22cm wide at 11pt
        max_y_chars = max((len(str(l)) for l in y_labels), default=2)
        AX_L_cm = 0.15 + 0.10 + max_y_chars * 0.22 + 0.10  # tick+gap+text+pad

        # Bottom axis height (cm): detect whether labels will be rotated
        # (mirrors the auto-rotation logic in _render_axis).
        max_x_chars = max((len(str(l)) for l in x_labels), default=3)
        x_breaks = ranges[0].get("x_major", np.array([])) if ranges else np.array([])
        x_rotated = False
        if len(x_breaks) > 1:
            n_xb = len(x_breaks)
            spacing = (x_breaks[-1] - x_breaks[0]) / max(n_xb - 1, 1)
            est_width = max_x_chars * 0.016
            x_rotated = n_xb > 1 and est_width > spacing * 0.9
        if x_rotated:
            # Rotated ~30°: projected height ≈ width * sin(30°) + charH * cos(30°)
            AX_B_cm = 0.15 + 0.10 + max_x_chars * 0.22 * 0.5 + 0.39 * 0.87 + 0.10
        else:
            AX_B_cm = 0.15 + 0.10 + 0.39 + 0.10  # tick+gap+1 line height+pad

        # Build gtable with mixed units: "cm" for axes, "null" for panels.
        # This matches R's approach where axis cells have fixed absolute
        # size and panels fill the remaining space.
        from grid_py._units import unit_c
        widths = unit_c(
            unit([AX_L_cm], "cm"),
            unit([1] * ncol, "null"),
            unit([0.1], "cm"),
        )
        heights = unit_c(
            unit([1] * nrow, "null"),
            unit([AX_B_cm], "cm"),
        )
        gt = Gtable(widths=widths, heights=heights, name="layout")

        for _, row_info in layout.iterrows():
            panel_id = int(row_info["PANEL"])
            r = int(row_info["ROW"])
            c = int(row_info["COL"])
            panel_idx = panel_id - 1
            pp = ranges[panel_idx] if panel_idx < len(ranges) else {}

            # --- Collect geom grobs for this panel ---
            panel_grobs = []
            for layer_grobs in panels:
                if isinstance(layer_grobs, list) and panel_idx < len(layer_grobs):
                    panel_grobs.append(layer_grobs[panel_idx])
                elif not isinstance(layer_grobs, list) and layer_grobs is not None:
                    panel_grobs.append(layer_grobs)

            # --- Decorate panel with background + foreground ---
            # coord.draw_panel wraps in a Viewport so NPC [0,1] maps
            # to the panel cell region (R semantics).
            if hasattr(coord, "draw_panel"):
                decorated = coord.draw_panel(panel_grobs, pp, theme)
            else:
                decorated = GTree(
                    children=GList(*panel_grobs),
                    name=f"panel-{panel_id}",
                )

            # Place panel in cell (r, c+1) — offset by 1 for left axis col
            gt = gtable_add_grob(
                gt, decorated, t=r, l=c + 1, name=f"panel-{r}-{c}",
            )

            # --- Axes: only on the edges, matching R's facet semantics ---
            # R adds axis grobs with clip="off" so labels are not clipped
            # at the cell boundary.
            # Bottom axis: only for the last row
            is_bottom_row = (r == nrow)
            if is_bottom_row and hasattr(coord, "render_axis_h"):
                axes_h = coord.render_axis_h(pp, theme)
                bottom_ax = axes_h.get("bottom")
                if bottom_ax is not None:
                    gt = gtable_add_grob(
                        gt, bottom_ax, t=nrow + 1, l=c + 1,
                        clip="off",
                        name=f"axis-b-{r}-{c}",
                    )

            # Left axis: only for the first column
            is_left_col = (c == 1)
            if is_left_col and hasattr(coord, "render_axis_v"):
                axes_v = coord.render_axis_v(pp, theme)
                left_ax = axes_v.get("left")
                if left_ax is not None:
                    gt = gtable_add_grob(
                        gt, left_ax, t=r, l=1,
                        clip="off",
                        name=f"axis-l-{r}-{c}",
                    )

        # --- Strip labels (R: FacetGrid adds top strips for cols,
        #     right strips for rows) ---
        gt = self._add_strip_labels(gt, layout, nrow, ncol, params)

        return gt

    def _add_strip_labels(
        self,
        gt: Any,
        layout: pd.DataFrame,
        nrow: int,
        ncol: int,
        params: Dict[str, Any],
    ) -> Any:
        """Add facet strip text labels to the gtable.

        Mirrors R's ``FacetGrid$draw_panels`` strip assembly:
        - Column strips (top): one per unique column, showing col var values
        - Row strips (right): one per unique row, showing row var values
        """
        from gtable_py import gtable_add_grob, gtable_add_rows, gtable_add_cols
        from grid_py import Unit as unit, text_grob, Gpar, rect_grob
        from grid_py._grob import grob_tree, GList

        # Detect faceting variables (columns besides PANEL/ROW/COL/SCALE_*/COORD)
        meta_cols = {"PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y", "COORD"}
        facet_vars = [c for c in layout.columns if c not in meta_cols]
        if not facet_vars:
            return gt

        col_vars = _resolve_facet_vars(params.get("cols"))
        row_vars = _resolve_facet_vars(params.get("rows"))

        # --- Top strip: one per column (col variable values) ---
        if col_vars:
            gt = gtable_add_rows(gt, unit([0.35], "cm"), pos=0)
            for c in range(1, ncol + 1):
                panel_row = layout[layout["COL"] == c].iloc[0]
                label_parts = [str(panel_row.get(v, "")) for v in col_vars]
                label_text = " : ".join(label_parts)

                strip = grob_tree(
                    rect_grob(
                        x=0.5, y=0.5, width=1, height=1,
                        gp=Gpar(fill="grey85", col=None),
                        name=f"strip.bg.top.{c}",
                    ),
                    text_grob(
                        label=label_text, x=0.5, y=0.5,
                        just="centre",
                        gp=Gpar(fontsize=7, col="grey10"),
                        name=f"strip.text.top.{c}",
                    ),
                    name=f"strip-t-{c}",
                )
                gt = gtable_add_grob(
                    gt, strip, t=1, l=c + 1,
                    clip="off", name=f"strip-t-{c}",
                )

        # --- Right strip: one per row (row variable values) ---
        if row_vars:
            gt = gtable_add_cols(gt, unit([0.4], "cm"), pos=-1)
            ncol_now = len(gt._widths)
            # Offset top strips if present
            row_offset = 1 if col_vars else 0
            for r in range(1, nrow + 1):
                panel_row = layout[layout["ROW"] == r].iloc[0]
                label_parts = [str(panel_row.get(v, "")) for v in row_vars]
                label_text = " : ".join(label_parts)

                strip = grob_tree(
                    rect_grob(
                        x=0.5, y=0.5, width=1, height=1,
                        gp=Gpar(fill="grey85", col=None),
                        name=f"strip.bg.right.{r}",
                    ),
                    text_grob(
                        label=label_text, x=0.5, y=0.5,
                        rot=270, just="centre",
                        gp=Gpar(fontsize=7, col="grey10"),
                        name=f"strip.text.right.{r}",
                    ),
                    name=f"strip-r-{r}",
                )
                gt = gtable_add_grob(
                    gt, strip, t=r + row_offset, l=ncol_now,
                    clip="off", name=f"strip-r-{r}",
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
        """Add axis title labels (xlab/ylab) to the panel table.

        Mirrors R's ``Facet$draw_labels``: adds a bottom row for the
        x-axis title and a left column for the y-axis title.

        Parameters
        ----------
        panels : gtable
        labels : dict
            Rendered label grobs keyed by ``"x"`` / ``"y"``, each a
            two-element list ``[primary, secondary]``.
        """
        from gtable_py import gtable_add_grob, gtable_add_rows, gtable_add_cols
        from grid_py import Unit as unit, text_grob, Gpar, null_grob

        gt = panels

        # --- x-axis title (bottom) ---
        x_label = None
        if "x" in labels:
            pair = labels["x"]
            if isinstance(pair, list) and len(pair) > 0:
                x_label = pair[0]  # primary

        if x_label is not None and not _is_null_grob(x_label):
            gt = gtable_add_rows(gt, unit([0.4], "cm"), pos=-1)
            nrow = len(gt._heights)
            ncol = len(gt._widths)
            gt = gtable_add_grob(
                gt, x_label, t=nrow, l=1, r=ncol,
                clip="off", name="xlab",
            )

        # --- y-axis title (left) ---
        y_label = None
        if "y" in labels:
            pair = labels["y"]
            if isinstance(pair, list) and len(pair) > 0:
                y_label = pair[0]  # primary

        if y_label is not None and not _is_null_grob(y_label):
            gt = gtable_add_cols(gt, unit([0.4], "cm"), pos=0)
            nrow = len(gt._heights)
            gt = gtable_add_grob(
                gt, y_label, t=1, b=nrow, l=1,
                clip="off", name="ylab",
            )

        return gt

    def vars(self) -> List[str]:
        """Return the faceting variable names.

        Returns
        -------
        list of str
        """
        return []


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
        """Build a single-panel gtable with background, axes, and geom content.

        Delegates to the base ``Facet.draw_panels`` which handles coord
        decoration and axis rendering.
        """
        return super().draw_panels(
            panels, layout, x_scales, y_scales, ranges,
            coord, data, theme, params,
        )


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
        dims = _wrap_dims(n, nrow, ncol)

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

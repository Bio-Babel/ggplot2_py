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
        """Assemble panels into a gtable.

        Parameters
        ----------
        panels : list of grobs (per-layer, each containing per-panel grobs)
        layout : pd.DataFrame
        x_scales, y_scales : list
        ranges : list
        coord : Coord
        data : list
        theme : Theme
        params : dict

        Returns
        -------
        gtable
        """
        from grid_py import GTree, GList, null_grob
        from gtable_py import Gtable, gtable_add_grob
        from grid_py import Unit as unit

        nrow = int(layout["ROW"].max()) if len(layout) > 0 else 1
        ncol = int(layout["COL"].max()) if len(layout) > 0 else 1

        gt = Gtable(
            widths=unit([1] * ncol, "null"),
            heights=unit([1] * nrow, "null"),
            name="layout",
        )

        for _, row_info in layout.iterrows():
            panel_id = int(row_info["PANEL"])
            r = int(row_info["ROW"])
            c = int(row_info["COL"])
            panel_idx = panel_id - 1

            panel_grobs = []
            for layer_grobs in panels:
                if isinstance(layer_grobs, list) and panel_idx < len(layer_grobs):
                    panel_grobs.append(layer_grobs[panel_idx])
                elif not isinstance(layer_grobs, list) and layer_grobs is not None:
                    panel_grobs.append(layer_grobs)

            if panel_grobs:
                content = GTree(
                    children=GList(*panel_grobs),
                    name=f"panel-{panel_id}",
                )
                gt = gtable_add_grob(gt, content, t=r, l=c, name=f"panel-{r}-{c}")

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
        """Add axis/facet labels to the panel table.

        Parameters
        ----------
        panels, layout, x_scales, y_scales, ranges, coord, data, theme,
        labels, params
            See ``draw_panels`` for shared arguments.

        Returns
        -------
        gtable
        """
        return panels

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
        """Build a single-panel gtable from all layer grobs."""
        from grid_py import GTree, GList, null_grob
        from gtable_py import Gtable, gtable_add_grob
        from grid_py import Unit as unit

        # Collect all grobs for panel 1 (index 0 from each layer)
        panel_grobs = []
        for layer_grobs in panels:
            if isinstance(layer_grobs, list):
                if len(layer_grobs) > 0:
                    panel_grobs.append(layer_grobs[0])
            elif layer_grobs is not None:
                panel_grobs.append(layer_grobs)

        if not panel_grobs:
            return null_grob()

        # Wrap all panel grobs in a single GTree
        panel_content = GTree(
            children=GList(*panel_grobs),
            name="panel-1",
        )

        # Create a 1x1 Gtable holding the panel
        gt = Gtable(
            widths=unit([1], "null"),
            heights=unit([1], "null"),
            name="layout",
        )
        gt = gtable_add_grob(gt, panel_content, t=1, l=1, name="panel-1-1")
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

"""
Layout: coordinate system + faceting + panel-scale management.

The :class:`Layout` class is the internal engine that connects facets,
coordinates, and per-panel scales during the build and render phases of
a ggplot.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort
from ggplot2_py.ggproto import GGProto, ggproto
from ggplot2_py._utils import data_frame

__all__ = ["Layout"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _scale_apply(
    data: pd.DataFrame,
    vars_: List[str],
    method: str,
    scale_id: pd.Series,
    scales: List[Any],
) -> Dict[str, Any]:
    """Apply a scale method to columns split by panel scale index.

    Parameters
    ----------
    data : DataFrame
        Layer data.
    vars_ : list of str
        Aesthetic column names to process.
    method : str
        Scale method name (e.g. ``"map"``).
    scale_id : array-like
        Per-row scale index (from ``layout.SCALE_X[match_id]``).
    scales : list of Scale
        The panel scales list.

    Returns
    -------
    dict
        Column-name -> mapped-values mapping.
    """
    if len(vars_) == 0 or data.shape[0] == 0:
        return {}

    result: Dict[str, Any] = {}
    for var in vars_:
        pieces: List[Any] = []
        indices: List[np.ndarray] = []
        for i, sc in enumerate(scales):
            mask = scale_id == (i + 1)  # scale indices are 1-based
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            chunk = data[var].iloc[idx]
            mapped = getattr(sc, method)(chunk)
            pieces.append(mapped)
            indices.append(idx)
        if pieces:
            # Reconstruct in original order
            out = pd.Series(np.nan, index=data.index, dtype=object)
            for idx_arr, piece in zip(indices, pieces):
                if isinstance(piece, pd.Series):
                    out.iloc[idx_arr] = piece.values
                elif isinstance(piece, np.ndarray):
                    out.iloc[idx_arr] = piece
                elif isinstance(piece, (list, tuple)):
                    out.iloc[idx_arr] = piece
                else:
                    out.iloc[idx_arr] = piece
            # Try to convert to numeric if possible
            try:
                result[var] = pd.to_numeric(out, errors="raise")
            except (ValueError, TypeError):
                result[var] = out
        else:
            result[var] = data[var].copy()
    return result


# ---------------------------------------------------------------------------
# Layout class
# ---------------------------------------------------------------------------

class Layout(GGProto):
    """Panel layout manager.

    The Layout manages panel creation and scale management during the
    build (``ggplot_build``) and render (``ggplot_gtable``) phases.

    Attributes
    ----------
    coord : Coord
        The coordinate system.
    coord_params : dict
        Parameters populated by ``Coord.setup_params()``.
    facet : Facet
        The faceting specification.
    facet_params : dict
        Parameters populated by ``Facet.setup_params()``.
    layout : DataFrame
        One row per panel with columns ``PANEL``, ``ROW``, ``COL``,
        ``SCALE_X``, ``SCALE_Y``, and possibly faceting variables.
    panel_scales_x : list of Scale
        Per-panel x scales (indexed by ``SCALE_X``).
    panel_scales_y : list of Scale
        Per-panel y scales (indexed by ``SCALE_Y``).
    panel_params : list of dict
        Per-panel coordinate parameters.
    """

    _class_name = "Layout"

    coord: Any = None
    coord_params: Dict[str, Any] = {}
    facet: Any = None
    facet_params: Dict[str, Any] = {}
    layout: Optional[pd.DataFrame] = None
    panel_scales_x: Optional[List[Any]] = None
    panel_scales_y: Optional[List[Any]] = None
    panel_params: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # ggplot_build phase
    # ------------------------------------------------------------------

    def setup(
        self,
        data: List[pd.DataFrame],
        plot_data: pd.DataFrame = None,
        plot_env: Any = None,
    ) -> List[pd.DataFrame]:
        """Initialise facet layout and assign panels to data.

        Parameters
        ----------
        data : list of DataFrame
            Layer data (one DataFrame per layer).
        plot_data : DataFrame, optional
            The plot-level default data.
        plot_env : object, optional
            The plot environment (unused in Python).

        Returns
        -------
        list of DataFrame
            Layer data with ``PANEL`` column assigned.
        """
        if plot_data is None:
            plot_data = pd.DataFrame()

        all_data = [plot_data] + list(data)

        # Setup facet
        if hasattr(self.facet, "setup_params"):
            self.facet_params = self.facet.setup_params(
                all_data, getattr(self.facet, "params", {})
            )
        else:
            self.facet_params = getattr(self.facet, "params", {})

        if plot_env is not None:
            self.facet_params["plot_env"] = plot_env

        if hasattr(self.facet, "setup_data"):
            all_data = self.facet.setup_data(all_data, self.facet_params)

        # Setup coord
        if hasattr(self.coord, "setup_params"):
            self.coord_params = self.coord.setup_params(all_data)
        else:
            self.coord_params = {}

        if hasattr(self.coord, "setup_data"):
            all_data = self.coord.setup_data(all_data, self.coord_params)

        # Generate panel layout
        if hasattr(self.facet, "compute_layout"):
            self.layout = self.facet.compute_layout(all_data, self.facet_params)
        else:
            self.layout = pd.DataFrame({
                "PANEL": pd.Categorical([1]),
                "ROW": [1],
                "COL": [1],
                "SCALE_X": [1],
                "SCALE_Y": [1],
            })

        if hasattr(self.coord, "setup_layout"):
            self.layout = self.coord.setup_layout(self.layout, self.coord_params)

        # Add COORD column if not present (used for deduplicating panel_params)
        if "COORD" not in self.layout.columns:
            # Default: unique combination of SCALE_X and SCALE_Y
            self.layout["COORD"] = (
                self.layout["SCALE_X"].astype(str) + "_" +
                self.layout["SCALE_Y"].astype(str)
            )

        # Map data to panels
        result = []
        for layer_data in all_data[1:]:  # skip plot_data (index 0)
            if hasattr(self.facet, "map_data"):
                mapped = self.facet.map_data(
                    layer_data,
                    layout=self.layout,
                    params=self.facet_params,
                )
                result.append(mapped)
            else:
                # Default: assign all rows to panel 1
                ld = layer_data.copy()
                if "PANEL" not in ld.columns:
                    ld["PANEL"] = pd.Categorical(
                        [1] * len(ld),
                        categories=self.layout["PANEL"].cat.categories
                        if hasattr(self.layout["PANEL"], "cat") else [1],
                    )
                result.append(ld)
        return result

    def train_position(
        self,
        data: List[pd.DataFrame],
        x_scale: Any,
        y_scale: Any,
    ) -> None:
        """Train position scales for each panel.

        Parameters
        ----------
        data : list of DataFrame
            Layer data.
        x_scale, y_scale : Scale
            Prototype position scales.
        """
        layout = self.layout

        # Initialise scales if needed
        if self.panel_scales_x is None and x_scale is not None:
            if hasattr(self.facet, "init_scales"):
                res = self.facet.init_scales(
                    layout, x_scale=x_scale, params=self.facet_params
                )
                self.panel_scales_x = res.get("x", [x_scale.clone()])
            else:
                n_x = int(layout["SCALE_X"].max()) if len(layout) > 0 else 1
                self.panel_scales_x = [x_scale.clone() for _ in range(n_x)]

        if self.panel_scales_y is None and y_scale is not None:
            if hasattr(self.facet, "init_scales"):
                res = self.facet.init_scales(
                    layout, y_scale=y_scale, params=self.facet_params
                )
                self.panel_scales_y = res.get("y", [y_scale.clone()])
            else:
                n_y = int(layout["SCALE_Y"].max()) if len(layout) > 0 else 1
                self.panel_scales_y = [y_scale.clone() for _ in range(n_y)]

        # Train scales
        if hasattr(self.facet, "train_scales"):
            self.facet.train_scales(
                self.panel_scales_x,
                self.panel_scales_y,
                layout,
                data,
                self.facet_params,
            )
        else:
            # Default training: train each scale on matching panel data
            for layer_data in data:
                if layer_data is None or layer_data.empty:
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
                    if self.panel_scales_x and sx_idx < len(self.panel_scales_x):
                        self.panel_scales_x[sx_idx].train_df(panel_data)
                    if self.panel_scales_y and sy_idx < len(self.panel_scales_y):
                        self.panel_scales_y[sy_idx].train_df(panel_data)

    def map_position(self, data: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Map position aesthetics through trained panel scales.

        Parameters
        ----------
        data : list of DataFrame
            Layer data.

        Returns
        -------
        list of DataFrame
            Data with mapped position columns.
        """
        layout = self.layout
        result = []

        for layer_data in data:
            if layer_data is None or layer_data.empty:
                result.append(layer_data)
                continue

            ld = layer_data.copy()

            if "PANEL" not in ld.columns:
                result.append(ld)
                continue

            # Match panels
            panel_vals = ld["PANEL"].values
            # Build match index: for each row, which layout row?
            layout_panels = layout["PANEL"].values
            match_id = np.searchsorted(
                np.sort(layout_panels),
                panel_vals,
            )
            # Safer: use a mapping
            panel_to_idx = {p: i for i, p in enumerate(layout_panels)}
            match_idx = np.array([
                panel_to_idx.get(p, 0) for p in panel_vals
            ])

            # Map x variables
            if self.panel_scales_x and len(self.panel_scales_x) > 0:
                x_aes = getattr(self.panel_scales_x[0], "aesthetics", ["x"])
                x_vars = [v for v in x_aes if v in ld.columns]
                if x_vars:
                    scale_x_ids = layout["SCALE_X"].values[match_idx]
                    mapped = _scale_apply(
                        ld, x_vars, "map", pd.Series(scale_x_ids),
                        self.panel_scales_x,
                    )
                    for k, v in mapped.items():
                        ld[k] = v

            # Map y variables
            if self.panel_scales_y and len(self.panel_scales_y) > 0:
                y_aes = getattr(self.panel_scales_y[0], "aesthetics", ["y"])
                y_vars = [v for v in y_aes if v in ld.columns]
                if y_vars:
                    scale_y_ids = layout["SCALE_Y"].values[match_idx]
                    mapped = _scale_apply(
                        ld, y_vars, "map", pd.Series(scale_y_ids),
                        self.panel_scales_y,
                    )
                    for k, v in mapped.items():
                        ld[k] = v

            result.append(ld)
        return result

    def reset_scales(self) -> None:
        """Reset scale ranges (called between stat computation and re-training).

        If the facet's ``shrink`` attribute is ``False``, this is a no-op.
        """
        if not getattr(self.facet, "shrink", True):
            return
        if self.panel_scales_x:
            for s in self.panel_scales_x:
                if hasattr(s, "reset"):
                    s.reset()
        if self.panel_scales_y:
            for s in self.panel_scales_y:
                if hasattr(s, "reset"):
                    s.reset()

    def setup_panel_params(self) -> None:
        """Compute per-panel coordinate parameters.

        Calls ``Coord.setup_panel_params()`` for each unique x/y scale
        combination.
        """
        if hasattr(self.coord, "modify_scales"):
            self.coord.modify_scales(self.panel_scales_x, self.panel_scales_y)

        layout = self.layout
        n_panels = len(layout)
        params_list: List[Dict[str, Any]] = []

        # Deduplicate by COORD column if available
        if "COORD" in layout.columns:
            unique_coords = layout["COORD"].unique()
            coord_to_params: Dict[Any, Dict[str, Any]] = {}
            for uc in unique_coords:
                row = layout.loc[layout["COORD"] == uc].iloc[0]
                sx_idx = int(row["SCALE_X"]) - 1
                sy_idx = int(row["SCALE_Y"]) - 1
                sx = self.panel_scales_x[sx_idx] if self.panel_scales_x else None
                sy = self.panel_scales_y[sy_idx] if self.panel_scales_y else None
                if hasattr(self.coord, "setup_panel_params"):
                    pp = self.coord.setup_panel_params(
                        sx, sy, params=self.coord_params,
                    )
                else:
                    pp = {}
                coord_to_params[uc] = pp

            # Expand to all panels
            for _, row in layout.iterrows():
                params_list.append(coord_to_params[row["COORD"]])
        else:
            for _, row in layout.iterrows():
                sx_idx = int(row["SCALE_X"]) - 1
                sy_idx = int(row["SCALE_Y"]) - 1
                sx = self.panel_scales_x[sx_idx] if self.panel_scales_x else None
                sy = self.panel_scales_y[sy_idx] if self.panel_scales_y else None
                if hasattr(self.coord, "setup_panel_params"):
                    pp = self.coord.setup_panel_params(
                        sx, sy, params=self.coord_params,
                    )
                else:
                    pp = {}
                params_list.append(pp)

        # Let facet modify panel_params
        if hasattr(self.facet, "setup_panel_params"):
            params_list = self.facet.setup_panel_params(params_list, self.coord)

        self.panel_params = params_list

    def setup_panel_guides(self, guides: Any, layers: List[Any]) -> None:
        """Set up and train position guides (axes) per panel.

        Parameters
        ----------
        guides : Guides
            The plot's guides specification.
        layers : list
            Plot layers.
        """
        if self.panel_params is None:
            return

        # Setup guides
        if hasattr(self.coord, "setup_panel_guides"):
            self.panel_params = [
                self.coord.setup_panel_guides(
                    pp, guides, self.coord_params,
                )
                for pp in self.panel_params
            ]

        # Train guides
        if hasattr(self.coord, "train_panel_guides"):
            self.panel_params = [
                self.coord.train_panel_guides(
                    pp, layers, self.coord_params,
                )
                for pp in self.panel_params
            ]

    def finish_data(self, data: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Apply facet's ``finish_data()`` hook.

        Parameters
        ----------
        data : list of DataFrame
            Layer data.

        Returns
        -------
        list of DataFrame
        """
        if hasattr(self.facet, "finish_data"):
            return [
                self.facet.finish_data(
                    d,
                    layout=self.layout,
                    x_scales=self.panel_scales_x,
                    y_scales=self.panel_scales_y,
                    params=self.facet_params,
                )
                for d in data
            ]
        return data

    # ------------------------------------------------------------------
    # ggplot_gtable phase (render)
    # ------------------------------------------------------------------

    def render(
        self,
        panels: List[Any],
        data: List[pd.DataFrame],
        theme: Any,
        labels: Dict[str, Any],
    ) -> Any:
        """Render panels, axes, and strips into a gtable.

        Parameters
        ----------
        panels : list
            Geom grobs per layer (list of lists).
        data : list of DataFrame
            Layer data.
        theme : Theme
            Complete theme.
        labels : dict
            Plot labels.

        Returns
        -------
        gtable
            The assembled plot table.
        """
        # Draw panel content
        if hasattr(self.facet, "draw_panel_content"):
            panels = self.facet.draw_panel_content(
                panels,
                self.layout,
                self.panel_scales_x,
                self.panel_scales_y,
                self.panel_params,
                self.coord,
                data,
                theme,
                self.facet_params,
            )

        # Draw panels into gtable
        if hasattr(self.facet, "draw_panels"):
            plot_table = self.facet.draw_panels(
                panels,
                self.layout,
                self.panel_scales_x,
                self.panel_scales_y,
                self.panel_params,
                self.coord,
                data,
                theme,
                self.facet_params,
            )
        else:
            # Minimal fallback
            from gtable_py import Gtable
            plot_table = Gtable()

        # Set panel sizes
        if hasattr(self.facet, "set_panel_size"):
            plot_table = self.facet.set_panel_size(plot_table, theme)

        # Resolve axis labels
        resolved_labels = {}
        if self.panel_scales_x and len(self.panel_scales_x) > 0:
            resolved_labels["x"] = self.resolve_label(
                self.panel_scales_x[0], labels,
            )
        if self.panel_scales_y and len(self.panel_scales_y) > 0:
            resolved_labels["y"] = self.resolve_label(
                self.panel_scales_y[0], labels,
            )

        # Let coord modify labels
        if hasattr(self.coord, "labels") and self.panel_params:
            resolved_labels = self.coord.labels(
                resolved_labels,
                self.panel_params[0] if self.panel_params else {},
            )

        # Render label grobs
        label_grobs = self.render_labels(resolved_labels, theme)

        # Draw axis title labels via facet
        if hasattr(self.facet, "draw_labels"):
            plot_table = self.facet.draw_labels(
                plot_table,
                self.layout,
                self.panel_scales_x,
                self.panel_scales_y,
                self.panel_params,
                self.coord,
                data,
                theme,
                label_grobs,
                self.facet_params,
            )

        return plot_table

    def resolve_label(
        self,
        scale: Any,
        labels: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve axis titles from guides, scales, or plot labels.

        Parameters
        ----------
        scale : Scale
            The position scale.
        labels : dict
            Plot labels dictionary.

        Returns
        -------
        dict
            ``{"primary": ..., "secondary": ...}`` title dict.
        """
        aes = scale.aesthetics[0] if scale.aesthetics else "x"

        # From scale name
        prim_scale = getattr(scale, "name", None)
        seco_scale = getattr(scale, "sec_name", None)
        if callable(seco_scale):
            seco_scale = seco_scale()

        # From plot labels
        prim_label = labels.get(aes)
        seco_label = labels.get(f"sec.{aes}")

        # From scale's make_title
        if hasattr(scale, "make_title"):
            primary = scale.make_title(
                prim_scale if not is_waiver(prim_scale) and prim_scale is not None
                else prim_label
            )
        else:
            primary = prim_scale if prim_scale is not None else prim_label

        secondary = seco_scale if seco_scale is not None else seco_label

        return {"primary": primary, "secondary": secondary}

    def render_labels(
        self,
        labels: Dict[str, Any],
        theme: Any,
    ) -> Dict[str, Any]:
        """Render axis title grobs.

        Mirrors R's ``Layout$render_labels``: produces text grobs for
        x-axis and y-axis titles.  Falls back to a simple ``text_grob``
        when theme ``element_render`` is unavailable.

        Parameters
        ----------
        labels : dict
            Resolved labels keyed by ``"x"`` / ``"y"``, each
            ``{"primary": ..., "secondary": ...}``.
        theme : Theme
            Complete theme.

        Returns
        -------
        dict
            ``{"x": [primary_grob, secondary_grob], "y": [...]}``
        """
        from grid_py import null_grob, text_grob, Gpar

        result: Dict[str, Any] = {}
        for axis, label_pair in labels.items():
            grobs = []
            if not isinstance(label_pair, dict):
                result[axis] = [null_grob(), null_grob()]
                continue
            for i, key in enumerate(["primary", "secondary"]):
                val = label_pair.get(key)
                if val is None or is_waiver(val):
                    grobs.append(null_grob())
                else:
                    # R: element_render(theme, "axis.title.x.bottom", label=...,
                    #    margin_x = label == "y", margin_y = label == "x")
                    from ggplot2_py.theme_elements import element_render as _el_render
                    pos = ".bottom" if axis == "x" else ".left"
                    if i == 1:
                        pos = ".top" if axis == "x" else ".right"
                    g = _el_render(
                        theme, f"axis.title.{axis}{pos}",
                        label=str(val),
                        margin_x=(axis == "y"),
                        margin_y=(axis == "x"),
                    )
                    grobs.append(g)
            result[axis] = grobs
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_scales(self, i: int) -> Dict[str, Any]:
        """Get scales for panel *i*.

        Parameters
        ----------
        i : int
            Panel index (1-based, matching ``PANEL`` column values).

        Returns
        -------
        dict
            ``{"x": Scale, "y": Scale}`` for the requested panel.
        """
        row = self.layout.loc[self.layout["PANEL"] == i]
        if row.empty:
            return {"x": None, "y": None}
        row = row.iloc[0]
        sx_idx = int(row["SCALE_X"]) - 1
        sy_idx = int(row["SCALE_Y"]) - 1
        return {
            "x": self.panel_scales_x[sx_idx] if self.panel_scales_x and sx_idx < len(self.panel_scales_x) else None,
            "y": self.panel_scales_y[sy_idx] if self.panel_scales_y and sy_idx < len(self.panel_scales_y) else None,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_layout(
    facet: Any,
    coord: Any,
    layout_cls: Any = None,
) -> Layout:
    """Create a :class:`Layout` instance for a plot.

    Parameters
    ----------
    facet : Facet
        Faceting specification.
    coord : Coord
        Coordinate system.
    layout_cls : type, optional
        Layout subclass to use (defaults to :class:`Layout`).

    Returns
    -------
    Layout
        A configured layout instance.
    """
    cls = layout_cls or Layout
    obj = cls()
    obj.facet = facet
    obj.coord = coord
    return obj

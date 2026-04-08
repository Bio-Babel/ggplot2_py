"""
Core ggplot class, build pipeline, and rendering machinery.

This module implements the central ``GGPlot`` object, the ``ggplot()``
constructor, the ``ggplot_build()`` / ``ggplot_gtable()`` pipeline, the
``+`` operator dispatch (``ggplot_add`` / ``update_ggplot``), last-plot
bookkeeping, and plot-introspection utilities.
"""

from __future__ import annotations

import copy
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd

from ggplot2_py._compat import (
    Waiver,
    is_waiver,
    waiver,
    cli_abort,
    cli_warn,
    cli_inform,
)
from ggplot2_py.ggproto import GGProto, ggproto, is_ggproto
from ggplot2_py.aes import Mapping, aes, is_mapping, standardise_aes_names
from ggplot2_py._utils import compact, modify_list, remove_missing, snake_class
from ggplot2_py.labels import Labels, is_labels, labs, make_labels, update_labels
from ggplot2_py.fortify import fortify

__all__ = [
    "ggplot",
    "is_ggplot",
    "is_ggproto",
    "ggplot_build",
    "ggplot_gtable",
    "ggplotGrob",
    "ggplot_add",
    "add_gg",
    "get_last_plot",
    "set_last_plot",
    "last_plot",
    "get_alt_text",
    "update_ggplot",
    "update_labels",
    "get_layer_data",
    "get_layer_grob",
    "get_panel_scales",
    "get_guide_data",
    "get_strip_labels",
    "get_labs",
    "layer_data",
    "layer_grob",
    "layer_scales",
    "summarise_plot",
    "summarise_coord",
    "summarise_layers",
    "summarise_layout",
    "find_panel",
    "panel_rows",
    "panel_cols",
    "print_plot",
]


# ---------------------------------------------------------------------------
# Last-plot bookkeeping
# ---------------------------------------------------------------------------

_last_plot: Optional["GGPlot"] = None


def get_last_plot() -> Optional["GGPlot"]:
    """Return the last plot created or displayed.

    Returns
    -------
    GGPlot or None
    """
    return _last_plot


def set_last_plot(plot: "GGPlot") -> None:
    """Store *plot* as the last plot (used by ``ggsave`` etc.).

    Parameters
    ----------
    plot : GGPlot
        Plot to record.
    """
    global _last_plot
    _last_plot = plot


last_plot = get_last_plot


# ---------------------------------------------------------------------------
# GGPlot class
# ---------------------------------------------------------------------------

class GGPlot:
    """A ggplot2 plot object.

    ``GGPlot`` is the central data structure.  It stores the default data,
    default mapping, layer stack, scales, coordinate system, faceting
    specification, theme, labels, and guides.

    Attributes
    ----------
    data : DataFrame or Waiver or callable or None
        Default data.
    mapping : Mapping
        Default aesthetic mapping.
    layers : list of Layer
        Layer stack.
    scales : ScalesList
        Scale container.
    theme : Theme or dict
        Theme specification.
    coordinates : Coord
        Coordinate system.
    facet : Facet
        Faceting specification.
    labels : Labels
        Axis / title labels.
    guides : object
        Guides specification.
    plot_env : object
        The environment the plot was created in (unused in Python).
    layout : type
        Layout class used during the build.
    """

    def __init__(
        self,
        data: Any = None,
        mapping: Optional[Mapping] = None,
        *,
        plot_env: Any = None,
    ) -> None:
        # Lazy imports to avoid circular dependencies
        from ggplot2_py.scale import ScalesList

        self.data = data
        self.mapping: Mapping = mapping if mapping is not None else aes()
        self.layers: List[Any] = []
        self.scales: "ScalesList" = ScalesList()
        self.theme: Any = {}  # will be merged with defaults
        self.coordinates: Any = None  # filled lazily via default
        self.facet: Any = None  # filled lazily via default
        self.labels: Labels = Labels()
        self.guides: Any = None  # filled lazily
        self.plot_env: Any = plot_env
        self.layout: Any = None  # Layout class reference
        self._meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def _clone(self) -> "GGPlot":
        """Create a shallow copy with a cloned scales list.

        Returns
        -------
        GGPlot
        """
        p = copy.copy(self)
        p.scales = self.scales.clone()
        p.layers = list(self.layers)  # shallow copy of list
        p.labels = Labels(self.labels)
        return p

    # ------------------------------------------------------------------
    # + operator
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> "GGPlot":
        if other is None:
            return self
        p = self._clone()
        p = ggplot_add(other, p)
        set_last_plot(p)
        return p

    def __radd__(self, other: Any) -> "GGPlot":
        if other is None or other == 0:
            return self
        return self.__add__(other)

    def __iadd__(self, other: Any) -> "GGPlot":
        return self.__add__(other)

    # ------------------------------------------------------------------
    # Attribute access helpers
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        meta = object.__getattribute__(self, "_meta")
        if name in meta:
            return meta[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        # Direct attributes go to __dict__; others to _meta
        if name in (
            "data", "mapping", "layers", "scales", "theme",
            "coordinates", "facet", "labels", "guides", "plot_env",
            "layout", "_meta",
        ):
            object.__setattr__(self, name, value)
        else:
            try:
                meta = object.__getattribute__(self, "_meta")
            except AttributeError:
                object.__setattr__(self, name, value)
                return
            meta[name] = value

    # ------------------------------------------------------------------
    # Repr / summary
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_layers = len(self.layers)
        data_info = ""
        if isinstance(self.data, pd.DataFrame):
            data_info = f" data={self.data.shape[0]}x{self.data.shape[1]}"
        return f"<GGPlot{data_info} layers={n_layers}>"

    # Default display size (inches) and DPI for Jupyter rendering.
    # Override per-plot: ``p.fig_width = 12; p.fig_height = 8``
    # Override globally: ``GGPlot.fig_width = 12``
    fig_width: float = 7.0
    fig_height: float = 5.0
    fig_dpi: int = 150

    def _repr_png_(self) -> Optional[bytes]:
        """Render the plot as PNG bytes for Jupyter notebook display."""
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from grid_py import grid_draw, grid_newpage

        try:
            grid_newpage()
            fig = plt.gcf()
            fig.set_size_inches(self.fig_width, self.fig_height)
            built = ggplot_build(self)
            gtable = ggplot_gtable(built)
            grid_draw(gtable)
            ax = fig.axes[0] if fig.axes else None
            # Add axes labels and title from plot labels
            if ax is not None:
                _mpl_add_labels(ax, self, built)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=self.fig_dpi, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception:
            return None

    def summary(self) -> str:
        """Return a human-readable summary of the plot.

        Returns
        -------
        str
        """
        parts: List[str] = []
        if isinstance(self.data, pd.DataFrame) and not self.data.empty:
            cols = ", ".join(self.data.columns[:10])
            parts.append(
                f"data:     {cols} [{self.data.shape[0]}x{self.data.shape[1]}]"
            )
        if self.mapping:
            parts.append(f"mapping:  {self.mapping}")
        if self.scales.n() > 0:
            parts.append(f"scales:   {', '.join(self.scales.input())}")
        if self.facet is not None and hasattr(self.facet, "vars"):
            fv = self.facet.vars()
            parts.append(f"faceting: {', '.join(fv) if fv else '<none>'}")
        if self.layers:
            parts.append("---")
            for layer in self.layers:
                parts.append(f"  {layer}")
        return "\n".join(parts)


def is_ggplot(x: Any) -> bool:
    """Return ``True`` if *x* is a :class:`GGPlot` instance.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(x, GGPlot)


# ---------------------------------------------------------------------------
# ggplot() constructor
# ---------------------------------------------------------------------------

def ggplot(
    data: Any = None,
    mapping: Optional[Mapping] = None,
    **kwargs: Any,
) -> GGPlot:
    """Create a new ggplot object.

    Parameters
    ----------
    data : DataFrame or dict or None, optional
        Default dataset for the plot.  Will be converted to a DataFrame
        via :func:`fortify` if necessary.
    mapping : Mapping, optional
        Default aesthetic mapping created via :func:`aes`.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    GGPlot
        A new plot object ready for layers to be added with ``+``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    >>> p = ggplot(df, aes(x="x", y="y"))
    """
    if callable(data) and not isinstance(data, (pd.DataFrame, dict, type)):
        cli_abort(
            "`data` cannot be a function. "
            "Have you misspelled the `data` argument in `ggplot()`?",
            cls=TypeError,
        )

    if mapping is not None and not isinstance(mapping, Mapping):
        # Maybe data and mapping were swapped?
        if isinstance(mapping, (pd.DataFrame, dict)):
            data, mapping = mapping, data
        elif is_mapping(mapping):
            pass
        else:
            cli_warn(
                f"Unexpected type for `mapping`: {type(mapping).__name__}. "
                "Expected a Mapping from `aes()`."
            )

    # Validate / convert mapping
    if mapping is None:
        mapping = aes()

    # Fortify data
    data = fortify(data)

    p = GGPlot(data=data, mapping=mapping)

    # Set defaults lazily (coord and facet are imported here to avoid circulars)
    from ggplot2_py.coord import CoordCartesian
    from ggplot2_py.facet import FacetNull

    p.coordinates = CoordCartesian()
    p.coordinates.default = True
    p.facet = FacetNull()

    # Set initial labels from mapping
    p.labels = labs(**make_labels(mapping))

    set_last_plot(p)
    return p


# ---------------------------------------------------------------------------
# BuiltGGPlot
# ---------------------------------------------------------------------------

class BuiltGGPlot:
    """Container returned by :func:`ggplot_build`.

    Attributes
    ----------
    data : list of DataFrame
        Computed data for each layer.
    layout : Layout
        Trained layout object.
    plot : GGPlot
        The (possibly modified) plot object.
    """

    def __init__(
        self,
        data: List[pd.DataFrame],
        layout: Any,
        plot: GGPlot,
    ) -> None:
        self.data = data
        self.layout = layout
        self.plot = plot

    def __repr__(self) -> str:
        return f"<BuiltGGPlot layers={len(self.data)}>"


# ---------------------------------------------------------------------------
# ggplot_build
# ---------------------------------------------------------------------------

def ggplot_build(plot: Any) -> BuiltGGPlot:
    """Build a ggplot for rendering.

    Transforms the plot object through the full data pipeline:
    prepare layer data, setup facets, compute aesthetics, transform
    scales, train positions, compute statistics, apply positions,
    map aesthetics, and compute geom parameters.

    Parameters
    ----------
    plot : GGPlot or BuiltGGPlot
        The plot to build.  If already a ``BuiltGGPlot``, returns it
        unchanged.

    Returns
    -------
    BuiltGGPlot
        Containing computed data, trained layout, and the plot reference.
    """
    # No-op for already-built plots
    if isinstance(plot, BuiltGGPlot):
        return plot

    from ggplot2_py.layout import Layout, create_layout
    from ggplot2_py.theme import complete_theme

    plot = plot._clone()

    # Ensure at least one layer
    if len(plot.layers) == 0:
        # Add a blank layer
        try:
            from ggplot2_py.geom import geom_blank
            blank = geom_blank()
            plot.layers.append(blank)
        except ImportError:
            pass

    layers = plot.layers
    data: List[Optional[pd.DataFrame]] = [None] * len(layers)
    scales = plot.scales

    # --- Layer data ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "layer_data"):
            data[i] = layer.layer_data(plot.data)
        elif hasattr(layer, "data") and layer.data is not None:
            d = layer.data
            if callable(d) and not isinstance(d, pd.DataFrame):
                d = d(plot.data)
            if isinstance(d, pd.DataFrame):
                data[i] = d
            else:
                data[i] = plot.data if isinstance(plot.data, pd.DataFrame) else pd.DataFrame()
        else:
            data[i] = plot.data if isinstance(plot.data, pd.DataFrame) else pd.DataFrame()

    # --- Setup layers ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "setup_layer"):
            data[i] = layer.setup_layer(data[i], plot)

    # --- Setup layout ---
    layout = create_layout(
        plot.facet,
        plot.coordinates,
        getattr(plot, "layout", None),
    )
    data = layout.setup(data, plot.data if isinstance(plot.data, pd.DataFrame) else pd.DataFrame(), plot.plot_env)

    # --- Compute aesthetics ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "compute_aesthetics"):
            data[i] = layer.compute_aesthetics(data[i], plot)

    # --- Add default scales for all aesthetics present in the data ---
    for i in range(len(data)):
        if data[i] is not None and not data[i].empty:
            scales.add_defaults(data[i], plot.plot_env)

    # --- Setup plot labels (from layer mappings) ---
    _setup_plot_labels(plot, layers, data)

    # --- Transform scales ---
    for i in range(len(data)):
        if data[i] is not None and not data[i].empty:
            data[i] = scales.transform_df(data[i])

    # --- Train and map positions ---
    scale_x = scales.get_scales("x")
    scale_y = scales.get_scales("y")
    layout.train_position(data, scale_x, scale_y)
    data = layout.map_position(data)

    # --- Compute statistics ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "compute_statistic"):
            data[i] = layer.compute_statistic(data[i], layout)

    # --- Map statistics ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "map_statistic"):
            data[i] = layer.map_statistic(data[i], plot)

    # --- Add missing scales ---
    scales.add_missing(["x", "y"], plot.plot_env)

    # --- Compute geom 1 (reparameterize) ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "compute_geom_1"):
            data[i] = layer.compute_geom_1(data[i])

    # --- Compute position adjustments ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "compute_position"):
            data[i] = layer.compute_position(data[i], layout)

    # --- Reset and retrain position scales ---
    layout.reset_scales()
    layout.train_position(data, scale_x, scale_y)
    layout.setup_panel_params()
    data = layout.map_position(data)

    # --- Setup panel guides ---
    layout.setup_panel_guides(plot.guides, plot.layers)

    # --- Complete theme ---
    if hasattr(plot, "theme"):
        try:
            plot.theme = complete_theme(plot.theme)
        except Exception:
            pass

    # --- Train non-position scales and guides ---
    npscales = scales.non_position_scales()
    if npscales.n() > 0:
        if hasattr(npscales, "set_palettes"):
            npscales.set_palettes(plot.theme)
        for d in data:
            if d is not None:
                npscales.train_df(d)
        # Build guides for non-position scales
        if plot.guides is not None and hasattr(plot.guides, "build"):
            plot.guides = plot.guides.build(
                npscales, plot.layers, plot.labels, data, plot.theme,
            )
        # Map non-position scales
        for i in range(len(data)):
            if data[i] is not None:
                data[i] = npscales.map_df(data[i])
    else:
        if plot.guides is not None and hasattr(plot.guides, "get_custom"):
            plot.guides = plot.guides.get_custom()

    # --- Compute geom 2 (fill defaults) ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "compute_geom_2"):
            data[i] = layer.compute_geom_2(data[i], theme=plot.theme)

    # --- Finish statistics ---
    for i, layer in enumerate(layers):
        if hasattr(layer, "finish_statistics"):
            data[i] = layer.finish_statistics(data[i])

    # --- Finish data ---
    data = layout.finish_data(data)

    # --- Consolidate alt-text ---
    plot.labels["alt"] = get_alt_text(plot)

    return BuiltGGPlot(data=data, layout=layout, plot=plot)


def _setup_plot_labels(
    plot: GGPlot,
    layers: List[Any],
    data: List[pd.DataFrame],
) -> None:
    """Collect default labels from layer mappings and merge with plot labels.

    Parameters
    ----------
    plot : GGPlot
        The plot (modified in-place).
    layers : list
        Plot layers.
    data : list of DataFrame
        Computed data per layer.
    """
    auto_labels: Dict[str, str] = {}
    for i, layer in enumerate(layers):
        mapping = getattr(layer, "computed_mapping", None)
        if mapping is None:
            mapping = getattr(layer, "mapping", None)
        if mapping is None:
            continue
        layer_labels = make_labels(mapping)
        # Default labels from stat
        if hasattr(layer, "stat") and hasattr(layer.stat, "default_aes"):
            stat_labels = make_labels(layer.stat.default_aes)
            for k, v in stat_labels.items():
                if k not in layer_labels:
                    layer_labels[k] = v
        # Merge: first layer wins
        for k, v in layer_labels.items():
            if k not in auto_labels:
                auto_labels[k] = v

    # Merge: user labels override auto labels
    merged = Labels(auto_labels)
    merged.update(plot.labels)
    plot.labels = merged


# ---------------------------------------------------------------------------
# ggplot_gtable
# ---------------------------------------------------------------------------

def ggplot_gtable(data: BuiltGGPlot) -> Any:
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

    # Legends
    if plot.guides is not None and hasattr(plot.guides, "assemble"):
        legend_box = plot.guides.assemble(theme)
        if legend_box is not None:
            # TODO: table_add_legends
            pass

    # Title / subtitle / caption / tag annotations
    plot_table = _table_add_titles(plot_table, labels, theme)

    # Add alt-text attribute
    if hasattr(plot_table, "__dict__"):
        plot_table._alt_label = labels.get("alt", "")

    return plot_table


def _table_add_titles(table: Any, labels: Dict[str, Any], theme: Any) -> Any:
    """Add title, subtitle, caption, tag annotations to the plot table.

    Parameters
    ----------
    table : gtable
        The plot gtable.
    labels : dict
        Plot labels.
    theme : Theme
        Complete theme.

    Returns
    -------
    gtable
        Modified table.
    """
    # Stub: in a full implementation this would render title/subtitle/caption
    # grobs and add them to the gtable.  For now, pass through.
    return table


# ---------------------------------------------------------------------------
# ggplotGrob
# ---------------------------------------------------------------------------

def ggplotGrob(plot: GGPlot) -> Any:
    """Build and convert a ggplot to a gtable grob.

    Parameters
    ----------
    plot : GGPlot
        A ggplot object.

    Returns
    -------
    gtable
    """
    return ggplot_gtable(ggplot_build(plot))


# ---------------------------------------------------------------------------
# ggplot_add / update_ggplot dispatch
# ---------------------------------------------------------------------------

def ggplot_add(obj: Any, plot: GGPlot, object_name: str = "") -> GGPlot:
    """Add an object to a ggplot (generic dispatch).

    This is the Python equivalent of R's ``ggplot_add()`` S3 generic.
    It dispatches based on the type of *obj*.

    Parameters
    ----------
    obj : object
        The component to add.
    plot : GGPlot
        The plot to modify.
    object_name : str, optional
        Name of the object (for error messages).

    Returns
    -------
    GGPlot
        The modified plot.
    """
    return update_ggplot(obj, plot, object_name)


def update_ggplot(obj: Any, plot: GGPlot, object_name: str = "") -> GGPlot:
    """Core dispatch for adding components to a ggplot.

    Parameters
    ----------
    obj : object
        Component to add.
    plot : GGPlot
        Target plot.
    object_name : str
        Name for error messages.

    Returns
    -------
    GGPlot
    """
    from ggplot2_py.layer import Layer, is_layer
    from ggplot2_py.scale import Scale, ScalesList
    from ggplot2_py.coord import Coord, is_coord
    from ggplot2_py.facet import Facet, is_facet
    from ggplot2_py.theme import Theme, is_theme, add_theme

    # None -> no-op
    if obj is None:
        return plot

    # Layer
    if is_layer(obj):
        plot.layers.append(obj)
        return plot

    # List of items
    if isinstance(obj, list):
        for item in obj:
            plot = ggplot_add(item, plot, object_name)
        return plot

    # Scale
    if isinstance(obj, Scale):
        plot.scales.add(obj)
        return plot

    # Labels
    if is_labels(obj):
        merged = Labels(plot.labels)
        merged.update(obj)
        plot.labels = merged
        return plot

    # Mapping (aes)
    if is_mapping(obj):
        # Merge new mapping with existing (new overrides)
        merged_mapping = aes(**{**plot.mapping, **obj})
        plot.mapping = merged_mapping
        return plot

    # Coord
    if is_coord(obj):
        if (
            not getattr(plot.coordinates, "default", True)
            and getattr(obj, "default", False)
        ):
            # Don't let a default coord override a non-default one
            return plot
        if not getattr(plot.coordinates, "default", True):
            cli_inform(
                "Coordinate system already present. "
                "Adding new coordinate system, which will replace the existing one."
            )
        plot.coordinates = obj
        return plot

    # Facet
    if is_facet(obj):
        plot.facet = obj
        return plot

    # Theme
    if is_theme(obj):
        plot.theme = add_theme(plot.theme, obj)
        return plot

    # Guides
    if hasattr(obj, "_is_guides") and obj._is_guides:
        if plot.guides is not None and hasattr(plot.guides, "add"):
            plot.guides.add(obj)
        else:
            plot.guides = obj
        return plot

    # DataFrame -> replace default data
    if isinstance(obj, pd.DataFrame):
        plot.data = obj
        return plot

    # GGProto (shouldn't be added directly)
    if is_ggproto(obj):
        cli_abort(
            "Cannot add ggproto objects together. "
            "Did you forget to add this object to a ggplot object?"
        )

    # Callable -> error with hint
    if callable(obj):
        name = object_name or getattr(obj, "__name__", "object")
        cli_abort(
            f"Cannot add `{name}` to a ggplot object. "
            f"Did you forget to add parentheses, as in `{name}()`?"
        )

    cli_abort(
        f"Cannot add `{object_name or type(obj).__name__}` to a ggplot object."
    )


def add_gg(e1: Any, e2: Any) -> Any:
    """Implement the ``+`` operator for gg objects.

    Parameters
    ----------
    e1 : GGPlot or Theme
        Left-hand side.
    e2 : object
        Right-hand side component.

    Returns
    -------
    GGPlot or Theme
    """
    from ggplot2_py.theme import is_theme, add_theme

    if is_theme(e1):
        return add_theme(e1, e2)
    elif is_ggplot(e1):
        return e1 + e2
    elif is_ggproto(e1):
        cli_abort(
            "Cannot add ggproto objects together. "
            "Did you forget to add this object to a ggplot object?"
        )
    else:
        cli_abort(f"Cannot use `+` with {type(e1).__name__}.")


# ---------------------------------------------------------------------------
# Alt-text
# ---------------------------------------------------------------------------

def get_alt_text(plot: Any) -> str:
    """Extract alt-text from a plot.

    Parameters
    ----------
    plot : GGPlot or BuiltGGPlot or gtable
        The plot or built plot.

    Returns
    -------
    str
        Alt-text string, or empty string if none is set.
    """
    if isinstance(plot, BuiltGGPlot):
        alt = plot.plot.labels.get("alt", "")
        if callable(alt):
            return alt(plot.plot)
        return alt or ""

    if isinstance(plot, GGPlot):
        alt = plot.labels.get("alt", "")
        if callable(alt):
            # Would need to build first; just return empty
            return ""
        return alt or ""

    # gtable
    if hasattr(plot, "_alt_label"):
        return plot._alt_label or ""

    return ""


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def get_layer_data(
    plot: Any = None,
    i: int = 1,
) -> pd.DataFrame:
    """Return the computed data for a given layer.

    Parameters
    ----------
    plot : GGPlot or None
        Plot to inspect.  ``None`` uses :func:`get_last_plot`.
    i : int
        Layer index (1-based).

    Returns
    -------
    DataFrame
    """
    if plot is None:
        plot = get_last_plot()
    built = ggplot_build(plot)
    idx = i - 1  # Convert to 0-based
    if idx < 0 or idx >= len(built.data):
        cli_abort(f"Layer index {i} out of range (plot has {len(built.data)} layers).")
    return built.data[idx]


layer_data = get_layer_data


def get_layer_grob(
    plot: Any = None,
    i: int = 1,
) -> Any:
    """Return the grob for a given layer.

    Parameters
    ----------
    plot : GGPlot or None
        Plot to inspect.
    i : int
        Layer index (1-based).

    Returns
    -------
    grob
    """
    if plot is None:
        plot = get_last_plot()
    built = ggplot_build(plot)
    idx = i - 1
    if idx < 0 or idx >= len(built.data):
        cli_abort(f"Layer index {i} out of range.")
    layer = built.plot.layers[idx]
    if hasattr(layer, "draw_geom"):
        return layer.draw_geom(built.data[idx], built.layout)
    return None


layer_grob = get_layer_grob


def get_panel_scales(
    plot: Any = None,
    i: int = 1,
    j: int = 1,
) -> Dict[str, Any]:
    """Return position scales for a specific panel.

    Parameters
    ----------
    plot : GGPlot or None
        Plot to inspect.
    i : int
        Row index (1-based).
    j : int
        Column index (1-based).

    Returns
    -------
    dict
        ``{"x": Scale, "y": Scale}``
    """
    if plot is None:
        plot = get_last_plot()
    built = ggplot_build(plot)
    layout_df = built.layout.layout
    sel = layout_df[(layout_df["ROW"] == i) & (layout_df["COL"] == j)]
    if sel.empty:
        return {"x": None, "y": None}
    row = sel.iloc[0]
    sx_idx = int(row["SCALE_X"]) - 1
    sy_idx = int(row["SCALE_Y"]) - 1
    return {
        "x": built.layout.panel_scales_x[sx_idx] if built.layout.panel_scales_x else None,
        "y": built.layout.panel_scales_y[sy_idx] if built.layout.panel_scales_y else None,
    }


layer_scales = get_panel_scales


def get_guide_data(
    plot: Any = None,
    aesthetic: str = "colour",
) -> Any:
    """Retrieve guide data for a given aesthetic (stub).

    Parameters
    ----------
    plot : GGPlot or None
        Plot to inspect.
    aesthetic : str
        Aesthetic name.

    Returns
    -------
    object
        Guide data or ``None``.
    """
    return None


def get_strip_labels(
    plot: Any = None,
) -> Any:
    """Retrieve strip labels from a faceted plot (stub).

    Parameters
    ----------
    plot : GGPlot or None
        Plot to inspect.

    Returns
    -------
    dict or None
    """
    return None


def get_labs(plot: Any = None) -> Labels:
    """Retrieve resolved labels from a plot.

    Parameters
    ----------
    plot : GGPlot or None
        Plot to inspect.

    Returns
    -------
    Labels
    """
    from ggplot2_py.labels import get_labs as _get_labs
    return _get_labs(plot)


# ---------------------------------------------------------------------------
# Summary / introspection
# ---------------------------------------------------------------------------

def summarise_plot(plot: GGPlot) -> Dict[str, Any]:
    """Summarise a plot's main components.

    Parameters
    ----------
    plot : GGPlot
        Plot to summarise.

    Returns
    -------
    dict
    """
    return {
        "data": type(plot.data).__name__,
        "mapping": dict(plot.mapping) if plot.mapping else {},
        "n_layers": len(plot.layers),
        "coord": type(plot.coordinates).__name__ if plot.coordinates else None,
        "facet": type(plot.facet).__name__ if plot.facet else None,
    }


def summarise_coord(plot: GGPlot) -> Dict[str, Any]:
    """Summarise the coordinate system.

    Parameters
    ----------
    plot : GGPlot

    Returns
    -------
    dict
    """
    coord = plot.coordinates
    if coord is None:
        return {}
    return {
        "class": type(coord).__name__,
        "default": getattr(coord, "default", None),
    }


def summarise_layers(plot: GGPlot) -> List[Dict[str, Any]]:
    """Summarise each layer.

    Parameters
    ----------
    plot : GGPlot

    Returns
    -------
    list of dict
    """
    result = []
    for layer in plot.layers:
        info: Dict[str, Any] = {}
        if hasattr(layer, "geom"):
            info["geom"] = type(layer.geom).__name__ if not isinstance(layer.geom, str) else layer.geom
        if hasattr(layer, "stat"):
            info["stat"] = type(layer.stat).__name__ if not isinstance(layer.stat, str) else layer.stat
        if hasattr(layer, "mapping") and layer.mapping:
            info["mapping"] = dict(layer.mapping)
        result.append(info)
    return result


def summarise_layout(plot: GGPlot) -> Dict[str, Any]:
    """Summarise the layout / faceting.

    Parameters
    ----------
    plot : GGPlot

    Returns
    -------
    dict
    """
    facet = plot.facet
    if facet is None:
        return {}
    return {
        "class": type(facet).__name__,
        "vars": list(facet.vars()) if hasattr(facet, "vars") else [],
    }


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

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


def _mpl_add_labels(ax: Any, plot: GGPlot, built: Any) -> None:
    """Add axes ticks, labels, and title to a matplotlib Axes.

    Uses the plot labels and panel_params from the build to annotate
    the matplotlib axes produced by grid_draw.
    """
    import numpy as np
    labels = plot.labels if hasattr(plot, "labels") else {}
    layout = built.layout

    # Axis labels
    x_lab = labels.get("x", "")
    y_lab = labels.get("y", "")
    if x_lab:
        ax.set_xlabel(str(x_lab), fontsize=11)
    if y_lab:
        ax.set_ylabel(str(y_lab), fontsize=11)

    # Title / subtitle
    title = labels.get("title", "")
    subtitle = labels.get("subtitle", "")
    if title:
        t = str(title)
        if subtitle:
            t += "\n" + str(subtitle)
        ax.set_title(t, fontsize=13, loc="left")

    # Axis ticks from panel_params
    if layout.panel_params:
        pp = layout.panel_params[0]
        x_range = pp.get("x_range") or pp.get("x.range")
        y_range = pp.get("y_range") or pp.get("y.range")

        if x_range is not None:
            xmin, xmax = float(x_range[0]), float(x_range[1])
            # Generate nice ticks
            ticks = np.linspace(xmin, xmax, 6)
            # Convert data ticks to NPC [0, 1]
            span = xmax - xmin
            if span > 0:
                npc_ticks = (ticks - xmin) / span
                ax.set_xticks(npc_ticks)
                ax.set_xticklabels([f"{t:.4g}" for t in ticks], fontsize=9)
            ax.tick_params(axis="x", direction="out", length=3)

        if y_range is not None:
            ymin, ymax = float(y_range[0]), float(y_range[1])
            ticks = np.linspace(ymin, ymax, 6)
            span = ymax - ymin
            if span > 0:
                npc_ticks = (ticks - ymin) / span
                ax.set_yticks(npc_ticks)
                ax.set_yticklabels([f"{t:.4g}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", direction="out", length=3)

    # Turn on axes frame
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("grey")
    ax.tick_params(bottom=True, left=True, top=False, right=False)


# ---------------------------------------------------------------------------
# print_plot
# ---------------------------------------------------------------------------

def print_plot(
    plot: GGPlot,
    newpage: bool = True,
    vp: Any = None,
) -> GGPlot:
    """Render a ggplot to the current matplotlib figure.

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

    # Add axes labels and title using matplotlib
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    ax = fig.axes[0] if fig.axes else None
    if ax is not None:
        _mpl_add_labels(ax, plot, built)

    return plot

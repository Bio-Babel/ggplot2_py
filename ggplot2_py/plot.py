"""
Core ggplot class, build pipeline, and operator dispatch.

This module implements the central ``GGPlot`` object, the ``ggplot()``
constructor, the ``ggplot_build()`` pipeline, the ``+`` operator dispatch
(``ggplot_add`` / ``update_ggplot``), last-plot bookkeeping, and
plot-introspection utilities.

Rendering functions (``ggplot_gtable``, ``_table_add_legends``,
``_table_add_titles``, ``ggplotGrob``, ``print_plot``, ``find_panel``,
``panel_rows``, ``panel_cols``) are in ``plot_render.py``, mirroring R's
separation of ``plot-build.R`` / ``plot-render.R``.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import warnings
from functools import singledispatch
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
    "by_layer",
    "BuildStage",
    "ggplot_defaults",
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
# Scoped defaults — ggplot_defaults context manager (Python-exclusive)
# ---------------------------------------------------------------------------

_ggplot_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "_ggplot_context", default={}
)


@contextlib.contextmanager
def ggplot_defaults(
    *,
    theme: Any = None,
    coord: Any = None,
    facet: Any = None,
    mapping: Any = None,
):
    """Context manager for scoped plot defaults.

    **Python-exclusive feature** — R has ``theme_set()`` for global state,
    but no scoped equivalent.  This context manager lets you set defaults
    that apply to all :func:`ggplot` calls within the ``with`` block,
    without affecting code outside.

    Parameters
    ----------
    theme : Theme or dict, optional
        Default theme applied to all plots in scope.
    coord : Coord, optional
        Default coordinate system.
    facet : Facet, optional
        Default faceting specification.
    mapping : Mapping, optional
        Default aesthetic mapping.

    Examples
    --------
    ::

        with ggplot_defaults(theme=theme_minimal(), coord=coord_fixed()):
            p1 = ggplot(df, aes("x", "y")) + geom_point()   # gets theme_minimal + coord_fixed
            p2 = ggplot(df, aes("x", "y")) + geom_bar()     # same defaults
        # Outside: no defaults applied
    """
    ctx: Dict[str, Any] = {}
    if theme is not None:
        ctx["theme"] = theme
    if coord is not None:
        ctx["coord"] = coord
    if facet is not None:
        ctx["facet"] = facet
    if mapping is not None:
        ctx["mapping"] = mapping

    token = _ggplot_context.set(ctx)
    try:
        yield
    finally:
        _ggplot_context.reset(token)


def _get_context_defaults() -> Dict[str, Any]:
    """Return the current scoped defaults (empty dict if none)."""
    return _ggplot_context.get()


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
        from ggplot2_py.theme import Theme

        self.data = data
        self.mapping: Mapping = mapping if mapping is not None else aes()
        self.layers: List[Any] = []
        self.scales: "ScalesList" = ScalesList()
        # R: plot$theme starts as an empty theme() object — not a bare list.
        # Must be a Theme instance so complete_theme() can access .complete
        # and ggplot2_py.theme.add_theme() works via operator overloads.
        self.theme: Theme = Theme()
        self.coordinates: Any = None  # filled lazily via default
        self.facet: Any = None  # filled lazily via default
        self.labels: Labels = Labels()
        self.guides: Any = None
        self.plot_env: Any = plot_env
        self.layout: Any = None  # Layout class reference
        self._meta: Dict[str, Any] = {}
        self._build_hooks: Dict[Tuple[str, str], List[Callable]] = {}

        # Apply scoped context defaults (Python-exclusive feature).
        ctx = _get_context_defaults()
        if ctx:
            if "theme" in ctx and not self.theme:
                self.theme = ctx["theme"]
            if "coord" in ctx and self.coordinates is None:
                self.coordinates = ctx["coord"]
            if "facet" in ctx and self.facet is None:
                self.facet = ctx["facet"]
            if "mapping" in ctx:
                # Merge: context defaults as base, explicit mapping overrides
                merged = aes(**{**ctx["mapping"], **self.mapping})
                self.mapping = merged

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
    # Build hooks (Python-exclusive — no R equivalent)
    # ------------------------------------------------------------------

    def add_build_hook(
        self,
        timing: str,
        stage: str,
        fn: Callable,
    ) -> "GGPlot":
        """Register a callback on a named pipeline stage.

        **Python-exclusive feature** — R's ggplot2 does not support
        build-stage hooks.

        Parameters
        ----------
        timing : ``"before"`` or ``"after"``
            Whether to run before or after the named stage.
        stage : str
            A :class:`BuildStage` constant (e.g.
            ``BuildStage.COMPUTE_STAT``).
        fn : callable
            ``fn(data, **ctx) -> data_or_None``.  Receives the current
            per-layer data list.  Return a new list to replace it, or
            ``None`` to leave it unchanged.

        Returns
        -------
        GGPlot
            ``self`` (for chaining).
        """
        if timing not in ("before", "after"):
            raise ValueError(f"timing must be 'before' or 'after', got {timing!r}")
        key = (timing, stage)
        self._build_hooks.setdefault(key, []).append(fn)
        return self

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
            "layout", "_meta", "_build_hooks",
            # Rendering hints — user can override before notebook display.
            "fig_width", "fig_height", "fig_dpi",
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
        """Render the plot as PNG bytes for Jupyter notebook display.

        R's ``print.ggplot`` raises rendering errors; here we surface them
        via :func:`cli_warn` (with the traceback) rather than silently
        swallowing. Returning ``None`` is required by the Jupyter display
        protocol so the next ``_repr_*_`` can be tried, but the warning
        ensures the failure is visible for debugging.
        """
        from grid_py import grid_draw, grid_newpage

        try:
            grid_newpage(
                width=self.fig_width,
                height=self.fig_height,
                dpi=float(self.fig_dpi),
            )
            built = ggplot_build(self)
            gtable = ggplot_gtable(built)
            grid_draw(gtable)

            from grid_py import get_state
            renderer = get_state().get_renderer()
            if renderer is not None:
                return renderer.to_png_bytes()
            return None
        except (KeyboardInterrupt, SystemExit):
            # Never swallow these: user / runtime signals.
            raise
        except Exception as exc:  # noqa: BLE001 — Jupyter _repr_* must not raise
            import traceback as _tb
            cli_warn(
                "Failed to render ggplot via _repr_png_: "
                f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"
            )
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

    # R parity: ``plot$labels`` stores ONLY user-set labels.  The
    # aesthetic-derived defaults (``x="carat"``, ``y="price"`` etc.)
    # are computed lazily at render time by ``_setup_plot_labels``
    # (mirrors R ``labels.R::make_labels`` invoked from
    # ``ggplot_build``).  Pre-filling here would silently mask the
    # distinction between "user wrote ``labs(x=...)``" and "label was
    # auto-derived from aes mapping" — a divergence that surfaced when
    # comparing ``p$labels$x`` between R and Python.
    p.labels = Labels()

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
# ---------------------------------------------------------------------------
# BuildStage — named pipeline stage constants (Python-exclusive feature).
#
# R's pipeline is fixed-sequence with no hook points.  This enum-like class
# gives each stage a stable name so that the hook system (below) can target
# specific stages.
# ---------------------------------------------------------------------------


class BuildStage:
    """Named constants for the ``ggplot_build`` pipeline stages.

    These are used with :meth:`GGPlot.add_build_hook` to register
    before/after callbacks on specific pipeline stages.  This is a
    **Python-exclusive** extension point — R's ggplot2 does not expose
    hooks on individual build stages.

    Example
    -------
    ::

        p = ggplot(df, aes("x", "y"))
        p.add_build_hook("after", BuildStage.COMPUTE_STAT, my_callback)
    """

    LAYER_DATA = "layer_data"
    SETUP_LAYER = "setup_layer"
    SETUP_LAYOUT = "setup_layout"
    COMPUTE_AESTHETICS = "compute_aesthetics"
    TRANSFORM_SCALES = "transform_scales"
    TRAIN_POSITION = "train_position"
    COMPUTE_STAT = "compute_stat"
    MAP_STAT = "map_stat"
    COMPUTE_GEOM_1 = "compute_geom_1"
    COMPUTE_POSITION = "compute_position"
    RETRAIN_POSITION = "retrain_position"
    SETUP_GUIDES = "setup_guides"
    TRAIN_NONPOSITION = "train_nonposition"
    COMPUTE_GEOM_2 = "compute_geom_2"
    FINISH_STAT = "finish_stat"
    FINISH_DATA = "finish_data"


def _run_hooks(
    plot: GGPlot,
    timing: str,
    stage: str,
    data: List[Any],
    **ctx: Any,
) -> List[Any]:
    """Execute registered build hooks for (*timing*, *stage*).

    Parameters
    ----------
    plot : GGPlot
        The plot whose hooks to run.
    timing : ``"before"`` or ``"after"``
        When relative to the stage.
    stage : str
        One of the :class:`BuildStage` constants.
    data : list
        Current per-layer data list.
    **ctx
        Additional context (e.g. ``layout``, ``scales``) passed to hooks.

    Returns
    -------
    list
        Possibly modified per-layer data.
    """
    hooks = getattr(plot, "_build_hooks", None)
    if not hooks:
        return data
    for hook in hooks.get((timing, stage), []):
        result = hook(data, **ctx)
        if result is not None:
            data = result
    return data


# ---------------------------------------------------------------------------
# by_layer — apply a function per-layer with error context
# (R ref: plot-build.R:194-211)
# ---------------------------------------------------------------------------


def by_layer(
    fn: Callable,
    layers: List[Any],
    data: List[Any],
    step: str = "",
) -> List[Any]:
    """Apply *fn(layer, data_i)* for each layer.

    Mirrors R's ``by_layer()`` helper in *plot-build.R:194-211*.  Wraps
    each call in a try/except so that errors include the layer index.

    Parameters
    ----------
    fn : callable
        ``fn(layer, data_i) -> data_i`` to apply.
    layers : list
        Layer objects.
    data : list
        Parallel list of per-layer DataFrames.
    step : str
        Human-readable description of the current pipeline stage
        (used in error messages).

    Returns
    -------
    list
        Updated per-layer DataFrames.
    """
    out: List[Any] = [None] * len(data)
    for i in range(len(data)):
        try:
            out[i] = fn(layers[i], data[i])
        except Exception as e:
            raise RuntimeError(
                f"Problem while {step}: error in layer {i + 1}."
            ) from e
    return out


# ---------------------------------------------------------------------------
# ggplot_build
# ---------------------------------------------------------------------------

@singledispatch
def ggplot_build(plot: Any) -> BuiltGGPlot:
    """Build a ggplot for rendering.

    This is a :func:`functools.singledispatch` generic (R ref:
    ``plot-build.R:28``, ``UseMethod("ggplot_build")``).  Extension
    packages can register custom plot types::

        @ggplot_build.register(MyPlotClass)
        def _build_my_plot(plot):
            ...

    Parameters
    ----------
    plot : GGPlot or BuiltGGPlot
        The plot to build.

    Returns
    -------
    BuiltGGPlot
    """
    raise TypeError(
        f"Cannot build object of type {type(plot).__name__}. "
        "Expected a GGPlot or BuiltGGPlot instance."
    )


@ggplot_build.register(BuiltGGPlot)
def _build_noop(plot):
    """Already-built plots are returned unchanged (R: no-op method)."""
    return plot


@ggplot_build.register(GGPlot)
def _build_ggplot(plot):
    """Build a GGPlot through the full data pipeline."""
    from ggplot2_py.layout import Layout, create_layout
    from ggplot2_py.theme import complete_theme

    plot = plot._clone()

    # Ensure at least one layer. ``geom_blank`` is an in-package constructor,
    # not an optional dependency, so any ImportError here is a real bug and
    # must propagate (matches R's ``ggplot_build`` which unconditionally
    # appends ``geom_blank()`` when ``plot$layers`` is empty).
    if len(plot.layers) == 0:
        from ggplot2_py.geom import geom_blank
        plot.layers.append(geom_blank())

    layers = plot.layers
    data: List[Optional[pd.DataFrame]] = [None] * len(layers)
    scales = plot.scales

    _h = _run_hooks  # local alias for brevity
    S = BuildStage

    # --- Layer data ---
    data = _h(plot, "before", S.LAYER_DATA, data)
    data = by_layer(lambda l, d: l.layer_data(plot.data), layers, data, "computing layer data")
    data = _h(plot, "after", S.LAYER_DATA, data)

    # --- Setup layers ---
    data = _h(plot, "before", S.SETUP_LAYER, data)
    data = by_layer(lambda l, d: l.setup_layer(d, plot), layers, data, "setting up layer")
    data = _h(plot, "after", S.SETUP_LAYER, data)

    # --- Setup layout ---
    layout = create_layout(plot.facet, plot.coordinates, getattr(plot, "layout", None))
    data = layout.setup(data, plot.data if isinstance(plot.data, pd.DataFrame) else pd.DataFrame(), plot.plot_env)

    # --- Compute aesthetics ---
    data = _h(plot, "before", S.COMPUTE_AESTHETICS, data)
    data = by_layer(lambda l, d: l.compute_aesthetics(d, plot), layers, data, "computing aesthetics")
    data = _h(plot, "after", S.COMPUTE_AESTHETICS, data)

    # --- Add default scales ---
    for i in range(len(data)):
        if data[i] is not None and not data[i].empty:
            scales.add_defaults(data[i], plot.plot_env)

    # --- Setup plot labels ---
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
    data = _h(plot, "before", S.COMPUTE_STAT, data)
    data = by_layer(lambda l, d: l.compute_statistic(d, layout), layers, data, "computing stat")
    data = _h(plot, "after", S.COMPUTE_STAT, data)

    # --- Map statistics ---
    data = _h(plot, "before", S.MAP_STAT, data)
    data = by_layer(lambda l, d: l.map_statistic(d, plot), layers, data, "mapping stat to aesthetics")
    data = _h(plot, "after", S.MAP_STAT, data)

    # R (layer.R:map_statistic, plot-build.R:83): after stats map new
    # aesthetics into data (e.g. ``fill = after_stat(count)`` from
    # StatBinhex), scales for those new columns must be registered
    # with the plot.  Without this step ``geom_hex`` ends up with a
    # ``fill`` column of numeric counts but no scale_fill, so the
    # colourbar legend never appears and the hex cells render
    # without the gradient.
    for i in range(len(data)):
        if data[i] is not None and not data[i].empty:
            scales.add_defaults(data[i], plot.plot_env)

    # --- Add missing scales ---
    scales.add_missing(["x", "y"], plot.plot_env)

    # --- Compute geom 1 ---
    data = _h(plot, "before", S.COMPUTE_GEOM_1, data)
    data = by_layer(lambda l, d: l.compute_geom_1(d), layers, data, "setting up geom")
    data = _h(plot, "after", S.COMPUTE_GEOM_1, data)

    # --- Compute position ---
    data = _h(plot, "before", S.COMPUTE_POSITION, data)
    data = by_layer(lambda l, d: l.compute_position(d, layout), layers, data, "computing position")
    data = _h(plot, "after", S.COMPUTE_POSITION, data)

    # --- Reset and retrain position scales ---
    scale_x = scales.get_scales("x")
    scale_y = scales.get_scales("y")
    layout.reset_scales()
    layout.train_position(data, scale_x, scale_y)
    layout.setup_panel_params()
    data = layout.map_position(data)

    # --- Setup panel guides ---
    layout.setup_panel_guides(plot.guides, plot.layers)

    # --- Complete theme ---
    # R: plot@theme <- plot_theme(plot)  (plot-build.R:107)
    # No try/except: R lets theme errors surface; silently discarding them
    # leaves ``plot.theme`` as an incomplete object and every downstream
    # ``calc_element`` then returns ``None`` (no bg, no grid, no titles).
    plot.theme = complete_theme(plot.theme)

    # --- Train non-position scales and guides ---
    npscales = scales.non_position_scales()
    if npscales.n() > 0:
        if hasattr(npscales, "set_palettes"):
            npscales.set_palettes(plot.theme)
        for d in data:
            if d is not None:
                npscales.train_df(d)
        if plot.guides is not None and hasattr(plot.guides, "build"):
            plot.guides = plot.guides.build(npscales, plot.layers, plot.labels, data, plot.theme)
        for i in range(len(data)):
            if data[i] is not None:
                data[i] = npscales.map_df(data[i])
    else:
        if plot.guides is not None and hasattr(plot.guides, "get_custom"):
            plot.guides = plot.guides.get_custom()

    # --- Compute geom 2 ---
    data = _h(plot, "before", S.COMPUTE_GEOM_2, data)
    data = by_layer(lambda l, d: l.compute_geom_2(d, theme=plot.theme), layers, data, "setting up geom aesthetics")
    data = _h(plot, "after", S.COMPUTE_GEOM_2, data)

    # --- Finish statistics ---
    data = _h(plot, "before", S.FINISH_STAT, data)
    data = by_layer(lambda l, d: l.finish_statistics(d), layers, data, "finishing layer stat")
    data = _h(plot, "after", S.FINISH_STAT, data)

    # --- Finish data ---
    data = _h(plot, "before", S.FINISH_DATA, data)
    data = layout.finish_data(data)
    data = _h(plot, "after", S.FINISH_DATA, data)

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
# Rendering functions — delegated to plot_render.py
# (mirrors R's separation of plot-build.R / plot-render.R)
# ---------------------------------------------------------------------------

from ggplot2_py.plot_render import (  # noqa: E402
    ggplot_gtable,
    ggplotGrob,
    _safe_colour,
    _table_add_legends,
    _table_add_titles,
    find_panel,
    panel_rows,
    panel_cols,
    print_plot,
)

# ---------------------------------------------------------------------------

def ggplot_add(obj: Any, plot: GGPlot, object_name: str = "") -> GGPlot:
    """Add an object to a ggplot (generic dispatch).

    This is the Python equivalent of R's ``ggplot_add()`` S3 generic.
    It dispatches based on the type of *obj* via :func:`update_ggplot`.

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


# ---------------------------------------------------------------------------
# update_ggplot — singledispatch generic (R ref: plot-construction.R:133,
# ``update_ggplot <- S7::new_generic("update_ggplot", c("object","plot"))``).
#
# Extension packages can register new types via:
#
#     from ggplot2_py.plot import update_ggplot
#     @update_ggplot.register(MyType)
#     def _add_my_type(obj, plot, object_name=""):
#         ...
#         return plot
# ---------------------------------------------------------------------------


@singledispatch
def update_ggplot(obj: Any, plot: GGPlot, object_name: str = "") -> GGPlot:
    """Add *obj* to *plot*.  Open generic — register new types with
    ``@update_ggplot.register(YourType)``.

    Registrations that would require importing from other ggplot2_py modules
    at this module's import time (e.g. ``Guides`` from :mod:`ggplot2_py.guide`)
    are registered in the other module's own file to keep the dependency
    graph acyclic — see ``guide.py``'s tail for an example.
    """
    # --- GGProto (error) ---
    if is_ggproto(obj):
        cli_abort(
            "Cannot add ggproto objects together. "
            "Did you forget to add this object to a ggplot object?"
        )
    # --- Callable (error with hint) ---
    if callable(obj):
        name = object_name or getattr(obj, "__name__", "object")
        cli_abort(
            f"Cannot add `{name}` to a ggplot object. "
            f"Did you forget to add parentheses, as in `{name}()`?"
        )
    cli_abort(
        f"Cannot add `{object_name or type(obj).__name__}` to a ggplot object."
    )


@update_ggplot.register(type(None))
def _update_none(obj, plot, object_name=""):
    return plot


@update_ggplot.register(list)
def _update_list(obj, plot, object_name=""):
    for item in obj:
        plot = ggplot_add(item, plot, object_name)
    return plot


@update_ggplot.register(pd.DataFrame)
def _update_dataframe(obj, plot, object_name=""):
    plot.data = obj
    return plot


@update_ggplot.register(Mapping)
def _update_mapping(obj, plot, object_name=""):
    merged_mapping = aes(**{**plot.mapping, **obj})
    plot.mapping = merged_mapping
    return plot


@update_ggplot.register(Labels)
def _update_labels(obj, plot, object_name=""):
    merged = Labels(plot.labels)
    merged.update(obj)
    plot.labels = merged
    return plot


# Registrations for types from other modules are deferred to avoid
# circular imports.  They are registered via _register_update_ggplot_types()
# called at the bottom of this module (after the lazy imports block).

def _register_update_ggplot_types():
    """Register update_ggplot handlers for types that require lazy imports."""
    from ggplot2_py.layer import Layer
    from ggplot2_py.scale import Scale
    from ggplot2_py.coord import Coord
    from ggplot2_py.facet import Facet
    from ggplot2_py.theme import Theme, add_theme

    @update_ggplot.register(Layer)
    def _update_layer(obj, plot, object_name=""):
        plot.layers.append(obj)
        return plot

    @update_ggplot.register(Scale)
    def _update_scale(obj, plot, object_name=""):
        plot.scales.add(obj)
        return plot

    @update_ggplot.register(Coord)
    def _update_coord(obj, plot, object_name=""):
        if (
            not getattr(plot.coordinates, "default", True)
            and getattr(obj, "default", False)
        ):
            return plot
        if not getattr(plot.coordinates, "default", True):
            cli_inform(
                "Coordinate system already present. "
                "Adding new coordinate system, which will replace the existing one."
            )
        plot.coordinates = obj
        return plot

    @update_ggplot.register(Facet)
    def _update_facet(obj, plot, object_name=""):
        plot.facet = obj
        return plot

    @update_ggplot.register(Theme)
    def _update_theme(obj, plot, object_name=""):
        plot.theme = add_theme(plot.theme, obj)
        return plot


# Perform deferred registrations.
_register_update_ggplot_types()


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

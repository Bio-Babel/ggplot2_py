"""
Layer: the core data structure combining geom, stat, and position.

A layer holds a geom, stat, position, data, mapping, and associated
parameters. Layers are typically created via ``geom_*`` or ``stat_*``
calls, but can also be assembled directly through ``layer()``.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn
from ggplot2_py.ggproto import GGProto, ggproto
from ggplot2_py.aes import (
    Mapping,
    standardise_aes_names,
    AfterStat,
    AfterScale,
    Stage,
    is_mapping,
    rename_aes,
)
from ggplot2_py._utils import (
    remove_missing,
    snake_class,
    compact,
    modify_list,
    plyr_id,
    data_frame,
    empty,
)

__all__ = [
    "Layer",
    "layer",
    "layer_sf",
    "is_layer",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_subclass(
    x: Any,
    subclass: str,
    registry: Optional[Dict[str, type]] = None,
) -> Any:
    """Validate and resolve *x* to a ggproto instance of *subclass*.

    Parameters
    ----------
    x : str or GGProto
        Either a string name (e.g. ``"point"``) that will be looked up, or
        an existing ggproto object.
    subclass : str
        Expected base class name (``"Geom"``, ``"Stat"``, ``"Position"``).
    registry : dict, optional
        Name -> class mapping used for string lookup.  If *None*, the
        object must already be a ggproto instance.

    Returns
    -------
    GGProto
        The resolved object.

    Raises
    ------
    TypeError
        If *x* cannot be resolved.
    """
    if isinstance(x, GGProto) or (isinstance(x, type) and issubclass(x, GGProto)):
        return x

    if isinstance(x, str):
        if registry is not None and x in registry:
            return registry[x]
        # Try CamelCase class name lookup
        camel = subclass + x.capitalize()
        if registry is not None and camel in registry:
            return registry[camel]
        cli_abort(
            f"Cannot find {subclass.lower()} called {x!r}.",
        )

    cli_abort(
        f"Expected a string or {subclass} object, got {type(x).__name__}.",
    )


def _resolve_class(name: str, prefix: str) -> Any:
    """Resolve a string like ``"identity"`` to a ggproto class.

    Looks up ``{prefix}{Name}`` (e.g. ``StatIdentity``) in the appropriate
    module (``ggplot2_py.stat``, ``ggplot2_py.geom``, ``ggplot2_py.position``).
    """
    import importlib

    module_map = {"Stat": "ggplot2_py.stat", "Geom": "ggplot2_py.geom", "Position": "ggplot2_py.position"}
    mod = importlib.import_module(module_map[prefix])

    # Try e.g. StatIdentity, GeomPoint, PositionDodge
    class_name = prefix + name.replace("_", " ").title().replace(" ", "")
    cls = getattr(mod, class_name, None)
    if cls is not None:
        return cls

    # Fallback: try exact attribute name
    cls = getattr(mod, name, None)
    if cls is not None:
        return cls

    cli_abort(f"Cannot find {prefix.lower()} called {name!r}.")


def _split_params(
    params: Dict[str, Any],
    geom: Any,
    stat: Any,
    position: Any,
) -> tuple:
    """Split *params* into ``(geom_params, stat_params, aes_params)``.

    Parameters
    ----------
    params : dict
        Combined parameters passed to the layer.
    geom, stat, position
        The ggproto objects whose parameter/aesthetic names determine the
        split.

    Returns
    -------
    tuple of (dict, dict, dict)
        ``geom_params``, ``stat_params``, ``aes_params``.
    """
    # Helper: call method on class or instance (ggproto objects blur the two)
    def _try_call(obj: Any, method: str, *args: Any) -> Optional[set]:
        # If obj is a class, instantiate it first so instance methods work
        if isinstance(obj, type):
            try:
                obj = obj()
            except Exception:
                pass
        fn = getattr(obj, method, None)
        if fn is None or not callable(fn):
            return None
        try:
            return set(fn(*args))
        except Exception:
            return None

    geom_aesthetics = _try_call(geom, "aesthetics") or set()
    stat_aesthetics = _try_call(stat, "aesthetics") or set()
    position_aesthetics = set()
    if hasattr(position, "required_aes"):
        position_aesthetics = set(getattr(position, "required_aes", ()))

    all_aes = geom_aesthetics | stat_aesthetics | position_aesthetics

    geom_param_names = _try_call(geom, "parameters", True) or set()
    stat_param_names = _try_call(stat, "parameters", True) or set()

    params = dict(rename_aes(params)) if isinstance(params, dict) else dict(params)
    aes_params: Dict[str, Any] = {}
    geom_params: Dict[str, Any] = {}
    stat_params: Dict[str, Any] = {}

    for k, v in params.items():
        if k in all_aes:
            aes_params[k] = v
        elif k in geom_param_names:
            geom_params[k] = v
        elif k in stat_param_names:
            stat_params[k] = v
        else:
            # Unknown params go to geom_params by default
            geom_params[k] = v

    return geom_params, stat_params, aes_params


# ---------------------------------------------------------------------------
# Layer class
# ---------------------------------------------------------------------------

class Layer(GGProto):
    """The Layer ggproto class.

    A Layer holds the Geom, Stat and Position trifecta together with data,
    mapping, and parameter state.  It is responsible for managing data flow
    during ``ggplot_build`` and producing grobs during ``ggplot_gtable``.

    Attributes
    ----------
    geom : GGProto or None
        Geom ggproto object.
    stat : GGProto or None
        Stat ggproto object.
    position : GGProto or None
        Position ggproto object.
    data : pd.DataFrame, callable, Waiver, or None
        Layer data.
    mapping : Mapping or None
        Aesthetic mapping for this layer.
    computed_mapping : Mapping or None
        Final mapping (may include inherited plot mapping).
    aes_params : dict
        Fixed aesthetic parameters.
    geom_params : dict
        Parameters for the geom.
    stat_params : dict
        Parameters for the stat.
    computed_geom_params : dict or None
        Geom parameters after ``Geom.setup_params``.
    computed_stat_params : dict or None
        Stat parameters after ``Stat.setup_params``.
    inherit_aes : bool
        Whether to inherit the plot-level mapping.
    show_legend : bool or None
        Whether to include this layer in the legend.
    key_glyph : callable or None
        Custom legend key drawing function.
    name : str or None
        Optional layer name.
    layout : Any
        Layout specification for the layer.
    constructor : str or None
        Name of the user-facing constructor, for error messaging.
    """

    # Fields ----------------------------------------------------------------
    constructor: Optional[str] = None
    geom: Any = None
    stat: Any = None
    position: Any = None
    data: Any = None
    mapping: Optional[Mapping] = None
    computed_mapping: Optional[Mapping] = None
    aes_params: Dict[str, Any] = {}
    geom_params: Dict[str, Any] = {}
    stat_params: Dict[str, Any] = {}
    computed_geom_params: Optional[Dict[str, Any]] = None
    computed_stat_params: Optional[Dict[str, Any]] = None
    inherit_aes: bool = True
    show_legend: Optional[bool] = None
    key_glyph: Any = None
    name: Optional[str] = None
    layout: Any = None

    # Methods ---------------------------------------------------------------

    def layer_data(self, plot_data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Resolve layer data against the global plot data.

        Parameters
        ----------
        plot_data : pd.DataFrame or None
            The ``data`` field of the ggplot object.

        Returns
        -------
        pd.DataFrame or None
            Resolved data for this layer.
        """
        if is_waiver(self.data):
            data = plot_data
        elif callable(self.data):
            data = self.data(plot_data)
            if not isinstance(data, pd.DataFrame):
                cli_abort("layer_data() must return a DataFrame.")
        else:
            data = self.data

        if data is None or is_waiver(data):
            return data
        # Strip row names / reset index
        if isinstance(data, pd.DataFrame):
            data = data.reset_index(drop=True)
        return data

    def setup_layer(self, data: pd.DataFrame, plot: Any) -> pd.DataFrame:
        """Prepare layer data and finalise the mapping.

        Merges the layer mapping with the global plot mapping when
        ``inherit_aes`` is True and stores the result in
        ``computed_mapping``.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data.
        plot : object
            The ggplot object (provides ``mapping``).

        Returns
        -------
        pd.DataFrame
            Possibly-modified layer data.
        """
        if self.inherit_aes:
            plot_mapping = getattr(plot, "mapping", None) or {}
            if self.mapping is not None:
                # Layer mapping overrides plot mapping
                merged = dict(plot_mapping)
                merged.update(self.mapping)
                self.computed_mapping = Mapping(merged) if isinstance(merged, dict) else merged
            else:
                self.computed_mapping = (
                    Mapping(plot_mapping) if isinstance(plot_mapping, dict) else plot_mapping
                )
        else:
            self.computed_mapping = self.mapping
        return data

    def compute_aesthetics(self, data: pd.DataFrame, plot: Any) -> pd.DataFrame:
        """Evaluate aesthetic mappings against the data.

        Evaluates column references in the mapping, infers a ``group``
        aesthetic if absent, and sets the ``PANEL`` column.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data.
        plot : object
            The ggplot object.

        Returns
        -------
        pd.DataFrame
            Data with evaluated aesthetics.
        """
        aesthetics = self.computed_mapping or {}

        # Remove aesthetics that are set as fixed params
        set_aes = set(self.aes_params.keys()) if self.aes_params else set()
        aesthetics = {k: v for k, v in aesthetics.items() if k not in set_aes}

        # Remove computed / staged aesthetics (after_stat, after_scale)
        evaluated: Dict[str, Any] = {}
        for aes_name, aes_val in aesthetics.items():
            if isinstance(aes_val, (AfterStat, AfterScale, Stage)):
                continue
            if isinstance(aes_val, str) and aes_val in data.columns:
                evaluated[aes_name] = data[aes_val].values
            elif isinstance(aes_val, str):
                # May be a computed column not yet available; skip
                continue
            else:
                evaluated[aes_name] = aes_val

        n = len(data)
        if n == 0 and evaluated:
            lengths = [
                len(v) if hasattr(v, "__len__") and not isinstance(v, str) else 1
                for v in evaluated.values()
            ]
            n = max(lengths) if lengths else 0

        # Build result DataFrame
        result_dict: Dict[str, Any] = {}
        for k, v in evaluated.items():
            if np.isscalar(v) or isinstance(v, str):
                result_dict[k] = np.repeat(v, n)
            elif hasattr(v, "__len__") and len(v) == n:
                result_dict[k] = v
            elif hasattr(v, "__len__") and len(v) == 1:
                result_dict[k] = np.repeat(v[0] if hasattr(v, "__getitem__") else v, n)
            else:
                result_dict[k] = v

        # PANEL
        if empty(data) and n > 0:
            result_dict["PANEL"] = np.ones(n, dtype=int)
        elif "PANEL" in data.columns:
            result_dict["PANEL"] = data["PANEL"].values

        result = pd.DataFrame(result_dict)

        # Add group if missing
        if "group" not in result.columns:
            result = _add_group(result)

        return result

    def compute_statistic(
        self, data: pd.DataFrame, layout: Any
    ) -> pd.DataFrame:
        """Compute statistics for this layer.

        Delegates to ``Stat.setup_params``, ``Stat.setup_data``,
        and ``Stat.compute_layer``.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data.
        layout : object
            Layout ggproto object.

        Returns
        -------
        pd.DataFrame
            Data with computed stat columns.
        """
        if empty(data):
            return pd.DataFrame()

        stat = self.stat
        self.computed_stat_params = stat.setup_params(data, self.stat_params)
        data = stat.setup_data(data, self.computed_stat_params)
        data = stat.compute_layer(data, self.computed_stat_params, layout)
        return data

    def map_statistic(self, data: pd.DataFrame, plot: Any) -> pd.DataFrame:
        """Map computed-stat output aesthetics back to the data.

        Evaluates ``after_stat()`` mappings from both the layer and the
        stat default aesthetics.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data after ``compute_statistic``.
        plot : object
            The ggplot object.

        Returns
        -------
        pd.DataFrame
            Data with stat-mapped columns.
        """
        if empty(data):
            return pd.DataFrame()

        # Merge computed_mapping with stat defaults
        aesthetics = dict(self.computed_mapping or {})
        stat_defaults = getattr(self.stat, "default_aes", {}) or {}
        for k, v in stat_defaults.items():
            if k not in aesthetics:
                aesthetics[k] = v
        aesthetics = compact(aesthetics)

        # Evaluate AfterStat mappings
        new_cols: Dict[str, Any] = {}
        for aes_name, aes_val in aesthetics.items():
            if isinstance(aes_val, AfterStat):
                col = aes_val.x
                if col in data.columns:
                    new_cols[aes_name] = data[col].values
            elif isinstance(aes_val, Stage):
                col = getattr(aes_val, "start", None)
                if isinstance(col, AfterStat) and col.x in data.columns:
                    new_cols[aes_name] = data[col.x].values

        for k, v in new_cols.items():
            data[k] = v

        return data

    def compute_geom_1(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for drawing (geom setup).

        Checks required aesthetics and delegates to
        ``Geom.setup_params`` and ``Geom.setup_data``.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data.

        Returns
        -------
        pd.DataFrame
            Data after geom setup.
        """
        if empty(data):
            return pd.DataFrame()

        geom = self.geom
        # Check required aesthetics
        required = getattr(geom, "REQUIRED_AES", None) or getattr(geom, "required_aes", ())
        if required:
            present = set(data.columns) | set(self.aes_params.keys())
            for req in required:
                alternatives = req.split("|")
                if not any(a in present for a in alternatives):
                    cli_abort(
                        f"{snake_class(geom)} requires the following missing "
                        f"aesthetics: {req}"
                    )

        all_params = dict(self.geom_params)
        all_params.update(self.aes_params)
        self.computed_geom_params = geom.setup_params(data, all_params)
        data = geom.setup_data(data, self.computed_geom_params)
        return data

    def compute_position(
        self, data: pd.DataFrame, layout: Any
    ) -> pd.DataFrame:
        """Apply position adjustment.

        Delegates to ``Position.use_defaults``, ``Position.setup_params``,
        ``Position.setup_data``, and ``Position.compute_layer``.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data.
        layout : object
            Layout ggproto object.

        Returns
        -------
        pd.DataFrame
            Position-adjusted data.
        """
        if empty(data):
            return pd.DataFrame()

        pos = self.position
        if hasattr(pos, "use_defaults"):
            data = pos.use_defaults(data, self.aes_params)
        params = pos.setup_params(data)
        data = pos.setup_data(data, params)
        data = pos.compute_layer(data, params, layout)
        return data

    def compute_geom_2(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        theme: Any = None,
    ) -> pd.DataFrame:
        """Fill in default and fixed aesthetic values.

        Wraps ``Geom.use_defaults``.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data.
        params : dict, optional
            Fixed aesthetic params.  Defaults to ``self.aes_params``.
        theme : object, optional
            Theme object.

        Returns
        -------
        pd.DataFrame
            Data with defaults filled in.
        """
        if params is None:
            params = self.aes_params
        if empty(data):
            return data

        geom = self.geom
        if hasattr(geom, "use_defaults"):
            modifiers = {}
            if self.computed_mapping:
                modifiers = {
                    k: v
                    for k, v in self.computed_mapping.items()
                    if isinstance(v, (AfterScale, Stage))
                }
            data = geom.use_defaults(data, params, modifiers, theme=theme)
        return data

    def finish_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the stat finish hook.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data.

        Returns
        -------
        pd.DataFrame
        """
        if hasattr(self.stat, "finish_layer"):
            return self.stat.finish_layer(data, self.computed_stat_params)
        return data

    def draw_geom(self, data: pd.DataFrame, layout: Any) -> list:
        """Produce grobs for every panel.

        Delegates to ``Geom.handle_na`` and ``Geom.draw_layer``.

        Parameters
        ----------
        data : pd.DataFrame
            Layer data.
        layout : object
            Layout ggproto object.

        Returns
        -------
        list
            A list of grobs, one per panel.
        """
        if empty(data):
            n = len(getattr(layout, "layout", pd.DataFrame()))
            from grid_py import null_grob
            return [null_grob()] * max(n, 1)

        geom = self.geom
        if hasattr(geom, "handle_na"):
            data = geom.handle_na(data, self.computed_geom_params)
        coord = getattr(layout, "coord", None)
        return geom.draw_layer(data, self.computed_geom_params, layout, coord)

    def __repr__(self) -> str:
        parts = []
        if self.mapping is not None:
            parts.append(f"mapping: {self.mapping}")
        if self.geom is not None:
            parts.append(f"geom: {snake_class(self.geom)}")
        if self.stat is not None:
            parts.append(f"stat: {snake_class(self.stat)}")
        if self.position is not None:
            parts.append(f"position: {snake_class(self.position)}")
        return "<Layer " + ", ".join(parts) + ">"


# ---------------------------------------------------------------------------
# Group detection helper
# ---------------------------------------------------------------------------

def _add_group(data: pd.DataFrame) -> pd.DataFrame:
    """Infer a ``group`` column from discrete aesthetics.

    Parameters
    ----------
    data : pd.DataFrame
        Data that may or may not contain a group column.

    Returns
    -------
    pd.DataFrame
        Data with a ``group`` column.
    """
    if "group" in data.columns:
        return data

    # Identify discrete columns (object, category, bool)
    disc_cols = [
        c
        for c in data.columns
        if c != "PANEL"
        and (
            data[c].dtype == object
            or hasattr(data[c], "cat")
            or data[c].dtype == bool
        )
    ]
    if disc_cols:
        # Create interaction of all discrete columns
        if len(disc_cols) == 1:
            groups = pd.Categorical(data[disc_cols[0]]).codes
        else:
            interaction = data[disc_cols].apply(
                lambda row: "|".join(str(v) for v in row), axis=1
            )
            groups = pd.Categorical(interaction).codes
        data = data.copy()
        data["group"] = groups
    else:
        data = data.copy()
        data["group"] = -1  # single group sentinel
    return data


# ---------------------------------------------------------------------------
# layer() constructor
# ---------------------------------------------------------------------------

def layer(
    geom: Any = None,
    stat: Any = None,
    data: Any = None,
    mapping: Optional[Mapping] = None,
    position: Any = None,
    params: Optional[Dict[str, Any]] = None,
    inherit_aes: bool = True,
    check_aes: bool = True,
    check_param: bool = True,
    show_legend: Optional[bool] = None,
    key_glyph: Any = None,
    layout: Any = None,
    layer_class: Type[Layer] = Layer,
    **kwargs: Any,
) -> Layer:
    """Create a new layer.

    Parameters
    ----------
    geom : str or GGProto
        Geom specification.
    stat : str or GGProto
        Stat specification.
    data : DataFrame, callable, or None
        Layer data.
    mapping : Mapping or None
        Aesthetic mapping.
    position : str or GGProto
        Position adjustment specification.
    params : dict, optional
        Combined geom/stat/aes parameters.
    inherit_aes : bool
        Whether to inherit the plot-level mapping.
    check_aes : bool
        Whether to check aesthetic validity.
    check_param : bool
        Whether to check parameter validity.
    show_legend : bool or None
        Whether to include in the legend.
    key_glyph : callable or str or None
        Legend key drawing function.
    layout : Any
        Layout specification.
    layer_class : type
        Class to instantiate.  Defaults to :class:`Layer`.
    **kwargs
        Additional keyword arguments merged into *params*.

    Returns
    -------
    Layer
        A new Layer instance.
    """
    if params is None:
        params = {}
    params.update(kwargs)

    # Ensure na_rm default
    params.setdefault("na_rm", False)

    # Validate/resolve geom, stat, position
    if geom is None:
        geom = "blank"
    if stat is None:
        stat = "identity"
    if position is None:
        position = "identity"

    # Resolve dict-form position (e.g. {"name": "jitter", "width": 0.2})
    if isinstance(position, dict):
        pos_name = position.pop("name", "identity")
        pos_kwargs = position
        position = pos_name
    else:
        pos_kwargs = {}

    # Resolve string names to ggproto classes and ensure instances
    if isinstance(stat, str):
        stat = _resolve_class(stat, "Stat")
    if isinstance(position, str):
        position = _resolve_class(position, "Position")
    if isinstance(geom, str):
        geom = _resolve_class(geom, "Geom")

    # Ensure we have instances, not classes (methods need bound self)
    if isinstance(stat, type):
        stat = stat()
    if isinstance(position, type):
        pos_inst = position()
        # Apply any dict-form position kwargs
        for k, v in pos_kwargs.items():
            if v is not None:
                setattr(pos_inst, k, v)
        position = pos_inst
    if isinstance(geom, type):
        geom = geom()

    # Split params
    geom_params: Dict[str, Any]
    stat_params: Dict[str, Any]
    aes_params: Dict[str, Any]

    if isinstance(geom, GGProto) or (isinstance(geom, type) and issubclass(geom, GGProto)):
        geom_params, stat_params, aes_params = _split_params(
            params, geom, stat, position
        )
    else:
        # Deferred: put everything into geom_params for now
        geom_params = dict(params)
        stat_params = {}
        aes_params = {}

    # Instantiate
    obj = object.__new__(layer_class)
    # Copy class defaults
    obj.constructor = None
    obj.geom = geom
    obj.stat = stat
    obj.position = position
    obj.data = waiver() if data is None else data
    obj.mapping = mapping
    obj.computed_mapping = None
    obj.geom_params = geom_params
    obj.stat_params = stat_params
    obj.aes_params = aes_params
    obj.computed_geom_params = None
    obj.computed_stat_params = None
    obj.inherit_aes = inherit_aes
    obj.show_legend = show_legend
    obj.key_glyph = key_glyph
    obj.name = params.get("name")
    obj.layout = layout or params.get("layout")
    return obj


def layer_sf(
    geom: Any = None,
    stat: Any = None,
    data: Any = None,
    mapping: Optional[Mapping] = None,
    position: Any = None,
    params: Optional[Dict[str, Any]] = None,
    inherit_aes: bool = True,
    check_aes: bool = True,
    check_param: bool = True,
    show_legend: Optional[bool] = None,
    key_glyph: Any = None,
    layout: Any = None,
    **kwargs: Any,
) -> Layer:
    """Create a layer for sf (spatial) data.

    This is a thin wrapper around :func:`layer` intended for use with
    sf-type geometries.

    Parameters
    ----------
    See :func:`layer`.

    Returns
    -------
    Layer
    """
    return layer(
        geom=geom,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        params=params,
        inherit_aes=inherit_aes,
        check_aes=check_aes,
        check_param=check_param,
        show_legend=show_legend,
        key_glyph=key_glyph,
        layout=layout,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def is_layer(x: Any) -> bool:
    """Test whether *x* is a Layer.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(x, Layer)

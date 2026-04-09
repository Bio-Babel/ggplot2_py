"""
Geom classes and constructor functions for ggplot2_py.

This module contains the base ``Geom`` class (a ``GGProto`` subclass) and all
concrete geom implementations (``GeomPoint``, ``GeomPath``, ``GeomBar``, etc.)
together with their user-facing constructor functions (``geom_point``,
``geom_path``, ``geom_bar``, etc.).

Each ``Geom*`` class defines:

* ``required_aes`` -- tuple of required aesthetic names.
* ``non_missing_aes`` -- aesthetics whose ``NA`` values trigger row removal.
* ``optional_aes`` -- aesthetics accepted but not required.
* ``default_aes`` -- a :class:`Mapping` with default values.
* ``extra_params`` -- extra non-aesthetic parameter names.
* ``draw_key`` -- legend key drawing function.
* ``setup_params`` / ``setup_data`` -- data/parameter preprocessing.
* ``draw_panel`` or ``draw_group`` -- the actual grob-creation method.

Each ``geom_*()`` function is a thin wrapper that calls ``layer()``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py.ggproto import GGProto, ggproto, ggproto_parent
from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn
from ggplot2_py._utils import (
    remove_missing,
    resolution,
    snake_class,
    compact,
    data_frame,
    empty,
)
from ggplot2_py.aes import aes, Mapping, standardise_aes_names

# Import draw_key functions
from ggplot2_py.draw_key import (
    draw_key_point,
    draw_key_path,
    draw_key_rect,
    draw_key_polygon,
    draw_key_blank,
    draw_key_boxplot,
    draw_key_crossbar,
    draw_key_dotplot,
    draw_key_label,
    draw_key_linerange,
    draw_key_pointrange,
    draw_key_smooth,
    draw_key_text,
    draw_key_abline,
    draw_key_vline,
    draw_key_timeseries,
    draw_key_vpath,
)

# grid_py grob creation imports
from grid_py import (
    points_grob,
    rect_grob,
    lines_grob,
    segments_grob,
    polygon_grob,
    polyline_grob,
    text_grob,
    circle_grob,
    raster_grob,
    path_grob,
    curve_grob,
    null_grob,
    Gpar,
    Unit,
    grob_tree,
    GTree,
    GList,
    clip_grob,
    Viewport,
    edit_grob,
    roundrect_grob,
)

from scales import alpha as _scales_alpha_raw

import re as _re

def _r_col_to_mpl(c):
    """Convert R-style grey names to RGB tuples for matplotlib."""
    if isinstance(c, str):
        m = _re.match(r'^gr[ae]y(\d{1,3})$', c)
        if m:
            v = int(m.group(1)) / 100.0
            return f"#{int(v*255):02x}{int(v*255):02x}{int(v*255):02x}"
    return c

def scales_alpha(colour, alpha):
    """Apply alpha to colours, converting R colour names first."""
    if isinstance(colour, (list, np.ndarray)):
        colour = [_r_col_to_mpl(c) for c in colour]
    elif isinstance(colour, str):
        colour = _r_col_to_mpl(colour)
    return _scales_alpha_raw(colour, alpha)

__all__ = [
    # Base class
    "Geom",
    # ggproto classes
    "GeomPoint", "GeomPath", "GeomLine", "GeomStep",
    "GeomBar", "GeomCol", "GeomRect", "GeomTile", "GeomRaster",
    "GeomText", "GeomLabel",
    "GeomBoxplot", "GeomViolin", "GeomDotplot",
    "GeomRibbon", "GeomArea", "GeomSmooth",
    "GeomPolygon",
    "GeomErrorbar", "GeomErrorbarh", "GeomCrossbar", "GeomLinerange", "GeomPointrange",
    "GeomSegment", "GeomCurve", "GeomSpoke",
    "GeomDensity", "GeomDensity2d", "GeomDensity2dFilled",
    "GeomContour", "GeomContourFilled",
    "GeomHex", "GeomBin2d",
    "GeomAbline", "GeomHline", "GeomVline",
    "GeomRug",
    "GeomBlank",
    "GeomFunction",
    "GeomFreqpoly", "GeomHistogram",
    "GeomCount",
    "GeomMap",
    "GeomQuantile",
    "GeomJitter",
    "GeomSf", "GeomAnnotationMap", "GeomCustomAnn", "GeomRasterAnn", "GeomLogticks",
    # Constructor functions
    "geom_point", "geom_path", "geom_line", "geom_step",
    "geom_bar", "geom_col", "geom_rect", "geom_tile", "geom_raster",
    "geom_text", "geom_label",
    "geom_boxplot", "geom_violin", "geom_dotplot",
    "geom_ribbon", "geom_area", "geom_smooth",
    "geom_polygon",
    "geom_errorbar", "geom_errorbarh", "geom_crossbar", "geom_linerange", "geom_pointrange",
    "geom_segment", "geom_curve", "geom_spoke",
    "geom_density", "geom_density2d", "geom_density2d_filled",
    "geom_density_2d", "geom_density_2d_filled",
    "geom_contour", "geom_contour_filled",
    "geom_hex", "geom_bin2d", "geom_bin_2d",
    "geom_abline", "geom_hline", "geom_vline",
    "geom_rug",
    "geom_blank",
    "geom_function",
    "geom_freqpoly", "geom_histogram",
    "geom_count",
    "geom_map",
    "geom_quantile",
    "geom_jitter",
    "geom_sf", "geom_sf_label", "geom_sf_text",
    "geom_qq", "geom_qq_line",
    # Utility
    "is_geom",
    "translate_shape_string",
]


# ===========================================================================
# Graphical-unit constants
# ===========================================================================

#: Points per mm  (``72.27 / 25.4``)
PT: float = 72.27 / 25.4
#: Stroke scale factor  (``96 / 25.4``)
STROKE: float = 96 / 25.4


# ===========================================================================
# Utilities
# ===========================================================================

def _fill_alpha(fill: Any, alpha_val: Any) -> Any:
    """Apply alpha to a fill colour, passing through ``None``."""
    if fill is None:
        return None
    try:
        return scales_alpha(fill, alpha_val)
    except Exception:
        return fill


def _gg_par(**kwargs: Any) -> Gpar:
    """Build a :class:`Gpar` filtering out ``None`` entries and converting
    ``linewidth`` (mm) to ``lwd`` (pts) when needed."""
    # Convert lwd from mm to pts
    if "lwd" in kwargs and kwargs["lwd"] is not None:
        try:
            kwargs["lwd"] = np.asarray(kwargs["lwd"], dtype=float) * PT
        except (TypeError, ValueError):
            pass
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    return Gpar(**filtered)


def _ggname(prefix: str, grob: Any) -> Any:
    """Attach a name prefix to a grob (used for identification in grob trees)."""
    try:
        grob.name = prefix
    except AttributeError:
        pass
    return grob


# ---------------------------------------------------------------------------
# Shape translation
# ---------------------------------------------------------------------------

_PCH_TABLE: Dict[str, int] = {
    "square open": 0,
    "circle open": 1,
    "triangle open": 2,
    "plus": 3,
    "cross": 4,
    "diamond open": 5,
    "triangle down open": 6,
    "square cross": 7,
    "asterisk": 8,
    "diamond plus": 9,
    "circle plus": 10,
    "star": 11,
    "square plus": 12,
    "circle cross": 13,
    "square triangle": 14,
    "triangle square": 14,
    "square": 15,
    "circle small": 16,
    "triangle": 17,
    "diamond": 18,
    "circle": 19,
    "bullet": 20,
    "circle filled": 21,
    "square filled": 22,
    "diamond filled": 23,
    "triangle filled": 24,
    "triangle down filled": 25,
}


def translate_shape_string(shape: Any) -> Any:
    """Translate point shape names to integer pch codes.

    Parameters
    ----------
    shape : str or array-like of str, or numeric
        Shape specification.  If numeric or single-character strings,
        returned as-is.

    Returns
    -------
    int or array-like
        Integer pch values.
    """
    if shape is None:
        return 19  # default circle
    if isinstance(shape, (int, float, np.integer, np.floating)):
        return int(shape)
    if isinstance(shape, str):
        if len(shape) <= 1:
            return shape
        lower = shape.lower()
        for name, code in _PCH_TABLE.items():
            if name.startswith(lower):
                return code
        cli_abort(f"Shape aesthetic contains invalid value: {shape!r}.")
    # array-like
    if hasattr(shape, "__iter__"):
        return np.array([translate_shape_string(s) for s in shape])
    return shape


def is_geom(x: Any) -> bool:
    """Return ``True`` if *x* is a ``Geom`` subclass or instance."""
    if isinstance(x, type):
        return issubclass(x, Geom)
    return isinstance(x, Geom)


# ===========================================================================
# Base Geom class
# ===========================================================================

class Geom(GGProto):
    """Base class for all geometry objects.

    Subclasses must override at least ``draw_panel`` or ``draw_group``.

    Attributes
    ----------
    required_aes : tuple of str
        Aesthetics that *must* be present.
    non_missing_aes : tuple of str
        Aesthetics that trigger row-removal if ``NA``.
    optional_aes : tuple of str
        Extra accepted aesthetics.
    default_aes : Mapping
        Default aesthetic values.
    extra_params : tuple of str
        Extra non-aesthetic parameters (e.g. ``"na_rm"``).
    draw_key : callable
        Legend key drawing function.
    rename_size : bool
        Whether to rename ``size`` to ``linewidth``.
    """

    required_aes: Tuple[str, ...] = ()
    non_missing_aes: Tuple[str, ...] = ()
    optional_aes: Tuple[str, ...] = ()
    default_aes: Mapping = Mapping()
    extra_params: Tuple[str, ...] = ("na_rm",)
    draw_key = draw_key_point
    rename_size: bool = False

    # -----------------------------------------------------------------------
    # Setup hooks (run before position adjustments)
    # -----------------------------------------------------------------------

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify or validate parameters given the data.

        Parameters
        ----------
        data : DataFrame
            Layer data.
        params : dict
            Current parameters.

        Returns
        -------
        dict
            Possibly modified parameters.
        """
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Modify or validate data before defaults are applied.

        Parameters
        ----------
        data : DataFrame
            Layer data.
        params : dict
            Parameters from ``setup_params``.

        Returns
        -------
        DataFrame
        """
        return data

    # -----------------------------------------------------------------------
    # Missing-value handling
    # -----------------------------------------------------------------------

    def handle_na(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Remove rows with missing values in required aesthetics.

        Parameters
        ----------
        data : DataFrame
        params : dict

        Returns
        -------
        DataFrame
        """
        na_rm = params.get("na_rm", params.get("na.rm", False))
        check_vars = list(self.required_aes) + list(self.non_missing_aes)
        return remove_missing(data, vars=check_vars, na_rm=na_rm, name=snake_class(self))

    # -----------------------------------------------------------------------
    # use_defaults -- fill in default aesthetics
    # -----------------------------------------------------------------------

    def use_defaults(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        modifiers: Optional[Mapping] = None,
        default_aes: Optional[Mapping] = None,
        theme: Any = None,
    ) -> pd.DataFrame:
        """Fill missing aesthetics with defaults and apply parameter overrides.

        Parameters
        ----------
        data : DataFrame
        params : dict, optional
        modifiers : Mapping, optional
        default_aes : Mapping, optional
        theme : optional

        Returns
        -------
        DataFrame
        """
        if params is None:
            params = {}
        if modifiers is None:
            modifiers = Mapping()
        if default_aes is None:
            default_aes = self.default_aes

        # Inherit size as linewidth when applicable
        if self.rename_size:
            if data is not None and "linewidth" not in data.columns and "size" in data.columns:
                data = data.copy()
                data["linewidth"] = data["size"]
            if "linewidth" not in params and "size" in params:
                params["linewidth"] = params["size"]

        # Fill in missing aesthetics with their defaults
        if data is not None and not data.empty:
            for aes_name, default_val in default_aes.items():
                if aes_name not in data.columns:
                    data[aes_name] = default_val

        # Override with params
        aes_params = set(self.aesthetics()) & set(params.keys())
        if data is not None and not data.empty:
            for ap in aes_params:
                data[ap] = params[ap]

        return data

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def draw_layer(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        layout: Any,
        coord: Any,
    ) -> List[Any]:
        """Orchestrate drawing for all panels.

        Parameters
        ----------
        data : DataFrame
        params : dict
        layout : Layout
        coord : Coord

        Returns
        -------
        list of grobs
        """
        if data is None or (hasattr(data, "empty") and data.empty):
            return [null_grob()]

        # Split by PANEL
        if "PANEL" in data.columns:
            panels = {k: v for k, v in data.groupby("PANEL")}
        else:
            panels = {1: data}

        grobs = []
        for panel_id, panel_data in panels.items():
            if panel_data.empty:
                grobs.append(null_grob())
                continue
            # PANEL is 1-based, panel_params list is 0-based
            idx = int(panel_id) - 1 if isinstance(panel_id, (int, np.integer)) else panel_id
            panel_params = layout.panel_params[idx]
            grobs.append(self.draw_panel(panel_data, panel_params, coord, **params))
        return grobs

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        **params: Any,
    ) -> Any:
        """Draw the geom for a single panel.

        The default implementation splits on ``group`` and delegates to
        ``draw_group``.

        Parameters
        ----------
        data : DataFrame
        panel_params : panel parameters
        coord : Coord

        Returns
        -------
        grob
        """
        if "group" not in data.columns:
            return self.draw_group(data, panel_params, coord, **params)

        groups = {k: v for k, v in data.groupby("group")}
        grobs = []
        for _, group_data in groups.items():
            grobs.append(self.draw_group(group_data, panel_params, coord, **params))

        return _ggname(
            snake_class(self),
            grob_tree(*grobs) if grobs else null_grob(),
        )

    def draw_group(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        **params: Any,
    ) -> Any:
        """Draw the geom for a single group.

        Must be overridden by subclasses that need per-group drawing.

        Parameters
        ----------
        data : DataFrame
        panel_params : panel parameters
        coord : Coord

        Returns
        -------
        grob
        """
        cli_abort(f"{snake_class(self)} has not implemented a draw_group method")

    # -----------------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------------

    def parameters(self, extra: bool = False) -> List[str]:
        """List acceptable parameters for this geom.

        Parameters
        ----------
        extra : bool
            Whether to include ``extra_params``.

        Returns
        -------
        list of str
        """
        import inspect
        sig = inspect.signature(self.draw_panel)
        args = [p for p in sig.parameters if p not in ("self", "data", "panel_params", "coord")]
        if extra:
            args = list(set(args) | set(self.extra_params))
        return args

    def aesthetics(self) -> List[str]:
        """List all accepted aesthetics.

        Returns
        -------
        list of str
        """
        required = []
        for aes_name in self.required_aes:
            required.extend(aes_name.split("|"))

        aes_names = list(dict.fromkeys(required + list(self.default_aes.keys())))
        aes_names.extend(a for a in self.optional_aes if a not in aes_names)
        if "group" not in aes_names:
            aes_names.append("group")
        return aes_names


# ===========================================================================
# Helper: _coord_transform
# ===========================================================================

def _coord_transform(coord: Any, data: pd.DataFrame, panel_params: Any) -> pd.DataFrame:
    """Safely apply coordinate transformation."""
    if coord is not None and hasattr(coord, "transform"):
        return coord.transform(data, panel_params)
    return data


# ===========================================================================
# GeomPoint
# ===========================================================================

class GeomPoint(Geom):
    """Point geom (scatterplot)."""

    required_aes: Tuple[str, ...] = ("x", "y")
    non_missing_aes: Tuple[str, ...] = ("size", "shape", "colour")
    default_aes: Mapping = Mapping(
        shape=19,
        colour="black",
        fill=None,
        size=1.5,
        alpha=None,
        stroke=0.5,
    )
    draw_key = draw_key_point

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        """Draw points.

        Parameters
        ----------
        data : DataFrame
        panel_params : panel parameters
        coord : Coord
        na_rm : bool

        Returns
        -------
        grob
        """
        data = data.copy()
        if "shape" in data.columns:
            data["shape"] = data["shape"].apply(translate_shape_string)
        coords = _coord_transform(coord, data, panel_params)

        return _ggname(
            "geom_point",
            points_grob(
                x=coords["x"].values,
                y=coords["y"].values,
                pch=coords["shape"].values if "shape" in coords.columns else 19,
                gp=Gpar(
                    col=scales_alpha(
                        coords["colour"].values if "colour" in coords.columns else "black",
                        coords["alpha"].values if "alpha" in coords.columns else None,
                    ),
                    fill=_fill_alpha(
                        coords["fill"].values if "fill" in coords.columns else None,
                        coords["alpha"].values if "alpha" in coords.columns else None,
                    ),
                    fontsize=(
                        coords["size"].values * PT + coords["stroke"].values * STROKE
                        if "size" in coords.columns and "stroke" in coords.columns
                        else 1.5 * PT + 0.5 * STROKE
                    ),
                    lwd=(
                        coords["stroke"].values * STROKE
                        if "stroke" in coords.columns
                        else 0.5 * STROKE
                    ),
                ),
            ),
        )


# ===========================================================================
# GeomPath / GeomLine / GeomStep
# ===========================================================================

class GeomPath(Geom):
    """Path geom -- connects observations in data order."""

    required_aes: Tuple[str, ...] = ("x", "y")
    non_missing_aes: Tuple[str, ...] = ("linewidth", "colour", "linetype")
    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_path
    rename_size: bool = True

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        arrow: Any = None,
        lineend: str = "butt",
        linejoin: str = "round",
        linemitre: float = 10,
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        """Draw connected paths.

        Parameters
        ----------
        data : DataFrame
        panel_params : panel parameters
        coord : Coord

        Returns
        -------
        grob
        """
        data = data.copy()
        if "group" in data.columns:
            data = data.sort_values("group")

        coords = _coord_transform(coord, data, panel_params)

        if coords.empty or len(coords) < 2:
            return null_grob()

        group_id = None
        if "group" in coords.columns:
            group_id = coords["group"].values

        return polyline_grob(
            x=coords["x"].values,
            y=coords["y"].values,
            id=group_id,
            default_units="native",
            gp=Gpar(
                col=scales_alpha(
                    coords["colour"].values if "colour" in coords.columns else "black",
                    coords["alpha"].values if "alpha" in coords.columns else None,
                ),
                lwd=(
                    coords["linewidth"].values * PT
                    if "linewidth" in coords.columns
                    else 0.5 * PT
                ),
                lty=coords["linetype"].values if "linetype" in coords.columns else 1,
                lineend=lineend,
                linejoin=linejoin,
                linemitre=linemitre,
            ),
            arrow=arrow,
        )


class GeomLine(GeomPath):
    """Line geom -- like path but sorted by x."""

    extra_params: Tuple[str, ...] = ("na_rm", "orientation")

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        flipped = params.get("flipped_aes", False)
        sort_col = "y" if flipped else "x"
        group_cols = ["PANEL", "group"] if "group" in data.columns else ["PANEL"]
        group_cols = [c for c in group_cols if c in data.columns]
        if sort_col in data.columns:
            data = data.sort_values(group_cols + [sort_col])
        return data


class GeomStep(GeomPath):
    """Step geom -- stairstep connections."""

    extra_params: Tuple[str, ...] = ("na_rm", "orientation")

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        direction: str = "hv",
        lineend: str = "butt",
        linejoin: str = "round",
        linemitre: float = 10,
        arrow: Any = None,
        flipped_aes: bool = False,
        **params: Any,
    ) -> Any:
        """Draw step connections."""
        data = data.copy()
        data = _stairstep(data, direction=direction)
        return GeomPath.draw_panel(
            self, data, panel_params, coord,
            lineend=lineend, linejoin=linejoin, linemitre=linemitre,
            arrow=arrow,
        )


def _stairstep(data: pd.DataFrame, direction: str = "hv") -> pd.DataFrame:
    """Calculate stairstep coordinates for :class:`GeomStep`."""
    if direction not in ("hv", "vh", "mid"):
        cli_abort(f"direction must be 'hv', 'vh', or 'mid', not {direction!r}")
    data = data.sort_values("x").reset_index(drop=True)
    n = len(data)
    if n <= 1:
        return data.iloc[:0]

    x = data["x"].values
    y = data["y"].values

    if direction == "hv":
        xs = np.repeat(x, 2)[1:]
        ys = np.repeat(y, 2)[:-1]
    elif direction == "vh":
        xs = np.repeat(x, 2)[:-1]
        ys = np.repeat(y, 2)[1:]
    else:  # mid
        gaps = np.diff(x)
        mid_x = x[:-1] + gaps / 2
        xs_idx = np.repeat(np.arange(n - 1), 2)
        ys_idx = np.repeat(np.arange(n), 2)
        xs_arr = np.concatenate([[x[0]], mid_x[xs_idx], [x[-1]]])
        ys_arr = y[ys_idx]
        result = data.iloc[[0]].copy()
        result = pd.DataFrame({"x": xs_arr, "y": ys_arr})
        # carry forward other columns
        for col in data.columns:
            if col not in ("x", "y"):
                result[col] = data[col].iloc[0]
        return result

    result = pd.DataFrame({"x": xs, "y": ys})
    for col in data.columns:
        if col not in ("x", "y"):
            result[col] = data[col].iloc[0]
    return result


# ===========================================================================
# GeomRect / GeomTile / GeomRaster
# ===========================================================================

class GeomRect(Geom):
    """Rectangle geom (defined by xmin, xmax, ymin, ymax)."""

    required_aes: Tuple[str, ...] = ("xmin", "xmax", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        colour=None,
        fill="grey35",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_polygon
    rename_size: bool = True

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Resolve rect aesthetics from center + size if corners are missing."""
        if all(c in data.columns for c in ("xmin", "xmax", "ymin", "ymax")):
            return data
        data = data.copy()
        # Resolve x-dimension
        if "xmin" not in data.columns or "xmax" not in data.columns:
            if "x" in data.columns and "width" in data.columns:
                data["xmin"] = data["x"] - data["width"] / 2
                data["xmax"] = data["x"] + data["width"] / 2
            elif "x" in data.columns:
                w = params.get("width", 0.9)
                data["xmin"] = data["x"] - w / 2
                data["xmax"] = data["x"] + w / 2
        # Resolve y-dimension
        if "ymin" not in data.columns or "ymax" not in data.columns:
            if "y" in data.columns and "height" in data.columns:
                data["ymin"] = data["y"] - data["height"] / 2
                data["ymax"] = data["y"] + data["height"] / 2
            elif "y" in data.columns:
                h = params.get("height", 0.9)
                data["ymin"] = data["y"] - h / 2
                data["ymax"] = data["y"] + h / 2
        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        linejoin: str = "mitre",
        **params: Any,
    ) -> Any:
        """Draw rectangles."""
        coords = _coord_transform(coord, data, panel_params)

        return _ggname(
            "geom_rect",
            rect_grob(
                x=coords["xmin"].values,
                y=coords["ymax"].values,
                width=coords["xmax"].values - coords["xmin"].values,
                height=coords["ymax"].values - coords["ymin"].values,
                default_units="native",
                just=("left", "top"),
                gp=Gpar(
                    col=coords["colour"].values if "colour" in coords.columns else None,
                    fill=_fill_alpha(
                        coords["fill"].values if "fill" in coords.columns else "grey35",
                        coords["alpha"].values if "alpha" in coords.columns else None,
                    ),
                    lwd=(
                        coords["linewidth"].values * PT
                        if "linewidth" in coords.columns
                        else 0.5 * PT
                    ),
                    lty=coords["linetype"].values if "linetype" in coords.columns else 1,
                    linejoin=linejoin,
                    lineend=lineend,
                ),
            ),
        )


class GeomTile(GeomRect):
    """Tile geom -- rectangles parameterised by center and size."""

    required_aes: Tuple[str, ...] = ("x", "y")
    non_missing_aes: Tuple[str, ...] = ("xmin", "xmax", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        fill="grey35",
        colour=None,
        linewidth=0.1,
        linetype=1,
        alpha=None,
        width=1,
        height=1,
    )
    draw_key = draw_key_polygon
    extra_params: Tuple[str, ...] = ("na_rm",)

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        w = data["width"].values if "width" in data.columns else params.get("width", 1)
        h = data["height"].values if "height" in data.columns else params.get("height", 1)
        data["xmin"] = data["x"] - np.asarray(w) / 2
        data["xmax"] = data["x"] + np.asarray(w) / 2
        data["ymin"] = data["y"] - np.asarray(h) / 2
        data["ymax"] = data["y"] + np.asarray(h) / 2
        return data


class GeomRaster(Geom):
    """Raster geom -- high-performance uniform tiles."""

    required_aes: Tuple[str, ...] = ("x", "y")
    non_missing_aes: Tuple[str, ...] = ("fill",)
    default_aes: Mapping = Mapping(fill="grey35", alpha=None)
    draw_key = draw_key_polygon

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        hjust = params.get("hjust", 0.5)
        vjust = params.get("vjust", 0.5)
        data = data.copy()

        x_vals = data["x"].values.astype(float)
        y_vals = data["y"].values.astype(float)
        x_diff = np.diff(np.sort(np.unique(x_vals)))
        y_diff = np.diff(np.sort(np.unique(y_vals)))
        w = x_diff[0] if len(x_diff) > 0 else 1
        h = y_diff[0] if len(y_diff) > 0 else 1

        data["xmin"] = data["x"] - w * (1 - hjust)
        data["xmax"] = data["x"] + w * hjust
        data["ymin"] = data["y"] - h * (1 - vjust)
        data["ymax"] = data["y"] + h * vjust
        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        interpolate: bool = False,
        hjust: float = 0.5,
        vjust: float = 0.5,
        **params: Any,
    ) -> Any:
        """Draw raster tiles."""
        coords = _coord_transform(coord, data, panel_params)

        x_rng = (coords["xmin"].min(), coords["xmax"].max())
        y_rng = (coords["ymin"].min(), coords["ymax"].max())

        return raster_grob(
            image=_fill_alpha(
                coords["fill"].values if "fill" in coords.columns else "grey35",
                coords["alpha"].values if "alpha" in coords.columns else None,
            ),
            x=np.mean(x_rng),
            y=np.mean(y_rng),
            width=x_rng[1] - x_rng[0],
            height=y_rng[1] - y_rng[0],
            default_units="native",
            interpolate=interpolate,
        )


# ===========================================================================
# GeomBar / GeomCol
# ===========================================================================

class GeomBar(GeomRect):
    """Bar geom -- rectangles with y anchored at zero."""

    required_aes: Tuple[str, ...] = ("x", "y")
    non_missing_aes: Tuple[str, ...] = ("xmin", "xmax", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        colour=None,
        fill="grey35",
        linewidth=0.5,
        linetype=1,
        alpha=None,
        width=0.9,
    )
    extra_params: Tuple[str, ...] = ("just", "na_rm", "orientation")
    rename_size: bool = False

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        width = params.get("width") or (data["width"].values if "width" in data.columns else 0.9)
        just = params.get("just", 0.5)

        if isinstance(width, (int, float)):
            data["width"] = width
        data["ymin"] = np.minimum(data["y"].values, 0)
        data["ymax"] = np.maximum(data["y"].values, 0)
        data["xmin"] = data["x"] - data["width"] * just
        data["xmax"] = data["x"] + data["width"] * (1 - just)
        return data


class GeomCol(GeomBar):
    """Column geom -- alias for GeomBar."""
    pass


# ===========================================================================
# GeomText / GeomLabel
# ===========================================================================

class GeomText(Geom):
    """Text geom."""

    required_aes: Tuple[str, ...] = ("x", "y", "label")
    non_missing_aes: Tuple[str, ...] = ("angle",)
    default_aes: Mapping = Mapping(
        colour="black",
        family="",
        size=3.88,
        angle=0,
        hjust=0.5,
        vjust=0.5,
        alpha=None,
        fontface=1,
        lineheight=1.2,
    )
    draw_key = draw_key_text

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        parse: bool = False,
        na_rm: bool = False,
        check_overlap: bool = False,
        size_unit: str = "mm",
        **params: Any,
    ) -> Any:
        """Draw text labels."""
        coords = _coord_transform(coord, data, panel_params)

        size_mul = PT  # default mm
        if size_unit == "pt":
            size_mul = 1
        elif size_unit == "cm":
            size_mul = PT * 10
        elif size_unit == "in":
            size_mul = 72.27
        elif size_unit == "pc":
            size_mul = 12

        return text_grob(
            label=coords["label"].values,
            x=coords["x"].values,
            y=coords["y"].values,
            default_units="native",
            hjust=coords["hjust"].values if "hjust" in coords.columns else 0.5,
            vjust=coords["vjust"].values if "vjust" in coords.columns else 0.5,
            rot=coords["angle"].values if "angle" in coords.columns else 0,
            gp=Gpar(
                col=scales_alpha(
                    coords["colour"].values if "colour" in coords.columns else "black",
                    coords["alpha"].values if "alpha" in coords.columns else None,
                ),
                fontsize=(
                    coords["size"].values * size_mul
                    if "size" in coords.columns
                    else 3.88 * size_mul
                ),
                fontfamily=coords["family"].values if "family" in coords.columns else "",
                fontface=coords["fontface"].values if "fontface" in coords.columns else 1,
                lineheight=coords["lineheight"].values if "lineheight" in coords.columns else 1.2,
            ),
            check_overlap=check_overlap,
        )


class GeomLabel(Geom):
    """Label geom -- text with background rectangle."""

    required_aes: Tuple[str, ...] = ("x", "y", "label")
    default_aes: Mapping = Mapping(
        colour="black",
        fill="white",
        family="",
        size=3.88,
        angle=0,
        hjust=0.5,
        vjust=0.5,
        alpha=None,
        fontface=1,
        lineheight=1.2,
        linewidth=0.25,
        linetype=1,
    )
    draw_key = draw_key_label

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        parse: bool = False,
        na_rm: bool = False,
        label_padding: Any = None,
        label_r: Any = None,
        size_unit: str = "mm",
        **params: Any,
    ) -> Any:
        """Draw labelled text."""
        coords = _coord_transform(coord, data, panel_params)
        size_mul = PT

        grobs = []
        for i in range(len(coords)):
            row = coords.iloc[i]
            label = str(row["label"])
            x_val = row["x"]
            y_val = row["y"]

            bg_grob = roundrect_grob(
                x=x_val,
                y=y_val,
                gp=Gpar(
                    col=row.get("colour", "black"),
                    fill=_fill_alpha(row.get("fill", "white"), row.get("alpha")),
                    lwd=row.get("linewidth", 0.25) * PT,
                    lty=row.get("linetype", 1),
                ),
            )
            txt_grob = text_grob(
                label=label,
                x=x_val,
                y=y_val,
                gp=Gpar(
                    col=scales_alpha(row.get("colour", "black"), row.get("alpha")),
                    fontsize=row.get("size", 3.88) * size_mul,
                    fontfamily=row.get("family", ""),
                    fontface=row.get("fontface", 1),
                ),
            )
            grobs.extend([bg_grob, txt_grob])

        return _ggname("geom_label", grob_tree(*grobs) if grobs else null_grob())


# ===========================================================================
# GeomPolygon
# ===========================================================================

class GeomPolygon(Geom):
    """Polygon geom."""

    required_aes: Tuple[str, ...] = ("x", "y")
    default_aes: Mapping = Mapping(
        colour=None,
        fill="grey35",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_polygon
    rename_size: bool = True

    def handle_na(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        rule: str = "evenodd",
        lineend: str = "butt",
        linejoin: str = "round",
        linemitre: float = 10,
        **params: Any,
    ) -> Any:
        """Draw filled polygons."""
        if len(data) <= 1:
            return null_grob()

        coords = _coord_transform(coord, data, panel_params)
        if "group" in coords.columns:
            coords = coords.sort_values("group")

        group_id = coords["group"].values if "group" in coords.columns else None

        # Take first value per group for gpar
        return _ggname(
            "geom_polygon",
            polygon_grob(
                x=coords["x"].values,
                y=coords["y"].values,
                id=group_id,
                default_units="native",
                gp=Gpar(
                    col=coords["colour"].iloc[0] if "colour" in coords.columns else None,
                    fill=_fill_alpha(
                        coords["fill"].iloc[0] if "fill" in coords.columns else "grey35",
                        coords["alpha"].iloc[0] if "alpha" in coords.columns else None,
                    ),
                    lwd=(
                        coords["linewidth"].iloc[0] * PT
                        if "linewidth" in coords.columns
                        else 0.5 * PT
                    ),
                    lty=coords["linetype"].iloc[0] if "linetype" in coords.columns else 1,
                    lineend=lineend,
                    linejoin=linejoin,
                    linemitre=linemitre,
                ),
            ),
        )


# ===========================================================================
# GeomRibbon / GeomArea
# ===========================================================================

class GeomRibbon(Geom):
    """Ribbon geom -- shaded region between ymin and ymax."""

    required_aes: Tuple[str, ...] = ("x", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        colour=None,
        fill="grey35",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    extra_params: Tuple[str, ...] = ("na_rm", "orientation")
    draw_key = draw_key_polygon
    rename_size: bool = True

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        sort_cols = ["PANEL", "group", "x"]
        sort_cols = [c for c in sort_cols if c in data.columns]
        if sort_cols:
            data = data.sort_values(sort_cols)
        if "y" not in data.columns:
            data["y"] = data.get("ymin", data.get("ymax", 0))
        return data

    def draw_group(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        linejoin: str = "round",
        linemitre: float = 10,
        na_rm: bool = False,
        flipped_aes: bool = False,
        outline_type: str = "both",
        **params: Any,
    ) -> Any:
        """Draw ribbon."""
        data = data.copy()
        if "group" in data.columns:
            data = data.sort_values("group")

        # Build polygon from upper + reversed lower
        upper = pd.DataFrame({"x": data["x"].values, "y": data["ymax"].values})
        lower = pd.DataFrame({"x": data["x"].values[::-1], "y": data["ymin"].values[::-1]})
        poly_data = pd.concat([upper, lower], ignore_index=True)

        # Copy aesthetics
        for col in ("colour", "fill", "linewidth", "linetype", "alpha"):
            if col in data.columns:
                poly_data[col] = data[col].iloc[0]

        coords = _coord_transform(coord, poly_data, panel_params)

        fill_val = data["fill"].iloc[0] if "fill" in data.columns else "grey35"
        alpha_val = data["alpha"].iloc[0] if "alpha" in data.columns else None
        colour_val = data["colour"].iloc[0] if "colour" in data.columns else None
        lwd = data["linewidth"].iloc[0] * PT if "linewidth" in data.columns else 0.5 * PT
        lty = data["linetype"].iloc[0] if "linetype" in data.columns else 1

        g_poly = polygon_grob(
            x=coords["x"].values,
            y=coords["y"].values,
            default_units="native",
            gp=Gpar(
                fill=_fill_alpha(fill_val, alpha_val),
                col=colour_val if outline_type == "full" else None,
                lwd=lwd if outline_type == "full" else 0,
                lty=lty if outline_type == "full" else 1,
                lineend=lineend,
                linejoin=linejoin,
            ),
        )

        if outline_type == "full":
            return _ggname("geom_ribbon", g_poly)

        # Draw outline lines
        upper_coords = _coord_transform(coord, pd.DataFrame({"x": data["x"].values, "y": data["ymax"].values}), panel_params)
        lower_coords = _coord_transform(coord, pd.DataFrame({"x": data["x"].values[::-1], "y": data["ymin"].values[::-1]}), panel_params)

        line_gp = Gpar(col=colour_val, lwd=lwd, lty=lty, lineend=lineend, linejoin=linejoin)

        line_grobs = []
        if outline_type in ("both", "upper"):
            line_grobs.append(
                lines_grob(x=upper_coords["x"].values, y=upper_coords["y"].values, default_units="native", gp=line_gp)
            )
        if outline_type in ("both", "lower"):
            line_grobs.append(
                lines_grob(x=lower_coords["x"].values, y=lower_coords["y"].values, default_units="native", gp=line_gp)
            )

        return _ggname("geom_ribbon", grob_tree(g_poly, *line_grobs))


class GeomArea(GeomRibbon):
    """Area geom -- ribbon anchored at y=0."""

    required_aes: Tuple[str, ...] = ("x", "y")

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        sort_cols = [c for c in ["PANEL", "group", "x"] if c in data.columns]
        if sort_cols:
            data = data.sort_values(sort_cols)
        data["ymin"] = 0
        data["ymax"] = data["y"]
        return data


# ===========================================================================
# GeomSmooth
# ===========================================================================

class GeomSmooth(Geom):
    """Smooth geom -- fitted line + optional confidence ribbon."""

    required_aes: Tuple[str, ...] = ("x", "y")
    optional_aes: Tuple[str, ...] = ("ymin", "ymax")
    default_aes: Mapping = Mapping(
        colour="blue",
        fill="grey60",
        linewidth=1.0,
        linetype=1,
        weight=1,
        alpha=0.4,
    )
    extra_params: Tuple[str, ...] = ("na_rm", "orientation")
    draw_key = draw_key_smooth
    rename_size: bool = True

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        if "se" not in params:
            params["se"] = all(c in data.columns for c in ("ymin", "ymax"))
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return GeomLine.setup_data(GeomLine(), data, params)

    def draw_group(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        linejoin: str = "round",
        linemitre: float = 10,
        se: bool = False,
        flipped_aes: bool = False,
        **params: Any,
    ) -> Any:
        """Draw smooth line + optional ribbon."""
        ribbon_data = data.copy()
        if "colour" in ribbon_data.columns:
            ribbon_data["colour"] = None

        path_data = data.copy()
        path_data["alpha"] = None

        grobs = []
        has_ribbon = se and "ymin" in data.columns and "ymax" in data.columns
        if has_ribbon:
            grobs.append(
                GeomRibbon.draw_group(
                    GeomRibbon(), ribbon_data, panel_params, coord,
                    flipped_aes=flipped_aes,
                )
            )
        grobs.append(
            GeomLine.draw_panel(
                GeomLine(), path_data, panel_params, coord,
                lineend=lineend, linejoin=linejoin, linemitre=linemitre,
            )
        )
        return grob_tree(*grobs)


# ===========================================================================
# GeomSegment / GeomCurve / GeomSpoke
# ===========================================================================

class GeomSegment(Geom):
    """Segment geom -- straight line between two points."""

    required_aes: Tuple[str, ...] = ("x", "y", "xend", "yend")
    non_missing_aes: Tuple[str, ...] = ("linetype", "linewidth")
    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_path
    rename_size: bool = True

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        arrow: Any = None,
        lineend: str = "butt",
        linejoin: str = "round",
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        """Draw line segments."""
        data = data.copy()
        if "xend" not in data.columns:
            data["xend"] = data["x"]
        if "yend" not in data.columns:
            data["yend"] = data["y"]

        coords = _coord_transform(coord, data, panel_params)

        if coords.empty:
            return null_grob()

        return segments_grob(
            x0=coords["x"].values,
            y0=coords["y"].values,
            x1=coords["xend"].values,
            y1=coords["yend"].values,
            default_units="native",
            gp=Gpar(
                col=scales_alpha(
                    coords["colour"].values if "colour" in coords.columns else "black",
                    coords["alpha"].values if "alpha" in coords.columns else None,
                ),
                lwd=(
                    coords["linewidth"].values * PT
                    if "linewidth" in coords.columns
                    else 0.5 * PT
                ),
                lty=coords["linetype"].values if "linetype" in coords.columns else 1,
                lineend=lineend,
                linejoin=linejoin,
            ),
            arrow=arrow,
        )


class GeomCurve(GeomSegment):
    """Curve geom -- curved line between two points."""

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        curvature: float = 0.5,
        angle: float = 90,
        ncp: int = 5,
        shape: float = 0.5,
        arrow: Any = None,
        lineend: str = "butt",
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        """Draw curved segments."""
        coords = _coord_transform(coord, data, panel_params)

        if coords.empty:
            return null_grob()

        return curve_grob(
            x1=coords["x"].values,
            y1=coords["y"].values,
            x2=coords["xend"].values if "xend" in coords.columns else coords["x"].values,
            y2=coords["yend"].values if "yend" in coords.columns else coords["y"].values,
            default_units="native",
            curvature=curvature,
            angle=angle,
            ncp=ncp,
            shape=shape,
            gp=Gpar(
                col=scales_alpha(
                    coords["colour"].values if "colour" in coords.columns else "black",
                    coords["alpha"].values if "alpha" in coords.columns else None,
                ),
                lwd=(
                    coords["linewidth"].values * PT
                    if "linewidth" in coords.columns
                    else 0.5 * PT
                ),
                lty=coords["linetype"].values if "linetype" in coords.columns else 1,
                lineend=lineend,
            ),
            arrow=arrow,
        )


class GeomSpoke(GeomSegment):
    """Spoke geom -- segment parameterised by location, angle, and radius."""

    required_aes: Tuple[str, ...] = ("x", "y", "angle", "radius")

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        if "radius" not in data.columns:
            data["radius"] = params.get("radius", 1)
        if "angle" not in data.columns:
            data["angle"] = params.get("angle", 0)
        data["xend"] = data["x"] + np.cos(data["angle"]) * data["radius"]
        data["yend"] = data["y"] + np.sin(data["angle"]) * data["radius"]
        return data


# ===========================================================================
# GeomErrorbar / GeomErrorbarh
# ===========================================================================

class GeomErrorbar(Geom):
    """Errorbar geom -- T-shaped error bars."""

    required_aes: Tuple[str, ...] = ("x", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        width=0.9,
        alpha=None,
    )
    extra_params: Tuple[str, ...] = ("na_rm", "orientation")
    draw_key = draw_key_path
    rename_size: bool = True

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        width = params.get("width") or (data["width"].values if "width" in data.columns else 0.9)
        if isinstance(width, (int, float)):
            data["width"] = width
        data["xmin"] = data["x"] - data["width"] / 2
        data["xmax"] = data["x"] + data["width"] / 2
        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        width: Optional[float] = None,
        flipped_aes: bool = False,
        **params: Any,
    ) -> Any:
        """Draw error bars as T-shapes."""
        n = len(data)
        # Build the three segments per bar:
        # top cap, vertical, bottom cap
        x_vals = np.concatenate([
            np.column_stack([data["xmin"].values, data["xmax"].values, np.full(n, np.nan),
                             data["x"].values, data["x"].values, np.full(n, np.nan),
                             data["xmin"].values, data["xmax"].values]).ravel()
        ])
        y_vals = np.concatenate([
            np.column_stack([data["ymax"].values, data["ymax"].values, np.full(n, np.nan),
                             data["ymax"].values, data["ymin"].values, np.full(n, np.nan),
                             data["ymin"].values, data["ymin"].values]).ravel()
        ])

        # Create a path-like data frame
        path_data = pd.DataFrame({
            "x": x_vals,
            "y": y_vals,
            "colour": np.repeat(data["colour"].values if "colour" in data.columns else "black", 8),
            "alpha": np.repeat(data["alpha"].values if "alpha" in data.columns else np.nan, 8),
            "linewidth": np.repeat(data["linewidth"].values if "linewidth" in data.columns else 0.5, 8),
            "linetype": np.repeat(data["linetype"].values if "linetype" in data.columns else 1, 8),
            "group": np.repeat(np.arange(n), 8),
        })

        return GeomPath.draw_panel(GeomPath(), path_data, panel_params, coord, lineend=lineend)


class GeomErrorbarh(GeomErrorbar):
    """Horizontal errorbar geom (deprecated -- use ``geom_errorbar(orientation='y')``)."""

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        warnings.warn(
            "geom_errorbarh() is deprecated. Use geom_errorbar(orientation='y').",
            FutureWarning,
            stacklevel=2,
        )
        return super().setup_params(data, params)


# ===========================================================================
# GeomCrossbar
# ===========================================================================

class GeomCrossbar(Geom):
    """Crossbar geom -- box with median line."""

    required_aes: Tuple[str, ...] = ("x", "y", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        colour="black",
        fill=None,
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    extra_params: Tuple[str, ...] = ("na_rm", "orientation")
    draw_key = draw_key_crossbar
    rename_size: bool = True

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params.setdefault("fatten", 2.5)
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return GeomErrorbar.setup_data(GeomErrorbar(), data, params)

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        linejoin: str = "mitre",
        fatten: float = 2.5,
        width: Optional[float] = None,
        flipped_aes: bool = False,
        middle_gp: Optional[Dict] = None,
        box_gp: Optional[Dict] = None,
        **params: Any,
    ) -> Any:
        """Draw crossbar."""
        # Build box polygon
        n = len(data)
        boxes = []
        middles = []

        for i in range(n):
            row = data.iloc[i]
            xmin = row.get("xmin", row["x"] - 0.45)
            xmax = row.get("xmax", row["x"] + 0.45)
            ymin_val = row["ymin"]
            ymax_val = row["ymax"]
            y_mid = row["y"]

            box_df = pd.DataFrame({
                "x": [xmin, xmin, xmax, xmax, xmin],
                "y": [ymax_val, ymin_val, ymin_val, ymax_val, ymax_val],
                "colour": row.get("colour", "black"),
                "fill": row.get("fill"),
                "linewidth": row.get("linewidth", 0.5),
                "linetype": row.get("linetype", 1),
                "alpha": row.get("alpha"),
                "group": i,
            })
            boxes.append(box_df)

            mid_df = pd.DataFrame({
                "x": [xmin],
                "y": [y_mid],
                "xend": [xmax],
                "yend": [y_mid],
                "colour": row.get("colour", "black"),
                "linewidth": row.get("linewidth", 0.5) * fatten,
                "linetype": row.get("linetype", 1),
                "alpha": [np.nan],
            })
            middles.append(mid_df)

        box_data = pd.concat(boxes, ignore_index=True)
        mid_data = pd.concat(middles, ignore_index=True)

        box_grob = GeomPolygon.draw_panel(
            GeomPolygon(), box_data, panel_params, coord,
            lineend=lineend, linejoin=linejoin,
        )
        mid_grob = GeomSegment.draw_panel(
            GeomSegment(), mid_data, panel_params, coord, lineend=lineend,
        )

        return _ggname("geom_crossbar", grob_tree(box_grob, mid_grob))


# ===========================================================================
# GeomLinerange / GeomPointrange
# ===========================================================================

class GeomLinerange(Geom):
    """Linerange geom -- vertical line segments."""

    required_aes: Tuple[str, ...] = ("x", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    extra_params: Tuple[str, ...] = ("na_rm", "orientation")
    draw_key = draw_key_linerange
    rename_size: bool = True

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        data["flipped_aes"] = params.get("flipped_aes", False)
        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        flipped_aes: bool = False,
        na_rm: bool = False,
        arrow: Any = None,
        **params: Any,
    ) -> Any:
        """Draw line ranges."""
        seg_data = data.copy()
        seg_data["xend"] = seg_data["x"]
        seg_data["yend"] = seg_data["ymax"]
        seg_data["y"] = seg_data["ymin"]
        grob = GeomSegment.draw_panel(
            GeomSegment(), seg_data, panel_params, coord,
            lineend=lineend, na_rm=na_rm, arrow=arrow,
        )
        return _ggname("geom_linerange", grob)


class GeomPointrange(Geom):
    """Pointrange geom -- line range with point at y."""

    required_aes: Tuple[str, ...] = ("x", "y", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        colour="black",
        size=0.5,
        linewidth=0.5,
        linetype=1,
        shape=19,
        fill=None,
        alpha=None,
        stroke=0.5,
    )
    extra_params: Tuple[str, ...] = ("na_rm", "orientation")
    draw_key = draw_key_pointrange

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params.setdefault("fatten", 4)
        return GeomLinerange.setup_params(GeomLinerange(), data, params)

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return GeomLinerange.setup_data(GeomLinerange(), data, params)

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        fatten: float = 4,
        flipped_aes: bool = False,
        na_rm: bool = False,
        arrow: Any = None,
        **params: Any,
    ) -> Any:
        """Draw point + range."""
        line_grob = GeomLinerange.draw_panel(
            GeomLinerange(), data, panel_params, coord,
            lineend=lineend, flipped_aes=flipped_aes, na_rm=na_rm, arrow=arrow,
        )
        pt_data = data.copy()
        if "size" in pt_data.columns:
            pt_data["size"] = pt_data["size"] * fatten
        point_grob = GeomPoint.draw_panel(GeomPoint(), pt_data, panel_params, coord, na_rm=na_rm)
        return _ggname("geom_pointrange", grob_tree(line_grob, point_grob))


# ===========================================================================
# GeomBoxplot
# ===========================================================================

class GeomBoxplot(Geom):
    """Boxplot geom."""

    required_aes: Tuple[str, ...] = ("x", "lower", "upper", "middle", "ymin", "ymax")
    default_aes: Mapping = Mapping(
        weight=1,
        colour="grey20",
        fill="white",
        size=1.5,
        alpha=None,
        shape=19,
        linetype=1,
        linewidth=0.5,
        width=0.9,
    )
    extra_params: Tuple[str, ...] = ("na_rm", "orientation", "outliers")
    draw_key = draw_key_boxplot
    rename_size: bool = True

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params.setdefault("fatten", 2)
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        width = params.get("width") or (data["width"].values if "width" in data.columns else 0.9)
        if isinstance(width, (int, float)):
            data["width"] = width
        data["xmin"] = data["x"] - data["width"] / 2
        data["xmax"] = data["x"] + data["width"] / 2
        return data

    def draw_group(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        linejoin: str = "mitre",
        fatten: float = 2,
        outlier_gp: Optional[Dict] = None,
        whisker_gp: Optional[Dict] = None,
        staple_gp: Optional[Dict] = None,
        median_gp: Optional[Dict] = None,
        box_gp: Optional[Dict] = None,
        notch: bool = False,
        notchwidth: float = 0.5,
        staplewidth: float = 0,
        varwidth: bool = False,
        flipped_aes: bool = False,
        **params: Any,
    ) -> Any:
        """Draw a single boxplot."""
        if outlier_gp is None:
            outlier_gp = {}
        if whisker_gp is None:
            whisker_gp = {}

        row = data.iloc[0] if len(data) > 0 else data

        # Whiskers
        whisker_data = pd.DataFrame({
            "x": [row["x"], row["x"]],
            "y": [row["upper"], row["lower"]],
            "xend": [row["x"], row["x"]],
            "yend": [row["ymax"], row["ymin"]],
            "colour": whisker_gp.get("colour", row.get("colour", "grey20")),
            "linewidth": whisker_gp.get("linewidth", row.get("linewidth", 0.5)),
            "linetype": whisker_gp.get("linetype", row.get("linetype", 1)),
            "alpha": [np.nan, np.nan],
        })

        # Box (simple rectangle)
        xmin = row.get("xmin", row["x"] - 0.45)
        xmax = row.get("xmax", row["x"] + 0.45)
        box_data = pd.DataFrame({
            "x": [xmin, xmin, xmax, xmax, xmin],
            "y": [row["upper"], row["lower"], row["lower"], row["upper"], row["upper"]],
            "colour": row.get("colour", "grey20"),
            "fill": row.get("fill", "white"),
            "linewidth": row.get("linewidth", 0.5),
            "linetype": row.get("linetype", 1),
            "alpha": row.get("alpha"),
            "group": 1,
        })

        # Median line
        median_data = pd.DataFrame({
            "x": [xmin],
            "y": [row["middle"]],
            "xend": [xmax],
            "yend": [row["middle"]],
            "colour": (median_gp or {}).get("colour", row.get("colour", "grey20")),
            "linewidth": (median_gp or {}).get("linewidth", row.get("linewidth", 0.5)) * fatten,
            "linetype": (median_gp or {}).get("linetype", row.get("linetype", 1)),
            "alpha": [np.nan],
        })

        grobs = [
            GeomSegment.draw_panel(GeomSegment(), whisker_data, panel_params, coord, lineend=lineend),
            GeomPolygon.draw_panel(GeomPolygon(), box_data, panel_params, coord, lineend=lineend, linejoin=linejoin),
            GeomSegment.draw_panel(GeomSegment(), median_data, panel_params, coord, lineend=lineend),
        ]

        # Outliers
        if "outliers" in data.columns and data["outliers"].iloc[0] is not None:
            outliers_list = data["outliers"].iloc[0]
            if hasattr(outliers_list, "__len__") and len(outliers_list) > 0:
                outlier_data = pd.DataFrame({
                    "x": row["x"],
                    "y": outliers_list,
                    "colour": outlier_gp.get("colour", row.get("colour", "grey20")),
                    "fill": None,
                    "shape": outlier_gp.get("shape", 19),
                    "size": outlier_gp.get("size", 1.5),
                    "stroke": outlier_gp.get("stroke", 0.5),
                    "alpha": outlier_gp.get("alpha", row.get("alpha")),
                })
                grobs.insert(0, GeomPoint.draw_panel(GeomPoint(), outlier_data, panel_params, coord))

        return _ggname("geom_boxplot", grob_tree(*grobs))


# ===========================================================================
# GeomViolin
# ===========================================================================

class GeomViolin(Geom):
    """Violin geom."""

    required_aes: Tuple[str, ...] = ("x", "y")
    default_aes: Mapping = Mapping(
        weight=1,
        colour="grey20",
        fill="white",
        linewidth=0.5,
        linetype=1,
        alpha=None,
        width=0.9,
    )
    extra_params: Tuple[str, ...] = ("na_rm", "orientation")
    draw_key = draw_key_polygon
    rename_size: bool = True

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params["flipped_aes"] = params.get("flipped_aes", False)
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        width = params.get("width") or (data["width"].values if "width" in data.columns else 0.9)
        if isinstance(width, (int, float)):
            data["width"] = width
        if "group" in data.columns:
            for grp, idx in data.groupby("group").groups.items():
                data.loc[idx, "xmin"] = data.loc[idx, "x"] - data.loc[idx, "width"] / 2
                data.loc[idx, "xmax"] = data.loc[idx, "x"] + data.loc[idx, "width"] / 2
        return data

    def draw_group(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        quantile_gp: Optional[Dict] = None,
        flipped_aes: bool = False,
        **params: Any,
    ) -> Any:
        """Draw a single violin."""
        data = data.copy()
        if "violinwidth" in data.columns:
            data["xminv"] = data["x"] - data["violinwidth"] * (data["x"] - data.get("xmin", data["x"] - 0.45))
            data["xmaxv"] = data["x"] + data["violinwidth"] * (data.get("xmax", data["x"] + 0.45) - data["x"])
        else:
            data["xminv"] = data.get("xmin", data["x"] - 0.45)
            data["xmaxv"] = data.get("xmax", data["x"] + 0.45)

        upper = data.sort_values("y")[["y"]].copy()
        upper["x"] = data.sort_values("y")["xminv"].values
        lower = data.sort_values("y", ascending=False)[["y"]].copy()
        lower["x"] = data.sort_values("y", ascending=False)["xmaxv"].values

        newdata = pd.concat([upper, lower], ignore_index=True)
        # Close polygon
        newdata = pd.concat([newdata, newdata.iloc[:1]], ignore_index=True)

        for col in ("colour", "fill", "linewidth", "linetype", "alpha"):
            if col in data.columns:
                newdata[col] = data[col].iloc[0]
        newdata["group"] = 1

        return _ggname(
            "geom_violin",
            GeomPolygon.draw_panel(GeomPolygon(), newdata, panel_params, coord),
        )


# ===========================================================================
# GeomDotplot
# ===========================================================================

class GeomDotplot(Geom):
    """Dotplot geom."""

    required_aes: Tuple[str, ...] = ("x", "y")
    non_missing_aes: Tuple[str, ...] = ("size", "shape")
    default_aes: Mapping = Mapping(
        colour="black",
        fill="black",
        alpha=None,
        stroke=1.0,
        linetype=1,
        weight=1,
        width=0.9,
    )
    draw_key = draw_key_dotplot

    def draw_group(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        na_rm: bool = False,
        binaxis: str = "x",
        stackdir: str = "up",
        stackratio: float = 1,
        dotsize: float = 1,
        stackgroups: bool = False,
        **params: Any,
    ) -> Any:
        """Draw dotplot."""
        coords = _coord_transform(coord, data, panel_params)
        return _ggname(
            "geom_dotplot",
            points_grob(
                x=coords["x"].values,
                y=coords["y"].values,
                pch=21,
                gp=Gpar(
                    col=scales_alpha(
                        coords["colour"].values if "colour" in coords.columns else "black",
                        coords["alpha"].values if "alpha" in coords.columns else None,
                    ),
                    fill=_fill_alpha(
                        coords["fill"].values if "fill" in coords.columns else "black",
                        coords["alpha"].values if "alpha" in coords.columns else None,
                    ),
                ),
            ),
        )


# ===========================================================================
# GeomDensity
# ===========================================================================

class GeomDensity(GeomArea):
    """Density geom -- smoothed histogram."""

    default_aes: Mapping = Mapping(
        colour="black",
        fill=None,
        weight=1,
        alpha=None,
        linewidth=0.5,
        linetype=1,
    )


# ===========================================================================
# GeomHistogram / GeomFreqpoly
# ===========================================================================

# GeomHistogram is just GeomBar with stat="bin"
GeomHistogram = GeomBar  # alias

# GeomFreqpoly is just GeomPath with stat="bin"
GeomFreqpoly = GeomPath  # alias


# ===========================================================================
# GeomAbline / GeomHline / GeomVline
# ===========================================================================

class GeomAbline(Geom):
    """Abline geom -- diagonal reference line."""

    required_aes: Tuple[str, ...] = ("slope", "intercept")
    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_abline
    rename_size: bool = True

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        **params: Any,
    ) -> Any:
        """Draw diagonal lines."""
        # Get x-range from panel_params
        if hasattr(panel_params, "x") and hasattr(panel_params.x, "range"):
            x_rng = panel_params.x.range
        elif isinstance(panel_params, dict) and "x_range" in panel_params:
            x_rng = panel_params["x_range"]
        else:
            x_rng = (0, 1)

        seg_data = data.copy()
        seg_data["x"] = x_rng[0]
        seg_data["xend"] = x_rng[1]
        seg_data["y"] = seg_data["slope"] * x_rng[0] + seg_data["intercept"]
        seg_data["yend"] = seg_data["slope"] * x_rng[1] + seg_data["intercept"]

        return GeomSegment.draw_panel(GeomSegment(), seg_data, panel_params, coord, lineend=lineend)


class GeomHline(Geom):
    """Horizontal line geom."""

    required_aes: Tuple[str, ...] = ("yintercept",)
    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_path
    rename_size: bool = True

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        **params: Any,
    ) -> Any:
        """Draw horizontal lines."""
        x_rng = (0, 1)
        if hasattr(panel_params, "x") and hasattr(panel_params.x, "range"):
            x_rng = panel_params.x.range
        elif isinstance(panel_params, dict) and "x_range" in panel_params:
            x_rng = panel_params["x_range"]

        seg_data = data.copy()
        seg_data["x"] = x_rng[0]
        seg_data["xend"] = x_rng[1]
        seg_data["y"] = seg_data["yintercept"]
        seg_data["yend"] = seg_data["yintercept"]

        return GeomSegment.draw_panel(GeomSegment(), seg_data, panel_params, coord, lineend=lineend)


class GeomVline(Geom):
    """Vertical line geom."""

    required_aes: Tuple[str, ...] = ("xintercept",)
    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_vline
    rename_size: bool = True

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        **params: Any,
    ) -> Any:
        """Draw vertical lines."""
        y_rng = (0, 1)
        if hasattr(panel_params, "y") and hasattr(panel_params.y, "range"):
            y_rng = panel_params.y.range
        elif isinstance(panel_params, dict) and "y_range" in panel_params:
            y_rng = panel_params["y_range"]

        seg_data = data.copy()
        seg_data["x"] = seg_data["xintercept"]
        seg_data["xend"] = seg_data["xintercept"]
        seg_data["y"] = y_rng[0]
        seg_data["yend"] = y_rng[1]

        return GeomSegment.draw_panel(GeomSegment(), seg_data, panel_params, coord, lineend=lineend)


# ===========================================================================
# GeomRug
# ===========================================================================

class GeomRug(Geom):
    """Rug geom -- marginal tick marks."""

    optional_aes: Tuple[str, ...] = ("x", "y")
    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_path
    rename_size: bool = True

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        params.setdefault("sides", "bl")
        return params

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        sides: str = "bl",
        outside: bool = False,
        length: Any = None,
        **params: Any,
    ) -> Any:
        """Draw rug marks."""
        coords = _coord_transform(coord, data, panel_params)

        gp = Gpar(
            col=scales_alpha(
                coords["colour"].values if "colour" in coords.columns else "black",
                coords["alpha"].values if "alpha" in coords.columns else None,
            ),
            lty=coords["linetype"].values if "linetype" in coords.columns else 1,
            lwd=coords["linewidth"].values * PT if "linewidth" in coords.columns else 0.5 * PT,
            lineend=lineend,
        )

        rug_len = 0.03  # fraction of npc
        grobs = []

        if "x" in coords.columns:
            x_vals = coords["x"].values
            if "b" in sides:
                grobs.append(
                    segments_grob(
                        x0=x_vals, y0=np.zeros_like(x_vals),
                        x1=x_vals, y1=np.full_like(x_vals, rug_len),
                        default_units="native", gp=gp,
                    )
                )
            if "t" in sides:
                grobs.append(
                    segments_grob(
                        x0=x_vals, y0=np.ones_like(x_vals),
                        x1=x_vals, y1=np.full_like(x_vals, 1 - rug_len),
                        default_units="native", gp=gp,
                    )
                )

        if "y" in coords.columns:
            y_vals = coords["y"].values
            if "l" in sides:
                grobs.append(
                    segments_grob(
                        x0=np.zeros_like(y_vals), y0=y_vals,
                        x1=np.full_like(y_vals, rug_len), y1=y_vals,
                        default_units="native", gp=gp,
                    )
                )
            if "r" in sides:
                grobs.append(
                    segments_grob(
                        x0=np.ones_like(y_vals), y0=y_vals,
                        x1=np.full_like(y_vals, 1 - rug_len), y1=y_vals,
                        default_units="native", gp=gp,
                    )
                )

        return grob_tree(*grobs) if grobs else null_grob()


# ===========================================================================
# GeomBlank
# ===========================================================================

class GeomBlank(Geom):
    """Blank geom -- draws nothing."""

    default_aes: Mapping = Mapping()

    def handle_na(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return data

    def draw_panel(self, data: pd.DataFrame = None, panel_params: Any = None,
                   coord: Any = None, **params: Any) -> Any:
        return null_grob()


# ===========================================================================
# GeomContour / GeomContourFilled
# ===========================================================================

class GeomContour(GeomPath):
    """Contour geom -- contour lines of a 3D surface."""

    default_aes: Mapping = Mapping(
        weight=1,
        colour="blue",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )


class GeomContourFilled(GeomPolygon):
    """Filled contour geom."""
    pass


# ===========================================================================
# GeomDensity2d / GeomDensity2dFilled
# ===========================================================================

class GeomDensity2d(GeomPath):
    """2D density contour lines."""

    default_aes: Mapping = Mapping(
        colour="blue",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )


class GeomDensity2dFilled(GeomPolygon):
    """Filled 2D density contours."""
    pass


# ===========================================================================
# GeomHex
# ===========================================================================

class GeomHex(Geom):
    """Hexagonal bin geom."""

    required_aes: Tuple[str, ...] = ("x", "y")
    default_aes: Mapping = Mapping(
        colour=None,
        fill="grey50",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )
    draw_key = draw_key_polygon
    rename_size: bool = True

    def draw_group(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        linejoin: str = "mitre",
        linemitre: float = 10,
        **params: Any,
    ) -> Any:
        """Draw hexagons."""
        if data.empty:
            return null_grob()

        # Approximate hexagon vertices
        n = len(data)
        dx = resolution(data["x"].values, zero=False) if len(data) > 1 else 0.5
        dy = dx / np.sqrt(3)

        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        hex_x = dx * np.cos(angles)
        hex_y = dy * np.sin(angles)

        all_x = np.repeat(data["x"].values, 6) + np.tile(hex_x, n)
        all_y = np.repeat(data["y"].values, 6) + np.tile(hex_y, n)

        hex_data = pd.DataFrame({"x": all_x, "y": all_y})
        hex_data["group"] = np.repeat(np.arange(n), 6)

        for col in ("colour", "fill", "linewidth", "linetype", "alpha"):
            if col in data.columns:
                hex_data[col] = np.repeat(data[col].values, 6)

        coords = _coord_transform(coord, hex_data, panel_params)

        return _ggname(
            "geom_hex",
            polygon_grob(
                x=coords["x"].values,
                y=coords["y"].values,
                id=coords["group"].values,
                default_units="native",
                gp=Gpar(
                    col=data["colour"].values if "colour" in data.columns else None,
                    fill=_fill_alpha(
                        data["fill"].values if "fill" in data.columns else "grey50",
                        data["alpha"].values if "alpha" in data.columns else None,
                    ),
                    lwd=data["linewidth"].values * PT if "linewidth" in data.columns else 0.5 * PT,
                    lty=data["linetype"].values if "linetype" in data.columns else 1,
                ),
            ),
        )


# ===========================================================================
# GeomBin2d
# ===========================================================================

class GeomBin2d(GeomTile):
    """2D bin heatmap geom."""
    pass


# ===========================================================================
# GeomFunction
# ===========================================================================

class GeomFunction(GeomPath):
    """Function geom -- draw a mathematical function as a path."""

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        arrow: Any = None,
        lineend: str = "butt",
        linejoin: str = "round",
        linemitre: float = 10,
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        return GeomPath.draw_panel(
            self, data, panel_params, coord,
            arrow=arrow, lineend=lineend, linejoin=linejoin,
            linemitre=linemitre, na_rm=na_rm,
        )


# ===========================================================================
# GeomMap
# ===========================================================================

class GeomMap(GeomPolygon):
    """Map polygon geom."""

    required_aes: Tuple[str, ...] = ("map_id",)

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        lineend: str = "butt",
        linejoin: str = "round",
        linemitre: float = 10,
        map: Optional[pd.DataFrame] = None,
        **params: Any,
    ) -> Any:
        """Draw map polygons."""
        if map is None:
            return null_grob()

        map_df = map.copy()
        if "lat" in map_df.columns:
            map_df["y"] = map_df["lat"]
        if "long" in map_df.columns:
            map_df["x"] = map_df["long"]
        if "region" in map_df.columns:
            map_df["id"] = map_df["region"]

        # Merge aesthetics
        common = set(data["map_id"]) & set(map_df["id"])
        map_df = map_df[map_df["id"].isin(common)]
        data_subset = data[data["map_id"].isin(common)]

        if map_df.empty:
            return null_grob()

        # Assign aesthetics from data to map
        for col in ("colour", "fill", "linewidth", "linetype", "alpha"):
            if col in data_subset.columns:
                id_to_val = dict(zip(data_subset["map_id"], data_subset[col]))
                map_df[col] = map_df["id"].map(id_to_val)

        map_df["group"] = map_df["id"]

        return GeomPolygon.draw_panel(
            GeomPolygon(), map_df, panel_params, coord,
            lineend=lineend, linejoin=linejoin, linemitre=linemitre,
        )


# ===========================================================================
# GeomQuantile
# ===========================================================================

class GeomQuantile(GeomPath):
    """Quantile regression lines."""

    default_aes: Mapping = Mapping(
        weight=1,
        colour="blue",
        linewidth=0.5,
        linetype=1,
        alpha=None,
    )


# ===========================================================================
# GeomSf and related
# ===========================================================================

# ---------------------------------------------------------------------------
# sf geometry type mapping (mirrors R's sf_types vector)
# ---------------------------------------------------------------------------

_SF_TYPES: Dict[str, str] = {
    "Point": "point", "MultiPoint": "point",
    "LineString": "line", "MultiLineString": "line",
    "CircularString": "line", "CompoundCurve": "line",
    "MultiCurve": "line", "Curve": "line",
    "Polygon": "other", "MultiPolygon": "other",
    "CurvePolygon": "other", "MultiSurface": "other",
    "Surface": "other", "PolyhedralSurface": "other",
    "TIN": "other", "Triangle": "other",
    "GeometryCollection": "collection",
    "Geometry": "other",
}

# R's .pt and .stroke constants
_PT = 72.27 / 25.4   # ≈ 2.845
_STROKE = 96 / 25.4  # ≈ 3.78


def _sf_geometry_to_grobs(
    geometry_series: Any,
    colour: Any,
    fill: Any,
    linewidth: Any,
    linetype: Any,
    point_size: Any,
    pch: Any,
    lineend: str = "butt",
    linejoin: str = "round",
) -> Any:
    """Convert a Series of shapely geometries to grid_py grobs.

    This reimplements R's ``sf::st_as_grob()`` using shapely + grid_py.
    Each geometry is rendered as the appropriate grob type:
    - Point/MultiPoint → points_grob
    - LineString/MultiLineString → polyline_grob / lines_grob
    - Polygon/MultiPolygon → polygon_grob / path_grob

    Returns a GTree containing all grobs.
    """
    from shapely.geometry import (
        Point as ShapelyPoint,
        MultiPoint as ShapelyMultiPoint,
        LineString as ShapelyLineString,
        MultiLineString as ShapelyMultiLineString,
        Polygon as ShapelyPolygon,
        MultiPolygon as ShapelyMultiPolygon,
        GeometryCollection as ShapelyGeometryCollection,
    )

    children = []

    for i, geom in enumerate(geometry_series):
        if geom is None or geom.is_empty:
            continue

        # Per-row graphical parameters
        col_i = colour[i] if hasattr(colour, "__getitem__") and len(colour) > i else colour
        fill_i = fill[i] if hasattr(fill, "__getitem__") and len(fill) > i else fill
        lwd_i = linewidth[i] if hasattr(linewidth, "__getitem__") and len(linewidth) > i else linewidth
        lty_i = linetype[i] if hasattr(linetype, "__getitem__") and len(linetype) > i else linetype
        sz_i = point_size[i] if hasattr(point_size, "__getitem__") and len(point_size) > i else point_size
        pch_i = pch[i] if hasattr(pch, "__getitem__") and len(pch) > i else pch

        gp = Gpar(
            col=col_i, fill=fill_i, lwd=lwd_i, lty=lty_i,
            lineend=lineend, linejoin=linejoin,
        )

        if isinstance(geom, (ShapelyPoint,)):
            x, y = geom.x, geom.y
            children.append(points_grob(
                x=[x], y=[y], pch=int(pch_i) if pch_i is not None else 19,
                size=Unit(float(sz_i) if sz_i is not None else 1, "char"),
                gp=gp, name=f"sf_point_{i}",
            ))

        elif isinstance(geom, (ShapelyMultiPoint,)):
            xs = [p.x for p in geom.geoms]
            ys = [p.y for p in geom.geoms]
            children.append(points_grob(
                x=xs, y=ys, pch=int(pch_i) if pch_i is not None else 19,
                size=Unit(float(sz_i) if sz_i is not None else 1, "char"),
                gp=gp, name=f"sf_mpoint_{i}",
            ))

        elif isinstance(geom, (ShapelyLineString,)):
            xs, ys = zip(*geom.coords) if len(geom.coords) > 0 else ([], [])
            children.append(lines_grob(
                x=list(xs), y=list(ys), gp=gp, name=f"sf_line_{i}",
            ))

        elif isinstance(geom, (ShapelyMultiLineString,)):
            for j, line in enumerate(geom.geoms):
                xs, ys = zip(*line.coords) if len(line.coords) > 0 else ([], [])
                children.append(lines_grob(
                    x=list(xs), y=list(ys), gp=gp,
                    name=f"sf_mline_{i}_{j}",
                ))

        elif isinstance(geom, (ShapelyPolygon,)):
            # Exterior ring
            xs, ys = geom.exterior.coords.xy
            children.append(polygon_grob(
                x=list(xs), y=list(ys), gp=gp, name=f"sf_poly_{i}",
            ))

        elif isinstance(geom, (ShapelyMultiPolygon,)):
            for j, poly in enumerate(geom.geoms):
                xs, ys = poly.exterior.coords.xy
                children.append(polygon_grob(
                    x=list(xs), y=list(ys), gp=gp,
                    name=f"sf_mpoly_{i}_{j}",
                ))

        elif isinstance(geom, (ShapelyGeometryCollection,)):
            # Recurse into collection
            sub_grobs = _sf_geometry_to_grobs(
                list(geom.geoms),
                colour=[col_i] * len(geom.geoms),
                fill=[fill_i] * len(geom.geoms),
                linewidth=[lwd_i] * len(geom.geoms),
                linetype=[lty_i] * len(geom.geoms),
                point_size=[sz_i] * len(geom.geoms),
                pch=[pch_i] * len(geom.geoms),
                lineend=lineend, linejoin=linejoin,
            )
            children.append(sub_grobs)

    if not children:
        return null_grob()

    return grob_tree(*children, name="sf_geometries")


class GeomSf(Geom):
    """Simple features geom.

    Draws different geometric objects depending on the geometry type:
    points, lines, or polygons — mirroring R's ``geom_sf()``.
    """

    required_aes: Tuple[str, ...] = ("geometry",)
    default_aes: Mapping = Mapping(
        shape=None,
        colour=None,
        fill=None,
        size=None,
        linewidth=None,
        linetype=None,
        alpha=None,
        stroke=0.5,
    )

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        legend: Any = None,
        lineend: str = "butt",
        linejoin: str = "round",
        linemitre: float = 10,
        arrow: Any = None,
        na_rm: bool = True,
        **params: Any,
    ) -> Any:
        """Draw sf geometries.

        Mirrors R's ``GeomSf$draw_panel``: classifies each geometry
        as point/line/other, computes per-type graphical parameters,
        and renders via ``_sf_geometry_to_grobs``.
        """
        coords = _coord_transform(coord, data, panel_params)

        if "geometry" not in coords.columns:
            return null_grob()

        import shapely

        n = len(coords)

        # Classify geometry types (mirrors R's sf_types vector)
        types = coords["geometry"].apply(
            lambda g: _SF_TYPES.get(g.geom_type, "other") if g is not None else "other"
        )
        is_point = types == "point"
        is_line = types == "line"
        is_collection = types == "collection"

        # Shape translation
        shape = coords.get("shape", pd.Series([19] * n))
        shape = shape.apply(
            lambda s: translate_shape_string(s) if isinstance(s, str) else (s if s is not None else 19)
        )

        # Fill with alpha (mirrors R: fill_alpha for all, arrow.fill for lines)
        fill_raw = coords.get("fill", pd.Series([np.nan] * n))
        alpha_raw = coords.get("alpha", pd.Series([1.0] * n)).fillna(1.0)
        fill_vals = _fill_alpha(fill_raw, alpha_raw)

        # Colour with alpha for points and lines
        colour = coords.get("colour", pd.Series(["black"] * n))

        # Point size vs linewidth (R: point_size for points/collections,
        # linewidth for everything else)
        size_raw = coords.get("size", pd.Series([1.5] * n)).fillna(1.5)
        lw_raw = coords.get("linewidth", pd.Series([0.5] * n)).fillna(0.5)
        point_size = size_raw.copy()
        point_size[~(is_point | is_collection)] = lw_raw[~(is_point | is_collection)]

        # Stroke
        stroke_raw = coords.get("stroke", pd.Series([0.5] * n)).fillna(0.5)
        stroke_vals = stroke_raw * _STROKE / 2
        font_size = point_size * _PT + stroke_vals

        # Linewidth
        linewidth = lw_raw * _PT
        linewidth[is_point] = stroke_vals[is_point]

        linetype = coords.get("linetype", pd.Series([1] * n))

        return _sf_geometry_to_grobs(
            coords["geometry"],
            colour=colour.values,
            fill=fill_vals if hasattr(fill_vals, '__len__') else [fill_vals] * n,
            linewidth=linewidth.values,
            linetype=linetype.values,
            point_size=font_size.values,
            pch=shape.values,
            lineend=lineend,
            linejoin=linejoin,
        )

    def draw_key(self, data: Any, params: Dict[str, Any], size: Any = None) -> Any:
        legend_type = params.get("legend", "other")
        if legend_type == "point":
            return draw_key_point(data, params, size)
        elif legend_type == "line":
            return draw_key_path(data, params, size)
        return draw_key_polygon(data, params, size)


# Placeholder classes for annotation geoms
class GeomAnnotationMap(GeomPolygon):
    """Annotation map geom."""
    pass


class GeomCustomAnn(Geom):
    """Custom annotation geom."""

    def draw_panel(self, data: pd.DataFrame = None, panel_params: Any = None,
                   coord: Any = None, grob: Any = None, **params: Any) -> Any:
        return grob if grob is not None else null_grob()


class GeomRasterAnn(Geom):
    """Raster annotation geom."""

    def draw_panel(self, data: pd.DataFrame = None, panel_params: Any = None,
                   coord: Any = None, raster: Any = None, **params: Any) -> Any:
        if raster is not None:
            return raster_grob(image=raster)
        return null_grob()


def _calc_logticks(
    base: float = 10,
    minpow: int = 0,
    maxpow: int = 1,
    start: float = 0.0,
    shortend: float = 0.1,
    midend: float = 0.2,
    longend: float = 0.3,
) -> pd.DataFrame:
    """Compute log tick mark positions and lengths.

    Mirrors R's ``calc_logticks()`` from ``annotation-logticks.R``.

    Returns
    -------
    pd.DataFrame
        Columns: ``value``, ``start``, ``end``.
    """
    ticks_per_base = int(base) - 1
    reps = maxpow - minpow

    if reps <= 0 or ticks_per_base <= 0:
        return pd.DataFrame({"value": [base ** maxpow], "start": [start], "end": [longend]})

    ticknums = np.tile(np.linspace(1, base - 1, ticks_per_base), reps)
    powers = np.repeat(np.arange(minpow, maxpow), ticks_per_base)
    ticks = ticknums * (base ** powers)
    ticks = np.append(ticks, base ** maxpow)

    tickend = np.full(len(ticks), shortend)
    cycle_idx = (ticknums - 1).astype(int)
    cycle_idx = np.append(cycle_idx, 0)

    # Major ticks (at each power of base)
    tickend[cycle_idx == 0] = longend

    # Mid ticks (at base/2, e.g. 5 for base 10)
    longtick_after = ticks_per_base // 2
    tickend[cycle_idx == longtick_after] = midend

    return pd.DataFrame({
        "value": ticks,
        "start": np.full(len(ticks), start),
        "end": tickend,
    })


class GeomLogticks(Geom):
    """Log-scale tick marks geom.

    Mirrors R's ``GeomLogticks`` from ``annotation-logticks.R``.
    Draws diminishing tick marks at log-spaced intervals on specified
    sides of the plot panel.
    """

    default_aes: Mapping = Mapping(
        colour="black",
        linewidth=0.5,
        linetype=1,
        alpha=1.0,
    )

    def draw_panel(
        self,
        data: pd.DataFrame = None,
        panel_params: Any = None,
        coord: Any = None,
        base: float = 10,
        sides: str = "bl",
        outside: bool = False,
        scaled: bool = True,
        short: float = 0.1,
        mid: float = 0.2,
        long: float = 0.3,
        **params: Any,
    ) -> Any:
        """Draw log tick marks on panel edges.

        Mirrors R's ``GeomLogticks$draw_panel``.
        """
        if panel_params is None:
            return null_grob()

        x_range = panel_params.get("x_range") or panel_params.get("x.range")
        y_range = panel_params.get("y_range") or panel_params.get("y.range")

        # Extract gp from data row
        colour = "black"
        linewidth_val = 0.5
        linetype_val = 1
        alpha_val = 1.0
        if data is not None and len(data) > 0:
            row = data.iloc[0]
            colour = row.get("colour", "black")
            linewidth_val = float(row.get("linewidth", 0.5))
            linetype_val = row.get("linetype", 1)
            alpha_val = float(row.get("alpha", 1.0) or 1.0)

        gp = Gpar(col=colour, lwd=linewidth_val, lty=linetype_val, alpha=alpha_val)

        ticks_grobs = []

        # X-axis ticks (bottom / top)
        if ("b" in sides or "t" in sides) and x_range is not None:
            xr = [float(x_range[0]), float(x_range[1])]
            if all(np.isfinite(xr)):
                xticks = _calc_logticks(
                    base=base,
                    minpow=int(np.floor(xr[0])),
                    maxpow=int(np.ceil(xr[1])),
                    start=0.0, shortend=short, midend=mid, longend=long,
                )
                if scaled:
                    xticks["value"] = np.log(xticks["value"]) / np.log(base)

                # Rescale to [0, 1] NPC
                span = xr[1] - xr[0]
                if span > 0:
                    xticks["x"] = (xticks["value"] - xr[0]) / span
                    xticks = xticks[(xticks["x"] >= 0) & (xticks["x"] <= 1)]

                    if outside:
                        xticks["end"] = -xticks["end"]

                    if "b" in sides and len(xticks) > 0:
                        ticks_grobs.append(segments_grob(
                            x0=xticks["x"].values, y0=np.zeros(len(xticks)),
                            x1=xticks["x"].values, y1=xticks["end"].values * 0.02,
                            gp=gp, name="logtick_x_b",
                        ))

                    if "t" in sides and len(xticks) > 0:
                        ticks_grobs.append(segments_grob(
                            x0=xticks["x"].values, y0=np.ones(len(xticks)),
                            x1=xticks["x"].values, y1=1.0 - xticks["end"].values * 0.02,
                            gp=gp, name="logtick_x_t",
                        ))

        # Y-axis ticks (left / right)
        if ("l" in sides or "r" in sides) and y_range is not None:
            yr = [float(y_range[0]), float(y_range[1])]
            if all(np.isfinite(yr)):
                yticks = _calc_logticks(
                    base=base,
                    minpow=int(np.floor(yr[0])),
                    maxpow=int(np.ceil(yr[1])),
                    start=0.0, shortend=short, midend=mid, longend=long,
                )
                if scaled:
                    yticks["value"] = np.log(yticks["value"]) / np.log(base)

                span = yr[1] - yr[0]
                if span > 0:
                    yticks["y"] = (yticks["value"] - yr[0]) / span
                    yticks = yticks[(yticks["y"] >= 0) & (yticks["y"] <= 1)]

                    if outside:
                        yticks["end"] = -yticks["end"]

                    if "l" in sides and len(yticks) > 0:
                        ticks_grobs.append(segments_grob(
                            x0=np.zeros(len(yticks)),
                            y0=yticks["y"].values,
                            x1=yticks["end"].values * 0.02,
                            y1=yticks["y"].values,
                            gp=gp, name="logtick_y_l",
                        ))

                    if "r" in sides and len(yticks) > 0:
                        ticks_grobs.append(segments_grob(
                            x0=np.ones(len(yticks)),
                            y0=yticks["y"].values,
                            x1=1.0 - yticks["end"].values * 0.02,
                            y1=yticks["y"].values,
                            gp=gp, name="logtick_y_r",
                        ))

        if not ticks_grobs:
            return null_grob()

        return grob_tree(*ticks_grobs, name="logticks")


# ===========================================================================
# GeomCount / GeomJitter (trivial wrappers)
# ===========================================================================

# GeomCount is GeomPoint + stat_sum
GeomCount = GeomPoint  # alias

# GeomJitter is GeomPoint + position_jitter
GeomJitter = GeomPoint  # alias


# ===========================================================================
# Constructor functions
# ===========================================================================

def _layer_import():
    """Lazy import of ``layer`` to avoid circular imports."""
    from ggplot2_py.layer import layer
    return layer


# ---------------------------------------------------------------------------
# Point / Path / Line / Step
# ---------------------------------------------------------------------------

def geom_point(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a point (scatter) layer.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame, optional
    stat, position : str
    na_rm : bool
    show_legend : bool or None
    inherit_aes : bool
    **kwargs : additional aesthetic or parameter overrides

    Returns
    -------
    Layer
    """
    layer = _layer_import()
    return layer(
        geom=GeomPoint, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_path(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a path layer (connects observations in data order)."""
    layer = _layer_import()
    return layer(
        geom=GeomPath, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_line(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a line layer (connects observations sorted by x)."""
    layer = _layer_import()
    return layer(
        geom=GeomLine, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, **kwargs},
    )


def geom_step(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    direction: str = "hv",
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a stairstep layer."""
    layer = _layer_import()
    return layer(
        geom=GeomStep, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "direction": direction, "orientation": orientation, **kwargs},
    )


# ---------------------------------------------------------------------------
# Bar / Col
# ---------------------------------------------------------------------------

def geom_bar(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "count",
    position: str = "stack",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    just: float = 0.5,
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a bar layer."""
    layer = _layer_import()
    return layer(
        geom=GeomBar, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "just": just, "orientation": orientation, **kwargs},
    )


def geom_col(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "stack",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    just: float = 0.5,
    **kwargs: Any,
) -> Any:
    """Create a column layer (bars with stat = identity)."""
    layer = _layer_import()
    return layer(
        geom=GeomCol, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "just": just, **kwargs},
    )


# ---------------------------------------------------------------------------
# Rect / Tile / Raster
# ---------------------------------------------------------------------------

def geom_rect(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a rectangle layer."""
    layer = _layer_import()
    return layer(
        geom=GeomRect, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_tile(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a tile layer."""
    layer = _layer_import()
    return layer(
        geom=GeomTile, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_raster(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    hjust: float = 0.5,
    vjust: float = 0.5,
    interpolate: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a raster layer."""
    layer = _layer_import()
    return layer(
        geom=GeomRaster, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "hjust": hjust, "vjust": vjust, "interpolate": interpolate, **kwargs},
    )


# ---------------------------------------------------------------------------
# Text / Label
# ---------------------------------------------------------------------------

def geom_text(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "nudge",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    parse: bool = False,
    check_overlap: bool = False,
    size_unit: str = "mm",
    **kwargs: Any,
) -> Any:
    """Create a text layer."""
    layer = _layer_import()
    return layer(
        geom=GeomText, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "parse": parse, "check_overlap": check_overlap, "size_unit": size_unit, **kwargs},
    )


def geom_label(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "nudge",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    parse: bool = False,
    size_unit: str = "mm",
    **kwargs: Any,
) -> Any:
    """Create a label layer (text with background box)."""
    layer = _layer_import()
    return layer(
        geom=GeomLabel, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "parse": parse, "size_unit": size_unit, **kwargs},
    )


# ---------------------------------------------------------------------------
# Boxplot / Violin / Dotplot
# ---------------------------------------------------------------------------

def geom_boxplot(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "boxplot",
    position: str = "dodge2",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    outliers: bool = True,
    notch: bool = False,
    notchwidth: float = 0.5,
    staplewidth: float = 0,
    varwidth: bool = False,
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a boxplot layer."""
    layer = _layer_import()
    return layer(
        geom=GeomBoxplot, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={
            "na_rm": na_rm, "outliers": outliers, "notch": notch,
            "notchwidth": notchwidth, "staplewidth": staplewidth,
            "varwidth": varwidth, "orientation": orientation, **kwargs,
        },
    )


def geom_violin(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "ydensity",
    position: str = "dodge",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    trim: bool = True,
    scale: str = "area",
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a violin layer."""
    layer = _layer_import()
    return layer(
        geom=GeomViolin, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "trim": trim, "scale": scale, "orientation": orientation, **kwargs},
    )


def geom_dotplot(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "bindot",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    binaxis: str = "x",
    method: str = "dotdensity",
    stackdir: str = "up",
    stackratio: float = 1,
    dotsize: float = 1,
    **kwargs: Any,
) -> Any:
    """Create a dotplot layer."""
    layer = _layer_import()
    return layer(
        geom=GeomDotplot, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={
            "na_rm": na_rm, "binaxis": binaxis, "method": method,
            "stackdir": stackdir, "stackratio": stackratio, "dotsize": dotsize,
            **kwargs,
        },
    )


# ---------------------------------------------------------------------------
# Ribbon / Area / Smooth
# ---------------------------------------------------------------------------

def geom_ribbon(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    orientation: Any = None,
    outline_type: str = "both",
    **kwargs: Any,
) -> Any:
    """Create a ribbon layer."""
    layer = _layer_import()
    return layer(
        geom=GeomRibbon, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, "outline_type": outline_type, **kwargs},
    )


def geom_area(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "align",
    position: str = "stack",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    orientation: Any = None,
    outline_type: str = "upper",
    **kwargs: Any,
) -> Any:
    """Create an area layer."""
    layer = _layer_import()
    return layer(
        geom=GeomArea, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, "outline_type": outline_type, **kwargs},
    )


def geom_smooth(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "smooth",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    method: Any = None,
    formula: Any = None,
    se: bool = True,
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a smooth layer."""
    layer = _layer_import()
    params: Dict[str, Any] = {
        "na_rm": na_rm, "orientation": orientation, "se": se, **kwargs,
    }
    if stat == "smooth":
        params["method"] = method
        params["formula"] = formula
    return layer(
        geom=GeomSmooth, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params=params,
    )


# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------

def geom_polygon(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    rule: str = "evenodd",
    **kwargs: Any,
) -> Any:
    """Create a polygon layer."""
    layer = _layer_import()
    return layer(
        geom=GeomPolygon, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "rule": rule, **kwargs},
    )


# ---------------------------------------------------------------------------
# Errorbar / Crossbar / Linerange / Pointrange
# ---------------------------------------------------------------------------

def geom_errorbar(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create an errorbar layer."""
    layer = _layer_import()
    return layer(
        geom=GeomErrorbar, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, **kwargs},
    )


def geom_errorbarh(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    orientation: str = "y",
    **kwargs: Any,
) -> Any:
    """Create a horizontal errorbar (deprecated -- use geom_errorbar)."""
    warnings.warn(
        "geom_errorbarh() is deprecated. Use geom_errorbar(orientation='y').",
        FutureWarning,
        stacklevel=2,
    )
    return geom_errorbar(mapping=mapping, data=data, orientation=orientation, **kwargs)


def geom_crossbar(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a crossbar layer."""
    layer = _layer_import()
    return layer(
        geom=GeomCrossbar, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, **kwargs},
    )


def geom_linerange(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a linerange layer."""
    layer = _layer_import()
    return layer(
        geom=GeomLinerange, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, **kwargs},
    )


def geom_pointrange(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    orientation: Any = None,
    fatten: float = 4,
    **kwargs: Any,
) -> Any:
    """Create a pointrange layer."""
    layer = _layer_import()
    return layer(
        geom=GeomPointrange, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, "fatten": fatten, **kwargs},
    )


# ---------------------------------------------------------------------------
# Segment / Curve / Spoke
# ---------------------------------------------------------------------------

def geom_segment(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a segment layer."""
    layer = _layer_import()
    return layer(
        geom=GeomSegment, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_curve(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    curvature: float = 0.5,
    angle: float = 90,
    ncp: int = 5,
    **kwargs: Any,
) -> Any:
    """Create a curve layer."""
    layer = _layer_import()
    return layer(
        geom=GeomCurve, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "curvature": curvature, "angle": angle, "ncp": ncp, **kwargs},
    )


def geom_spoke(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a spoke layer."""
    layer = _layer_import()
    return layer(
        geom=GeomSpoke, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def geom_density(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "density",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    outline_type: str = "upper",
    **kwargs: Any,
) -> Any:
    """Create a density layer."""
    layer = _layer_import()
    return layer(
        geom=GeomDensity, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "outline_type": outline_type, **kwargs},
    )


def geom_density_2d(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "density_2d",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    contour_var: str = "density",
    **kwargs: Any,
) -> Any:
    """Create a 2D density contour layer."""
    layer = _layer_import()
    return layer(
        geom=GeomDensity2d, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "contour_var": contour_var, **kwargs},
    )


# Aliases
geom_density2d = geom_density_2d


def geom_density_2d_filled(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "density_2d_filled",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    contour_var: str = "density",
    **kwargs: Any,
) -> Any:
    """Create a filled 2D density contour layer."""
    layer = _layer_import()
    return layer(
        geom=GeomDensity2dFilled, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "contour_var": contour_var, **kwargs},
    )


geom_density2d_filled = geom_density_2d_filled


# ---------------------------------------------------------------------------
# Contour
# ---------------------------------------------------------------------------

def geom_contour(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "contour",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    bins: Optional[int] = None,
    binwidth: Optional[float] = None,
    breaks: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a contour line layer."""
    layer = _layer_import()
    return layer(
        geom=GeomContour, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "bins": bins, "binwidth": binwidth, "breaks": breaks, **kwargs},
    )


def geom_contour_filled(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "contour_filled",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    bins: Optional[int] = None,
    binwidth: Optional[float] = None,
    breaks: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a filled contour layer."""
    layer = _layer_import()
    return layer(
        geom=GeomContourFilled, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "bins": bins, "binwidth": binwidth, "breaks": breaks, **kwargs},
    )


# ---------------------------------------------------------------------------
# Hex / Bin2d
# ---------------------------------------------------------------------------

def geom_hex(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "binhex",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a hex bin layer."""
    layer = _layer_import()
    return layer(
        geom=GeomHex, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_bin_2d(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "bin2d",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a 2D bin heatmap layer."""
    layer = _layer_import()
    return layer(
        geom=GeomBin2d, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


geom_bin2d = geom_bin_2d


# ---------------------------------------------------------------------------
# Abline / Hline / Vline
# ---------------------------------------------------------------------------

def geom_abline(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    slope: Any = None,
    intercept: Any = None,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = False,
    **kwargs: Any,
) -> Any:
    """Create an abline layer."""
    layer = _layer_import()
    if slope is not None or intercept is not None:
        if slope is None:
            slope = 1
        if intercept is None:
            intercept = 0
        data = pd.DataFrame({"intercept": [intercept] if not hasattr(intercept, "__len__") else intercept,
                              "slope": [slope] if not hasattr(slope, "__len__") else slope})
        mapping = Mapping(intercept="intercept", slope="slope")
        show_legend = False
    elif mapping is None:
        slope = 1
        intercept = 0
        data = pd.DataFrame({"intercept": [intercept], "slope": [slope]})
        mapping = Mapping(intercept="intercept", slope="slope")

    return layer(
        geom=GeomAbline, stat=stat, data=data, mapping=mapping,
        position="identity", show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_hline(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    yintercept: Any = None,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a horizontal line layer."""
    layer = _layer_import()
    if yintercept is not None:
        data = pd.DataFrame({"yintercept": [yintercept] if not hasattr(yintercept, "__len__") else yintercept})
        mapping = Mapping(yintercept="yintercept")
        show_legend = False

    return layer(
        geom=GeomHline, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_vline(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    xintercept: Any = None,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a vertical line layer."""
    layer = _layer_import()
    if xintercept is not None:
        data = pd.DataFrame({"xintercept": [xintercept] if not hasattr(xintercept, "__len__") else xintercept})
        mapping = Mapping(xintercept="xintercept")
        show_legend = False

    return layer(
        geom=GeomVline, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


# ---------------------------------------------------------------------------
# Rug
# ---------------------------------------------------------------------------

def geom_rug(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    sides: str = "bl",
    outside: bool = False,
    length: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a rug layer."""
    layer = _layer_import()
    return layer(
        geom=GeomRug, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "sides": sides, "outside": outside, "length": length, **kwargs},
    )


# ---------------------------------------------------------------------------
# Blank
# ---------------------------------------------------------------------------

def geom_blank(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a blank layer (draws nothing)."""
    layer = _layer_import()
    return layer(
        geom=GeomBlank, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params=kwargs,
    )


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------

def geom_function(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "function",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a function layer."""
    layer = _layer_import()
    return layer(
        geom=GeomFunction, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


# ---------------------------------------------------------------------------
# Histogram / Freqpoly
# ---------------------------------------------------------------------------

def geom_histogram(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "bin",
    position: str = "stack",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    binwidth: Any = None,
    bins: Optional[int] = None,
    orientation: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a histogram layer."""
    layer = _layer_import()
    return layer(
        geom=GeomBar, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "binwidth": binwidth, "bins": bins, "orientation": orientation, **kwargs},
    )


def geom_freqpoly(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "bin",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a frequency polygon layer."""
    layer = _layer_import()
    params: Dict[str, Any] = {"na_rm": na_rm, **kwargs}
    if stat == "bin":
        params["pad"] = True
    return layer(
        geom=GeomPath, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params=params,
    )


# ---------------------------------------------------------------------------
# Count / Jitter
# ---------------------------------------------------------------------------

def geom_count(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "sum",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a count layer (points sized by n at each location)."""
    layer = _layer_import()
    return layer(
        geom=GeomPoint, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_jitter(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "jitter",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    width: Optional[float] = None,
    height: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Create a jittered point layer."""
    layer = _layer_import()
    if width is not None or height is not None:
        position = {"name": "jitter", "width": width, "height": height}
    return layer(
        geom=GeomPoint, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def geom_map(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    map: Optional[pd.DataFrame] = None,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a map polygon layer."""
    layer = _layer_import()
    return layer(
        geom=GeomMap, stat=stat, data=data, mapping=mapping,
        position="identity", show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "map": map, **kwargs},
    )


# ---------------------------------------------------------------------------
# Quantile
# ---------------------------------------------------------------------------

def geom_quantile(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "quantile",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a quantile regression line layer."""
    layer = _layer_import()
    return layer(
        geom=GeomQuantile, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


# ---------------------------------------------------------------------------
# Sf
# ---------------------------------------------------------------------------

def geom_sf(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "sf",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a simple-features layer."""
    layer = _layer_import()
    return layer(
        geom=GeomSf, stat=stat, data=data, mapping=mapping or Mapping(),
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_sf_text(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "sf_coordinates",
    position: str = "nudge",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    parse: bool = False,
    check_overlap: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a text layer for sf geometries."""
    layer = _layer_import()
    return layer(
        geom=GeomText, stat=stat, data=data, mapping=mapping or Mapping(),
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "parse": parse, "check_overlap": check_overlap, **kwargs},
    )


def geom_sf_label(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "sf_coordinates",
    position: str = "nudge",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    parse: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a label layer for sf geometries."""
    layer = _layer_import()
    return layer(
        geom=GeomLabel, stat=stat, data=data, mapping=mapping or Mapping(),
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "parse": parse, **kwargs},
    )


# ---------------------------------------------------------------------------
# QQ (geom only -- delegates to stat_qq / stat_qq_line)
# ---------------------------------------------------------------------------

def geom_qq(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "qq",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a QQ-plot point layer."""
    layer = _layer_import()
    return layer(
        geom=GeomPoint, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def geom_qq_line(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "qq_line",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a QQ-line layer."""
    layer = _layer_import()
    return layer(
        geom=GeomPath, stat=stat, data=data, mapping=mapping,
        position=position, show_legend=show_legend, inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )

"""
Coordinate systems for ggplot2.

Coordinate systems control how position aesthetics are mapped to the 2-D
plane of the plot. They also provide axes, panel backgrounds, and
foreground decorations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn
from ggplot2_py.ggproto import GGProto, ggproto
from ggplot2_py._utils import snake_class, modify_list, compact

__all__ = [
    "Coord",
    "CoordCartesian",
    "CoordFixed",
    "CoordFlip",
    "CoordPolar",
    "CoordRadial",
    "CoordTrans",
    "CoordTransform",
    "coord_cartesian",
    "coord_equal",
    "coord_fixed",
    "coord_flip",
    "coord_polar",
    "coord_radial",
    "coord_trans",
    "coord_transform",
    "coord_munch",
    "is_coord",
    "is_Coord",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rescale(
    x: np.ndarray,
    to: Tuple[float, float] = (0.0, 1.0),
    from_: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Linearly rescale *x* from *from_* range to *to* range."""
    x = np.asarray(x, dtype=float)
    if from_ is None:
        from_ = (float(np.nanmin(x)), float(np.nanmax(x)))
    rng = from_[1] - from_[0]
    if rng == 0:
        return np.full_like(x, (to[0] + to[1]) / 2.0)
    return (x - from_[0]) / rng * (to[1] - to[0]) + to[0]


def _squish_infinite(
    x: np.ndarray,
    range_: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Squish infinite values to the range endpoints."""
    x = np.asarray(x, dtype=float)
    if range_ is not None:
        x = np.where(np.isneginf(x), range_[0], x)
        x = np.where(np.isposinf(x), range_[1], x)
    else:
        x = np.where(np.isneginf(x), 0.0, x)
        x = np.where(np.isposinf(x), 1.0, x)
    return x


def _dist_euclidean(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Euclidean distance between successive points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 0 or len(x) < 2:
        return np.array([0.0])
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx ** 2 + dy ** 2)


def _dist_polar(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Distance in polar coordinates between successive points."""
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if len(r) < 2:
        return np.array([0.0])
    dr = np.diff(r)
    dtheta = np.diff(theta)
    r1 = r[:-1]
    r2 = r[1:]
    return np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(dtheta))


def _theta_rescale(
    x: np.ndarray,
    range_: Tuple[float, float],
    arc: Tuple[float, float] = (0, 2 * math.pi),
    direction: int = 1,
) -> np.ndarray:
    """Rescale theta to arc range, squishing and wrapping."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, range_[0], range_[1])
    out = _rescale(x, to=arc, from_=range_)
    return (out % (2 * math.pi)) * direction


def _theta_rescale_no_clip(
    x: np.ndarray,
    range_: Tuple[float, float],
    arc: Tuple[float, float] = (0, 2 * math.pi),
    direction: int = 1,
) -> np.ndarray:
    """Rescale theta without clipping."""
    x = np.asarray(x, dtype=float)
    return _rescale(x, to=arc, from_=range_) * direction


def _r_rescale(
    x: np.ndarray,
    range_: Tuple[float, float],
    donut: Tuple[float, float] = (0.0, 0.4),
) -> np.ndarray:
    """Rescale radius to donut range."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, range_[0], range_[1])
    return _rescale(x, to=donut, from_=range_)


def _parse_coord_expand(expand: Any) -> List[bool]:
    """Expand argument to a length-4 list of booleans (top, right, bottom, left)."""
    if isinstance(expand, bool):
        return [expand] * 4
    if isinstance(expand, (list, tuple)):
        result = list(expand)
        while len(result) < 4:
            result.append(result[-1] if result else True)
        return [bool(v) for v in result[:4]]
    return [True, True, True, True]


def _transform_position(
    data: pd.DataFrame,
    trans_x: Any = None,
    trans_y: Any = None,
) -> pd.DataFrame:
    """Apply transformation functions to position aesthetics.

    Parameters
    ----------
    data : pd.DataFrame
        Data to transform.
    trans_x, trans_y : callable, optional
        Transformation functions for x and y families.

    Returns
    -------
    pd.DataFrame
        Transformed data.
    """
    data = data.copy()
    x_cols = [c for c in data.columns if c in ("x", "xmin", "xmax", "xend", "xintercept")]
    y_cols = [c for c in data.columns if c in ("y", "ymin", "ymax", "yend", "yintercept")]
    if trans_x is not None:
        for c in x_cols:
            data[c] = trans_x(data[c].values)
    if trans_y is not None:
        for c in y_cols:
            data[c] = trans_y(data[c].values)
    return data


# ---------------------------------------------------------------------------
# Base Coord
# ---------------------------------------------------------------------------

class Coord(GGProto):
    """Base coordinate system.

    Attributes
    ----------
    default : bool
        Whether this is the default coordinate system.
    clip : str
        Clipping setting: ``"on"``, ``"off"``, or ``"inherit"``.
    reverse : str
        Which directions to reverse: ``"none"``, ``"x"``, ``"y"``, or ``"xy"``.
    """

    default: bool = False
    clip: str = "on"
    reverse: str = "none"

    # -- setup ---------------------------------------------------------------

    def setup_params(self, data: Any) -> Dict[str, Any]:
        """Modify or check parameters based on data.

        Parameters
        ----------
        data : list of DataFrames
            Global data followed by layer data.

        Returns
        -------
        dict
            Parameters, including parsed ``expand``.
        """
        expand = getattr(self, "expand", True)
        return {"expand": _parse_coord_expand(expand)}

    def setup_data(self, data: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        """Hook for modifying data before defaults are added.

        Parameters
        ----------
        data : list of DataFrames
        params : dict

        Returns
        -------
        list of DataFrames
        """
        return data

    def setup_layout(self, layout: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Hook for the coord to influence layout.

        Parameters
        ----------
        layout : pd.DataFrame
            Layout table with ``ROW``, ``COL``, ``PANEL``, ``SCALE_X``, ``SCALE_Y``.
        params : dict

        Returns
        -------
        pd.DataFrame
            Layout with an added ``COORD`` column.
        """
        layout = layout.copy()
        scales = layout[["SCALE_X", "SCALE_Y"]]
        unique_scales = scales.drop_duplicates().reset_index(drop=True)
        unique_scales = unique_scales.copy()
        unique_scales["COORD"] = range(1, len(unique_scales) + 1)
        layout = layout.drop(columns="COORD", errors="ignore")
        layout = pd.merge(layout, unique_scales, on=["SCALE_X", "SCALE_Y"], how="left")
        return layout

    # -- panel params --------------------------------------------------------

    def modify_scales(self, scales_x: list, scales_y: list) -> None:
        """Optionally modify scales in-place.

        Parameters
        ----------
        scales_x, scales_y : list
            Lists of trained x and y scales.
        """
        pass

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create panel parameters for one panel.

        Parameters
        ----------
        scale_x, scale_y : Scale
            Trained position scales.
        params : dict

        Returns
        -------
        dict
            Panel parameters including view scales and ranges.
        """
        return {}

    def setup_panel_guides(
        self,
        panel_params: Dict[str, Any],
        guides: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Initiate position guides for a panel.

        Parameters
        ----------
        panel_params : dict
            Output from ``setup_panel_params``.
        guides : Guides
            Guides ggproto.
        params : dict

        Returns
        -------
        dict
            ``panel_params`` with guides appended.
        """
        panel_params["guides"] = guides
        return panel_params

    def train_panel_guides(
        self,
        panel_params: Dict[str, Any],
        layers: list,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train and transform position guides.

        Parameters
        ----------
        panel_params : dict
        layers : list
        params : dict

        Returns
        -------
        dict
        """
        return panel_params

    # -- transform -----------------------------------------------------------

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        """Transform data coordinates to [0, 1] range.

        Parameters
        ----------
        data : pd.DataFrame
            Data with numeric position columns.
        panel_params : dict
            Panel parameters.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        cli_abort(f"{snake_class(self)} has not implemented transform().")

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        panel_params: Dict[str, Any],
    ) -> np.ndarray:
        """Compute distances between successive points.

        Parameters
        ----------
        x, y : array-like
        panel_params : dict

        Returns
        -------
        np.ndarray
        """
        cli_abort(f"{snake_class(self)} has not implemented distance().")

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ranges back to data coordinates.

        Parameters
        ----------
        panel_params : dict

        Returns
        -------
        dict
            With ``x`` and ``y`` ranges.
        """
        cli_abort(f"{snake_class(self)} has not implemented backtransform_range().")

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        """Extract x/y ranges from panel_params.

        Parameters
        ----------
        panel_params : dict

        Returns
        -------
        dict
            ``{"x": [lo, hi], "y": [lo, hi]}``.
        """
        cli_abort(f"{snake_class(self)} has not implemented range().")

    # -- render --------------------------------------------------------------

    def draw_panel(self, panel: Any, params: Dict[str, Any], theme: Any) -> Any:
        """Decorate panel with foreground and background.

        Parameters
        ----------
        panel : grob
        params : dict
        theme : Theme

        Returns
        -------
        grob
        """
        from grid_py import GTree, GList, Viewport
        fg = self.render_fg(params, theme)
        bg = self.render_bg(params, theme)
        children = [bg] + (list(panel) if isinstance(panel, (list, tuple)) else [panel]) + [fg]
        return GTree(
            children=GList(*children),
            vp=Viewport(clip=self.clip),
        )

    def render_fg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        """Render panel foreground.

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        grob
        """
        from grid_py import null_grob
        return null_grob()

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        """Render panel background.

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        grob
        """
        cli_abort(f"{snake_class(self)} has not implemented render_bg().")

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        """Render horizontal axes.

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        dict
            ``{"top": grob, "bottom": grob}``.
        """
        from grid_py import null_grob
        return {"top": null_grob(), "bottom": null_grob()}

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        """Render vertical axes.

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        dict
            ``{"left": grob, "right": grob}``.
        """
        from grid_py import null_grob
        return {"left": null_grob(), "right": null_grob()}

    def labels(self, labels: Dict[str, Any], panel_params: Dict[str, Any]) -> Dict[str, Any]:
        """Format axis labels.

        Parameters
        ----------
        labels : dict
            Label structure with ``x`` and ``y`` sub-dicts.
        panel_params : dict

        Returns
        -------
        dict
        """
        return labels

    def aspect(self, ranges: Any) -> Optional[float]:
        """Return the aspect ratio for panels.

        Parameters
        ----------
        ranges : dict
            Panel parameters.

        Returns
        -------
        float or None
        """
        return None

    # -- utilities -----------------------------------------------------------

    def is_linear(self) -> bool:
        """Whether this coordinate system is linear.

        Returns
        -------
        bool
        """
        return False

    def is_free(self) -> bool:
        """Whether this coord supports free-scaling in facets.

        Returns
        -------
        bool
        """
        return False


# ---------------------------------------------------------------------------
# CoordCartesian
# ---------------------------------------------------------------------------

class CoordCartesian(Coord):
    """Cartesian coordinate system.

    Attributes
    ----------
    limits : dict
        ``{"x": (lo, hi) or None, "y": (lo, hi) or None}``.
    ratio : float or None
        Aspect ratio ``y/x``.
    """

    limits: Dict[str, Any] = {"x": None, "y": None}
    ratio: Optional[float] = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def is_linear(self) -> bool:
        return True

    def is_free(self) -> bool:
        return self.ratio is None

    def aspect(self, ranges: Any) -> Optional[float]:
        """Compute aspect ratio from ranges.

        Parameters
        ----------
        ranges : dict
            Must have ``x.range`` and ``y.range`` or similar.

        Returns
        -------
        float or None
        """
        if self.ratio is None:
            return None
        y_range = ranges.get("y.range") or ranges.get("y_range", [0, 1])
        x_range = ranges.get("x.range") or ranges.get("x_range", [0, 1])
        return (y_range[1] - y_range[0]) / max(x_range[1] - x_range[0], 1e-10) * self.ratio

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        panel_params: Dict[str, Any],
    ) -> np.ndarray:
        """Euclidean distance normalised by the panel diagonal."""
        x_dim = panel_params.get("x_range") or panel_params.get("x.range", [0, 1])
        y_dim = panel_params.get("y_range") or panel_params.get("y.range", [0, 1])
        max_dist = np.sqrt((x_dim[1] - x_dim[0]) ** 2 + (y_dim[1] - y_dim[0]) ** 2)
        if max_dist == 0:
            max_dist = 1.0
        return _dist_euclidean(np.asarray(x), np.asarray(y)) / max_dist

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        x_range = panel_params.get("x_range") or panel_params.get("x.range", [0, 1])
        y_range = panel_params.get("y_range") or panel_params.get("y.range", [0, 1])
        return {"x": list(x_range), "y": list(y_range)}

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return self.range(panel_params)

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        """Rescale x/y into [0, 1].

        Parameters
        ----------
        data : pd.DataFrame
        panel_params : dict

        Returns
        -------
        pd.DataFrame
        """
        reverse = panel_params.get("reverse") or getattr(self, "reverse", "none")
        x_range = panel_params.get("x_range") or panel_params.get("x.range", [0, 1])
        y_range = panel_params.get("y_range") or panel_params.get("y.range", [0, 1])

        def rescale_x(vals: np.ndarray) -> np.ndarray:
            r = x_range
            if reverse in ("x", "xy"):
                r = list(reversed(r))
            return _rescale(vals, to=(0, 1), from_=tuple(r))

        def rescale_y(vals: np.ndarray) -> np.ndarray:
            r = y_range
            if reverse in ("y", "xy"):
                r = list(reversed(r))
            return _rescale(vals, to=(0, 1), from_=tuple(r))

        data = _transform_position(data, rescale_x, rescale_y)
        data = _transform_position(data, _squish_infinite, _squish_infinite)
        return data

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build panel parameters from scales.

        Parameters
        ----------
        scale_x, scale_y : Scale
        params : dict

        Returns
        -------
        dict
        """
        params = params or {}
        # Simple implementation: extract limits from scales
        x_limits = self.limits.get("x")
        y_limits = self.limits.get("y")

        if hasattr(scale_x, "get_limits"):
            x_range = list(scale_x.get_limits())
        else:
            x_range = list(x_limits or [0, 1])

        if hasattr(scale_y, "get_limits"):
            y_range = list(scale_y.get_limits())
        else:
            y_range = list(y_limits or [0, 1])

        # Apply coord limits as zoom
        if x_limits is not None:
            x_range = list(x_limits)
        if y_limits is not None:
            y_range = list(y_limits)

        return {
            "x_range": x_range,
            "y_range": y_range,
            "x.range": x_range,
            "y.range": y_range,
            "reverse": getattr(self, "reverse", "none"),
        }

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        """Render panel background (grid lines, background fill).

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        grob
        """
        from grid_py import null_grob
        return null_grob()

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"top": null_grob(), "bottom": null_grob()}

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"left": null_grob(), "right": null_grob()}


# ---------------------------------------------------------------------------
# CoordFixed
# ---------------------------------------------------------------------------

class CoordFixed(CoordCartesian):
    """Fixed-ratio Cartesian coordinate system.

    Attributes
    ----------
    ratio : float
        Aspect ratio (y per x unit).  Default is 1.
    """

    ratio: float = 1.0

    def is_free(self) -> bool:
        return False

    def aspect(self, ranges: Any) -> float:
        y_range = ranges.get("y.range") or ranges.get("y_range", [0, 1])
        x_range = ranges.get("x.range") or ranges.get("x_range", [0, 1])
        return (y_range[1] - y_range[0]) / max(x_range[1] - x_range[0], 1e-10) * self.ratio


# ---------------------------------------------------------------------------
# CoordFlip
# ---------------------------------------------------------------------------

class CoordFlip(CoordCartesian):
    """Flipped Cartesian coordinates (swap x and y)."""

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        data = _flip_axis_labels(data)
        return super().transform(data, panel_params)

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        r = self.range(panel_params)
        return r

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        un_flipped = super().range(panel_params)
        return {"x": un_flipped["y"], "y": un_flipped["x"]}

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        expand = params.get("expand", [True, True, True, True])
        if len(expand) >= 4:
            params["expand"] = [expand[1], expand[0], expand[3], expand[2]]
        pp = super().setup_panel_params(scale_x, scale_y, params)
        return _flip_axis_labels(pp) if isinstance(pp, dict) else pp

    def labels(self, labels: Dict[str, Any], panel_params: Dict[str, Any]) -> Dict[str, Any]:
        return _flip_axis_labels(labels)

    def setup_layout(self, layout: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        layout = super().setup_layout(layout, params)
        layout = layout.copy()
        layout[["SCALE_X", "SCALE_Y"]] = layout[["SCALE_Y", "SCALE_X"]].values
        return layout


def _flip_axis_labels(x: Any) -> Any:
    """Swap x/y prefixed names in a dict or DataFrame."""
    if isinstance(x, dict):
        new = {}
        for k, v in x.items():
            nk = k.replace("x", "__Z__").replace("y", "x").replace("__Z__", "y") if k.startswith(("x", "y")) else k
            new[nk] = v
        return new
    if isinstance(x, pd.DataFrame):
        rename_map = {}
        for c in x.columns:
            if c.startswith("x"):
                rename_map[c] = "y" + c[1:]
            elif c.startswith("y"):
                rename_map[c] = "x" + c[1:]
        return x.rename(columns=rename_map)
    return x


# ---------------------------------------------------------------------------
# CoordPolar
# ---------------------------------------------------------------------------

class CoordPolar(Coord):
    """Polar coordinate system.

    Attributes
    ----------
    theta : str
        Which variable to map to angle (``"x"`` or ``"y"``).
    r : str
        Which variable to map to radius.
    start : float
        Offset from 12 o'clock in radians.
    direction : int
        1 for clockwise, -1 for anticlockwise.
    """

    theta: str = "x"
    r: str = "y"
    start: float = 0.0
    direction: int = 1
    limits: Dict[str, Any] = {"x": None, "y": None}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.theta == "x":
            self.r = "y"
        else:
            self.r = "x"

    def aspect(self, ranges: Any) -> float:
        return 1.0

    def is_free(self) -> bool:
        return True

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        panel_params: Dict[str, Any],
        boost: float = 0.75,
    ) -> np.ndarray:
        arc = (self.start, self.start + 2 * math.pi)
        if self.theta == "x":
            r = _rescale(np.asarray(y), from_=tuple(panel_params.get("r.range", [0, 1])))
            theta = _theta_rescale_no_clip(
                np.asarray(x),
                tuple(panel_params.get("theta.range", [0, 1])),
                arc,
                self.direction,
            )
        else:
            r = _rescale(np.asarray(x), from_=tuple(panel_params.get("r.range", [0, 1])))
            theta = _theta_rescale_no_clip(
                np.asarray(y),
                tuple(panel_params.get("theta.range", [0, 1])),
                arc,
                self.direction,
            )
        return _dist_polar(r ** boost, theta)

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return self.range(panel_params)

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return {
            self.theta: list(panel_params.get("theta.range", [0, 1])),
            self.r: list(panel_params.get("r.range", [0, 1])),
        }

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        result: Dict[str, Any] = {}
        for name, scale in [("x", scale_x), ("y", scale_y)]:
            limits = self.limits.get(name)
            if hasattr(scale, "get_limits"):
                rng = list(scale.get_limits())
            else:
                rng = list(limits or [0, 1])
            if limits is not None:
                rng = list(limits)

            is_theta = (self.theta == name)
            prefix = "theta" if is_theta else "r"

            result[f"{prefix}.range"] = rng
            if hasattr(scale, "break_info"):
                info = scale.break_info(rng)
                result[f"{prefix}.major"] = info.get("major_source")
                result[f"{prefix}.minor"] = info.get("minor_source")
                result[f"{prefix}.labels"] = info.get("labels")
            else:
                result[f"{prefix}.major"] = None
                result[f"{prefix}.minor"] = None
                result[f"{prefix}.labels"] = None

        return result

    def setup_panel_guides(
        self,
        panel_params: Dict[str, Any],
        guides: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # CoordPolar cannot render standard guides
        return panel_params

    def train_panel_guides(
        self,
        panel_params: Dict[str, Any],
        layers: list,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return panel_params

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        arc = (self.start, self.start + 2 * math.pi)
        direction = self.direction
        data = data.copy()

        # Rename x/y to theta/r based on self.theta
        if self.theta == "x":
            theta_col, r_col = "x", "y"
        else:
            theta_col, r_col = "y", "x"

        r_range = panel_params.get("r.range", [0, 1])
        theta_range = panel_params.get("theta.range", [0, 1])

        if r_col in data.columns:
            data["__r__"] = _r_rescale(data[r_col].values, tuple(r_range))
        else:
            data["__r__"] = 0.0

        if theta_col in data.columns:
            data["__theta__"] = _theta_rescale(
                data[theta_col].values, tuple(theta_range), arc, direction
            )
        else:
            data["__theta__"] = 0.0

        data["x"] = data["__r__"] * np.sin(data["__theta__"]) + 0.5
        data["y"] = data["__r__"] * np.cos(data["__theta__"]) + 0.5
        data.drop(columns=["__r__", "__theta__"], inplace=True, errors="ignore")
        return data

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        from grid_py import null_grob
        return null_grob()

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"top": null_grob(), "bottom": null_grob()}

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"left": null_grob(), "right": null_grob()}

    def labels(self, labels: Dict[str, Any], panel_params: Dict[str, Any]) -> Dict[str, Any]:
        if self.theta == "y":
            return {"x": labels.get("y", {}), "y": labels.get("x", {})}
        return labels


# ---------------------------------------------------------------------------
# CoordRadial
# ---------------------------------------------------------------------------

class CoordRadial(Coord):
    """Modern radial (polar) coordinate system.

    Attributes
    ----------
    theta : str
        Angle variable (``"x"`` or ``"y"``).
    r : str
        Radius variable.
    arc : tuple of float
        Start and end of the arc in radians.
    r_axis_inside : bool or float or None
        Whether the r-axis is drawn inside the panel.
    rotate_angle : bool
        Whether to transform the ``angle`` aesthetic.
    inner_radius : tuple of float
        Inner and outer radius proportions.
    """

    theta: str = "x"
    r: str = "y"
    arc: Tuple[float, float] = (0.0, 2 * math.pi)
    r_axis_inside: Any = None
    rotate_angle: bool = False
    inner_radius: Tuple[float, float] = (0.0, 0.4)
    limits: Dict[str, Any] = {"theta": None, "r": None}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.theta == "x":
            self.r = "y"
        else:
            self.r = "x"

    def aspect(self, details: Any) -> float:
        bbox = details.get("bbox", {"x": [0, 1], "y": [0, 1]})
        dx = bbox["x"][1] - bbox["x"][0]
        dy = bbox["y"][1] - bbox["y"][0]
        return dy / max(dx, 1e-10)

    def is_free(self) -> bool:
        return True

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        details: Dict[str, Any],
        boost: float = 0.75,
    ) -> np.ndarray:
        arc = details.get("arc") or self.arc
        inner = self.inner_radius
        if self.theta == "x":
            r = _rescale(np.asarray(y), from_=tuple(details.get("r.range", [0, 1])),
                         to=(inner[0] / 0.4, inner[1] / 0.4))
            theta = _theta_rescale_no_clip(np.asarray(x), tuple(details.get("theta.range", [0, 1])), arc)
        else:
            r = _rescale(np.asarray(x), from_=tuple(details.get("r.range", [0, 1])),
                         to=(inner[0] / 0.4, inner[1] / 0.4))
            theta = _theta_rescale_no_clip(np.asarray(y), tuple(details.get("theta.range", [0, 1])), arc)
        return _dist_polar(r ** boost, theta)

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return self.range(panel_params)

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return {
            self.theta: list(panel_params.get("theta.range", [0, 1])),
            self.r: list(panel_params.get("r.range", [0, 1])),
        }

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        result: Dict[str, Any] = {}

        if self.theta == "x":
            theta_limits = self.limits.get("theta")
            r_limits = self.limits.get("r")
            theta_scale, r_scale = scale_x, scale_y
        else:
            theta_limits = self.limits.get("theta")
            r_limits = self.limits.get("r")
            theta_scale, r_scale = scale_y, scale_x

        # Theta
        if hasattr(theta_scale, "get_limits"):
            theta_range = list(theta_scale.get_limits())
        else:
            theta_range = list(theta_limits or [0, 1])
        if theta_limits is not None:
            theta_range = list(theta_limits)

        # R
        if hasattr(r_scale, "get_limits"):
            r_range = list(r_scale.get_limits())
        else:
            r_range = list(r_limits or [0, 1])
        if r_limits is not None:
            r_range = list(r_limits)

        result["theta.range"] = theta_range
        result["r.range"] = r_range
        result["bbox"] = _polar_bbox(self.arc, inner_radius=self.inner_radius)
        result["arc"] = self.arc
        result["inner_radius"] = self.inner_radius

        return result

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        bbox = panel_params.get("bbox", {"x": [0, 1], "y": [0, 1]})
        arc = panel_params.get("arc", self.arc)
        inner_radius = panel_params.get("inner_radius", self.inner_radius)

        if self.theta == "x":
            theta_col, r_col = "x", "y"
        else:
            theta_col, r_col = "y", "x"

        r_range = panel_params.get("r.range", [0, 1])
        theta_range = panel_params.get("theta.range", [0, 1])

        if r_col in data.columns:
            data["__r__"] = _r_rescale(data[r_col].values, tuple(r_range), donut=inner_radius)
        else:
            data["__r__"] = 0.0

        if theta_col in data.columns:
            data["__theta__"] = _theta_rescale(
                data[theta_col].values, tuple(theta_range), arc
            )
        else:
            data["__theta__"] = 0.0

        raw_x = data["__r__"] * np.sin(data["__theta__"]) + 0.5
        raw_y = data["__r__"] * np.cos(data["__theta__"]) + 0.5
        data["x"] = _rescale(raw_x, from_=tuple(bbox["x"]))
        data["y"] = _rescale(raw_y, from_=tuple(bbox["y"]))
        data.drop(columns=["__r__", "__theta__"], inplace=True, errors="ignore")
        return data

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        from grid_py import null_grob
        return null_grob()

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"top": null_grob(), "bottom": null_grob()}

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"left": null_grob(), "right": null_grob()}

    def labels(self, labels: Dict[str, Any], panel_params: Dict[str, Any]) -> Dict[str, Any]:
        if self.theta == "y":
            return {"x": labels.get("y", {}), "y": labels.get("x", {})}
        return labels


def _polar_bbox(
    arc: Tuple[float, float],
    margin: Tuple[float, float, float, float] = (0.05, 0.05, 0.05, 0.05),
    inner_radius: Tuple[float, float] = (0.0, 0.4),
) -> Dict[str, list]:
    """Compute bounding box for a partial polar chart.

    Parameters
    ----------
    arc : tuple of float
        Start and end angles in radians.
    margin : tuple
        Margins (top, right, bottom, left).
    inner_radius : tuple
        Inner and outer radii.

    Returns
    -------
    dict
        ``{"x": [xmin, xmax], "y": [ymin, ymax]}``.
    """
    if abs(arc[1] - arc[0]) >= 2 * math.pi:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    sorted_arc = (min(arc), max(arc))
    angles = np.array([sorted_arc[0], sorted_arc[1]])
    x_outer = 0.5 * np.sin(angles) + 0.5
    y_outer = 0.5 * np.cos(angles) + 0.5

    # Check cardinal directions
    cardinal = np.array([0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
    in_sector = _in_arc(cardinal, sorted_arc)

    # top, right, bottom, left extremes
    bounds = [
        1.0 if in_sector[0] else max(float(np.max(y_outer)), 0.5 + margin[0]),
        1.0 if in_sector[1] else max(float(np.max(x_outer)), 0.5 + margin[1]),
        0.0 if in_sector[2] else min(float(np.min(y_outer)), 0.5 - margin[2]),
        0.0 if in_sector[3] else min(float(np.min(x_outer)), 0.5 - margin[3]),
    ]
    return {"x": [bounds[3], bounds[1]], "y": [bounds[2], bounds[0]]}


def _in_arc(theta: np.ndarray, arc: Tuple[float, float]) -> np.ndarray:
    """Test whether angles are inside an arc."""
    theta = np.asarray(theta)
    if abs(arc[1] - arc[0]) >= 2 * math.pi - 1e-8:
        return np.ones(len(theta), dtype=bool)
    a0 = arc[0] % (2 * math.pi)
    a1 = arc[1] % (2 * math.pi)
    if a0 < a1:
        return (theta >= a0) & (theta <= a1)
    else:
        return ~((theta < a0) & (theta > a1))


# ---------------------------------------------------------------------------
# CoordTransform
# ---------------------------------------------------------------------------

class CoordTransform(Coord):
    """Transformed Cartesian coordinate system.

    Applies arbitrary transformations to x and y.

    Attributes
    ----------
    trans : dict
        ``{"x": transform, "y": transform}`` where each transform has
        ``transform()`` and ``inverse()`` methods.
    limits : dict
        Coordinate limits.
    """

    trans: Dict[str, Any] = {"x": None, "y": None}
    limits: Dict[str, Any] = {"x": None, "y": None}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def is_free(self) -> bool:
        return True

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        panel_params: Dict[str, Any],
    ) -> np.ndarray:
        x_range = panel_params.get("x.range", [0, 1])
        y_range = panel_params.get("y.range", [0, 1])
        max_dist = np.sqrt((x_range[1] - x_range[0]) ** 2 + (y_range[1] - y_range[0]) ** 2)
        if max_dist == 0:
            max_dist = 1.0
        tx = self.trans["x"].transform(np.asarray(x)) if self.trans.get("x") else np.asarray(x)
        ty = self.trans["y"].transform(np.asarray(y)) if self.trans.get("y") else np.asarray(y)
        return _dist_euclidean(tx, ty) / max_dist

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        x_range = panel_params.get("x.range", [0, 1])
        y_range = panel_params.get("y.range", [0, 1])
        inv_x = self.trans["x"].inverse(np.array(x_range)) if self.trans.get("x") else x_range
        inv_y = self.trans["y"].inverse(np.array(y_range)) if self.trans.get("y") else y_range
        return {"x": list(inv_x), "y": list(inv_y)}

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return {
            "x": list(panel_params.get("x.range", [0, 1])),
            "y": list(panel_params.get("y.range", [0, 1])),
        }

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        reverse = panel_params.get("reverse") or getattr(self, "reverse", "none")
        x_range = list(panel_params.get("x.range", [0, 1]))
        y_range = list(panel_params.get("y.range", [0, 1]))

        if reverse in ("x", "xy"):
            x_range = list(reversed(x_range))
        if reverse in ("y", "xy"):
            y_range = list(reversed(y_range))

        trans_x = self.trans.get("x")
        trans_y = self.trans.get("y")

        def apply_trans_x(vals: np.ndarray) -> np.ndarray:
            vals = np.asarray(vals, dtype=float)
            finite = np.isfinite(vals)
            if trans_x is not None and np.any(finite):
                vals[finite] = _rescale(
                    trans_x.transform(vals[finite]),
                    to=(0, 1),
                    from_=tuple(x_range),
                )
            else:
                vals = _rescale(vals, to=(0, 1), from_=tuple(x_range))
            return vals

        def apply_trans_y(vals: np.ndarray) -> np.ndarray:
            vals = np.asarray(vals, dtype=float)
            finite = np.isfinite(vals)
            if trans_y is not None and np.any(finite):
                vals[finite] = _rescale(
                    trans_y.transform(vals[finite]),
                    to=(0, 1),
                    from_=tuple(y_range),
                )
            else:
                vals = _rescale(vals, to=(0, 1), from_=tuple(y_range))
            return vals

        data = _transform_position(data, apply_trans_x, apply_trans_y)
        data = _transform_position(data, _squish_infinite, _squish_infinite)
        return data

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        x_limits = self.limits.get("x")
        y_limits = self.limits.get("y")

        if hasattr(scale_x, "get_limits"):
            x_range = list(scale_x.get_limits())
        else:
            x_range = list(x_limits or [0, 1])
        if x_limits is not None:
            x_range = list(x_limits)

        if hasattr(scale_y, "get_limits"):
            y_range = list(scale_y.get_limits())
        else:
            y_range = list(y_limits or [0, 1])
        if y_limits is not None:
            y_range = list(y_limits)

        return {
            "x_range": x_range,
            "y_range": y_range,
            "x.range": x_range,
            "y.range": y_range,
            "reverse": getattr(self, "reverse", "none"),
        }

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        from grid_py import null_grob
        return null_grob()


# Alias for backward compatibility
CoordTrans = CoordTransform


# ---------------------------------------------------------------------------
# coord_munch
# ---------------------------------------------------------------------------

def coord_munch(
    coord: Coord,
    data: pd.DataFrame,
    range_: Dict[str, Any],
    n: int = 50,
    is_closed: bool = False,
) -> pd.DataFrame:
    """Interpolate path data for non-linear coordinate systems.

    For linear coordinates, the data is returned unchanged (after
    transformation).  For non-linear coordinates, points are
    interpolated so that straight lines in data space become curves
    in plot space.

    Parameters
    ----------
    coord : Coord
        Coordinate system.
    data : pd.DataFrame
        Data with ``x`` and ``y`` columns (at minimum).
    range_ : dict
        Panel parameters / ranges.
    n : int
        Maximum number of interpolation points per segment.
    is_closed : bool
        Whether the path is closed (polygon).

    Returns
    -------
    pd.DataFrame
        Transformed (and possibly interpolated) data.
    """
    if coord.is_linear():
        return coord.transform(data, range_)

    # For non-linear coords, interpolate
    if len(data) < 2:
        return coord.transform(data, range_)

    # Compute distances to determine segment counts
    x = data["x"].values
    y = data["y"].values
    dist = coord.distance(x, y, range_)

    # Interpolate segments that are long
    if len(dist) == 0:
        return coord.transform(data, range_)

    # Determine how many points each segment needs
    max_dist = float(np.nanmax(dist)) if len(dist) > 0 else 0.0
    if max_dist == 0:
        return coord.transform(data, range_)

    # Simple approach: subdivide each segment proportionally
    segments = np.ceil(dist / max_dist * n).astype(int)
    segments = np.clip(segments, 1, n)

    rows = []
    for i in range(len(data) - 1):
        nseg = int(segments[i]) if i < len(segments) else 1
        row_start = data.iloc[i]
        row_end = data.iloc[i + 1]
        for j in range(nseg):
            t = j / nseg
            new_row = {}
            for col in data.columns:
                v0 = row_start[col]
                v1 = row_end[col]
                if isinstance(v0, (int, float, np.integer, np.floating)):
                    new_row[col] = v0 + (v1 - v0) * t
                else:
                    new_row[col] = v0
            rows.append(new_row)
    # Last point
    rows.append(dict(data.iloc[-1]))

    munched = pd.DataFrame(rows)
    return coord.transform(munched, range_)


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------

def coord_cartesian(
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    expand: Union[bool, List[bool]] = True,
    default: bool = False,
    clip: str = "on",
    reverse: str = "none",
    ratio: Optional[float] = None,
) -> CoordCartesian:
    """Create a Cartesian coordinate system.

    Parameters
    ----------
    xlim, ylim : sequence of float or None
        Limits for zooming (does not filter data).
    expand : bool or list of bool
        Whether to expand limits to avoid data/axis overlap.
    default : bool
        Whether this is the default coord.
    clip : str
        Clipping: ``"on"`` or ``"off"``.
    reverse : str
        ``"none"``, ``"x"``, ``"y"``, or ``"xy"``.
    ratio : float or None
        Fixed aspect ratio.

    Returns
    -------
    CoordCartesian
    """
    return CoordCartesian(
        limits={"x": list(xlim) if xlim is not None else None,
                "y": list(ylim) if ylim is not None else None},
        expand=expand,
        default=default,
        clip=clip,
        reverse=reverse,
        ratio=ratio,
    )


def coord_fixed(ratio: float = 1.0, **kwargs: Any) -> CoordFixed:
    """Create a fixed-ratio coordinate system.

    Parameters
    ----------
    ratio : float
        Aspect ratio (y/x).
    **kwargs
        Passed to :class:`CoordFixed`.

    Returns
    -------
    CoordFixed
    """
    obj = CoordFixed(ratio=ratio, **kwargs)
    if "limits" not in kwargs:
        obj.limits = {
            "x": kwargs.get("xlim"),
            "y": kwargs.get("ylim"),
        }
    return obj


coord_equal = coord_fixed


def coord_flip(
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    expand: Union[bool, List[bool]] = True,
    clip: str = "on",
) -> CoordFlip:
    """Create a flipped Cartesian coordinate system.

    Parameters
    ----------
    xlim, ylim : sequence of float or None
    expand : bool or list
    clip : str

    Returns
    -------
    CoordFlip
    """
    return CoordFlip(
        limits={"x": list(xlim) if xlim is not None else None,
                "y": list(ylim) if ylim is not None else None},
        expand=expand,
        clip=clip,
    )


def coord_polar(
    theta: str = "x",
    start: float = 0.0,
    direction: int = 1,
    clip: str = "on",
) -> CoordPolar:
    """Create a polar coordinate system.

    Parameters
    ----------
    theta : str
        ``"x"`` or ``"y"``.
    start : float
        Offset from 12 o'clock in radians.
    direction : int
        1 for clockwise, -1 for anticlockwise.
    clip : str

    Returns
    -------
    CoordPolar
    """
    if theta not in ("x", "y"):
        cli_abort("theta must be 'x' or 'y'.")
    return CoordPolar(
        theta=theta,
        start=start,
        direction=int(np.sign(direction)),
        clip=clip,
    )


def coord_radial(
    theta: str = "x",
    start: float = 0.0,
    end: Optional[float] = None,
    thetalim: Optional[Sequence[float]] = None,
    rlim: Optional[Sequence[float]] = None,
    expand: Union[bool, List[bool]] = True,
    clip: str = "off",
    r_axis_inside: Any = None,
    rotate_angle: bool = False,
    inner_radius: float = 0.0,
    reverse: str = "none",
) -> CoordRadial:
    """Create a radial coordinate system.

    Parameters
    ----------
    theta : str
        ``"x"`` or ``"y"``.
    start : float
        Start angle in radians.
    end : float or None
        End angle.  Defaults to ``start + 2*pi``.
    thetalim, rlim : sequence or None
        Limits for theta and r.
    expand : bool or list
    clip : str
    r_axis_inside : bool, float, or None
    rotate_angle : bool
    inner_radius : float
        Between 0 and 1.
    reverse : str
        ``"none"``, ``"theta"``, ``"r"``, or ``"thetar"``.

    Returns
    -------
    CoordRadial
    """
    if theta not in ("x", "y"):
        cli_abort("theta must be 'x' or 'y'.")
    if reverse not in ("none", "theta", "r", "thetar"):
        cli_abort("reverse must be 'none', 'theta', 'r', or 'thetar'.")

    arc_end = end if end is not None else (start + 2 * math.pi)
    arc = (start, arc_end)

    if arc[0] > arc[1]:
        n_rot = int((arc[0] - arc[1]) // (2 * math.pi)) + 1
        arc = (arc[0] - n_rot * 2 * math.pi, arc[1])

    if reverse in ("theta", "thetar"):
        arc = (arc[1], arc[0])

    inner = (inner_radius, 1.0)
    inner = (inner[0] * 0.4, inner[1] * 0.4)
    if reverse in ("r", "thetar"):
        inner = (inner[1], inner[0])

    return CoordRadial(
        theta=theta,
        arc=arc,
        limits={"theta": list(thetalim) if thetalim is not None else None,
                "r": list(rlim) if rlim is not None else None},
        expand=expand,
        clip=clip,
        r_axis_inside=r_axis_inside,
        rotate_angle=rotate_angle,
        inner_radius=inner,
        reverse=reverse,
    )


def coord_transform(
    x: Any = "identity",
    y: Any = "identity",
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    clip: str = "on",
    expand: Union[bool, List[bool]] = True,
    reverse: str = "none",
) -> CoordTransform:
    """Create a transformed coordinate system.

    Parameters
    ----------
    x, y : str or transform
        Transformations for x and y.
    xlim, ylim : sequence or None
    clip : str
    expand : bool or list
    reverse : str

    Returns
    -------
    CoordTransform
    """
    from scales import as_transform

    if isinstance(x, str):
        x = as_transform(x)
    if isinstance(y, str):
        y = as_transform(y)

    return CoordTransform(
        trans={"x": x, "y": y},
        limits={"x": list(xlim) if xlim is not None else None,
                "y": list(ylim) if ylim is not None else None},
        expand=expand,
        reverse=reverse,
        clip=clip,
    )


def coord_trans(**kwargs: Any) -> CoordTransform:
    """Deprecated alias for :func:`coord_transform`.

    Parameters
    ----------
    **kwargs
        Passed to :func:`coord_transform`.

    Returns
    -------
    CoordTransform
    """
    cli_warn("coord_trans() is deprecated; use coord_transform().")
    return coord_transform(**kwargs)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def is_coord(x: Any) -> bool:
    """Test whether *x* is a Coord.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
    """
    return isinstance(x, Coord)


def is_Coord(x: Any) -> bool:
    """Deprecated alias for :func:`is_coord`.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
    """
    return is_coord(x)

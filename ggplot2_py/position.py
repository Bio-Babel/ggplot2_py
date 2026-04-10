"""
Position adjustments for ggplot2.

Position adjustments control how overlapping geoms are arranged.
Each position is a GGProto object with ``setup_params``,
``setup_data``, ``compute_layer``, and ``compute_panel`` methods.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn
from ggplot2_py.ggproto import GGProto, ggproto
from ggplot2_py._utils import snake_class, compact, empty

__all__ = [
    "Position",
    "PositionIdentity",
    "PositionDodge",
    "PositionDodge2",
    "PositionJitter",
    "PositionJitterdodge",
    "PositionNudge",
    "PositionStack",
    "PositionFill",
    "position_identity",
    "position_dodge",
    "position_dodge2",
    "position_jitter",
    "position_jitterdodge",
    "position_nudge",
    "position_stack",
    "position_fill",
    "is_position",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transform_position(
    df: pd.DataFrame,
    trans_x: Optional[Callable] = None,
    trans_y: Optional[Callable] = None,
) -> pd.DataFrame:
    """Apply transformation functions to position aesthetics.

    Parameters
    ----------
    df : pd.DataFrame
    trans_x, trans_y : callable or None

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    x_cols = [c for c in df.columns if c in ("x", "xmin", "xmax", "xend", "xintercept")]
    y_cols = [c for c in df.columns if c in ("y", "ymin", "ymax", "yend", "yintercept")]
    if trans_x is not None:
        for c in x_cols:
            df[c] = trans_x(df[c].values)
    if trans_y is not None:
        for c in y_cols:
            df[c] = trans_y(df[c].values)
    return df


def _check_required_aesthetics(
    required: Sequence[str],
    present: Sequence[str],
    name: str,
) -> None:
    """Check that required aesthetics are present.

    Parameters
    ----------
    required : sequence of str
        Aesthetic names, possibly with ``|`` for alternatives.
    present : sequence of str
    name : str
        Name of the component for error messages.

    Raises
    ------
    ValueError
        If a required aesthetic is missing.
    """
    present_set = set(present)
    for req in required:
        alternatives = req.split("|")
        if not any(a in present_set for a in alternatives):
            cli_abort(f"{name} requires the following missing aesthetics: {req}")


def _resolution(x: np.ndarray, zero: bool = True) -> float:
    """Compute the resolution of a numeric vector.

    Parameters
    ----------
    x : array-like
    zero : bool

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 1.0
    unique_vals = np.unique(x)
    if len(unique_vals) < 2:
        return 1.0
    diffs = np.diff(np.sort(unique_vals))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    res = float(np.min(diffs))
    if zero:
        res = min(res, abs(float(unique_vals[0]))) if unique_vals[0] != 0 else res
    return res


def _collide(
    data: pd.DataFrame,
    width: Optional[float],
    name: str,
    strategy: Callable,
    reverse: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Set up and execute a collision strategy (dodge, stack).

    Parameters
    ----------
    data : pd.DataFrame
    width : float or None
    name : str
    strategy : callable
    reverse : bool
    **kwargs
        Extra args for the strategy function.

    Returns
    -------
    pd.DataFrame
    """
    data = data.copy()

    # Determine width
    if width is not None:
        if "xmin" not in data.columns or "xmax" not in data.columns:
            data["xmin"] = data["x"] - width / 2
            data["xmax"] = data["x"] + width / 2
    else:
        if "xmin" not in data.columns or "xmax" not in data.columns:
            data["xmin"] = data["x"]
            data["xmax"] = data["x"]
        widths = (data["xmax"] - data["xmin"]).dropna().unique()
        width = widths[0] if len(widths) > 0 else 0.0

    # Sort
    if reverse:
        data = data.sort_values(["xmin", "group"], ascending=[True, True]).reset_index(drop=True)
    else:
        data = data.sort_values(["xmin", "group"], ascending=[True, False]).reset_index(drop=True)

    original_order = data.index.copy()

    # Apply strategy per xmin group
    if "ymax" in data.columns:
        groups = data.groupby("xmin", sort=False)
        parts = []
        for _, grp in groups:
            parts.append(strategy(grp.copy(), width, **kwargs))
        data = pd.concat(parts, ignore_index=True) if parts else data
    elif "y" in data.columns:
        data["ymax"] = data["y"]
        groups = data.groupby("xmin", sort=False)
        parts = []
        for _, grp in groups:
            parts.append(strategy(grp.copy(), width, **kwargs))
        data = pd.concat(parts, ignore_index=True) if parts else data
        data["y"] = data["ymax"]

    return data


# ---------------------------------------------------------------------------
# Base Position
# ---------------------------------------------------------------------------

class Position(GGProto):
    """Base position adjustment class.

    Attributes
    ----------
    required_aes : tuple of str
        Aesthetics required for this position.
    default_aes : dict
        Default aesthetic values.
    """

    required_aes: Tuple[str, ...] = ()
    default_aes: Dict[str, Any] = {}

    def use_defaults(
        self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Fill in default position aesthetics.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict
            Fixed aesthetic params from the layer.

        Returns
        -------
        pd.DataFrame
        """
        if empty(data):
            return data

        params = params or {}
        aes_names = self.aesthetics()

        # Filter params to only position aesthetics not already in data
        relevant = {k: v for k, v in params.items() if k in aes_names and k not in data.columns}
        defaults = {
            k: v for k, v in self.default_aes.items()
            if k not in data.columns and k not in relevant
        }

        if not relevant and not defaults:
            return data

        data = data.copy()
        for k, v in defaults.items():
            if callable(v):
                data[k] = v(data)
            elif np.isscalar(v):
                data[k] = v
        for k, v in relevant.items():
            if np.isscalar(v) or (hasattr(v, "__len__") and len(v) == 1):
                data[k] = v
            elif hasattr(v, "__len__") and len(v) == len(data):
                data[k] = v
        return data

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Modify or validate parameters.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        dict
        """
        return {}

    def setup_data(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Modify or validate data.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        _check_required_aesthetics(self.required_aes, data.columns, snake_class(self))
        return data

    def compute_layer(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        layout: Any,
    ) -> pd.DataFrame:
        """Apply position adjustment across all panels.

        Splits data by ``PANEL`` and delegates to ``compute_panel``.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict
        layout : Layout

        Returns
        -------
        pd.DataFrame
        """
        if empty(data):
            return data

        panels = []
        for panel_id, panel_data in data.groupby("PANEL", sort=False, observed=True):
            if len(panel_data) == 0:
                continue
            scales = None
            if hasattr(layout, "get_scales"):
                scales = layout.get_scales(panel_id)
            result = self.compute_panel(
                data=panel_data.copy(),
                params=params,
                scales=scales,
            )
            panels.append(result)

        if panels:
            return pd.concat(panels, ignore_index=True)
        return data

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        """Apply position adjustment for one panel.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict
        scales : dict or None

        Returns
        -------
        pd.DataFrame
        """
        cli_abort(f"{snake_class(self)} has not implemented compute_panel().")

    def aesthetics(self) -> List[str]:
        """List position aesthetics.

        Returns
        -------
        list of str
        """
        required = list(self.required_aes) if self.required_aes else []
        # Expand pipe-separated alternatives
        expanded = []
        for r in required:
            expanded.extend(r.split("|"))
        return list(set(expanded) | set(self.default_aes.keys()))


# ---------------------------------------------------------------------------
# PositionIdentity
# ---------------------------------------------------------------------------

class PositionIdentity(Position):
    """No position adjustment (pass-through)."""

    def compute_layer(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        layout: Any,
    ) -> pd.DataFrame:
        return data

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        return data


# ---------------------------------------------------------------------------
# PositionDodge
# ---------------------------------------------------------------------------

class PositionDodge(Position):
    """Dodge overlapping elements side-to-side.

    Attributes
    ----------
    width : float or None
        Dodging width.
    preserve : str
        ``"total"`` or ``"single"``.
    orientation : str
        ``"x"`` or ``"y"``.
    reverse : bool
        Whether to reverse dodge order.
    """

    width: Optional[float] = None
    preserve: str = "total"
    orientation: str = "x"
    reverse: bool = False
    default_aes: Dict[str, Any] = {"order": None}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Set up dodge parameters.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        dict
        """
        flipped = self.orientation == "y"
        n = None
        if self.preserve == "single" and "group" in data.columns:
            # Count max groups per position
            if "x" in data.columns:
                n = data.groupby("x")["group"].nunique().max()
            elif "xmin" in data.columns:
                n = data.groupby("xmin")["group"].nunique().max()
            if n is not None:
                n = int(n)

        return {
            "width": self.width,
            "n": n,
            "flipped_aes": flipped,
            "reverse": self.reverse,
        }

    def setup_data(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        data = data.copy()
        if "x" not in data.columns and "xmin" in data.columns and "xmax" in data.columns:
            data["x"] = (data["xmin"] + data["xmax"]) / 2
        return data

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        """Dodge elements within a panel.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict
        scales : ignored

        Returns
        -------
        pd.DataFrame
        """
        return _pos_dodge(data, params.get("width"), n=params.get("n"))


def _pos_dodge(
    df: pd.DataFrame,
    width: Optional[float] = None,
    n: Optional[int] = None,
) -> pd.DataFrame:
    """Core dodge algorithm.

    Parameters
    ----------
    df : pd.DataFrame
    width : float or None
    n : int or None

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if "group" not in df.columns:
        return df

    if n is None:
        n = df["group"].nunique()
    if n <= 1:
        return df

    if "xmin" not in df.columns or "xmax" not in df.columns:
        df["xmin"] = df["x"]
        df["xmax"] = df["x"]

    d_width = float((df["xmax"] - df["xmin"]).max())
    if width is None:
        width = d_width

    # Map groups to 0..n-1
    unique_groups = np.sort(df["group"].unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    group_idx = df["group"].map(group_map).values

    df["x"] = df["x"].values + width * ((group_idx + 0.5) / n - 0.5)
    df["xmin"] = df["x"] - d_width / n / 2
    df["xmax"] = df["x"] + d_width / n / 2

    return df


# ---------------------------------------------------------------------------
# PositionDodge2
# ---------------------------------------------------------------------------

class PositionDodge2(PositionDodge):
    """Dodge with variable widths.

    Attributes
    ----------
    padding : float
        Proportion of space between elements (0 to 1).
    group_row : str
        ``"single"`` or ``"many"``.
    """

    padding: float = 0.1
    group_row: str = "single"

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        n = None
        if self.preserve == "single":
            if "x" in data.columns:
                groups = data.groupby(["PANEL", "x"]).ngroups
            else:
                groups = 1
            n = int(groups) if groups else None

        return {
            "width": self.width,
            "n": n,
            "padding": self.padding,
            "reverse": self.reverse,
            "flipped_aes": False,
            "group_row": self.group_row,
        }

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        return _pos_dodge2(
            data,
            params.get("width"),
            n=params.get("n"),
            padding=params.get("padding", 0.1),
        )


def _pos_dodge2(
    df: pd.DataFrame,
    width: Optional[float] = None,
    n: Optional[int] = None,
    padding: float = 0.1,
) -> pd.DataFrame:
    """Core dodge2 algorithm.

    Parameters
    ----------
    df : pd.DataFrame
    width : float or None
    n : int or None
    padding : float

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if "xmin" not in df.columns or "xmax" not in df.columns:
        if "x" in df.columns:
            df["xmin"] = df["x"]
            df["xmax"] = df["x"]
        else:
            return df

    # Find overlapping groups
    if n is None and "group" in df.columns:
        n = df["group"].nunique()

    if n is None or n <= 1:
        return df

    original_width = df["xmax"] - df["xmin"]
    new_width = original_width / n

    if "group" in df.columns:
        unique_groups = np.sort(df["group"].unique())
        group_map = {g: i for i, g in enumerate(unique_groups)}
        group_idx = df["group"].map(group_map).values
    else:
        group_idx = np.zeros(len(df), dtype=int)

    center = (df["xmin"] + df["xmax"]) / 2
    total_width = new_width * n
    start = center - total_width / 2

    df["xmin"] = start + group_idx * new_width
    df["xmax"] = df["xmin"] + new_width
    df["x"] = (df["xmin"] + df["xmax"]) / 2

    # Apply padding
    if padding > 0:
        pad_width = new_width * (1 - padding)
        df["xmin"] = df["x"] - pad_width / 2
        df["xmax"] = df["x"] + pad_width / 2

    return df


# ---------------------------------------------------------------------------
# PositionJitter
# ---------------------------------------------------------------------------

class PositionJitter(Position):
    """Random jitter.

    Attributes
    ----------
    width : float or None
        Jitter width (each direction).
    height : float or None
        Jitter height (each direction).
    seed : int or None
        Random seed for reproducibility.
    """

    width: Optional[float] = None
    height: Optional[float] = None
    seed: Any = None  # NA -> random, None -> don't reset
    required_aes: Tuple[str, ...] = ("x", "y")

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        seed = self.seed
        if seed is not None and (isinstance(seed, float) and np.isnan(seed)):
            seed = np.random.randint(0, 2 ** 31)
        return {
            "width": self.width,
            "height": self.height,
            "seed": seed,
        }

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        return _compute_jitter(
            data,
            width=params.get("width"),
            height=params.get("height"),
            seed=params.get("seed"),
        )


def _compute_jitter(
    data: pd.DataFrame,
    width: Optional[float] = None,
    height: Optional[float] = None,
    seed: Any = None,
) -> pd.DataFrame:
    """Apply jitter to data.

    Parameters
    ----------
    data : pd.DataFrame
    width, height : float or None
    seed : int or None

    Returns
    -------
    pd.DataFrame
    """
    data = data.copy()
    n = len(data)

    if width is None:
        width = _resolution(data["x"].values, zero=False) * 0.4 if "x" in data.columns else 0.0
    if height is None:
        height = _resolution(data["y"].values, zero=False) * 0.4 if "y" in data.columns else 0.0

    rng = np.random.RandomState(seed) if seed is not None else np.random

    if width > 0 and "x" in data.columns:
        x_jit = rng.uniform(-width, width, size=n)
        data["x"] = data["x"].values + x_jit
        for c in ("xmin", "xmax", "xend"):
            if c in data.columns:
                data[c] = data[c].values + x_jit

    if height > 0 and "y" in data.columns:
        y_jit = rng.uniform(-height, height, size=n)
        data["y"] = data["y"].values + y_jit
        for c in ("ymin", "ymax", "yend"):
            if c in data.columns:
                data[c] = data[c].values + y_jit

    return data


# ---------------------------------------------------------------------------
# PositionJitterdodge
# ---------------------------------------------------------------------------

class PositionJitterdodge(Position):
    """Simultaneously dodge and jitter.

    Attributes
    ----------
    jitter_width : float or None
    jitter_height : float
    dodge_width : float
    preserve : str
    reverse : bool
    seed : int or None
    """

    jitter_width: Optional[float] = None
    jitter_height: float = 0.0
    dodge_width: float = 0.75
    preserve: str = "total"
    reverse: bool = False
    seed: Any = None
    required_aes: Tuple[str, ...] = ("x", "y")

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        n = None
        if self.preserve == "single" and "group" in data.columns and "x" in data.columns:
            n = int(data.groupby(["PANEL", "x"])["group"].nunique().max())

        jw = self.jitter_width
        if jw is None and "x" in data.columns:
            jw = _resolution(data["x"].values, zero=False) * 0.4
        if jw is None:
            jw = 0.0
        jw = jw / max(n or 1, 1)

        return {
            "dodge_width": self.dodge_width,
            "jitter_width": jw,
            "jitter_height": self.jitter_height,
            "n": n,
            "seed": self.seed,
            "reverse": self.reverse,
        }

    def setup_data(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        data = data.copy()
        if "x" not in data.columns and "xmin" in data.columns and "xmax" in data.columns:
            data["x"] = (data["xmin"] + data["xmax"]) / 2
        return data

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        # First dodge
        data = _pos_dodge(data, params.get("dodge_width"), n=params.get("n"))
        # Then jitter
        data = _compute_jitter(
            data,
            width=params.get("jitter_width"),
            height=params.get("jitter_height"),
            seed=params.get("seed"),
        )
        return data


# ---------------------------------------------------------------------------
# PositionNudge
# ---------------------------------------------------------------------------

class PositionNudge(Position):
    """Constant offset in x and/or y.

    Attributes
    ----------
    x : float or None
        Horizontal nudge amount.
    y : float or None
        Vertical nudge amount.
    """

    x: Optional[float] = None
    y: Optional[float] = None
    default_aes: Dict[str, Any] = {"nudge_x": 0, "nudge_y": 0}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        nx = self.x
        ny = self.y
        if nx is None:
            nx = data["nudge_x"].values if "nudge_x" in data.columns else 0.0
        if ny is None:
            ny = data["nudge_y"].values if "nudge_y" in data.columns else 0.0
        return {"x": nx, "y": ny}

    def compute_layer(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        layout: Any,
    ) -> pd.DataFrame:
        px = params.get("x", 0)
        py = params.get("y", 0)
        trans_x = (lambda v: v + px) if np.any(np.asarray(px) != 0) else None
        trans_y = (lambda v: v + py) if np.any(np.asarray(py) != 0) else None
        return _transform_position(data, trans_x, trans_y)

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        # Nudge is handled at compute_layer level
        return data


# ---------------------------------------------------------------------------
# PositionStack / PositionFill
# ---------------------------------------------------------------------------

class PositionStack(Position):
    """Stack overlapping elements on top of each other.

    Attributes
    ----------
    vjust : float
        Vertical justification (0 = bottom, 0.5 = middle, 1 = top).
    fill : bool
        If True, normalise stacks to fill [0, 1].
    reverse : bool
        Whether to reverse stacking order.
    """

    vjust: float = 1.0
    fill: bool = False
    reverse: bool = False

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        var = _stack_var(data)
        return {
            "var": var,
            "fill": self.fill,
            "vjust": self.vjust,
            "reverse": self.reverse,
        }

    def setup_data(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        if params.get("var") is None:
            return data
        data = data.copy()
        var = params["var"]
        if var == "y" and "y" in data.columns:
            data["ymax"] = data["y"]
        elif var == "ymax" and "ymax" in data.columns and "ymin" in data.columns:
            mask = (data["ymax"] == 0)
            data.loc[mask, "ymax"] = data.loc[mask, "ymin"]
        return data

    def compute_panel(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        scales: Any = None,
    ) -> pd.DataFrame:
        if params.get("var") is None:
            return data

        data = data.copy()
        vjust = params.get("vjust", 1.0)
        fill = params.get("fill", False)
        reverse = params.get("reverse", False)

        # Split positive and negative
        if "ymax" in data.columns:
            negative_mask = data["ymax"] < 0
            negative_mask = negative_mask.fillna(False)
        else:
            negative_mask = pd.Series([False] * len(data), index=data.index)

        neg = data[negative_mask].copy()
        pos = data[~negative_mask].copy()

        if len(neg) > 0:
            neg = _pos_stack(neg, vjust=vjust, fill=fill, reverse=reverse)
        if len(pos) > 0:
            pos = _pos_stack(pos, vjust=vjust, fill=fill, reverse=reverse)

        # Recombine in original order
        result = pd.concat([neg, pos], ignore_index=False)
        result = result.loc[data.index].reset_index(drop=True)
        return result


def _stack_var(data: pd.DataFrame) -> Optional[str]:
    """Determine the stacking variable.

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    str or None
    """
    if "ymax" in data.columns:
        return "ymax"
    elif "y" in data.columns:
        return "y"
    else:
        cli_warn("Stacking requires y or ymax aesthetics.")
        return None


def _pos_stack(
    df: pd.DataFrame,
    vjust: float = 1.0,
    fill: bool = False,
    reverse: bool = False,
) -> pd.DataFrame:
    """Core stacking algorithm.

    Stacks overlapping bars *within* each x-position group.  In R's
    ggplot2 this corresponds to ``collide()`` + ``stack_var()``, which
    groups rows sharing the same ``xmin``/``xmax`` interval before
    cumulating y values.

    Parameters
    ----------
    df : pd.DataFrame
    vjust : float
    fill : bool
    reverse : bool

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if "group" in df.columns:
        if reverse:
            df = df.sort_values("group", ascending=True)
        else:
            df = df.sort_values("group", ascending=False)

    # Determine the x-grouping key.  Use xmin if available (matches R's
    # collide), otherwise fall back to x.
    if "xmin" in df.columns:
        x_key = df["xmin"].values
    elif "x" in df.columns:
        x_key = df["x"].values
    else:
        x_key = np.zeros(len(df))

    y = df["y"].values if "y" in df.columns else np.zeros(len(df))
    y = np.where(np.isnan(y), 0, y)

    ymin_out = np.zeros(len(df))
    ymax_out = np.zeros(len(df))

    # Stack within each unique x position
    for xval in np.unique(x_key):
        mask = x_key == xval
        y_group = y[mask]
        heights = np.concatenate([[0], np.cumsum(y_group)])
        if fill:
            total = abs(heights[-1])
            if total > np.sqrt(np.finfo(float).eps):
                heights = heights / total
        n = len(y_group)
        ymin_out[mask] = np.minimum(heights[:n], heights[1:])
        ymax_out[mask] = np.maximum(heights[:n], heights[1:])

    df["y"] = (1 - vjust) * ymin_out + vjust * ymax_out
    df["ymin"] = ymin_out
    df["ymax"] = ymax_out
    return df


class PositionFill(PositionStack):
    """Stack and normalise to fill [0, 1].

    This is ``PositionStack`` with ``fill=True``.
    """

    fill: bool = True


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------

def position_identity() -> PositionIdentity:
    """Create an identity position (no adjustment).

    Returns
    -------
    PositionIdentity
    """
    return PositionIdentity()


def position_dodge(
    width: Optional[float] = None,
    preserve: str = "total",
    orientation: str = "x",
    reverse: bool = False,
) -> PositionDodge:
    """Create a dodge position adjustment.

    Parameters
    ----------
    width : float or None
    preserve : str
        ``"total"`` or ``"single"``.
    orientation : str
        ``"x"`` or ``"y"``.
    reverse : bool

    Returns
    -------
    PositionDodge
    """
    return PositionDodge(
        width=width,
        preserve=preserve,
        orientation=orientation,
        reverse=reverse,
    )


def position_dodge2(
    width: Optional[float] = None,
    preserve: str = "total",
    padding: float = 0.1,
    reverse: bool = False,
    group_row: str = "single",
) -> PositionDodge2:
    """Create a dodge2 position adjustment.

    Parameters
    ----------
    width : float or None
    preserve : str
    padding : float
    reverse : bool
    group_row : str

    Returns
    -------
    PositionDodge2
    """
    return PositionDodge2(
        width=width,
        preserve=preserve,
        padding=padding,
        reverse=reverse,
        group_row=group_row,
    )


def position_jitter(
    width: Optional[float] = None,
    height: Optional[float] = None,
    seed: Any = None,
) -> PositionJitter:
    """Create a jitter position adjustment.

    Parameters
    ----------
    width, height : float or None
    seed : int or None

    Returns
    -------
    PositionJitter
    """
    return PositionJitter(width=width, height=height, seed=seed)


def position_jitterdodge(
    jitter_width: Optional[float] = None,
    jitter_height: float = 0.0,
    dodge_width: float = 0.75,
    reverse: bool = False,
    preserve: str = "total",
    seed: Any = None,
) -> PositionJitterdodge:
    """Create a jitter+dodge position adjustment.

    Parameters
    ----------
    jitter_width : float or None
    jitter_height : float
    dodge_width : float
    reverse : bool
    preserve : str
    seed : int or None

    Returns
    -------
    PositionJitterdodge
    """
    return PositionJitterdodge(
        jitter_width=jitter_width,
        jitter_height=jitter_height,
        dodge_width=dodge_width,
        reverse=reverse,
        preserve=preserve,
        seed=seed,
    )


def position_nudge(
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> PositionNudge:
    """Create a nudge position adjustment.

    Parameters
    ----------
    x, y : float or None

    Returns
    -------
    PositionNudge
    """
    return PositionNudge(x=x or 0.0, y=y or 0.0)


def position_stack(
    vjust: float = 1.0,
    reverse: bool = False,
) -> PositionStack:
    """Create a stack position adjustment.

    Parameters
    ----------
    vjust : float
    reverse : bool

    Returns
    -------
    PositionStack
    """
    return PositionStack(vjust=vjust, reverse=reverse)


def position_fill(
    vjust: float = 1.0,
    reverse: bool = False,
) -> PositionFill:
    """Create a fill position adjustment (stack + normalise).

    Parameters
    ----------
    vjust : float
    reverse : bool

    Returns
    -------
    PositionFill
    """
    return PositionFill(vjust=vjust, reverse=reverse, fill=True)


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def is_position(x: Any) -> bool:
    """Test whether *x* is a Position.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
    """
    return isinstance(x, Position)

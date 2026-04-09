"""
Stat classes for ggplot2.

All ``stat_*()`` functions (like ``stat_bin()``) return a layer that contains
a ``Stat*`` object (like ``StatBin``).  The ``Stat*`` object is responsible
for computing summary statistics from the data before the geom renders it.

This module contains the base :class:`Stat` class and all built-in stat
implementations.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py.ggproto import GGProto, ggproto
from ggplot2_py._compat import (
    Waiver,
    is_waiver,
    waiver,
    cli_abort,
    cli_warn,
    cli_inform,
)
from ggplot2_py._utils import (
    remove_missing,
    resolution,
    snake_class,
    compact,
    data_frame,
    empty,
    has_groups,
)
from ggplot2_py.aes import aes, Mapping, standardise_aes_names, AfterStat

__all__ = [
    # Base class
    "Stat",
    # Concrete stat classes
    "StatIdentity",
    "StatBin",
    "StatCount",
    "StatDensity",
    "StatSmooth",
    "StatBoxplot",
    "StatSummary",
    "StatSummaryBin",
    "StatSummary2d",
    "StatSummaryHex",
    "StatFunction",
    "StatEcdf",
    "StatQq",
    "StatQqLine",
    "StatBin2d",
    "StatBinhex",
    "StatContour",
    "StatContourFilled",
    "StatDensity2d",
    "StatDensity2dFilled",
    "StatEllipse",
    "StatUnique",
    "StatSum",
    "StatYdensity",
    "StatBindot",
    "StatAlign",
    "StatConnect",
    "StatManual",
    "StatQuantile",
    "StatSf",
    "StatSfCoordinates",
    # Constructor functions
    "stat_identity",
    "stat_bin",
    "stat_count",
    "stat_density",
    "stat_smooth",
    "stat_boxplot",
    "stat_summary",
    "stat_summary_bin",
    "stat_summary2d",
    "stat_summary_2d",
    "stat_summary_hex",
    "stat_function",
    "stat_ecdf",
    "stat_qq",
    "stat_qq_line",
    "stat_bin2d",
    "stat_bin_2d",
    "stat_bin_hex",
    "stat_binhex",
    "stat_contour",
    "stat_contour_filled",
    "stat_density2d",
    "stat_density2d_filled",
    "stat_density_2d",
    "stat_density_2d_filled",
    "stat_ellipse",
    "stat_unique",
    "stat_sum",
    "stat_ydensity",
    "stat_align",
    "stat_connect",
    "stat_manual",
    "stat_quantile",
    "stat_sf",
    "stat_sf_coordinates",
    "stat_spoke",
    # Utilities
    "is_stat",
    # Summary helpers
    "mean_se",
    "mean_cl_boot",
    "mean_cl_normal",
    "mean_sdl",
    "median_hilow",
]


# ---------------------------------------------------------------------------
# Lazy layer import to avoid circular dependencies
# ---------------------------------------------------------------------------

def _layer(**kwargs: Any) -> Any:
    """Lazy import of ``layer`` to break circular dependency."""
    from ggplot2_py.layer import layer
    return layer(**kwargs)


def _layer_sf(**kwargs: Any) -> Any:
    """Lazy import of ``layer_sf`` to break circular dependency."""
    from ggplot2_py.layer import layer_sf
    return layer_sf(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flip_data(data: pd.DataFrame, flipped_aes: bool) -> pd.DataFrame:
    """Swap x/y columns when orientation is flipped.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    flipped_aes : bool
        Whether aesthetics are flipped.

    Returns
    -------
    pd.DataFrame
        Data with x/y columns swapped if *flipped_aes* is truthy.
    """
    if not flipped_aes:
        return data
    data = data.copy()
    swap_pairs = [
        ("x", "y"),
        ("xmin", "ymin"),
        ("xmax", "ymax"),
        ("xend", "yend"),
        ("xlower", "lower"),
        ("xupper", "upper"),
        ("xmiddle", "middle"),
    ]
    for a, b in swap_pairs:
        a_val = data.get(a)
        b_val = data.get(b)
        if a_val is not None and a in data.columns:
            data[b] = a_val
        if b_val is not None and b in data.columns:
            data[a] = b_val
    return data


def _has_flipped_aes(
    data: pd.DataFrame,
    params: Dict[str, Any],
    *,
    main_is_orthogonal: bool = False,
    range_is_orthogonal: bool = False,
    group_has_equal: bool = False,
    ambiguous: bool = False,
    main_is_optional: bool = False,
    main_is_continuous: bool = False,
    default: bool = False,
) -> bool:
    """Determine if the aesthetic orientation is flipped.

    Parameters
    ----------
    data : pd.DataFrame
        Layer data.
    params : dict
        Layer params.

    Returns
    -------
    bool
    """
    # Explicit orientation parameter takes precedence
    orientation = params.get("orientation", None)
    if orientation is not None and orientation is not np.nan:
        if isinstance(orientation, str):
            if orientation.lower() == "y":
                return True
            elif orientation.lower() == "x":
                return False

    # Check for explicit flipped_aes parameter
    if "flipped_aes" in params:
        fa = params["flipped_aes"]
        if isinstance(fa, bool):
            return fa

    has_x = "x" in data.columns or "x" in params
    has_y = "y" in data.columns or "y" in params

    if main_is_orthogonal:
        if has_x and not has_y:
            return True
        if has_y and not has_x:
            return False

    if main_is_continuous:
        if has_x and not has_y:
            return False
        if has_y and not has_x:
            return True

    return default


def _is_mapped_discrete(x: Any) -> bool:
    """Check if a variable represents discrete/categorical data.

    Parameters
    ----------
    x : Any
        Column data to check.

    Returns
    -------
    bool
    """
    if x is None:
        return False
    if isinstance(x, pd.Categorical):
        return True
    if isinstance(x, pd.Series):
        if pd.api.types.is_categorical_dtype(x):
            return True
        if pd.api.types.is_object_dtype(x):
            return True
        if pd.api.types.is_bool_dtype(x):
            return True
    return False


def _rescale_max(x: np.ndarray) -> np.ndarray:
    """Rescale a numeric array to [0, 1] by dividing by its max.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    mx = np.nanmax(np.abs(x))
    if mx == 0 or not np.isfinite(mx):
        return x
    return x / mx


def _check_required_aesthetics(
    required: Sequence[str],
    present: Sequence[str],
    name: str,
) -> None:
    """Raise if required aesthetics are missing.

    Parameters
    ----------
    required : sequence of str
        Required aesthetics, may contain ``|`` for alternatives.
    present : sequence of str
        Available aesthetic/parameter names.
    name : str
        Name for error messages.
    """
    present = set(present)
    missing = []
    for req in required:
        options = [s.strip() for s in req.split("|")]
        if not any(o in present for o in options):
            missing.append(req)
    if missing:
        cli_abort(
            f"{name} requires the following missing aesthetics: "
            + ", ".join(missing)
        )


def _inner_runs(x: np.ndarray) -> np.ndarray:
    """Mark inner runs of values as True, outer zero-runs as False.

    Parameters
    ----------
    x : np.ndarray of bool
        Boolean array where True represents non-zero bins.

    Returns
    -------
    np.ndarray of bool
    """
    if len(x) == 0:
        return np.array([], dtype=bool)
    # Find run lengths
    changes = np.diff(x.astype(int))
    run_starts = np.concatenate([[0], np.where(changes != 0)[0] + 1])
    run_lengths = np.diff(np.concatenate([run_starts, [len(x)]]))
    run_values = x[run_starts]
    nruns = len(run_values)
    inner = np.ones(nruns, dtype=bool)
    for i in [0, nruns - 1]:
        if i < nruns:
            inner[i] = inner[i] and run_values[i]
    result = np.repeat(inner, run_lengths)
    return result


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

class _Bins:
    """Histogram bin specification.

    Parameters
    ----------
    breaks : np.ndarray
        Sorted break points.
    closed : str
        ``"right"`` or ``"left"``.
    """

    def __init__(self, breaks: np.ndarray, closed: str = "right") -> None:
        self.breaks = np.sort(np.asarray(breaks, dtype=float))
        self.closed = closed
        self.right_closed = closed == "right"
        # Fuzzy breaks for binning robustness
        finite_diffs = np.diff(self.breaks[np.isfinite(self.breaks)])
        if len(finite_diffs) > 0:
            fuzz = 1e-08 * np.median(finite_diffs)
        else:
            fuzz = np.finfo(float).eps * 1e3
        if self.right_closed:
            fuzzes = np.concatenate([[-fuzz], np.full(len(self.breaks) - 1, fuzz)])
        else:
            fuzzes = np.concatenate([np.full(len(self.breaks) - 1, -fuzz), [fuzz]])
        self.fuzzy = self.breaks + fuzzes


def _compute_bins(
    x: np.ndarray,
    scale: Any = None,
    breaks: Any = None,
    binwidth: Any = None,
    bins: Any = None,
    center: Optional[float] = None,
    boundary: Optional[float] = None,
    closed: str = "right",
) -> _Bins:
    """Compute bin specification for histogram-like stats.

    Parameters
    ----------
    x : array-like
        Data values.
    scale : optional
        Scale object with ``dimension()`` method.
    breaks, binwidth, bins : optional
        Binning parameters.
    center, boundary : optional
        Bin alignment parameters.
    closed : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    _Bins
    """
    x = np.asarray(x, dtype=float)
    x_finite = x[np.isfinite(x)]

    if scale is not None and hasattr(scale, "dimension"):
        x_range = np.array(scale.dimension(), dtype=float)
    else:
        if len(x_finite) == 0:
            x_range = np.array([0.0, 1.0])
        else:
            x_range = np.array([np.nanmin(x_finite), np.nanmax(x_finite)])

    # Explicit breaks
    if breaks is not None:
        if callable(breaks):
            breaks = breaks(x)
        return _Bins(np.asarray(breaks, dtype=float), closed)

    if boundary is not None and center is not None:
        cli_abort("Only one of 'boundary' and 'center' may be specified.")

    # Binwidth → breaks
    if binwidth is not None:
        if callable(binwidth):
            binwidth = binwidth(x)
        binwidth = float(binwidth)
        return _bin_breaks_width(x_range, binwidth, center, boundary, closed)

    # Bins (count) → breaks
    if bins is None:
        bins = 30
    if callable(bins):
        bins = bins(x)
    bins = int(bins)
    return _bin_breaks_bins(x_range, bins, center, boundary, closed)


def _bin_breaks_width(
    x_range: np.ndarray,
    width: float,
    center: Optional[float] = None,
    boundary: Optional[float] = None,
    closed: str = "right",
) -> _Bins:
    """Compute bin breaks given a fixed width.

    Parameters
    ----------
    x_range : np.ndarray
        ``[min, max]`` of data range.
    width : float
        Bin width.
    center, boundary : optional
        Alignment params.
    closed : str

    Returns
    -------
    _Bins
    """
    if boundary is None:
        if center is None:
            boundary = width / 2
        else:
            boundary = center - width / 2
    shift = np.floor((x_range[0] - boundary) / width)
    origin = boundary + shift * width
    max_x = x_range[1] + (1 - 1e-08) * width
    brks = np.arange(origin, max_x + width, width)
    # Trim to not overshoot excessively
    brks = brks[brks <= max_x + width]
    if len(brks) < 2:
        brks = np.array([brks[0], brks[0] + width])
    return _Bins(brks, closed)


def _bin_breaks_bins(
    x_range: np.ndarray,
    bins: int = 30,
    center: Optional[float] = None,
    boundary: Optional[float] = None,
    closed: str = "right",
) -> _Bins:
    """Compute bin breaks for a given number of bins.

    Parameters
    ----------
    x_range : np.ndarray
        ``[min, max]`` of data range.
    bins : int
        Number of bins.
    center, boundary : optional
        Alignment params.
    closed : str

    Returns
    -------
    _Bins
    """
    rng = x_range[1] - x_range[0]
    if rng == 0 or not np.isfinite(rng):
        width = 0.1
    elif bins == 1:
        width = rng
        boundary = x_range[0]
        center = None
    else:
        width = rng / (bins - 1)
        if center is None:
            if boundary is None:
                boundary = x_range[0] - width / 2
    return _bin_breaks_width(x_range, width, center=center, boundary=boundary, closed=closed)


def _bin_vector(
    x: np.ndarray,
    bins_obj: _Bins,
    weight: Optional[np.ndarray] = None,
    pad: bool = False,
) -> pd.DataFrame:
    """Place data into bins and return count/density summary.

    Parameters
    ----------
    x : array-like
        Data values.
    bins_obj : _Bins
        Bin specification.
    weight : array-like, optional
        Observation weights.
    pad : bool
        Whether to add padding bins with 0 count.

    Returns
    -------
    pd.DataFrame
        With columns: count, x, xmin, xmax, width, density, ncount, ndensity.
    """
    x = np.asarray(x, dtype=float)
    brk = bins_obj.breaks
    if weight is None:
        weight = np.ones_like(x)
    else:
        weight = np.asarray(weight, dtype=float)
        weight = np.where(np.isnan(weight), 0, weight)

    # Assign each x to a bin
    if bins_obj.right_closed:
        bin_idx = np.searchsorted(bins_obj.fuzzy[1:], x, side="left")
    else:
        bin_idx = np.searchsorted(bins_obj.fuzzy[:-1], x, side="right") - 1

    nbins = len(brk) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    # Count per bin
    bin_count = np.zeros(nbins, dtype=float)
    for i in range(len(x)):
        if np.isfinite(x[i]):
            bin_count[bin_idx[i]] += weight[i]

    bin_x = (brk[:-1] + brk[1:]) / 2
    bin_widths = np.diff(brk)

    if pad:
        bin_count = np.concatenate([[0], bin_count, [0]])
        w1, wn = bin_widths[0], bin_widths[-1]
        bin_widths = np.concatenate([[w1], bin_widths, [wn]])
        bin_x = np.concatenate([[bin_x[0] - w1], bin_x, [bin_x[-1] + wn]])

    total_count = np.sum(np.abs(bin_count))
    density = np.where(
        (bin_widths > 0) & (total_count > 0),
        bin_count / bin_widths / total_count,
        0.0,
    )
    max_count = np.max(np.abs(bin_count)) if len(bin_count) > 0 else 1
    max_density = np.max(np.abs(density)) if len(density) > 0 else 1
    if max_count == 0:
        max_count = 1
    if max_density == 0:
        max_density = 1

    return pd.DataFrame({
        "count": bin_count,
        "x": bin_x,
        "xmin": bin_x - bin_widths / 2,
        "xmax": bin_x + bin_widths / 2,
        "width": bin_widths,
        "density": density,
        "ncount": bin_count / max_count,
        "ndensity": density / max_density,
    })


def _bin_cut(x: np.ndarray, bins_obj: _Bins) -> np.ndarray:
    """Assign each value to a bin index.

    Parameters
    ----------
    x : np.ndarray
    bins_obj : _Bins

    Returns
    -------
    np.ndarray of int
        Bin indices (1-based).
    """
    x = np.asarray(x, dtype=float)
    if bins_obj.right_closed:
        idx = np.searchsorted(bins_obj.fuzzy[1:], x, side="left")
    else:
        idx = np.searchsorted(bins_obj.fuzzy[:-1], x, side="right") - 1
    nbins = len(bins_obj.breaks) - 1
    idx = np.clip(idx, 0, nbins - 1)
    return idx + 1  # 1-based


def _bin_loc(breaks: np.ndarray, idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Get bin locations for given bin indices.

    Parameters
    ----------
    breaks : np.ndarray
    idx : np.ndarray
        1-based bin indices.

    Returns
    -------
    dict
        With keys ``left``, ``right``, ``mid``, ``length``.
    """
    breaks = np.asarray(breaks, dtype=float)
    idx = np.asarray(idx, dtype=int) - 1  # to 0-based
    left = breaks[:-1][idx]
    right = breaks[1:][idx]
    return {
        "left": left,
        "right": right,
        "mid": (left + right) / 2,
        "length": right - left,
    }


def _dual_param(
    x: Any,
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize a parameter to x/y dict.

    Parameters
    ----------
    x : scalar, list, or dict
    default : dict, optional

    Returns
    -------
    dict
        With keys ``x`` and ``y``.
    """
    if default is None:
        default = {"x": None, "y": None}
    if x is None:
        return default
    if isinstance(x, dict):
        return x
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return {"x": x[0], "y": x[1]}
    return {"x": x, "y": x}


# ---------------------------------------------------------------------------
# Summary helper functions
# ---------------------------------------------------------------------------

def mean_se(x: np.ndarray, mult: float = 1.0) -> pd.DataFrame:
    """Compute mean and standard error.

    Parameters
    ----------
    x : array-like
        Numeric values.
    mult : float, optional
        Multiplier for the standard error. Default 1.

    Returns
    -------
    pd.DataFrame
        With columns ``y``, ``ymin``, ``ymax``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return pd.DataFrame({"y": [np.nan], "ymin": [np.nan], "ymax": [np.nan]})
    m = np.mean(x)
    se = mult * np.sqrt(np.var(x, ddof=1) / len(x)) if len(x) > 1 else 0.0
    return pd.DataFrame({"y": [m], "ymin": [m - se], "ymax": [m + se]})


def mean_cl_boot(
    x: np.ndarray,
    confidence: float = 0.95,
    B: int = 1000,
) -> pd.DataFrame:
    """Compute bootstrapped confidence interval of the mean.

    Parameters
    ----------
    x : array-like
        Numeric values.
    confidence : float, optional
        Confidence level. Default 0.95.
    B : int, optional
        Number of bootstrap samples. Default 1000.

    Returns
    -------
    pd.DataFrame
        With columns ``y``, ``ymin``, ``ymax``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return pd.DataFrame({"y": [np.nan], "ymin": [np.nan], "ymax": [np.nan]})
    rng = np.random.default_rng()
    boot_means = np.array([
        np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(B)
    ])
    alpha = 1 - confidence
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return pd.DataFrame({"y": [np.mean(x)], "ymin": [lo], "ymax": [hi]})


def mean_cl_normal(
    x: np.ndarray,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute normal-theory confidence interval of the mean.

    Parameters
    ----------
    x : array-like
        Numeric values.
    confidence : float, optional
        Confidence level. Default 0.95.

    Returns
    -------
    pd.DataFrame
        With columns ``y``, ``ymin``, ``ymax``.
    """
    from scipy import stats as scipy_stats

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return pd.DataFrame({"y": [np.nan], "ymin": [np.nan], "ymax": [np.nan]})
    m = np.mean(x)
    se = np.sqrt(np.var(x, ddof=1) / len(x)) if len(x) > 1 else 0.0
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=len(x) - 1) if len(x) > 1 else 0.0
    return pd.DataFrame({"y": [m], "ymin": [m - t_crit * se], "ymax": [m + t_crit * se]})


def mean_sdl(x: np.ndarray, mult: float = 2.0) -> pd.DataFrame:
    """Compute mean plus/minus a multiple of the standard deviation.

    Parameters
    ----------
    x : array-like
        Numeric values.
    mult : float, optional
        Multiplier for sd. Default 2.

    Returns
    -------
    pd.DataFrame
        With columns ``y``, ``ymin``, ``ymax``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return pd.DataFrame({"y": [np.nan], "ymin": [np.nan], "ymax": [np.nan]})
    m = np.mean(x)
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return pd.DataFrame({"y": [m], "ymin": [m - mult * sd], "ymax": [m + mult * sd]})


def median_hilow(x: np.ndarray, confidence: float = 0.95) -> pd.DataFrame:
    """Compute median and quantile range.

    Parameters
    ----------
    x : array-like
        Numeric values.
    confidence : float, optional
        Confidence level determining quantile range. Default 0.95.

    Returns
    -------
    pd.DataFrame
        With columns ``y``, ``ymin``, ``ymax``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return pd.DataFrame({"y": [np.nan], "ymin": [np.nan], "ymax": [np.nan]})
    alpha = 1 - confidence
    return pd.DataFrame({
        "y": [np.median(x)],
        "ymin": [np.percentile(x, 100 * alpha / 2)],
        "ymax": [np.percentile(x, 100 * (1 - alpha / 2))],
    })


def _make_summary_fun(
    fun_data: Any = None,
    fun: Any = None,
    fun_max: Any = None,
    fun_min: Any = None,
    fun_args: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Create a summary function from user-supplied components.

    Parameters
    ----------
    fun_data : callable or None
        A function that takes a DataFrame and returns a DataFrame with
        ``y``, ``ymin``, ``ymax``.
    fun, fun_max, fun_min : callable or None
        Individual summary functions.
    fun_args : dict, optional
        Extra arguments to pass to the functions.

    Returns
    -------
    callable
        Function that takes a DataFrame and returns summary DataFrame.
    """
    if fun_args is None:
        fun_args = {}

    def _resolve(f: Any) -> Optional[Callable]:
        if f is None:
            return None
        if isinstance(f, str):
            # Look up common summary functions
            builtins_map = {
                "mean_se": mean_se,
                "mean_cl_boot": mean_cl_boot,
                "mean_cl_normal": mean_cl_normal,
                "mean_sdl": mean_sdl,
                "median_hilow": median_hilow,
            }
            if f in builtins_map:
                return builtins_map[f]
            # Try numpy
            if hasattr(np, f):
                return getattr(np, f)
            cli_abort(f"Cannot find summary function '{f}'.")
        return f

    if fun_data is not None:
        fun_data = _resolve(fun_data)

        def _summarise(df: pd.DataFrame) -> pd.DataFrame:
            return fun_data(df["y"], **fun_args)

        return _summarise

    if fun is not None or fun_max is not None or fun_min is not None:
        fun = _resolve(fun)
        fun_max = _resolve(fun_max)
        fun_min = _resolve(fun_min)

        def _call(f: Optional[Callable], vals: np.ndarray) -> float:
            if f is None:
                return np.nan
            return f(vals, **fun_args)

        def _summarise_three(df: pd.DataFrame) -> pd.DataFrame:
            y_vals = df["y"].values
            return pd.DataFrame({
                "ymin": [_call(fun_min, y_vals)],
                "y": [_call(fun, y_vals)],
                "ymax": [_call(fun_max, y_vals)],
            })

        return _summarise_three

    # Default to mean_se
    def _default_summarise(df: pd.DataFrame) -> pd.DataFrame:
        return mean_se(df["y"])

    return _default_summarise


# ---------------------------------------------------------------------------
# Density computation helpers
# ---------------------------------------------------------------------------

def _compute_density(
    x: np.ndarray,
    w: Optional[np.ndarray] = None,
    from_: float = 0.0,
    to: float = 1.0,
    bw: Union[str, float] = "nrd0",
    adjust: float = 1.0,
    kernel: str = "gaussian",
    n: int = 512,
    bounds: Tuple[float, float] = (-np.inf, np.inf),
) -> pd.DataFrame:
    """Compute kernel density estimate.

    Parameters
    ----------
    x : array-like
        Sample data.
    w : array-like, optional
        Weights.
    from_, to : float
        Range over which to evaluate the density.
    bw : str or float
        Bandwidth specification.
    adjust : float
        Bandwidth adjustment multiplier.
    kernel : str
        Kernel name (ignored for scipy kde, always gaussian).
    n : int
        Number of evaluation points.
    bounds : tuple
        Known bounds for boundary correction.

    Returns
    -------
    pd.DataFrame
        With columns: x, density, scaled, ndensity, count, wdensity, n.
    """
    from scipy import stats as scipy_stats

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nx = len(x)

    if w is None:
        weights = np.ones(nx) / nx
        w_sum = nx
    else:
        w = np.asarray(w, dtype=float)[:nx]
        w_sum = np.sum(w)
        if w_sum > 0:
            weights = w / w_sum
        else:
            weights = np.ones(nx) / nx

    # Filter data to bounds
    if np.isfinite(bounds[0]) or np.isfinite(bounds[1]):
        lo = bounds[0] if np.isfinite(bounds[0]) else -np.inf
        hi = bounds[1] if np.isfinite(bounds[1]) else np.inf
        inside = (x >= lo) & (x <= hi)
        if not np.all(inside):
            cli_warn("Some data points are outside of `bounds`. Removing them.")
            x = x[inside]
            weights = weights[inside]
            w_new = np.sum(weights)
            if w_new > 0:
                weights = weights / w_new
            w_sum = w_new * w_sum
            nx = len(x)

    if nx < 2:
        cli_warn("Groups with fewer than two data points have been dropped.")
        return pd.DataFrame({
            "x": [np.nan],
            "density": [np.nan],
            "scaled": [np.nan],
            "ndensity": [np.nan],
            "count": [np.nan],
            "wdensity": [np.nan],
            "n": [np.nan],
        })

    # Compute bandwidth
    bw_val = _precompute_bw(x, bw)
    bw_val = bw_val * adjust

    if bw_val <= 0 or not np.isfinite(bw_val):
        cli_abort(f"Bandwidth must be a finite positive number, got {bw_val}.")

    # Evaluate density
    try:
        kde = scipy_stats.gaussian_kde(x, bw_method=bw_val / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else bw_val)
    except Exception:
        kde = scipy_stats.gaussian_kde(x)

    grid = np.linspace(from_, to, n)
    dens_y = kde(grid)

    # Boundary correction via reflection
    if np.isfinite(bounds[0]) or np.isfinite(bounds[1]):
        dens_y = _reflect_density(grid, dens_y, bounds, from_, to)

    max_dens = np.nanmax(dens_y)
    if max_dens == 0 or not np.isfinite(max_dens):
        max_dens = 1.0

    return pd.DataFrame({
        "x": grid,
        "density": dens_y,
        "scaled": dens_y / max_dens,
        "ndensity": dens_y / max_dens,
        "count": dens_y * nx,
        "wdensity": dens_y * w_sum,
        "n": np.full(len(grid), nx, dtype=int),
    })


def _reflect_density(
    grid: np.ndarray,
    dens: np.ndarray,
    bounds: Tuple[float, float],
    from_: float,
    to: float,
) -> np.ndarray:
    """Apply boundary correction via reflection.

    Parameters
    ----------
    grid : np.ndarray
        Evaluation points.
    dens : np.ndarray
        Density values.
    bounds : tuple
        Known data bounds.
    from_, to : float
        Evaluation range.

    Returns
    -------
    np.ndarray
        Corrected density values.
    """
    from scipy.interpolate import interp1d

    f = interp1d(grid, dens, kind="linear", bounds_error=False, fill_value=0.0)

    left = max(from_, bounds[0]) if np.isfinite(bounds[0]) else from_
    right = min(to, bounds[1]) if np.isfinite(bounds[1]) else to
    out_x = np.linspace(left, right, len(grid))

    out_y = f(out_x)
    if np.isfinite(bounds[0]):
        left_ref = f(bounds[0] + (bounds[0] - out_x))
        out_y = out_y + left_ref
    if np.isfinite(bounds[1]):
        right_ref = f(bounds[1] + (bounds[1] - out_x))
        out_y = out_y + right_ref

    return out_y


def _precompute_bw(x: np.ndarray, bw: Union[str, float] = "nrd0") -> float:
    """Compute bandwidth from a string rule or return numeric value.

    Parameters
    ----------
    x : np.ndarray
        Sample data.
    bw : str or float
        Bandwidth rule name or numeric bandwidth.

    Returns
    -------
    float
        Bandwidth value.
    """
    if isinstance(bw, (int, float)):
        return float(bw)
    from scipy import stats as scipy_stats

    bw = bw.lower()
    n = len(x)
    sd = np.std(x, ddof=1) if n > 1 else 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25]))

    if bw in ("nrd0",):
        # Silverman's rule
        hi = sd
        lo = min(hi, iqr / 1.34) if iqr > 0 else hi
        if lo == 0:
            lo = hi if hi > 0 else 1.0
        return 0.9 * lo * n ** (-0.2)
    elif bw == "nrd":
        hi = sd
        lo = min(hi, iqr / 1.34) if iqr > 0 else hi
        if lo == 0:
            lo = hi if hi > 0 else 1.0
        return 1.06 * lo * n ** (-0.2)
    elif bw in ("sj", "sj-ste", "sj-dpi"):
        # Use scipy's Sheather-Jones equivalent
        try:
            kde = scipy_stats.gaussian_kde(x, bw_method="silverman")
            return kde.factor * np.std(x, ddof=1)
        except Exception:
            return _precompute_bw(x, "nrd0")
    elif bw in ("ucv", "bcv"):
        try:
            kde = scipy_stats.gaussian_kde(x, bw_method="silverman")
            return kde.factor * np.std(x, ddof=1)
        except Exception:
            return _precompute_bw(x, "nrd0")
    else:
        cli_abort(f"Unknown bandwidth rule: {bw!r}")


# ---------------------------------------------------------------------------
# Contour helpers
# ---------------------------------------------------------------------------

def _contour_breaks(
    z_range: np.ndarray,
    bins: Optional[int] = None,
    binwidth: Optional[float] = None,
    breaks: Any = None,
) -> np.ndarray:
    """Compute break values for contouring.

    Parameters
    ----------
    z_range : array-like
        ``[min, max]`` of z values.
    bins, binwidth : optional
        Binning parameters.
    breaks : optional
        Explicit breaks.

    Returns
    -------
    np.ndarray
    """
    if breaks is not None:
        if callable(breaks):
            breaks = breaks(z_range)
        return np.asarray(breaks, dtype=float)

    z_range = np.asarray(z_range, dtype=float)

    if bins is None and binwidth is None:
        from matplotlib.ticker import MaxNLocator
        locator = MaxNLocator(nbins=10)
        return np.asarray(locator.tick_values(z_range[0], z_range[1]))

    if bins is not None:
        if bins == 1:
            return z_range.copy()
        bw = (z_range[1] - z_range[0]) / (bins - 1)
        result = np.arange(z_range[0], z_range[1] + bw, bw)
        if len(result) < bins + 1:
            bw = (z_range[1] - z_range[0]) / bins
            result = np.arange(z_range[0], z_range[1] + bw, bw)
        return result

    if binwidth is not None:
        return np.arange(z_range[0], z_range[1] + binwidth, binwidth)

    return np.linspace(z_range[0], z_range[1], 11)


# ---------------------------------------------------------------------------
# Hexagonal binning helper
# ---------------------------------------------------------------------------

def _hex_binwidth(bins: int, scales: Any) -> Tuple[float, float]:
    """Compute hex bin widths from number of bins.

    Parameters
    ----------
    bins : int
    scales : dict-like

    Returns
    -------
    tuple of float
    """
    if hasattr(scales, "x") and hasattr(scales.x, "dimension"):
        x_range = np.array(scales.x.dimension())
    else:
        x_range = np.array([0.0, 1.0])
    if hasattr(scales, "y") and hasattr(scales.y, "dimension"):
        y_range = np.array(scales.y.dimension())
    else:
        y_range = np.array([0.0, 1.0])

    x_bw = (x_range[1] - x_range[0]) / bins
    y_bw = (y_range[1] - y_range[0]) / bins
    return (x_bw, y_bw)


def _hex_bin_summarise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    binwidth: Tuple[float, float],
    fun: Callable = np.sum,
    fun_args: Optional[Dict[str, Any]] = None,
    drop: bool = True,
) -> pd.DataFrame:
    """Summarise z values in hexagonal bins.

    Parameters
    ----------
    x, y, z : array-like
    binwidth : tuple
    fun : callable
    fun_args : dict, optional
    drop : bool

    Returns
    -------
    pd.DataFrame
        With columns x, y, value.
    """
    if fun_args is None:
        fun_args = {}

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    bw_x, bw_y = binwidth

    # Compute hex grid coordinates
    # Using offset hex coordinates
    ix = np.floor(x / bw_x).astype(int)
    iy = np.floor(y / (bw_y * np.sqrt(3) / 2)).astype(int)

    # Offset every other row
    shifted = iy % 2 == 1
    ix_adj = ix.copy()
    x_shifted = x - (0.5 * bw_x * shifted.astype(float))
    ix_adj = np.floor(x_shifted / bw_x).astype(int)

    # Create bin keys
    keys = ix_adj.astype(str) + "_" + iy.astype(str)
    unique_keys = np.unique(keys)

    results = []
    for key in unique_keys:
        mask = keys == key
        z_vals = z[mask]
        if drop and len(z_vals) == 0:
            continue
        val = fun(z_vals, **fun_args)
        # Center of hex
        parts = key.split("_")
        bi, bj = int(parts[0]), int(parts[1])
        cx = (bi + 0.5) * bw_x + (0.5 * bw_x if bj % 2 == 1 else 0)
        cy = (bj + 0.5) * bw_y * np.sqrt(3) / 2
        results.append({"x": cx, "y": cy, "value": val})

    if len(results) == 0:
        return pd.DataFrame({"x": [], "y": [], "value": []})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Weighted ECDF helper
# ---------------------------------------------------------------------------

def _wecdf(
    x: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a weighted empirical CDF function.

    Parameters
    ----------
    x : array-like
    weights : array-like, optional

    Returns
    -------
    callable
        Function mapping query values to ECDF values.
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) == 1:
            weights = np.full_like(x, weights[0])

    # Remove NaN
    valid = np.isfinite(x) & np.isfinite(weights)
    x = x[valid]
    weights = weights[valid]

    # Sort
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = weights[order]

    total = np.sum(w_sorted)
    if total == 0:
        cli_abort("Cannot compute ECDF when weights sum to 0.")

    # Aggregate weights by unique x values
    unique_vals, inv = np.unique(x_sorted, return_inverse=True)
    agg_weights = np.zeros(len(unique_vals))
    for i, w in zip(inv, w_sorted):
        agg_weights[i] += w
    cum_weights = np.cumsum(agg_weights) / total

    def ecdf_fun(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=float)
        result = np.zeros_like(query)
        for i, q in enumerate(query):
            if q < unique_vals[0]:
                result[i] = 0.0
            elif q >= unique_vals[-1]:
                result[i] = 1.0
            else:
                idx = np.searchsorted(unique_vals, q, side="right") - 1
                result[i] = cum_weights[idx]
        return result

    return ecdf_fun


# ---------------------------------------------------------------------------
# Density bin helper (for dot plots)
# ---------------------------------------------------------------------------

def _densitybin(
    x: np.ndarray,
    weight: Optional[np.ndarray] = None,
    binwidth: Optional[float] = None,
    range_: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Density-based binning for dot plots.

    Parameters
    ----------
    x : array-like
    weight : array-like, optional
    binwidth : float, optional
    range_ : tuple, optional

    Returns
    -------
    pd.DataFrame
        With columns x, bin, binwidth, weight, bincenter.
    """
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return pd.DataFrame()

    if weight is None:
        weight = np.ones_like(x)
    else:
        weight = np.asarray(weight, dtype=float)
        weight = np.where(np.isnan(weight), 0, weight)

    if range_ is None:
        range_ = (np.nanmin(x), np.nanmax(x))
    if binwidth is None:
        binwidth = (range_[1] - range_[0]) / 30

    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = weight[order]

    bin_ids = np.empty(len(x_sorted), dtype=int)
    cbin = 0
    binend = -np.inf

    for i in range(len(x_sorted)):
        if x_sorted[i] >= binend:
            binend = x_sorted[i] + binwidth
            cbin += 1
        bin_ids[i] = cbin

    df = pd.DataFrame({
        "x": x_sorted,
        "bin": bin_ids,
        "binwidth": binwidth,
        "weight": w_sorted,
    })

    # Compute bin centers
    centers = df.groupby("bin")["x"].transform(lambda v: (v.min() + v.max()) / 2)
    df["bincenter"] = centers

    return df


# ============================================================================
# BASE STAT CLASS
# ============================================================================

class Stat(GGProto):
    """Base class for all ggplot2 statistics.

    Stat objects compute statistical transformations of the data before
    it is passed to the geom for rendering.  Subclasses typically override
    :meth:`compute_group` (or :meth:`compute_panel`) and set
    ``required_aes``, ``default_aes``, etc.

    Attributes
    ----------
    required_aes : list of str
        Aesthetics that must be present in the data.
    non_missing_aes : list of str
        Additional aesthetics that trigger missing-value removal.
    optional_aes : list of str
        Accepted but not required aesthetics.
    default_aes : dict
        Default aesthetic values.
    dropped_aes : list of str
        Aesthetics consumed by the computation (removed afterwards).
    extra_params : list of str
        Extra parameter names beyond those in ``compute_group``.
    retransform : bool
        Whether computed values should be retransformed.
    """

    required_aes: List[str] = []
    non_missing_aes: List[str] = []
    optional_aes: List[str] = []
    default_aes: Dict[str, Any] = {}
    dropped_aes: List[str] = []
    extra_params: List[str] = ["na_rm"]
    retransform: bool = True

    # -- Methods ---------------------------------------------------------------

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate or modify parameters based on the data.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Validate or modify data before computation.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        return data

    def compute_layer(self, data: pd.DataFrame, params: Dict[str, Any], layout: Any) -> pd.DataFrame:
        """Orchestrate per-panel stat computation.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict
        layout : Layout

        Returns
        -------
        pd.DataFrame
        """
        _check_required_aesthetics(
            self.required_aes,
            list(data.columns) + list(params.keys()),
            snake_class(self),
        )

        # Determine which required aes are actually present
        req_aes = []
        for r in self.required_aes:
            for opt in r.split("|"):
                opt = opt.strip()
                if opt in data.columns:
                    req_aes.append(opt)

        data = remove_missing(
            data,
            vars=req_aes + list(self.non_missing_aes),
            na_rm=params.get("na_rm", params.get("na.rm", False)),
            name=snake_class(self),
            finite=True,
        )

        # Trim params to those accepted by the stat
        accepted = set(self.parameters(extra=True))
        trimmed = {k: v for k, v in params.items() if k in accepted}

        results = []
        if "PANEL" in data.columns:
            for panel, panel_data in data.groupby("PANEL", sort=False):
                scales = layout.get_scales(panel) if layout is not None else {}
                try:
                    result = self.compute_panel(panel_data, scales, **trimmed)
                except Exception as e:
                    cli_warn(f"Computation failed in {snake_class(self)}: {e}")
                    result = pd.DataFrame()
                if not result.empty:
                    results.append(result)
        else:
            scales = {}
            try:
                result = self.compute_panel(data, scales, **trimmed)
            except Exception as e:
                cli_warn(f"Computation failed in {snake_class(self)}: {e}")
                result = pd.DataFrame()
            if not result.empty:
                results.append(result)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def compute_panel(self, data: pd.DataFrame, scales: Any, **params: Any) -> pd.DataFrame:
        """Compute stat for a single panel (splits by group).

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        **params
            Additional computation parameters.

        Returns
        -------
        pd.DataFrame
        """
        if data.empty:
            return pd.DataFrame()

        if "group" not in data.columns:
            data = data.copy()
            data["group"] = 1

        results = []
        for group_val, group_data in data.groupby("group", sort=False):
            try:
                new = self.compute_group(group_data, scales, **params)
            except Exception as e:
                cli_warn(f"Computation failed in {snake_class(self)}: {e}")
                new = pd.DataFrame()

            if new is None or (isinstance(new, pd.DataFrame) and new.empty):
                continue

            if not isinstance(new, pd.DataFrame):
                new = pd.DataFrame(new)

            # Preserve constant columns from original group
            for col in group_data.columns:
                if col not in new.columns:
                    vals = group_data[col].values
                    if len(set(vals)) == 1:
                        new[col] = vals[0]

            results.append(new)

        if not results:
            return pd.DataFrame()

        combined = pd.concat(results, ignore_index=True)

        # Drop non-constant columns that weren't in the computed result
        # (warn if they weren't in dropped_aes)
        return combined

    def compute_group(self, data: pd.DataFrame, scales: Any, **params: Any) -> pd.DataFrame:
        """Compute stat for a single group.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        **params

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.compute_group() is not implemented."
        )

    def finish_layer(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Post-processing hook after scales are applied.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        return data

    def parameters(self, extra: bool = False) -> List[str]:
        """List accepted parameter names.

        Parameters
        ----------
        extra : bool
            Whether to include ``extra_params``.

        Returns
        -------
        list of str
        """
        # Inspect compute_group signature
        sig = inspect.signature(self.compute_group)
        args = [
            p.name
            for p in sig.parameters.values()
            if p.name not in ("self", "data", "scales")
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if extra:
            args = list(set(args) | set(self.extra_params))
        return args

    def aesthetics(self) -> List[str]:
        """List accepted aesthetic names.

        Returns
        -------
        list of str
        """
        req = []
        for r in (self.required_aes or []):
            req.extend(r.split("|"))
        return list(set(req) | set(self.default_aes.keys()) | set(self.optional_aes) | {"group"})


def is_stat(x: Any) -> bool:
    """Test whether *x* is a Stat object.

    Parameters
    ----------
    x : Any

    Returns
    -------
    bool
    """
    return isinstance(x, type) and issubclass(x, Stat) or isinstance(x, Stat)


# ============================================================================
# StatIdentity
# ============================================================================

class StatIdentity(Stat):
    """Leave data unchanged.

    The identity stat passes data through without any transformation.
    """

    def compute_layer(self, data: pd.DataFrame, params: Dict[str, Any], layout: Any) -> pd.DataFrame:
        """Return data unaltered.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict
        layout : Any

        Returns
        -------
        pd.DataFrame
        """
        return data


def stat_identity(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Leave data unchanged.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom : str
    position : str
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs
        Additional parameters.

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatIdentity,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


# ============================================================================
# StatBin
# ============================================================================

class StatBin(Stat):
    """Histogram binning stat.

    Divides continuous x values into bins and counts observations per bin.

    Attributes
    ----------
    required_aes : list
        ``["x|y"]``
    default_aes : dict
        ``weight=1``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x|y"]
    default_aes: Dict[str, Any] = {
        "x": AfterStat("count"),
        "y": AfterStat("count"),
        "weight": 1,
    }
    dropped_aes: List[str] = ["weight"]
    extra_params: List[str] = ["na_rm", "orientation"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate binning parameters.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(
            data, params, main_is_orthogonal=False
        )

        has_x = "x" in data.columns or "x" in params
        has_y = "y" in data.columns or "y" in params
        if not has_x and not has_y:
            cli_abort(f"{snake_class(self)} requires an x or y aesthetic.")
        if has_x and has_y:
            cli_abort(f"{snake_class(self)} must only have an x or y aesthetic.")

        # Default bins
        if params.get("breaks") is None and params.get("binwidth") is None and params.get("bins") is None:
            cli_inform(f"{snake_class(self)} using bins = 30. Pick better value with binwidth.")
            params["bins"] = 30

        # Handle drop parameter
        drop = params.get("drop", "none")
        if isinstance(drop, bool):
            drop = "all" if drop else "none"
        if drop not in ("all", "none", "extremes"):
            drop = "none"
        params["drop"] = drop

        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        binwidth: Optional[float] = None,
        bins: Optional[int] = None,
        center: Optional[float] = None,
        boundary: Optional[float] = None,
        closed: str = "right",
        pad: bool = False,
        breaks: Any = None,
        flipped_aes: bool = False,
        drop: str = "none",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Bin x values and compute counts.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        binwidth : float, optional
        bins : int, optional
        center, boundary : float, optional
        closed : str
        pad : bool
        breaks : array-like, optional
        flipped_aes : bool
        drop : str

        Returns
        -------
        pd.DataFrame
            With columns: count, x, xmin, xmax, width, density, ncount, ndensity.
        """
        data = _flip_data(data, flipped_aes)
        x_col = "x"

        x_vals = data[x_col].values
        scale = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)

        bins_obj = _compute_bins(
            x_vals, scale,
            breaks=breaks, binwidth=binwidth, bins=bins,
            center=center, boundary=boundary, closed=closed,
        )

        weight = data["weight"].values if "weight" in data.columns else None
        result = _bin_vector(x_vals, bins_obj, weight=weight, pad=pad)

        # Apply drop
        if drop == "all":
            result = result[result["count"] != 0].reset_index(drop=True)
        elif drop == "extremes":
            keep = _inner_runs(result["count"].values != 0)
            result = result[keep].reset_index(drop=True)

        result["flipped_aes"] = flipped_aes
        return _flip_data(result, flipped_aes)


def stat_bin(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "bar",
    position: str = "stack",
    binwidth: Optional[float] = None,
    bins: Optional[int] = None,
    center: Optional[float] = None,
    boundary: Optional[float] = None,
    breaks: Any = None,
    closed: str = "right",
    pad: bool = False,
    drop: str = "none",
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Bin data and compute counts.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom : str
    position : str
    binwidth, bins : optional
    center, boundary : optional
    breaks : optional
    closed : str
    pad : bool
    drop : str
    na_rm : bool
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatBin,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "binwidth": binwidth, "bins": bins, "center": center,
            "boundary": boundary, "breaks": breaks, "closed": closed,
            "pad": pad, "drop": drop, "na_rm": na_rm,
            "orientation": orientation, **kwargs,
        },
    )


# ============================================================================
# StatCount
# ============================================================================

class StatCount(Stat):
    """Count unique x values.

    Attributes
    ----------
    required_aes : list
        ``["x|y"]``
    default_aes : dict
        ``y=after_stat(count), weight=1``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x|y"]
    default_aes: Dict[str, Any] = {"y": AfterStat("count"), "weight": 1}
    dropped_aes: List[str] = ["weight"]
    extra_params: List[str] = ["na_rm", "orientation"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(
            data, params, main_is_orthogonal=False
        )
        has_x = "x" in data.columns or "x" in params
        has_y = "y" in data.columns or "y" in params
        if not has_x and not has_y:
            cli_abort(f"{snake_class(self)} requires an x or y aesthetic.")
        if has_x and has_y:
            cli_abort(f"{snake_class(self)} must only have an x or y aesthetic.")

        if params.get("width") is None:
            x_col = "y" if params.get("flipped_aes") else "x"
            if x_col in data.columns:
                params["width"] = resolution(data[x_col].values, discrete=True) * 0.9
            else:
                params["width"] = 0.9

        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        width: Optional[float] = None,
        flipped_aes: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Count occurrences of each unique x.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        width : float, optional
        flipped_aes : bool

        Returns
        -------
        pd.DataFrame
            With columns: count, prop, x, width.
        """
        data = _flip_data(data, flipped_aes)
        x = data["x"].values
        weight = data["weight"].values if "weight" in data.columns else np.ones(len(x))

        unique_x = np.sort(np.unique(x))
        counts = np.array([np.sum(weight[x == ux]) for ux in unique_x])
        total = np.sum(np.abs(counts))

        result = pd.DataFrame({
            "count": counts,
            "prop": counts / total if total > 0 else counts,
            "x": unique_x,
            "width": width or 0.9,
            "flipped_aes": flipped_aes,
        })

        return _flip_data(result, flipped_aes)


def stat_count(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "bar",
    position: str = "stack",
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Count unique x values.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    na_rm : bool
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatCount,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, **kwargs},
    )


# ============================================================================
# StatDensity
# ============================================================================

class StatDensity(Stat):
    """Kernel density estimation.

    Attributes
    ----------
    required_aes : list
        ``["x|y"]``
    default_aes : dict
        ``fill=None, weight=None``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x|y"]
    default_aes: Dict[str, Any] = {
        "y": AfterStat("density"),
        "fill": None,
        "weight": None,
    }
    dropped_aes: List[str] = ["weight"]
    extra_params: List[str] = ["na_rm", "orientation"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(
            data, params, main_is_orthogonal=False, main_is_continuous=True
        )
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        bw: Union[str, float] = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        n: int = 512,
        trim: bool = False,
        na_rm: bool = False,
        bounds: Tuple[float, float] = (-np.inf, np.inf),
        flipped_aes: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute kernel density estimate.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        bw : str or float
        adjust : float
        kernel : str
        n : int
        trim : bool
        na_rm : bool
        bounds : tuple
        flipped_aes : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, density, scaled, ndensity, count, wdensity, n.
        """
        data = _flip_data(data, flipped_aes)

        if trim:
            x_range = (np.nanmin(data["x"]), np.nanmax(data["x"]))
        else:
            scale = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
            if scale is not None and hasattr(scale, "dimension"):
                x_range = tuple(scale.dimension())
            else:
                x_range = (np.nanmin(data["x"]), np.nanmax(data["x"]))

        weight = data["weight"].values if "weight" in data.columns and data["weight"].notna().any() else None

        density = _compute_density(
            data["x"].values, weight,
            from_=x_range[0], to=x_range[1],
            bw=bw, adjust=adjust, kernel=kernel, n=n, bounds=bounds,
        )

        density["flipped_aes"] = flipped_aes
        return _flip_data(density, flipped_aes)


def stat_density(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "area",
    position: str = "stack",
    bw: Union[str, float] = "nrd0",
    adjust: float = 1.0,
    kernel: str = "gaussian",
    n: int = 512,
    trim: bool = False,
    na_rm: bool = False,
    bounds: Tuple[float, float] = (-np.inf, np.inf),
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Compute kernel density estimate.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    bw : str or float
    adjust : float
    kernel : str
    n : int
    trim : bool
    na_rm : bool
    bounds : tuple
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatDensity,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "bw": bw, "adjust": adjust, "kernel": kernel, "n": n,
            "trim": trim, "na_rm": na_rm, "bounds": bounds,
            "orientation": orientation, **kwargs,
        },
    )


# ============================================================================
# StatSmooth
# ============================================================================

class StatSmooth(Stat):
    """Smoothing with confidence interval.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x", "y"]
    dropped_aes: List[str] = ["weight"]
    extra_params: List[str] = ["na_rm", "orientation"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve smoothing method.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(data, params, ambiguous=True)

        method = params.get("method")
        if method is None or method == "auto":
            if "group" in data.columns and "PANEL" in data.columns:
                max_group = data.groupby(["group", "PANEL"]).size().max()
            else:
                max_group = len(data)
            method = "loess" if max_group < 1000 else "lm"
            cli_inform(f"geom_smooth using method = '{method}'")

        params["method"] = method
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        method: str = "lm",
        formula: Any = None,
        se: bool = True,
        n: int = 80,
        span: float = 0.75,
        fullrange: bool = False,
        xseq: Optional[np.ndarray] = None,
        level: float = 0.95,
        method_args: Optional[Dict[str, Any]] = None,
        na_rm: bool = False,
        flipped_aes: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute smooth curve with optional confidence band.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        method : str
            ``"lm"``, ``"loess"``/``"lowess"``, ``"glm"``.
        formula : optional
        se : bool
        n : int
        span : float
        fullrange : bool
        xseq : array-like, optional
        level : float
        method_args : dict, optional
        na_rm : bool
        flipped_aes : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y, ymin, ymax, se.
        """
        from scipy import stats as scipy_stats

        data = _flip_data(data, flipped_aes)

        if len(data["x"].unique()) < 2:
            return pd.DataFrame()

        if "weight" not in data.columns:
            data = data.copy()
            data["weight"] = 1.0

        if xseq is None:
            if fullrange:
                scale = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
                if scale is not None and hasattr(scale, "dimension"):
                    rng = scale.dimension()
                else:
                    rng = (data["x"].min(), data["x"].max())
            else:
                rng = (data["x"].min(), data["x"].max())
            xseq = np.linspace(rng[0], rng[1], n)

        if method_args is None:
            method_args = {}

        try:
            if method in ("lm", "linear"):
                result = self._fit_lm(data, xseq, se, level)
            elif method in ("loess", "lowess"):
                result = self._fit_lowess(data, xseq, se, level, span)
            elif method == "glm":
                result = self._fit_lm(data, xseq, se, level)
            else:
                result = self._fit_lm(data, xseq, se, level)
        except Exception as e:
            cli_warn(f"Failed to fit group {data.get('group', [None])[0] if 'group' in data.columns else ''}: {e}")
            return pd.DataFrame()

        if result is None:
            return pd.DataFrame()

        result["flipped_aes"] = flipped_aes
        return _flip_data(result, flipped_aes)

    @staticmethod
    def _fit_lm(
        data: pd.DataFrame,
        xseq: np.ndarray,
        se: bool,
        level: float,
    ) -> pd.DataFrame:
        """Fit linear model.

        Parameters
        ----------
        data : pd.DataFrame
        xseq : np.ndarray
        se : bool
        level : float

        Returns
        -------
        pd.DataFrame
        """
        from scipy import stats as scipy_stats

        x = data["x"].values
        y = data["y"].values
        w = data["weight"].values if "weight" in data.columns else np.ones(len(x))

        # Weighted least squares
        coeffs = np.polyfit(x, y, deg=1, w=np.sqrt(w))
        y_pred = np.polyval(coeffs, xseq)

        result = pd.DataFrame({"x": xseq, "y": y_pred})

        if se and len(x) > 2:
            y_hat = np.polyval(coeffs, x)
            resid = y - y_hat
            mse = np.sum(w * resid ** 2) / (np.sum(w) - 2)
            x_mean = np.average(x, weights=w)
            ss_x = np.sum(w * (x - x_mean) ** 2)

            se_pred = np.sqrt(mse * (1.0 / np.sum(w) + (xseq - x_mean) ** 2 / ss_x))
            t_val = scipy_stats.t.ppf((1 + level) / 2, df=len(x) - 2)

            result["ymin"] = y_pred - t_val * se_pred
            result["ymax"] = y_pred + t_val * se_pred
            result["se"] = se_pred
        else:
            result["ymin"] = np.nan
            result["ymax"] = np.nan
            result["se"] = np.nan

        return result

    @staticmethod
    def _fit_lowess(
        data: pd.DataFrame,
        xseq: np.ndarray,
        se: bool,
        level: float,
        span: float = 0.75,
    ) -> pd.DataFrame:
        """Fit LOWESS smoother.

        Parameters
        ----------
        data : pd.DataFrame
        xseq : np.ndarray
        se : bool
        level : float
        span : float

        Returns
        -------
        pd.DataFrame
        """
        from scipy.interpolate import interp1d
        import statsmodels.api as sm

        lowess = sm.nonparametric.lowess
        x = data["x"].values
        y = data["y"].values
        result_raw = lowess(y, x, frac=span, return_sorted=True)
        f = interp1d(
            result_raw[:, 0], result_raw[:, 1],
            kind="linear", bounds_error=False, fill_value="extrapolate",
        )
        y_pred = f(xseq)

        result = pd.DataFrame({"x": xseq, "y": y_pred})

        if se:
            # Approximate confidence interval using residuals
            from scipy import stats as scipy_stats

            x = data["x"].values
            y = data["y"].values
            try:
                y_hat_data = f(x) if "f" in dir() else np.polyval(coeffs, x)
            except Exception:
                y_hat_data = y_pred[:len(x)]
            resid_var = np.var(y - y_hat_data[:len(y)], ddof=2) if len(y) > 2 else 0.0
            se_val = np.sqrt(resid_var)
            from scipy import stats as scipy_stats
            t_val = scipy_stats.t.ppf((1 + level) / 2, df=max(len(x) - 2, 1))
            result["ymin"] = y_pred - t_val * se_val
            result["ymax"] = y_pred + t_val * se_val
            result["se"] = se_val
        else:
            result["ymin"] = np.nan
            result["ymax"] = np.nan
            result["se"] = np.nan

        return result


def stat_smooth(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "smooth",
    position: str = "identity",
    method: Optional[str] = None,
    formula: Any = None,
    se: bool = True,
    n: int = 80,
    span: float = 0.75,
    fullrange: bool = False,
    level: float = 0.95,
    method_args: Optional[Dict[str, Any]] = None,
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Smoothing with confidence interval.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    method : str, optional
    formula : optional
    se : bool
    n : int
    span : float
    fullrange : bool
    level : float
    method_args : dict, optional
    na_rm : bool
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatSmooth,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "method": method, "formula": formula, "se": se, "n": n,
            "span": span, "fullrange": fullrange, "level": level,
            "method_args": method_args, "na_rm": na_rm,
            "orientation": orientation, **kwargs,
        },
    )


# ============================================================================
# StatBoxplot
# ============================================================================

class StatBoxplot(Stat):
    """Box-and-whisker statistics.

    Computes quantiles, whisker limits, and outliers for box plots.

    Attributes
    ----------
    required_aes : list
        ``["y|x"]``
    non_missing_aes : list
        ``["weight"]``
    optional_aes : list
        ``["width"]``
    dropped_aes : list
        ``["x", "y", "weight"]``
    """

    required_aes: List[str] = ["y|x"]
    non_missing_aes: List[str] = ["weight"]
    optional_aes: List[str] = ["width"]
    dropped_aes: List[str] = ["x", "y", "weight"]
    extra_params: List[str] = ["na_rm", "orientation"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(
            data, params,
            main_is_orthogonal=True,
            group_has_equal=True,
            main_is_optional=True,
            default=False,
        )
        data_f = _flip_data(data, params["flipped_aes"])

        if params.get("width") is None:
            x_vals = data_f["x"].values if "x" in data_f.columns else np.array([0])
            params["width"] = resolution(x_vals, discrete=True) * 0.75

        return params

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Ensure x column exists.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        data = _flip_data(data, params.get("flipped_aes", False))
        if "x" not in data.columns:
            data = data.copy()
            data["x"] = 0
        return _flip_data(data, params.get("flipped_aes", False))

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        width: Optional[float] = None,
        na_rm: bool = False,
        coef: float = 1.5,
        flipped_aes: bool = False,
        quantile_type: int = 7,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute boxplot statistics.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        width : float, optional
        na_rm : bool
        coef : float
            IQR multiplier for whiskers.
        flipped_aes : bool
        quantile_type : int

        Returns
        -------
        pd.DataFrame
            With columns: ymin, lower, middle, upper, ymax, outliers,
            notchupper, notchlower, x, width, relvarwidth.
        """
        data = _flip_data(data, flipped_aes)
        y = data["y"].values.astype(float)
        y = y[~np.isnan(y)]

        if len(y) == 0:
            return pd.DataFrame()

        # Compute quantiles
        qs = np.percentile(y, [0, 25, 50, 75, 100])
        ymin, lower, middle, upper, ymax = qs
        iqr = upper - lower

        # Identify outliers
        outliers = y[(y < lower - coef * iqr) | (y > upper + coef * iqr)]
        non_outliers = y[(y >= lower - coef * iqr) & (y <= upper + coef * iqr)]

        if len(non_outliers) > 0:
            ymin = np.min(non_outliers)
            ymax = np.max(non_outliers)

        # Width
        if "width" in data.columns and len(data["width"]) > 0:
            width = data["width"].iloc[0]
        elif width is None and len(data["x"].unique()) > 1:
            width = (data["x"].max() - data["x"].min()) * 0.9

        n = len(y)
        x_val = np.mean([data["x"].min(), data["x"].max()]) if "x" in data.columns else 0

        result = pd.DataFrame({
            "ymin": [ymin],
            "lower": [lower],
            "middle": [middle],
            "upper": [upper],
            "ymax": [ymax],
            "outliers": [list(outliers)],
            "notchupper": [middle + 1.58 * iqr / np.sqrt(n) if n > 0 else np.nan],
            "notchlower": [middle - 1.58 * iqr / np.sqrt(n) if n > 0 else np.nan],
            "x": [x_val],
            "width": [width],
            "relvarwidth": [np.sqrt(n)],
            "flipped_aes": [flipped_aes],
        })

        return _flip_data(result, flipped_aes)


def stat_boxplot(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "boxplot",
    position: str = "dodge2",
    coef: float = 1.5,
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Compute boxplot statistics.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    coef : float
    na_rm : bool
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatBoxplot,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"coef": coef, "na_rm": na_rm, "orientation": orientation, **kwargs},
    )


# ============================================================================
# StatSummary
# ============================================================================

class StatSummary(Stat):
    """Summarise y values at unique x positions.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    extra_params : list
        Includes ``fun_data``, ``fun_max``, ``fun_min``, ``fun_args``.
    """

    required_aes: List[str] = ["x", "y"]
    extra_params: List[str] = [
        "na_rm", "orientation", "fun_data", "fun_max", "fun_min", "fun_args",
    ]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build summary function.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(data, params)
        params["fun"] = _make_summary_fun(
            params.get("fun_data"),
            params.get("fun"),
            params.get("fun_max"),
            params.get("fun_min"),
            params.get("fun_args", {}),
        )
        return params

    def compute_panel(
        self,
        data: pd.DataFrame,
        scales: Any,
        fun: Optional[Callable] = None,
        na_rm: bool = False,
        flipped_aes: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Summarise by unique x value.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        fun : callable, optional
        na_rm : bool
        flipped_aes : bool

        Returns
        -------
        pd.DataFrame
        """
        data = _flip_data(data, flipped_aes)

        if fun is None:
            fun = lambda df: mean_se(df["y"])

        results = []
        group_cols = ["group", "x"] if "group" in data.columns else ["x"]
        for keys, grp in data.groupby(group_cols, sort=False):
            try:
                summary = fun(grp)
            except Exception:
                summary = mean_se(grp["y"])
            if isinstance(summary, pd.DataFrame):
                for col in group_cols:
                    if col in grp.columns:
                        summary[col] = grp[col].iloc[0]
                results.append(summary)

        if not results:
            return pd.DataFrame()

        out = pd.concat(results, ignore_index=True)
        out["flipped_aes"] = flipped_aes
        return _flip_data(out, flipped_aes)


def stat_summary(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "pointrange",
    position: str = "identity",
    fun_data: Any = None,
    fun: Any = None,
    fun_max: Any = None,
    fun_min: Any = None,
    fun_args: Optional[Dict[str, Any]] = None,
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Summarise y values at unique x.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    fun_data, fun, fun_max, fun_min : callable or str, optional
    fun_args : dict, optional
    na_rm : bool
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatSummary,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "fun_data": fun_data, "fun": fun, "fun_max": fun_max,
            "fun_min": fun_min, "fun_args": fun_args or {},
            "na_rm": na_rm, "orientation": orientation, **kwargs,
        },
    )


# ============================================================================
# StatSummaryBin
# ============================================================================

class StatSummaryBin(Stat):
    """Summarise y values in binned x.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    """

    required_aes: List[str] = ["x", "y"]
    extra_params: List[str] = [
        "na_rm", "orientation", "fun_data", "fun_max", "fun_min", "fun_args",
    ]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build summary function.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(data, params)
        params["fun"] = _make_summary_fun(
            params.get("fun_data"),
            params.get("fun"),
            params.get("fun_max"),
            params.get("fun_min"),
            params.get("fun_args", {}),
        )
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        fun: Optional[Callable] = None,
        bins: int = 30,
        binwidth: Optional[float] = None,
        breaks: Any = None,
        na_rm: bool = False,
        flipped_aes: bool = False,
        width: Optional[float] = None,
        center: Optional[float] = None,
        boundary: Optional[float] = None,
        closed: str = "right",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Bin x then summarise y within each bin.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        fun : callable, optional
        bins, binwidth, breaks : optional
        na_rm : bool
        flipped_aes : bool
        width : float, optional
        center, boundary : optional
        closed : str

        Returns
        -------
        pd.DataFrame
        """
        x_col = "y" if flipped_aes else "x"
        x_vals = data[x_col].values if x_col in data.columns else data["x"].values
        scale = scales.get(x_col) if isinstance(scales, dict) else getattr(scales, x_col, None)

        bins_obj = _compute_bins(
            x_vals, scale,
            breaks=breaks, binwidth=binwidth, bins=bins,
            center=center, boundary=boundary, closed=closed,
        )

        bin_idx = _bin_cut(x_vals, bins_obj)

        data = _flip_data(data, flipped_aes)
        data = data.copy()
        data["bin"] = bin_idx

        if fun is None:
            fun = lambda df: mean_se(df["y"])

        results = []
        for bin_val, grp in data.groupby("bin", sort=True):
            try:
                summary = fun(grp)
            except Exception:
                summary = mean_se(grp["y"])
            if isinstance(summary, pd.DataFrame):
                summary["bin"] = bin_val
                results.append(summary)

        if not results:
            return pd.DataFrame()

        out = pd.concat(results, ignore_index=True)

        # Compute bin locations
        locs = _bin_loc(bins_obj.breaks, out["bin"].values)
        out["x"] = locs["mid"]
        out["width"] = width if width is not None else locs["length"]
        out["flipped_aes"] = flipped_aes

        return _flip_data(out, flipped_aes)


def stat_summary_bin(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "pointrange",
    position: str = "identity",
    fun_data: Any = None,
    fun: Any = None,
    fun_max: Any = None,
    fun_min: Any = None,
    fun_args: Optional[Dict[str, Any]] = None,
    bins: int = 30,
    binwidth: Optional[float] = None,
    breaks: Any = None,
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Summarise y values in binned x.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    fun_data, fun, fun_max, fun_min : optional
    fun_args : dict, optional
    bins, binwidth, breaks : optional
    na_rm : bool
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatSummaryBin,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "fun_data": fun_data, "fun": fun, "fun_max": fun_max,
            "fun_min": fun_min, "fun_args": fun_args or {},
            "bins": bins, "binwidth": binwidth, "breaks": breaks,
            "na_rm": na_rm, "orientation": orientation, **kwargs,
        },
    )


# ============================================================================
# StatSummary2d
# ============================================================================

class StatSummary2d(Stat):
    """2D binning and summary.

    Attributes
    ----------
    required_aes : list
        ``["x", "y", "z"]``
    dropped_aes : list
        ``["z"]``
    """

    required_aes: List[str] = ["x", "y", "z"]
    dropped_aes: List[str] = ["z"]
    extra_params: List[str] = ["na_rm", "origin"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        binwidth: Any = None,
        bins: int = 30,
        breaks: Any = None,
        drop: bool = True,
        fun: Any = "mean",
        fun_args: Optional[Dict[str, Any]] = None,
        boundary: Any = None,
        closed: Any = None,
        center: Any = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Bin in 2D and summarise z.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        binwidth, bins, breaks : optional
        drop : bool
        fun : callable or str
        fun_args : dict, optional
        boundary, closed, center : optional

        Returns
        -------
        pd.DataFrame
            With columns: x, y, value, width, height.
        """
        if fun_args is None:
            fun_args = {}

        if isinstance(fun, str):
            fun_map = {
                "mean": np.nanmean, "sum": np.nansum, "median": np.nanmedian,
                "min": np.nanmin, "max": np.nanmax, "var": np.nanvar,
                "std": np.nanstd, "count": len,
            }
            fun = fun_map.get(fun, np.nanmean)

        bins_d = _dual_param(bins, {"x": 30, "y": 30})
        binwidth_d = _dual_param(binwidth)
        breaks_d = _dual_param(breaks)
        boundary_d = _dual_param(boundary, {"x": 0, "y": 0})
        closed_d = _dual_param(closed, {"x": "right", "y": "right"})
        center_d = _dual_param(center)

        x_scale = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
        y_scale = scales.get("y") if isinstance(scales, dict) else getattr(scales, "y", None)

        xbin = _compute_bins(
            data["x"].values, x_scale,
            breaks=breaks_d.get("x"), binwidth=binwidth_d.get("x"),
            bins=bins_d.get("x"), center=center_d.get("x"),
            boundary=boundary_d.get("x"), closed=closed_d.get("x", "right"),
        )
        ybin = _compute_bins(
            data["y"].values, y_scale,
            breaks=breaks_d.get("y"), binwidth=binwidth_d.get("y"),
            bins=bins_d.get("y"), center=center_d.get("y"),
            boundary=boundary_d.get("y"), closed=closed_d.get("y", "right"),
        )

        xidx = _bin_cut(data["x"].values, xbin)
        yidx = _bin_cut(data["y"].values, ybin)

        # Aggregate
        z_vals = data["z"].values
        keys = {}
        for i in range(len(z_vals)):
            key = (int(xidx[i]), int(yidx[i]))
            if key not in keys:
                keys[key] = []
            keys[key].append(z_vals[i])

        rows = []
        for (xi, yi), zs in keys.items():
            val = fun(np.array(zs), **fun_args)
            x_loc = _bin_loc(xbin.breaks, np.array([xi]))
            y_loc = _bin_loc(ybin.breaks, np.array([yi]))
            rows.append({
                "x": x_loc["mid"][0],
                "y": y_loc["mid"][0],
                "value": val,
                "width": x_loc["length"][0],
                "height": y_loc["length"][0],
            })

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        if drop:
            out = out.dropna(subset=["value"]).reset_index(drop=True)
        return out


def stat_summary_2d(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "tile",
    position: str = "identity",
    bins: int = 30,
    binwidth: Any = None,
    breaks: Any = None,
    drop: bool = True,
    fun: Any = "mean",
    fun_args: Optional[Dict[str, Any]] = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """2D binning and summary.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    bins, binwidth, breaks : optional
    drop : bool
    fun : callable or str
    fun_args : dict, optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatSummary2d,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "bins": bins, "binwidth": binwidth, "breaks": breaks,
            "drop": drop, "fun": fun, "fun_args": fun_args or {},
            "na_rm": na_rm, **kwargs,
        },
    )


stat_summary2d = stat_summary_2d


# ============================================================================
# StatSummaryHex
# ============================================================================

class StatSummaryHex(Stat):
    """Hexagonal binning and summary.

    Attributes
    ----------
    required_aes : list
        ``["x", "y", "z"]``
    dropped_aes : list
        ``["z"]``
    """

    required_aes: List[str] = ["x", "y", "z"]
    dropped_aes: List[str] = ["z"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        binwidth: Any = None,
        bins: int = 30,
        drop: bool = True,
        fun: Any = "mean",
        fun_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Hex-bin and summarise z.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        binwidth : optional
        bins : int
        drop : bool
        fun : callable or str
        fun_args : dict, optional

        Returns
        -------
        pd.DataFrame
        """
        if fun_args is None:
            fun_args = {}

        if isinstance(fun, str):
            fun_map = {
                "mean": np.nanmean, "sum": np.nansum, "median": np.nanmedian,
                "min": np.nanmin, "max": np.nanmax,
            }
            fun = fun_map.get(fun, np.nanmean)

        if binwidth is None:
            binwidth = _hex_binwidth(bins, scales)

        if not isinstance(binwidth, (list, tuple)):
            binwidth = (binwidth, binwidth)

        return _hex_bin_summarise(
            data["x"].values, data["y"].values, data["z"].values,
            binwidth, fun=fun, fun_args=fun_args, drop=drop,
        )


def stat_summary_hex(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "hex",
    position: str = "identity",
    bins: int = 30,
    binwidth: Any = None,
    drop: bool = True,
    fun: Any = "mean",
    fun_args: Optional[Dict[str, Any]] = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Hexagonal binning and summary.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    bins, binwidth : optional
    drop : bool
    fun : callable or str
    fun_args : dict, optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatSummaryHex,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "bins": bins, "binwidth": binwidth, "drop": drop,
            "fun": fun, "fun_args": fun_args or {},
            "na_rm": na_rm, **kwargs,
        },
    )


# ============================================================================
# StatFunction
# ============================================================================

class StatFunction(Stat):
    """Evaluate a function over an x range.

    Attributes
    ----------
    default_aes : dict
        ``x=None``
    """

    default_aes: Dict[str, Any] = {"x": None}

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        fun: Optional[Callable] = None,
        xlim: Optional[Tuple[float, float]] = None,
        n: int = 101,
        args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Evaluate function on a grid.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        fun : callable
        xlim : tuple, optional
        n : int
        args : dict, optional

        Returns
        -------
        pd.DataFrame
            With columns: x, y.
        """
        if fun is None:
            cli_abort("stat_function requires a `fun` argument.")

        if args is None:
            args = {}

        scale_x = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
        if scale_x is None:
            rng = xlim if xlim is not None else (0, 1)
        else:
            rng = xlim if xlim is not None else tuple(scale_x.dimension())

        xseq = np.linspace(rng[0], rng[1], n)
        y_out = fun(xseq, **args)

        return pd.DataFrame({"x": xseq, "y": y_out})


def stat_function(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "function",
    position: str = "identity",
    fun: Optional[Callable] = None,
    xlim: Optional[Tuple[float, float]] = None,
    n: int = 101,
    args: Optional[Dict[str, Any]] = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Evaluate a function over x range.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    fun : callable
    xlim : tuple, optional
    n : int
    args : dict, optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    if data is None:
        data = pd.DataFrame({"group": [1]})
    return _layer(
        stat=StatFunction,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "fun": fun, "xlim": xlim, "n": n, "args": args or {},
            "na_rm": na_rm, **kwargs,
        },
    )


# ============================================================================
# StatEcdf
# ============================================================================

class StatEcdf(Stat):
    """Empirical cumulative distribution function.

    Attributes
    ----------
    required_aes : list
        ``["x|y"]``
    default_aes : dict
        ``weight=None``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x|y"]
    default_aes: Dict[str, Any] = {"weight": None}
    dropped_aes: List[str] = ["weight"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(
            data, params, main_is_orthogonal=False, main_is_continuous=True,
        )
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        n: Optional[int] = None,
        pad: bool = True,
        flipped_aes: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute ECDF.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        n : int, optional
        pad : bool
        flipped_aes : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y, ecdf.
        """
        data = _flip_data(data, flipped_aes)
        x_vals = data["x"].values

        if n is None:
            x_grid = np.sort(np.unique(x_vals))
        else:
            x_grid = np.linspace(np.min(x_vals), np.max(x_vals), n)

        if pad:
            x_grid = np.concatenate([[-np.inf], x_grid, [np.inf]])

        weight = data["weight"].values if "weight" in data.columns and data["weight"].notna().any() else None
        ecdf_fun = _wecdf(x_vals, weight)
        ecdf_vals = ecdf_fun(x_grid)

        result = pd.DataFrame({
            "x": x_grid,
            "y": ecdf_vals,
            "ecdf": ecdf_vals,
            "flipped_aes": flipped_aes,
        })

        return _flip_data(result, flipped_aes)


def stat_ecdf(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "step",
    position: str = "identity",
    n: Optional[int] = None,
    pad: bool = True,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Compute empirical CDF.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    n : int, optional
    pad : bool
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatEcdf,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"n": n, "pad": pad, "na_rm": na_rm, **kwargs},
    )


# ============================================================================
# StatQq
# ============================================================================

class StatQq(Stat):
    """Q-Q plot statistics.

    Attributes
    ----------
    required_aes : list
        ``["sample"]``
    """

    required_aes: List[str] = ["sample"]
    default_aes: Dict[str, Any] = {}

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        quantiles: Optional[np.ndarray] = None,
        distribution: Any = None,
        dparams: Optional[Dict[str, Any]] = None,
        na_rm: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute Q-Q plot points.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        quantiles : array-like, optional
        distribution : scipy distribution, optional
        dparams : dict, optional
        na_rm : bool

        Returns
        -------
        pd.DataFrame
            With columns: sample, theoretical.
        """
        from scipy import stats as scipy_stats

        sample = np.sort(data["sample"].values)
        n = len(sample)

        if quantiles is None:
            quantiles = (np.arange(1, n + 1) - 0.5) / n
        elif len(quantiles) != n:
            cli_abort(
                f"The length of quantiles ({len(quantiles)}) must match "
                f"the length of the data ({n})."
            )

        if distribution is None:
            distribution = scipy_stats.norm

        if dparams is None:
            dparams = {}

        if hasattr(distribution, "ppf"):
            theoretical = distribution.ppf(quantiles, **dparams)
        elif callable(distribution):
            theoretical = distribution(quantiles, **dparams)
        else:
            theoretical = scipy_stats.norm.ppf(quantiles)

        return pd.DataFrame({"sample": sample, "theoretical": theoretical})


def stat_qq(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    distribution: Any = None,
    dparams: Optional[Dict[str, Any]] = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Q-Q plot.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    distribution : scipy distribution, optional
    dparams : dict, optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatQq,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "distribution": distribution, "dparams": dparams or {},
            "na_rm": na_rm, **kwargs,
        },
    )


# ============================================================================
# StatQqLine
# ============================================================================

class StatQqLine(Stat):
    """Q-Q line (reference line through chosen quantiles).

    Attributes
    ----------
    required_aes : list
        ``["sample"]``
    dropped_aes : list
        ``["sample"]``
    """

    required_aes: List[str] = ["sample"]
    dropped_aes: List[str] = ["sample"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        quantiles: Optional[np.ndarray] = None,
        distribution: Any = None,
        dparams: Optional[Dict[str, Any]] = None,
        na_rm: bool = False,
        line_p: Tuple[float, float] = (0.25, 0.75),
        fullrange: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute Q-Q line endpoints.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        quantiles : array-like, optional
        distribution : scipy distribution, optional
        dparams : dict, optional
        na_rm : bool
        line_p : tuple
        fullrange : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y, slope, intercept.
        """
        from scipy import stats as scipy_stats

        sample = np.sort(data["sample"].values)
        n = len(sample)

        if quantiles is None:
            quantiles = (np.arange(1, n + 1) - 0.5) / n

        if distribution is None:
            distribution = scipy_stats.norm

        if dparams is None:
            dparams = {}

        if hasattr(distribution, "ppf"):
            theoretical = distribution.ppf(quantiles, **dparams)
            x_coords = distribution.ppf(np.array(line_p), **dparams)
        elif callable(distribution):
            theoretical = distribution(quantiles, **dparams)
            x_coords = distribution(np.array(line_p), **dparams)
        else:
            theoretical = scipy_stats.norm.ppf(quantiles)
            x_coords = scipy_stats.norm.ppf(np.array(line_p))

        y_coords = np.percentile(sample, np.array(line_p) * 100)

        slope = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0]) if x_coords[1] != x_coords[0] else 0
        intercept = y_coords[0] - slope * x_coords[0]

        if fullrange:
            scale_x = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
            if scale_x is not None and hasattr(scale_x, "dimension"):
                x_range = np.array(scale_x.dimension())
            else:
                x_range = np.array([theoretical.min(), theoretical.max()])
        else:
            x_range = np.array([theoretical.min(), theoretical.max()])

        return pd.DataFrame({
            "x": x_range,
            "y": slope * x_range + intercept,
            "slope": slope,
            "intercept": intercept,
        })


def stat_qq_line(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "abline",
    position: str = "identity",
    distribution: Any = None,
    dparams: Optional[Dict[str, Any]] = None,
    line_p: Tuple[float, float] = (0.25, 0.75),
    fullrange: bool = False,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Q-Q reference line.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    distribution : optional
    dparams : dict, optional
    line_p : tuple
    fullrange : bool
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatQqLine,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "distribution": distribution, "dparams": dparams or {},
            "line_p": line_p, "fullrange": fullrange,
            "na_rm": na_rm, **kwargs,
        },
    )


# ============================================================================
# StatBin2d
# ============================================================================

class StatBin2d(Stat):
    """2D rectangular binning.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    default_aes : dict
        ``weight=1``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x", "y"]
    default_aes: Dict[str, Any] = {"weight": 1}
    dropped_aes: List[str] = ["weight"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        binwidth: Any = None,
        bins: int = 30,
        breaks: Any = None,
        drop: bool = True,
        boundary: Any = None,
        closed: Any = None,
        center: Any = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Bin in 2D rectangles and count.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        binwidth, bins, breaks : optional
        drop : bool
        boundary, closed, center : optional

        Returns
        -------
        pd.DataFrame
            With columns: x, y, count, ncount, density, ndensity, width, height.
        """
        # Set z = weight for counting
        data = data.copy()
        data["z"] = data["weight"].values if "weight" in data.columns else 1.0

        if boundary is None and center is None:
            boundary = {"x": 0, "y": 0}

        out = StatSummary2d.compute_group(
            StatSummary2d(), data, scales,
            binwidth=binwidth, bins=bins, breaks=breaks, drop=drop,
            fun=np.nansum, boundary=boundary, closed=closed, center=center,
        )

        if out.empty:
            return out

        out["count"] = out["value"]
        max_count = out["count"].max()
        out["ncount"] = out["count"] / max_count if max_count > 0 else 0
        total_count = out["count"].sum()
        out["density"] = out["count"] / total_count if total_count > 0 else 0
        max_density = out["density"].max()
        out["ndensity"] = out["density"] / max_density if max_density > 0 else 0

        return out


def stat_bin_2d(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "tile",
    position: str = "identity",
    bins: int = 30,
    binwidth: Any = None,
    breaks: Any = None,
    drop: bool = True,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """2D rectangular binning.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    bins, binwidth, breaks : optional
    drop : bool
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatBin2d,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "bins": bins, "binwidth": binwidth, "breaks": breaks,
            "drop": drop, "na_rm": na_rm, **kwargs,
        },
    )


stat_bin2d = stat_bin_2d


# ============================================================================
# StatBinhex
# ============================================================================

class StatBinhex(Stat):
    """Hexagonal binning.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    default_aes : dict
        ``weight=1``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x", "y"]
    default_aes: Dict[str, Any] = {"weight": 1}
    dropped_aes: List[str] = ["weight"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        binwidth: Any = None,
        bins: int = 30,
        na_rm: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Hex-bin and count.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        binwidth : optional
        bins : int
        na_rm : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y, count, density, ncount, ndensity.
        """
        if binwidth is None:
            binwidth = _hex_binwidth(bins, scales)
        if not isinstance(binwidth, (list, tuple)):
            binwidth = (binwidth, binwidth)

        wt = data["weight"].values if "weight" in data.columns else np.ones(len(data))
        out = _hex_bin_summarise(data["x"].values, data["y"].values, wt, binwidth, fun=np.sum)

        if out.empty:
            return out

        total = out["value"].sum()
        out["density"] = out["value"] / total if total > 0 else 0
        max_dens = out["density"].max()
        out["ndensity"] = out["density"] / max_dens if max_dens > 0 else 0
        out["count"] = out["value"]
        max_count = out["count"].max()
        out["ncount"] = out["count"] / max_count if max_count > 0 else 0
        out.drop(columns=["value"], inplace=True, errors="ignore")

        return out


def stat_bin_hex(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "hex",
    position: str = "identity",
    bins: int = 30,
    binwidth: Any = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Hexagonal binning.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    bins, binwidth : optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatBinhex,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"bins": bins, "binwidth": binwidth, "na_rm": na_rm, **kwargs},
    )


stat_binhex = stat_bin_hex


# ============================================================================
# StatContour / StatContourFilled
# ============================================================================

class StatContour(Stat):
    """Contour lines from gridded data.

    Attributes
    ----------
    required_aes : list
        ``["x", "y", "z"]``
    dropped_aes : list
        ``["z", "weight"]``
    """

    required_aes: List[str] = ["x", "y", "z"]
    dropped_aes: List[str] = ["z", "weight"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record z range.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        z_vals = data["z"].values[np.isfinite(data["z"].values)]
        params["z_range"] = [float(z_vals.min()), float(z_vals.max())] if len(z_vals) > 0 else [0.0, 1.0]
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        z_range: Optional[List[float]] = None,
        bins: Optional[int] = None,
        binwidth: Optional[float] = None,
        breaks: Any = None,
        na_rm: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute contour lines.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        z_range : list, optional
        bins, binwidth, breaks : optional
        na_rm : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y, level, piece, group.
        """
        if z_range is None:
            z_vals = data["z"].values[np.isfinite(data["z"].values)]
            z_range = [z_vals.min(), z_vals.max()] if len(z_vals) > 0 else [0, 1]

        brks = _contour_breaks(z_range, bins, binwidth, breaks)

        # Build z matrix
        x_unique = np.sort(data["x"].unique())
        y_unique = np.sort(data["y"].unique())

        z_matrix = np.full((len(y_unique), len(x_unique)), np.nan)
        x_idx = np.searchsorted(x_unique, data["x"].values)
        y_idx = np.searchsorted(y_unique, data["y"].values)
        for i in range(len(data)):
            if x_idx[i] < len(x_unique) and y_idx[i] < len(y_unique):
                z_matrix[y_idx[i], x_idx[i]] = data["z"].values[i]

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            cs = ax.contour(x_unique, y_unique, z_matrix, levels=brks)

            rows = []
            group_base = data["group"].iloc[0] if "group" in data.columns else 1
            piece = 0
            for i, level_val in enumerate(cs.levels):
                for seg in cs.allsegs[i]:
                    piece += 1
                    for pt in seg:
                        rows.append({
                            "x": pt[0],
                            "y": pt[1],
                            "level": float(level_val),
                            "piece": piece,
                            "group": f"{group_base}-{piece:03d}",
                        })
            plt.close(fig)
        except Exception:
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        max_level = result["level"].max()
        result["nlevel"] = result["level"] / max_level if max_level > 0 else 0
        return result


class StatContourFilled(Stat):
    """Filled contour bands from gridded data.

    Attributes
    ----------
    required_aes : list
        ``["x", "y", "z"]``
    dropped_aes : list
        ``["z", "weight"]``
    """

    required_aes: List[str] = ["x", "y", "z"]
    dropped_aes: List[str] = ["z", "weight"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record z range.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        z_vals = data["z"].values[np.isfinite(data["z"].values)]
        params["z_range"] = [float(z_vals.min()), float(z_vals.max())] if len(z_vals) > 0 else [0.0, 1.0]
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        z_range: Optional[List[float]] = None,
        bins: Optional[int] = None,
        binwidth: Optional[float] = None,
        breaks: Any = None,
        na_rm: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute filled contour bands.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        z_range : list, optional
        bins, binwidth, breaks : optional
        na_rm : bool

        Returns
        -------
        pd.DataFrame
        """
        if z_range is None:
            z_vals = data["z"].values[np.isfinite(data["z"].values)]
            z_range = [z_vals.min(), z_vals.max()] if len(z_vals) > 0 else [0, 1]

        brks = _contour_breaks(z_range, bins, binwidth, breaks)

        # Build z matrix
        x_unique = np.sort(data["x"].unique())
        y_unique = np.sort(data["y"].unique())

        z_matrix = np.full((len(y_unique), len(x_unique)), np.nan)
        x_idx = np.searchsorted(x_unique, data["x"].values)
        y_idx = np.searchsorted(y_unique, data["y"].values)
        for i in range(len(data)):
            if x_idx[i] < len(x_unique) and y_idx[i] < len(y_unique):
                z_matrix[y_idx[i], x_idx[i]] = data["z"].values[i]

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            cs = ax.contourf(x_unique, y_unique, z_matrix, levels=brks)

            rows = []
            group_base = data["group"].iloc[0] if "group" in data.columns else 1
            piece = 0
            for i, collection in enumerate(cs.collections):
                level_low = brks[i] if i < len(brks) else brks[-1]
                level_high = brks[i + 1] if i + 1 < len(brks) else brks[-1]
                for path in collection.get_paths():
                    piece += 1
                    vertices = path.vertices
                    for pt in vertices:
                        rows.append({
                            "x": pt[0],
                            "y": pt[1],
                            "level": f"({level_low}, {level_high}]",
                            "level_low": level_low,
                            "level_high": level_high,
                            "level_mid": (level_low + level_high) / 2,
                            "piece": piece,
                            "group": f"{group_base}-{piece:03d}",
                        })
            plt.close(fig)
        except Exception:
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        max_level = result["level_high"].max() if "level_high" in result.columns else 1
        result["nlevel"] = result["level_high"] / max_level if max_level > 0 else 0
        return result


def stat_contour(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "contour",
    position: str = "identity",
    bins: Optional[int] = None,
    binwidth: Optional[float] = None,
    breaks: Any = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Contour lines from gridded data.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    bins, binwidth, breaks : optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatContour,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "bins": bins, "binwidth": binwidth, "breaks": breaks,
            "na_rm": na_rm, **kwargs,
        },
    )


def stat_contour_filled(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "contour_filled",
    position: str = "identity",
    bins: Optional[int] = None,
    binwidth: Optional[float] = None,
    breaks: Any = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Filled contour bands.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    bins, binwidth, breaks : optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatContourFilled,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "bins": bins, "binwidth": binwidth, "breaks": breaks,
            "na_rm": na_rm, **kwargs,
        },
    )


# ============================================================================
# StatDensity2d / StatDensity2dFilled
# ============================================================================

class StatDensity2d(Stat):
    """2D kernel density estimation.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    default_aes : dict
        ``colour="#3366FF", size=0.5``
    """

    required_aes: List[str] = ["x", "y"]
    default_aes: Dict[str, Any] = {"colour": "#3366FF", "size": 0.5}
    dropped_aes: List[str] = []
    extra_params: List[str] = [
        "na_rm", "contour", "contour_var", "bins", "binwidth", "breaks",
    ]
    contour_type: str = "lines"

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        na_rm: bool = False,
        h: Optional[Tuple[float, float]] = None,
        adjust: Union[float, Tuple[float, float]] = (1.0, 1.0),
        n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute 2D density on a grid.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        na_rm : bool
        h : tuple, optional
            Bandwidth vector (length 2).
        adjust : tuple
        n : int

        Returns
        -------
        pd.DataFrame
            With columns: x, y, density, ndensity, count, n, level, piece.
        """
        from scipy import stats as scipy_stats

        x = data["x"].values
        y = data["y"].values
        nx = len(x)

        if isinstance(adjust, (int, float)):
            adjust = (adjust, adjust)

        if h is None:
            h_x = _precompute_bw(x) * 4 * adjust[0]
            h_y = _precompute_bw(y) * 4 * adjust[1]
        else:
            h_x, h_y = h

        # Get evaluation range
        x_scale = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
        y_scale = scales.get("y") if isinstance(scales, dict) else getattr(scales, "y", None)

        if x_scale is not None and hasattr(x_scale, "dimension"):
            x_range = x_scale.dimension()
        else:
            x_range = (x.min(), x.max())

        if y_scale is not None and hasattr(y_scale, "dimension"):
            y_range = y_scale.dimension()
        else:
            y_range = (y.min(), y.max())

        # Compute kde
        try:
            values = np.vstack([x, y])
            kde = scipy_stats.gaussian_kde(values)
            x_grid = np.linspace(x_range[0], x_range[1], n)
            y_grid = np.linspace(y_range[0], y_range[1], n)
            xx, yy = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(positions).reshape(n, n)
        except Exception:
            return pd.DataFrame()

        # Expand grid
        df = pd.DataFrame({
            "x": xx.ravel(),
            "y": yy.ravel(),
            "density": z.ravel(),
        })
        group_val = data["group"].iloc[0] if "group" in data.columns else 1
        df["group"] = group_val
        max_dens = df["density"].max()
        df["ndensity"] = df["density"] / max_dens if max_dens > 0 else 0
        df["count"] = nx * df["density"]
        df["n"] = nx
        df["level"] = 1
        df["piece"] = 1

        return df


class StatDensity2dFilled(StatDensity2d):
    """2D density with filled contour bands.

    Attributes
    ----------
    contour_type : str
        ``"bands"``
    """

    default_aes: Dict[str, Any] = {"colour": None, "fill": None}
    contour_type: str = "bands"


def stat_density_2d(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "density_2d",
    position: str = "identity",
    contour: bool = True,
    contour_var: str = "density",
    n: int = 100,
    h: Optional[Tuple[float, float]] = None,
    adjust: Union[float, Tuple[float, float]] = (1.0, 1.0),
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """2D kernel density estimation.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    contour : bool
    contour_var : str
    n : int
    h : tuple, optional
    adjust : float or tuple
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatDensity2d,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "contour": contour, "contour_var": contour_var, "n": n,
            "h": h, "adjust": adjust, "na_rm": na_rm, **kwargs,
        },
    )


stat_density2d = stat_density_2d


def stat_density_2d_filled(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "density_2d_filled",
    position: str = "identity",
    contour: bool = True,
    contour_var: str = "density",
    n: int = 100,
    h: Optional[Tuple[float, float]] = None,
    adjust: Union[float, Tuple[float, float]] = (1.0, 1.0),
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """2D density with filled contours.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    contour : bool
    contour_var : str
    n : int
    h : tuple, optional
    adjust : float or tuple
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatDensity2dFilled,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "contour": contour, "contour_var": contour_var, "n": n,
            "h": h, "adjust": adjust, "na_rm": na_rm, **kwargs,
        },
    )


stat_density2d_filled = stat_density_2d_filled


# ============================================================================
# StatEllipse
# ============================================================================

class StatEllipse(Stat):
    """Confidence ellipse.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    optional_aes : list
        ``["weight"]``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x", "y"]
    optional_aes: List[str] = ["weight"]
    dropped_aes: List[str] = ["weight"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        type: str = "t",
        level: float = 0.95,
        segments: int = 51,
        na_rm: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute ellipse points.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        type : str
            ``"t"``, ``"norm"``, or ``"euclid"``.
        level : float
        segments : int
        na_rm : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y.
        """
        from scipy import stats as scipy_stats

        x = data["x"].values.astype(float)
        y = data["y"].values.astype(float)
        n = len(x)

        if n < 3:
            cli_warn("Too few points to calculate an ellipse.")
            return pd.DataFrame({"x": [np.nan], "y": [np.nan]})

        weight = data["weight"].values if "weight" in data.columns else np.ones(n)
        weight = weight / np.sum(weight)

        xy = np.column_stack([x, y])

        if type == "t":
            # Robust covariance (simplified)
            center = np.average(xy, axis=0, weights=weight)
            diff = xy - center
            cov_mat = np.cov(diff.T, aweights=weight)
        elif type in ("norm", "euclid"):
            center = np.average(xy, axis=0, weights=weight)
            diff = xy - center
            cov_mat = np.cov(diff.T, aweights=weight)
            if type == "euclid":
                min_var = np.min(np.diag(cov_mat))
                cov_mat = np.diag([min_var, min_var])
        else:
            return pd.DataFrame({"x": [np.nan], "y": [np.nan]})

        try:
            chol = np.linalg.cholesky(cov_mat)
        except np.linalg.LinAlgError:
            return pd.DataFrame({"x": [np.nan], "y": [np.nan]})

        dfn = 2
        dfd = n - 1

        if type == "euclid":
            radius = level / np.max(chol)
        else:
            radius = np.sqrt(dfn * scipy_stats.f.ppf(level, dfn, dfd))

        angles = np.linspace(0, 2 * np.pi, segments + 1)
        unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])
        ellipse = center + radius * (unit_circle @ chol.T)

        return pd.DataFrame({"x": ellipse[:, 0], "y": ellipse[:, 1]})


def stat_ellipse(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "path",
    position: str = "identity",
    type: str = "t",
    level: float = 0.95,
    segments: int = 51,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Compute confidence ellipse.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    type : str
    level : float
    segments : int
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatEllipse,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "type": type, "level": level, "segments": segments,
            "na_rm": na_rm, **kwargs,
        },
    )


# ============================================================================
# StatUnique
# ============================================================================

class StatUnique(Stat):
    """Remove duplicate rows.

    Keeps only unique rows in the data.
    """

    def compute_panel(self, data: pd.DataFrame, scales: Any, **params: Any) -> pd.DataFrame:
        """Return unique rows.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like

        Returns
        -------
        pd.DataFrame
        """
        return data.drop_duplicates().reset_index(drop=True)


def stat_unique(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Remove duplicate observations.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatUnique,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


# ============================================================================
# StatSum
# ============================================================================

class StatSum(Stat):
    """Count coincident points.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    default_aes : dict
        ``weight=1``
    """

    required_aes: List[str] = ["x", "y"]
    default_aes: Dict[str, Any] = {"weight": 1}

    def compute_panel(self, data: pd.DataFrame, scales: Any, **params: Any) -> pd.DataFrame:
        """Count coincident (x, y) points.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like

        Returns
        -------
        pd.DataFrame
            With columns: x, y, n, prop, plus original constant columns.
        """
        if "weight" not in data.columns:
            data = data.copy()
            data["weight"] = 1

        # Group by all aesthetic columns except weight
        group_cols = [c for c in data.columns if c != "weight"]
        grouped = data.groupby(group_cols, sort=False, dropna=False)
        counts = grouped["weight"].sum().reset_index()
        counts.rename(columns={"weight": "n"}, inplace=True)

        # Compute prop within each group
        if "group" in counts.columns:
            group_totals = counts.groupby("group")["n"].transform("sum")
            counts["prop"] = counts["n"] / group_totals
        else:
            counts["prop"] = counts["n"] / counts["n"].sum()

        return counts


def stat_sum(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Count coincident points.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatSum,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


# ============================================================================
# StatYdensity
# ============================================================================

class StatYdensity(Stat):
    """Violin plot density (y density per group at each x).

    Attributes
    ----------
    required_aes : list
        ``["x|y"]``
    non_missing_aes : list
        ``["weight"]``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x|y"]
    non_missing_aes: List[str] = ["weight"]
    dropped_aes: List[str] = ["weight"]
    extra_params: List[str] = ["na_rm", "orientation"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(
            data, params, main_is_orthogonal=True, group_has_equal=True,
        )
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        width: Optional[float] = None,
        bw: Union[str, float] = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        trim: bool = True,
        na_rm: bool = False,
        drop: bool = True,
        flipped_aes: bool = False,
        bounds: Tuple[float, float] = (-np.inf, np.inf),
        quantiles: Optional[Sequence[float]] = (0.25, 0.50, 0.75),
        scale: str = "area",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute density of y for violin plots.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        width : float, optional
        bw : str or float
        adjust : float
        kernel : str
        trim : bool
        na_rm : bool
        drop : bool
        flipped_aes : bool
        bounds : tuple
        quantiles : sequence of float, optional
        scale : str

        Returns
        -------
        pd.DataFrame
        """
        if len(data) < 2:
            if drop:
                cli_warn("Groups with fewer than two datapoints have been dropped.")
                return pd.DataFrame()
            return pd.DataFrame({
                "x": [data["x"].iloc[0] if "x" in data.columns else 0],
                "n": [len(data)],
            })

        y = data["y"].values if "y" in data.columns else data["x"].values
        w = data["weight"].values if "weight" in data.columns and data["weight"].notna().any() else None

        modifier = 0 if trim else 3
        bw_val = _precompute_bw(y, bw) if isinstance(bw, str) else float(bw)

        y_range = (np.nanmin(y), np.nanmax(y))
        dens = _compute_density(
            y, w,
            from_=y_range[0] - modifier * bw_val,
            to=y_range[1] + modifier * bw_val,
            bw=bw_val, adjust=adjust, kernel=kernel, bounds=bounds,
        )

        dens["y"] = dens["x"]
        x_val = data["x"].iloc[0] if "x" in data.columns else 0
        dens["x"] = x_val

        if width is None and "x" in data.columns and len(data["x"].unique()) > 1:
            width = (data["x"].max() - data["x"].min()) * 0.9
        dens["width"] = width

        # Add quantile information
        if quantiles is not None:
            quantiles = list(quantiles)
            quant_vals = np.percentile(y, np.array(quantiles) * 100)
            quant_rows = []
            for i, q in enumerate(quantiles):
                row = {"y": quant_vals[i], "quantile": q, "x": x_val, "width": width}
                # Interpolate density
                if len(dens) > 1:
                    from scipy.interpolate import interp1d
                    try:
                        f = interp1d(dens["y"], dens["density"], bounds_error=False, fill_value=0)
                        row["density"] = float(f(quant_vals[i]))
                        row["scaled"] = row["density"] / dens["density"].max() if dens["density"].max() > 0 else 0
                        row["ndensity"] = row["scaled"]
                    except Exception:
                        row["density"] = 0
                        row["scaled"] = 0
                        row["ndensity"] = 0
                quant_rows.append(row)

            if quant_rows:
                quant_df = pd.DataFrame(quant_rows)
                for col in dens.columns:
                    if col not in quant_df.columns:
                        quant_df[col] = np.nan
                dens = pd.concat([dens, quant_df], ignore_index=True)

        return dens


def stat_ydensity(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "violin",
    position: str = "dodge",
    bw: Union[str, float] = "nrd0",
    adjust: float = 1.0,
    kernel: str = "gaussian",
    trim: bool = True,
    scale: str = "area",
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Violin plot density.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    bw : str or float
    adjust : float
    kernel : str
    trim : bool
    scale : str
    na_rm : bool
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatYdensity,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "bw": bw, "adjust": adjust, "kernel": kernel, "trim": trim,
            "scale": scale, "na_rm": na_rm, "orientation": orientation,
            **kwargs,
        },
    )


# ============================================================================
# StatBindot
# ============================================================================

class StatBindot(Stat):
    """Dot-density binning for dot plots.

    Attributes
    ----------
    required_aes : list
        ``["x"]``
    non_missing_aes : list
        ``["weight"]``
    dropped_aes : list
        ``["weight", "bin", "bincenter"]``
    """

    required_aes: List[str] = ["x"]
    non_missing_aes: List[str] = ["weight"]
    default_aes: Dict[str, Any] = {}
    dropped_aes: List[str] = ["weight", "bin", "bincenter"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        if params.get("binwidth") is None:
            cli_inform(
                "Bin width defaults to 1/30 of the range of the data. "
                "Pick better value with binwidth."
            )
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        binwidth: Optional[float] = None,
        binaxis: str = "x",
        method: str = "dotdensity",
        binpositions: str = "bygroup",
        origin: Optional[float] = None,
        width: float = 0.9,
        drop: bool = False,
        right: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute dot-density bins.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        binwidth : float, optional
        binaxis : str
        method : str
        binpositions : str
        origin : float, optional
        width : float
        drop : bool
        right : bool

        Returns
        -------
        pd.DataFrame
        """
        if binaxis == "x":
            values = data["x"].values
            scale = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
        else:
            values = data["y"].values if "y" in data.columns else data["x"].values
            scale = scales.get("y") if isinstance(scales, dict) else getattr(scales, "y", None)

        weight = data["weight"].values if "weight" in data.columns else None

        if scale is not None and hasattr(scale, "dimension"):
            range_ = tuple(scale.dimension())
        else:
            range_ = (np.nanmin(values), np.nanmax(values))

        if method == "histodot":
            bins_obj = _compute_bins(
                values, scale,
                binwidth=binwidth, bins=30, boundary=origin,
                closed="right" if right else "left",
            )
            result = _bin_vector(values, bins_obj, weight=weight, pad=False)
            result.rename(columns={"width": "binwidth", "x": "bincenter"}, inplace=True)
        else:
            # Dot density method
            result = _densitybin(values, weight=weight, binwidth=binwidth, range_=range_)
            if result.empty:
                return pd.DataFrame()

            # Collapse bins
            collapsed = result.groupby("bincenter").agg(
                binwidth=("binwidth", "first"),
                count=("weight", "sum"),
            ).reset_index()

            total = collapsed["count"].sum()
            if total > 0:
                collapsed.loc[collapsed["count"].isna(), "count"] = 0
                collapsed["ncount"] = collapsed["count"] / collapsed["count"].abs().max()
                if drop:
                    collapsed = collapsed[collapsed["count"] > 0]
            result = collapsed

        if binaxis == "x":
            if "bincenter" in result.columns:
                result.rename(columns={"bincenter": "x"}, inplace=True)
            result["width"] = result.get("binwidth", binwidth)
        else:
            if "bincenter" in result.columns:
                result.rename(columns={"bincenter": "y"}, inplace=True)
            if "x" in data.columns:
                result["x"] = np.mean([data["x"].min(), data["x"].max()])

        return result


# No separate stat_bindot constructor since it's typically used via geom_dotplot


# ============================================================================
# StatAlign
# ============================================================================

class StatAlign(Stat):
    """Align y values across groups for area/ribbon plots.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    """

    required_aes: List[str] = ["x", "y"]
    extra_params: List[str] = ["na_rm", "orientation"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(data, params, ambiguous=True)
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        flipped_aes: bool = False,
        unique_loc: Optional[np.ndarray] = None,
        adjust: float = 0.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Align values by interpolation.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        flipped_aes : bool
        unique_loc : array-like, optional
        adjust : float

        Returns
        -------
        pd.DataFrame
        """
        data = _flip_data(data, flipped_aes)

        if len(data["x"].unique()) < 2:
            return pd.DataFrame()

        x = data["x"].values
        y = data["y"].values

        if unique_loc is None:
            unique_loc = np.sort(np.unique(x))

        from scipy.interpolate import interp1d
        try:
            f = interp1d(x, y, bounds_error=False, fill_value=np.nan)
            y_val = f(unique_loc)
        except Exception:
            return pd.DataFrame()

        keep = ~np.isnan(y_val)
        x_val = unique_loc[keep]
        y_val = y_val[keep]

        # Add padding
        x_val = np.concatenate([[x_val[0] - adjust] if len(x_val) > 0 else [], x_val, [x_val[-1] + adjust] if len(x_val) > 0 else []])
        y_val = np.concatenate([[0], y_val, [0]])
        padding = np.concatenate([[True], np.zeros(len(x_val) - 2, dtype=bool), [True]])

        result = pd.DataFrame({
            "x": x_val,
            "y": y_val,
            "align_padding": padding,
            "flipped_aes": flipped_aes,
        })

        # Carry over constant columns
        for col in data.columns:
            if col not in result.columns and col not in ("x", "y"):
                vals = data[col].values
                if len(set(vals)) == 1:
                    result[col] = vals[0]

        return _flip_data(result, flipped_aes)


def stat_align(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "area",
    position: str = "identity",
    na_rm: bool = False,
    orientation: Any = None,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Align y values across groups.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    na_rm : bool
    orientation : optional
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatAlign,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"na_rm": na_rm, "orientation": orientation, **kwargs},
    )


# ============================================================================
# StatConnect
# ============================================================================

class StatConnect(Stat):
    """Connect observations with step/interpolation patterns.

    Attributes
    ----------
    required_aes : list
        ``["x|xmin|xmax", "y|ymin|ymax"]``
    """

    required_aes: List[str] = ["x|xmin|xmax", "y|ymin|ymax"]

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve connection type.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        params["flipped_aes"] = _has_flipped_aes(
            data, params, range_is_orthogonal=True, ambiguous=True,
        )

        connection = params.get("connection", "hv")
        if isinstance(connection, str):
            conn_map = {
                "hv": np.array([[1, 0], [1, 1]]),
                "vh": np.array([[0, 0], [0, 1]]),
                "mid": np.array([[0.5, 0], [0.5, 1]]),
                "linear": np.array([[0, 0], [1, 1]]),
            }
            if connection in conn_map:
                connection = conn_map[connection]
            else:
                cli_abort(f"Unknown connection type: {connection!r}")

        params["connection"] = connection
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        connection: Any = "hv",
        flipped_aes: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Connect points with interpolated steps.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        connection : str or np.ndarray
        flipped_aes : bool

        Returns
        -------
        pd.DataFrame
        """
        data = _flip_data(data, flipped_aes)
        n = len(data)
        if n <= 1:
            return data.iloc[:0]

        if not isinstance(connection, np.ndarray):
            return data

        m = connection.shape[0]

        # Sort by x
        if "x" in data.columns:
            data = data.sort_values("x").reset_index(drop=True)
        elif "xmin" in data.columns:
            data = data.sort_values("xmin").reset_index(drop=True)

        before = np.repeat(np.arange(n - 1), m)
        after = np.repeat(np.arange(1, n), m)
        xjust = np.tile(connection[:, 0], n - 1)
        yjust = np.tile(connection[:, 1], n - 1)

        new_data = data.iloc[before].reset_index(drop=True)

        # Interpolate x columns
        x_cols = [c for c in data.columns if c in ("x", "xmin", "xmax", "xend")]
        for col in x_cols:
            if col in data.columns:
                x_before = data[col].values[before]
                x_after = data[col].values[after]
                new_data[col] = x_before * (1 - xjust) + x_after * xjust

        # Interpolate y columns
        y_cols = [c for c in data.columns if c in ("y", "ymin", "ymax", "yend")]
        for col in y_cols:
            if col in data.columns:
                y_before = data[col].values[before]
                y_after = data[col].values[after]
                new_data[col] = y_before * (1 - yjust) + y_after * yjust

        # Ensure start/end integrity
        if not np.allclose(connection[0], [0, 0]):
            new_data = pd.concat([data.iloc[[0]], new_data], ignore_index=True)
        if not np.allclose(connection[-1], [1, 1]):
            new_data = pd.concat([new_data, data.iloc[[-1]]], ignore_index=True)

        return _flip_data(new_data, flipped_aes)


def stat_connect(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "path",
    position: str = "identity",
    connection: Any = "hv",
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Connect observations.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    connection : str or matrix
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatConnect,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"connection": connection, "na_rm": na_rm, **kwargs},
    )


# ============================================================================
# StatManual
# ============================================================================

class StatManual(Stat):
    """Apply a user-supplied function to each group.

    This stat passes data through a user-supplied function that returns
    a DataFrame.
    """

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function parameter.

        Parameters
        ----------
        data : pd.DataFrame
        params : dict

        Returns
        -------
        dict
        """
        fun = params.get("fun")
        if fun is not None and not callable(fun):
            cli_abort("`fun` must be a callable.")
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        fun: Optional[Callable] = None,
        args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Apply custom function.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        fun : callable, optional
        args : dict, optional

        Returns
        -------
        pd.DataFrame
        """
        if fun is None:
            return data

        if args is None:
            args = {}

        result = fun(data, **args)
        if isinstance(result, dict):
            result = pd.DataFrame(result)
        return result


def stat_manual(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    fun: Optional[Callable] = None,
    args: Optional[Dict[str, Any]] = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Apply custom transformation per group.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    fun : callable, optional
    args : dict, optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatManual,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"fun": fun, "args": args or {}, "na_rm": na_rm, **kwargs},
    )


# ============================================================================
# StatQuantile
# ============================================================================

class StatQuantile(Stat):
    """Quantile regression lines.

    Attributes
    ----------
    required_aes : list
        ``["x", "y"]``
    dropped_aes : list
        ``["weight"]``
    """

    required_aes: List[str] = ["x", "y"]
    dropped_aes: List[str] = ["weight"]

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        quantiles: Sequence[float] = (0.25, 0.5, 0.75),
        formula: Any = None,
        xseq: Optional[np.ndarray] = None,
        method: str = "rq",
        method_args: Optional[Dict[str, Any]] = None,
        na_rm: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute quantile regression lines.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        quantiles : sequence of float
        formula : optional
        xseq : array-like, optional
        method : str
        method_args : dict, optional
        na_rm : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y, quantile, group.
        """
        if xseq is None:
            xseq = np.linspace(data["x"].min(), data["x"].max(), 100)

        x = data["x"].values
        y_data = data["y"].values
        group_val = data["group"].iloc[0] if "group" in data.columns else 1

        results = []
        for q in quantiles:
            # Simple linear quantile regression approximation
            # using weighted least squares with iteratively reweighted method
            try:
                y_pred = self._quantile_regression(x, y_data, xseq, q)
            except Exception:
                y_pred = np.full_like(xseq, np.percentile(y_data, q * 100))

            results.append(pd.DataFrame({
                "x": xseq,
                "y": y_pred,
                "quantile": q,
                "group": f"{group_val}-{q}",
            }))

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def _quantile_regression(
        x: np.ndarray,
        y: np.ndarray,
        xseq: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        """Simple linear quantile regression.

        Parameters
        ----------
        x, y : np.ndarray
            Training data.
        xseq : np.ndarray
            Prediction points.
        tau : float
            Quantile (0-1).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        import statsmodels.formula.api as smf

        df = pd.DataFrame({"x": x, "y": y})
        mod = smf.quantreg("y ~ x", df)
        res = mod.fit(q=tau, max_iter=1000)
        pred_df = pd.DataFrame({"x": xseq})
        return res.predict(pred_df)


def stat_quantile(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "quantile",
    position: str = "identity",
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    formula: Any = None,
    method: str = "rq",
    method_args: Optional[Dict[str, Any]] = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Quantile regression lines.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    quantiles : sequence of float
    formula : optional
    method : str
    method_args : dict, optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatQuantile,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "quantiles": quantiles, "formula": formula, "method": method,
            "method_args": method_args or {}, "na_rm": na_rm, **kwargs,
        },
    )


# ============================================================================
# StatSf / StatSfCoordinates (stub implementations)
# ============================================================================

class StatSf(Stat):
    """Spatial feature statistics (stub).

    This stat computes bounding boxes from spatial feature geometries.
    It is a stub that provides the interface; full spatial support
    requires the ``geopandas`` package.

    Attributes
    ----------
    required_aes : list
        ``["geometry"]``
    """

    required_aes: List[str] = ["geometry"]

    def compute_panel(self, data: pd.DataFrame, scales: Any, coord: Any = None, **params: Any) -> pd.DataFrame:
        """Compute bounding boxes.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        coord : optional

        Returns
        -------
        pd.DataFrame
        """
        if "geometry" not in data.columns:
            return data

        import geopandas as gpd

        if hasattr(data, "total_bounds"):
            bounds = data.total_bounds
        else:
            geom = gpd.GeoSeries(data["geometry"])
            bounds = geom.total_bounds
        data = data.copy()
        data["xmin"] = bounds[0]
        data["ymin"] = bounds[1]
        data["xmax"] = bounds[2]
        data["ymax"] = bounds[3]

        return data


class StatSfCoordinates(Stat):
    """Extract coordinates from spatial features (stub).

    Attributes
    ----------
    required_aes : list
        ``["geometry"]``
    """

    required_aes: List[str] = ["geometry"]
    default_aes: Dict[str, Any] = {}

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        coord: Any = None,
        fun_geometry: Optional[Callable] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Extract point coordinates.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        coord : optional
        fun_geometry : callable, optional

        Returns
        -------
        pd.DataFrame
        """
        if "geometry" not in data.columns:
            return data

        import geopandas as gpd

        geom = gpd.GeoSeries(data["geometry"])

        if fun_geometry is not None:
            points = fun_geometry(geom)
        else:
            points = geom.centroid

        data = data.copy()
        data["x"] = points.x.values
        data["y"] = points.y.values

        return data


def stat_sf(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "rect",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Spatial feature stat.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer_sf(
        stat=StatSf,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )


def stat_sf_coordinates(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "point",
    position: str = "identity",
    fun_geometry: Optional[Callable] = None,
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Extract coordinates from spatial features.

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    fun_geometry : callable, optional
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer_sf(
        stat=StatSfCoordinates,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"fun_geometry": fun_geometry, "na_rm": na_rm, **kwargs},
    )


# ============================================================================
# stat_spoke (alias for identity with geom spoke)
# ============================================================================

def stat_spoke(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: str = "spoke",
    position: str = "identity",
    na_rm: bool = False,
    show_legend: Optional[bool] = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Stat for spoke geom (uses identity stat).

    Parameters
    ----------
    mapping : Mapping, optional
    data : DataFrame or callable, optional
    geom, position : str
    na_rm : bool
    show_legend : bool, optional
    inherit_aes : bool
    **kwargs

    Returns
    -------
    Layer
    """
    return _layer(
        stat=StatIdentity,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"na_rm": na_rm, **kwargs},
    )

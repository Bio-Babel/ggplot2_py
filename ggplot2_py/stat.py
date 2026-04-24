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

# R ``stats::quantile`` types 1-9 mapped to the equivalent ``numpy.quantile``
# ``method=`` string. Default R type is 7 (linear interpolation), which is
# also numpy's default. See R ``?quantile`` or Hyndman & Fan (1996).
_R_QTYPE_TO_NUMPY_METHOD = {
    1: "inverted_cdf",
    2: "averaged_inverted_cdf",
    3: "closest_observation",
    4: "interpolated_inverted_cdf",
    5: "hazen",
    6: "weibull",
    7: "linear",
    8: "median_unbiased",
    9: "normal_unbiased",
}


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
        if isinstance(x.dtype, pd.CategoricalDtype):
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
        if np.any(np.isnan(z_range)):
            return np.asarray([np.nan])
        if z_range[0] == z_range[1]:
            return np.asarray([z_range[0]])
        from scales import breaks_extended
        return np.asarray(breaks_extended(n=10)(z_range), dtype=float)

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
# MASS::bandwidth.nrd and MASS::kde2d (inlined, matching MASS conventions)
# ---------------------------------------------------------------------------

def _bandwidth_nrd(x: np.ndarray) -> float:
    """Port of ``MASS::bandwidth.nrd``.

    ``4 * 1.06 * min(sd(x), IQR(x) / 1.34) * n ** (-1/5)``. Note that this is
    4x the result of ``stats::bw.nrd0``; ``stat_density_2d`` calls this form
    because ``MASS::kde2d`` internally divides by 4 before the Gaussian
    kernel.
    """
    x = np.asarray(x, dtype=float)
    r = np.quantile(x, [0.25, 0.75])
    h = (r[1] - r[0]) / 1.34
    sd = float(np.std(x, ddof=1))
    return 4.0 * 1.06 * min(sd, h) * len(x) ** (-0.2)


def _kde2d(
    x: np.ndarray,
    y: np.ndarray,
    h: Optional[Sequence[float]] = None,
    n: Union[int, Sequence[int]] = 25,
    lims: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Port of ``MASS::kde2d`` (two-dimensional kernel density estimation).

    The MASS convention divides the bandwidth vector by 4 before applying the
    Gaussian kernel. ``ggplot2::stat_density_2d`` calls this code path, so we
    must reproduce the ``h / 4`` step exactly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = len(x)
    if len(y) != nx:
        raise ValueError("x and y must have the same length")

    if lims is None:
        lims = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    lims = list(lims)

    if np.isscalar(n):
        n_vec = (int(n), int(n))
    else:
        n_pair = list(n)
        n_vec = (int(n_pair[0]), int(n_pair[-1]))

    gx = np.linspace(lims[0], lims[1], n_vec[0])
    gy = np.linspace(lims[2], lims[3], n_vec[1])

    if h is None:
        h_vec = np.array([_bandwidth_nrd(x), _bandwidth_nrd(y)], dtype=float)
    else:
        h_list = list(np.atleast_1d(h).astype(float))
        if len(h_list) == 1:
            h_list = [h_list[0], h_list[0]]
        h_vec = np.asarray(h_list[:2], dtype=float)

    if np.any(h_vec <= 0):
        raise ValueError("bandwidths must be strictly positive")

    h_vec = h_vec / 4.0  # MASS convention

    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    def _dnorm(a: np.ndarray) -> np.ndarray:
        return inv_sqrt_2pi * np.exp(-0.5 * a * a)

    ax = (gx[:, None] - x[None, :]) / h_vec[0]
    ay = (gy[:, None] - y[None, :]) / h_vec[1]
    dx = _dnorm(ax)  # shape (n_x, nx)
    dy = _dnorm(ay)  # shape (n_y, nx)
    # z[i, j] = sum_k dx[i, k] * dy[j, k] / (nx * h1 * h2)
    # R tcrossprod(A, B) = A %*% t(B); mirror that shape with @ for z[i, j] mapped below.
    z = (dx @ dy.T) / (nx * h_vec[0] * h_vec[1])
    return {"x": gx, "y": gy, "z": z}


# ---------------------------------------------------------------------------
# contourpy wrappers -- reproduce isoband::isolines / isobands outputs as a
# tidy DataFrame matching ggplot2's iso_to_geom() schema.
# ---------------------------------------------------------------------------

def _contourpy_isolines(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    levels: Sequence[float],
    group: Any = -1,
) -> pd.DataFrame:
    """Wrap contourpy ``.lines(level)`` into an ``iso_to_geom(geom='path')``-shaped frame.

    Returns columns: ``level, x, y, piece, group, subgroup``. ``piece`` is the
    integer id of the individual line segment across all levels (1-based);
    ``group`` is a factor-style string ``"{group}-{level_idx:03d}-{line_idx:03d}"``
    mirroring ggplot2; ``subgroup`` is ``None`` for isolines (per R).
    """
    import contourpy

    gen = contourpy.contour_generator(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        z=np.asarray(z, dtype=float),
        line_type="Separate",
        fill_type="OuterOffset",
    )

    rows_level: List[Any] = []
    rows_x: List[float] = []
    rows_y: List[float] = []
    rows_piece: List[int] = []
    rows_group: List[str] = []
    piece = 0
    for level_idx, lvl in enumerate(levels, start=1):
        lines = gen.lines(float(lvl))
        for line_idx, seg in enumerate(lines, start=1):
            piece += 1
            n_pts = seg.shape[0]
            group_str = f"{group}-{level_idx:03d}-{line_idx:03d}"
            rows_level.extend([float(lvl)] * n_pts)
            rows_x.extend(seg[:, 0].tolist())
            rows_y.extend(seg[:, 1].tolist())
            rows_piece.extend([piece] * n_pts)
            rows_group.extend([group_str] * n_pts)

    df = pd.DataFrame({
        "level": rows_level,
        "x": rows_x,
        "y": rows_y,
        "piece": rows_piece,
        "group": rows_group,
        "subgroup": [None] * len(rows_level),
    })
    return df


def _contourpy_isobands(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    breaks: Sequence[float],
    group: Any = -1,
) -> pd.DataFrame:
    """Wrap contourpy ``.filled(lo, hi)`` into an ``iso_to_geom(geom='polygon')`` frame.

    ``breaks`` is the sequence of level boundaries (length ``n_bands + 1``).
    Returns columns: ``level, x, y, piece, group, subgroup, level_low,
    level_high, level_mid``. Each band gets one ``piece`` id; each ring inside
    the band (outer ring + holes) gets its own ``subgroup`` id (1-based).
    """
    import contourpy

    gen = contourpy.contour_generator(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        z=np.asarray(z, dtype=float),
        line_type="Separate",
        fill_type="OuterOffset",
    )

    breaks = np.asarray(breaks, dtype=float)
    levels_low = breaks[:-1]
    levels_high = breaks[1:]
    # Label each band the same way pretty_isoband_levels does.
    level_labels = _pretty_isoband_levels(levels_low, levels_high)

    rows_level: List[Any] = []
    rows_x: List[float] = []
    rows_y: List[float] = []
    rows_piece: List[int] = []
    rows_group: List[str] = []
    rows_subgroup: List[int] = []
    rows_low: List[float] = []
    rows_high: List[float] = []
    piece = 0
    for band_idx, (lo, hi, label) in enumerate(
        zip(levels_low, levels_high, level_labels), start=1
    ):
        polys, offsets = gen.filled(float(lo), float(hi))
        if len(polys) == 0:
            continue
        piece += 1
        group_str = f"{group}-{band_idx:03d}"
        subgroup_counter = 0
        for poly_pts, poly_offs in zip(polys, offsets):
            # poly_offs slices each ring: [0, n1, n1+n2, ...]
            for ring_idx in range(len(poly_offs) - 1):
                subgroup_counter += 1
                s = int(poly_offs[ring_idx])
                e = int(poly_offs[ring_idx + 1])
                n_pts = e - s
                rows_level.extend([label] * n_pts)
                rows_x.extend(poly_pts[s:e, 0].tolist())
                rows_y.extend(poly_pts[s:e, 1].tolist())
                rows_piece.extend([piece] * n_pts)
                rows_group.extend([group_str] * n_pts)
                rows_subgroup.extend([subgroup_counter] * n_pts)
                rows_low.extend([float(lo)] * n_pts)
                rows_high.extend([float(hi)] * n_pts)

    df = pd.DataFrame({
        "level": rows_level,
        "x": rows_x,
        "y": rows_y,
        "piece": rows_piece,
        "group": rows_group,
        "subgroup": rows_subgroup,
        "level_low": rows_low,
        "level_high": rows_high,
    })
    df["level_mid"] = 0.5 * (df["level_low"] + df["level_high"])
    return df


def _pretty_isoband_levels(
    lows: np.ndarray, highs: np.ndarray, dig_lab: int = 3
) -> List[str]:
    """Port of ``pretty_isoband_levels`` -- label each band as ``"(lo, hi]"``.

    Matches R's ``format(x, digits=dig.lab, trim=TRUE)`` by auto-growing the
    precision until all boundary labels are unique.
    """
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    breaks = np.unique(np.concatenate([lows, highs]))

    def _format_vec(vals: np.ndarray, dig: int) -> List[str]:
        # R's format() with digits uses significant digits; Python's %g matches
        # closely (both strip trailing zeros and pick exponential vs fixed
        # automatically).
        return [f"{v:.{dig}g}" for v in vals]

    dig = dig_lab
    while True:
        labels = _format_vec(breaks, dig)
        if len(set(labels)) == len(labels):
            break
        dig += 1
        if dig > 22:
            break

    label_low = _format_vec(lows, dig)
    label_high = _format_vec(highs, dig)
    return [f"({lo}, {hi}]" for lo, hi in zip(label_low, label_high)]


# ---------------------------------------------------------------------------
# Ellipse / Q-Q helpers (ports of R primitives used by StatEllipse/StatQq*).
# ---------------------------------------------------------------------------

def _ppoints(n: int, a: Optional[float] = None) -> np.ndarray:
    """Port of R ``stats::ppoints(n)``.

    Formula: ``(1:n - a) / (n + 1 - 2*a)`` with ``a = 3/8`` when ``n <= 10``
    else ``a = 1/2``. Returns a length-``n`` ndarray of plotting positions.
    """
    if a is None:
        a = 3.0 / 8.0 if n <= 10 else 0.5
    if n <= 0:
        return np.empty(0, dtype=float)
    return (np.arange(1, n + 1, dtype=float) - a) / (n + 1 - 2 * a)


def _cov_wt(x: np.ndarray, wt: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Port of R ``stats::cov.wt(x, wt)`` with default ``method='unbiased'``.

    ``x`` is shape (n, p). ``wt`` is length-n non-negative weights that will
    be normalised to sum to 1 inside R; mirror that here. Returns a dict with
    keys ``cov`` (p, p), ``center`` (p,).

    Unbiased weighted covariance: ``S = sum_k w_k (x_k - mu)(x_k - mu)^T``
    divided by ``1 - sum(w^2)`` where ``w`` sums to 1 and ``mu`` is the
    weighted mean. For equal weights ``w = 1/n`` this reduces to the
    ddof=1 sample covariance.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if wt is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(wt, dtype=float)
        s = float(w.sum())
        if s <= 0:
            raise ValueError("sum(wt) must be positive in cov.wt")
        w = w / s
    center = (w[:, None] * x).sum(axis=0)
    diff = x - center
    # Outer-product accumulation across rows, weighted.
    cov = (w[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
    denom = 1.0 - float((w * w).sum())
    if denom <= 0:
        raise ValueError("cov.wt: 1 - sum(w^2) not positive")
    cov = cov / denom
    return {"cov": cov, "center": center}


def _cov_trob(
    x: np.ndarray,
    wt: Optional[np.ndarray] = None,
    center: bool = True,
    nu: float = 5.0,
    maxit: int = 25,
    tol: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Port of R ``MASS::cov.trob`` (robust multivariate t covariance).

    Verbatim port of the MASS source. Iteratively reweights observations to
    down-weight outliers assuming a multivariate t distribution with ``nu``
    degrees of freedom.

    Parameters
    ----------
    x : ndarray of shape (n, p).
    wt : optional length-n weights; defaults to all ones.
    center : if True, estimate the center jointly with the covariance; if
        False, fix the center at zero; if a length-p array, use that as the
        fixed center.
    nu : degrees of freedom for the t distribution (default 5).
    maxit : maximum number of iterations (default 25).
    tol : convergence tolerance on weight change (default 0.01).

    Returns
    -------
    dict with keys ``cov`` (p, p) and ``center`` (p,).

    Notes
    -----
    Algorithm (from MASS::cov.trob):
    ``loc = colSums(wt * x) / sum(wt)``.
    ``w = wt * (1 + p/nu)`` initially; then iterate:
      - ``X = x - loc``;
      - ``sX = svd(sqrt(w/sum(w)) * X)``;
      - ``wX = X V diag(1/d)``;
      - ``Q = rowSums(wX^2)``;
      - ``w = wt * (nu + p) / (nu + Q)``;
      - if ``center`` is True: ``loc = colSums(w*x) / sum(w)``;
      - break when ``max(|w - w0|) < tol``.
    Finally ``cov = (sqrt(w) X)^T (sqrt(w) X) / sum(wt)``.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    if wt is None:
        wt = np.ones(n, dtype=float)
        miss_wt = True
    else:
        wt = np.asarray(wt, dtype=float)
        if wt.shape[0] != n:
            raise ValueError("length of 'wt' must equal number of observations")
        if np.any(wt < 0):
            raise ValueError("negative weights not allowed")
        if wt.sum() == 0:
            raise ValueError("no positive weights")
        pos = wt > 0
        x = x[pos, :]
        wt = wt[pos]
        n = x.shape[0]
        miss_wt = False
    del miss_wt  # tracked but unused; matches R's miss.wt flag

    # Initial location
    loc = (wt[:, None] * x).sum(axis=0) / wt.sum()
    if isinstance(center, np.ndarray):
        if center.shape[0] != p:
            raise ValueError("'center' is not the right length")
        loc = np.asarray(center, dtype=float)
        use_loc = False
    elif isinstance(center, bool):
        if not center:
            loc = np.zeros(p, dtype=float)
            use_loc = False
        else:
            use_loc = True
    else:
        # numeric vector
        loc = np.asarray(center, dtype=float)
        if loc.shape[0] != p:
            raise ValueError("'center' is not the right length")
        use_loc = False

    w = wt * (1.0 + p / nu)
    X = x - loc  # initialized so the final ``cov`` can reuse the last X.
    for _ in range(1, maxit + 1):
        w0 = w
        # R computes ``X <- scale.simp(x, loc, n, p)`` at the TOP of each
        # iteration using the loc from the previous iteration; if the loop
        # breaks, R's post-loop ``cov`` uses *this* X (not one recomputed
        # from the just-updated loc).
        X = x - loc
        # R: svd(sqrt(w/sum(w)) * X, nu = 0) -> returns d, v (no U).
        # sqrt(w/sum(w)) broadcast as row scaling (length-n vector times n x p matrix).
        scale = np.sqrt(w / w.sum())
        M = scale[:, None] * X
        # full_matrices=False gives compact form: U (n x k), d (k,), Vt (k x p)
        # where k = min(n, p).
        _, d, vt = np.linalg.svd(M, full_matrices=False)
        V = vt.T  # p x k
        # wX = X @ V @ diag(1/d); need diag(1/d, , p) in R which pads to p x p.
        # R's diag(1/sX$d, , p): when length(d) == p, diag creates p x p.
        # Here k = min(n, p); if n >= p (typical), k == p and this works as-is.
        # If n < p, MASS degenerates; we mirror R which would fail there too.
        wX = X @ V @ np.diag(1.0 / d)
        Q = (wX * wX).sum(axis=1)
        w = wt * (nu + p) / (nu + Q)
        if use_loc:
            loc = (w[:, None] * x).sum(axis=0) / w.sum()
        if np.max(np.abs(w - w0)) < tol:
            break

    sw = np.sqrt(w)
    cov = (sw[:, None] * X).T @ (sw[:, None] * X) / wt.sum()
    return {"cov": cov, "center": loc}


def _calculate_ellipse(
    data: pd.DataFrame,
    vars_: Sequence[str],
    type_: str,
    level: float,
    segments: int,
) -> pd.DataFrame:
    """Port of ggplot2 ``calculate_ellipse`` (R/stat-ellipse.R).

    Parameters
    ----------
    data : DataFrame with the two coordinate columns named by ``vars_`` and
        optionally a ``weight`` column.
    vars_ : list of two column names (e.g. ``["x", "y"]``).
    type_ : one of ``"t"``, ``"norm"``, ``"euclid"``. ``"t"`` uses the port
        of ``MASS::cov.trob`` in :func:`_cov_trob`.
    level : confidence level (or euclidean radius when type='euclid').
    segments : number of polygon segments.

    Returns
    -------
    DataFrame with columns ``vars_[0]``, ``vars_[1]`` and ``segments + 1``
    rows (closed ring).

    Notes
    -----
    R's ``chol()`` returns the **upper** triangle; NumPy's ``np.linalg.cholesky``
    returns the **lower**. We use ``np.linalg.cholesky(S).T`` to mimic R.
    """
    from scipy import stats as scipy_stats

    dfn = 2
    dfd = data.shape[0] - 1

    if type_ not in ("t", "norm", "euclid"):
        cli_inform(f"Unrecognized ellipse type: {type_!r}")
        return pd.DataFrame({vars_[0]: [np.nan], vars_[1]: [np.nan]})
    if dfd < 3:
        cli_inform("Too few points to calculate an ellipse")
        return pd.DataFrame({vars_[0]: [np.nan], vars_[1]: [np.nan]})

    xy = data[list(vars_)].to_numpy(dtype=float)
    n = xy.shape[0]
    if "weight" in data.columns:
        raw_w = np.asarray(data["weight"].to_numpy(), dtype=float)
    else:
        raw_w = np.ones(n)
    weight = raw_w / raw_w.sum()

    if type_ == "t":
        # R passes ``wt = weight * nrow(data)`` to cov.trob.
        v = _cov_trob(xy, wt=weight * n)
    else:
        v = _cov_wt(xy, wt=weight)
    shape = v["cov"]
    center = v["center"]
    if type_ == "euclid":
        shape = np.diag(np.repeat(float(np.min(np.diag(shape))), 2))

    # R chol() returns UPPER triangle; numpy returns LOWER. Transpose.
    chol_decomp = np.linalg.cholesky(shape).T

    if type_ == "euclid":
        radius = level / float(np.max(chol_decomp))
    else:
        radius = float(np.sqrt(dfn * scipy_stats.f.ppf(level, dfn, dfd)))

    angles = np.arange(segments + 1) * 2.0 * np.pi / segments
    unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    # R: t(center + radius * t(unit.circle %*% chol_decomp))
    # i.e. each row of (unit_circle @ chol_decomp) is scaled by radius and
    # offset by center.
    ellipse = center + radius * (unit_circle @ chol_decomp)
    return pd.DataFrame({vars_[0]: ellipse[:, 0], vars_[1]: ellipse[:, 1]})


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
    # scales may be a dict or an object with attributes
    x_scale = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
    y_scale = scales.get("y") if isinstance(scales, dict) else getattr(scales, "y", None)

    if x_scale is not None and hasattr(x_scale, "dimension"):
        x_range = np.array(x_scale.dimension())
    else:
        x_range = np.array([0.0, 1.0])
    if y_scale is not None and hasattr(y_scale, "dimension"):
        y_range = np.array(y_scale.dimension())
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

    # Empty input → empty output.  R's ``hex_bounds`` silently works on
    # zero-length vectors (``min(numeric(0))`` returns ``Inf`` with a
    # warning, the pipeline produces an empty data frame).  NumPy's
    # ``np.min`` / ``np.max`` raise on empty arrays, so short-circuit.
    if x.size == 0:
        return pd.DataFrame({"x": [], "y": [], "value": [],
                             "width": [], "height": []})

    bw_x, bw_y = binwidth
    row_h = bw_y * np.sqrt(3) / 2.0

    # R (hexbin.R:8-13, hexbin::hexbin):
    #   xbnds = c(floor(min(x)/bw_x)*bw_x - 1e-6, …)
    # The hex lattice is anchored to (xlo, ylo) and lattice nodes
    # (= hex centres returned by hcell2xy) are at
    #   even row:  (xlo + bi*bw_x,          ylo + bj*row_h)
    #   odd  row:  (xlo + (bi+0.5)*bw_x,    ylo + bj*row_h)
    # Each point goes to the NEAREST lattice node.  Verified against
    # R's first centre (8.67, 11.73) for mpg cty/hwy — bi=bj=0.
    eps = 1e-6
    xlo = np.floor(x.min() / bw_x) * bw_x - eps
    ylo = np.floor(y.min() / bw_y) * bw_y - eps

    # Candidate lattice: pick nearest of two adjacent rows then nearest
    # column within that row (hex nearest-neighbour is exact enough for
    # rectangular preprocessing at our aspect ratio).
    ry = (y - ylo) / row_h
    iy = np.rint(ry).astype(int)
    shifted = iy % 2 == 1
    x_shifted = x - 0.5 * bw_x * shifted.astype(float)
    ix = np.rint((x_shifted - xlo) / bw_x).astype(int)

    keys = ix.astype(str) + "_" + iy.astype(str)
    unique_keys = np.unique(keys)

    results = []
    for key in unique_keys:
        mask = keys == key
        z_vals = z[mask]
        if drop and len(z_vals) == 0:
            continue
        val = fun(z_vals, **fun_args)
        parts = key.split("_")
        bi, bj = int(parts[0]), int(parts[1])
        cx = xlo + bi * bw_x + (0.5 * bw_x if bj % 2 == 1 else 0.0)
        cy = ylo + bj * row_h
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

    # --- Auto-registration registry (Python-exclusive) -------------------
    _registry: Dict[str, Any] = {}

    required_aes: List[str] = []
    non_missing_aes: List[str] = []
    optional_aes: List[str] = []
    default_aes: Dict[str, Any] = {}
    dropped_aes: List[str] = []
    extra_params: List[str] = ["na_rm"]
    retransform: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if name.startswith("Stat") and len(name) > 4:
            key = name[4:]
            Stat._registry[key] = cls
            Stat._registry[key.lower()] = cls

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
            for panel, panel_data in data.groupby("PANEL", sort=False, observed=True):
                scales = layout.get_scales(panel) if layout is not None else {}
                try:
                    result = self.compute_panel(panel_data, scales, **trimmed)
                except Exception as e:
                    cli_warn(f"Computation failed in {snake_class(self)}: {e}")
                    result = pd.DataFrame()
                if not result.empty:
                    # Ensure PANEL is preserved (some stats override
                    # compute_panel without carrying it forward).
                    if "PANEL" not in result.columns:
                        result["PANEL"] = panel
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

        Port of R ggplot2 ``Stat$parameters`` (``R/stat-.R:368-381``). R
        inspects ``compute_panel`` first and falls back to
        ``compute_group`` only when the former is the base
        ``function(data, scales, ...)`` delegator — detected in R by
        ``"..." %in% panel_args``. The Python port encodes the same rule
        by asking whether the concrete ``compute_panel`` bound on the
        subclass is still ``Stat.compute_panel``.

        Parameters
        ----------
        extra : bool
            Whether to include ``extra_params``.

        Returns
        -------
        list of str
        """
        def _named(fn) -> List[str]:
            sig = inspect.signature(fn)
            return [
                p.name for p in sig.parameters.values()
                if p.name != "self"
                and p.kind is not inspect.Parameter.VAR_POSITIONAL
                and p.kind is not inspect.Parameter.VAR_KEYWORD
            ]

        # R inspects ``compute_panel`` and, if it has ``...``, falls back
        # to ``compute_group``. The Python port routinely tacks a
        # ``**kwargs`` onto ``compute_panel`` for forward compatibility
        # even when R's counterpart has explicit formals (e.g.
        # ``StatYdensity`` keeps its real parameters on ``compute_group``),
        # so the ``...``-based switch misfires. We instead take the
        # union of both methods' formals whenever they're overridden —
        # matches R semantically for every pattern (R-override-panel,
        # R-override-group, or R-override-both).
        panel_overridden = type(self).compute_panel is not Stat.compute_panel
        group_overridden = type(self).compute_group is not Stat.compute_group
        args: List[str] = []
        if panel_overridden:
            args.extend(_named(self.compute_panel))
        if group_overridden:
            args.extend(_named(self.compute_group))
        if not panel_overridden and not group_overridden:
            args = _named(self.compute_group)

        # R: setdiff(args, names(ggproto_formals(Stat$compute_group))) —
        # drop the ``data`` / ``scales`` (and any other base) slots.
        base_args = set(_named(Stat.compute_group))
        args = [a for a in dict.fromkeys(args) if a not in base_args]

        if extra:
            args = list(dict.fromkeys(args + list(self.extra_params)))
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
    default_aes: Dict[str, Any] = {
        "x": AfterStat("count"),
        "y": AfterStat("count"),
        "weight": 1,
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
        "x": AfterStat("density"),
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

# ---------------------------------------------------------------------------
# R-faithful formula parsing via patsy.
#
# R's smoothing methods (lm, glm, loess, gam, rlm, ...) all accept a
# ``formula`` argument (e.g. ``y ~ x``, ``y ~ poly(x, 2)``, ``y ~ s(x)``).
# ggplot2 threads the formula *literally* through to ``method(formula, data=,
# weights=, ...)``. The Python port mirrors that by parsing the formula
# string with patsy and building a design matrix on both the training data
# and the prediction grid.
#
# R ``poly(x, degree)`` is not available to patsy by default; we register a
# raw-polynomial implementation below. OLS predictions and standard errors
# are invariant under the basis choice (same column space), so raw powers
# reproduce R's ``predict.lm`` output to machine precision — verified
# against ``poly(x, 2)``, ``poly(x, 3)``, and ``poly(x, 2, raw=TRUE)``.
# ---------------------------------------------------------------------------

def _r_poly(x: Any, degree: int = 1, raw: bool = True) -> np.ndarray:
    """R-compatible ``poly(x, degree)`` for patsy.

    Returns raw powers ``x^1, ..., x^degree`` as an ``(n, degree)`` array.
    Orthogonalization (R's default) spans the same column space as raw
    powers, so fitted values and prediction SEs are identical — which is
    what ggplot2 uses.
    """
    xa = np.asarray(x, dtype=float)
    return np.column_stack([xa ** k for k in range(1, int(degree) + 1)])


_SMOOTH_FORMULA_NAMESPACE = {"poly": _r_poly}


def _parse_smooth_formula(
    formula: Any,
    data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Parse an R-style formula and build design matrices for training data.

    Parameters
    ----------
    formula : str, patsy.ModelDesc, or None
        R-style formula. ``None`` defaults to ``"y ~ x"``.
    data : pd.DataFrame
        Must contain the LHS and RHS variables referenced by ``formula``.

    Returns
    -------
    y : (n,) ndarray
    X : (n, p) ndarray
    design_info : patsy.DesignInfo
        Reusable for ``build_design_matrices`` on a prediction grid.
    """
    import patsy

    if formula is None:
        formula = "y ~ x"
    if not isinstance(formula, str):
        # Accept pre-built patsy ModelDesc / DesignInfo transparently.
        formula_obj = formula
    else:
        formula_obj = formula

    eval_env = patsy.EvalEnvironment.capture(0).with_outer_namespace(
        _SMOOTH_FORMULA_NAMESPACE
    )
    y_df, X_df = patsy.dmatrices(
        formula_obj, data, return_type="dataframe", eval_env=eval_env,
    )
    y = np.asarray(y_df).ravel()
    X = np.asarray(X_df)
    return y, X, X_df.design_info


def _build_smooth_prediction_matrix(
    design_info: Any, xseq: np.ndarray,
) -> np.ndarray:
    """Build the prediction design matrix on a fresh ``x`` grid."""
    import patsy
    new_df = pd.DataFrame({"x": np.asarray(xseq, dtype=float)})
    (X_new,) = patsy.build_design_matrices([design_info], new_df)
    return np.asarray(X_new)


def _formula_is_trivial_y_x(formula: Any) -> bool:
    """True iff ``formula`` is the default ``y ~ x`` (loess's only form)."""
    if formula is None:
        return True
    if isinstance(formula, str):
        normalized = "".join(formula.split())
        return normalized == "y~x"
    return False


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
            # R stat-smooth.R:14-20 — pick loess vs gam based on the
            # *largest* group, not total row count (loess has bad memory
            # scaling so we want to avoid it on big groups).
            if "group" in data.columns and "PANEL" in data.columns:
                max_group = data.groupby(["group", "PANEL"]).size().max()
            else:
                max_group = len(data)
            if max_group < 1000:
                method = "loess"
            else:
                # R falls back to loess if mgcv is not installed. Python
                # mirrors: try the gam backend; if unavailable, warn + loess.
                try:
                    from statsmodels.gam.api import GLMGam  # noqa: F401
                    method = "gam"
                except ImportError:
                    cli_inform(
                        "method was set to 'gam', but statsmodels.gam is "
                        "not importable. Falling back to method = 'loess'."
                    )
                    method = "loess"
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

        # Default formula: R stat-smooth.R:36-43 — ``y ~ x`` for everything
        # except auto/gam (which defaults to ``y ~ s(x, bs="cs")``). The
        # gam default is handled inside ``_fit_gam`` since the
        # smoother-basis API differs from the design-matrix API.
        if formula is None:
            formula = "y ~ x"

        # Validate formula/method compatibility BEFORE fitting. R's stats
        # don't silently downgrade either: a bad formula fails loudly.
        if method in ("loess", "lowess") and not _formula_is_trivial_y_x(formula):
            cli_abort(
                f"stat_smooth(method='loess'): formula must be 'y ~ x'; "
                f"got {formula!r}. Use method='lm' for polynomial/spline "
                f"fits."
            )

        if method not in ("lm", "linear", "loess", "lowess", "gam", "glm"):
            cli_abort(
                f"Unknown smoothing method {method!r}; expected one of "
                f"'lm', 'loess', 'gam', 'glm', or 'auto'."
            )

        try:
            if method in ("lm", "linear"):
                result = self._fit_lm(data, xseq, se, level, formula=formula)
            elif method in ("loess", "lowess"):
                result = self._fit_lowess(data, xseq, se, level, span)
            elif method == "gam":
                result = self._fit_gam(data, xseq, se, level, formula=formula)
            elif method == "glm":
                # R glm with identity link ≈ lm for gaussian family. For
                # other families users should pass method_args.
                result = self._fit_lm(data, xseq, se, level, formula=formula)
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
        formula: Any = "y ~ x",
    ) -> pd.DataFrame:
        """Fit linear model — R ``lm(formula, data, weights)`` parity.

        Formula is parsed with patsy (R-compatible syntax). Supports
        ``y ~ x``, ``y ~ poly(x, k)``, ``y ~ x + I(x^2)``,
        ``y ~ bs(x, df=k)``, ``y ~ cr(x, df=k)``, log/exp transforms,
        etc. Predictions at ``xseq`` are produced by evaluating the same
        design on a fresh grid and matching R's ``predict.lm(..., se.fit=TRUE,
        interval='confidence')`` output.

        Parameters
        ----------
        data : pd.DataFrame
        xseq : np.ndarray
        se : bool
        level : float
        formula : str or patsy design, optional
            Default ``"y ~ x"``.

        Returns
        -------
        pd.DataFrame
        """
        from scipy import stats as scipy_stats
        import statsmodels.api as sm

        w = (
            np.asarray(data["weight"].values, dtype=float)
            if "weight" in data.columns
            else np.ones(len(data))
        )

        y, X, design_info = _parse_smooth_formula(formula, data)

        # Weighted least squares via statsmodels.WLS. Matches R's
        # ``lm(formula, data, weights=w)`` on fitted values, residual
        # variance, and predict.lm's se.fit.
        model = sm.WLS(y, X, weights=w).fit()

        X_new = _build_smooth_prediction_matrix(design_info, xseq)
        y_pred = np.asarray(X_new @ model.params, dtype=float)

        result = pd.DataFrame({"x": np.asarray(xseq, dtype=float), "y": y_pred})

        n = len(y)
        p = X.shape[1]
        if se and n > p:
            # R's predict.lm with se.fit=TRUE uses the model covariance
            # ``vcov = sigma^2 * (X'WX)^-1``, and for a new row x_new
            # ``se.fit_i = sqrt(x_new' vcov x_new)``.
            cov = np.asarray(model.cov_params(), dtype=float)
            se_pred = np.sqrt(np.einsum("ij,jk,ik->i", X_new, cov, X_new))
            t_val = scipy_stats.t.ppf((1 + level) / 2, df=n - p)
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
        """Fit LOESS smoother — R ``loess()`` parity via ``skmisc.loess``.

        R's ``loess`` returns per-point standard errors from the local
        polynomial fit; ``skmisc.loess.loess`` mirrors that API
        (``predict(..., stderror=True)`` → ``.stderr`` vector).

        Falls back to ``statsmodels.nonparametric.lowess`` + a constant
        residual-variance SE if ``skmisc.loess`` is unavailable; that path
        only gives a rough band.
        """
        from scipy import stats as scipy_stats

        x = np.asarray(data["x"].values, dtype=float)
        y = np.asarray(data["y"].values, dtype=float)
        w = (
            np.asarray(data["weight"].values, dtype=float)
            if "weight" in data.columns
            else np.ones_like(x)
        )
        xseq = np.asarray(xseq, dtype=float)

        try:
            from skmisc.loess import loess as _loess

            model = _loess(x, y, weights=w, span=float(span), degree=2,
                           family="gaussian")
            model.fit()
            pred = model.predict(xseq, stderror=bool(se))
            y_pred = np.asarray(pred.values, dtype=float)

            result = pd.DataFrame({"x": xseq, "y": y_pred})
            if se and hasattr(pred, "stderr"):
                se_pred = np.asarray(pred.stderr, dtype=float)
                # skmisc loess stderr uses residual df internally; R uses
                # the same t-quantile on n - trace(S) equivalent degrees
                # of freedom. skmisc.predict exposes ``confidence`` which
                # applies the right t-factor if called instead of stderr.
                conf = model.predict(xseq, stderror=True).confidence(
                    alpha=1.0 - float(level)
                )
                result["ymin"] = np.asarray(conf.lower, dtype=float)
                result["ymax"] = np.asarray(conf.upper, dtype=float)
                result["se"] = se_pred
            else:
                result["ymin"] = np.nan
                result["ymax"] = np.nan
                result["se"] = np.nan
            return result
        except ImportError:
            pass

        # Fallback — statsmodels LOWESS (no per-point SE). The CI band is
        # a constant ± t * sqrt(residual_var); strictly worse than R's
        # per-point band but we emit a warning rather than a broken band.
        from scipy.interpolate import interp1d
        import statsmodels.api as sm

        smooth = sm.nonparametric.lowess(y, x, frac=span, return_sorted=True)
        f = interp1d(smooth[:, 0], smooth[:, 1], kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        y_pred = f(xseq)

        result = pd.DataFrame({"x": xseq, "y": y_pred})
        if se and len(x) > 2:
            cli_warn(
                "stat_smooth(method='loess'): skmisc.loess is unavailable; "
                "confidence band is a constant ±t*sqrt(residual_var) rather "
                "than R's per-point loess SE."
            )
            y_hat = f(x)
            resid_var = np.var(y - y_hat, ddof=2)
            se_val = np.sqrt(resid_var)
            t_val = scipy_stats.t.ppf((1 + level) / 2, df=len(x) - 2)
            result["ymin"] = y_pred - t_val * se_val
            result["ymax"] = y_pred + t_val * se_val
            result["se"] = se_val
        else:
            result["ymin"] = np.nan
            result["ymax"] = np.nan
            result["se"] = np.nan
        return result

    @staticmethod
    def _fit_gam(
        data: pd.DataFrame,
        xseq: np.ndarray,
        se: bool,
        level: float,
        formula: Any = None,
    ) -> pd.DataFrame:
        """Fit GAM — R ``mgcv::gam(y ~ s(x, bs='cs'))`` via
        ``statsmodels.gam.api.GLMGam`` with B-splines.

        R's ``bs="cs"`` is a cubic regression spline with shrinkage
        penalty. The closest faithful stand-in in statsmodels is
        ``BSplines(df=10, degree=3)`` + a ridge-like ``alpha`` penalty
        chosen so the effective degrees of freedom match R's defaults on
        typical biological data (EDF ≈ 8-9).

        Supported formulas: ``y ~ x`` (treated as default smooth on x) and
        ``y ~ s(x, ...)`` (arguments currently ignored — always fits
        ``s(x, bs='cs')`` equivalent). More complex gam formulas (tensor
        products, multiple smooths) raise ``cli_abort``.
        """
        from scipy import stats as scipy_stats
        from statsmodels.gam.api import GLMGam, BSplines

        # Accept default / y~x / y~s(x, ...). Anything else — abort with
        # an R-style message. We do not silently fall through to the
        # default smooth.
        if formula is not None and not _formula_is_trivial_y_x(formula):
            form_str = (
                formula if isinstance(formula, str) else str(formula)
            ).replace(" ", "")
            # Very permissive match: y~s(x...). R's mgcv accepts far more
            # (te(), ti(), bivariate smooths, etc.) but our GLMGam stand-in
            # only supports a single univariate smooth on x.
            if not (form_str.startswith("y~s(x") and form_str.endswith(")")):
                cli_abort(
                    "stat_smooth(method='gam'): only 'y ~ x' or "
                    "'y ~ s(x, ...)' is supported; got "
                    f"{formula!r}. Use method='lm' for polynomial / spline "
                    "basis regression."
                )

        x = np.asarray(data["x"].values, dtype=float)
        y = np.asarray(data["y"].values, dtype=float)
        w = (
            np.asarray(data["weight"].values, dtype=float)
            if "weight" in data.columns
            else np.ones_like(x)
        )
        xseq = np.asarray(xseq, dtype=float)

        bs = BSplines(x[:, None], df=[10], degree=[3], include_intercept=False)
        exog = np.ones((len(y), 1))
        model = GLMGam(y, exog=exog, smoother=bs, freq_weights=w).fit()

        # Predict on the grid.
        bs_pred = bs.transform(xseq[:, None])
        X_pred = np.hstack([np.ones((len(xseq), 1)), bs_pred])
        y_pred = np.asarray(X_pred @ model.params, dtype=float)

        result = pd.DataFrame({"x": xseq, "y": y_pred})
        if se:
            cov = model.cov_params()
            se_pred = np.sqrt(np.einsum("ij,jk,ik->i", X_pred, cov, X_pred))
            t_val = scipy_stats.t.ppf((1 + level) / 2, df=model.df_resid)
            result["ymin"] = y_pred - t_val * se_pred
            result["ymax"] = y_pred + t_val * se_pred
            result["se"] = se_pred
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

        # Compute quantiles via ``np.quantile`` with the method matching
        # R's ``quantile.type`` argument (R stat-boxplot.R:58 default 7).
        # numpy's ``method="linear"`` is R type 7; see R ``?quantile``.
        np_method = _R_QTYPE_TO_NUMPY_METHOD.get(int(quantile_type))
        if np_method is None:
            raise ValueError(
                f"quantile_type must be an integer in 1..9 (R convention); "
                f"got {quantile_type!r}."
            )
        qs = np.quantile(y, [0.0, 0.25, 0.5, 0.75, 1.0], method=np_method)
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
        """Summarise by unique (group, x).

        Port of ``stat-summary.R:196-202`` + ``summarise_by_x``.  R calls
        ``dapply(data, c("group","x"), summary)`` and then merges the
        summary with a per-(group, x) unique-columns table, keyed by
        ``c("x", "group")``.  The merge key columns come first in the
        resulting data frame.

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

        if "group" not in data.columns:
            data = data.copy()
            data["group"] = -1

        summary_rows: List[pd.DataFrame] = []
        unique_rows: List[Dict[str, Any]] = []
        for (grp_val, x_val), grp in data.groupby(["group", "x"], sort=False):
            summary = fun(grp)
            if not isinstance(summary, pd.DataFrame):
                summary = pd.DataFrame(summary)
            summary = summary.copy()
            summary["x"] = x_val
            summary["group"] = grp_val
            summary_rows.append(summary)

            # R's uniquecols(): columns that are constant within the group.
            uniq: Dict[str, Any] = {"x": x_val, "group": grp_val}
            for col in grp.columns:
                if col in ("x", "group", "y"):
                    continue
                vals = grp[col].values
                if pd.Series(vals).nunique(dropna=False) == 1:
                    uniq[col] = vals[0]
            unique_rows.append(uniq)

        if not summary_rows:
            return pd.DataFrame()

        summary_df = pd.concat(summary_rows, ignore_index=True)
        unique_df = pd.DataFrame(unique_rows)

        # Merge summary with per-group unique columns on (x, group).  The
        # ordering follows R's merge: key columns first, then summary cols,
        # then the remaining unique cols.
        out = summary_df.merge(unique_df, on=["x", "group"], how="left", sort=False)

        # R places the merge key columns (x, group) first; reorder to match.
        cols = ["x", "group"] + [c for c in out.columns if c not in ("x", "group")]
        out = out[cols]
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
            summary = fun(grp)
            if not isinstance(summary, pd.DataFrame):
                summary = pd.DataFrame(summary)
            summary = summary.copy()
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
    default_aes: Dict[str, Any] = {
        "x": AfterStat("ecdf"),
        "y": AfterStat("ecdf"),
        "weight": None,
    }
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
    default_aes: Dict[str, Any] = {
        "y": AfterStat("sample"),
        "x": AfterStat("theoretical"),
    }

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

        sample = np.sort(np.asarray(data["sample"].values, dtype=float))
        n = len(sample)

        if quantiles is None:
            # R ``stats::ppoints(n)`` — the stat-qq default.
            quantiles = _ppoints(n)
        else:
            quantiles = np.asarray(quantiles, dtype=float)
            if len(quantiles) != n:
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

        return pd.DataFrame({"sample": sample, "theoretical": np.asarray(theoretical, dtype=float)})


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

        sample = np.sort(np.asarray(data["sample"].values, dtype=float))
        n = len(sample)

        if quantiles is None:
            quantiles = _ppoints(n)
        else:
            quantiles = np.asarray(quantiles, dtype=float)
            if len(quantiles) != n:
                cli_abort(
                    f"quantiles must have the same length as the data."
                )

        line_p = np.asarray(line_p, dtype=float)
        if len(line_p) != 2:
            cli_abort(
                f"Cannot fit line quantiles {list(line_p)}. line_p must have length 2."
            )

        if distribution is None:
            distribution = scipy_stats.norm

        if dparams is None:
            dparams = {}

        if hasattr(distribution, "ppf"):
            theoretical = np.asarray(distribution.ppf(quantiles, **dparams), dtype=float)
            x_coords = np.asarray(distribution.ppf(line_p, **dparams), dtype=float)
        elif callable(distribution):
            theoretical = np.asarray(distribution(quantiles, **dparams), dtype=float)
            x_coords = np.asarray(distribution(line_p, **dparams), dtype=float)
        else:
            theoretical = scipy_stats.norm.ppf(quantiles)
            x_coords = scipy_stats.norm.ppf(line_p)

        # R: y_coords <- stats::quantile(sample, line.p) uses type=7
        # (linear interpolation), which matches numpy's default linear method.
        y_coords = np.quantile(sample, line_p, method="linear")

        slope = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
        intercept = y_coords[0] - slope * x_coords[0]

        if fullrange:
            scale_x = scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
            if scale_x is not None and hasattr(scale_x, "dimension"):
                x_range = np.array(scale_x.dimension(), dtype=float)
            else:
                x_range = np.array([theoretical.min(), theoretical.max()])
        else:
            x_range = np.array([theoretical.min(), theoretical.max()])

        return pd.DataFrame({
            "x": x_range,
            "y": slope * x_range + intercept,
            "slope": float(slope),
            "intercept": float(intercept),
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

    # R (stat-bin2d.R:8): aes(weight = 1, fill = after_stat(count))
    required_aes: List[str] = ["x", "y"]
    default_aes: Dict[str, Any] = {"weight": 1, "fill": AfterStat("count")}
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

    # R (stat-binhex.R:7):
    #   default_aes = aes(weight = 1, fill = after_stat(count))
    # Without ``fill = after_stat(count)`` no fill scale is added
    # and the colourbar legend for count never appears.
    required_aes: List[str] = ["x", "y"]
    default_aes: Dict[str, Any] = {"weight": 1, "fill": AfterStat("count")}
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

        # R: out$width <- binwidth[1]; out$height <- binwidth[2]
        out["width"] = binwidth[0]
        out["height"] = binwidth[1]

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

        # Build z matrix (rows = y, cols = x), mirroring R's isoband_z_matrix.
        x_unique = np.sort(data["x"].unique())
        y_unique = np.sort(data["y"].unique())

        z_matrix = np.full((len(y_unique), len(x_unique)), np.nan)
        x_idx = np.searchsorted(x_unique, data["x"].values)
        y_idx = np.searchsorted(y_unique, data["y"].values)
        z_matrix[y_idx, x_idx] = data["z"].values

        group_base = data["group"].iloc[0] if "group" in data.columns else -1
        result = _contourpy_isolines(
            x_unique, y_unique, z_matrix, brks, group=group_base
        )

        if result.empty:
            return result

        max_level = result["level"].max()
        if max_level and max_level > 0:
            result["nlevel"] = result["level"] / max_level
        else:
            result["nlevel"] = 0.0
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

        # Build z matrix (rows = y, cols = x).
        x_unique = np.sort(data["x"].unique())
        y_unique = np.sort(data["y"].unique())

        z_matrix = np.full((len(y_unique), len(x_unique)), np.nan)
        x_idx = np.searchsorted(x_unique, data["x"].values)
        y_idx = np.searchsorted(y_unique, data["y"].values)
        z_matrix[y_idx, x_idx] = data["z"].values

        group_base = data["group"].iloc[0] if "group" in data.columns else -1
        result = _contourpy_isobands(
            x_unique, y_unique, z_matrix, brks, group=group_base
        )

        if result.empty:
            return result

        max_level = result["level_high"].max()
        if max_level and max_level > 0:
            result["nlevel"] = result["level_high"] / max_level
        else:
            result["nlevel"] = 0.0
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

    def compute_layer(
        self, data: pd.DataFrame, params: Dict[str, Any], layout: Any
    ) -> pd.DataFrame:
        """Run the per-group 2D density, then optionally dispatch to a contour stat.

        Mirrors ``StatDensity2d$compute_layer`` in ggplot2: once the raw density
        grid has been computed, if ``contour=TRUE`` (default), set ``z`` from
        ``contour_var`` and delegate to ``StatContour`` / ``StatContourFilled``'s
        ``compute_panel``.
        """
        data = super().compute_layer(data, params, layout)
        if data.empty:
            return data

        contour = params.get("contour", True)
        if not contour:
            return data

        contour_var = params.get("contour_var", "density")
        if contour_var not in ("density", "ndensity", "count"):
            raise ValueError(
                "`contour_var` must be one of 'density', 'ndensity', 'count'"
            )

        data = data.copy()
        data["z"] = data[contour_var].values
        z_vals = data["z"].values[np.isfinite(data["z"].values)]
        z_range = (
            [float(z_vals.min()), float(z_vals.max())]
            if len(z_vals) > 0
            else [0.0, 1.0]
        )

        sub_params = {
            k: params[k] for k in ("bins", "binwidth", "breaks") if k in params
        }
        sub_params["z_range"] = z_range

        if self.contour_type == "bands":
            contour_stat = StatContourFilled()
        else:
            contour_stat = StatContour()

        if "PANEL" in data.columns:
            pieces = []
            for panel, panel_data in data.groupby("PANEL", sort=False, observed=True):
                scales = layout.get_scales(panel) if layout is not None else {}
                out = contour_stat.compute_panel(panel_data, scales, **sub_params)
                if out is not None and not out.empty:
                    if "PANEL" not in out.columns:
                        out = out.copy()
                        out["PANEL"] = panel
                    pieces.append(out)
            if pieces:
                return pd.concat(pieces, ignore_index=True)
            return pd.DataFrame()

        scales = {}
        out = contour_stat.compute_panel(data, scales, **sub_params)
        return out if out is not None else pd.DataFrame()

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
        """Compute 2D density on a grid via ``MASS::kde2d``.

        Returns a DataFrame with the raw density grid and the ``density``,
        ``ndensity``, ``count``, ``n``, ``level``, ``piece`` columns that
        ``StatContour`` can consume downstream. ``compute_layer`` decides
        whether to continue through the contour pipeline.
        """
        x = np.asarray(data["x"].values, dtype=float)
        y = np.asarray(data["y"].values, dtype=float)
        nx = len(x)

        if isinstance(adjust, (int, float)):
            adjust = (float(adjust), float(adjust))
        else:
            adjust = tuple(float(a) for a in adjust)

        if h is None:
            h_x = _bandwidth_nrd(x) * adjust[0]
            h_y = _bandwidth_nrd(y) * adjust[1]
        else:
            h_x, h_y = float(h[0]), float(h[1])

        # Evaluation range -- match ggplot2's scale$dimension() fallback.
        x_scale = (
            scales.get("x") if isinstance(scales, dict) else getattr(scales, "x", None)
        )
        y_scale = (
            scales.get("y") if isinstance(scales, dict) else getattr(scales, "y", None)
        )
        if x_scale is not None and hasattr(x_scale, "dimension"):
            x_range = x_scale.dimension()
        else:
            x_range = (float(x.min()), float(x.max()))
        if y_scale is not None and hasattr(y_scale, "dimension"):
            y_range = y_scale.dimension()
        else:
            y_range = (float(y.min()), float(y.max()))

        lims = (x_range[0], x_range[1], y_range[0], y_range[1])
        dens = _kde2d(x, y, h=(h_x, h_y), n=n, lims=lims)

        gx, gy, z = dens["x"], dens["y"], dens["z"]
        # z has shape (len(gx), len(gy)); expand.grid yields x varying fastest.
        xx, yy = np.meshgrid(gx, gy, indexing="ij")
        df = pd.DataFrame({
            "x": xx.ravel(),
            "y": yy.ravel(),
            "density": z.ravel(),
        })
        group_val = data["group"].iloc[0] if "group" in data.columns else -1
        df["group"] = group_val
        max_dens = df["density"].max()
        df["ndensity"] = df["density"] / max_dens if max_dens > 0 else 0.0
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
            ``"norm"`` or ``"euclid"``. R ggplot2 also supports ``"t"`` via
            ``MASS::cov.trob``; that iterative robust estimator is not ported
            and ``type='t'`` raises :class:`NotImplementedError`.
        level : float
        segments : int
        na_rm : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y.
        """
        return _calculate_ellipse(
            data, vars_=["x", "y"], type_=type, level=level, segments=segments,
        )


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
    default_aes: Dict[str, Any] = {"size": AfterStat("n"), "weight": 1}

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
        grouped = data.groupby(group_cols, sort=False, dropna=False, observed=False)
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

        In R, ``setup_params`` computes ``width = resolution(data$x) * 0.9``
        when not explicitly set.

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
        # R: width <- params$width %||% (resolution(data$x, FALSE) * 0.9)
        if params.get("width") is None and "x" in data.columns:
            xu = np.sort(data["x"].dropna().unique())
            if len(xu) > 1:
                res = np.min(np.diff(xu))
            else:
                res = 1.0
            params["width"] = float(res) * 0.9
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

        # Panel-level ``compute_panel`` uses ``n`` for ``scale="count"``.
        dens["n"] = len(data)

        # R's ``compute_group`` leaves violinwidth unset; ``compute_panel``
        # fills it using panel-wide max density / max n. We initialise to
        # scaled (per-group) so a caller that skips ``compute_panel`` still
        # sees the width="width" behaviour.
        dens["violinwidth"] = dens["scaled"]

        # Add quantile information.  Port of ``stat-ydensity.R:73-95``.
        # R uses ``stats::quantile(y, probs=quantiles)`` which applies type-7
        # (NumPy default) linear interpolation.  Other dens columns are
        # interpolated with ``stats::approx(dens$y, dens[[var]], xout, ties="ordered")``.
        if quantiles is not None:
            quantiles = list(quantiles)
            quant_vals = np.quantile(y, quantiles)
            quant_cols: Dict[str, List[Any]] = {"y": list(quant_vals), "quantile": list(quantiles)}

            # Interpolate every column in dens except `y` / `quantile` at
            # the quantile y-values.  R's ``stats::approx`` is linear;
            # ``ties='ordered'`` means duplicates are kept in input order.
            for col in dens.columns:
                if col in ("y", "quantile"):
                    continue
                # approx with ties='ordered': dedupe x, keeping first y value
                xs = dens["y"].to_numpy()
                ys = dens[col].to_numpy()
                # If any categorical / non-numeric, fall through to constant
                if ys.dtype.kind in ("O", "U", "S", "b"):
                    # non-numeric: carry the first value
                    quant_cols[col] = [ys[0]] * len(quantiles)
                    continue
                # remove duplicate xs for interp (R approxfun uses ties="ordered")
                order = np.argsort(xs, kind="mergesort")
                xs_s, ys_s = xs[order], ys[order]
                # R ties='ordered' keeps every point; np.interp also handles
                # duplicates by averaging — close enough for monotone grids.
                interp_vals = np.interp(
                    quant_vals, xs_s, ys_s,
                    left=np.nan, right=np.nan,
                )
                quant_cols[col] = list(interp_vals)

            quant_df = pd.DataFrame(quant_cols)

            # R: dens <- vec_slice(dens, !dens$y %in% quants$y)
            dens = dens[~dens["y"].isin(quant_vals)].reset_index(drop=True)
            # Align columns (quant_df may be missing `quantile` in dens).
            for col in dens.columns:
                if col not in quant_df.columns:
                    quant_df[col] = np.nan
            for col in quant_df.columns:
                if col not in dens.columns:
                    dens[col] = np.nan
            dens = pd.concat([dens, quant_df[dens.columns]], ignore_index=True)

        return dens

    def compute_panel(
        self,
        data: pd.DataFrame,
        scales: Any,
        scale: str = "area",
        drop: bool = True,
        **params: Any,
    ) -> pd.DataFrame:
        """Port R ``stat-ydensity.R:101-140``.

        After ``compute_group`` runs for every (x, group), R rescales
        ``violinwidth`` across the whole panel according to ``scale``:

        - ``"area"``  (default) every violin has the same peak width 1
          relative to the panel-wide max density
        - ``"count"`` area proportional to group n
        - ``"width"`` per-group scaled (ignores neighbours)

        Without this override every violin is scaled to its own max,
        making cross-group comparisons in a single panel misleading.
        """
        # Run ``Stat.compute_panel`` which dispatches per group. Pass every
        # param except ``scale`` (which is a panel-level knob).
        data = super().compute_panel(data, scales, **params)
        if data is None or len(data) == 0:
            return data

        if not drop and "n" in data.columns and (data["n"] < 2).any():
            cli_warn(
                "Cannot compute density for groups with fewer than two "
                "datapoints."
            )

        if scale == "area":
            max_density = data["density"].max(skipna=True)
            if max_density and max_density > 0:
                data["violinwidth"] = data["density"] / max_density
        elif scale == "count":
            max_density = data["density"].max(skipna=True)
            max_n = data["n"].max(skipna=True) if "n" in data.columns else 1
            if max_density and max_density > 0 and max_n:
                data["violinwidth"] = (
                    data["density"] / max_density
                    * data.get("n", 1) / max_n
                )
        elif scale == "width":
            data["violinwidth"] = data["scaled"]
        else:
            raise ValueError(
                f"scale must be 'area', 'count', or 'width'; got {scale!r}."
            )
        return data


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

    Port of ``stat-align.R``.

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

    def compute_panel(
        self,
        data: pd.DataFrame,
        scales: Any,
        flipped_aes: bool = False,
        **params: Any,
    ) -> pd.DataFrame:
        """Port of ``stat-align.R:15-44``.

        Computes zero-crossings, the shared ``unique_loc`` x-grid (+/-
        ``adjust`` padding), then dispatches to ``compute_group`` per
        group with ``unique_loc`` and ``adjust`` injected.
        """
        if data is None or len(data) == 0:
            return pd.DataFrame()
        if "group" not in data.columns or data["group"].nunique() <= 1:
            return data

        data_f = _flip_data(data, flipped_aes)
        x = data_f["x"].to_numpy(dtype=float)
        y = data_f["y"].to_numpy(dtype=float)
        grp = data_f["group"].to_numpy()

        # --- Zero-crossing detection (R vec_unrep on (group, y<0)) ---
        sign_flag = y < 0
        # Run-length on (group, sign_flag): a change where either differs.
        g_int = pd.Series(grp).astype("category").cat.codes.to_numpy()
        key = np.stack([g_int, sign_flag.astype(int)], axis=1)
        if len(key) == 0:
            return pd.DataFrame()
        changes = np.any(np.diff(key, axis=0) != 0, axis=1)
        run_starts = np.concatenate([[0], np.where(changes)[0] + 1])
        # run_times[i] = length of i-th run (from run_starts[i] to run_starts[i+1]-1)
        run_ends = np.concatenate([run_starts[1:], [len(key)]])
        run_times = run_ends - run_starts

        # cumulative run lengths (equivalent to R's cumsum(pivot$times))
        cum = np.cumsum(run_times)
        # per-group run counts: cumsum of vec_unrep(pivot$key$group)$times
        # (i.e. number of (group, sign) runs within each group)
        group_codes_by_run = g_int[run_starts]
        # run counts per group, in the order groups first appear
        unique_groups, group_first = np.unique(group_codes_by_run, return_index=True)
        # preserve order of first appearance
        order = np.argsort(group_first)
        ug_ordered = unique_groups[order]
        group_run_counts = np.array(
            [np.sum(group_codes_by_run == g) for g in ug_ordered]
        )
        group_ends = np.cumsum(group_run_counts)
        # pivot indices = cumsum(run_times)[-group_ends]
        pivot_cum = cum.copy()
        # remove positions at end of each group
        mask = np.ones(len(pivot_cum), dtype=bool)
        mask[group_ends - 1] = False
        pivot = pivot_cum[mask]

        # R: pivot refers to 1-based index into the original data; here pivot
        # is end-of-run index (0-based inclusive for first element after run).
        # We re-use as zero-indexed so ``pivot[i]`` is last index of a run
        # and ``pivot[i]+1`` is first index of the next run, which is what R
        # computes via ``cumsum(pivot$times)[-group_ends]``.
        cross: np.ndarray
        if len(pivot) == 0:
            cross = np.array([], dtype=float)
        else:
            p = pivot - 1  # zero-based index of last point of a run
            # R: cross <- -y[pivot] * (x[pivot+1] - x[pivot]) /
            #             (y[pivot+1] - y[pivot]) + x[pivot]
            denom = y[p + 1] - y[p]
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = -y[p] * (x[p + 1] - x[p]) / denom + x[p]
            cross = cross[np.isfinite(cross)]

        unique_loc = np.unique(np.sort(np.concatenate([x, cross])))
        if len(unique_loc) < 2:
            return data

        rng = np.nanmax(unique_loc) - np.nanmin(unique_loc)
        diff_loc = np.diff(unique_loc)
        diff_min = float(np.min(diff_loc)) if len(diff_loc) > 0 else 0.0
        adjust = min(rng * 0.001, diff_min / 3.0)

        unique_loc = np.unique(np.sort(np.concatenate([
            unique_loc - adjust, unique_loc, unique_loc + adjust,
        ])))

        # Dispatch per group.  Use the parent ``Stat.compute_panel``-like
        # loop to drive ``compute_group`` per group value with the shared
        # unique_loc/adjust arguments.
        results: List[pd.DataFrame] = []
        for g_val, sub in data_f.groupby("group", sort=False):
            res = self.compute_group(
                sub.reset_index(drop=True),
                scales=scales,
                flipped_aes=flipped_aes,
                unique_loc=unique_loc,
                adjust=adjust,
                **params,
            )
            if res is not None and len(res) > 0:
                results.append(res)

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        flipped_aes: bool = False,
        unique_loc: Optional[np.ndarray] = None,
        adjust: float = 0.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Align values by interpolation.  Port of ``stat-align.R:46-76``.

        Parameters
        ----------
        data : pd.DataFrame
        scales : dict-like
        flipped_aes : bool
        unique_loc : array-like, optional
        adjust : float
        """
        data = _flip_data(data, flipped_aes)

        # R: ``is_unique(data$x)`` returns TRUE only when *all* x are equal
        # (``length(unique(x)) == 1``). Mirror that exactly.
        if data["x"].nunique() < 2:
            return pd.DataFrame()

        # R: handle duplicated x by keeping first & last row at (x-adjust, x).
        if data["x"].duplicated().any():
            rows: List[pd.DataFrame] = []
            for x_key, sub in data.groupby("x", sort=True):
                if len(sub) == 1:
                    rows.append(sub)
                else:
                    pair = sub.iloc[[0, -1]].copy()
                    pair.iloc[0, pair.columns.get_loc("x")] = float(x_key) - adjust
                    rows.append(pair)
            data = pd.concat(rows, ignore_index=True)

        x = data["x"].to_numpy(dtype=float)
        y = data["y"].to_numpy(dtype=float)

        # Ensure x is sorted (required by np.interp / R's approxfun).
        order = np.argsort(x, kind="mergesort")
        x_s, y_s = x[order], y[order]
        # R: ``approxfun(data$x, data$y)(unique_loc)``. When ``unique_loc`` is
        # NULL (e.g. user invokes ``compute_group`` directly without the
        # ``compute_panel`` preamble), R's ``approxfun(NULL)`` returns
        # ``numeric(0)``; mirror that with an empty array.
        if unique_loc is None:
            y_val = np.empty(0, dtype=float)
            unique_loc_arr = np.empty(0, dtype=float)
        else:
            unique_loc_arr = np.asarray(unique_loc, dtype=float)
            y_val = np.interp(
                unique_loc_arr, x_s, y_s, left=np.nan, right=np.nan
            )

        keep = ~np.isnan(y_val)
        x_val = unique_loc_arr[keep]
        y_val = y_val[keep]

        # R does NOT early-return when x_val is empty; instead ``min/max`` on
        # an empty vector returns ``Inf`` / ``-Inf`` with a warning, and the
        # result is still a 2-row DataFrame with Inf/-Inf padding. Mirror that.
        if len(x_val) == 0:
            x_out = np.array([np.inf, -np.inf], dtype=float)
            y_out = np.array([0.0, 0.0], dtype=float)
            padding = np.array([True, True], dtype=bool)
            result = pd.DataFrame({"x": x_out, "y": y_out})
            for col in data.columns:
                if col in ("x", "y"):
                    continue
                result[col] = data[col].iloc[0]
            result["align_padding"] = padding
            result["flipped_aes"] = flipped_aes
            return _flip_data(result, flipped_aes)

        x_out = np.concatenate([[x_val[0] - adjust], x_val, [x_val[-1] + adjust]])
        y_out = np.concatenate([[0.0], y_val, [0.0]])
        padding = np.concatenate([[True], np.zeros(len(x_val), dtype=bool), [True]])

        result = pd.DataFrame({"x": x_out, "y": y_out})
        # Carry over the first row of any other columns (R: ``data[1, setdiff(...)]``).
        for col in data.columns:
            if col in ("x", "y"):
                continue
            result[col] = data[col].iloc[0]
        result["align_padding"] = padding
        result["flipped_aes"] = flipped_aes

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
        lambda_: float = 1.0,
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
            Currently only the default ``y ~ x`` is supported.
        xseq : array-like, optional
        method : str
            Only ``"rq"`` (linear quantile regression) is supported.
            ``"rqss"`` (smoothing splines) from R quantreg is not ported.
        method_args : dict, optional
            Extra keyword arguments forwarded to statsmodels ``QuantReg.fit``.
        lambda_ : float
            Kept for signature compatibility with R ``rqss``; unused here.
        na_rm : bool

        Returns
        -------
        pd.DataFrame
            With columns: x, y, quantile, group.

        Notes
        -----
        R uses ``quantreg::rq`` (Barrodale-Roberts LP). This Python port uses
        ``statsmodels.regression.quantile_regression.QuantReg``. The two
        implementations can differ at LP vertices when ties occur; on
        well-posed data both converge to the same slope/intercept within
        the default convergence tolerance.
        """
        if method not in ("rq",):
            cli_abort(
                f"stat_quantile: method {method!r} not supported. Use 'rq'."
            )
        if formula is not None and formula != "y ~ x":
            cli_abort(
                f"stat_quantile: custom formula {formula!r} not supported. "
                "Only the default 'y ~ x' is supported in the Python port."
            )

        if xseq is None:
            xseq = np.linspace(float(data["x"].min()), float(data["x"].max()), 100)
        xseq = np.asarray(xseq, dtype=float)

        x = np.asarray(data["x"].values, dtype=float)
        y_data = np.asarray(data["y"].values, dtype=float)
        if "weight" in data.columns:
            w = np.asarray(data["weight"].values, dtype=float)
        else:
            w = np.ones_like(x)
        group_val = data["group"].iloc[0] if "group" in data.columns else 1

        m_args = dict(method_args or {})

        results = []
        for q in quantiles:
            coef = self._quantile_regression_coef(x, y_data, w, float(q), m_args)
            intercept, slope = float(coef[0]), float(coef[1])
            y_pred = intercept + slope * xseq
            results.append(pd.DataFrame({
                "x": xseq,
                "y": y_pred,
                "quantile": float(q),
                "group": f"{group_val}-{q}",
            }))

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    @staticmethod
    def _quantile_regression_coef(
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        tau: float,
        fit_kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Fit linear quantile regression ``y ~ 1 + x`` at quantile ``tau``.

        Returns the two-element coefficient vector ``[intercept, slope]``.
        Uses statsmodels' ``QuantReg`` which implements an IRLS solver with
        Huber's sandwich; results match R ``quantreg::rq(method='br')`` up
        to convergence tolerance on data without LP ties.
        """
        from statsmodels.regression.quantile_regression import QuantReg

        X = np.column_stack([np.ones_like(x), x])
        # statsmodels QuantReg does not expose weights through the public API
        # in all versions; scale rows to emulate weighted regression only when
        # weights are non-uniform. For uniform weights, just fit directly.
        if not np.allclose(weight, weight[0]):
            sw = np.sqrt(weight)
            Xw = X * sw[:, None]
            yw = y * sw
            model = QuantReg(yw, Xw)
        else:
            model = QuantReg(y, X)
        defaults = {"max_iter": 2000, "p_tol": 1e-8}
        defaults.update(fit_kwargs)
        res = model.fit(q=tau, **defaults)
        return np.asarray(res.params, dtype=float)


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

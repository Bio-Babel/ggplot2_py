"""
Base Scale classes and constructor functions for the ggplot2 scale system.

This module implements the core Scale hierarchy:

- :class:`Scale` -- abstract base
- :class:`ScaleContinuous` -- continuous data
- :class:`ScaleDiscrete` -- discrete / categorical data
- :class:`ScaleBinned` -- binned continuous data
- Position sub-classes (:class:`ScaleContinuousPosition`, etc.)
- Identity sub-classes (:class:`ScaleContinuousIdentity`, etc.)
- Date/datetime sub-classes

Also provides constructor helpers:

- :func:`continuous_scale`
- :func:`discrete_scale`
- :func:`binned_scale`

Container class :class:`ScalesList`, secondary axis support
(:class:`AxisSecondary`, :func:`sec_axis`, :func:`dup_axis`),
and expansion helpers (:func:`expansion`, :func:`expand_scale`).
"""

from __future__ import annotations

import copy
import math
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

from scales import (
    ContinuousRange,
    DiscreteRange,
    as_transform,
    breaks_extended,
    censor,
    discard,
    expand_range,
    is_transform,
    oob_censor,
    oob_squish,
    oob_squish_infinite,
    rescale,
    rescale_mid,
    rescale_max,
    squish,
    train_continuous,
    train_discrete,
    transform_identity,
    zero_range,
)

from ggplot2_py._compat import (
    Waiver,
    cli_abort,
    cli_inform,
    cli_warn,
    deprecate_warn,
    is_waiver,
    waiver,
)
from ggplot2_py.aes import standardise_aes_names
from ggplot2_py.ggproto import GGProto, fetch_ggproto, ggproto, ggproto_parent

__all__ = [
    # Base classes
    "Scale",
    "ScaleContinuous",
    "ScaleDiscrete",
    "ScaleBinned",
    # Position sub-classes
    "ScaleContinuousPosition",
    "ScaleDiscretePosition",
    "ScaleBinnedPosition",
    # Identity sub-classes
    "ScaleContinuousIdentity",
    "ScaleDiscreteIdentity",
    # Date/datetime sub-classes
    "ScaleContinuousDate",
    "ScaleContinuousDatetime",
    # Constructors
    "continuous_scale",
    "discrete_scale",
    "binned_scale",
    # Container
    "ScalesList",
    "scales_list",
    # Secondary axis
    "AxisSecondary",
    "sec_axis",
    "dup_axis",
    "derive",
    "is_derived",
    "is_sec_axis",
    # Expansion helpers
    "expansion",
    "expand_scale",
    "expand_range4",
    "default_expansion",
    # Scale detection
    "find_scale",
    "is_scale",
    # Mapped discrete sentinel
    "mapped_discrete",
    "is_mapped_discrete",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_POSITION_AESTHETICS = frozenset(
    [
        "x",
        "xmin",
        "xmax",
        "xend",
        "xintercept",
        "xmin_final",
        "xmax_final",
        "xlower",
        "xmiddle",
        "xupper",
        "x0",
        "y",
        "ymin",
        "ymax",
        "yend",
        "yintercept",
        "ymin_final",
        "ymax_final",
        "ylower",
        "ymiddle",
        "yupper",
        "y0",
    ]
)

_X_AESTHETICS = [
    "x", "xmin", "xmax", "xend", "xintercept",
    "xmin_final", "xmax_final", "xlower", "xmiddle", "xupper", "x0",
]

_Y_AESTHETICS = [
    "y", "ymin", "ymax", "yend", "yintercept",
    "ymin_final", "ymax_final", "ylower", "ymiddle", "yupper", "y0",
]


def _is_position_aes(aesthetics: Union[str, Sequence[str]]) -> bool:
    """Return True if any of *aesthetics* is a position aesthetic."""
    if isinstance(aesthetics, str):
        return aesthetics in _POSITION_AESTHETICS
    return any(a in _POSITION_AESTHETICS for a in aesthetics)


def _is_discrete(x: Any) -> bool:
    """Check whether *x* should be treated as discrete data."""
    if isinstance(x, pd.Categorical) or isinstance(x, pd.CategoricalDtype):
        return True
    if isinstance(x, pd.Series):
        if isinstance(x.dtype, pd.CategoricalDtype):
            return True
        if x.dtype == object:
            return True
        if pd.api.types.is_bool_dtype(x.dtype):
            return True
        return False
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("U", "S", "O", "b"):
            return True
        return False
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        first = x[0]
        return isinstance(first, (str, bool))
    if isinstance(x, (str, bool)):
        return True
    return False


def _empty(df: Any) -> bool:
    """Check whether *df* is empty (None or zero-length)."""
    if df is None:
        return True
    if isinstance(df, pd.DataFrame):
        return len(df) == 0
    if isinstance(df, dict):
        return len(df) == 0
    return False


def _unique0(x: Any) -> np.ndarray:
    """Unique values preserving order."""
    if x is None:
        return np.array([])
    arr = np.asarray(x)
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def _check_breaks_labels(
    breaks: Any,
    labels: Any,
) -> None:
    """Validate that breaks and labels are compatible."""
    if breaks is None or labels is None:
        return
    if isinstance(breaks, np.ndarray) and np.isscalar(breaks) and np.isnan(breaks):
        cli_abort("Invalid breaks specification. Use None, not NaN.")
    if (
        not callable(breaks)
        and not callable(labels)
        and not is_waiver(breaks)
        and not is_waiver(labels)
    ):
        breaks_arr = np.asarray(breaks) if not isinstance(breaks, (list, tuple)) else breaks
        labels_arr = labels if not isinstance(labels, (list, tuple)) else labels
        if hasattr(breaks_arr, "__len__") and hasattr(labels_arr, "__len__"):
            if len(breaks_arr) != len(labels_arr):
                cli_abort("breaks and labels must have the same length.")


def _is_finite(x: Any) -> np.ndarray:
    """Element-wise finite check."""
    arr = np.asarray(x, dtype=float)
    return np.isfinite(arr)


# ---------------------------------------------------------------------------
# Mapped discrete sentinel
# ---------------------------------------------------------------------------

class _MappedDiscrete(np.ndarray):
    """Sentinel wrapper for discrete values that have been mapped to numeric."""

    def __new__(cls, x: Any) -> "_MappedDiscrete":
        arr = np.asarray(x, dtype=float).view(cls)
        return arr


def mapped_discrete(x: Any) -> Optional[_MappedDiscrete]:
    """Wrap *x* as a mapped-discrete array."""
    if x is None:
        return None
    return _MappedDiscrete(x)


def is_mapped_discrete(x: Any) -> bool:
    """Check whether *x* is a mapped discrete array."""
    return isinstance(x, _MappedDiscrete)


# ---------------------------------------------------------------------------
# Expansion helpers
# ---------------------------------------------------------------------------

def expansion(
    mult: Union[float, Sequence[float]] = 0,
    add: Union[float, Sequence[float]] = 0,
) -> np.ndarray:
    """Generate an expansion vector for scale padding.

    Parameters
    ----------
    mult : float or sequence of float
        Multiplicative range expansion factors.  If length 1, both limits
        use the same value; if length 2, ``(lower, upper)``.
    add : float or sequence of float
        Additive range expansion constants.

    Returns
    -------
    numpy.ndarray
        Length-4 array ``[mult_lower, add_lower, mult_upper, add_upper]``.
    """
    mult = np.atleast_1d(np.asarray(mult, dtype=float))
    add = np.atleast_1d(np.asarray(add, dtype=float))
    if len(mult) == 1:
        mult = np.repeat(mult, 2)
    if len(add) == 1:
        add = np.repeat(add, 2)
    if len(mult) != 2 or len(add) != 2:
        cli_abort("mult and add must be numeric vectors with 1 or 2 elements.")
    return np.array([mult[0], add[0], mult[1], add[1]])


def expand_scale(
    mult: Union[float, Sequence[float]] = 0,
    add: Union[float, Sequence[float]] = 0,
) -> np.ndarray:
    """Deprecated. Use :func:`expansion` instead.

    Parameters
    ----------
    mult : float or sequence of float
        Multiplicative range expansion factors.
    add : float or sequence of float
        Additive range expansion constants.

    Returns
    -------
    numpy.ndarray
        Length-4 expansion vector.
    """
    deprecate_warn("3.3.0", "expand_scale()", with_="expansion()")
    return expansion(mult, add)


def expand_range4(
    limits: Sequence[float],
    expand: np.ndarray,
) -> np.ndarray:
    """Expand a numeric range with a 2- or 4-element expansion vector.

    Parameters
    ----------
    limits : array-like
        Length-2 numeric range.
    expand : array-like
        2- or 4-element expansion vector ``[mult_lo, add_lo, mult_hi, add_hi]``
        or ``[mult, add]`` (duplicated for both sides).

    Returns
    -------
    numpy.ndarray
        Expanded limits (length 2).
    """
    expand = np.asarray(expand, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if len(expand) not in (2, 4):
        cli_abort("expand must be a numeric vector with 2 or 4 elements.")
    if not np.any(np.isfinite(limits)):
        return np.array([-np.inf, np.inf])
    if len(expand) == 2:
        expand = np.tile(expand, 2)
    # expand = [mult_lower, add_lower, mult_upper, add_upper]
    # Compute expansion inline to handle asymmetric mult/add correctly.
    # scales.expand_range only accepts scalar mul/add, so we compute manually.
    extent = limits[1] - limits[0]
    if extent == 0:
        extent = 1.0
    lower = limits[0] - extent * expand[0] - expand[1]
    upper = limits[1] + extent * expand[2] + expand[3]
    return np.array([lower, upper])


def default_expansion(
    scale: Any,
    discrete: Optional[np.ndarray] = None,
    continuous: Optional[np.ndarray] = None,
    expand: bool = True,
) -> np.ndarray:
    """Compute the default expansion for a scale.

    Parameters
    ----------
    scale : Scale
        A position scale.
    discrete : array-like, optional
        Default expansion for discrete scales.
    continuous : array-like, optional
        Default expansion for continuous scales.
    expand : bool
        Whether to apply expansion at all.

    Returns
    -------
    numpy.ndarray
        Length-4 expansion vector.
    """
    if discrete is None:
        discrete = expansion(add=0.6)
    if continuous is None:
        continuous = expansion(mult=0.05)
    out = expansion()
    if not expand:
        return out
    scale_expand = scale.expand
    if is_waiver(scale_expand):
        scale_expand = discrete if scale.is_discrete() else continuous
    scale_expand = np.asarray(scale_expand, dtype=float)
    if len(scale_expand) < 4:
        scale_expand = np.tile(scale_expand, 2)[:4]
    out[0:2] = scale_expand[0:2]
    out[2:4] = scale_expand[2:4]
    return out


# ---------------------------------------------------------------------------
# Base Scale class
# ---------------------------------------------------------------------------

class Scale(GGProto):
    """Abstract base class for all ggplot2 scales.

    Scales translate data values to aesthetic values and populate
    breaks and labels.
    """

    # --- Auto-registration registry (Python-exclusive) -------------------
    _registry: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if name.startswith("Scale") and len(name) > 5:
            key = name[5:]  # strip "Scale" prefix
            Scale._registry[key] = cls
            Scale._registry[key.lower()] = cls

    call: Optional[str] = None
    aesthetics: List[str] = []
    palette: Optional[Callable] = None
    fallback_palette: Optional[Callable] = None
    limits: Any = None
    na_value: Any = np.nan
    expand: Any = waiver()
    name: Any = waiver()
    breaks: Any = waiver()
    labels: Any = waiver()
    guide: Any = "legend"
    position: str = "left"

    # -- Transformation -------------------------------------------------------

    def transform_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply transform to matching columns in *df*.

        Parameters
        ----------
        df : pandas.DataFrame
            Layer data.

        Returns
        -------
        dict
            Transformed columns keyed by aesthetic name.
        """
        if _empty(df):
            return {}
        aesthetics = [a for a in self.aesthetics if a in df.columns]
        if not aesthetics:
            return {}
        return {a: self.transform(df[a]) for a in aesthetics}

    def transform(self, x: Any) -> Any:
        """Transform raw data values.  Must be overridden."""
        cli_abort("Not implemented.")

    # -- Training -------------------------------------------------------------

    def train_df(self, df: pd.DataFrame) -> None:
        """Train scale on matching columns of *df*.

        Parameters
        ----------
        df : pandas.DataFrame
            Layer data.
        """
        if _empty(df):
            return
        aesthetics = [a for a in self.aesthetics if a in df.columns]
        for a in aesthetics:
            self.train(df[a])

    def train(self, x: Any) -> None:
        """Train on a vector.  Must be overridden."""
        cli_abort("Not implemented.")

    # -- Mapping --------------------------------------------------------------

    def map_df(self, df: pd.DataFrame, i: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Map matching columns in *df* to aesthetic values.

        Parameters
        ----------
        df : pandas.DataFrame
            Layer data.
        i : array-like, optional
            Row index subset.

        Returns
        -------
        dict
            Mapped columns keyed by aesthetic name.
        """
        if _empty(df):
            return {}
        if self.palette is None:
            pal = getattr(self, "fallback_palette", None)
            if pal is not None:
                self.palette = pal
        aesthetics = [a for a in self.aesthetics if a in df.columns]
        if not aesthetics:
            return {}
        result = {}
        for a in aesthetics:
            col = df[a].values if i is None else df[a].values[i]
            result[a] = self.map(col)
        return result

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        """Map data values to aesthetic values.  Must be overridden."""
        cli_abort("Not implemented.")

    def rescale(
        self,
        x: Any,
        limits: Optional[Any] = None,
        range: Optional[Any] = None,
    ) -> Any:
        """Rescale to 0-1 range.  Must be overridden."""
        cli_abort("Not implemented.")

    # -- Getters --------------------------------------------------------------

    def get_limits(self) -> Any:
        """Return the current scale limits (without expansion).

        Returns
        -------
        array-like
            Scale limits.
        """
        if self.is_empty():
            return np.array([0.0, 1.0])
        if self.limits is None:
            return self.range.range
        if callable(self.limits):
            return self.limits(self.range.range)
        return self.limits

    def dimension(
        self,
        expand: Optional[np.ndarray] = None,
        limits: Optional[Any] = None,
    ) -> Any:
        """Return continuous dimension of the scale.  Must be overridden."""
        cli_abort("Not implemented.")

    def get_breaks(self, limits: Optional[Any] = None) -> Any:
        """Resolve and return scale breaks.  Must be overridden."""
        cli_abort("Not implemented.")

    def get_breaks_minor(
        self,
        n: int = 2,
        b: Optional[Any] = None,
        limits: Optional[Any] = None,
    ) -> Any:
        """Resolve and return minor breaks.  Must be overridden."""
        cli_abort("Not implemented.")

    def get_labels(self, breaks: Optional[Any] = None) -> Any:
        """Resolve and return labels for the given breaks.  Must be overridden."""
        cli_abort("Not implemented.")

    def get_transformation(self) -> Any:
        """Return the scale's transformation object.

        Returns
        -------
        Transform
            A scales-package transform object.
        """
        return getattr(self, "trans", transform_identity())

    def break_positions(self, range: Optional[Any] = None) -> Any:
        """Return mapped break positions.

        Parameters
        ----------
        range : array-like, optional
            Scale limits; defaults to ``get_limits()``.

        Returns
        -------
        array-like
            Mapped break positions.
        """
        if range is None:
            range = self.get_limits()
        return self.map(self.get_breaks(range))

    def break_info(self, range: Optional[Any] = None) -> Dict[str, Any]:
        """Return all break-related information.  Must be overridden."""
        cli_abort("Not implemented.")

    # -- Titles ---------------------------------------------------------------

    def make_title(
        self,
        guide_title: Any = None,
        scale_title: Any = None,
        label_title: Any = None,
    ) -> Any:
        """Resolve scale title from guide, scale, and label titles.

        Parameters
        ----------
        guide_title : str or Waiver, optional
            Title from the guide.
        scale_title : str or Waiver, optional
            Title from the scale ``name`` field.
        label_title : str or Waiver, optional
            Title from ``labs()``.

        Returns
        -------
        str or None
            Resolved title.
        """
        if guide_title is None:
            guide_title = waiver()
        if scale_title is None:
            scale_title = waiver()
        if label_title is None:
            label_title = waiver()
        title = label_title
        if callable(scale_title) and not is_waiver(scale_title):
            title = scale_title(title)
        elif not is_waiver(scale_title):
            title = scale_title
        if callable(guide_title) and not is_waiver(guide_title):
            title = guide_title(title)
        elif not is_waiver(guide_title):
            title = guide_title
        return title

    def make_sec_title(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve secondary axis title (delegates to ``make_title``)."""
        return self.make_title(*args, **kwargs)

    # -- Axis order -----------------------------------------------------------

    def axis_order(self) -> List[str]:
        """Return axis order as ``['primary', 'secondary']`` or reversed."""
        order = ["primary", "secondary"]
        if self.position in ("right", "bottom"):
            order = list(reversed(order))
        return order

    # -- Utilities ------------------------------------------------------------

    def clone(self) -> "Scale":
        """Create an untrained copy of this scale.

        Returns
        -------
        Scale
            A new Scale with a fresh ``range``.
        """
        cli_abort("Not implemented.")

    def reset(self) -> None:
        """Reset the scale's range, un-training it."""
        self.range.reset()

    def is_empty(self) -> bool:
        """Whether the scale contains no information for limits.

        Returns
        -------
        bool
        """
        return self.range.range is None and self.limits is None

    def is_discrete(self) -> bool:
        """Whether the scale is discrete.  Must be overridden.

        Returns
        -------
        bool
        """
        cli_abort("Not implemented.")


# ---------------------------------------------------------------------------
# ScaleContinuous
# ---------------------------------------------------------------------------

def _default_transform(self: Any, x: Any) -> Any:
    """Apply the scale's transformation to data values."""
    transformation = self.get_transformation()
    x_arr = np.asarray(x, dtype=float)
    new_x = transformation.transform(x_arr)
    new_x = np.asarray(new_x, dtype=float)
    # Check for introduced infinities
    finite_orig = np.isfinite(x_arr)
    finite_new = np.isfinite(new_x)
    if np.any(finite_orig & ~finite_new):
        cli_warn(
            f"{transformation.name} transformation introduced infinite values."
        )
    return new_x


class ScaleContinuous(Scale):
    """Scale for continuous data.

    Attributes
    ----------
    trans : Transform
        Transformation object from the ``scales`` package.
    rescaler : callable
        Function to rescale values (default ``rescale``).
    oob : callable
        Out-of-bounds handler (default ``censor``).
    minor_breaks : any
        Minor break specification.
    n_breaks : int or None
        Desired number of major breaks.
    """

    na_value: Any = np.nan
    rescaler: Callable = staticmethod(rescale)
    oob: Callable = staticmethod(censor)
    minor_breaks: Any = waiver()
    n_breaks: Optional[int] = None
    trans: Any = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.trans is None:
            cls.trans = transform_identity()

    def is_discrete(self) -> bool:
        return False

    def train(self, x: Any) -> None:
        """Train the continuous range on *x*.

        Parameters
        ----------
        x : array-like
            Numeric data values.
        """
        x_arr = np.asarray(x, dtype=float)
        if len(x_arr) == 0:
            return
        self.range.train(x_arr)

    def is_empty(self) -> bool:
        has_data = self.range.range is not None
        has_limits = callable(self.limits) or (
            self.limits is not None
            and np.all(np.isfinite(np.asarray(self.limits, dtype=float)))
        )
        return not has_data and not has_limits

    def transform(self, x: Any) -> Any:
        """Transform data values using the scale's transformation.

        Parameters
        ----------
        x : array-like
            Raw data values.

        Returns
        -------
        numpy.ndarray
            Transformed values.
        """
        return _default_transform(self, x)

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        """Map data values to aesthetic values via palette.

        Parameters
        ----------
        x : array-like
            Values in transformed space.
        limits : array-like, optional
            Scale limits; defaults to ``get_limits()``.

        Returns
        -------
        numpy.ndarray
            Mapped aesthetic values.
        """
        if limits is None:
            limits = self.get_limits()
        x_arr = np.asarray(x, dtype=float)
        x_oob = self.oob(x_arr, range=limits)
        x_rescaled = self.rescale(x_oob, limits)

        uniq = _unique0(x_rescaled)
        if len(uniq) == 0:
            return np.full_like(x_arr, self.na_value)
        pal = np.asarray(self.palette(uniq))

        # Determine output dtype: colour strings need object dtype
        out_dtype = pal.dtype if pal.dtype.kind in ("U", "S", "O") else float
        scaled = np.full(len(x_arr), self.na_value, dtype=out_dtype)

        for i, u in enumerate(uniq):
            mask = x_rescaled == u
            if np.any(mask):
                scaled[mask] = pal[i] if i < len(pal) else self.na_value

        # Fill NaN from x_rescaled
        nan_mask = np.isnan(x_rescaled)
        scaled[nan_mask] = self.na_value
        return scaled

    def rescale(
        self,
        x: Any,
        limits: Optional[Any] = None,
        range: Optional[Any] = None,
    ) -> np.ndarray:
        """Rescale *x* to [0, 1].

        Parameters
        ----------
        x : array-like
            Values to rescale.
        limits : array-like, optional
            Scale limits.
        range : array-like, optional
            Range to rescale from.  Defaults to *limits*.

        Returns
        -------
        numpy.ndarray
            Rescaled values.
        """
        if limits is None:
            limits = self.get_limits()
        if range is None:
            range = limits
        return self.rescaler(x, from_range=range)

    def get_limits(self) -> np.ndarray:
        if self.is_empty():
            return np.array([0.0, 1.0])
        if self.limits is None:
            return np.asarray(self.range.range)
        if callable(self.limits):
            transformation = self.get_transformation()
            inv = transformation.inverse(np.asarray(self.range.range))
            user_limits = self.limits(inv)
            return np.asarray(transformation.transform(user_limits))
        limits = np.asarray(self.limits, dtype=float)
        r = np.asarray(self.range.range, dtype=float) if self.range.range is not None else limits
        return np.where(np.isnan(limits), r, limits)

    def dimension(
        self,
        expand: Optional[np.ndarray] = None,
        limits: Optional[Any] = None,
    ) -> np.ndarray:
        """Return the (optionally expanded) continuous range.

        Mirrors R ``Scale$dimension`` (scale-.R:713):

            dimension = function(self, expand = expansion(0, 0), limits = ...)

        R's default is **no expansion** (``expansion(0, 0)``).  The
        caller applies expansion explicitly when needed (e.g. at
        panel-param setup).  Python used to default to
        ``expansion(0.05, 0)`` which inflated values that downstream
        consumers — notably ``hex_binwidth = diff(x$dimension()) /
        bins`` — expected to be the raw data extent.

        Parameters
        ----------
        expand : array-like, optional
            Expansion vector (defaults to none).
        limits : array-like, optional
            Scale limits.

        Returns
        -------
        numpy.ndarray
            Length-2 range.
        """
        if expand is None:
            expand = expansion(0, 0)
        if limits is None:
            limits = self.get_limits()
        return expand_range4(limits, expand)

    def get_breaks(self, limits: Optional[Any] = None) -> Optional[np.ndarray]:
        """Resolve and return break positions.

        Parameters
        ----------
        limits : array-like, optional
            Scale limits.

        Returns
        -------
        numpy.ndarray or None
        """
        if self.is_empty():
            return np.array([])
        if limits is None:
            limits = self.get_limits()
        limits = np.asarray(limits, dtype=float)
        transformation = self.get_transformation()

        breaks = self.breaks
        if is_waiver(breaks):
            breaks = transformation.breaks_func

        if breaks is None:
            return None

        if zero_range(limits.astype(float)):
            return np.array([limits[0]])

        if callable(breaks):
            inv_limits = transformation.inverse(limits)
            n_brk = getattr(self, "n_breaks", None)
            if n_brk is not None:
                try:
                    result = breaks(inv_limits, n=n_brk)
                except TypeError:
                    result = breaks(inv_limits)
            else:
                result = breaks(inv_limits)
            breaks_val = np.asarray(result, dtype=float)
        else:
            breaks_val = np.asarray(breaks, dtype=float)

        return np.asarray(transformation.transform(breaks_val), dtype=float)

    def get_breaks_minor(
        self,
        n: int = 2,
        b: Optional[Any] = None,
        limits: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """Resolve minor breaks.

        Parameters
        ----------
        n : int
            Number of minor breaks between major breaks.
        b : array-like, optional
            Major break positions.
        limits : array-like, optional
            Scale limits.

        Returns
        -------
        numpy.ndarray or None
        """
        if limits is None:
            limits = self.get_limits()
        limits = np.asarray(limits, dtype=float)
        if zero_range(limits):
            return None
        if b is None:
            b = self.break_positions()

        minor = self.minor_breaks
        if minor is None:
            return None

        if is_waiver(minor):
            if b is None:
                return None
            transformation = self.get_transformation()
            if not callable(getattr(transformation, "minor_breaks_func", None)):
                return None
            b_finite = np.asarray(b, dtype=float)
            b_finite = b_finite[np.isfinite(b_finite)]
            return np.asarray(transformation.minor_breaks_func(b_finite, limits, n))
        elif callable(minor):
            transformation = self.get_transformation()
            inv_limits = transformation.inverse(limits)
            result = minor(inv_limits)
            return np.asarray(transformation.transform(result), dtype=float)
        else:
            transformation = self.get_transformation()
            return np.asarray(transformation.transform(minor), dtype=float)

    def get_labels(self, breaks: Optional[Any] = None) -> Optional[Any]:
        """Resolve labels for the given breaks.

        Parameters
        ----------
        breaks : array-like, optional
            Break positions.

        Returns
        -------
        list or None
        """
        if breaks is None:
            breaks = self.get_breaks()
        if breaks is None:
            return None

        transformation = self.get_transformation()
        breaks_data = transformation.inverse(np.asarray(breaks, dtype=float))

        labels = self.labels
        if labels is None:
            return None
        if is_waiver(labels):
            return list(transformation.format_func(breaks_data))
        if callable(labels):
            return list(labels(breaks_data))
        return list(labels)

    def clone(self) -> "ScaleContinuous":
        new = copy.copy(self)
        new.range = ContinuousRange()
        return new

    def break_info(self, range: Optional[Any] = None) -> Dict[str, Any]:
        """Compute all break info for position scales.

        Parameters
        ----------
        range : array-like, optional
            The continuous range to compute breaks for.

        Returns
        -------
        dict
        """
        if range is None:
            range = self.dimension()
        range = np.asarray(range, dtype=float)

        major = self.get_breaks(range)
        labels = self.get_labels(major)
        minor = self.get_breaks_minor(b=major, limits=range)
        if minor is not None:
            minor = minor[~np.isnan(minor)]

        # Censor out-of-range
        if major is not None:
            major_arr = np.asarray(major, dtype=float)
            oob_mask = (major_arr < range[0]) | (major_arr > range[1])
            if labels is not None:
                labels = [l for l, m in zip(labels, ~oob_mask) if m]
            major_arr = major_arr[~oob_mask]
        else:
            major_arr = None

        major_n = rescale(major_arr, from_range=range) if major_arr is not None else None
        minor_n = rescale(minor, from_range=range) if minor is not None else None

        return {
            "range": range,
            "labels": labels,
            "major": major_n,
            "minor": minor_n,
            "major_source": major_arr,
            "minor_source": minor,
        }


# ---------------------------------------------------------------------------
# ScaleDiscrete
# ---------------------------------------------------------------------------

class ScaleDiscrete(Scale):
    """Scale for discrete / categorical data.

    Attributes
    ----------
    drop : bool
        Whether to drop unused factor levels.
    na_translate : bool
        Whether to include NA in the scale.
    """

    drop: bool = True
    na_value: Any = np.nan
    na_translate: bool = True
    n_breaks_cache: Optional[int] = None
    palette_cache: Optional[Any] = None

    def is_discrete(self) -> bool:
        return True

    def train(self, x: Any) -> None:
        """Train the discrete range on *x*.

        Parameters
        ----------
        x : array-like
            Discrete data values.
        """
        if isinstance(x, pd.Series):
            x_arr = x.values
        else:
            x_arr = np.asarray(x)
        if len(x_arr) == 0:
            return
        self.range.train(x_arr, drop=self.drop)

    def transform(self, x: Any) -> Any:
        """Identity transform for discrete scales."""
        return x

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        """Map discrete values to palette values.

        Parameters
        ----------
        x : array-like
            Discrete data values.
        limits : array-like, optional
            Scale limits.

        Returns
        -------
        numpy.ndarray
        """
        if limits is None:
            limits = self.get_limits()
        if limits is None or len(limits) == 0:
            return np.full(len(np.asarray(x)), self.na_value)

        limits = [l for l in limits if l is not None and not (isinstance(l, float) and np.isnan(l))]
        n = len(limits)
        if n < 1:
            return np.full(len(np.asarray(x)), self.na_value)

        if self.n_breaks_cache is not None and self.n_breaks_cache == n:
            pal = self.palette_cache
        else:
            pal = self.palette(n)
            self.palette_cache = pal
            self.n_breaks_cache = n

        x_str = [str(v) for v in np.asarray(x)]
        limits_str = [str(l) for l in limits]

        if isinstance(pal, dict):
            pal_list = list(pal.values())
        elif isinstance(pal, np.ndarray):
            pal_list = list(pal)
        else:
            pal_list = list(pal) if hasattr(pal, "__iter__") else [pal]

        na_val = self.na_value if self.na_translate else np.nan

        result = []
        for v in x_str:
            if v in limits_str:
                idx = limits_str.index(v)
                if idx < len(pal_list):
                    result.append(pal_list[idx])
                else:
                    result.append(na_val)
            else:
                result.append(na_val)
        return np.array(result)

    def rescale(
        self,
        x: Any,
        limits: Optional[Any] = None,
        range: Optional[Any] = None,
    ) -> np.ndarray:
        """Rescale discrete values."""
        if limits is None:
            limits = self.get_limits()
        if range is None:
            range = (1, len(limits))
        x_arr = np.asarray(x)
        limits_str = [str(l) for l in limits]
        matched = np.array([limits_str.index(str(v)) + 1 if str(v) in limits_str else np.nan for v in x_arr])
        return rescale(matched, from_range=range)

    def dimension(
        self,
        expand: Optional[np.ndarray] = None,
        limits: Optional[Any] = None,
    ) -> np.ndarray:
        if expand is None:
            # R default for discrete position scales: expansion(add = 0.6)
            expand = expansion(0, 0.6)
        if limits is None:
            limits = self.get_limits()
        n = len(limits) if limits is not None else 0
        if n == 0:
            return np.array([0.0, 1.0])
        return expand_range4(np.array([1.0, float(n)]), expand)

    def get_breaks(self, limits: Optional[Any] = None) -> Optional[Any]:
        if self.is_empty():
            return np.array([])
        if limits is None:
            limits = self.get_limits()
        breaks = self.breaks
        if breaks is None:
            return None
        if is_waiver(breaks):
            return limits
        if callable(breaks):
            return breaks(limits)
        # Filter breaks to those in limits
        if limits is not None:
            limits_str = set(str(l) for l in limits)
            return [b for b in breaks if str(b) in limits_str]
        return breaks

    def get_breaks_minor(
        self,
        n: int = 2,
        b: Optional[Any] = None,
        limits: Optional[Any] = None,
    ) -> Optional[Any]:
        minor = self.minor_breaks if hasattr(self, "minor_breaks") else waiver()
        if is_waiver(minor) or minor is None:
            return None
        if callable(minor):
            if limits is None:
                limits = self.get_limits()
            return minor(limits)
        return minor

    def get_labels(self, breaks: Optional[Any] = None) -> Optional[Any]:
        if self.is_empty():
            return []
        if breaks is None:
            breaks = self.get_breaks()
        if breaks is None:
            return None
        labels = self.labels
        if labels is None:
            return None
        if is_waiver(labels):
            return [str(b) for b in breaks]
        if callable(labels):
            return list(labels(breaks))
        return list(labels)

    def clone(self) -> "ScaleDiscrete":
        new = copy.copy(self)
        new.range = DiscreteRange()
        return new

    def break_info(self, range: Optional[Any] = None) -> Dict[str, Any]:
        limits = self.get_limits()
        major = self.get_breaks(limits)
        if major is None:
            return {
                "range": range,
                "labels": None,
                "major": None,
                "minor": None,
                "major_source": None,
                "minor_source": None,
            }
        labels = self.get_labels(major)
        major_mapped = self.map(major)
        major_mapped = major_mapped[~np.isnan(major_mapped.astype(float))]
        major_n = rescale(major_mapped, from_range=range) if range is not None else None
        return {
            "range": range,
            "labels": labels,
            "major": major_n,
            "minor": None,
            "major_source": major_mapped,
            "minor_source": None,
        }


# ---------------------------------------------------------------------------
# ScaleBinned
# ---------------------------------------------------------------------------

class ScaleBinned(Scale):
    """Scale for binned continuous data.

    Attributes
    ----------
    n_breaks : int or None
        Desired number of bins.
    nice_breaks : bool
        Whether to use nicely-spaced breaks.
    right : bool
        Whether bins are closed on the right.
    show_limits : bool
        Whether to show scale limits as ticks.
    """

    na_value: Any = np.nan
    rescaler: Callable = staticmethod(rescale)
    oob: Callable = staticmethod(squish)
    n_breaks: Optional[int] = None
    nice_breaks: bool = True
    right: bool = True
    after_stat: bool = False
    show_limits: bool = False
    trans: Any = None
    palette_cache: Optional[Any] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.trans is None:
            cls.trans = transform_identity()

    def is_discrete(self) -> bool:
        return False

    def train(self, x: Any) -> None:
        x_arr = np.asarray(x, dtype=float)
        if len(x_arr) == 0:
            return
        if not np.issubdtype(x_arr.dtype, np.number):
            cli_abort("Binned scales only support continuous data.")
        self.range.train(x_arr)

    def transform(self, x: Any) -> Any:
        return _default_transform(self, x)

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        if limits is None:
            limits = self.get_limits()
        limits = np.asarray(limits, dtype=float)

        if self.after_stat:
            return x

        breaks = self.get_breaks(limits)
        if breaks is None:
            breaks = np.array([])
        all_breaks = np.unique(np.sort(np.concatenate([limits[:1], np.asarray(breaks), limits[1:]])))

        x_arr = np.asarray(self.oob(np.asarray(x, dtype=float), range=limits), dtype=float)
        x_arr = np.where(~np.isnan(x_arr), x_arr, self.na_value)

        # Rescale breaks
        breaks_resc = self.rescale(all_breaks, limits)
        if len(breaks_resc) > 1:
            # Mirror R's cut(x, breaks, include.lowest=TRUE, right=self$right, labels=FALSE):
            # right-closed intervals (a, b] with the first interval also including its
            # left endpoint (or mirrored for right=FALSE).  ``np.digitize`` cannot
            # express the ``include.lowest`` rule, so delegate to ``pd.cut`` which
            # matches R's semantics exactly.  Codes are 0-based (or -1 for NA);
            # shift to 1-based so downstream palette indexing with ``bins - 1``
            # remains correct.
            x_resc = np.asarray(self.rescale(x_arr, limits), dtype=float)
            unique_breaks = np.unique(breaks_resc)
            codes = pd.cut(
                x_resc,
                unique_breaks,
                labels=False,
                include_lowest=True,
                right=self.right,
            )
            codes_arr = np.asarray(codes, dtype=float)
            # Values outside the break range → NaN; assign bin 1 temporarily so
            # palette lookup stays in-bounds, then mask via the final NaN check.
            out_of_range = np.isnan(codes_arr)
            bins = np.where(out_of_range, 0, codes_arr).astype(int) + 1
            midpoints = breaks_resc[:-1] + np.diff(breaks_resc) / 2.0

            if self.palette_cache is not None:
                pal = self.palette_cache
            else:
                pal = self.palette(midpoints)
                self.palette_cache = pal

            if isinstance(pal, np.ndarray):
                scaled = pal[np.clip(bins - 1, 0, len(pal) - 1)]
            else:
                pal_arr = np.asarray(pal)
                scaled = pal_arr[np.clip(bins - 1, 0, len(pal_arr) - 1)]
            # np.isnan doesn't work on object arrays (e.g. colour strings)
            if scaled.dtype.kind in ("U", "S", "O"):
                na_mask = np.array([v is None or (isinstance(v, float) and np.isnan(v))
                                    for v in scaled]) | out_of_range
                scaled = scaled.astype(object)
                scaled[na_mask] = self.na_value
                return scaled
            scaled = np.where(out_of_range, np.nan, scaled)
            return np.where(~np.isnan(scaled), scaled, self.na_value)
        else:
            return np.full_like(x_arr, self.na_value)

    def rescale(
        self,
        x: Any,
        limits: Optional[Any] = None,
        range: Optional[Any] = None,
    ) -> np.ndarray:
        if limits is None:
            limits = self.get_limits()
        if range is None:
            range = limits
        return self.rescaler(x, from_range=range)

    def dimension(
        self,
        expand: Optional[np.ndarray] = None,
        limits: Optional[Any] = None,
    ) -> np.ndarray:
        if expand is None:
            expand = np.array([0.0, 0.0, 0.0, 0.0])
        if limits is None:
            limits = self.get_limits()
        return expand_range4(np.asarray(limits), expand)

    def get_limits(self) -> np.ndarray:
        # Delegate to continuous logic
        if self.is_empty():
            return np.array([0.0, 1.0])
        if self.limits is None:
            return np.asarray(self.range.range)
        if callable(self.limits):
            transformation = self.get_transformation()
            inv = transformation.inverse(np.asarray(self.range.range))
            user_limits = self.limits(inv)
            return np.asarray(transformation.transform(user_limits))
        limits = np.asarray(self.limits, dtype=float)
        r = np.asarray(self.range.range, dtype=float) if self.range.range is not None else limits
        return np.where(np.isnan(limits), r, limits)

    def get_breaks(self, limits: Optional[Any] = None) -> Optional[np.ndarray]:
        if self.is_empty():
            return np.array([])
        if limits is None:
            limits = self.get_limits()

        transformation = self.get_transformation()
        inv_limits = transformation.inverse(np.asarray(limits, dtype=float))
        inv_limits_sorted = np.sort(inv_limits)

        breaks = self.breaks
        if breaks is None:
            return None
        if is_waiver(breaks):
            if self.nice_breaks:
                n = self.n_breaks or 5
                try:
                    result = transformation.breaks_func(inv_limits_sorted, n=n)
                except TypeError:
                    result = transformation.breaks_func(inv_limits_sorted)
            else:
                n = self.n_breaks or 5
                result = np.linspace(inv_limits_sorted[0], inv_limits_sorted[1], n + 2)[1:-1]
            breaks_val = np.asarray(result, dtype=float)
            # Discard out of range
            breaks_val = breaks_val[(breaks_val >= inv_limits_sorted[0]) & (breaks_val <= inv_limits_sorted[1])]
        elif callable(breaks):
            n = self.n_breaks or 5
            try:
                breaks_val = np.asarray(breaks(inv_limits_sorted, n=n), dtype=float)
            except TypeError:
                breaks_val = np.asarray(breaks(inv_limits_sorted), dtype=float)
        else:
            breaks_val = np.asarray(breaks, dtype=float)

        return np.asarray(transformation.transform(breaks_val), dtype=float)

    def get_breaks_minor(self, **kwargs: Any) -> None:
        return None

    def get_labels(self, breaks: Optional[Any] = None) -> Optional[Any]:
        if breaks is None:
            breaks = self.get_breaks()
        if breaks is None:
            return None
        transformation = self.get_transformation()
        breaks_data = transformation.inverse(np.asarray(breaks, dtype=float))
        labels = self.labels
        if labels is None:
            return None
        if is_waiver(labels):
            return list(transformation.format_func(breaks_data))
        if callable(labels):
            return list(labels(breaks_data))
        return list(labels)

    def clone(self) -> "ScaleBinned":
        new = copy.copy(self)
        new.range = ContinuousRange()
        return new

    def break_info(self, range: Optional[Any] = None) -> Dict[str, Any]:
        if range is None:
            range = self.dimension()
        range = np.asarray(range, dtype=float)
        major = self.get_breaks(range)
        labels = self.get_labels(major)
        return {
            "range": range,
            "labels": labels,
            "major": major,
            "minor": None,
            "major_source": major,
            "minor_source": None,
        }


# ---------------------------------------------------------------------------
# Position sub-classes
# ---------------------------------------------------------------------------

class ScaleContinuousPosition(ScaleContinuous):
    """Continuous scale for position aesthetics (x/y)."""

    secondary_axis: Any = None  # waiver or AxisSecondary

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.secondary_axis is None:
            cls.secondary_axis = waiver()

    def map(self, x: Any, limits: Optional[Any] = None) -> np.ndarray:
        """Map position values (oob only, no palette).

        Parameters
        ----------
        x : array-like
            Values in transformed space.
        limits : array-like, optional
            Scale limits.

        Returns
        -------
        numpy.ndarray
        """
        if limits is None:
            limits = self.get_limits()
        x_arr = np.asarray(x, dtype=float)
        scaled = np.asarray(self.oob(x_arr, range=limits), dtype=float)
        nan_mask = np.isnan(scaled)
        if np.any(nan_mask):
            scaled[nan_mask] = self.na_value
        return scaled

    def break_info(self, range: Optional[Any] = None) -> Dict[str, Any]:
        info = super().break_info(range)
        sec = getattr(self, "secondary_axis", None)
        if sec is not None and not is_waiver(sec) and not sec.empty():
            sec.init(self)
            sec_info = sec.break_info(info["range"], self)
            info.update(sec_info)
        return info

    def sec_name(self) -> Any:
        sec = getattr(self, "secondary_axis", None)
        if sec is None or is_waiver(sec):
            return waiver()
        return sec.name

    def make_sec_title(self, *args: Any, **kwargs: Any) -> Any:
        sec = getattr(self, "secondary_axis", None)
        if sec is not None and not is_waiver(sec):
            return sec.make_title(*args, **kwargs)
        return super().make_sec_title(*args, **kwargs)


class ScaleDiscretePosition(ScaleDiscrete):
    """Discrete scale for position aesthetics (x/y)."""

    secondary_axis: Any = None
    continuous_limits: Any = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.secondary_axis is None:
            cls.secondary_axis = waiver()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not hasattr(self, "range_c") or self.range_c is None:
            self.range_c = ContinuousRange()

    def train(self, x: Any) -> None:
        if _is_discrete(x):
            super().train(x)
        else:
            self.range_c.train(np.asarray(x, dtype=float))

    def get_limits(self) -> Any:
        if self.is_empty():
            return np.array([0.0, 1.0])
        if callable(self.limits):
            return self.limits(self.range.range)
        return self.limits if self.limits is not None else (
            self.range.range if self.range.range is not None else []
        )

    def is_empty(self) -> bool:
        r = self.range.range
        return (
            r is None
            and (self.limits is None or callable(self.limits))
            and (not hasattr(self, "range_c") or self.range_c is None or self.range_c.range is None)
        )

    def reset(self) -> None:
        if hasattr(self, "range_c") and self.range_c is not None:
            self.range_c.reset()

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        if limits is None:
            limits = self.get_limits()
        if _is_discrete(x):
            if limits is None or len(limits) == 0:
                return np.array([])
            values = self.palette(len(limits))
            if not isinstance(values, (np.ndarray, list)):
                cli_abort("The palette function must return a numeric vector.")
            values = np.asarray(values)
            x_str = [str(v) for v in np.asarray(x)]
            limits_str = [str(l) for l in limits]
            mapped = np.array([
                values[limits_str.index(v)] if v in limits_str else np.nan
                for v in x_str
            ], dtype=float)
            return mapped_discrete(mapped)
        return mapped_discrete(np.asarray(x, dtype=float))

    def dimension(
        self,
        expand: Optional[np.ndarray] = None,
        limits: Optional[Any] = None,
    ) -> np.ndarray:
        if expand is None:
            # R default for discrete position scales: expansion(add = 0.6)
            expand = expansion(0, 0.6)
        if limits is None:
            limits = self.get_limits()
        mapped = self.map(limits)
        if mapped is None or len(mapped) == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(np.nanmin(mapped)), float(np.nanmax(mapped))
        return expand_range4(np.array([lo, hi]), expand)

    def clone(self) -> "ScaleDiscretePosition":
        new = copy.copy(self)
        new.range = DiscreteRange()
        new.range_c = ContinuousRange()
        return new

    def sec_name(self) -> Any:
        sec = getattr(self, "secondary_axis", None)
        if sec is None or is_waiver(sec):
            return waiver()
        return sec.name


class ScaleBinnedPosition(ScaleBinned):
    """Binned scale for position aesthetics (x/y)."""

    after_stat: bool = False

    def train(self, x: Any) -> None:
        x_arr = np.asarray(x, dtype=float)
        if not np.issubdtype(x_arr.dtype, np.number):
            cli_abort("Binned scales only support continuous data.")
        if len(x_arr) == 0 or self.after_stat:
            return
        self.range.train(x_arr)

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        if limits is None:
            limits = self.get_limits()
        limits = np.asarray(limits, dtype=float)
        x_arr = np.asarray(x, dtype=float)

        breaks = self.get_breaks(limits)
        if breaks is None:
            breaks = np.array([])
        all_breaks = np.unique(np.sort(np.concatenate([limits[:1], np.asarray(breaks), limits[1:]])))

        x_oob = np.asarray(self.oob(x_arr, range=limits), dtype=float)
        x_oob = np.where(~np.isnan(x_oob), x_oob, self.na_value)
        # Mirror R's cut(x, all_breaks, include.lowest=TRUE, right=self$right, labels=FALSE)
        # semantics.  pd.cut returns 0-based integer codes (or NaN for out-of-range);
        # shift to 1-based integers to match R's cut().
        unique_breaks = np.unique(all_breaks)
        if len(unique_breaks) <= 1:
            return np.full_like(x_oob, self.na_value)
        codes = pd.cut(
            x_oob,
            unique_breaks,
            labels=False,
            include_lowest=True,
            right=self.right,
        )
        codes_arr = np.asarray(codes, dtype=float)
        # Out-of-range inputs are NaN; R clamps these via oob already, but preserve
        # NaN if oob did not catch them (e.g. with custom oob functions).
        bins = np.where(np.isnan(codes_arr), np.nan, codes_arr + 1.0)
        return bins

    def reset(self) -> None:
        self.after_stat = True
        limits = self.get_limits()
        breaks = self.get_breaks(limits)
        self.range.reset()
        combined = np.concatenate([np.asarray(limits), np.asarray(breaks) if breaks is not None else np.array([])])
        self.range.train(combined)

    def get_breaks(self, limits: Optional[Any] = None) -> Optional[np.ndarray]:
        breaks = super().get_breaks(limits)
        if self.show_limits and breaks is not None:
            lims = self.get_limits()
            breaks = np.sort(np.unique(np.concatenate([lims, np.asarray(breaks)])))
        return breaks


# ---------------------------------------------------------------------------
# Identity sub-classes
# ---------------------------------------------------------------------------

class ScaleContinuousIdentity(ScaleContinuous):
    """Continuous identity scale -- data values are used as-is."""

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        x_arr = np.asarray(x)
        if isinstance(x, pd.Categorical):
            return np.asarray(x.astype(str))
        return x_arr

    def train(self, x: Any) -> None:
        if self.guide == "none":
            return
        super().train(x)


class ScaleDiscreteIdentity(ScaleDiscrete):
    """Discrete identity scale -- data values are used as-is."""

    def map(self, x: Any, limits: Optional[Any] = None) -> Any:
        x_arr = np.asarray(x)
        if isinstance(x, pd.Categorical):
            return np.asarray(x.astype(str))
        return x_arr

    def train(self, x: Any) -> None:
        if self.guide == "none":
            return
        super().train(x)


# ---------------------------------------------------------------------------
# Date/Datetime sub-classes
# ---------------------------------------------------------------------------

class ScaleContinuousDate(ScaleContinuous):
    """Continuous scale for date-valued data."""
    pass


class ScaleContinuousDatetime(ScaleContinuous):
    """Continuous scale for datetime-valued data."""
    pass


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------

def continuous_scale(
    aesthetics: Union[str, List[str]],
    palette: Optional[Callable] = None,
    *,
    name: Any = None,
    breaks: Any = None,
    minor_breaks: Any = None,
    n_breaks: Optional[int] = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    rescaler: Optional[Callable] = None,
    oob: Optional[Callable] = None,
    expand: Any = None,
    na_value: Any = np.nan,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: Any = "legend",
    position: str = "left",
    fallback_palette: Optional[Callable] = None,
    super_class: Optional[Type[ScaleContinuous]] = None,
) -> ScaleContinuous:
    """Construct a continuous scale.

    Parameters
    ----------
    aesthetics : str or list of str
        Aesthetic names this scale applies to.
    palette : callable, optional
        Palette function mapping [0,1] to aesthetic values.
    name : str or Waiver, optional
        Scale title.
    breaks : array-like, callable, or None
        Break specification.
    minor_breaks : array-like, callable, or None
        Minor break specification.
    n_breaks : int, optional
        Desired number of breaks.
    labels : array-like, callable, or None
        Label specification.
    limits : array-like, callable, or None
        Scale limits.
    rescaler : callable, optional
        Rescaling function (default ``rescale``).
    oob : callable, optional
        Out-of-bounds handler (default ``censor``).
    expand : array-like or Waiver, optional
        Expansion.
    na_value : any
        Value to use for missing data.
    transform : str or Transform
        Transformation name or object.
    trans : str or Transform, optional
        Deprecated alias for *transform*.
    guide : str
        Guide type.
    position : str
        Axis position.
    fallback_palette : callable, optional
        Palette to use when *palette* is None and theme provides none.
    super_class : type, optional
        Scale class to instantiate (default ``ScaleContinuous``).

    Returns
    -------
    ScaleContinuous
    """
    if name is None:
        name = waiver()
    if breaks is None:
        breaks = waiver()
    if minor_breaks is None:
        minor_breaks = waiver()
    if labels is None:
        labels = waiver()
    if expand is None:
        expand = waiver()

    if trans is not None:
        deprecate_warn("3.5.0", "continuous_scale(trans=)", with_="continuous_scale(transform=)")
        transform = trans

    if isinstance(aesthetics, str):
        aesthetics = [aesthetics]
    aesthetics = standardise_aes_names(aesthetics)

    _check_breaks_labels(breaks, labels)

    if position not in ("left", "right", "top", "bottom"):
        cli_abort(f"position must be one of 'left', 'right', 'top', 'bottom', got '{position}'.")

    # If non-positional scale with breaks=None, remove guide
    if breaks is None and not _is_position_aes(aesthetics):
        guide = "none"

    if isinstance(transform, str):
        transform = as_transform(transform)

    # Transform limits if provided
    if limits is not None and not callable(limits):
        limits_arr = np.asarray(limits, dtype=float)
        limits_arr = transform.transform(limits_arr)
        if not np.any(np.isnan(limits_arr)):
            limits_arr = np.sort(limits_arr)
        limits = limits_arr

    if super_class is None:
        super_class = ScaleContinuous

    sc = super_class()
    sc.aesthetics = list(aesthetics)
    sc.palette = palette
    sc.fallback_palette = fallback_palette
    sc.range = ContinuousRange()
    sc.limits = limits
    sc.trans = transform
    sc.na_value = na_value
    sc.expand = expand
    sc.rescaler = rescaler if rescaler is not None else rescale
    sc.oob = oob if oob is not None else censor
    sc.name = name
    sc.breaks = breaks
    sc.minor_breaks = minor_breaks
    sc.n_breaks = n_breaks
    sc.labels = labels
    sc.guide = guide
    sc.position = position
    return sc


def discrete_scale(
    aesthetics: Union[str, List[str]],
    palette: Optional[Callable] = None,
    *,
    name: Any = None,
    breaks: Any = None,
    minor_breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    na_translate: bool = True,
    na_value: Any = np.nan,
    drop: bool = True,
    guide: Any = "legend",
    position: str = "left",
    fallback_palette: Optional[Callable] = None,
    super_class: Optional[Type[ScaleDiscrete]] = None,
) -> ScaleDiscrete:
    """Construct a discrete scale.

    Parameters
    ----------
    aesthetics : str or list of str
        Aesthetic names this scale applies to.
    palette : callable, optional
        Palette function taking an integer and returning *n* values.
    name : str or Waiver, optional
        Scale title.
    breaks : array-like, callable, or None
        Break specification.
    minor_breaks : array-like, callable, or None
        Minor break specification.
    labels : array-like, callable, or None
        Label specification.
    limits : array-like, callable, or None
        Scale limits.
    expand : array-like or Waiver, optional
        Expansion.
    na_translate : bool
        Whether to translate NAs.
    na_value : any
        Value for missing data.
    drop : bool
        Whether to drop unused levels.
    guide : str
        Guide type.
    position : str
        Axis position.
    fallback_palette : callable, optional
        Fallback palette.
    super_class : type, optional
        Scale class to instantiate (default ``ScaleDiscrete``).

    Returns
    -------
    ScaleDiscrete
    """
    if name is None:
        name = waiver()
    if breaks is None:
        breaks = waiver()
    if minor_breaks is None:
        minor_breaks = waiver()
    if labels is None:
        labels = waiver()
    if expand is None:
        expand = waiver()

    if isinstance(aesthetics, str):
        aesthetics = [aesthetics]
    aesthetics = standardise_aes_names(aesthetics)

    _check_breaks_labels(breaks, labels)

    if position not in ("left", "right", "top", "bottom"):
        cli_abort(f"position must be one of 'left', 'right', 'top', 'bottom', got '{position}'.")

    # If non-positional scale with breaks=None, remove guide
    if breaks is None and not _is_position_aes(aesthetics):
        guide = "none"

    if super_class is None:
        super_class = ScaleDiscrete

    sc = super_class()
    sc.aesthetics = list(aesthetics)
    sc.palette = palette
    sc.fallback_palette = fallback_palette
    sc.range = DiscreteRange()
    sc.limits = limits
    sc.na_value = na_value
    sc.na_translate = na_translate
    sc.expand = expand
    sc.name = name
    sc.breaks = breaks
    sc.minor_breaks = minor_breaks
    sc.labels = labels
    sc.drop = drop
    sc.guide = guide
    sc.position = position
    return sc


def binned_scale(
    aesthetics: Union[str, List[str]],
    palette: Optional[Callable] = None,
    *,
    name: Any = None,
    breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    rescaler: Optional[Callable] = None,
    oob: Optional[Callable] = None,
    expand: Any = None,
    na_value: Any = np.nan,
    n_breaks: Optional[int] = None,
    nice_breaks: bool = True,
    right: bool = True,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    show_limits: bool = False,
    guide: Any = "bins",
    position: str = "left",
    fallback_palette: Optional[Callable] = None,
    super_class: Optional[Type[ScaleBinned]] = None,
) -> ScaleBinned:
    """Construct a binned scale.

    Parameters
    ----------
    aesthetics : str or list of str
        Aesthetic names this scale applies to.
    palette : callable, optional
        Palette function.
    name : str or Waiver, optional
        Scale title.
    breaks : array-like, callable, or None
        Break specification.
    labels : array-like, callable, or None
        Label specification.
    limits : array-like, callable, or None
        Scale limits.
    rescaler : callable, optional
        Rescaling function.
    oob : callable, optional
        Out-of-bounds handler (default ``squish``).
    expand : array-like or Waiver, optional
        Expansion.
    na_value : any
        Value for missing data.
    n_breaks : int, optional
        Desired number of breaks.
    nice_breaks : bool
        Use nicely-spaced breaks.
    right : bool
        Bins closed on the right.
    transform : str or Transform
        Transformation.
    trans : str or Transform, optional
        Deprecated alias for *transform*.
    show_limits : bool
        Show limits as ticks.
    guide : str
        Guide type.
    position : str
        Axis position.
    fallback_palette : callable, optional
        Fallback palette.
    super_class : type, optional
        Scale class to instantiate (default ``ScaleBinned``).

    Returns
    -------
    ScaleBinned
    """
    if name is None:
        name = waiver()
    if breaks is None:
        breaks = waiver()
    if labels is None:
        labels = waiver()
    if expand is None:
        expand = waiver()

    if trans is not None:
        deprecate_warn("3.5.0", "binned_scale(trans=)", with_="binned_scale(transform=)")
        transform = trans

    if isinstance(aesthetics, str):
        aesthetics = [aesthetics]
    aesthetics = standardise_aes_names(aesthetics)

    _check_breaks_labels(breaks, labels)

    if position not in ("left", "right", "top", "bottom"):
        cli_abort(f"position must be one of 'left', 'right', 'top', 'bottom', got '{position}'.")

    if breaks is None and not _is_position_aes(aesthetics) and guide != "none":
        guide = "none"

    if isinstance(transform, str):
        transform = as_transform(transform)

    if limits is not None and not callable(limits):
        limits_arr = np.asarray(limits, dtype=float)
        limits_arr = transform.transform(limits_arr)
        if not np.any(np.isnan(limits_arr)):
            limits_arr = np.sort(limits_arr)
        limits = limits_arr

    if super_class is None:
        super_class = ScaleBinned

    sc = super_class()
    sc.aesthetics = list(aesthetics)
    sc.palette = palette
    sc.fallback_palette = fallback_palette
    sc.range = ContinuousRange()
    sc.limits = limits
    sc.trans = transform
    sc.na_value = na_value
    sc.expand = expand
    sc.rescaler = rescaler if rescaler is not None else rescale
    sc.oob = oob if oob is not None else squish
    sc.n_breaks = n_breaks
    sc.nice_breaks = nice_breaks
    sc.right = right
    sc.show_limits = show_limits
    sc.name = name
    sc.breaks = breaks
    sc.labels = labels
    sc.guide = guide
    sc.position = position
    return sc


# ---------------------------------------------------------------------------
# ScalesList container
# ---------------------------------------------------------------------------

class ScalesList:
    """Container for a plot's collection of scales.

    Attributes
    ----------
    scales : list of Scale
        The individual scales.
    """

    def __init__(self) -> None:
        self.scales: List[Scale] = []

    def find(self, aesthetic: str) -> List[bool]:
        """Return a boolean mask of scales matching *aesthetic*.

        Parameters
        ----------
        aesthetic : str
            Aesthetic name.

        Returns
        -------
        list of bool
        """
        return [any(aesthetic in s.aesthetics for aesthetic in [aesthetic]) for s in self.scales]

    def has_scale(self, aesthetic: str) -> bool:
        """Check whether a scale exists for *aesthetic*.

        Parameters
        ----------
        aesthetic : str
            Aesthetic name.

        Returns
        -------
        bool
        """
        return any(aesthetic in s.aesthetics for s in self.scales)

    def add(self, scale: Optional[Scale]) -> None:
        """Add a scale, replacing any existing scale for the same aesthetics.

        Parameters
        ----------
        scale : Scale or None
            Scale to add.  ``None`` is silently ignored.
        """
        if scale is None:
            return
        prev_aes = [any(a in s.aesthetics for a in scale.aesthetics) for s in self.scales]
        if any(prev_aes):
            first_name = next(
                s.aesthetics[0] for s, p in zip(self.scales, prev_aes) if p
            )
            cli_inform(
                f"Scale for {first_name} is already present. "
                f"Adding another scale for {first_name}, which will replace the existing scale."
            )
        self.scales = [s for s, p in zip(self.scales, prev_aes) if not p]
        self.scales.append(scale)

    def n(self) -> int:
        """Return the number of scales."""
        return len(self.scales)

    def input(self) -> List[str]:
        """Return all aesthetic names across all scales."""
        result: List[str] = []
        for s in self.scales:
            result.extend(s.aesthetics)
        return result

    def clone(self) -> "ScalesList":
        """Clone the scales list and all its scales."""
        new = ScalesList()
        new.scales = [s.clone() for s in self.scales]
        return new

    def non_position_scales(self) -> "ScalesList":
        """Return a new ScalesList with only non-position scales."""
        new = ScalesList()
        new.scales = [
            s for s in self.scales
            if not any(a in _POSITION_AESTHETICS for a in s.aesthetics)
        ]
        return new

    def get_scales(self, output: str) -> Optional[Scale]:
        """Get the scale for a given aesthetic.

        Parameters
        ----------
        output : str
            Aesthetic name.

        Returns
        -------
        Scale or None
        """
        for s in self.scales:
            if output in s.aesthetics:
                return s
        return None

    def train_df(self, df: pd.DataFrame) -> None:
        """Train all scales on *df*.

        Parameters
        ----------
        df : pandas.DataFrame
            Layer data.
        """
        if _empty(df) or len(self.scales) == 0:
            return
        for s in self.scales:
            s.train_df(df)

    def map_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map all scales on *df*.

        Parameters
        ----------
        df : pandas.DataFrame
            Layer data.

        Returns
        -------
        pandas.DataFrame
            Data with mapped columns.
        """
        if _empty(df) or len(self.scales) == 0:
            return df
        for s in self.scales:
            mapped = s.map_df(df)
            for k, v in mapped.items():
                df[k] = v
        return df

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform all scale columns in *df*.

        Parameters
        ----------
        df : pandas.DataFrame
            Layer data.

        Returns
        -------
        pandas.DataFrame
            Data with transformed columns.
        """
        if _empty(df):
            return df
        for s in self.scales:
            transformed = s.transform_df(df)
            for k, v in transformed.items():
                df[k] = v
        return df

    def add_defaults(self, data: pd.DataFrame, env: Optional[Any] = None) -> None:
        """Add default scales for aesthetics in *data* not yet covered.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data.
        env : any, optional
            Lookup environment (unused in Python port).
        """
        existing = set(self.input())
        for aes_name in data.columns:
            if aes_name not in existing:
                sc = find_scale(aes_name, data[aes_name])
                if sc is not None:
                    self.add(sc)

    def add_missing(self, aesthetics: List[str], env: Optional[Any] = None) -> None:
        """Add missing but required scales.

        Parameters
        ----------
        aesthetics : list of str
            Required aesthetic names (typically ``['x', 'y']``).
        env : any, optional
            Lookup environment (unused).
        """
        existing = set(self.input())
        for aes_name in aesthetics:
            if aes_name not in existing:
                sc = _default_continuous_scale(aes_name)
                if sc is not None:
                    self.add(sc)


def scales_list() -> ScalesList:
    """Create a new empty :class:`ScalesList`.

    Returns
    -------
    ScalesList
    """
    return ScalesList()


def _default_continuous_scale(aes: str) -> Optional[Scale]:
    """Create a default continuous scale for the given aesthetic."""
    if aes in ("x", "xmin", "xmax", "xend", "xintercept"):
        return continuous_scale(
            _X_AESTHETICS,
            palette=lambda x: x,
            position="bottom",
            super_class=ScaleContinuousPosition,
        )
    if aes in ("y", "ymin", "ymax", "yend", "yintercept"):
        return continuous_scale(
            _Y_AESTHETICS,
            palette=lambda x: x,
            position="left",
            super_class=ScaleContinuousPosition,
        )
    return None


# ---------------------------------------------------------------------------
# Secondary axis support
# ---------------------------------------------------------------------------

class _Derived:
    """Sentinel for inheriting settings from the primary axis."""

    def __repr__(self) -> str:
        return "derive()"


def derive() -> _Derived:
    """Return a ``derive()`` sentinel for secondary axis inheritance.

    Returns
    -------
    _Derived
    """
    return _Derived()


def is_derived(x: Any) -> bool:
    """Check whether *x* is a ``derive()`` sentinel.

    Parameters
    ----------
    x : Any

    Returns
    -------
    bool
    """
    return isinstance(x, _Derived)


class AxisSecondary:
    """Specification for a secondary axis.

    Parameters
    ----------
    trans : callable
        Monotonic transformation from primary to secondary scale.
    name : any
        Axis title.
    breaks : any
        Break specification.
    labels : any
        Label specification.
    guide : any
        Guide specification.
    """

    def __init__(
        self,
        trans: Optional[Callable] = None,
        name: Any = None,
        breaks: Any = None,
        labels: Any = None,
        guide: Any = None,
    ) -> None:
        self.trans = trans
        self.name = name if name is not None else waiver()
        self.breaks = breaks if breaks is not None else waiver()
        self.labels = labels if labels is not None else waiver()
        self.guide = guide if guide is not None else waiver()
        self.detail = 1000

    def empty(self) -> bool:
        """Whether this secondary axis is empty (no transform)."""
        return self.trans is None

    def init(self, scale: Scale) -> None:
        """Inherit settings from the primary scale.

        Parameters
        ----------
        scale : Scale
            The primary scale.
        """
        if self.empty():
            return
        if is_derived(self.name) and not is_waiver(scale.name):
            self.name = scale.name
        if is_derived(self.breaks):
            self.breaks = scale.breaks
        if is_waiver(self.breaks):
            transformation = scale.get_transformation()
            self.breaks = transformation.breaks_func
        if is_derived(self.labels):
            self.labels = scale.labels
        if is_derived(self.guide):
            self.guide = scale.guide

    def transform_range(self, range: np.ndarray) -> np.ndarray:
        """Apply the secondary transform to a range.

        Parameters
        ----------
        range : numpy.ndarray
            Primary range values.

        Returns
        -------
        numpy.ndarray
            Transformed values.
        """
        return np.asarray(self.trans(range))

    def break_info(self, range: np.ndarray, scale: Scale) -> Dict[str, Any]:
        """Compute secondary-axis break information.

        Parameters
        ----------
        range : numpy.ndarray
            Primary axis range.
        scale : Scale
            Primary scale.

        Returns
        -------
        dict
            Keys are prefixed with ``sec.``.
        """
        if self.empty():
            return {}

        transformation = scale.get_transformation()
        along = np.linspace(range[0], range[1], self.detail)
        old_range = transformation.inverse(along)
        full_range = self.transform_range(old_range)

        new_range = np.array([np.nanmin(full_range), np.nanmax(full_range)])

        # Create a temporary scale for break info
        temp_sc = ScaleContinuousPosition()
        temp_sc.name = self.name
        temp_sc.breaks = self.breaks
        temp_sc.labels = self.labels
        temp_sc.limits = new_range
        temp_sc.expand = np.array([0.0, 0.0])
        temp_sc.minor_breaks = None  # no minor breaks for secondary axis
        temp_sc.trans = transformation
        temp_sc.range = ContinuousRange()
        temp_sc.train(new_range)
        range_info = temp_sc.break_info()

        result = {}
        for k, v in range_info.items():
            result[f"sec.{k}"] = v
        return result

    def make_title(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve the secondary axis title."""
        return ScaleContinuous.make_title(None, *args, **kwargs)


def sec_axis(
    transform: Optional[Callable] = None,
    name: Any = None,
    breaks: Any = None,
    labels: Any = None,
    guide: Any = None,
    trans: Optional[Callable] = None,
) -> AxisSecondary:
    """Create a secondary axis specification.

    Parameters
    ----------
    transform : callable, optional
        Monotonic transformation function.
    name : any, optional
        Axis title.
    breaks : any, optional
        Break specification.
    labels : any, optional
        Label specification.
    guide : any, optional
        Guide specification.
    trans : callable, optional
        Deprecated alias for *transform*.

    Returns
    -------
    AxisSecondary
    """
    if trans is not None:
        deprecate_warn("3.5.0", "sec_axis(trans=)", with_="sec_axis(transform=)")
        transform = trans

    if name is None:
        name = waiver()
    if breaks is None:
        breaks = waiver()
    if labels is None:
        labels = waiver()
    if guide is None:
        guide = waiver()

    return AxisSecondary(
        trans=transform,
        name=name,
        breaks=breaks,
        labels=labels,
        guide=guide,
    )


def dup_axis(
    transform: Optional[Callable] = None,
    name: Any = None,
    breaks: Any = None,
    labels: Any = None,
    guide: Any = None,
    trans: Optional[Callable] = None,
) -> AxisSecondary:
    """Create a secondary axis that duplicates the primary.

    Parameters
    ----------
    transform : callable, optional
        Transformation (default: identity).
    name : any, optional
        Axis title (default: derive from primary).
    breaks : any, optional
        Breaks (default: derive from primary).
    labels : any, optional
        Labels (default: derive from primary).
    guide : any, optional
        Guide (default: derive from primary).
    trans : callable, optional
        Deprecated alias for *transform*.

    Returns
    -------
    AxisSecondary
    """
    if transform is None:
        transform = lambda x: x
    if name is None:
        name = derive()
    if breaks is None:
        breaks = derive()
    if labels is None:
        labels = derive()
    if guide is None:
        guide = derive()
    return sec_axis(transform=transform, name=name, breaks=breaks, labels=labels, guide=guide, trans=trans)


def is_sec_axis(x: Any) -> bool:
    """Check whether *x* is an :class:`AxisSecondary`.

    Parameters
    ----------
    x : Any

    Returns
    -------
    bool
    """
    return isinstance(x, AxisSecondary)


def _set_sec_axis(sec_axis_obj: Any, scale: Scale) -> Scale:
    """Attach a secondary axis to a scale (internal helper).

    Parameters
    ----------
    sec_axis_obj : AxisSecondary or Waiver
        Secondary axis specification.
    scale : Scale
        Target scale.

    Returns
    -------
    Scale
        The scale (potentially modified).
    """
    if not is_waiver(sec_axis_obj):
        if scale.is_discrete():
            if sec_axis_obj.trans is not None and sec_axis_obj.trans is not (lambda x: x):
                pass  # discrete axes must have identity transform
        if not is_sec_axis(sec_axis_obj):
            cli_abort("Secondary axes must be specified using sec_axis().")
        scale.secondary_axis = sec_axis_obj
    return scale


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

def is_scale(x: Any) -> bool:
    """Check whether *x* is a Scale instance.

    Parameters
    ----------
    x : Any

    Returns
    -------
    bool
    """
    return isinstance(x, Scale)


def scale_type(x: Any) -> List[str]:
    """Determine the appropriate scale type for data *x*.

    Parameters
    ----------
    x : array-like
        Data values.

    Returns
    -------
    list of str
        Scale type names (e.g. ``['continuous']`` or ``['discrete']``).
    """
    if isinstance(x, pd.Series):
        if isinstance(x.dtype, pd.CategoricalDtype):
            if x.cat.ordered:
                return ["ordinal", "discrete"]
            return ["discrete"]
        if pd.api.types.is_bool_dtype(x.dtype):
            return ["discrete"]
        if pd.api.types.is_datetime64_any_dtype(x.dtype):
            return ["datetime", "continuous"]
        if pd.api.types.is_numeric_dtype(x.dtype):
            return ["continuous"]
        if x.dtype == object:
            return ["discrete"]
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("U", "S", "O", "b"):
            return ["discrete"]
        if np.issubdtype(x.dtype, np.datetime64):
            return ["datetime", "continuous"]
        if np.issubdtype(x.dtype, np.number):
            return ["continuous"]
    return ["continuous"]


#: Aesthetics that R does **not** auto-create a default scale for.
#:
#: R ships no ``scale_stroke_*`` family — ``stroke`` data flows through
#: as raw aesthetic values and is not trained by a scale nor given a
#: legend.  Python has ``scale_stroke_*()`` available for explicit user
#: control, but ``find_scale`` must not auto-instantiate one or a
#: spurious "stroke" guide would appear that R never produces.
_NO_DEFAULT_SCALE_AES: frozenset = frozenset({"stroke"})


def find_scale(aes: str, x: Any, env: Optional[Any] = None) -> Optional[Scale]:
    """Auto-detect an appropriate scale for aesthetic *aes* and data *x*.

    Mirrors R's ``find_scale()`` (ggplot2/R/scales-.R) which only looks
    up scales for aesthetics that R has registered scale constructors
    for.  ``stroke`` is intentionally excluded — see
    :data:`_NO_DEFAULT_SCALE_AES`.

    Parameters
    ----------
    aes : str
        Aesthetic name.
    x : array-like
        Data values.
    env : any, optional
        Lookup environment (unused).

    Returns
    -------
    Scale or None
    """
    if x is None:
        return None
    if aes in _NO_DEFAULT_SCALE_AES:
        return None

    types = scale_type(x)

    for stype in types:
        # Try to import from the scales module
        func_name = f"scale_{aes}_{stype}"
        try:
            from ggplot2_py import scales as scales_mod
            func = getattr(scales_mod, func_name, None)
            if func is not None:
                return func()
        except (ImportError, AttributeError):
            pass

    return None

"""
Scale constructor functions for ggplot2.

This module provides all ``scale_*`` factory functions that users call.
Each creates and returns a :class:`~ggplot2_py.scale.Scale` object configured
for the requested aesthetic, palette, and parameters.

Examples
--------
>>> from ggplot2_py.scales import scale_x_continuous, scale_colour_hue
>>> sc = scale_x_continuous(limits=[0, 10])
>>> sc = scale_colour_hue(l=65, c=100)
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

from scales import (
    ContinuousRange,
    DiscreteRange,
    abs_area,
    as_transform,
    censor,
    muted,
    oob_censor,
    oob_squish,
    pal_area,
    pal_brewer,
    pal_div_gradient,
    pal_gradient_n,
    pal_grey,
    pal_hue,
    pal_identity,
    pal_linetype,
    pal_manual,
    pal_rescale,
    pal_seq_gradient,
    pal_shape,
    pal_viridis,
    rescale,
    rescale_max,
    rescale_mid,
    squish,
    transform_date,
    transform_identity,
    transform_log10,
    transform_reverse,
    transform_sqrt,
    transform_time,
)

from ggplot2_py._compat import (
    Waiver,
    cli_abort,
    cli_warn,
    deprecate_warn,
    is_waiver,
    waiver,
)
from ggplot2_py.scale import (
    AxisSecondary,
    ScaleBinned,
    ScaleBinnedPosition,
    ScaleContinuous,
    ScaleContinuousDate,
    ScaleContinuousDatetime,
    ScaleContinuousIdentity,
    ScaleContinuousPosition,
    ScaleDiscrete,
    ScaleDiscreteIdentity,
    ScaleDiscretePosition,
    _X_AESTHETICS,
    _Y_AESTHETICS,
    _set_sec_axis,
    binned_scale,
    continuous_scale,
    derive,
    discrete_scale,
    dup_axis,
    expansion,
    find_scale,
    is_sec_axis,
    scale_type,
    sec_axis,
)

__all__: List[str] = [
    # Position continuous
    "scale_x_continuous",
    "scale_y_continuous",
    "scale_x_log10",
    "scale_y_log10",
    "scale_x_sqrt",
    "scale_y_sqrt",
    "scale_x_reverse",
    "scale_y_reverse",
    # Position discrete
    "scale_x_discrete",
    "scale_y_discrete",
    # Position binned
    "scale_x_binned",
    "scale_y_binned",
    # Position date/datetime
    "scale_x_date",
    "scale_y_date",
    "scale_x_datetime",
    "scale_y_datetime",
    "scale_x_time",
    "scale_y_time",
    # Colour/fill continuous
    "scale_colour_continuous",
    "scale_fill_continuous",
    "scale_colour_gradient",
    "scale_fill_gradient",
    "scale_colour_gradient2",
    "scale_fill_gradient2",
    "scale_colour_gradientn",
    "scale_fill_gradientn",
    # Colour/fill discrete
    "scale_colour_discrete",
    "scale_fill_discrete",
    "scale_colour_hue",
    "scale_fill_hue",
    "scale_colour_brewer",
    "scale_fill_brewer",
    "scale_colour_grey",
    "scale_fill_grey",
    # Colour/fill viridis
    "scale_colour_viridis_c",
    "scale_fill_viridis_c",
    "scale_colour_viridis_d",
    "scale_fill_viridis_d",
    "scale_colour_viridis_b",
    "scale_fill_viridis_b",
    # Colour/fill distiller/fermenter
    "scale_colour_distiller",
    "scale_fill_distiller",
    "scale_colour_fermenter",
    "scale_fill_fermenter",
    # Colour/fill binned / steps
    "scale_colour_binned",
    "scale_fill_binned",
    "scale_colour_steps",
    "scale_fill_steps",
    "scale_colour_steps2",
    "scale_fill_steps2",
    "scale_colour_stepsn",
    "scale_fill_stepsn",
    # Colour/fill identity / manual
    "scale_colour_identity",
    "scale_fill_identity",
    "scale_colour_manual",
    "scale_fill_manual",
    # Colour/fill date/datetime/ordinal
    "scale_colour_date",
    "scale_fill_date",
    "scale_colour_datetime",
    "scale_fill_datetime",
    "scale_colour_ordinal",
    "scale_fill_ordinal",
    # American spelling aliases
    "scale_color_continuous",
    "scale_color_discrete",
    "scale_color_gradient",
    "scale_color_gradient2",
    "scale_color_gradientn",
    "scale_color_hue",
    "scale_color_brewer",
    "scale_color_distiller",
    "scale_color_fermenter",
    "scale_color_grey",
    "scale_color_viridis_c",
    "scale_color_viridis_d",
    "scale_color_viridis_b",
    "scale_color_binned",
    "scale_color_steps",
    "scale_color_steps2",
    "scale_color_stepsn",
    "scale_color_identity",
    "scale_color_manual",
    "scale_color_date",
    "scale_color_datetime",
    "scale_color_ordinal",
    # Alpha
    "scale_alpha",
    "scale_alpha_continuous",
    "scale_alpha_discrete",
    "scale_alpha_binned",
    "scale_alpha_identity",
    "scale_alpha_manual",
    "scale_alpha_ordinal",
    "scale_alpha_date",
    "scale_alpha_datetime",
    # Size
    "scale_size",
    "scale_size_continuous",
    "scale_size_discrete",
    "scale_size_binned",
    "scale_size_area",
    "scale_size_binned_area",
    "scale_size_identity",
    "scale_size_manual",
    "scale_size_ordinal",
    "scale_size_date",
    "scale_size_datetime",
    "scale_radius",
    # Shape
    "scale_shape",
    "scale_shape_discrete",
    "scale_shape_binned",
    "scale_shape_continuous",
    "scale_shape_identity",
    "scale_shape_manual",
    "scale_shape_ordinal",
    # Linetype
    "scale_linetype",
    "scale_linetype_discrete",
    "scale_linetype_continuous",
    "scale_linetype_binned",
    "scale_linetype_identity",
    "scale_linetype_manual",
    # Linewidth
    "scale_linewidth",
    "scale_linewidth_continuous",
    "scale_linewidth_discrete",
    "scale_linewidth_binned",
    "scale_linewidth_identity",
    "scale_linewidth_manual",
    "scale_linewidth_ordinal",
    "scale_linewidth_date",
    "scale_linewidth_datetime",
    # Generic identity / manual
    "scale_continuous_identity",
    "scale_discrete_identity",
    "scale_discrete_manual",
    # Helpers re-exported
    "scale_type",
    "find_scale",
    "expansion",
    "sec_axis",
    "dup_axis",
    "derive",
    "continuous_scale",
    "discrete_scale",
    "binned_scale",
]


# =========================================================================
# Internal helpers
# =========================================================================

def _identity(x: Any) -> Any:
    """Identity function used as default palette for position scales."""
    return x


def _seq_len(n: int) -> List[int]:
    """Return ``list(range(1, n+1))``."""
    return list(range(1, n + 1))


def _mid_rescaler(
    mid: float = 0,
    transform: Union[str, Any] = "identity",
) -> Callable:
    """Return a rescaler centred on *mid*.

    Parameters
    ----------
    mid : float
        Midpoint in data space.
    transform : str or Transform
        Transformation object.

    Returns
    -------
    callable
    """
    if isinstance(transform, str):
        transform = as_transform(transform)
    trans_mid = float(transform.transform(np.array([mid]))[0])

    def _rescaler(x: Any, _from: Any = None, **kwargs: Any) -> Any:
        if _from is None:
            x_arr = np.asarray(x, dtype=float)
            _from = np.array([np.nanmin(x_arr), np.nanmax(x_arr)])
        return rescale_mid(x, to=np.array([0.0, 1.0]), from_range=_from, mid=trans_mid)

    return _rescaler


def _manual_scale(
    aesthetic: Union[str, List[str]],
    values: Any = None,
    breaks: Any = None,
    *,
    name: Any = None,
    na_value: Any = np.nan,
    **kwargs: Any,
) -> ScaleDiscrete:
    """Internal helper for manual scales.

    Parameters
    ----------
    aesthetic : str or list of str
        Aesthetics.
    values : dict or list
        Manual values.
    breaks : any
        Break specification.
    name : any, optional
        Scale title.
    na_value : any
        Missing-data value.
    **kwargs
        Extra arguments passed to ``discrete_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if name is None:
        name = waiver()
    if breaks is None:
        breaks = waiver()

    limits = kwargs.pop("limits", None)

    if values is None:
        values = []

    if isinstance(values, dict):
        if limits is None:
            _values_dict = values

            def _limits_func(x: Any) -> Any:
                keys = list(_values_dict.keys())
                x_list = list(x) if not isinstance(x, list) else x
                shared = [v for v in x_list if v in keys or str(v) in keys]
                if not shared:
                    shared = [v for v in x_list if v is not None and not (isinstance(v, float) and np.isnan(v))]
                return shared if shared else x_list

            limits = _limits_func

    # Order values by breaks if values is a plain list
    if (
        isinstance(values, (list, np.ndarray))
        and not isinstance(values, dict)
        and not is_waiver(breaks)
        and breaks is not None
        and not callable(breaks)
    ):
        values_list = list(values)
        breaks_list = list(breaks)
        named_values = {}
        for i, b in enumerate(breaks_list):
            if i < len(values_list):
                named_values[b] = values_list[i]
        values = named_values

    if isinstance(values, dict):
        _vals = values

        def pal(n: int) -> list:
            return list(_vals.values())[:n] if len(_vals) >= n else list(_vals.values())
    else:
        _vals_list = list(values)

        def pal(n: int) -> list:
            if n > len(_vals_list):
                cli_abort(
                    f"Insufficient values in manual scale. {n} needed but only {len(_vals_list)} provided."
                )
            return _vals_list[:n]

    return discrete_scale(
        aesthetic,
        palette=pal,
        name=name,
        breaks=breaks,
        limits=limits,
        na_value=na_value,
        **kwargs,
    )


# =========================================================================
# Position continuous scales
# =========================================================================

def scale_x_continuous(
    name: Any = None,
    *,
    breaks: Any = None,
    minor_breaks: Any = None,
    n_breaks: Optional[int] = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    oob: Optional[Callable] = None,
    na_value: float = np.nan,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: Any = None,
    position: str = "bottom",
    sec_axis: Any = None,
) -> ScaleContinuousPosition:
    """Continuous scale for the x position aesthetic.

    Parameters
    ----------
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
    limits : array-like or None
        Scale limits.
    expand : array-like or Waiver, optional
        Expansion.
    oob : callable, optional
        Out-of-bounds handler.
    na_value : float
        Value for missing data.
    transform : str or Transform
        Transformation.
    trans : str or Transform, optional
        Deprecated alias for *transform*.
    guide : any, optional
        Guide specification.
    position : str
        Axis position.
    sec_axis : AxisSecondary or Waiver, optional
        Secondary axis specification.

    Returns
    -------
    ScaleContinuousPosition
    """
    if guide is None:
        guide = waiver()
    if oob is None:
        oob = censor
    if sec_axis is None:
        sec_axis_obj = waiver()
    else:
        sec_axis_obj = sec_axis

    sc = continuous_scale(
        _X_AESTHETICS,
        palette=_identity,
        name=name,
        breaks=breaks,
        n_breaks=n_breaks,
        minor_breaks=minor_breaks,
        labels=labels,
        limits=limits,
        expand=expand,
        oob=oob,
        na_value=na_value,
        transform=transform,
        trans=trans,
        guide=guide,
        position=position,
        super_class=ScaleContinuousPosition,
    )
    return _set_sec_axis(sec_axis_obj, sc)


def scale_y_continuous(
    name: Any = None,
    *,
    breaks: Any = None,
    minor_breaks: Any = None,
    n_breaks: Optional[int] = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    oob: Optional[Callable] = None,
    na_value: float = np.nan,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: Any = None,
    position: str = "left",
    sec_axis: Any = None,
) -> ScaleContinuousPosition:
    """Continuous scale for the y position aesthetic.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    breaks, minor_breaks, n_breaks, labels, limits, expand, oob, na_value,
    transform, trans, guide, position, sec_axis
        See :func:`scale_x_continuous`.

    Returns
    -------
    ScaleContinuousPosition
    """
    if guide is None:
        guide = waiver()
    if oob is None:
        oob = censor
    if sec_axis is None:
        sec_axis_obj = waiver()
    else:
        sec_axis_obj = sec_axis

    sc = continuous_scale(
        _Y_AESTHETICS,
        palette=_identity,
        name=name,
        breaks=breaks,
        n_breaks=n_breaks,
        minor_breaks=minor_breaks,
        labels=labels,
        limits=limits,
        expand=expand,
        oob=oob,
        na_value=na_value,
        transform=transform,
        trans=trans,
        guide=guide,
        position=position,
        super_class=ScaleContinuousPosition,
    )
    return _set_sec_axis(sec_axis_obj, sc)


def scale_x_log10(**kwargs: Any) -> ScaleContinuousPosition:
    """Log10-transformed continuous x scale.

    Parameters
    ----------
    **kwargs
        Passed to :func:`scale_x_continuous`.

    Returns
    -------
    ScaleContinuousPosition
    """
    kwargs.setdefault("transform", transform_log10())
    return scale_x_continuous(**kwargs)


def scale_y_log10(**kwargs: Any) -> ScaleContinuousPosition:
    """Log10-transformed continuous y scale.

    Parameters
    ----------
    **kwargs
        Passed to :func:`scale_y_continuous`.

    Returns
    -------
    ScaleContinuousPosition
    """
    kwargs.setdefault("transform", transform_log10())
    return scale_y_continuous(**kwargs)


def scale_x_sqrt(**kwargs: Any) -> ScaleContinuousPosition:
    """Square-root-transformed continuous x scale.

    Parameters
    ----------
    **kwargs
        Passed to :func:`scale_x_continuous`.

    Returns
    -------
    ScaleContinuousPosition
    """
    kwargs.setdefault("transform", transform_sqrt())
    return scale_x_continuous(**kwargs)


def scale_y_sqrt(**kwargs: Any) -> ScaleContinuousPosition:
    """Square-root-transformed continuous y scale.

    Parameters
    ----------
    **kwargs
        Passed to :func:`scale_y_continuous`.

    Returns
    -------
    ScaleContinuousPosition
    """
    kwargs.setdefault("transform", transform_sqrt())
    return scale_y_continuous(**kwargs)


def scale_x_reverse(**kwargs: Any) -> ScaleContinuousPosition:
    """Reverse-transformed continuous x scale.

    Parameters
    ----------
    **kwargs
        Passed to :func:`scale_x_continuous`.

    Returns
    -------
    ScaleContinuousPosition
    """
    kwargs.setdefault("transform", transform_reverse())
    return scale_x_continuous(**kwargs)


def scale_y_reverse(**kwargs: Any) -> ScaleContinuousPosition:
    """Reverse-transformed continuous y scale.

    Parameters
    ----------
    **kwargs
        Passed to :func:`scale_y_continuous`.

    Returns
    -------
    ScaleContinuousPosition
    """
    kwargs.setdefault("transform", transform_reverse())
    return scale_y_continuous(**kwargs)


# =========================================================================
# Position discrete scales
# =========================================================================

def scale_x_discrete(
    name: Any = None,
    *,
    palette: Optional[Callable] = None,
    expand: Any = None,
    guide: Any = None,
    position: str = "bottom",
    sec_axis: Any = None,
    continuous_limits: Optional[Any] = None,
    **kwargs: Any,
) -> ScaleDiscretePosition:
    """Discrete scale for the x position aesthetic.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    palette : callable, optional
        Palette function.
    expand : array-like or Waiver, optional
        Expansion.
    guide : any, optional
        Guide specification.
    position : str
        Axis position.
    sec_axis : AxisSecondary or Waiver, optional
        Secondary axis specification.
    continuous_limits : array-like, optional
        Display range for continuous data on a discrete scale.
    **kwargs
        Extra arguments passed to ``discrete_scale``.

    Returns
    -------
    ScaleDiscretePosition
    """
    if palette is None:
        palette = _seq_len
    if guide is None:
        guide = waiver()
    if sec_axis is None:
        sec_axis_obj = waiver()
    else:
        sec_axis_obj = sec_axis

    sc = discrete_scale(
        _X_AESTHETICS,
        palette=palette,
        name=name,
        expand=expand,
        guide=guide,
        position=position,
        super_class=ScaleDiscretePosition,
        **kwargs,
    )
    sc.range_c = ContinuousRange()
    sc.continuous_limits = continuous_limits
    return _set_sec_axis(sec_axis_obj, sc)


def scale_y_discrete(
    name: Any = None,
    *,
    palette: Optional[Callable] = None,
    expand: Any = None,
    guide: Any = None,
    position: str = "left",
    sec_axis: Any = None,
    continuous_limits: Optional[Any] = None,
    **kwargs: Any,
) -> ScaleDiscretePosition:
    """Discrete scale for the y position aesthetic.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    palette, expand, guide, position, sec_axis, continuous_limits
        See :func:`scale_x_discrete`.
    **kwargs
        Extra arguments passed to ``discrete_scale``.

    Returns
    -------
    ScaleDiscretePosition
    """
    if palette is None:
        palette = _seq_len
    if guide is None:
        guide = waiver()
    if sec_axis is None:
        sec_axis_obj = waiver()
    else:
        sec_axis_obj = sec_axis

    sc = discrete_scale(
        _Y_AESTHETICS,
        palette=palette,
        name=name,
        expand=expand,
        guide=guide,
        position=position,
        super_class=ScaleDiscretePosition,
        **kwargs,
    )
    sc.range_c = ContinuousRange()
    sc.continuous_limits = continuous_limits
    return _set_sec_axis(sec_axis_obj, sc)


# =========================================================================
# Position binned scales
# =========================================================================

def scale_x_binned(
    name: Any = None,
    *,
    n_breaks: int = 10,
    nice_breaks: bool = True,
    breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    oob: Optional[Callable] = None,
    na_value: float = np.nan,
    right: bool = True,
    show_limits: bool = False,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: Any = None,
    position: str = "bottom",
) -> ScaleBinnedPosition:
    """Binned scale for the x position aesthetic.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    n_breaks : int
        Number of breaks.
    nice_breaks : bool
        Whether to use nice breaks.
    breaks, labels, limits, expand, oob, na_value, right, show_limits,
    transform, trans, guide, position
        See :func:`binned_scale`.

    Returns
    -------
    ScaleBinnedPosition
    """
    if guide is None:
        guide = waiver()
    if oob is None:
        oob = squish
    return binned_scale(
        _X_AESTHETICS,
        palette=_identity,
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        expand=expand,
        oob=oob,
        na_value=na_value,
        n_breaks=n_breaks,
        nice_breaks=nice_breaks,
        right=right,
        transform=transform,
        trans=trans,
        show_limits=show_limits,
        guide=guide,
        position=position,
        super_class=ScaleBinnedPosition,
    )


def scale_y_binned(
    name: Any = None,
    *,
    n_breaks: int = 10,
    nice_breaks: bool = True,
    breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    oob: Optional[Callable] = None,
    na_value: float = np.nan,
    right: bool = True,
    show_limits: bool = False,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: Any = None,
    position: str = "left",
) -> ScaleBinnedPosition:
    """Binned scale for the y position aesthetic.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    n_breaks, nice_breaks, breaks, labels, limits, expand, oob, na_value,
    right, show_limits, transform, trans, guide, position
        See :func:`scale_x_binned`.

    Returns
    -------
    ScaleBinnedPosition
    """
    if guide is None:
        guide = waiver()
    if oob is None:
        oob = squish
    return binned_scale(
        _Y_AESTHETICS,
        palette=_identity,
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        expand=expand,
        oob=oob,
        na_value=na_value,
        n_breaks=n_breaks,
        nice_breaks=nice_breaks,
        right=right,
        transform=transform,
        trans=trans,
        show_limits=show_limits,
        guide=guide,
        position=position,
        super_class=ScaleBinnedPosition,
    )


# =========================================================================
# Position date/datetime/time scales
# =========================================================================

def scale_x_date(
    name: Any = None,
    *,
    breaks: Any = None,
    date_breaks: Optional[str] = None,
    labels: Any = None,
    date_labels: Optional[str] = None,
    minor_breaks: Any = None,
    date_minor_breaks: Optional[str] = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    oob: Optional[Callable] = None,
    guide: Any = None,
    position: str = "bottom",
    sec_axis: Any = None,
) -> ScaleContinuousPosition:
    """Date scale for the x position aesthetic.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    breaks, date_breaks, labels, date_labels, minor_breaks,
    date_minor_breaks, limits, expand, oob, guide, position, sec_axis
        Standard scale parameters.

    Returns
    -------
    ScaleContinuousPosition
    """
    if oob is None:
        oob = censor
    return scale_x_continuous(
        name=name,
        breaks=breaks,
        labels=labels,
        minor_breaks=minor_breaks,
        limits=limits,
        expand=expand,
        oob=oob,
        guide=guide,
        position=position,
        sec_axis=sec_axis,
        transform=transform_date(),
    )


def scale_y_date(
    name: Any = None,
    *,
    breaks: Any = None,
    labels: Any = None,
    minor_breaks: Any = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    oob: Optional[Callable] = None,
    guide: Any = None,
    position: str = "left",
    sec_axis: Any = None,
) -> ScaleContinuousPosition:
    """Date scale for the y position aesthetic.

    Parameters
    ----------
    name, breaks, labels, minor_breaks, limits, expand, oob, guide,
    position, sec_axis
        Standard scale parameters.

    Returns
    -------
    ScaleContinuousPosition
    """
    if oob is None:
        oob = censor
    return scale_y_continuous(
        name=name,
        breaks=breaks,
        labels=labels,
        minor_breaks=minor_breaks,
        limits=limits,
        expand=expand,
        oob=oob,
        guide=guide,
        position=position,
        sec_axis=sec_axis,
        transform=transform_date(),
    )


def scale_x_datetime(
    name: Any = None,
    *,
    breaks: Any = None,
    date_breaks: Optional[str] = None,
    labels: Any = None,
    date_labels: Optional[str] = None,
    minor_breaks: Any = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    oob: Optional[Callable] = None,
    guide: Any = None,
    position: str = "bottom",
    sec_axis: Any = None,
) -> ScaleContinuousPosition:
    """Datetime scale for the x position aesthetic.

    Parameters
    ----------
    name, breaks, date_breaks, labels, date_labels, minor_breaks,
    limits, expand, oob, guide, position, sec_axis
        Standard scale parameters.

    Returns
    -------
    ScaleContinuousPosition
    """
    if oob is None:
        oob = censor
    return scale_x_continuous(
        name=name,
        breaks=breaks,
        labels=labels,
        minor_breaks=minor_breaks,
        limits=limits,
        expand=expand,
        oob=oob,
        guide=guide,
        position=position,
        sec_axis=sec_axis,
        transform=transform_time(),
    )


def scale_y_datetime(
    name: Any = None,
    *,
    breaks: Any = None,
    labels: Any = None,
    minor_breaks: Any = None,
    limits: Optional[Any] = None,
    expand: Any = None,
    oob: Optional[Callable] = None,
    guide: Any = None,
    position: str = "left",
    sec_axis: Any = None,
) -> ScaleContinuousPosition:
    """Datetime scale for the y position aesthetic.

    Parameters
    ----------
    name, breaks, labels, minor_breaks, limits, expand, oob, guide,
    position, sec_axis
        Standard scale parameters.

    Returns
    -------
    ScaleContinuousPosition
    """
    if oob is None:
        oob = censor
    return scale_y_continuous(
        name=name,
        breaks=breaks,
        labels=labels,
        minor_breaks=minor_breaks,
        limits=limits,
        expand=expand,
        oob=oob,
        guide=guide,
        position=position,
        sec_axis=sec_axis,
        transform=transform_time(),
    )


# Aliases
scale_x_time = scale_x_datetime
scale_y_time = scale_y_datetime


# =========================================================================
# Colour / fill continuous (gradient) scales
# =========================================================================

def scale_colour_continuous(
    name: Any = None,
    *,
    palette: Optional[Any] = None,
    aesthetics: Union[str, List[str]] = "colour",
    guide: str = "colourbar",
    na_value: str = "grey50",
    **kwargs: Any,
) -> ScaleContinuous:
    """Default continuous colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    palette : callable or list of str, optional
        Palette specification.
    aesthetics : str or list of str
        Aesthetic names.
    guide : str
        Guide type.
    na_value : str
        Colour for missing values.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    pal = palette
    if pal is not None and not callable(pal):
        if isinstance(pal, (list, tuple)):
            pal = pal_gradient_n(pal)
    return continuous_scale(
        aesthetics,
        palette=pal,
        name=name,
        guide=guide,
        na_value=na_value,
        fallback_palette=pal_seq_gradient("#132B43", "#56B1F7"),
        **kwargs,
    )


def scale_fill_continuous(
    name: Any = None,
    *,
    palette: Optional[Any] = None,
    aesthetics: Union[str, List[str]] = "fill",
    guide: str = "colourbar",
    na_value: str = "grey50",
    **kwargs: Any,
) -> ScaleContinuous:
    """Default continuous fill scale.

    Parameters
    ----------
    name, palette, aesthetics, guide, na_value
        See :func:`scale_colour_continuous`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    pal = palette
    if pal is not None and not callable(pal):
        if isinstance(pal, (list, tuple)):
            pal = pal_gradient_n(pal)
    return continuous_scale(
        aesthetics,
        palette=pal,
        name=name,
        guide=guide,
        na_value=na_value,
        fallback_palette=pal_seq_gradient("#132B43", "#56B1F7"),
        **kwargs,
    )


def scale_colour_gradient(
    name: Any = None,
    *,
    low: str = "#132B43",
    high: str = "#56B1F7",
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleContinuous:
    """Two-colour gradient colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    low : str
        Colour for the low end.
    high : str
        Colour for the high end.
    space : str
        Colour interpolation space.
    na_value : str
        Colour for missing values.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        palette=pal_seq_gradient(low, high, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_fill_gradient(
    name: Any = None,
    *,
    low: str = "#132B43",
    high: str = "#56B1F7",
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleContinuous:
    """Two-colour gradient fill scale.

    Parameters
    ----------
    name, low, high, space, na_value, guide, aesthetics
        See :func:`scale_colour_gradient`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        palette=pal_seq_gradient(low, high, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_colour_gradient2(
    name: Any = None,
    *,
    low: Optional[str] = None,
    mid: str = "white",
    high: Optional[str] = None,
    midpoint: float = 0,
    space: str = "Lab",
    na_value: str = "grey50",
    transform: Union[str, Any] = "identity",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleContinuous:
    """Diverging colour gradient scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    low : str, optional
        Colour for the low end (default: muted red).
    mid : str
        Colour for the midpoint.
    high : str, optional
        Colour for the high end (default: muted blue).
    midpoint : float
        Data value at the midpoint.
    space : str
        Colour interpolation space.
    na_value : str
        Colour for missing values.
    transform : str or Transform
        Transformation.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    if low is None:
        low = muted("red")
    if high is None:
        high = muted("blue")
    return continuous_scale(
        aesthetics,
        palette=pal_div_gradient(low, mid, high, space),
        name=name,
        na_value=na_value,
        transform=transform,
        guide=guide,
        rescaler=_mid_rescaler(mid=midpoint, transform=transform),
        **kwargs,
    )


def scale_fill_gradient2(
    name: Any = None,
    *,
    low: Optional[str] = None,
    mid: str = "white",
    high: Optional[str] = None,
    midpoint: float = 0,
    space: str = "Lab",
    na_value: str = "grey50",
    transform: Union[str, Any] = "identity",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleContinuous:
    """Diverging fill gradient scale.

    Parameters
    ----------
    name, low, mid, high, midpoint, space, na_value, transform, guide, aesthetics
        See :func:`scale_colour_gradient2`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    if low is None:
        low = muted("red")
    if high is None:
        high = muted("blue")
    return continuous_scale(
        aesthetics,
        palette=pal_div_gradient(low, mid, high, space),
        name=name,
        na_value=na_value,
        transform=transform,
        guide=guide,
        rescaler=_mid_rescaler(mid=midpoint, transform=transform),
        **kwargs,
    )


def scale_colour_gradientn(
    name: Any = None,
    *,
    colours: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleContinuous:
    """N-colour gradient colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    colours : list of str, optional
        Colour values.
    colors : list of str, optional
        Alias for *colours*.
    values : list of float, optional
        Positions of colours.
    space : str
        Colour interpolation space.
    na_value : str
        Colour for missing values.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    cols = colours if colours is not None else colors
    if cols is None:
        cli_abort("Must provide either 'colours' or 'colors'.")
    return continuous_scale(
        aesthetics,
        palette=pal_gradient_n(cols, values, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_fill_gradientn(
    name: Any = None,
    *,
    colours: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleContinuous:
    """N-colour gradient fill scale.

    Parameters
    ----------
    name, colours, colors, values, space, na_value, guide, aesthetics
        See :func:`scale_colour_gradientn`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    cols = colours if colours is not None else colors
    if cols is None:
        cli_abort("Must provide either 'colours' or 'colors'.")
    return continuous_scale(
        aesthetics,
        palette=pal_gradient_n(cols, values, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


# =========================================================================
# Colour / fill discrete scales
# =========================================================================

def scale_colour_discrete(
    name: Any = None,
    *,
    palette: Optional[Any] = None,
    aesthetics: Union[str, List[str]] = "colour",
    na_value: str = "grey50",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Default discrete colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    palette : callable, optional
        Palette function.
    aesthetics : str or list of str
        Aesthetic names.
    na_value : str
        Colour for missing values.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=palette,
        name=name,
        na_value=na_value,
        fallback_palette=pal_hue(),
        **kwargs,
    )


def scale_fill_discrete(
    name: Any = None,
    *,
    palette: Optional[Any] = None,
    aesthetics: Union[str, List[str]] = "fill",
    na_value: str = "grey50",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Default discrete fill scale.

    Parameters
    ----------
    name, palette, aesthetics, na_value
        See :func:`scale_colour_discrete`.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=palette,
        name=name,
        na_value=na_value,
        fallback_palette=pal_hue(),
        **kwargs,
    )


def scale_colour_hue(
    name: Any = None,
    *,
    h: Sequence[float] = (15, 375),
    c: float = 100,
    l: float = 65,
    h_start: float = 0,
    direction: int = 1,
    na_value: str = "grey50",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Evenly spaced hue colour scale for discrete data.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    h : tuple of float
        Range of hues (0--360).
    c : float
        Chroma.
    l : float
        Luminance.
    h_start : float
        Hue offset.
    direction : int
        Direction of hue traverse (1 or -1).
    na_value : str
        Colour for missing values.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_hue(h, c, l, h_start, direction),
        name=name,
        na_value=na_value,
        **kwargs,
    )


def scale_fill_hue(
    name: Any = None,
    *,
    h: Sequence[float] = (15, 375),
    c: float = 100,
    l: float = 65,
    h_start: float = 0,
    direction: int = 1,
    na_value: str = "grey50",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Evenly spaced hue fill scale for discrete data.

    Parameters
    ----------
    name, h, c, l, h_start, direction, na_value, aesthetics
        See :func:`scale_colour_hue`.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_hue(h, c, l, h_start, direction),
        name=name,
        na_value=na_value,
        **kwargs,
    )


# =========================================================================
# Brewer scales
# =========================================================================

def scale_colour_brewer(
    name: Any = None,
    *,
    type: str = "seq",
    palette: Union[int, str] = 1,
    direction: int = 1,
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleDiscrete:
    """ColorBrewer discrete colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    type : str
        Palette type (``'seq'``, ``'div'``, ``'qual'``).
    palette : int or str
        Palette name or index.
    direction : int
        Colour order direction.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_brewer(type, palette, direction),
        name=name,
        **kwargs,
    )


def scale_fill_brewer(
    name: Any = None,
    *,
    type: str = "seq",
    palette: Union[int, str] = 1,
    direction: int = 1,
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleDiscrete:
    """ColorBrewer discrete fill scale.

    Parameters
    ----------
    name, type, palette, direction, aesthetics
        See :func:`scale_colour_brewer`.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_brewer(type, palette, direction),
        name=name,
        **kwargs,
    )


def scale_colour_distiller(
    name: Any = None,
    *,
    type: str = "seq",
    palette: Union[int, str] = 1,
    direction: int = -1,
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleContinuous:
    """Continuous colour scale interpolated from a Brewer palette.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    type : str
        Palette type.
    palette : int or str
        Palette name or index.
    direction : int
        Colour order direction.
    values : list of float, optional
        Positions for colours.
    space : str
        Interpolation space.
    na_value : str
        Colour for missing values.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        palette=pal_gradient_n(pal_brewer(type, palette, direction)(7), values, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_fill_distiller(
    name: Any = None,
    *,
    type: str = "seq",
    palette: Union[int, str] = 1,
    direction: int = -1,
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleContinuous:
    """Continuous fill scale interpolated from a Brewer palette.

    Parameters
    ----------
    name, type, palette, direction, values, space, na_value, guide, aesthetics
        See :func:`scale_colour_distiller`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        palette=pal_gradient_n(pal_brewer(type, palette, direction)(7), values, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_colour_fermenter(
    name: Any = None,
    *,
    type: str = "seq",
    palette: Union[int, str] = 1,
    direction: int = -1,
    na_value: str = "grey50",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleBinned:
    """Binned colour scale from a Brewer palette.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    type : str
        Palette type.
    palette : int or str
        Palette name or index.
    direction : int
        Colour order direction.
    na_value : str
        Colour for missing values.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=pal_brewer(type, palette, direction),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_fill_fermenter(
    name: Any = None,
    *,
    type: str = "seq",
    palette: Union[int, str] = 1,
    direction: int = -1,
    na_value: str = "grey50",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleBinned:
    """Binned fill scale from a Brewer palette.

    Parameters
    ----------
    name, type, palette, direction, na_value, guide, aesthetics
        See :func:`scale_colour_fermenter`.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=pal_brewer(type, palette, direction),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


# =========================================================================
# Grey scales
# =========================================================================

def scale_colour_grey(
    name: Any = None,
    *,
    start: float = 0.2,
    end: float = 0.8,
    na_value: str = "red",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Sequential grey discrete colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    start : float
        Grey level for the lightest colour.
    end : float
        Grey level for the darkest colour.
    na_value : str
        Colour for missing values.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_grey(start, end),
        name=name,
        na_value=na_value,
        **kwargs,
    )


def scale_fill_grey(
    name: Any = None,
    *,
    start: float = 0.2,
    end: float = 0.8,
    na_value: str = "red",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Sequential grey discrete fill scale.

    Parameters
    ----------
    name, start, end, na_value, aesthetics
        See :func:`scale_colour_grey`.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_grey(start, end),
        name=name,
        na_value=na_value,
        **kwargs,
    )


# =========================================================================
# Viridis scales
# =========================================================================

def scale_colour_viridis_d(
    name: Any = None,
    *,
    alpha: float = 1,
    begin: float = 0,
    end: float = 1,
    direction: int = 1,
    option: str = "D",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Viridis discrete colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    alpha : float
        Alpha transparency.
    begin, end : float
        Range within the colour map (0--1).
    direction : int
        Colour order direction.
    option : str
        Colour map variant.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_viridis(alpha, begin, end, direction, option),
        name=name,
        **kwargs,
    )


def scale_fill_viridis_d(
    name: Any = None,
    *,
    alpha: float = 1,
    begin: float = 0,
    end: float = 1,
    direction: int = 1,
    option: str = "D",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Viridis discrete fill scale.

    Parameters
    ----------
    name, alpha, begin, end, direction, option, aesthetics
        See :func:`scale_colour_viridis_d`.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_viridis(alpha, begin, end, direction, option),
        name=name,
        **kwargs,
    )


def scale_colour_viridis_c(
    name: Any = None,
    *,
    alpha: float = 1,
    begin: float = 0,
    end: float = 1,
    direction: int = 1,
    option: str = "D",
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleContinuous:
    """Viridis continuous colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    alpha, begin, end, direction, option
        Viridis palette parameters.
    values : list of float, optional
        Positions for colours.
    space : str
        Interpolation space.
    na_value : str
        Colour for missing values.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        palette=pal_gradient_n(
            pal_viridis(alpha, begin, end, direction, option)(6),
            values,
            space,
        ),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_fill_viridis_c(
    name: Any = None,
    *,
    alpha: float = 1,
    begin: float = 0,
    end: float = 1,
    direction: int = 1,
    option: str = "D",
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "colourbar",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleContinuous:
    """Viridis continuous fill scale.

    Parameters
    ----------
    name, alpha, begin, end, direction, option, values, space, na_value,
    guide, aesthetics
        See :func:`scale_colour_viridis_c`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        palette=pal_gradient_n(
            pal_viridis(alpha, begin, end, direction, option)(6),
            values,
            space,
        ),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_colour_viridis_b(
    name: Any = None,
    *,
    alpha: float = 1,
    begin: float = 0,
    end: float = 1,
    direction: int = 1,
    option: str = "D",
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleBinned:
    """Viridis binned colour scale.

    Parameters
    ----------
    name, alpha, begin, end, direction, option, values, space, na_value,
    guide, aesthetics
        See :func:`scale_colour_viridis_c`.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=pal_viridis(alpha, begin, end, direction, option),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_fill_viridis_b(
    name: Any = None,
    *,
    alpha: float = 1,
    begin: float = 0,
    end: float = 1,
    direction: int = 1,
    option: str = "D",
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleBinned:
    """Viridis binned fill scale.

    Parameters
    ----------
    name, alpha, begin, end, direction, option, values, space, na_value,
    guide, aesthetics
        See :func:`scale_colour_viridis_b`.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=pal_viridis(alpha, begin, end, direction, option),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


# =========================================================================
# Colour / fill binned (steps) scales
# =========================================================================

def scale_colour_binned(
    name: Any = None,
    *,
    palette: Optional[Any] = None,
    aesthetics: Union[str, List[str]] = "colour",
    guide: str = "coloursteps",
    na_value: str = "grey50",
    **kwargs: Any,
) -> ScaleBinned:
    """Default binned colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    palette : callable or list of str, optional
        Palette specification.
    aesthetics : str or list of str
        Aesthetic names.
    guide : str
        Guide type.
    na_value : str
        Colour for missing values.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=palette,
        name=name,
        guide=guide,
        na_value=na_value,
        fallback_palette=pal_seq_gradient("#132B43", "#56B1F7"),
        **kwargs,
    )


def scale_fill_binned(
    name: Any = None,
    *,
    palette: Optional[Any] = None,
    aesthetics: Union[str, List[str]] = "fill",
    guide: str = "coloursteps",
    na_value: str = "grey50",
    **kwargs: Any,
) -> ScaleBinned:
    """Default binned fill scale.

    Parameters
    ----------
    name, palette, aesthetics, guide, na_value
        See :func:`scale_colour_binned`.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=palette,
        name=name,
        guide=guide,
        na_value=na_value,
        fallback_palette=pal_seq_gradient("#132B43", "#56B1F7"),
        **kwargs,
    )


def scale_colour_steps(
    name: Any = None,
    *,
    low: str = "#132B43",
    high: str = "#56B1F7",
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleBinned:
    """Binned two-colour gradient colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    low, high : str
        Gradient endpoint colours.
    space : str
        Interpolation space.
    na_value : str
        Colour for missing values.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=pal_seq_gradient(low, high, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_fill_steps(
    name: Any = None,
    *,
    low: str = "#132B43",
    high: str = "#56B1F7",
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleBinned:
    """Binned two-colour gradient fill scale.

    Parameters
    ----------
    name, low, high, space, na_value, guide, aesthetics
        See :func:`scale_colour_steps`.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=pal_seq_gradient(low, high, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_colour_steps2(
    name: Any = None,
    *,
    low: Optional[str] = None,
    mid: str = "white",
    high: Optional[str] = None,
    midpoint: float = 0,
    space: str = "Lab",
    na_value: str = "grey50",
    transform: Union[str, Any] = "identity",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleBinned:
    """Diverging binned colour gradient scale.

    Parameters
    ----------
    name, low, mid, high, midpoint, space, na_value, transform, guide, aesthetics
        See :func:`scale_colour_gradient2` for parameter descriptions.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    if low is None:
        low = muted("red")
    if high is None:
        high = muted("blue")
    return binned_scale(
        aesthetics,
        palette=pal_div_gradient(low, mid, high, space),
        name=name,
        na_value=na_value,
        transform=transform,
        guide=guide,
        rescaler=_mid_rescaler(mid=midpoint, transform=transform),
        **kwargs,
    )


def scale_fill_steps2(
    name: Any = None,
    *,
    low: Optional[str] = None,
    mid: str = "white",
    high: Optional[str] = None,
    midpoint: float = 0,
    space: str = "Lab",
    na_value: str = "grey50",
    transform: Union[str, Any] = "identity",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleBinned:
    """Diverging binned fill gradient scale.

    Parameters
    ----------
    name, low, mid, high, midpoint, space, na_value, transform, guide, aesthetics
        See :func:`scale_colour_steps2`.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    if low is None:
        low = muted("red")
    if high is None:
        high = muted("blue")
    return binned_scale(
        aesthetics,
        palette=pal_div_gradient(low, mid, high, space),
        name=name,
        na_value=na_value,
        transform=transform,
        guide=guide,
        rescaler=_mid_rescaler(mid=midpoint, transform=transform),
        **kwargs,
    )


def scale_colour_stepsn(
    name: Any = None,
    *,
    colours: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleBinned:
    """N-colour binned gradient colour scale.

    Parameters
    ----------
    name, colours, colors, values, space, na_value, guide, aesthetics
        See :func:`scale_colour_gradientn`.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    cols = colours if colours is not None else colors
    if cols is None:
        cli_abort("Must provide either 'colours' or 'colors'.")
    return binned_scale(
        aesthetics,
        palette=pal_gradient_n(cols, values, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


def scale_fill_stepsn(
    name: Any = None,
    *,
    colours: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
    na_value: str = "grey50",
    guide: str = "coloursteps",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleBinned:
    """N-colour binned gradient fill scale.

    Parameters
    ----------
    name, colours, colors, values, space, na_value, guide, aesthetics
        See :func:`scale_colour_stepsn`.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    cols = colours if colours is not None else colors
    if cols is None:
        cli_abort("Must provide either 'colours' or 'colors'.")
    return binned_scale(
        aesthetics,
        palette=pal_gradient_n(cols, values, space),
        name=name,
        na_value=na_value,
        guide=guide,
        **kwargs,
    )


# =========================================================================
# Identity scales
# =========================================================================

def scale_colour_identity(
    name: Any = None,
    *,
    guide: str = "none",
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleDiscreteIdentity:
    """Identity colour scale -- data values used as colours directly.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscreteIdentity
    """
    return discrete_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleDiscreteIdentity,
        **kwargs,
    )


def scale_fill_identity(
    name: Any = None,
    *,
    guide: str = "none",
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleDiscreteIdentity:
    """Identity fill scale.

    Parameters
    ----------
    name, guide, aesthetics
        See :func:`scale_colour_identity`.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscreteIdentity
    """
    return discrete_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleDiscreteIdentity,
        **kwargs,
    )


def scale_continuous_identity(
    aesthetics: Union[str, List[str]],
    name: Any = None,
    *,
    guide: str = "none",
    **kwargs: Any,
) -> ScaleContinuousIdentity:
    """Generic continuous identity scale.

    Parameters
    ----------
    aesthetics : str or list of str
        Aesthetic names.
    name : str or Waiver, optional
        Scale title.
    guide : str
        Guide type.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuousIdentity
    """
    return continuous_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleContinuousIdentity,
        **kwargs,
    )


def scale_discrete_identity(
    aesthetics: Union[str, List[str]],
    name: Any = None,
    *,
    guide: str = "none",
    **kwargs: Any,
) -> ScaleDiscreteIdentity:
    """Generic discrete identity scale.

    Parameters
    ----------
    aesthetics : str or list of str
        Aesthetic names.
    name : str or Waiver, optional
        Scale title.
    guide : str
        Guide type.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscreteIdentity
    """
    return discrete_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleDiscreteIdentity,
        **kwargs,
    )


# =========================================================================
# Manual scales
# =========================================================================

def scale_colour_manual(
    *,
    values: Any,
    aesthetics: Union[str, List[str]] = "colour",
    breaks: Any = None,
    na_value: str = "grey50",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Manual colour scale.

    Parameters
    ----------
    values : dict or list
        Mapping from data values to colours.
    aesthetics : str or list of str
        Aesthetic names.
    breaks : any, optional
        Break specification.
    na_value : str
        Colour for missing values.
    **kwargs
        Passed to ``_manual_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if breaks is None:
        breaks = waiver()
    return _manual_scale(aesthetics, values, breaks, na_value=na_value, **kwargs)


def scale_fill_manual(
    *,
    values: Any,
    aesthetics: Union[str, List[str]] = "fill",
    breaks: Any = None,
    na_value: str = "grey50",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Manual fill scale.

    Parameters
    ----------
    values, aesthetics, breaks, na_value
        See :func:`scale_colour_manual`.
    **kwargs
        Passed to ``_manual_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if breaks is None:
        breaks = waiver()
    return _manual_scale(aesthetics, values, breaks, na_value=na_value, **kwargs)


def scale_discrete_manual(
    aesthetics: Union[str, List[str]],
    *,
    values: Any,
    breaks: Any = None,
    **kwargs: Any,
) -> ScaleDiscrete:
    """Generic discrete manual scale.

    Parameters
    ----------
    aesthetics : str or list of str
        Aesthetic names.
    values : dict or list
        Manual values.
    breaks : any, optional
        Break specification.
    **kwargs
        Passed to ``_manual_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if breaks is None:
        breaks = waiver()
    return _manual_scale(aesthetics, values, breaks, **kwargs)


# =========================================================================
# Colour/fill date, datetime, ordinal
# =========================================================================

def scale_colour_date(
    name: Any = None,
    *,
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleContinuous:
    """Date colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        name=name,
        transform=transform_date(),
        fallback_palette=pal_seq_gradient("#132B43", "#56B1F7"),
        **kwargs,
    )


def scale_fill_date(
    name: Any = None,
    *,
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleContinuous:
    """Date fill scale.

    Parameters
    ----------
    name, aesthetics
        See :func:`scale_colour_date`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        name=name,
        transform=transform_date(),
        fallback_palette=pal_seq_gradient("#132B43", "#56B1F7"),
        **kwargs,
    )


def scale_colour_datetime(
    name: Any = None,
    *,
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleContinuous:
    """Datetime colour scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        name=name,
        transform=transform_time(),
        fallback_palette=pal_seq_gradient("#132B43", "#56B1F7"),
        **kwargs,
    )


def scale_fill_datetime(
    name: Any = None,
    *,
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleContinuous:
    """Datetime fill scale.

    Parameters
    ----------
    name, aesthetics
        See :func:`scale_colour_datetime`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        name=name,
        transform=transform_time(),
        fallback_palette=pal_seq_gradient("#132B43", "#56B1F7"),
        **kwargs,
    )


def scale_colour_ordinal(
    name: Any = None,
    *,
    aesthetics: Union[str, List[str]] = "colour",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Ordinal colour scale (viridis palette).

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_viridis(),
        name=name,
        **kwargs,
    )


def scale_fill_ordinal(
    name: Any = None,
    *,
    aesthetics: Union[str, List[str]] = "fill",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Ordinal fill scale (viridis palette).

    Parameters
    ----------
    name, aesthetics
        See :func:`scale_colour_ordinal`.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=pal_viridis(),
        name=name,
        **kwargs,
    )


# =========================================================================
# Alpha scales
# =========================================================================

def scale_alpha(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "alpha",
    **kwargs: Any,
) -> ScaleContinuous:
    """Alpha transparency continuous scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    range : tuple of float, optional
        Output alpha range (default ``(0.1, 1)``).
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_rescale(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        fallback_palette=pal_rescale((0.1, 1)),
        **kwargs,
    )


scale_alpha_continuous = scale_alpha


def scale_alpha_binned(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "alpha",
    **kwargs: Any,
) -> ScaleBinned:
    """Alpha transparency binned scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    range : tuple of float, optional
        Output alpha range.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    palette = pal_rescale(range) if range is not None else None
    return binned_scale(
        aesthetics,
        palette=palette,
        name=name,
        fallback_palette=pal_rescale((0.1, 1)),
        **kwargs,
    )


def scale_alpha_discrete(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "alpha",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Alpha transparency discrete scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    range : tuple of float, optional
        Output alpha range.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`scale_alpha_ordinal`.

    Returns
    -------
    ScaleDiscrete
    """
    cli_warn("Using alpha for a discrete variable is not advised.")
    return scale_alpha_ordinal(name=name, range=range, aesthetics=aesthetics, **kwargs)


def scale_alpha_ordinal(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "alpha",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Alpha transparency ordinal scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    range : tuple of float, optional
        Output alpha range.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    if range is not None:
        _r = range

        def palette(n: int) -> List[float]:
            return list(np.linspace(_r[0], _r[1], n))
    else:
        palette = None
    return discrete_scale(
        aesthetics,
        palette=palette,
        name=name,
        fallback_palette=lambda n: list(np.linspace(0.1, 1, n)),
        **kwargs,
    )


def scale_alpha_identity(
    name: Any = None,
    *,
    guide: str = "none",
    aesthetics: Union[str, List[str]] = "alpha",
    **kwargs: Any,
) -> ScaleContinuousIdentity:
    """Alpha identity scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuousIdentity
    """
    return continuous_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleContinuousIdentity,
        **kwargs,
    )


def scale_alpha_manual(
    *,
    values: Any,
    breaks: Any = None,
    na_value: Any = np.nan,
    aesthetics: Union[str, List[str]] = "alpha",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Alpha manual scale.

    Parameters
    ----------
    values : dict or list
        Manual alpha values.
    breaks : any, optional
        Break specification.
    na_value : any
        Value for missing data.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to ``_manual_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if breaks is None:
        breaks = waiver()
    return _manual_scale(aesthetics, values, breaks, na_value=na_value, **kwargs)


def scale_alpha_date(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "alpha",
    **kwargs: Any,
) -> ScaleContinuous:
    """Alpha date scale.

    Parameters
    ----------
    name, range, aesthetics
        See :func:`scale_alpha`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_rescale(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        transform=transform_date(),
        fallback_palette=pal_rescale((0.1, 1)),
        **kwargs,
    )


def scale_alpha_datetime(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "alpha",
    **kwargs: Any,
) -> ScaleContinuous:
    """Alpha datetime scale.

    Parameters
    ----------
    name, range, aesthetics
        See :func:`scale_alpha`.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_rescale(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        transform=transform_time(),
        fallback_palette=pal_rescale((0.1, 1)),
        **kwargs,
    )


# =========================================================================
# Size scales
# =========================================================================

def scale_size_continuous(
    name: Any = None,
    *,
    breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    range: Optional[Sequence[float]] = None,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: str = "legend",
    aesthetics: Union[str, List[str]] = "size",
) -> ScaleContinuous:
    """Continuous size scale (area-based).

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    breaks, labels, limits
        Standard break/label/limit parameters.
    range : tuple of float, optional
        Size range.
    transform : str or Transform
        Transformation.
    trans : str or Transform, optional
        Deprecated alias.
    guide : str
        Guide type.
    aesthetics : str or list of str
        Aesthetic names.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_area(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        transform=transform,
        trans=trans,
        guide=guide,
        fallback_palette=pal_area(),
    )


scale_size = scale_size_continuous


def scale_radius(
    name: Any = None,
    *,
    breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    range: Sequence[float] = (1, 6),
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: str = "legend",
    aesthetics: Union[str, List[str]] = "size",
) -> ScaleContinuous:
    """Radius-based size scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    breaks, labels, limits
        Standard parameters.
    range : tuple of float
        Radius range.
    transform, trans, guide, aesthetics
        Standard parameters.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        palette=pal_rescale(range),
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        transform=transform,
        trans=trans,
        guide=guide,
    )


def scale_size_binned(
    name: Any = None,
    *,
    breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    range: Optional[Sequence[float]] = None,
    n_breaks: Optional[int] = None,
    nice_breaks: bool = True,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: str = "bins",
    aesthetics: Union[str, List[str]] = "size",
) -> ScaleBinned:
    """Binned size scale.

    Parameters
    ----------
    name, breaks, labels, limits, range, n_breaks, nice_breaks,
    transform, trans, guide, aesthetics
        Standard scale parameters.

    Returns
    -------
    ScaleBinned
    """
    palette = pal_area(range) if range is not None else None
    return binned_scale(
        aesthetics,
        palette=palette,
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        transform=transform,
        trans=trans,
        n_breaks=n_breaks,
        nice_breaks=nice_breaks,
        guide=guide,
        fallback_palette=pal_area(),
    )


def scale_size_discrete(
    name: Any = None,
    **kwargs: Any,
) -> ScaleDiscrete:
    """Discrete size scale (not recommended).

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    **kwargs
        Passed to :func:`scale_size_ordinal`.

    Returns
    -------
    ScaleDiscrete
    """
    cli_warn("Using size for a discrete variable is not advised.")
    return scale_size_ordinal(name=name, **kwargs)


def scale_size_ordinal(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "size",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Ordinal size scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    range : tuple of float, optional
        Size range.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    if range is not None:
        _r = range

        def palette(n: int) -> List[float]:
            return list(np.sqrt(np.linspace(_r[0] ** 2, _r[1] ** 2, n)))
    else:
        palette = None
    return discrete_scale(
        aesthetics,
        palette=palette,
        name=name,
        fallback_palette=lambda n: list(np.sqrt(np.linspace(4, 36, n))),
        **kwargs,
    )


def scale_size_area(
    name: Any = None,
    *,
    max_size: float = 6,
    aesthetics: Union[str, List[str]] = "size",
    **kwargs: Any,
) -> ScaleContinuous:
    """Size scale where 0 maps to size 0.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    max_size : float
        Maximum point size.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    return continuous_scale(
        aesthetics,
        palette=abs_area(max_size),
        name=name,
        rescaler=rescale_max,
        **kwargs,
    )


def scale_size_binned_area(
    name: Any = None,
    *,
    max_size: float = 6,
    aesthetics: Union[str, List[str]] = "size",
    **kwargs: Any,
) -> ScaleBinned:
    """Binned size scale where 0 maps to size 0.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    max_size : float
        Maximum point size.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=abs_area(max_size),
        name=name,
        rescaler=rescale_max,
        **kwargs,
    )


def scale_size_identity(
    name: Any = None,
    *,
    guide: str = "none",
    aesthetics: Union[str, List[str]] = "size",
    **kwargs: Any,
) -> ScaleContinuousIdentity:
    """Size identity scale.

    Parameters
    ----------
    name, guide, aesthetics
        Standard parameters.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuousIdentity
    """
    return continuous_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleContinuousIdentity,
        **kwargs,
    )


def scale_size_manual(
    *,
    values: Any,
    breaks: Any = None,
    na_value: Any = np.nan,
    aesthetics: Union[str, List[str]] = "size",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Size manual scale.

    Parameters
    ----------
    values, breaks, na_value, aesthetics
        Standard parameters.
    **kwargs
        Passed to ``_manual_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if breaks is None:
        breaks = waiver()
    return _manual_scale(aesthetics, values, breaks, na_value=na_value, **kwargs)


def scale_size_date(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "size",
    **kwargs: Any,
) -> ScaleContinuous:
    """Size date scale.

    Parameters
    ----------
    name, range, aesthetics
        Standard parameters.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_area(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        transform=transform_date(),
        fallback_palette=pal_area(),
        **kwargs,
    )


def scale_size_datetime(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "size",
    **kwargs: Any,
) -> ScaleContinuous:
    """Size datetime scale.

    Parameters
    ----------
    name, range, aesthetics
        Standard parameters.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_area(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        transform=transform_time(),
        fallback_palette=pal_area(),
        **kwargs,
    )


# =========================================================================
# Shape scales
# =========================================================================

def scale_shape(
    name: Any = None,
    *,
    solid: Optional[bool] = None,
    aesthetics: Union[str, List[str]] = "shape",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Discrete shape scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    solid : bool, optional
        Whether shapes are solid (default) or hollow.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    palette = pal_shape(solid) if solid is not None else None
    return discrete_scale(
        aesthetics,
        palette=palette,
        name=name,
        fallback_palette=pal_shape(),
        **kwargs,
    )


scale_shape_discrete = scale_shape


def scale_shape_binned(
    name: Any = None,
    *,
    solid: bool = True,
    aesthetics: Union[str, List[str]] = "shape",
    **kwargs: Any,
) -> ScaleBinned:
    """Binned shape scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    solid : bool
        Whether shapes are solid.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=pal_shape(solid),
        name=name,
        **kwargs,
    )


def scale_shape_continuous(**kwargs: Any) -> None:
    """Raise an error -- continuous data cannot be mapped to shape.

    Raises
    ------
    ValueError
    """
    cli_abort(
        "A continuous variable cannot be mapped to the shape aesthetic. "
        "Choose a different aesthetic or use scale_shape_binned()."
    )


def scale_shape_identity(
    name: Any = None,
    *,
    guide: str = "none",
    aesthetics: Union[str, List[str]] = "shape",
    **kwargs: Any,
) -> ScaleContinuousIdentity:
    """Shape identity scale.

    Parameters
    ----------
    name, guide, aesthetics
        Standard parameters.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuousIdentity
    """
    return continuous_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleContinuousIdentity,
        **kwargs,
    )


def scale_shape_manual(
    *,
    values: Any,
    breaks: Any = None,
    na_value: Any = np.nan,
    aesthetics: Union[str, List[str]] = "shape",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Shape manual scale.

    Parameters
    ----------
    values, breaks, na_value, aesthetics
        Standard parameters.
    **kwargs
        Passed to ``_manual_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if breaks is None:
        breaks = waiver()
    return _manual_scale(aesthetics, values, breaks, na_value=na_value, **kwargs)


def scale_shape_ordinal(
    name: Any = None,
    **kwargs: Any,
) -> ScaleDiscrete:
    """Ordinal shape scale (not recommended).

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    **kwargs
        Passed to :func:`scale_shape`.

    Returns
    -------
    ScaleDiscrete
    """
    cli_warn("Using shapes for an ordinal variable is not advised.")
    return scale_shape(name=name, **kwargs)


# =========================================================================
# Linetype scales
# =========================================================================

def scale_linetype(
    name: Any = None,
    *,
    aesthetics: Union[str, List[str]] = "linetype",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Discrete linetype scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    return discrete_scale(
        aesthetics,
        palette=None,
        name=name,
        fallback_palette=pal_linetype(),
        **kwargs,
    )


scale_linetype_discrete = scale_linetype


def scale_linetype_binned(
    name: Any = None,
    *,
    aesthetics: Union[str, List[str]] = "linetype",
    **kwargs: Any,
) -> ScaleBinned:
    """Binned linetype scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`binned_scale`.

    Returns
    -------
    ScaleBinned
    """
    return binned_scale(
        aesthetics,
        palette=None,
        name=name,
        fallback_palette=pal_linetype(),
        **kwargs,
    )


def scale_linetype_continuous(**kwargs: Any) -> None:
    """Raise an error -- continuous data cannot be mapped to linetype.

    Raises
    ------
    ValueError
    """
    cli_abort(
        "A continuous variable cannot be mapped to the linetype aesthetic. "
        "Choose a different aesthetic or use scale_linetype_binned()."
    )


def scale_linetype_identity(
    name: Any = None,
    *,
    guide: str = "none",
    aesthetics: Union[str, List[str]] = "linetype",
    **kwargs: Any,
) -> ScaleDiscreteIdentity:
    """Linetype identity scale.

    Parameters
    ----------
    name, guide, aesthetics
        Standard parameters.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscreteIdentity
    """
    return discrete_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleDiscreteIdentity,
        **kwargs,
    )


def scale_linetype_manual(
    *,
    values: Any,
    breaks: Any = None,
    na_value: Any = np.nan,
    aesthetics: Union[str, List[str]] = "linetype",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Linetype manual scale.

    Parameters
    ----------
    values, breaks, na_value, aesthetics
        Standard parameters.
    **kwargs
        Passed to ``_manual_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if breaks is None:
        breaks = waiver()
    return _manual_scale(aesthetics, values, breaks, na_value=na_value, **kwargs)


# =========================================================================
# Linewidth scales
# =========================================================================

def scale_linewidth_continuous(
    name: Any = None,
    *,
    breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    range: Optional[Sequence[float]] = None,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: str = "legend",
    aesthetics: Union[str, List[str]] = "linewidth",
) -> ScaleContinuous:
    """Continuous linewidth scale.

    Parameters
    ----------
    name, breaks, labels, limits, range, transform, trans, guide, aesthetics
        Standard parameters.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_rescale(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        transform=transform,
        trans=trans,
        guide=guide,
        fallback_palette=pal_rescale((1, 6)),
    )


scale_linewidth = scale_linewidth_continuous


def scale_linewidth_binned(
    name: Any = None,
    *,
    breaks: Any = None,
    labels: Any = None,
    limits: Optional[Any] = None,
    range: Optional[Sequence[float]] = None,
    n_breaks: Optional[int] = None,
    nice_breaks: bool = True,
    transform: Union[str, Any] = "identity",
    trans: Optional[Any] = None,
    guide: str = "bins",
    aesthetics: Union[str, List[str]] = "linewidth",
) -> ScaleBinned:
    """Binned linewidth scale.

    Parameters
    ----------
    name, breaks, labels, limits, range, n_breaks, nice_breaks,
    transform, trans, guide, aesthetics
        Standard parameters.

    Returns
    -------
    ScaleBinned
    """
    palette = pal_rescale(range) if range is not None else None
    return binned_scale(
        aesthetics,
        palette=palette,
        name=name,
        breaks=breaks,
        labels=labels,
        limits=limits,
        transform=transform,
        trans=trans,
        n_breaks=n_breaks,
        nice_breaks=nice_breaks,
        guide=guide,
        fallback_palette=pal_rescale((1, 6)),
    )


def scale_linewidth_discrete(
    name: Any = None,
    **kwargs: Any,
) -> ScaleDiscrete:
    """Discrete linewidth scale (not recommended).

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    **kwargs
        Passed to :func:`scale_linewidth_ordinal`.

    Returns
    -------
    ScaleDiscrete
    """
    cli_warn("Using linewidth for a discrete variable is not advised.")
    return scale_linewidth_ordinal(name=name, **kwargs)


def scale_linewidth_ordinal(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "linewidth",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Ordinal linewidth scale.

    Parameters
    ----------
    name : str or Waiver, optional
        Scale title.
    range : tuple of float, optional
        Linewidth range.
    aesthetics : str or list of str
        Aesthetic names.
    **kwargs
        Passed to :func:`discrete_scale`.

    Returns
    -------
    ScaleDiscrete
    """
    if range is not None:
        _r = range

        def palette(n: int) -> List[float]:
            return list(np.linspace(_r[0], _r[1], n))
    else:
        palette = None
    return discrete_scale(
        aesthetics,
        palette=palette,
        name=name,
        fallback_palette=lambda n: list(np.linspace(2, 6, n)),
        **kwargs,
    )


def scale_linewidth_identity(
    name: Any = None,
    *,
    guide: str = "none",
    aesthetics: Union[str, List[str]] = "linewidth",
    **kwargs: Any,
) -> ScaleContinuousIdentity:
    """Linewidth identity scale.

    Parameters
    ----------
    name, guide, aesthetics
        Standard parameters.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuousIdentity
    """
    return continuous_scale(
        aesthetics,
        palette=pal_identity(),
        name=name,
        guide=guide,
        super_class=ScaleContinuousIdentity,
        **kwargs,
    )


def scale_linewidth_manual(
    *,
    values: Any,
    breaks: Any = None,
    na_value: Any = np.nan,
    aesthetics: Union[str, List[str]] = "linewidth",
    **kwargs: Any,
) -> ScaleDiscrete:
    """Linewidth manual scale.

    Parameters
    ----------
    values, breaks, na_value, aesthetics
        Standard parameters.
    **kwargs
        Passed to ``_manual_scale``.

    Returns
    -------
    ScaleDiscrete
    """
    if breaks is None:
        breaks = waiver()
    return _manual_scale(aesthetics, values, breaks, na_value=na_value, **kwargs)


def scale_linewidth_date(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "linewidth",
    **kwargs: Any,
) -> ScaleContinuous:
    """Linewidth date scale.

    Parameters
    ----------
    name, range, aesthetics
        Standard parameters.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_rescale(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        transform=transform_date(),
        fallback_palette=pal_rescale((1, 6)),
        **kwargs,
    )


def scale_linewidth_datetime(
    name: Any = None,
    *,
    range: Optional[Sequence[float]] = None,
    aesthetics: Union[str, List[str]] = "linewidth",
    **kwargs: Any,
) -> ScaleContinuous:
    """Linewidth datetime scale.

    Parameters
    ----------
    name, range, aesthetics
        Standard parameters.
    **kwargs
        Passed to :func:`continuous_scale`.

    Returns
    -------
    ScaleContinuous
    """
    palette = pal_rescale(range) if range is not None else None
    return continuous_scale(
        aesthetics,
        palette=palette,
        name=name,
        transform=transform_time(),
        fallback_palette=pal_rescale((1, 6)),
        **kwargs,
    )


# =========================================================================
# American spelling aliases (color -> colour)
# =========================================================================

scale_color_continuous = scale_colour_continuous
scale_color_discrete = scale_colour_discrete
scale_color_gradient = scale_colour_gradient
scale_color_gradient2 = scale_colour_gradient2
scale_color_gradientn = scale_colour_gradientn
scale_color_hue = scale_colour_hue
scale_color_brewer = scale_colour_brewer
scale_color_distiller = scale_colour_distiller
scale_color_fermenter = scale_colour_fermenter
scale_color_grey = scale_colour_grey
scale_color_viridis_c = scale_colour_viridis_c
scale_color_viridis_d = scale_colour_viridis_d
scale_color_viridis_b = scale_colour_viridis_b
scale_color_binned = scale_colour_binned
scale_color_steps = scale_colour_steps
scale_color_steps2 = scale_colour_steps2
scale_color_stepsn = scale_colour_stepsn
scale_color_identity = scale_colour_identity
scale_color_manual = scale_colour_manual
scale_color_date = scale_colour_date
scale_color_datetime = scale_colour_datetime
scale_color_ordinal = scale_colour_ordinal

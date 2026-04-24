"""
Aesthetic mapping system for ggplot2_py.

This module implements the core aesthetic-mapping machinery that translates
user-specified column references (and computed-variable references) into a
structured :class:`Mapping` object consumed by layers, scales, and stats.

In R's ggplot2, ``aes()`` uses non-standard evaluation (quosures).  In this
Python port we use plain strings for column names and the helper classes
:class:`AfterStat`, :class:`AfterScale`, and :class:`Stage` for deferred
references.

Examples
--------
>>> from ggplot2_py.aes import aes, after_stat, after_scale, stage
>>> aes(x="displ", y="hwy", colour="class")
>>> aes(x="displ", y=after_stat("count"))
>>> aes(colour=stage(start="class", after_scale="fill"))
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

__all__ = [
    "aes",
    "after_stat",
    "after_scale",
    "stage",
    "vars",
    "is_mapping",
    "standardise_aes_names",
    "rename_aes",
    "Mapping",
    "AfterStat",
    "AfterScale",
    "Stage",
    "AESTHETIC_ALIASES",
    "eval_aes_value",
]

# ---------------------------------------------------------------------------
# Aesthetic-alias lookup (R name -> canonical ggplot2 name)
# ---------------------------------------------------------------------------

AESTHETIC_ALIASES: Dict[str, str] = {
    "col":   "colour",
    "color": "colour",
    "pch":   "shape",
    "cex":   "size",
    "lty":   "linetype",
    "lwd":   "linewidth",
    "srt":   "angle",
    "adj":   "hjust",
    "bg":    "fill",
    "fg":    "colour",
    "min":   "ymin",
    "max":   "ymax",
}
"""Port of R ``ggplot_global$base_to_ggplot`` (ggplot-global.R:31-44)."""

# ---------------------------------------------------------------------------
# Helper classes for deferred / staged aesthetics
# ---------------------------------------------------------------------------


class AfterStat:
    """Reference to a variable computed by the stat layer.

    Parameters
    ----------
    x : str or callable
        Name of the computed-stat variable (e.g. ``"count"``, ``"density"``),
        or a callable ``f(data: DataFrame) -> array`` evaluated after stat
        computation.

    Examples
    --------
    >>> AfterStat("count")
    AfterStat('count')
    >>> AfterStat(lambda d: d["count"] / d["count"].max())
    AfterStat(<lambda>)
    """

    __slots__ = ("x",)

    def __init__(self, x: Union[str, Callable[..., Any]]) -> None:
        if not isinstance(x, str) and not callable(x):
            raise TypeError(
                f"AfterStat expects a str or callable, got {type(x).__name__}"
            )
        self.x = x

    # -- repr / eq / hash ---------------------------------------------------

    def __repr__(self) -> str:
        if callable(self.x) and not isinstance(self.x, str):
            name = getattr(self.x, "__name__", "<callable>")
            return f"AfterStat({name})"
        return f"AfterStat({self.x!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AfterStat):
            return self.x == other.x
        return NotImplemented

    def __hash__(self) -> int:
        if callable(self.x) and not isinstance(self.x, str):
            return hash(("AfterStat", id(self.x)))
        return hash(("AfterStat", self.x))


class AfterScale:
    """Reference to a variable available after scale transformation.

    Parameters
    ----------
    x : str or callable
        Name of the post-scale variable (e.g. ``"fill"``), or a callable
        ``f(data: DataFrame) -> array`` evaluated after scale mapping.

    Examples
    --------
    >>> AfterScale("fill")
    AfterScale('fill')
    >>> AfterScale(lambda d: alpha(d["colour"], 0.3))
    AfterScale(<lambda>)
    """

    __slots__ = ("x",)

    def __init__(self, x: Union[str, Callable[..., Any]]) -> None:
        if not isinstance(x, str) and not callable(x):
            raise TypeError(
                f"AfterScale expects a str or callable, got {type(x).__name__}"
            )
        self.x = x

    def __repr__(self) -> str:
        if callable(self.x) and not isinstance(self.x, str):
            name = getattr(self.x, "__name__", "<callable>")
            return f"AfterScale({name})"
        return f"AfterScale({self.x!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AfterScale):
            return self.x == other.x
        return NotImplemented

    def __hash__(self) -> int:
        if callable(self.x) and not isinstance(self.x, str):
            return hash(("AfterScale", id(self.x)))
        return hash(("AfterScale", self.x))


class Stage:
    """Staged aesthetic with potentially different values at each pipeline stage.

    At most one of *start*, *after_stat*, or *after_scale* should be the
    "primary" mapping; the others provide overrides for later stages.

    Each slot accepts a column-name string, a callable
    ``f(data: DataFrame) -> array``, or a wrapper object.

    Parameters
    ----------
    start : str, callable, or None, optional
        Column name or callable used at the initial data stage.
    after_stat : str, callable, AfterStat, or None, optional
        Reference used after the stat computation.
    after_scale : str, callable, AfterScale, or None, optional
        Reference used after scale transformation.

    Examples
    --------
    >>> Stage(start="class", after_scale="fill")
    Stage(start='class', after_stat=None, after_scale='fill')
    >>> Stage(after_stat=lambda d: d["count"] / d["count"].max())
    Stage(start=None, after_stat=AfterStat(<lambda>), after_scale=None)
    """

    __slots__ = ("start", "after_stat", "after_scale")

    def __init__(
        self,
        start: Optional[Union[str, Callable[..., Any]]] = None,
        after_stat: Optional[Union[str, Callable[..., Any], AfterStat]] = None,
        after_scale: Optional[Union[str, Callable[..., Any], AfterScale]] = None,
    ) -> None:
        self.start = start

        # Normalise bare strings and callables to wrapper objects.
        if isinstance(after_stat, str) or (
            callable(after_stat) and not isinstance(after_stat, AfterStat)
        ):
            after_stat = AfterStat(after_stat)
        self.after_stat: Optional[AfterStat] = after_stat  # type: ignore[assignment]

        if isinstance(after_scale, str) or (
            callable(after_scale) and not isinstance(after_scale, AfterScale)
        ):
            after_scale = AfterScale(after_scale)
        self.after_scale: Optional[AfterScale] = after_scale  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Stage(start={self.start!r}, after_stat={self.after_stat!r}, "
            f"after_scale={self.after_scale!r})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Stage):
            return (
                self.start == other.start
                and self.after_stat == other.after_stat
                and self.after_scale == other.after_scale
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("Stage", self.start, self.after_stat, self.after_scale))


# ---------------------------------------------------------------------------
# Mapping class (dict subclass)
# ---------------------------------------------------------------------------

#: Type alias for a single aesthetic value.
AesValue = Union[str, AfterStat, AfterScale, Stage, Callable[..., Any], int, float]


class Mapping(dict):
    """Aesthetic mapping object, analogous to the output of R's ``aes()``.

    A thin :class:`dict` subclass whose keys are canonical aesthetic names
    (e.g. ``"x"``, ``"colour"``) and whose values describe how to map data
    to that aesthetic.

    Values may be:

    * **str** — column-name reference (most common).
    * **AfterStat** — computed variable from a stat layer.
    * **AfterScale** — variable available after scale transformation.
    * **Stage** — staged aesthetic with per-stage overrides.
    * **callable** — a lambda or function applied to the data.
    * **scalar** (int, float, str) — constant aesthetic value.

    Examples
    --------
    >>> m = Mapping(x="displ", y="hwy", colour="class")
    >>> m["x"]
    'displ'
    """

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"aes({inner})"

    # Convenience: attribute access mirrors key access (read-only).
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from None


# ---------------------------------------------------------------------------
# Centralised aesthetic-value evaluator
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd


def eval_aes_value(
    val: Any,
    data: "pd.DataFrame",
) -> Any:
    """Evaluate a single aesthetic value against a DataFrame.

    This is the Python equivalent of R's ``eval_tidy(quo, data)`` pattern.
    It handles the three kinds of aesthetic values that appear in
    :class:`Mapping` objects:

    * **str** — looked up as a column name in *data*.
    * **callable** — called with *data* as the single argument.
    * **scalar / other** — returned unchanged (will be broadcast by the
      caller).

    Parameters
    ----------
    val : str, callable, or scalar
        The aesthetic value to evaluate.
    data : DataFrame
        The layer data to evaluate against.

    Returns
    -------
    numpy array, scalar, or ``None``
        ``None`` is returned when *val* is a string that does not match any
        column in *data* (the column may appear in a later pipeline stage).
    """
    if callable(val) and not isinstance(val, (str, type)):
        result = val(data)
        if isinstance(result, pd.Series):
            return result.values
        return result
    elif isinstance(val, str):
        if val in data.columns:
            return data[val].values
        return None  # column not yet available
    else:
        return val  # scalar — broadcast by caller


# ---------------------------------------------------------------------------
# Public convenience constructors
# ---------------------------------------------------------------------------


def after_stat(x: Union[str, Callable[..., Any]]) -> AfterStat:
    """Create an :class:`AfterStat` reference.

    Parameters
    ----------
    x : str or callable
        Name of a stat-computed variable (e.g. ``"count"``), or a callable
        ``f(data) -> array`` to be evaluated after stat computation.

    Returns
    -------
    AfterStat

    Examples
    --------
    >>> after_stat("density")
    AfterStat('density')
    >>> after_stat(lambda d: d["count"] / d["count"].max())
    AfterStat(<lambda>)
    """
    return AfterStat(x)


def after_scale(x: Union[str, Callable[..., Any]]) -> AfterScale:
    """Create an :class:`AfterScale` reference.

    Parameters
    ----------
    x : str or callable
        Name of a post-scale variable (e.g. ``"fill"``), or a callable
        ``f(data) -> array`` to be evaluated after scale mapping.

    Returns
    -------
    AfterScale

    Examples
    --------
    >>> after_scale("fill")
    AfterScale('fill')
    >>> after_scale(lambda d: d["colour"].str.replace("FF", "80"))
    AfterScale(<lambda>)
    """
    return AfterScale(x)


def stage(
    start: Optional[Union[str, Callable[..., Any]]] = None,
    after_stat: Optional[Union[str, Callable[..., Any], AfterStat]] = None,
    after_scale: Optional[Union[str, Callable[..., Any], AfterScale]] = None,
) -> Stage:
    """Create a :class:`Stage` aesthetic with per-stage overrides.

    Parameters
    ----------
    start : str, callable, or None, optional
        Column name or callable used at the initial data stage.
    after_stat : str, callable, AfterStat, or None, optional
        Reference used after the stat computation.
    after_scale : str, callable, AfterScale, or None, optional
        Reference used after scale transformation.

    Returns
    -------
    Stage

    Examples
    --------
    >>> stage(start="class", after_scale="fill")
    Stage(start='class', after_stat=None, after_scale='fill')
    """
    return Stage(start=start, after_stat=after_stat, after_scale=after_scale)


# ---------------------------------------------------------------------------
# aes() — main entry point
# ---------------------------------------------------------------------------


def aes(
    x: Optional[AesValue] = None,
    y: Optional[AesValue] = None,
    **kwargs: AesValue,
) -> Mapping:
    """Create an aesthetic mapping.

    This is the Python equivalent of R's ``aes()`` function.  Column
    references are passed as plain strings; deferred references use
    :func:`after_stat`, :func:`after_scale`, or :func:`stage`.

    Parameters
    ----------
    x : str, callable, or scalar, optional
        Mapping for the *x* aesthetic.
    y : str, callable, or scalar, optional
        Mapping for the *y* aesthetic.
    **kwargs
        Additional aesthetic mappings.  Names are automatically
        standardised (e.g. ``color`` becomes ``colour``).

    Returns
    -------
    Mapping
        An aesthetic-mapping dictionary.

    Examples
    --------
    >>> aes(x="displ", y="hwy")
    aes(x='displ', y='hwy')

    >>> aes(x="displ", y="hwy", color="class")
    aes(x='displ', y='hwy', colour='class')

    >>> aes(x="displ", y=after_stat("count"))
    aes(x='displ', y=AfterStat('count'))
    """
    raw: Dict[str, AesValue] = {}
    if x is not None:
        raw["x"] = x
    if y is not None:
        raw["y"] = y
    raw.update(kwargs)

    # Standardise names (e.g. "color" -> "colour"). R's ``aes()`` also
    # ``cli::cli_abort``s on duplicate x/y (aes.R:103-106). Python's
    # ``**kwargs`` makes literal duplicates a ``SyntaxError`` at parse
    # time, but ``color=`` + ``colour=`` both collapse to ``colour`` — we
    # catch that post-standardisation because both refer to the same
    # aesthetic.
    result = Mapping()
    seen_aliases: Dict[str, str] = {}
    for key, value in raw.items():
        canonical = _standardise_single(key)
        if canonical in seen_aliases and seen_aliases[canonical] != key:
            from ggplot2_py._compat import cli_abort
            cli_abort(
                f"Duplicate aesthetic {canonical!r} — passed as both "
                f"{seen_aliases[canonical]!r} and {key!r}."
            )
        seen_aliases[canonical] = key
        result[canonical] = value

    return result


# ---------------------------------------------------------------------------
# Name standardisation
# ---------------------------------------------------------------------------


def _standardise_single(name: str) -> str:
    """Canonicalise a single aesthetic name.

    Port of R ``standardise_aes_names`` (aes.R:180-191) — two steps:

    1. ``sub("color", "colour", x, fixed = TRUE)`` — first-occurrence
       substring rewrite (so ``segment_color`` → ``segment_colour``,
       ``bg.color`` → ``bg.colour``, ``color_palette`` → ``colour_palette``).
    2. Whole-name lookup in ``ggplot_global$base_to_ggplot``
       (:data:`AESTHETIC_ALIASES`), which maps R-style names like ``pch``,
       ``cex``, ``srt``, ``fg`` to their ggplot2 canonical names.
    """
    # Step 1 — R uses `sub(..., fixed = TRUE)` which replaces only the
    # first occurrence; `str.replace(..., 1)` matches that semantics.
    if "color" in name:
        name = name.replace("color", "colour", 1)
    # Step 2 — whole-name alias lookup.
    return AESTHETIC_ALIASES.get(name, name)


def standardise_aes_names(aes_names: Iterable[str]) -> List[str]:
    """Standardise a sequence of aesthetic names to their canonical forms.

    Parameters
    ----------
    aes_names : Iterable[str]
        Aesthetic names, possibly containing aliases.

    Returns
    -------
    list of str
        Canonical aesthetic names in the same order.

    Examples
    --------
    >>> standardise_aes_names(["color", "lwd", "x"])
    ['colour', 'linewidth', 'x']
    """
    return [_standardise_single(n) for n in aes_names]


def rename_aes(mapping: Mapping) -> Mapping:
    """Return a copy of *mapping* with all keys standardised.

    Port of R ``rename_aes`` (aes.R:193-201): standardises every key of
    an aesthetic mapping and warns when collapsing duplicates (e.g. when
    both ``color`` and ``colour`` are supplied).

    Parameters
    ----------
    mapping : Mapping
        An aesthetic mapping (or plain dict).

    Returns
    -------
    Mapping
        New mapping with canonical aesthetic names as keys.

    Examples
    --------
    >>> rename_aes(Mapping(color="class"))
    aes(colour='class')
    """
    items = [(_standardise_single(k), v) for k, v in mapping.items()]
    # R warns on duplicated names after standardisation.
    seen: Dict[str, int] = {}
    for k, _ in items:
        seen[k] = seen.get(k, 0) + 1
    dupes = sorted({k for k, c in seen.items() if c > 1})
    if dupes:
        from ggplot2_py._compat import cli_warn as _cli_warn
        _cli_warn(
            "Duplicated aesthetics after name standardisation: "
            + ", ".join(dupes)
        )
    # Later entry wins (dict semantics) — matches R's ``names(x) <- ...``.
    return Mapping(dict(items))


# ---------------------------------------------------------------------------
# vars() helper (for facets)
# ---------------------------------------------------------------------------


def vars(*args: str, **kwargs: str) -> List[str]:
    """Specify variables for faceting.

    A thin helper that collects positional and keyword variable names into a
    flat list of strings, suitable for passing to ``facet_wrap`` or
    ``facet_grid``.

    Parameters
    ----------
    *args : str
        Variable names given positionally.
    **kwargs : str
        Variable names given as keyword arguments (values are used, keys
        are ignored).

    Returns
    -------
    list of str
        Flat list of variable-name strings.

    Examples
    --------
    >>> vars("cyl", "drv")
    ['cyl', 'drv']

    >>> vars(rows="cyl")
    ['cyl']
    """
    result: List[str] = list(args)
    result.extend(kwargs.values())
    return result


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------


def is_mapping(x: Any) -> bool:
    """Test whether *x* is an aesthetic :class:`Mapping`.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a :class:`Mapping` instance.

    Examples
    --------
    >>> is_mapping(aes(x="a"))
    True
    >>> is_mapping({"x": "a"})
    False
    """
    return isinstance(x, Mapping)

"""
Internal utilities for ggplot2.

Replaces functionality from ``utilities.R``, ``compat-plyr.R``, ``bin.R``,
and ``grouping.R`` in the R package.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd

__all__ = [
    "remove_missing",
    "resolution",
    "snake_class",
    "has_groups",
    "empty",
    "is_empty",
    "try_fetch",
    "compact",
    "modify_list",
    "data_frame",
    "unique_default",
    "rename",
    "id_var",
    "plyr_id",
    "interleave",
    "width_cm",
    "height_cm",
    "stapled_to_list",
]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Missing-data helpers
# ---------------------------------------------------------------------------

def remove_missing(
    df: pd.DataFrame,
    vars: Optional[List[str]] = None,
    na_rm: bool = False,
    name: str = "",
    finite: bool = False,
) -> pd.DataFrame:
    """Remove rows with missing (or non-finite) values from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    vars : list of str, optional
        Column names to check.  If ``None``, all columns are checked.
    na_rm : bool, optional
        If ``False``, emit a warning when rows are removed but still
        remove them.  If ``True``, remove silently.  Default ``False``.
    name : str, optional
        Name of the calling layer (used in warning messages).
    finite : bool, optional
        If ``True``, also remove rows where checked columns contain
        ``inf`` / ``-inf``.  Default ``False``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with offending rows removed.
    """
    if df.empty:
        return df

    if vars is None:
        check_cols = list(df.columns)
    else:
        check_cols = [c for c in vars if c in df.columns]

    if not check_cols:
        return df

    if finite:
        # Check for NA *and* non-finite in numeric columns.
        mask = pd.Series(True, index=df.index)
        for col in check_cols:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                mask = mask & np.isfinite(s.to_numpy(dtype=float, na_value=np.nan))
            else:
                mask = mask & s.notna()
    else:
        mask = df[check_cols].notna().all(axis=1)

    n_removed = int((~mask).sum())
    if n_removed > 0 and not na_rm:
        qual = "non-finite" if finite else "missing"
        where = f" ({name})" if name else ""
        warnings.warn(
            f"Removed {n_removed} rows containing {qual} values{where}.",
            UserWarning,
            stacklevel=2,
        )

    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def resolution(x: Any, zero: bool = True, discrete: bool = False) -> float:
    """Compute the resolution of a numeric vector.

    The resolution is the smallest non-zero difference between adjacent
    unique sorted values.  This is useful for choosing default bin widths.

    Parameters
    ----------
    x : array-like
        Numeric values.
    zero : bool, optional
        If ``True`` (default), include zero in the set of differences so
        that the result is at most 1 when there is only one unique value.

    Returns
    -------
    float
        The resolution.
    """
    if discrete:
        return 1.0
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0

    x = np.sort(np.unique(x))
    if len(x) == 1:
        return 1.0 if zero else 0.0

    diffs = np.diff(x)
    if zero:
        diffs = np.concatenate([[0.0], diffs])
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.min(diffs))


# ---------------------------------------------------------------------------
# String / naming helpers
# ---------------------------------------------------------------------------

def snake_class(x: Any) -> str:
    """Convert an object's class name to snake_case.

    Parameters
    ----------
    x : Any
        An object (or class).

    Returns
    -------
    str
        The class name in ``snake_case``.

    Examples
    --------
    >>> snake_class(pd.DataFrame())
    'data_frame'
    """
    name = type(x).__name__ if not isinstance(x, type) else x.__name__
    # Insert underscore before uppercase letters that follow a lowercase
    # letter or another uppercase followed by a lowercase.
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s2 = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def rename(
    x: Dict[str, Any],
    mapping: Optional[Dict[str, str]] = None,
    **kwargs: str,
) -> Dict[str, Any]:
    """Rename keys in a dictionary.

    Parameters
    ----------
    x : dict
        Original dictionary.
    mapping : dict, optional
        ``{old_name: new_name}`` pairs.
    **kwargs : str
        Additional ``old_name=new_name`` pairs (convenience).

    Returns
    -------
    dict
        A new dictionary with renamed keys.
    """
    if mapping is None:
        mapping = {}
    mapping.update(kwargs)
    return {mapping.get(k, k): v for k, v in x.items()}


# ---------------------------------------------------------------------------
# DataFrame / grouping utilities
# ---------------------------------------------------------------------------

def has_groups(df: pd.DataFrame) -> bool:
    """Check whether a DataFrame has a ``"group"`` column with >1 group.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    bool
    """
    if "group" not in df.columns:
        return False
    return int(df["group"].nunique()) > 1


def empty(df: pd.DataFrame) -> bool:
    """Check whether a DataFrame is conceptually empty.

    A DataFrame is considered empty if it has zero rows **or** zero
    columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    bool
    """
    return df.shape[0] == 0 or df.shape[1] == 0


def is_empty(x: Any) -> bool:
    """Check whether an object is empty.

    Parameters
    ----------
    x : Any
        Object to test.  Works with DataFrames, dicts, lists, and other
        sized objects.

    Returns
    -------
    bool
    """
    if isinstance(x, pd.DataFrame):
        return empty(x)
    if x is None:
        return True
    try:
        return len(x) == 0  # type: ignore[arg-type]
    except TypeError:
        return False


def data_frame(**kwargs: Any) -> pd.DataFrame:
    """Create a ``pd.DataFrame`` from keyword arguments.

    Each keyword becomes a column.  Scalar values are broadcast to match
    the length of the longest array-like argument.

    Parameters
    ----------
    **kwargs : Any
        Column name/value pairs.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(kwargs)


def unique_default(x: Any) -> np.ndarray:
    """Return unique values preserving the order of first occurrence.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x)
    _, idx = np.unique(x, return_index=True)
    return x[np.sort(idx)]


def id_var(x: Any) -> np.ndarray:
    """Compute integer group IDs for a single variable.

    Parameters
    ----------
    x : array-like
        Values (may be numeric, string, or categorical).

    Returns
    -------
    np.ndarray
        Integer array of 1-based group IDs, one per element.
    """
    x = np.asarray(x)
    uniques, inverse = np.unique(x, return_inverse=True)
    return inverse + 1  # 1-based to match R convention


def plyr_id(
    df: pd.DataFrame,
    drop: bool = False,
) -> np.ndarray:
    """Compute interaction-style group IDs across multiple columns.

    Mimics ``plyr::id()`` — each unique combination of values across all
    columns of *df* receives a unique integer ID (1-based).

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns to interact.
    drop : bool, optional
        If ``True``, re-number IDs so that unused combinations are
        removed.  Default ``False``.

    Returns
    -------
    np.ndarray
        1-based integer group IDs with length ``len(df)``.
    """
    if df.shape[1] == 0 or df.shape[0] == 0:
        return np.ones(len(df), dtype=int)

    # Build a single interaction key using tuples.
    cols = [df.iloc[:, i].to_numpy() for i in range(df.shape[1])]
    keys = list(zip(*cols))

    # Map each unique key to an integer.
    seen: Dict[Any, int] = {}
    ids = np.empty(len(keys), dtype=int)
    counter = 0
    for i, k in enumerate(keys):
        if k not in seen:
            counter += 1
            seen[k] = counter
        ids[i] = seen[k]
    return ids


# ---------------------------------------------------------------------------
# Dict / list helpers
# ---------------------------------------------------------------------------

def try_fetch(expr: Any, default: Any = None) -> Any:
    """Execute a callable and return *default* on any exception.

    If *expr* is not callable it is returned as-is.

    Parameters
    ----------
    expr : callable or Any
        A zero-argument callable, or a plain value.
    default : Any, optional
        Value to return if *expr* raises.

    Returns
    -------
    Any
    """
    if callable(expr):
        try:
            return expr()
        except Exception:
            return default
    return expr


def compact(x: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ``None`` values from a dictionary.

    Parameters
    ----------
    x : dict
        Input dictionary.

    Returns
    -------
    dict
        A new dictionary with all ``None``-valued entries removed.
    """
    return {k: v for k, v in x.items() if v is not None}


def modify_list(
    old: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge two dictionaries, with *new* overriding *old*.

    Parameters
    ----------
    old : dict
        Base dictionary.
    new : dict
        Override dictionary.  Keys whose values are ``None`` will
        **remove** the corresponding key from the result (mirroring
        ``utils::modifyList`` in R).

    Returns
    -------
    dict
        Merged dictionary (a fresh copy).
    """
    result = dict(old)
    for k, v in new.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def interleave(*args: Sequence[Any]) -> list:
    """Interleave elements from multiple sequences.

    Parameters
    ----------
    *args : Sequence
        Input sequences.  They should all have the same length.

    Returns
    -------
    list
        A flat list with elements taken round-robin from each input.

    Examples
    --------
    >>> interleave([1, 2, 3], [10, 20, 30])
    [1, 10, 2, 20, 3, 30]
    """
    if not args:
        return []
    result: list = []
    max_len = max(len(a) for a in args)
    for i in range(max_len):
        for a in args:
            if i < len(a):
                result.append(a[i])
    return result


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def width_cm(x: Any) -> Union[float, np.ndarray]:
    """Convert a grid unit to centimetres (width / x-axis).

    Parameters
    ----------
    x : Unit or numeric
        A ``grid_py.Unit`` object or a plain number (assumed to be in
        centimetres already).

    Returns
    -------
    float or np.ndarray
        The width in centimetres.
    """
    from grid_py import Unit, convert_unit

    if isinstance(x, Unit):
        result = convert_unit(x, "cm", axisFrom="x", typeFrom="dimension", valueOnly=True)
        return result
    return float(x) if np.isscalar(x) else np.asarray(x, dtype=float)


def height_cm(x: Any) -> Union[float, np.ndarray]:
    """Convert a grid unit to centimetres (height / y-axis).

    Parameters
    ----------
    x : Unit or numeric
        A ``grid_py.Unit`` object or a plain number (assumed to be in
        centimetres already).

    Returns
    -------
    float or np.ndarray
        The height in centimetres.
    """
    from grid_py import Unit, convert_unit

    if isinstance(x, Unit):
        result = convert_unit(x, "cm", axisFrom="y", typeFrom="dimension", valueOnly=True)
        return result
    return float(x) if np.isscalar(x) else np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def stapled_to_list(x: Any) -> list:
    """Convert a "stapled" object to a plain list.

    In R, ``vctrs::vec_proxy()`` sometimes returns stapled vectors.  In
    Python this is a no-op — if *x* is already a list it is returned
    unchanged; otherwise it is wrapped in a list.

    Parameters
    ----------
    x : Any
        Object to convert.

    Returns
    -------
    list
    """
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]

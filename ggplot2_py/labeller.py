"""
Facet labeller functions — faithful port of R's facet-labeller.R.

Labeller functions format the strip labels of facet grids and wraps.
They accept a dict of ``{variable_name: [values...]}`` and return
a list of formatted label strings.

R references
------------
* ``ggplot2/R/facet-labeller.R`` — label_value, label_both, etc.
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable, Dict, List, Optional, Union

__all__ = [
    "label_value",
    "label_both",
    "label_context",
    "label_parsed",
    "label_wrap_gen",
    "as_labeller",
]


# ---------------------------------------------------------------------------
# label_value
# ---------------------------------------------------------------------------

def label_value(
    labels: Dict[str, List[str]],
    multi_line: bool = True,
) -> List[str]:
    """Return only the factor values.

    This is the default labeller in ggplot2.

    Mirrors ``label_value`` in R (facet-labeller.R:105-112).

    Parameters
    ----------
    labels : dict
        ``{variable_name: [value_per_panel...]}``.
    multi_line : bool
        If ``True`` (default), return one line per variable.
        If ``False``, collapse all variables into one line.

    Returns
    -------
    list of str
        Formatted labels, one per panel.

    Examples
    --------
    >>> label_value({"drv": ["4", "f", "r"]})
    ["4", "f", "r"]
    >>> label_value({"drv": ["4", "f"], "cyl": ["4", "6"]}, multi_line=False)
    ["4, 4", "f, 6"]
    """
    var_names = list(labels.keys())
    if not var_names:
        return []

    n_panels = len(next(iter(labels.values())))

    if multi_line and len(var_names) == 1:
        return [str(v) for v in labels[var_names[0]]]

    if multi_line:
        # Return one label per variable per panel — for multi-line strips,
        # join with newline
        result = []
        for i in range(n_panels):
            parts = [str(labels[v][i]) for v in var_names]
            result.append("\n".join(parts))
        return result
    else:
        # Single line: collapse all variables
        result = []
        for i in range(n_panels):
            parts = [str(labels[v][i]) for v in var_names]
            result.append(", ".join(parts))
        return result


# ---------------------------------------------------------------------------
# label_both
# ---------------------------------------------------------------------------

def label_both(
    labels: Dict[str, List[str]],
    multi_line: bool = True,
    sep: str = ": ",
) -> List[str]:
    """Return ``"variable: value"`` labels.

    Mirrors ``label_both`` in R (facet-labeller.R:119-142).

    Parameters
    ----------
    labels : dict
        ``{variable_name: [value_per_panel...]}``.
    multi_line : bool
        If ``True``, one line per variable; if ``False``, collapse.
    sep : str
        Separator between variable name and value.

    Returns
    -------
    list of str
        Formatted labels.

    Examples
    --------
    >>> label_both({"drv": ["4", "f", "r"]})
    ["drv: 4", "drv: f", "drv: r"]
    """
    var_names = list(labels.keys())
    if not var_names:
        return []

    n_panels = len(next(iter(labels.values())))

    if multi_line:
        result = []
        for i in range(n_panels):
            parts = [f"{v}{sep}{labels[v][i]}" for v in var_names]
            result.append("\n".join(parts))
        return result
    else:
        result = []
        for i in range(n_panels):
            var_part = ", ".join(var_names)
            val_part = ", ".join(str(labels[v][i]) for v in var_names)
            result.append(f"{var_part}{sep}{val_part}")
        return result


# ---------------------------------------------------------------------------
# label_context
# ---------------------------------------------------------------------------

def label_context(
    labels: Dict[str, List[str]],
    multi_line: bool = True,
    sep: str = ": ",
) -> List[str]:
    """Context-dependent labeller.

    Uses ``label_value`` for single-variable faceting, ``label_both``
    when multiple variables are involved.

    Mirrors ``label_context`` in R (facet-labeller.R:147-153).

    Parameters
    ----------
    labels : dict
        ``{variable_name: [value_per_panel...]}``.
    multi_line : bool
        Multi-line mode.
    sep : str
        Separator for ``label_both``.

    Returns
    -------
    list of str
        Formatted labels.
    """
    if len(labels) <= 1:
        return label_value(labels, multi_line)
    else:
        return label_both(labels, multi_line, sep)


# ---------------------------------------------------------------------------
# label_parsed
# ---------------------------------------------------------------------------

def label_parsed(
    labels: Dict[str, List[str]],
    multi_line: bool = True,
) -> List[str]:
    """Interpret labels as expressions (Python: return as-is).

    In R, this parses labels as plotmath expressions.  In Python we
    simply return the string values, since matplotlib mathtext or
    LaTeX rendering is handled downstream by the text grob.

    Mirrors ``label_parsed`` in R (facet-labeller.R:158-173).

    Parameters
    ----------
    labels : dict
        ``{variable_name: [value_per_panel...]}``.
    multi_line : bool
        Multi-line mode.

    Returns
    -------
    list of str
        Labels (unchanged).
    """
    return label_value(labels, multi_line)


# ---------------------------------------------------------------------------
# label_wrap_gen
# ---------------------------------------------------------------------------

def label_wrap_gen(width: int = 25) -> Callable:
    """Generate a labeller that wraps text at *width* characters.

    Mirrors ``label_wrap_gen`` in R (facet-labeller.R:220-229).

    Parameters
    ----------
    width : int
        Maximum number of characters before wrapping.

    Returns
    -------
    callable
        A labeller function.

    Examples
    --------
    >>> labeller = label_wrap_gen(10)
    >>> labeller({"class": ["compact car", "midsize SUV"]})
    ["compact\\ncar", "midsize\\nSUV"]
    """
    def _wrap_labeller(
        labels: Dict[str, List[str]],
        multi_line: bool = True,
    ) -> List[str]:
        base = label_value(labels, multi_line)
        return ["\n".join(textwrap.wrap(str(lab), width)) for lab in base]

    return _wrap_labeller


# ---------------------------------------------------------------------------
# as_labeller
# ---------------------------------------------------------------------------

_LABELLER_REGISTRY: Dict[str, Callable] = {
    "label_value": label_value,
    "label_both": label_both,
    "label_context": label_context,
    "label_parsed": label_parsed,
}


def as_labeller(x: Any) -> Callable:
    """Coerce *x* to a labeller function.

    Mirrors ``as_labeller`` in R (facet-labeller.R:284-299).

    Parameters
    ----------
    x : str, dict, or callable
        - ``str``: lookup by name (e.g. ``"label_value"``).
        - ``dict``: a lookup table mapping values to labels.
        - ``callable``: returned as-is.

    Returns
    -------
    callable
        A labeller function.
    """
    if callable(x):
        return x

    if isinstance(x, str):
        if x in _LABELLER_REGISTRY:
            return _LABELLER_REGISTRY[x]
        raise ValueError(
            f"Unknown labeller: {x!r}. "
            f"Available: {list(_LABELLER_REGISTRY.keys())}"
        )

    if isinstance(x, dict):
        # Lookup-table labeller: maps values to custom labels
        lookup = x

        def _dict_labeller(
            labels: Dict[str, List[str]],
            multi_line: bool = True,
        ) -> List[str]:
            base = label_value(labels, multi_line)
            return [lookup.get(lab, lab) for lab in base]

        return _dict_labeller

    raise TypeError(f"Cannot coerce {type(x).__name__} to a labeller.")

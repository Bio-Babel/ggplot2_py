"""
Label helpers for ggplot2 plots.

Provides ``labs()``, ``xlab()``, ``ylab()``, ``ggtitle()`` for setting
axis titles, plot titles, subtitles, captions, tags, and alt-text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_warn

__all__ = [
    "labs",
    "xlab",
    "ylab",
    "ggtitle",
    "Labels",
    "is_labels",
    "get_labs",
    "update_labels",
    "make_labels",
]


# ---------------------------------------------------------------------------
# Labels container
# ---------------------------------------------------------------------------

class Labels(dict):
    """A thin ``dict`` subclass marking an object as a labels specification.

    Behaves like a regular dictionary but carries a type tag so that the
    ``+`` operator on a ``GGPlot`` object can dispatch correctly.
    """

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"Labels({items})"


def is_labels(x: Any) -> bool:
    """Return ``True`` if *x* is a :class:`Labels` instance.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(x, Labels)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Canonical label keys that are NOT aesthetics.
_SPECIAL_KEYS = frozenset({
    "title", "subtitle", "caption", "tag",
    "alt", "alt_insight", "dictionary",
})


def labs(
    *,
    title: Any = None,
    subtitle: Any = None,
    caption: Any = None,
    tag: Any = None,
    alt: Any = None,
    alt_insight: Any = None,
    dictionary: Any = None,
    **kwargs: Any,
) -> Labels:
    """Create a labels specification.

    Parameters
    ----------
    title : str or None, optional
        Plot title.
    subtitle : str or None, optional
        Plot subtitle (displayed below the title).
    caption : str or None, optional
        Plot caption (typically data source info).
    tag : str or None, optional
        Plot tag (e.g. ``"A"`` for sub-figures).
    alt : str or callable or None, optional
        Alt-text for accessibility.
    alt_insight : str or None, optional
        Short insight appended to auto-generated alt-text.
    dictionary : dict or None, optional
        Named dictionary for label look-ups.
    **kwargs
        Aesthetic-name = label pairs, e.g. ``x="Weight"``, ``colour="Class"``.

    Returns
    -------
    Labels
        A labels specification suitable for adding to a ggplot via ``+``.

    Examples
    --------
    >>> labs(x="Engine displacement", y="Highway MPG", colour="Vehicle class")
    """
    # Collect special keyword args
    args: Dict[str, Any] = {}
    if title is not None:
        args["title"] = title
    if subtitle is not None:
        args["subtitle"] = subtitle
    if caption is not None:
        args["caption"] = caption
    if tag is not None:
        args["tag"] = tag
    if alt is not None:
        args["alt"] = alt
    if alt_insight is not None:
        args["alt_insight"] = alt_insight
    if dictionary is not None:
        args["dictionary"] = dictionary

    # Aesthetic label kwargs -- apply alias normalisation
    from ggplot2_py.aes import standardise_aes_names  # lazy to avoid circular

    for k, v in kwargs.items():
        canonical = standardise_aes_names([k])[0]
        args[canonical] = v

    return Labels(args)


def xlab(label: Optional[str]) -> Labels:
    """Set the x-axis label.

    Parameters
    ----------
    label : str or None
        Label text.  ``None`` removes the label.

    Returns
    -------
    Labels
    """
    return labs(x=label)


def ylab(label: Optional[str]) -> Labels:
    """Set the y-axis label.

    Parameters
    ----------
    label : str or None
        Label text.  ``None`` removes the label.

    Returns
    -------
    Labels
    """
    return labs(y=label)


def ggtitle(
    label: Optional[str],
    subtitle: Optional[str] = None,
) -> Labels:
    """Set the plot title (and optionally subtitle).

    Parameters
    ----------
    label : str or None
        Title text.
    subtitle : str or None, optional
        Subtitle text.

    Returns
    -------
    Labels
    """
    return labs(title=label, subtitle=subtitle)


# ---------------------------------------------------------------------------
# update_labels helper
# ---------------------------------------------------------------------------

def update_labels(plot: Any, labels: Union[Labels, Dict[str, Any]]) -> Any:
    """Merge *labels* into *plot*'s existing labels.

    Parameters
    ----------
    plot : GGPlot
        The plot to update (will be cloned).
    labels : Labels or dict
        New labels to merge in.

    Returns
    -------
    GGPlot
        A shallow copy of *plot* with updated labels.
    """
    # Import here to avoid circular dependency
    p = plot._clone()
    # Merge: new labels override existing
    merged = dict(p.labels)
    merged.update(labels)
    p.labels = Labels(merged)
    return p


# ---------------------------------------------------------------------------
# make_labels
# ---------------------------------------------------------------------------

def make_labels(mapping: Any) -> Dict[str, str]:
    """Convert an aesthetic mapping into text labels.

    Parameters
    ----------
    mapping : Mapping
        An aesthetic mapping (from ``aes()``).

    Returns
    -------
    dict
        A dictionary of aesthetic-name -> label-string.
    """
    from ggplot2_py.aes import Mapping, AfterStat, AfterScale, Stage

    # Accept both ``Mapping`` (from ``aes()``) and plain ``dict``
    # (used by Stat.default_aes / Geom.default_aes).  R's
    # ``make_labels()`` treats any named list uniformly (labels.R:124).
    if isinstance(mapping, Mapping):
        mapping_items = mapping.items()
    elif isinstance(mapping, dict):
        mapping_items = mapping.items()
    else:
        return {}

    def _label_for(val: Any) -> str:
        """Generate a human-readable label for an aesthetic value."""
        if val is None:
            return ""
        if callable(val) and not isinstance(val, str):
            return getattr(val, "__name__", "<expr>")
        return str(val)

    result: Dict[str, str] = {}
    for aes_name, val in mapping_items:
        if val is None:
            result[aes_name] = aes_name
        elif isinstance(val, str):
            result[aes_name] = val
        elif isinstance(val, AfterStat):
            result[aes_name] = _label_for(val.x)
        elif isinstance(val, AfterScale):
            result[aes_name] = _label_for(val.x)
        elif isinstance(val, Stage):
            # Use the start mapping if available, then after_stat, then after_scale
            if val.start is not None:
                result[aes_name] = _label_for(val.start)
            elif val.after_stat is not None:
                result[aes_name] = _label_for(val.after_stat)
            elif val.after_scale is not None:
                result[aes_name] = _label_for(val.after_scale)
            else:
                result[aes_name] = aes_name
        elif callable(val):
            result[aes_name] = _label_for(val)
        else:
            result[aes_name] = str(val)
    return result


# ---------------------------------------------------------------------------
# get_labs (requires a built plot)
# ---------------------------------------------------------------------------

def get_labs(plot: Any = None) -> Labels:
    """Retrieve completed labels from a plot.

    Parameters
    ----------
    plot : GGPlot or None
        A ggplot object.  If ``None``, uses ``get_last_plot()``.

    Returns
    -------
    Labels
        Resolved label dictionary.
    """
    if plot is None:
        # lazy import to avoid circular
        from ggplot2_py.plot import get_last_plot
        plot = get_last_plot()

    if plot is None:
        return Labels()

    # If already built, grab labels directly
    if hasattr(plot, "plot") and hasattr(plot, "layout"):
        # BuiltGGPlot
        return Labels(plot.plot.labels)

    return Labels(plot.labels)

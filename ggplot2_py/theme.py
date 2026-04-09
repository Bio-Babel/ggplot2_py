"""
Theme creation and management for ggplot2.

Provides the ``Theme`` class, the ``theme()`` constructor, global theme
state functions (``theme_get``, ``theme_set``, etc.), and the ``+`` /
``%+replace%`` operators for combining themes.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from ggplot2_py._compat import cli_abort, cli_warn
from ggplot2_py.theme_elements import (
    Element,
    ElementBlank,
    is_theme_element,
    merge_element,
    combine_elements,
    _ggplot_global,
    get_element_tree,
    calc_element,
)

__all__ = [
    "Theme",
    "theme",
    "is_theme",
    "complete_theme",
    "add_theme",
    "theme_get",
    "theme_set",
    "theme_update",
    "theme_replace",
    "set_theme",
    "get_theme",
    "reset_theme_settings",
    "update_theme",
    "replace_theme",
]


# ---------------------------------------------------------------------------
# Theme class
# ---------------------------------------------------------------------------

class Theme:
    """A ggplot2 theme object.

    Internally a ``Theme`` is a dictionary-like container mapping element
    names (strings) to element objects (``Element`` subclasses, ``Unit``,
    scalars, etc.).  It supports ``+`` to merge themes and item access via
    ``[]`` or ``.get()``.

    Parameters
    ----------
    elements : dict
        Mapping of element names to values.
    complete : bool
        ``True`` for complete themes (e.g. ``theme_grey()``).
    validate : bool
        Whether to validate elements against the element tree.
    """

    def __init__(
        self,
        elements: Optional[Dict[str, Any]] = None,
        complete: bool = False,
        validate: bool = True,
    ) -> None:
        self._elements: Dict[str, Any] = dict(elements) if elements else {}
        self.complete = complete
        self.validate = validate

    # -- dict-like access ---------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._elements[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._elements[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._elements

    def __iter__(self):
        return iter(self._elements)

    def __len__(self) -> int:
        return len(self._elements)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the element for *key*, or *default* if absent.

        Parameters
        ----------
        key : str
            Element name.
        default : Any
            Fallback value.

        Returns
        -------
        Any
        """
        return self._elements.get(key, default)

    def keys(self):
        return self._elements.keys()

    def values(self):
        return self._elements.values()

    def items(self):
        return self._elements.items()

    def names(self):
        """Return element names (R-compatible alias for ``keys``)."""
        return list(self._elements.keys())

    def update(self, other: Dict[str, Any]) -> None:
        """Update elements in-place from *other*.

        Parameters
        ----------
        other : dict
            Mapping of element names to values.
        """
        self._elements.update(other)

    def copy(self) -> "Theme":
        """Return a shallow copy of this theme.

        Returns
        -------
        Theme
        """
        return Theme(
            elements=dict(self._elements),
            complete=self.complete,
            validate=self.validate,
        )

    # -- operators ----------------------------------------------------------

    def __add__(self, other: Any) -> "Theme":
        """Merge another theme into this one (``self + other``).

        Properties in *other* override those in *self*; ``None`` properties
        in *other* are filled from *self* (element-level merge).

        Parameters
        ----------
        other : Theme or None
            Theme to merge.

        Returns
        -------
        Theme
        """
        if other is None:
            return self.copy()
        if not isinstance(other, Theme):
            cli_abort(
                f"Cannot add {type(other).__name__} to a Theme object."
            )
        return add_theme(self, other)

    def __radd__(self, other: Any) -> "Theme":
        """Support ``None + theme``."""
        if other is None or other == 0:
            return self.copy()
        if isinstance(other, Theme):
            return add_theme(other, self)
        return NotImplemented

    def __repr__(self) -> str:
        n = len(self._elements)
        tag = "complete " if self.complete else ""
        return f"<Theme ({tag}{n} elements)>"


# ---------------------------------------------------------------------------
# theme() constructor
# ---------------------------------------------------------------------------

def theme(complete: bool = False, validate: bool = True, **kwargs: Any) -> Theme:
    """Create a ``Theme`` object.

    Parameters
    ----------
    complete : bool
        If ``True``, marks this as a complete theme (like ``theme_grey()``).
        Complete themes set ``inherit_blank=True`` on all elements.
    validate : bool
        If ``True``, element values are checked against the element tree.
    **kwargs
        Named theme elements (e.g. ``line=element_line(...)``).

    Returns
    -------
    Theme
        A new theme object.

    Examples
    --------
    >>> from ggplot2_py.theme_elements import element_text, element_rect, rel
    >>> t = theme(plot_title=element_text(size=rel(1.2), hjust=0))
    """
    # Normalise dots to dashes (Python kwargs use underscores but theme
    # element names use dots in R).  We accept both forms.
    elements: Dict[str, Any] = {}
    for key, value in kwargs.items():
        # Convert underscores to dots so that ``axis_text_x`` -> ``axis.text.x``
        canonical = key.replace("_", ".")
        # But preserve 'inherit.blank' as-is (it's an element property, not
        # a theme key).  Actually theme keys never contain "inherit.blank"
        # so this is fine.
        elements[canonical] = value

    # If complete, set inherit_blank = True on all elements
    if complete:
        for key, el in elements.items():
            if isinstance(el, Element) and hasattr(el, "inherit_blank"):
                el.inherit_blank = True

    return Theme(elements=elements, complete=complete, validate=validate)


# ---------------------------------------------------------------------------
# is_theme
# ---------------------------------------------------------------------------

def is_theme(x: Any) -> bool:
    """Test whether *x* is a ``Theme`` object.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(x, Theme)


# ---------------------------------------------------------------------------
# add_theme — the engine behind ``+``
# ---------------------------------------------------------------------------

def add_theme(t1: Theme, t2: Theme) -> Theme:
    """Merge theme *t2* into *t1*.

    Parameters
    ----------
    t1 : Theme
        The base theme (may also be a plain dict for the initial plot theme).
    t2 : Theme
        The theme to add (its elements override *t1*'s).

    Returns
    -------
    Theme
        A new theme with merged elements.
    """
    if t2 is None:
        if isinstance(t1, dict):
            return Theme(t1)
        return t1.copy()

    # If t2 is complete, it replaces t1 entirely
    if t2.complete:
        return t2.copy()

    if t1 is None:
        return t2.copy()

    # If t1 is a plain dict (e.g. initial plot theme), wrap it in a Theme
    # so it has .validate, .copy(), .keys() etc.  Mirrors R's
    # ``if (!is_theme(t1) && is.list(t1)) t1 <- theme(!!!t1)``
    if isinstance(t1, dict) and not isinstance(t1, Theme):
        t1 = Theme(t1)

    result = t1.copy()
    for item in t2.keys():
        try:
            old_val = result.get(item)
            new_val = t2[item]
            merged = merge_element(new_val, old_val)
            result[item] = merged
        except Exception as exc:
            cli_warn(f"Problem merging theme element '{item}': {exc}")
            result[item] = t2[item]

    result.validate = t1.validate and t2.validate
    return result


def theme_replace_op(e1: Theme, e2: Theme) -> Theme:
    """The ``%+replace%`` operator: replace elements wholesale.

    Unlike ``+``, this does not merge element-level properties.
    Missing elements in *e2* result in ``None`` in the output.

    Parameters
    ----------
    e1 : Theme
        The base theme.
    e2 : Theme
        The replacement theme.

    Returns
    -------
    Theme
    """
    if not isinstance(e1, Theme) or not isinstance(e2, Theme):
        cli_abort("%+replace% requires two Theme objects.")
    result = e1.copy()
    for key in e2.keys():
        result[key] = e2[key]
    return result


# ---------------------------------------------------------------------------
# complete_theme
# ---------------------------------------------------------------------------

def complete_theme(
    theme_obj: Optional[Theme] = None,
    default: Optional[Theme] = None,
) -> Theme:
    """Complete a theme so every element is fully resolved.

    Parameters
    ----------
    theme_obj : Theme or None
        An incomplete theme to complete.
    default : Theme or None
        A complete theme to fill in missing elements.  Falls back to the
        current global theme.

    Returns
    -------
    Theme
        A fully resolved theme.
    """
    if default is None:
        default = get_theme()
    if default is None:
        # No global theme set yet; return what we have
        if theme_obj is None:
            return Theme(complete=True, validate=False)
        result = theme_obj.copy()
        result.complete = True
        result.validate = False
        return result

    if theme_obj is None:
        result = default.copy()
    elif theme_obj.complete:
        # For complete themes, only fill missing elements
        result = theme_obj.copy()
        for key in default.keys():
            if key not in result:
                result[key] = default[key]
    else:
        result = default + theme_obj

    # Fill from global default as last resort
    global_default = _ggplot_global.theme_default
    if global_default is not None:
        for key in global_default.keys():
            if key not in result:
                result[key] = global_default[key]

    result.complete = True
    result.validate = False
    return result


# ---------------------------------------------------------------------------
# Global theme state
# ---------------------------------------------------------------------------

def get_theme() -> Optional[Theme]:
    """Return the currently active global theme.

    Returns
    -------
    Theme or None
    """
    return _ggplot_global.theme_current


def set_theme(new: Optional[Theme] = None) -> Optional[Theme]:
    """Set the global theme, returning the previous one.

    Parameters
    ----------
    new : Theme or None
        The theme to make active.  If ``None``, resets to the default.

    Returns
    -------
    Theme or None
        The previously active theme.
    """
    if new is None:
        new = _ggplot_global.theme_default
    if new is not None and not isinstance(new, Theme):
        cli_abort("set_theme() requires a Theme object.")
    old = _ggplot_global.theme_current
    _ggplot_global.theme_current = new
    return old


def update_theme(**kwargs: Any) -> Optional[Theme]:
    """Update the current global theme in-place.

    Parameters
    ----------
    **kwargs
        Theme element overrides (passed to ``theme()``).

    Returns
    -------
    Theme or None
        The previously active theme.
    """
    current = get_theme()
    new_theme = theme(**kwargs)
    if current is not None:
        return set_theme(current + new_theme)
    return set_theme(new_theme)


def replace_theme(**kwargs: Any) -> Optional[Theme]:
    """Replace elements in the current global theme.

    Unlike ``update_theme``, this replaces elements wholesale
    (no element-level merging).

    Parameters
    ----------
    **kwargs
        Theme element overrides.

    Returns
    -------
    Theme or None
        The previously active theme.
    """
    current = get_theme()
    new_theme = theme(**kwargs)
    if current is not None:
        return set_theme(theme_replace_op(current, new_theme))
    return set_theme(new_theme)


def reset_theme_settings(reset_current: bool = True) -> None:
    """Reset the global theme state to built-in defaults.

    Parameters
    ----------
    reset_current : bool
        If ``True`` (default), also reset the currently active theme.
    """
    from ggplot2_py.theme_elements import reset_theme_settings as _reset_tree

    _reset_tree(reset_current=False)
    # Lazy import to avoid circular dependency at module load time
    try:
        from ggplot2_py.theme_defaults import theme_grey

        _ggplot_global.theme_default = theme_grey()
        if reset_current:
            _ggplot_global.theme_current = _ggplot_global.theme_default
    except ImportError:
        # theme_defaults may not be available yet during initial import
        pass


# R-compatible aliases
theme_get = get_theme
theme_set = set_theme
theme_update = update_theme
theme_replace = replace_theme

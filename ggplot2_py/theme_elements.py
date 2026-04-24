"""
Theme element classes for ggplot2.

Provides the element hierarchy (blank, line, rect, text, point, polygon, geom),
the ``Rel`` and ``Margin`` helper types, factory functions such as
``element_blank()``, ``element_line()``, etc., and the element-tree machinery
used to resolve theme inheritance via ``calc_element()``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from grid_py import (
    Unit,
    Gpar,
    unit_c,
    Grob,
    GTree,
    rect_grob,
    lines_grob,
    polyline_grob,
    text_grob,
    polygon_grob,
    null_grob,
    Viewport,
    edit_grob,
    grob_width,
    grob_height,
)

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn

__all__ = [
    "Element",
    "ElementBlank",
    "ElementLine",
    "ElementRect",
    "ElementText",
    "ElementPoint",
    "ElementPolygon",
    "ElementGeom",
    "element_blank",
    "element_line",
    "element_rect",
    "element_text",
    "element_point",
    "element_polygon",
    "element_geom",
    "element_grob",
    "element_render",
    "el_def",
    "merge_element",
    "combine_elements",
    "is_theme_element",
    "Margin",
    "margin",
    "margin_auto",
    "margin_part",
    "is_margin",
    "Rel",
    "rel",
    "is_rel",
    "calc_element",
    "get_element_tree",
    "register_theme_elements",
    "reset_theme_settings",
]


# ---------------------------------------------------------------------------
# Graphical-unit conversion constants (R: ggplot2/R/geom-.R, lines 513-517)
# ---------------------------------------------------------------------------
# Multiply a size in mm by these to convert to the units that grid uses
# internally for ``lwd`` and ``fontsize``.
#   .pt    = 72.27 / 25.4  — mm → points (for lwd & fontsize)
#   .stroke = 96 / 25.4    — mm → stroke units (for point border widths)
_PT: float = 72.27 / 25.4
_STROKE: float = 96 / 25.4


# ---------------------------------------------------------------------------
# Rel — relative size multiplier
# ---------------------------------------------------------------------------

class Rel:
    """A relative-size wrapper.

    Parameters
    ----------
    x : float
        The multiplier applied relative to the parent element's value.
    """

    __slots__ = ("_x",)

    def __init__(self, x: float) -> None:
        self._x = float(x)

    @property
    def value(self) -> float:
        """The numeric multiplier."""
        return self._x

    # Arithmetic so that ``Rel(0.8) * 11`` works transparently.
    def __mul__(self, other: Any) -> Any:
        if isinstance(other, Rel):
            return Rel(self._x * other._x)
        if isinstance(other, (int, float)):
            return self._x * other
        if isinstance(other, Unit):
            return self._x * other
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __float__(self) -> float:
        return self._x

    def __repr__(self) -> str:
        return f"rel({self._x})"


def rel(x: float) -> Rel:
    """Create a ``Rel`` (relative-size) object.

    Parameters
    ----------
    x : float
        Numeric multiplier specifying size relative to the parent element.

    Returns
    -------
    Rel
        A relative-size wrapper.
    """
    return Rel(x)


def is_rel(x: Any) -> bool:
    """Test whether *x* is a ``Rel`` object.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(x, Rel)


# ---------------------------------------------------------------------------
# Margin
# ---------------------------------------------------------------------------

class Margin:
    """A four-sided margin (top, right, bottom, left) stored as a ``Unit``.

    Parameters
    ----------
    t : float
        Top margin value.
    r : float
        Right margin value.
    b : float
        Bottom margin value.
    l : float
        Left margin value.
    unit : str
        Unit string (default ``"pt"``).
    """

    __slots__ = ("_values", "_unit_str", "_unit")

    def __init__(
        self,
        t: float = 0.0,
        r: float = 0.0,
        b: float = 0.0,
        l: float = 0.0,
        unit: str = "pt",
    ) -> None:
        self._values: Tuple[float, float, float, float] = (
            float(t),
            float(r),
            float(b),
            float(l),
        )
        self._unit_str = unit
        self._unit = Unit(list(self._values), unit)

    # Named accessors
    @property
    def t(self) -> float:
        """Top margin."""
        return self._values[0]

    @property
    def r(self) -> float:
        """Right margin."""
        return self._values[1]

    @property
    def b(self) -> float:
        """Bottom margin."""
        return self._values[2]

    @property
    def l(self) -> float:
        """Left margin."""
        return self._values[3]

    @property
    def unit_str(self) -> str:
        """The unit string."""
        return self._unit_str

    @property
    def unit(self) -> Unit:
        """The underlying ``grid_py.Unit`` object."""
        return self._unit

    def __getitem__(self, idx: int) -> float:
        return self._values[idx]

    def __len__(self) -> int:
        return 4

    def __iter__(self):
        return iter(self._values)

    def __repr__(self) -> str:
        return (
            f"margin(t={self.t}, r={self.r}, b={self.b}, l={self.l}, "
            f"unit={self._unit_str!r})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Margin):
            return self._values == other._values and self._unit_str == other._unit_str
        return NotImplemented


def margin(
    t: float = 0.0,
    r: float = 0.0,
    b: float = 0.0,
    l: float = 0.0,
    unit: str = "pt",
) -> Margin:
    """Create a ``Margin`` object.

    Parameters
    ----------
    t : float
        Top margin.
    r : float
        Right margin.
    b : float
        Bottom margin.
    l : float
        Left margin.
    unit : str
        Measurement unit (default ``"pt"``).

    Returns
    -------
    Margin
        A four-sided margin.
    """
    return Margin(t=t, r=r, b=b, l=l, unit=unit)


def margin_auto(
    t: float = 0.0,
    r: Optional[float] = None,
    b: Optional[float] = None,
    l: Optional[float] = None,
    unit: str = "pt",
) -> Margin:
    """Create a ``Margin`` with auto-recycling (CSS-like shorthand).

    Parameters
    ----------
    t : float
        Top margin.
    r : float, optional
        Right margin.  Defaults to *t*.
    b : float, optional
        Bottom margin.  Defaults to *t*.
    l : float, optional
        Left margin.  Defaults to *r*.
    unit : str
        Measurement unit (default ``"pt"``).

    Returns
    -------
    Margin
    """
    if r is None:
        r = t
    if b is None:
        b = t
    if l is None:
        l = r
    return Margin(t=t, r=r, b=b, l=l, unit=unit)


def margin_part(
    t: float = float("nan"),
    r: float = float("nan"),
    b: float = float("nan"),
    l: float = float("nan"),
    unit: str = "pt",
) -> Margin:
    """Create a partial ``Margin`` (unset sides are ``NaN``).

    Parameters
    ----------
    t, r, b, l : float
        Margin values; NaN means "inherit from parent".
    unit : str
        Measurement unit.

    Returns
    -------
    Margin
    """
    return Margin(t=t, r=r, b=b, l=l, unit=unit)


def is_margin(x: Any) -> bool:
    """Test whether *x* is a ``Margin`` object.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(x, Margin)


# ---------------------------------------------------------------------------
# Element base & subclasses
# ---------------------------------------------------------------------------

class Element:
    """Abstract base class for theme elements.

    All concrete element classes inherit from this.
    """

    @property
    def blank(self) -> bool:
        """Whether this element draws nothing."""
        return False

    def merge(self, other: "Element") -> "Element":
        """Merge *other* (parent) into this element, filling ``None`` slots.

        Parameters
        ----------
        other : Element
            The parent element to inherit from.

        Returns
        -------
        Element
            A new element with ``None`` properties filled from *other*.
        """
        return merge_element(self, other)


class ElementBlank(Element):
    """An element that draws nothing and allocates no space.

    Parameters
    ----------
    inherit_blank : bool
        Kept for interface consistency; always ``True`` conceptually.
    """

    def __init__(self, inherit_blank: bool = True) -> None:
        self.inherit_blank = inherit_blank

    @property
    def blank(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "element_blank()"


class ElementLine(Element):
    """Theme element for lines.

    Parameters
    ----------
    colour : str or None
        Line colour.
    linewidth : float or None
        Line width in mm.
    linetype : int, str, or None
        Line type.
    lineend : str or None
        Line end style (``"round"``, ``"butt"``, ``"square"``).
    linejoin : str or None
        Line join style (``"round"``, ``"mitre"``, ``"bevel"``).
    arrow : object or None
        Arrow specification (from ``grid_py.arrow``).
    arrow_fill : str or None
        Fill colour for closed arrow heads.
    inherit_blank : bool
        Whether to inherit ``element_blank`` from parents.
    """

    def __init__(
        self,
        colour: Optional[str] = None,
        linewidth: Optional[float] = None,
        linetype: Optional[Union[int, str]] = None,
        lineend: Optional[str] = None,
        linejoin: Optional[str] = None,
        arrow: Optional[Any] = None,
        arrow_fill: Optional[str] = None,
        inherit_blank: bool = False,
    ) -> None:
        self.colour = colour
        self.linewidth = linewidth
        self.linetype = linetype
        self.lineend = lineend
        self.linejoin = linejoin
        self.arrow = arrow
        self.arrow_fill = arrow_fill
        self.inherit_blank = inherit_blank

    def __repr__(self) -> str:
        parts = []
        for attr in (
            "colour",
            "linewidth",
            "linetype",
            "lineend",
            "linejoin",
            "arrow",
            "arrow_fill",
            "inherit_blank",
        ):
            val = getattr(self, attr)
            if val is not None and val is not False:
                parts.append(f"{attr}={val!r}")
        return f"element_line({', '.join(parts)})"


class ElementRect(Element):
    """Theme element for rectangles (borders and backgrounds).

    Parameters
    ----------
    fill : str or None
        Fill colour.
    colour : str or None
        Border colour.
    linewidth : float or None
        Border width in mm.
    linetype : int, str, or None
        Border line type.
    linejoin : str or None
        Line join style.
    inherit_blank : bool
        Whether to inherit ``element_blank`` from parents.
    """

    def __init__(
        self,
        fill: Optional[str] = None,
        colour: Optional[str] = None,
        linewidth: Optional[float] = None,
        linetype: Optional[Union[int, str]] = None,
        linejoin: Optional[str] = None,
        inherit_blank: bool = False,
    ) -> None:
        self.fill = fill
        self.colour = colour
        self.linewidth = linewidth
        self.linetype = linetype
        self.linejoin = linejoin
        self.inherit_blank = inherit_blank

    def __repr__(self) -> str:
        parts = []
        for attr in ("fill", "colour", "linewidth", "linetype", "linejoin", "inherit_blank"):
            val = getattr(self, attr)
            if val is not None and val is not False:
                parts.append(f"{attr}={val!r}")
        return f"element_rect({', '.join(parts)})"


class ElementText(Element):
    """Theme element for text.

    Parameters
    ----------
    family : str or None
        Font family.
    face : str or None
        Font face (``"plain"``, ``"italic"``, ``"bold"``, ``"bold.italic"``).
    colour : str or None
        Text colour.
    size : float, Rel, or None
        Font size in points (or ``Rel`` for relative sizing).
    hjust : float or None
        Horizontal justification (0--1).
    vjust : float or None
        Vertical justification (0--1).
    angle : float or None
        Rotation angle in degrees.
    lineheight : float or None
        Line height multiplier.
    margin : Margin or None
        Margins around the text.
    debug : bool or None
        If ``True``, draw debugging annotations.
    inherit_blank : bool
        Whether to inherit ``element_blank`` from parents.
    """

    def __init__(
        self,
        family: Optional[str] = None,
        face: Optional[str] = None,
        colour: Optional[str] = None,
        size: Optional[Union[float, Rel]] = None,
        hjust: Optional[float] = None,
        vjust: Optional[float] = None,
        angle: Optional[float] = None,
        lineheight: Optional[float] = None,
        margin: Optional[Margin] = None,
        debug: Optional[bool] = None,
        inherit_blank: bool = False,
    ) -> None:
        self.family = family
        self.face = face
        self.colour = colour
        self.size = size
        self.hjust = hjust
        self.vjust = vjust
        self.angle = angle
        self.lineheight = lineheight
        self.margin = margin
        self.debug = debug
        self.inherit_blank = inherit_blank

    def __repr__(self) -> str:
        parts = []
        for attr in (
            "family",
            "face",
            "colour",
            "size",
            "hjust",
            "vjust",
            "angle",
            "lineheight",
            "margin",
            "debug",
            "inherit_blank",
        ):
            val = getattr(self, attr)
            if val is not None and val is not False:
                parts.append(f"{attr}={val!r}")
        return f"element_text({', '.join(parts)})"


class ElementPoint(Element):
    """Theme element for points.

    Parameters
    ----------
    shape : int, str, or None
        Point shape.
    colour : str or None
        Point colour.
    fill : str or None
        Point fill colour.
    size : float or None
        Point size in mm.
    stroke : float or None
        Stroke width.
    inherit_blank : bool
        Whether to inherit ``element_blank`` from parents.
    """

    def __init__(
        self,
        shape: Optional[Union[int, str]] = None,
        colour: Optional[str] = None,
        fill: Optional[str] = None,
        size: Optional[float] = None,
        stroke: Optional[float] = None,
        inherit_blank: bool = False,
    ) -> None:
        self.shape = shape
        self.colour = colour
        self.fill = fill
        self.size = size
        self.stroke = stroke
        self.inherit_blank = inherit_blank

    def __repr__(self) -> str:
        parts = []
        for attr in ("shape", "colour", "fill", "size", "stroke", "inherit_blank"):
            val = getattr(self, attr)
            if val is not None and val is not False:
                parts.append(f"{attr}={val!r}")
        return f"element_point({', '.join(parts)})"


class ElementPolygon(Element):
    """Theme element for polygons.

    Parameters
    ----------
    colour : str or None
        Border colour.
    fill : str or None
        Fill colour.
    linewidth : float or None
        Border width in mm.
    linetype : int, str, or None
        Border line type.
    linejoin : str or None
        Line join style.
    inherit_blank : bool
        Whether to inherit ``element_blank`` from parents.
    """

    def __init__(
        self,
        colour: Optional[str] = None,
        fill: Optional[str] = None,
        linewidth: Optional[float] = None,
        linetype: Optional[Union[int, str]] = None,
        linejoin: Optional[str] = None,
        inherit_blank: bool = False,
    ) -> None:
        self.colour = colour
        self.fill = fill
        self.linewidth = linewidth
        self.linetype = linetype
        self.linejoin = linejoin
        self.inherit_blank = inherit_blank

    def __repr__(self) -> str:
        parts = []
        for attr in ("colour", "fill", "linewidth", "linetype", "linejoin", "inherit_blank"):
            val = getattr(self, attr)
            if val is not None and val is not False:
                parts.append(f"{attr}={val!r}")
        return f"element_polygon({', '.join(parts)})"


class ElementGeom(Element):
    """Theme element for global geom defaults.

    Parameters
    ----------
    ink : str or None
        Foreground colour.
    paper : str or None
        Background colour.
    accent : str or None
        Accent colour.
    linewidth : float or None
        Default line width in mm.
    borderwidth : float or None
        Default border width in mm.
    linetype : int, str, or None
        Default line type.
    bordertype : int, str, or None
        Default border type.
    family : str or None
        Default font family.
    fontsize : float or None
        Default font size in points.
    pointsize : float or None
        Default point size in mm.
    pointshape : int or None
        Default point shape.
    colour : str or None
        Explicit colour override.
    fill : str or None
        Explicit fill override.
    """

    def __init__(
        self,
        ink: Optional[str] = None,
        paper: Optional[str] = None,
        accent: Optional[str] = None,
        linewidth: Optional[float] = None,
        borderwidth: Optional[float] = None,
        linetype: Optional[Union[int, str]] = None,
        bordertype: Optional[Union[int, str]] = None,
        family: Optional[str] = None,
        fontsize: Optional[float] = None,
        pointsize: Optional[float] = None,
        pointshape: Optional[int] = None,
        colour: Optional[str] = None,
        fill: Optional[str] = None,
    ) -> None:
        self.ink = ink
        self.paper = paper
        self.accent = accent
        self.linewidth = linewidth
        self.borderwidth = borderwidth
        self.linetype = linetype
        self.bordertype = bordertype
        self.family = family
        self.fontsize = fontsize
        self.pointsize = pointsize
        self.pointshape = pointshape
        self.colour = colour
        self.fill = fill

    def __repr__(self) -> str:
        parts = []
        for attr in (
            "ink",
            "paper",
            "accent",
            "linewidth",
            "borderwidth",
            "linetype",
            "bordertype",
            "family",
            "fontsize",
            "pointsize",
            "pointshape",
            "colour",
            "fill",
        ):
            val = getattr(self, attr)
            if val is not None:
                parts.append(f"{attr}={val!r}")
        return f"element_geom({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def element_blank() -> ElementBlank:
    """Create a blank element that draws nothing.

    Returns
    -------
    ElementBlank
    """
    return ElementBlank()


def element_line(
    colour: Optional[str] = None,
    linewidth: Optional[float] = None,
    linetype: Optional[Union[int, str]] = None,
    lineend: Optional[str] = None,
    linejoin: Optional[str] = None,
    arrow: Optional[Any] = None,
    arrow_fill: Optional[str] = None,
    color: Optional[str] = None,
    inherit_blank: bool = False,
) -> ElementLine:
    """Create a line theme element.

    Parameters
    ----------
    colour : str, optional
        Line colour.
    linewidth : float, optional
        Line width in mm.
    linetype : int or str, optional
        Line type.
    lineend : str, optional
        Line end style.
    linejoin : str, optional
        Line join style.
    arrow : object, optional
        Arrow specification.
    arrow_fill : str, optional
        Arrow fill colour.
    color : str, optional
        Alias for *colour*.
    inherit_blank : bool
        Inherit blank from parents (default ``False``).

    Returns
    -------
    ElementLine
    """
    colour = color if colour is None else colour
    return ElementLine(
        colour=colour,
        linewidth=linewidth,
        linetype=linetype,
        lineend=lineend,
        linejoin=linejoin,
        arrow=arrow,
        arrow_fill=arrow_fill,
        inherit_blank=inherit_blank,
    )


def element_rect(
    fill: Optional[str] = None,
    colour: Optional[str] = None,
    linewidth: Optional[float] = None,
    linetype: Optional[Union[int, str]] = None,
    color: Optional[str] = None,
    linejoin: Optional[str] = None,
    inherit_blank: bool = False,
) -> ElementRect:
    """Create a rectangle theme element.

    Parameters
    ----------
    fill : str, optional
        Fill colour.
    colour : str, optional
        Border colour.
    linewidth : float, optional
        Border width in mm.
    linetype : int or str, optional
        Border line type.
    color : str, optional
        Alias for *colour*.
    linejoin : str, optional
        Line join style.
    inherit_blank : bool
        Inherit blank from parents (default ``False``).

    Returns
    -------
    ElementRect
    """
    colour = color if colour is None else colour
    return ElementRect(
        fill=fill,
        colour=colour,
        linewidth=linewidth,
        linetype=linetype,
        linejoin=linejoin,
        inherit_blank=inherit_blank,
    )


def element_text(
    family: Optional[str] = None,
    face: Optional[str] = None,
    colour: Optional[str] = None,
    size: Optional[Union[float, Rel]] = None,
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    angle: Optional[float] = None,
    lineheight: Optional[float] = None,
    color: Optional[str] = None,
    margin: Optional[Margin] = None,
    debug: Optional[bool] = None,
    inherit_blank: bool = False,
) -> ElementText:
    """Create a text theme element.

    Parameters
    ----------
    family : str, optional
        Font family.
    face : str, optional
        Font face.
    colour : str, optional
        Text colour.
    size : float or Rel, optional
        Font size in points, or a ``Rel`` for relative sizing.
    hjust : float, optional
        Horizontal justification (0--1).
    vjust : float, optional
        Vertical justification (0--1).
    angle : float, optional
        Text rotation angle in degrees.
    lineheight : float, optional
        Line height multiplier.
    color : str, optional
        Alias for *colour*.
    margin : Margin, optional
        Margins around the text.
    debug : bool, optional
        Draw debug annotations.
    inherit_blank : bool
        Inherit blank from parents (default ``False``).

    Returns
    -------
    ElementText
    """
    colour = color if colour is None else colour
    return ElementText(
        family=family,
        face=face,
        colour=colour,
        size=size,
        hjust=hjust,
        vjust=vjust,
        angle=angle,
        lineheight=lineheight,
        margin=margin,
        debug=debug,
        inherit_blank=inherit_blank,
    )


def element_point(
    shape: Optional[Union[int, str]] = None,
    colour: Optional[str] = None,
    fill: Optional[str] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    color: Optional[str] = None,
    inherit_blank: bool = False,
) -> ElementPoint:
    """Create a point theme element.

    Parameters
    ----------
    shape : int or str, optional
        Point shape.
    colour : str, optional
        Point colour.
    fill : str, optional
        Point fill colour.
    size : float, optional
        Point size in mm.
    stroke : float, optional
        Stroke width.
    color : str, optional
        Alias for *colour*.
    inherit_blank : bool
        Inherit blank from parents (default ``False``).

    Returns
    -------
    ElementPoint
    """
    colour = color if colour is None else colour
    return ElementPoint(
        shape=shape,
        colour=colour,
        fill=fill,
        size=size,
        stroke=stroke,
        inherit_blank=inherit_blank,
    )


def element_polygon(
    colour: Optional[str] = None,
    fill: Optional[str] = None,
    linewidth: Optional[float] = None,
    linetype: Optional[Union[int, str]] = None,
    color: Optional[str] = None,
    linejoin: Optional[str] = None,
    inherit_blank: bool = False,
) -> ElementPolygon:
    """Create a polygon theme element.

    Parameters
    ----------
    colour : str, optional
        Border colour.
    fill : str, optional
        Fill colour.
    linewidth : float, optional
        Border width in mm.
    linetype : int or str, optional
        Border line type.
    color : str, optional
        Alias for *colour*.
    linejoin : str, optional
        Line join style.
    inherit_blank : bool
        Inherit blank from parents (default ``False``).

    Returns
    -------
    ElementPolygon
    """
    colour = color if colour is None else colour
    return ElementPolygon(
        colour=colour,
        fill=fill,
        linewidth=linewidth,
        linetype=linetype,
        linejoin=linejoin,
        inherit_blank=inherit_blank,
    )


def element_geom(
    ink: Optional[str] = None,
    paper: Optional[str] = None,
    accent: Optional[str] = None,
    linewidth: Optional[float] = None,
    borderwidth: Optional[float] = None,
    linetype: Optional[Union[int, str]] = None,
    bordertype: Optional[Union[int, str]] = None,
    family: Optional[str] = None,
    fontsize: Optional[float] = None,
    pointsize: Optional[float] = None,
    pointshape: Optional[int] = None,
    colour: Optional[str] = None,
    color: Optional[str] = None,
    fill: Optional[str] = None,
) -> ElementGeom:
    """Create a geom defaults theme element.

    Parameters
    ----------
    ink : str, optional
        Foreground colour.
    paper : str, optional
        Background colour.
    accent : str, optional
        Accent colour.
    linewidth : float, optional
        Default line width in mm.
    borderwidth : float, optional
        Default border width in mm.
    linetype : int or str, optional
        Default line type.
    bordertype : int or str, optional
        Default border type.
    family : str, optional
        Default font family.
    fontsize : float, optional
        Default font size in points.
    pointsize : float, optional
        Default point size in mm.
    pointshape : int, optional
        Default point shape.
    colour : str, optional
        Explicit colour override.
    color : str, optional
        Alias for *colour*.
    fill : str, optional
        Explicit fill override.

    Returns
    -------
    ElementGeom
    """
    colour = color if colour is None else colour
    return ElementGeom(
        ink=ink,
        paper=paper,
        accent=accent,
        linewidth=linewidth,
        borderwidth=borderwidth,
        linetype=linetype,
        bordertype=bordertype,
        family=family,
        fontsize=fontsize,
        pointsize=pointsize,
        pointshape=pointshape,
        colour=colour,
        fill=fill,
    )


# ---------------------------------------------------------------------------
# Type predicates
# ---------------------------------------------------------------------------

_TYPE_MAP: Dict[str, type] = {
    "any": Element,
    "blank": ElementBlank,
    "line": ElementLine,
    "rect": ElementRect,
    "text": ElementText,
    "point": ElementPoint,
    "polygon": ElementPolygon,
    "geom": ElementGeom,
}


def is_theme_element(x: Any, type_: str = "any") -> bool:
    """Test whether *x* is a theme element, optionally of a specific type.

    Parameters
    ----------
    x : Any
        Object to test.
    type_ : str
        One of ``"any"``, ``"blank"``, ``"rect"``, ``"line"``, ``"text"``,
        ``"polygon"``, ``"point"``, ``"geom"``.

    Returns
    -------
    bool
    """
    cls = _TYPE_MAP.get(type_, None)
    if cls is None:
        return False
    return isinstance(x, cls)


# ---------------------------------------------------------------------------
# Helper to get "properties" dict from an element (for merging)
# ---------------------------------------------------------------------------

def _element_props(el: Element) -> Dict[str, Any]:
    """Return a dict of the element's settable properties."""
    if isinstance(el, ElementBlank):
        return {"inherit_blank": el.inherit_blank}
    return {k: v for k, v in el.__dict__.items()}


def _element_prop_names(el: Element) -> List[str]:
    """Return the property names of an element (excluding inherit_blank for checks)."""
    return list(el.__dict__.keys())


# ---------------------------------------------------------------------------
# Merge & combine
# ---------------------------------------------------------------------------

def merge_element(new: Any, old: Any) -> Any:
    """Merge a child element (*new*) with a parent element (*old*).

    Properties that are ``None`` in *new* are filled from *old*.

    Parameters
    ----------
    new : Element or other
        The child element.
    old : Element or other
        The parent element.

    Returns
    -------
    Element or other
        A copy of *new* with ``None`` properties filled from *old*.
    """
    if old is None or isinstance(old, ElementBlank):
        return new
    if new is None or isinstance(new, (str, int, float, bool)):
        return new
    if isinstance(new, ElementBlank):
        return new
    if isinstance(new, Unit):
        return new
    if isinstance(new, Margin):
        return new
    if not isinstance(new, Element) or not isinstance(old, Element):
        return new

    # Classes must be compatible for merging
    if type(new) is not type(old):
        # Allow merging if new's class is a subclass of old's
        if not isinstance(new, type(old)):
            cli_abort(
                f"Only elements of the same class can be merged, "
                f"got {type(new).__name__} and {type(old).__name__}."
            )

    result = copy.copy(new)
    for attr in old.__dict__:
        if attr in result.__dict__ and getattr(result, attr) is None:
            setattr(result, attr, getattr(old, attr))
    return result


def combine_elements(e1: Any, e2: Any) -> Any:
    """Combine element *e1* with its parent *e2* (full inheritance resolution).

    Unlike ``merge_element``, this also resolves ``Rel`` sizes and
    handles ``element_blank`` inheritance.

    Parameters
    ----------
    e1 : Any
        The child element (or value).
    e2 : Any
        The parent element (or value) from which *e1* inherits.

    Returns
    -------
    Any
        The resolved element.
    """
    # If e2 is None, nothing to inherit
    if e2 is None or isinstance(e1, ElementBlank):
        return e1

    # If e1 is None, inherit everything from e2
    if e1 is None:
        return e2

    # Rel handling
    if isinstance(e1, Rel):
        if isinstance(e2, Rel):
            return Rel(e1.value * e2.value)
        if isinstance(e2, (int, float)):
            return e1.value * e2
        if isinstance(e2, Unit):
            return e1.value * e2
        return e1

    # Margin merging
    if isinstance(e1, Margin) and isinstance(e2, Margin):
        import math

        t = e2.t if math.isnan(e1.t) else e1.t
        r = e2.r if math.isnan(e1.r) else e1.r
        b = e2.b if math.isnan(e1.b) else e1.b
        l = e2.l if math.isnan(e1.l) else e1.l
        return Margin(t=t, r=r, b=b, l=l, unit=e1.unit_str)

    # If neither is an Element, return e1
    if not isinstance(e1, Element) and not isinstance(e2, Element):
        return e1

    # If e2 is blank and e1 inherits blank, return e2
    if isinstance(e2, ElementBlank):
        if isinstance(e1, Element) and getattr(e1, "inherit_blank", False):
            return e2
        return e1

    # Fill None properties of e1 from e2
    if isinstance(e1, Element) and isinstance(e2, Element):
        result = copy.copy(e1)
        for attr in e2.__dict__:
            if attr in result.__dict__ and getattr(result, attr) is None:
                setattr(result, attr, getattr(e2, attr))

        # Resolve relative sizes
        if hasattr(result, "size") and isinstance(result.size, Rel):
            parent_size = getattr(e2, "size", None)
            if parent_size is not None and not isinstance(parent_size, Rel):
                result.size = result.size.value * parent_size

        # Resolve relative linewidth
        if hasattr(result, "linewidth") and isinstance(result.linewidth, Rel):
            parent_lw = getattr(e2, "linewidth", None)
            if parent_lw is not None and not isinstance(parent_lw, Rel):
                result.linewidth = result.linewidth.value * parent_lw

        # Resolve margin inheritance for text elements
        if isinstance(result, ElementText) and result.margin is not None:
            parent_margin = getattr(e2, "margin", None)
            if parent_margin is not None:
                result.margin = combine_elements(result.margin, parent_margin)

        return result

    return e1


# ---------------------------------------------------------------------------
# Element grob rendering
# ---------------------------------------------------------------------------

def element_grob(element: Element, **kwargs: Any) -> Any:
    """Generate a grid grob from a theme element.

    Parameters
    ----------
    element : Element
        A theme element (``ElementLine``, ``ElementRect``, ``ElementText``,
        ``ElementBlank``, etc.).
    **kwargs
        Additional arguments controlling rendering (e.g. position, labels).

    Returns
    -------
    Grob
        A grid grob.
    """
    if isinstance(element, ElementBlank):
        return null_grob()

    if isinstance(element, ElementRect):
        return _grob_from_rect(element, **kwargs)

    if isinstance(element, ElementLine):
        return _grob_from_line(element, **kwargs)

    if isinstance(element, ElementText):
        return _grob_from_text(element, **kwargs)

    if isinstance(element, ElementPoint):
        return _grob_from_point(element, **kwargs)

    if isinstance(element, ElementPolygon):
        return _grob_from_polygon(element, **kwargs)

    if isinstance(element, ElementGeom):
        # ElementGeom defines global defaults; not directly rendered.
        return null_grob()

    # Fallback
    return null_grob()


def _grob_from_rect(
    element: ElementRect,
    x: float = 0.5,
    y: float = 0.5,
    width: float = 1.0,
    height: float = 1.0,
    fill: Optional[str] = None,
    colour: Optional[str] = None,
    linewidth: Optional[float] = None,
    linetype: Optional[Union[int, str]] = None,
    **kwargs: Any,
) -> Any:
    """Render an ``ElementRect`` as a rect grob.

    Mirrors R's ``element_grob(element_rect, ...)`` which converts
    linewidth (mm) → lwd (points) via ``gg_par(lwd = linewidth)``.
    """
    lwd_mm = linewidth if linewidth is not None else element.linewidth
    gp = Gpar(
        fill=fill if fill is not None else element.fill,
        col=colour if colour is not None else element.colour,
        lwd=float(lwd_mm) * _PT if lwd_mm is not None else None,
        lty=linetype if linetype is not None else element.linetype,
    )
    return rect_grob(x=x, y=y, width=width, height=height, gp=gp, **kwargs)


def _grob_from_line(
    element: ElementLine,
    x: Any = None,
    y: Any = None,
    colour: Optional[str] = None,
    linewidth: Optional[float] = None,
    linetype: Optional[Union[int, str]] = None,
    lineend: Optional[str] = None,
    id: Optional[Any] = None,
    id_lengths: Optional[Any] = None,
    default_units: str = "npc",
    **kwargs: Any,
) -> Any:
    """Render an ``ElementLine`` as a polyline grob.

    Mirrors R's ``element_grob.element_line`` (theme-elements.R:558-595):
    always emits a ``polylineGrob`` (accepts ``id.lengths`` for
    multi-segment lines — used for panel gridlines); converts
    ``linewidth`` (mm) → ``lwd`` (points) via ``gg_par(lwd = linewidth)``.
    """
    if x is None:
        x = [0, 1]
    if y is None:
        y = [0, 1]
    lwd_mm = linewidth if linewidth is not None else element.linewidth
    gp = Gpar(
        col=colour if colour is not None else element.colour,
        lwd=float(lwd_mm) * _PT if lwd_mm is not None else None,
        lty=linetype if linetype is not None else element.linetype,
        lineend=lineend if lineend is not None else element.lineend,
    )
    return polyline_grob(
        x=x, y=y,
        id=id, id_lengths=id_lengths,
        default_units=default_units,
        gp=gp, **kwargs,
    )


def _rotate_just(angle: float, hjust: float, vjust: float) -> Tuple[float, float]:
    """Rotate (hjust, vjust) counter-clockwise into the rotated text frame.

    Mirrors R's ``rotate_just()`` (margins.R:216-276).  Used by
    ``titleGrob`` to compute default x/y anchors so rotated text
    sits at the correct point within its parent viewport.
    """
    import bisect
    a = (float(angle) if angle is not None else 0.0) % 360
    hj = 0.5 if hjust is None else float(hjust)
    vj = 0.5 if vjust is None else float(vjust)
    # R: case <- findInterval(angle, c(0, 90, 180, 270, 360))
    # findInterval is right-open: case=1 for [0,90), 2 for [90,180), etc.
    case = bisect.bisect_right([0.0, 90.0, 180.0, 270.0, 360.0], a)
    if case == 2:          # 90 <= a < 180
        return (1 - vj, hj)
    if case == 3:          # 180 <= a < 270
        return (1 - hj, 1 - vj)
    if case == 4:          # 270 <= a < 360
        return (vj, 1 - hj)
    return (hj, vj)        # 0 <= a < 90


class _TitleGrob(GTree):
    """A text grob wrapped with its margin — simplified port of R's titleGrob.

    When ``margin_x`` or ``margin_y`` is ``True``, the grob carries
    ``_widths`` and ``_heights`` vectors that include the surrounding
    margin, so ``grob_height()`` / ``grob_width()`` return the full
    extent including padding.

    R reference: ``ggplot2/R/margins.R`` lines 88-206.
    """

    def __init__(self, text_grob_child: Any, widths: Any, heights: Any,
                 name: str = "title") -> None:
        from grid_py import GList
        super().__init__(children=GList(text_grob_child), name=name)
        self._title_widths = widths
        self._title_heights = heights

    # R: widthDetails.titleGrob <- function(x) sum(x$widths)
    # (``ggplot2/R/margins.R:199-201``). R's ``sum()`` on a unit vector
    # dispatches to ``Summary.unit`` which wraps the full vector in a
    # *single-element* L_SUM compound (``grid/R/unit.R:300-347``). That
    # is what ``evaluateGrobUnit`` expects at ``grid/src/unit.c:535``,
    # where it reads ``transformWidthtoINCHES(unitx, 0, ...)``. Using
    # Python's builtin ``sum`` here would leave a multi-element unit
    # unchanged (because ``0 + Unit`` short-circuits to identity),
    # which makes grid_py read only the left-margin element and
    # underestimate the grob's width.
    def width_details(self) -> Any:
        if self._title_widths is not None:
            from grid_py import unit_summary_sum
            return unit_summary_sum(self._title_widths)
        return Unit(0, "cm")

    def height_details(self) -> Any:
        if self._title_heights is not None:
            from grid_py import unit_summary_sum
            return unit_summary_sum(self._title_heights)
        return Unit(0, "cm")


def _grob_from_text(
    element: ElementText,
    label: Optional[str] = None,
    x: Any = None,
    y: Any = None,
    family: Optional[str] = None,
    face: Optional[str] = None,
    colour: Optional[str] = None,
    size: Optional[float] = None,
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    angle: Optional[float] = None,
    lineheight: Optional[float] = None,
    margin: Any = None,
    margin_x: bool = False,
    margin_y: bool = False,
    **kwargs: Any,
) -> Any:
    """Render an ``ElementText`` as a text grob.

    When *margin_x* or *margin_y* is ``True``, wraps the result in a
    ``_TitleGrob`` whose ``width_details``/``height_details`` include
    the element's margin — matching R's ``titleGrob()`` (margins.R:88-206).
    """
    if label is None:
        return null_grob()
    gp = Gpar(
        fontfamily=family if family is not None else element.family,
        fontface=face if face is not None else element.face,
        fontsize=size if size is not None else element.size,
        col=colour if colour is not None else element.colour,
        lineheight=lineheight if lineheight is not None else element.lineheight,
    )
    hj = hjust if hjust is not None else (element.hjust if element.hjust is not None else 0.5)
    vj = vjust if vjust is not None else (element.vjust if element.vjust is not None else 0.5)
    ang = angle if angle is not None else (element.angle if element.angle is not None else 0)

    # R titleGrob (margins.R:95-107): when x/y are NULL, default them to the
    # ROTATED justification anchor inside the parent viewport.
    just_hj, just_vj = _rotate_just(ang, hj, vj)
    if x is None:
        x = Unit(just_hj, "npc")
    if y is None:
        y = Unit(just_vj, "npc")

    # If no margin wrapping requested, emit a plain text grob.
    if not margin_x and not margin_y:
        return text_grob(label=label, x=x, y=y, hjust=hj, vjust=vj, rot=ang,
                         gp=gp, **kwargs)

    # --- titleGrob (R: margins.R:88-196) ---------------------------------
    # Resolve the element's margin (R: element_text has a `margin` slot).
    el_margin = margin if margin is not None else getattr(element, "margin", None)
    if el_margin is None:
        m_t = m_r = m_b = m_l = Unit(0, "pt")
    elif isinstance(el_margin, Margin):
        m_t = Unit(el_margin.t, el_margin.unit_str)
        m_r = Unit(el_margin.r, el_margin.unit_str)
        m_b = Unit(el_margin.b, el_margin.unit_str)
        m_l = Unit(el_margin.l, el_margin.unit_str)
    else:
        m_t = m_r = m_b = m_l = Unit(0, "pt")

    # Shift x/y inward by margin before constructing the text grob
    # (R: margins.R:150, 156):
    #   new_x = x - margin[2] * just$hjust + margin[4] * (1 - just$hjust)
    #   new_y = y - margin[1] * just$vjust + margin[3] * (1 - just$vjust)
    # This is what creates the actual visual gap between the text and the
    # edge of its cell.  Without it the margin only affects cell sizing.
    if margin_x:
        if just_hj != 0:
            x = x - m_r * just_hj
        if just_hj != 1:
            x = x + m_l * (1 - just_hj)
    if margin_y:
        if just_vj != 0:
            y = y - m_t * just_vj
        if just_vj != 1:
            y = y + m_b * (1 - just_vj)

    grob = text_grob(label=label, x=x, y=y, hjust=hj, vjust=vj, rot=ang,
                     gp=gp, **kwargs)

    # --- Compute widths/heights for width_details/height_details ---------
    # R emits *lazy* grobwidth/grobheight units referencing the rotated
    # textGrob.  grid resolves them at draw time using the rotated
    # bounding box (tested: grid_py handles this correctly).
    #
    # R: width  = unit(1, "grobwidth",  grob) + x_descent
    # R: height = unit(1, "grobheight", grob) + y_descent
    # where x_descent = abs(sin(rad)) * font_descent
    #       y_descent = abs(cos(rad)) * font_descent
    import math
    from grid_py._size import calc_string_metric

    fontsize_val = size if size is not None else (element.size if element.size is not None else 12)
    # Use R's exact descender probe. R ``font_descent`` (margins.R:295)
    # runs ``grobDescent(textGrob(label = "gjpqyQ", ...))`` — the trailing
    # ``Q`` is present in R, and omitting it here was a pure probe-string
    # mismatch. Keep the string identical so descent values align with R
    # even though cairo / AFM fonts render "Q" slightly differently.
    metrics = calc_string_metric("gjpqyQ", Gpar(
        fontsize=fontsize_val,
        fontfamily=family if family is not None else element.family,
        fontface=face if face is not None else element.face,
    ))
    descent_in = metrics["descent"]

    ang_val = float(ang) if ang is not None else 0.0
    rad = math.radians(ang_val % 360)
    x_descent_unit = Unit(abs(math.sin(rad)) * descent_in, "inches")
    y_descent_unit = Unit(abs(math.cos(rad)) * descent_in, "inches")

    width = grob_width(grob) + x_descent_unit
    height = grob_height(grob) + y_descent_unit

    # Build widths/heights vectors including margins.
    # R: new_width  = unit.c(margin[4], width, margin[2])  # left, w, right
    # R: new_height = unit.c(margin[1], height, margin[3]) # top, h, bottom
    widths = unit_c(m_l, width, m_r) if margin_x else width
    heights = unit_c(m_t, height, m_b) if margin_y else height

    return _TitleGrob(grob, widths=widths, heights=heights,
                      name=kwargs.get("name", "title"))


def _grob_from_point(
    element: "ElementPoint",
    x: float = 0.5,
    y: float = 0.5,
    colour: Optional[str] = None,
    shape: Optional[int] = None,
    fill: Optional[str] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Render an ``ElementPoint`` as a points grob.

    Mirrors R's ``element_grob(element_point, ...)`` which converts
    pointsize (mm) and stroke (mm) via ``gg_par(pointsize=size, stroke=stroke)``:
        fontsize = pointsize_mm * .pt + stroke_mm * .stroke / 2
    """
    from grid_py import points_grob, Gpar
    col = colour or element.colour or "black"
    sh = shape if shape is not None else (element.shape if element.shape is not None else 19)
    fl = fill or element.fill
    sz = size if size is not None else (element.size if element.size is not None else 1.5)
    st = stroke if stroke is not None else (element.stroke if element.stroke is not None else 0.5)
    # R: gg_par(pointsize=sz, stroke=st) → fontsize = sz * .pt + st * .stroke / 2
    fontsize = float(sz) * _PT + float(st) * _STROKE / 2
    gp = Gpar(col=col, fill=fl, fontsize=fontsize)
    try:
        return points_grob(x=x, y=y, pch=int(sh), gp=gp, **kwargs)
    except Exception:
        from grid_py import null_grob
        return null_grob()


def _grob_from_polygon(
    element: "ElementPolygon",
    x=None, y=None,
    fill: Optional[str] = None,
    colour: Optional[str] = None,
    linewidth: Optional[float] = None,
    linetype: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Render an ``ElementPolygon`` as a path grob.

    Mirrors R's ``element_grob(element_polygon, ...)`` which converts
    linewidth (mm) → lwd (points) via ``gg_par(lwd = linewidth)``.
    """
    from grid_py import polygon_grob, Gpar
    if x is None:
        x = [0, 0.5, 1, 0.5]
    if y is None:
        y = [0.5, 1, 0.5, 0]
    fl = fill or element.fill or "grey20"
    col = colour or element.colour
    lwd_mm = linewidth if linewidth is not None else (element.linewidth if element.linewidth is not None else 0.5)
    lty = linetype if linetype is not None else (element.linetype if element.linetype is not None else 1)
    gp = Gpar(fill=fl, col=col, lwd=float(lwd_mm) * _PT, lty=lty)
    return polygon_grob(x=x, y=y, gp=gp, **kwargs)


def element_render(theme: Any, element_name: str, name: Optional[str] = None, **kwargs: Any) -> Any:
    """Render a named theme element into a grob.

    Parameters
    ----------
    theme : Theme
        The theme object.
    element_name : str
        The element name (e.g., ``"axis.line.x"``).
    name : str, optional
        Additional name component for the grob.
    **kwargs
        Passed through to ``element_grob()``.

    Returns
    -------
    Grob
        A grid grob for the element.
    """
    el = calc_element(element_name, theme)
    if el is None:
        return null_grob()
    grob = element_grob(el, **kwargs)
    return grob


# ---------------------------------------------------------------------------
# Element tree definition (el_def)
# ---------------------------------------------------------------------------

def el_def(
    class_: Any = None,
    inherit: Optional[Union[str, List[str]]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Define an entry in the element tree.

    Parameters
    ----------
    class_ : type or str or list of str, optional
        The expected element class (e.g. ``ElementLine``, ``"character"``).
    inherit : str or list of str, optional
        Name(s) of the parent element(s) from which this element inherits.
    description : str, optional
        Human-readable description.

    Returns
    -------
    dict
        A dictionary with keys ``"class"``, ``"inherit"``, ``"description"``.
    """
    if isinstance(inherit, str):
        inherit = [inherit]
    return {"class": class_, "inherit": inherit, "description": description}


# ---------------------------------------------------------------------------
# The default element tree
# ---------------------------------------------------------------------------

_ELEMENT_TREE: Dict[str, Dict[str, Any]] = {
    "line": el_def(ElementLine),
    "rect": el_def(ElementRect),
    "text": el_def(ElementText),
    "point": el_def(ElementPoint),
    "polygon": el_def(ElementPolygon),
    "geom": el_def(ElementGeom),
    "title": el_def(ElementText, "text"),
    "spacing": el_def("unit"),
    "margins": el_def("margin"),

    # Axis lines
    "axis.line": el_def(ElementLine, "line"),
    "axis.line.x": el_def(ElementLine, "axis.line"),
    "axis.line.x.top": el_def(ElementLine, "axis.line.x"),
    "axis.line.x.bottom": el_def(ElementLine, "axis.line.x"),
    "axis.line.y": el_def(ElementLine, "axis.line"),
    "axis.line.y.left": el_def(ElementLine, "axis.line.y"),
    "axis.line.y.right": el_def(ElementLine, "axis.line.y"),
    "axis.line.theta": el_def(ElementLine, "axis.line.x"),
    "axis.line.r": el_def(ElementLine, "axis.line.y"),

    # Axis text
    "axis.text": el_def(ElementText, "text"),
    "axis.text.x": el_def(ElementText, "axis.text"),
    "axis.text.x.top": el_def(ElementText, "axis.text.x"),
    "axis.text.x.bottom": el_def(ElementText, "axis.text.x"),
    "axis.text.y": el_def(ElementText, "axis.text"),
    "axis.text.y.left": el_def(ElementText, "axis.text.y"),
    "axis.text.y.right": el_def(ElementText, "axis.text.y"),
    "axis.text.theta": el_def(ElementText, "axis.text.x"),
    "axis.text.r": el_def(ElementText, "axis.text.y"),

    # Axis ticks
    "axis.ticks": el_def(ElementLine, "line"),
    "axis.ticks.x": el_def(ElementLine, "axis.ticks"),
    "axis.ticks.x.top": el_def(ElementLine, "axis.ticks.x"),
    "axis.ticks.x.bottom": el_def(ElementLine, "axis.ticks.x"),
    "axis.ticks.y": el_def(ElementLine, "axis.ticks"),
    "axis.ticks.y.left": el_def(ElementLine, "axis.ticks.y"),
    "axis.ticks.y.right": el_def(ElementLine, "axis.ticks.y"),
    "axis.ticks.theta": el_def(ElementLine, "axis.ticks.x"),
    "axis.ticks.r": el_def(ElementLine, "axis.ticks.y"),

    # Axis tick lengths
    "axis.ticks.length": el_def("unit_or_rel", "spacing"),
    "axis.ticks.length.x": el_def("unit_or_rel", "axis.ticks.length"),
    "axis.ticks.length.x.top": el_def("unit_or_rel", "axis.ticks.length.x"),
    "axis.ticks.length.x.bottom": el_def("unit_or_rel", "axis.ticks.length.x"),
    "axis.ticks.length.y": el_def("unit_or_rel", "axis.ticks.length"),
    "axis.ticks.length.y.left": el_def("unit_or_rel", "axis.ticks.length.y"),
    "axis.ticks.length.y.right": el_def("unit_or_rel", "axis.ticks.length.y"),
    "axis.ticks.length.theta": el_def("unit_or_rel", "axis.ticks.length.x"),
    "axis.ticks.length.r": el_def("unit_or_rel", "axis.ticks.length.y"),

    # Axis minor ticks
    "axis.minor.ticks.x.top": el_def(ElementLine, "axis.ticks.x.top"),
    "axis.minor.ticks.x.bottom": el_def(ElementLine, "axis.ticks.x.bottom"),
    "axis.minor.ticks.y.left": el_def(ElementLine, "axis.ticks.y.left"),
    "axis.minor.ticks.y.right": el_def(ElementLine, "axis.ticks.y.right"),
    "axis.minor.ticks.theta": el_def(ElementLine, "axis.ticks.theta"),
    "axis.minor.ticks.r": el_def(ElementLine, "axis.ticks.r"),

    # Axis minor tick lengths
    "axis.minor.ticks.length": el_def("unit_or_rel"),
    "axis.minor.ticks.length.x": el_def("unit_or_rel", "axis.minor.ticks.length"),
    "axis.minor.ticks.length.x.top": el_def(
        "unit_or_rel", ["axis.minor.ticks.length.x", "axis.ticks.length.x.top"]
    ),
    "axis.minor.ticks.length.x.bottom": el_def(
        "unit_or_rel", ["axis.minor.ticks.length.x", "axis.ticks.length.x.bottom"]
    ),
    "axis.minor.ticks.length.y": el_def("unit_or_rel", "axis.minor.ticks.length"),
    "axis.minor.ticks.length.y.left": el_def(
        "unit_or_rel", ["axis.minor.ticks.length.y", "axis.ticks.length.y.left"]
    ),
    "axis.minor.ticks.length.y.right": el_def(
        "unit_or_rel", ["axis.minor.ticks.length.y", "axis.ticks.length.y.right"]
    ),
    "axis.minor.ticks.length.theta": el_def(
        "unit_or_rel", ["axis.minor.ticks.length.x", "axis.ticks.length.theta"]
    ),
    "axis.minor.ticks.length.r": el_def(
        "unit_or_rel", ["axis.minor.ticks.length.y", "axis.ticks.length.r"]
    ),

    # Axis titles
    "axis.title": el_def(ElementText, "title"),
    "axis.title.x": el_def(ElementText, "axis.title"),
    "axis.title.x.top": el_def(ElementText, "axis.title.x"),
    "axis.title.x.bottom": el_def(ElementText, "axis.title.x"),
    "axis.title.y": el_def(ElementText, "axis.title"),
    "axis.title.y.left": el_def(ElementText, "axis.title.y"),
    "axis.title.y.right": el_def(ElementText, "axis.title.y"),

    # Legend
    "legend.background": el_def(ElementRect, "rect"),
    "legend.margin": el_def("margin", "margins"),
    "legend.spacing": el_def("unit_or_rel", "spacing"),
    "legend.spacing.x": el_def("unit_or_rel", "legend.spacing"),
    "legend.spacing.y": el_def("unit_or_rel", "legend.spacing"),
    "legend.key": el_def(ElementRect, "panel.background"),
    "legend.key.size": el_def("unit_or_rel", "spacing"),
    "legend.key.height": el_def("unit_or_rel", "legend.key.size"),
    "legend.key.width": el_def("unit_or_rel", "legend.key.size"),
    "legend.key.spacing": el_def("unit_or_rel", "spacing"),
    "legend.key.spacing.x": el_def("unit_or_rel", "legend.key.spacing"),
    "legend.key.spacing.y": el_def("unit_or_rel", "legend.key.spacing"),
    "legend.key.justification": el_def("character"),
    "legend.frame": el_def(ElementRect, "rect"),
    "legend.axis.line": el_def(ElementLine, "line"),
    "legend.ticks": el_def(ElementLine, "legend.axis.line"),
    "legend.ticks.length": el_def("unit_or_rel", "legend.key.size"),
    "legend.text": el_def(ElementText, "text"),
    "legend.text.position": el_def("character"),
    "legend.title": el_def(ElementText, "title"),
    "legend.title.position": el_def("character"),
    "legend.byrow": el_def("logical"),
    "legend.position": el_def("character"),
    "legend.position.inside": el_def("numeric"),
    "legend.direction": el_def("character"),
    "legend.justification": el_def("character"),
    "legend.justification.top": el_def("character", "legend.justification"),
    "legend.justification.bottom": el_def("character", "legend.justification"),
    "legend.justification.left": el_def("character", "legend.justification"),
    "legend.justification.right": el_def("character", "legend.justification"),
    "legend.justification.inside": el_def("character", "legend.justification"),
    "legend.location": el_def("character"),
    "legend.box": el_def("character"),
    "legend.box.just": el_def("character"),
    "legend.box.margin": el_def("margin", "margins"),
    "legend.box.background": el_def(ElementRect, "rect"),
    "legend.box.spacing": el_def("unit_or_rel", "spacing"),

    # Panel
    "panel.background": el_def(ElementRect, "rect"),
    "panel.border": el_def(ElementRect, "rect"),
    "panel.spacing": el_def("unit_or_rel", "spacing"),
    "panel.spacing.x": el_def("unit_or_rel", "panel.spacing"),
    "panel.spacing.y": el_def("unit_or_rel", "panel.spacing"),
    "panel.grid": el_def(ElementLine, "line"),
    "panel.grid.major": el_def(ElementLine, "panel.grid"),
    "panel.grid.minor": el_def(ElementLine, "panel.grid"),
    "panel.grid.major.x": el_def(ElementLine, "panel.grid.major"),
    "panel.grid.major.y": el_def(ElementLine, "panel.grid.major"),
    "panel.grid.minor.x": el_def(ElementLine, "panel.grid.minor"),
    "panel.grid.minor.y": el_def(ElementLine, "panel.grid.minor"),
    "panel.ontop": el_def("logical"),
    "panel.widths": el_def("unit"),
    "panel.heights": el_def("unit"),

    # Strip
    "strip.background": el_def(ElementRect, "rect"),
    "strip.background.x": el_def(ElementRect, "strip.background"),
    "strip.background.y": el_def(ElementRect, "strip.background"),
    "strip.clip": el_def("character"),
    "strip.text": el_def(ElementText, "text"),
    "strip.text.x": el_def(ElementText, "strip.text"),
    "strip.text.x.top": el_def(ElementText, "strip.text.x"),
    "strip.text.x.bottom": el_def(ElementText, "strip.text.x"),
    "strip.text.y": el_def(ElementText, "strip.text"),
    "strip.text.y.left": el_def(ElementText, "strip.text.y"),
    "strip.text.y.right": el_def(ElementText, "strip.text.y"),
    "strip.placement": el_def("character"),
    "strip.placement.x": el_def("character", "strip.placement"),
    "strip.placement.y": el_def("character", "strip.placement"),
    "strip.switch.pad.grid": el_def("unit_or_rel", "spacing"),
    "strip.switch.pad.wrap": el_def("unit_or_rel", "spacing"),

    # Plot
    "plot.background": el_def(ElementRect, "rect"),
    "plot.title": el_def(ElementText, "title"),
    "plot.title.position": el_def("character"),
    "plot.subtitle": el_def(ElementText, "text"),
    "plot.caption": el_def(ElementText, "text"),
    "plot.caption.position": el_def("character"),
    "plot.tag": el_def(ElementText, "text"),
    "plot.tag.position": el_def("character"),
    "plot.tag.location": el_def("character"),
    "plot.margin": el_def("margin", "margins"),

    # Aspect ratio
    "aspect.ratio": el_def("numeric"),
}


# ---------------------------------------------------------------------------
# Global element-tree state
# ---------------------------------------------------------------------------

class _ThemeGlobal:
    """Module-level singleton holding current theme and element tree state."""

    def __init__(self) -> None:
        self.element_tree: Dict[str, Dict[str, Any]] = dict(_ELEMENT_TREE)
        self.theme_default: Any = None
        self.theme_current: Any = None


_ggplot_global = _ThemeGlobal()


def get_element_tree() -> Dict[str, Dict[str, Any]]:
    """Return the currently active element tree.

    Returns
    -------
    dict
        A mapping of element names to their definitions (created by ``el_def``).
    """
    return _ggplot_global.element_tree


def register_theme_elements(
    element_tree: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> None:
    """Register new theme elements globally.

    Parameters
    ----------
    element_tree : dict, optional
        Additional element tree entries (name -> ``el_def(...)``).
    **kwargs
        Element default values to merge into the default theme.
    """
    if element_tree is not None:
        _ggplot_global.element_tree.update(element_tree)
    # Defaults are handled by the theme module once it is imported.


def reset_theme_settings(reset_current: bool = True) -> None:
    """Reset the element tree and default theme to built-in defaults.

    Mirrors R's ``reset_theme_settings()`` (theme-elements.R:714-723):
    restores the element tree, sets ``theme_default = theme_grey()``,
    and (unless disabled) sets ``theme_current = theme_default``.

    Parameters
    ----------
    reset_current : bool
        If ``True`` (default), also reset the currently active theme.
    """
    _ggplot_global.element_tree = dict(_ELEMENT_TREE)
    # Local import: theme_defaults depends on this module (circular at top level).
    from ggplot2_py.theme_defaults import theme_grey
    _ggplot_global.theme_default = theme_grey()
    if reset_current:
        _ggplot_global.theme_current = _ggplot_global.theme_default


# ---------------------------------------------------------------------------
# calc_element — element inheritance resolution
# ---------------------------------------------------------------------------

def calc_element(
    element: str,
    theme: Any,
    verbose: bool = False,
    skip_blank: bool = False,
) -> Any:
    """Resolve a theme element by walking the inheritance tree.

    Parameters
    ----------
    element : str
        Name of the element to resolve (e.g. ``"axis.text.x"``).
    theme : Theme
        The theme object.
    verbose : bool
        If ``True``, print inheritance chain.
    skip_blank : bool
        If ``True``, skip ``element_blank`` ancestors.

    Returns
    -------
    Element or other
        The fully resolved element, or ``None`` if not found.
    """
    if verbose:
        print(f"{element} --> ", end="")

    # Look up the element value in the theme
    el_out = theme.get(element) if hasattr(theme, "get") else getattr(theme, element, None)

    # If blank, decide whether to skip
    if isinstance(el_out, ElementBlank):
        if skip_blank:
            el_out = None
        else:
            if verbose:
                print("element_blank (no inheritance)")
            return el_out

    # Get element tree
    element_tree = get_element_tree()

    # Validate element class against tree definition (R: check_element)
    tree_entry = element_tree.get(element)
    if tree_entry is None:
        if verbose:
            print("(not in element tree)")
        return el_out

    if el_out is not None and not isinstance(el_out, ElementBlank):
        expected_class = tree_entry.get("class")
        if expected_class is not None and isinstance(expected_class, type):
            if not isinstance(el_out, (expected_class, ElementBlank)):
                import warnings
                warnings.warn(
                    f"Theme element '{element}' must be a "
                    f"{expected_class.__name__} object, "
                    f"got {type(el_out).__name__}.",
                    stacklevel=3,
                )

    # Get parent names
    pnames = tree_entry.get("inherit")

    # If no parents, this is a root node
    if pnames is None:
        if verbose:
            print("(top level)")

        if el_out is not None:
            # Check for None properties
            if isinstance(el_out, Element):
                null_props = [k for k, v in el_out.__dict__.items() if v is None]
            else:
                null_props = []
            if not null_props:
                return el_out

            # Try to fill from default theme
            default_theme = _ggplot_global.theme_default
            if default_theme is not None:
                default_el = (
                    default_theme.get(element)
                    if hasattr(default_theme, "get")
                    else getattr(default_theme, element, None)
                )
                el_out = combine_elements(el_out, default_el)

        return el_out

    if verbose:
        print(f"{pnames}")

    # If el_out has inherit_blank=False, start skipping blanks
    if (
        not skip_blank
        and el_out is not None
        and isinstance(el_out, Element)
        and not getattr(el_out, "inherit_blank", True)
    ):
        skip_blank = True

    # Recursively calculate parents
    parents = [
        calc_element(pname, theme, verbose=verbose, skip_blank=skip_blank)
        for pname in pnames
    ]

    # Combine with parents using reduce
    result = el_out
    for parent in parents:
        result = combine_elements(result, parent)

    return result

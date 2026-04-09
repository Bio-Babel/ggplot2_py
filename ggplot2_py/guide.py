"""
Guide system for ggplot2_py.

Ports the R guide infrastructure (``guide-.R``, ``guide-legend.R``,
``guide-colorbar.R``, ``guide-axis.R``, ``guide-none.R``, ``guide-bins.R``,
``guide-colorsteps.R``, ``guide-custom.R``, ``guide-axis-logticks.R``,
``guide-axis-stack.R``, ``guide-axis-theta.R``, ``guide-old.R``, and
``guides-.R``) into a unified Python module.

The module defines:

* **Guide** -- base GGProto class for all guides.
* Concrete guide classes (``GuideLegend``, ``GuideColourbar``, etc.).
* Constructor functions (``guide_legend()``, ``guide_colourbar()``, etc.).
* The **Guides** container and the ``guides()`` helper.
* Legacy S3-style shims (``guide_train``, ``guide_merge``, etc.).
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py.ggproto import GGProto, ggproto, ggproto_parent, is_ggproto
from ggplot2_py._compat import (
    Waiver,
    cli_abort,
    cli_warn,
    is_waiver,
    waiver,
)
from ggplot2_py._utils import compact, modify_list, snake_class
from ggplot2_py.aes import standardise_aes_names, rename_aes

__all__ = [
    # Classes
    "Guide",
    "GuideAxis",
    "GuideAxisLogticks",
    "GuideAxisStack",
    "GuideAxisTheta",
    "GuideBins",
    "GuideColourbar",
    "GuideColoursteps",
    "GuideCustom",
    "GuideLegend",
    "GuideNone",
    "GuideOld",
    # Constructors
    "guide_axis",
    "guide_axis_logticks",
    "guide_axis_stack",
    "guide_axis_theta",
    "guide_bins",
    "guide_colourbar",
    "guide_colorbar",
    "guide_coloursteps",
    "guide_colorsteps",
    "guide_custom",
    "guide_legend",
    "guide_none",
    # Guides container
    "guides",
    "Guides",
    # Helpers
    "new_guide",
    "old_guide",
    "guide_gengrob",
    "guide_geom",
    "guide_merge",
    "guide_train",
    "guide_transform",
    "is_guide",
    "is_guides",
]


# ---------------------------------------------------------------------------
# Positional constants
# ---------------------------------------------------------------------------

_TRBL: List[str] = ["top", "right", "bottom", "left"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _hash_object(obj: Any) -> str:
    """Return a deterministic hash string for *obj*.

    Parameters
    ----------
    obj : Any
        The object to hash.  Converted to ``repr`` then hashed with MD5.

    Returns
    -------
    str
        Hexadecimal hash digest.
    """
    return hashlib.md5(repr(obj).encode("utf-8")).hexdigest()


def _defaults(target: dict, defaults: dict) -> dict:
    """Return a new dict with *defaults* filled in for missing keys.

    Parameters
    ----------
    target : dict
        Primary values.
    defaults : dict
        Fall-back values.

    Returns
    -------
    dict
        Merged dictionary.
    """
    out = dict(defaults)
    out.update(target)
    return out


def _validate_guide(guide: Any) -> Any:
    """Ensure *guide* is a Guide class/instance.

    Parameters
    ----------
    guide : str or Guide
        Either a guide shorthand name (e.g. ``"legend"``) or a Guide
        class / instance.

    Returns
    -------
    Guide
        A validated guide object.

    Raises
    ------
    ValueError
        If *guide* cannot be resolved.
    """
    if isinstance(guide, str):
        guide = _resolve_guide_name(guide)
    if isinstance(guide, type) and issubclass(guide, GGProto):
        # It is a class; instantiate with default params
        return guide()
    if isinstance(guide, GGProto):
        return guide
    cli_abort(f"Cannot resolve guide: {guide!r}")


def _resolve_guide_name(name: str) -> type:
    """Map a short string name to a Guide class.

    Parameters
    ----------
    name : str
        Short name, e.g. ``"legend"``, ``"colourbar"``, ``"none"``.

    Returns
    -------
    type
        The corresponding Guide class.
    """
    _REGISTRY: Dict[str, type] = {
        "none": GuideNone,
        "legend": GuideLegend,
        "colourbar": GuideColourbar,
        "colorbar": GuideColourbar,
        "coloursteps": GuideColoursteps,
        "colorsteps": GuideColoursteps,
        "bins": GuideBins,
        "axis": GuideAxis,
        "axis_logticks": GuideAxisLogticks,
        "axis_theta": GuideAxisTheta,
        "axis_stack": GuideAxisStack,
        "custom": GuideCustom,
    }
    key = name.lower().replace("-", "_")
    cls = _REGISTRY.get(key)
    if cls is None:
        cli_abort(f"Unknown guide type: {name!r}")
    return cls


# ============================================================================
# Guide base class
# ============================================================================

class Guide(GGProto):
    """Base class for all ggplot2 guides.

    A ``Guide`` is responsible for rendering the visual representation of a
    scale -- axis tick marks, legends, colour bars, and so on.

    Attributes
    ----------
    params : dict
        Default parameters.  Subclasses extend this dict.
    elements : dict
        Theme element names used by this guide.
    hashables : list[str]
        Parameter keys used to compute a deduplication hash.
    available_aes : list[str]
        Aesthetics that this guide can represent.
    """

    _class_name: str = "Guide"

    # -- Fields --------------------------------------------------------------

    params: Dict[str, Any] = {
        "title": waiver(),
        "theme": None,
        "name": "",
        "position": waiver(),
        "direction": None,
        "order": 0,
        "hash": "",
    }

    available_aes: List[str] = []

    elements: Dict[str, str] = {}

    hashables: List[str] = ["title", "name"]

    # -- Key extraction ------------------------------------------------------

    @staticmethod
    def extract_key(
        scale: Any,
        aesthetic: str,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """Extract key (break positions / labels) from a scale.

        Parameters
        ----------
        scale : Scale
            The scale from which to extract breaks.
        aesthetic : str
            Name of the aesthetic this guide represents.
        **kwargs : Any
            Additional arguments forwarded by subclasses.

        Returns
        -------
        pd.DataFrame or None
            A DataFrame with columns for the aesthetic, ``.value``, and
            ``.label``; or ``None`` if the scale has no breaks.
        """
        breaks = getattr(scale, "get_breaks", lambda: None)()
        if breaks is None:
            return None
        mapped = getattr(scale, "map", lambda x: x)(breaks)
        labels = getattr(scale, "get_labels", lambda x: x)(breaks)

        key = pd.DataFrame({
            aesthetic: mapped,
            ".value": breaks,
            ".label": labels if labels is not None else [str(b) for b in breaks],
        })
        return key

    @staticmethod
    def extract_decor(
        scale: Any,
        aesthetic: str,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """Extract decoration data from a scale.

        Parameters
        ----------
        scale : Scale
            The scale.
        aesthetic : str
            Aesthetic name.
        **kwargs : Any
            Extra arguments.

        Returns
        -------
        pd.DataFrame or None
            Decoration data or ``None``.
        """
        return None

    @staticmethod
    def extract_params(
        scale: Any,
        params: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Post-process guide parameters after extraction.

        Parameters
        ----------
        scale : Scale
            The source scale.
        params : dict
            Current guide parameters.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        dict
            Possibly-modified parameters.
        """
        title = kwargs.get("title", waiver())
        scale_name = getattr(scale, "name", None)
        if is_waiver(params.get("title")):
            if not is_waiver(title):
                params["title"] = title
            elif scale_name is not None:
                params["title"] = scale_name
        return params

    # -- Training / transform ------------------------------------------------

    def train(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale: Any = None,
        aesthetic: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Train the guide on a scale.

        Parameters
        ----------
        params : dict, optional
            Guide parameters.
        scale : Scale, optional
            The scale to train on.
        aesthetic : str, optional
            Aesthetic name.
        **kwargs : Any
            Extra arguments (e.g. ``title``).

        Returns
        -------
        dict or None
            Updated parameters, or ``None`` to drop this guide.
        """
        if params is None:
            params = dict(self.params)
        if scale is None:
            return params

        params["aesthetic"] = aesthetic or ""

        # Extract key — mirrors R's inject(self$extract_key(scale, !!!params))
        key = self.extract_key(scale, **{k: v for k, v in params.items()
                                         if k != "key"}, **kwargs)
        if key is not None and key.empty:
            return None
        params["key"] = key

        # Extract decor — mirrors R's inject(self$extract_decor(scale, !!!params))
        # Pass scale as positional, then splat all params as kwargs (except
        # "decor" itself to avoid recursion).
        params["decor"] = self.extract_decor(
            scale, **{k: v for k, v in params.items()
                      if k != "decor"},
        )

        # Post-process
        params = self.extract_params(scale, params, **kwargs)

        # Compute hash
        hash_vals = []
        for h in self.hashables:
            if h in params:
                hash_vals.append(params[h])
            elif isinstance(params.get("key"), pd.DataFrame) and h.startswith("key."):
                col = h.split(".", 1)[1]
                if col in params["key"].columns:
                    hash_vals.append(list(params["key"][col]))
        params["hash"] = _hash_object(hash_vals)

        return params

    @staticmethod
    def transform(
        params: Dict[str, Any],
        coord: Any,
        panel_params: Any,
    ) -> Dict[str, Any]:
        """Transform guide data through coordinate system.

        Parameters
        ----------
        params : dict
            Guide parameters including ``key`` and ``decor``.
        coord : Coord
            Coordinate system.
        panel_params : object
            Panel parameters from the coordinate system.

        Returns
        -------
        dict
            Parameters with transformed ``key`` / ``decor``.
        """
        key = params.get("key")
        if key is not None and hasattr(coord, "transform") and not key.empty:
            params["key"] = coord.transform(key, panel_params)
        return params

    def get_layer_key(
        self,
        params: Dict[str, Any],
        layers: List[Any],
        data: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Map layer key information into the guide parameters.

        Parameters
        ----------
        params : dict
            Guide parameters.
        layers : list
            Plot layers.
        data : list, optional
            Layer data.

        Returns
        -------
        dict
            Updated parameters.
        """
        return params

    def process_layers(
        self,
        params: Dict[str, Any],
        layers: List[Any],
        data: Optional[List[Any]] = None,
        theme: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Process layer information to generate geom info.

        Parameters
        ----------
        params : dict
            Guide parameters.
        layers : list
            Plot layers.
        data : list, optional
            Layer data.
        theme : Theme, optional
            Plot theme.

        Returns
        -------
        dict or None
            Updated parameters or ``None`` if guide should be dropped.
        """
        return self.get_layer_key(params, layers, data)

    # -- Setup / override ----------------------------------------------------

    @staticmethod
    def setup_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set up parameters before drawing.

        Parameters
        ----------
        params : dict
            Guide parameters.

        Returns
        -------
        dict
            Validated parameters.
        """
        return params

    @staticmethod
    def override_elements(
        params: Dict[str, Any],
        elements: Dict[str, Any],
        theme: Any,
    ) -> Dict[str, Any]:
        """Resolve theme elements for this guide.

        Parameters
        ----------
        params : dict
            Guide parameters.
        elements : dict
            Element name -> theme element name mapping.
        theme : Theme
            The plot theme.

        Returns
        -------
        dict
            Resolved element objects.
        """
        return elements

    def setup_elements(
        self,
        params: Dict[str, Any],
        elements: Optional[Dict[str, str]] = None,
        theme: Any = None,
    ) -> Dict[str, Any]:
        """Set up theme elements used by this guide.

        Parameters
        ----------
        params : dict
            Guide parameters.
        elements : dict, optional
            Element specifications.  Falls back to ``self.elements``.
        theme : Theme, optional
            Plot theme.

        Returns
        -------
        dict
            Resolved elements.
        """
        if elements is None:
            elements = dict(self.elements)
        return self.override_elements(params, elements, theme)

    # -- Build methods -------------------------------------------------------

    @staticmethod
    def build_title(
        label: Any,
        elements: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Any:
        """Build the guide title grob.

        Parameters
        ----------
        label : str or None
            Title text.
        elements : dict
            Resolved theme elements.
        params : dict
            Guide parameters.

        Returns
        -------
        grob
            A title grob or ``None``.
        """
        return None

    @staticmethod
    def build_labels(
        key: pd.DataFrame,
        elements: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Any:
        """Build label grobs from the key.

        Parameters
        ----------
        key : pd.DataFrame
            The guide key.
        elements : dict
            Resolved theme elements.
        params : dict
            Guide parameters.

        Returns
        -------
        grob or list of grobs
            Label grobs.
        """
        return None

    @staticmethod
    def build_decor(
        decor: Any,
        grobs: Any,
        elements: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Any:
        """Build decoration grobs.

        Parameters
        ----------
        decor : pd.DataFrame or None
            Decoration data.
        grobs : dict
            Previously built grobs.
        elements : dict
            Resolved theme elements.
        params : dict
            Guide parameters.

        Returns
        -------
        grob or list of grobs
            Decoration grobs.
        """
        return None

    @staticmethod
    def build_ticks(
        key: pd.DataFrame,
        elements: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Any:
        """Build tick mark grobs.

        Parameters
        ----------
        key : pd.DataFrame
            The guide key.
        elements : dict
            Resolved theme elements.
        params : dict
            Guide parameters.

        Returns
        -------
        grob
            Tick mark grobs.
        """
        return None

    @staticmethod
    def measure_grobs(
        grobs: Dict[str, Any],
        params: Dict[str, Any],
        elements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Measure built grobs for layout.

        Parameters
        ----------
        grobs : dict
            Named dictionary of grobs.
        params : dict
            Guide parameters.
        elements : dict
            Resolved elements.

        Returns
        -------
        dict
            Dictionary with ``width`` and ``height`` keys.
        """
        return {"width": None, "height": None}

    @staticmethod
    def arrange_layout(
        key: pd.DataFrame,
        sizes: Dict[str, Any],
        params: Dict[str, Any],
        elements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute the layout specification.

        Parameters
        ----------
        key : pd.DataFrame
            The guide key.
        sizes : dict
            Size measurements from :meth:`measure_grobs`.
        params : dict
            Guide parameters.
        elements : dict
            Resolved elements.

        Returns
        -------
        dict
            Layout specification.
        """
        return {}

    @staticmethod
    def assemble_drawing(
        grobs: Dict[str, Any],
        layout: Dict[str, Any],
        sizes: Dict[str, Any],
        params: Dict[str, Any],
        elements: Dict[str, Any],
    ) -> Any:
        """Assemble the final guide drawing (gtable).

        Parameters
        ----------
        grobs : dict
            Named grobs.
        layout : dict
            Layout specification.
        sizes : dict
            Size measurements.
        params : dict
            Guide parameters.
        elements : dict
            Resolved elements.

        Returns
        -------
        gtable or grob
            The final assembled guide graphic.
        """
        return None

    # -- Merge ---------------------------------------------------------------

    def merge(
        self,
        params: Dict[str, Any],
        new_guide: "Guide",
        new_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge another guide into this one.

        Parameters
        ----------
        params : dict
            This guide's parameters.
        new_guide : Guide
            The other guide.
        new_params : dict
            The other guide's parameters.

        Returns
        -------
        dict
            A dict with keys ``guide`` and ``params`` representing the
            merged result.
        """
        new_key = new_params.get("key")
        if new_key is not None and isinstance(new_key, pd.DataFrame):
            key = params.get("key")
            if key is not None and isinstance(key, pd.DataFrame):
                # Merge keys by joining on shared columns
                common = [c for c in key.columns if c in new_key.columns
                          and c.startswith(".")]
                if common:
                    new_cols = [c for c in new_key.columns if c not in common]
                    if new_cols:
                        params["key"] = pd.merge(
                            key, new_key[common + new_cols],
                            on=common, how="left",
                        )
                else:
                    # Just add new aesthetic columns
                    for col in new_key.columns:
                        if col not in key.columns:
                            params["key"][col] = new_key[col].values
        return {"guide": self, "params": params}

    # -- Draw ----------------------------------------------------------------

    def draw(
        self,
        theme: Any = None,
        position: Optional[str] = None,
        direction: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Draw the guide.

        Parameters
        ----------
        theme : Theme, optional
            Plot theme.
        position : str, optional
            Position (``"top"``, ``"right"``, ``"bottom"``, ``"left"``,
            ``"inside"``).
        direction : str, optional
            ``"horizontal"`` or ``"vertical"``.
        params : dict, optional
            Guide parameters.  Defaults to ``self.params``.

        Returns
        -------
        grob or gtable
            The rendered guide.
        """
        if params is None:
            params = dict(self.params)

        # Update position/direction if provided
        if position is not None:
            params["position"] = position
        if direction is not None:
            params["direction"] = direction

        params = self.setup_params(params)
        elems = self.setup_elements(params, dict(self.elements), theme)

        # Build components
        key = params.get("key")
        if key is None:
            return None

        grobs: Dict[str, Any] = {}
        title = params.get("title")
        if not is_waiver(title) and title is not None:
            grobs["title"] = self.build_title(title, elems, params)

        grobs["labels"] = self.build_labels(key, elems, params)
        grobs["ticks"] = self.build_ticks(key, elems, params)

        decor = params.get("decor")
        grobs["decor"] = self.build_decor(decor, grobs, elems, params)

        sizes = self.measure_grobs(grobs, params, elems)
        layout = self.arrange_layout(key, sizes, params, elems)

        return self.assemble_drawing(grobs, layout, sizes, params, elems)


# ============================================================================
# GuideNone -- suppresses the guide
# ============================================================================

class GuideNone(Guide):
    """A guide that draws nothing.

    Attributes
    ----------
    _class_name : str
        ``"GuideNone"``.
    """

    _class_name: str = "GuideNone"

    params: Dict[str, Any] = {
        "title": waiver(),
        "theme": None,
        "name": "none",
        "position": waiver(),
        "direction": None,
        "order": 0,
        "hash": "",
    }

    available_aes: List[str] = ["any"]

    def train(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale: Any = None,
        aesthetic: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Perform no training.

        Returns
        -------
        dict
            The unmodified parameters.
        """
        return params if params is not None else dict(self.params)

    @staticmethod
    def transform(params: Dict[str, Any], coord: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Pass through without transformation.

        Returns
        -------
        dict
            Unmodified parameters.
        """
        return params

    def draw(self, **kwargs: Any) -> None:
        """Draw nothing.

        Returns
        -------
        None
        """
        return None


# ============================================================================
# GuideAxis -- position axis guide
# ============================================================================

class GuideAxis(Guide):
    """Guide for position axes (x / y).

    Renders tick marks, labels, and axis lines for position scales.

    Attributes
    ----------
    _class_name : str
        ``"GuideAxis"``.
    """

    _class_name: str = "GuideAxis"

    params: Dict[str, Any] = {
        "title": waiver(),
        "theme": None,
        "name": "axis",
        "hash": "",
        "position": waiver(),
        "direction": None,
        "angle": None,
        "n.dodge": 1,
        "minor.ticks": False,
        "cap": "none",
        "order": 0,
        "check.overlap": False,
    }

    available_aes: List[str] = ["x", "y"]

    hashables: List[str] = ["title", "name"]

    elements: Dict[str, str] = {
        "line": "axis.line",
        "text": "axis.text",
        "ticks": "axis.ticks",
        "minor": "axis.minor.ticks",
        "major_length": "axis.ticks.length",
        "minor_length": "axis.minor.ticks.length",
    }

    @staticmethod
    def extract_key(
        scale: Any,
        aesthetic: str,
        minor_ticks: bool = False,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """Extract break positions for axis guide.

        Parameters
        ----------
        scale : Scale
            Position scale.
        aesthetic : str
            ``"x"`` or ``"y"``.
        minor_ticks : bool
            Whether to include minor tick positions.
        **kwargs : Any
            Extra arguments.

        Returns
        -------
        pd.DataFrame or None
            Key with break positions.
        """
        major = Guide.extract_key(scale, aesthetic)
        if major is None:
            major = pd.DataFrame()
        if not minor_ticks:
            return major

        minor_breaks = getattr(scale, "get_breaks_minor", lambda: [])()
        if minor_breaks is None:
            minor_breaks = []
        if major is not None and not major.empty:
            major_vals = set(major[".value"].tolist())
            minor_breaks = [b for b in minor_breaks
                           if b not in major_vals and np.isfinite(b)]
        else:
            minor_breaks = [b for b in minor_breaks if np.isfinite(b)]

        if not minor_breaks:
            return major

        mapped = getattr(scale, "map", lambda x: x)(minor_breaks)
        minor = pd.DataFrame({
            aesthetic: mapped,
            ".value": minor_breaks,
            ".type": ["minor"] * len(minor_breaks),
        })

        if major is not None and not major.empty:
            major = major.copy()
            major[".type"] = "major"
            return pd.concat([major, minor], ignore_index=True)
        return minor

    @staticmethod
    def extract_params(
        scale: Any,
        params: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Append aesthetic name to the guide name.

        Parameters
        ----------
        scale : Scale
            The position scale.
        params : dict
            Guide parameters.
        **kwargs : Any
            Extra arguments.

        Returns
        -------
        dict
            Updated parameters.
        """
        aes = params.get("aesthetic", "")
        params["name"] = f"{params.get('name', 'axis')}_{aes}"
        return params

    @staticmethod
    def extract_decor(
        scale: Any,
        aesthetic: str,
        key: Optional[pd.DataFrame] = None,
        cap: str = "none",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Build axis line decoration data.

        Parameters
        ----------
        scale : Scale
            The position scale.
        aesthetic : str
            ``"x"`` or ``"y"``.
        key : pd.DataFrame, optional
            The guide key.
        cap : str
            One of ``"none"``, ``"both"``, ``"upper"``, ``"lower"``.
        **kwargs : Any
            Extra arguments.

        Returns
        -------
        pd.DataFrame
            Axis line positions.
        """
        value = [-np.inf, np.inf]
        has_key = key is not None and not key.empty
        if cap in ("both", "upper") and has_key:
            value[1] = key[aesthetic].max()
        if cap in ("both", "lower") and has_key:
            value[0] = key[aesthetic].min()
        return pd.DataFrame({aesthetic: value})

    @staticmethod
    def transform(
        params: Dict[str, Any],
        coord: Any,
        panel_params: Any,
    ) -> Dict[str, Any]:
        """Transform axis data through coordinate system.

        Parameters
        ----------
        params : dict
            Guide parameters.
        coord : Coord
            Coordinate system.
        panel_params : object
            Panel parameters.

        Returns
        -------
        dict
            Transformed parameters.
        """
        key = params.get("key")
        if key is not None and hasattr(coord, "transform") and not key.empty:
            aesthetic = params.get("aesthetic", "x")
            ortho = "y" if aesthetic == "x" else "x"
            position = params.get("position")
            if position in ("bottom", "left"):
                override = -np.inf
            else:
                override = np.inf

            if not key.empty:
                if ortho not in key.columns:
                    key = key.copy()
                    key[ortho] = override
                params["key"] = coord.transform(key, panel_params)

        decor = params.get("decor")
        if decor is not None and hasattr(coord, "transform"):
            aesthetic = params.get("aesthetic", "x")
            ortho = "y" if aesthetic == "x" else "x"
            if ortho not in decor.columns:
                decor = decor.copy()
                position = params.get("position")
                decor[ortho] = -np.inf if position in ("bottom", "left") else np.inf
            params["decor"] = coord.transform(decor, panel_params)

        return params


# ============================================================================
# GuideLegend -- legend for non-position aesthetics
# ============================================================================

class GuideLegend(Guide):
    """Legend guide for non-position aesthetics.

    Shows keys (geoms) mapped onto discrete or discretised values.

    Attributes
    ----------
    _class_name : str
        ``"GuideLegend"``.
    """

    _class_name: str = "GuideLegend"

    params: Dict[str, Any] = {
        "title": waiver(),
        "theme": None,
        "override.aes": {},
        "nrow": None,
        "ncol": None,
        "reverse": False,
        "order": 0,
        "name": "legend",
        "hash": "",
        "position": None,
        "direction": None,
    }

    available_aes: List[str] = ["any"]

    hashables: List[str] = ["title", "name"]

    elements: Dict[str, str] = {
        "background": "legend.background",
        "margin": "legend.margin",
        "key": "legend.key",
        "key_height": "legend.key.height",
        "key_width": "legend.key.width",
        "key_just": "legend.key.justification",
        "text": "legend.text",
        "theme.title": "legend.title",
        "spacing_x": "legend.key.spacing.x",
        "spacing_y": "legend.key.spacing.y",
        "text_position": "legend.text.position",
        "title_position": "legend.title.position",
        "byrow": "legend.byrow",
    }

    @staticmethod
    def extract_params(
        scale: Any,
        params: Dict[str, Any],
        title: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Extract and validate legend parameters.

        Parameters
        ----------
        scale : Scale
            The mapped scale.
        params : dict
            Guide parameters.
        title : str or Waiver, optional
            Title override.
        **kwargs : Any
            Extra arguments.

        Returns
        -------
        dict
            Updated parameters.
        """
        if title is None:
            title = waiver()
        # Resolve title
        scale_name = getattr(scale, "name", None)
        if is_waiver(params.get("title")):
            if not is_waiver(title):
                params["title"] = title
            elif scale_name is not None:
                params["title"] = scale_name

        # Reverse key order if requested
        if params.get("reverse", False):
            key = params.get("key")
            if key is not None and isinstance(key, pd.DataFrame) and not key.empty:
                params["key"] = key.iloc[::-1].reset_index(drop=True)
        return params


# ============================================================================
# GuideColourbar -- continuous colour bar guide
# ============================================================================

class GuideColourbar(GuideLegend):
    """Continuous colour bar guide.

    Shows a smooth colour gradient representing continuous colour/fill
    scales.

    Attributes
    ----------
    _class_name : str
        ``"GuideColourbar"``.
    """

    _class_name: str = "GuideColourbar"

    params: Dict[str, Any] = {
        "title": waiver(),
        "theme": None,
        "nbin": 300,
        "display": "raster",
        "alpha": float("nan"),
        "draw_lim": [True, True],
        "angle": None,
        "position": None,
        "direction": None,
        "reverse": False,
        "order": 0,
        "name": "colourbar",
        "hash": "",
    }

    available_aes: List[str] = ["colour", "color", "fill"]

    hashables: List[str] = ["title", "name"]

    elements: Dict[str, str] = {
        "background": "legend.background",
        "margin": "legend.margin",
        "key": "legend.key",
        "key_height": "legend.key.height",
        "key_width": "legend.key.width",
        "text": "legend.text",
        "theme.title": "legend.title",
        "ticks": "legend.ticks",
        "ticks_length": "legend.ticks.length",
        "frame": "legend.frame",
        "text_position": "legend.text.position",
        "title_position": "legend.title.position",
    }


# ============================================================================
# GuideColoursteps -- stepped colour bar guide
# ============================================================================

class GuideColoursteps(GuideColourbar):
    """Discretised (stepped) colour bar guide.

    Displays areas between breaks as single constant colours instead of
    a smooth gradient.

    Attributes
    ----------
    _class_name : str
        ``"GuideColoursteps"``.
    """

    _class_name: str = "GuideColoursteps"

    params: Dict[str, Any] = {
        **GuideColourbar.params,
        "even.steps": True,
        "show.limits": None,
        "name": "coloursteps",
    }

    available_aes: List[str] = ["colour", "color", "fill"]


# ============================================================================
# GuideBins -- binned legend guide
# ============================================================================

class GuideBins(GuideLegend):
    """Binned legend guide.

    A version of the legend guide for binned scales.  Places ticks between
    keys and optionally shows a small axis.

    Attributes
    ----------
    _class_name : str
        ``"GuideBins"``.
    """

    _class_name: str = "GuideBins"

    params: Dict[str, Any] = {
        **GuideLegend.params,
        "angle": None,
        "show.limits": None,
        "name": "bins",
    }

    available_aes: List[str] = ["any"]

    elements: Dict[str, str] = {
        **GuideLegend.elements,
        "axis_line": "legend.axis.line",
    }


# ============================================================================
# GuideCustom -- user-supplied grob guide
# ============================================================================

class GuideCustom(Guide):
    """Custom guide that displays a user-supplied grob.

    Attributes
    ----------
    _class_name : str
        ``"GuideCustom"``.
    """

    _class_name: str = "GuideCustom"

    params: Dict[str, Any] = {
        **Guide.params,
        "grob": None,
        "width": None,
        "height": None,
        "name": "custom",
    }

    available_aes: List[str] = ["any"]

    hashables: List[str] = ["title", "grob"]

    elements: Dict[str, str] = {
        "background": "legend.background",
        "margin": "legend.margin",
        "title": "legend.title",
        "title_position": "legend.title.position",
    }

    def train(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale: Any = None,
        aesthetic: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Custom guides skip training.

        Returns
        -------
        dict
            Unchanged parameters.
        """
        return params if params is not None else dict(self.params)

    @staticmethod
    def transform(params: Dict[str, Any], coord: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Pass through without transformation.

        Returns
        -------
        dict
            Unmodified parameters.
        """
        return params

    def draw(
        self,
        theme: Any = None,
        position: Optional[str] = None,
        direction: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Draw the custom grob with optional title.

        Parameters
        ----------
        theme : Theme, optional
            Plot theme.
        position : str, optional
            Legend position.
        direction : str, optional
            Legend direction.
        params : dict, optional
            Guide parameters.

        Returns
        -------
        grob
            The custom grob.
        """
        if params is None:
            params = dict(self.params)
        return params.get("grob")


# ============================================================================
# GuideAxisLogticks -- log-scale tick marks
# ============================================================================

class GuideAxisLogticks(GuideAxis):
    """Axis guide with logarithmic tick marks.

    Replaces standard tick placement with ticks at log10-spaced intervals.

    Attributes
    ----------
    _class_name : str
        ``"GuideAxisLogticks"``.
    """

    _class_name: str = "GuideAxisLogticks"

    params: Dict[str, Any] = {
        **GuideAxis.params,
        "long": 2.25,
        "mid": 1.5,
        "short": 0.75,
        "prescale.base": None,
        "negative.small": None,
        "short.theme": None,
        "expanded": True,
        "name": "axis_logticks",
    }

    available_aes: List[str] = ["x", "y"]


# ============================================================================
# GuideAxisStack -- stacked axis guides
# ============================================================================

class GuideAxisStack(GuideAxis):
    """Stacked axis guide combining multiple axis guides.

    Attributes
    ----------
    _class_name : str
        ``"GuideAxisStack"``.
    """

    _class_name: str = "GuideAxisStack"

    params: Dict[str, Any] = {
        "guides": [],
        "guide_params": [],
        "spacing": None,
        "name": "stacked_axis",
        "title": waiver(),
        "theme": None,
        "angle": waiver(),
        "hash": "",
        "position": waiver(),
        "direction": None,
        "order": 0,
    }

    available_aes: List[str] = ["x", "y", "theta", "r"]


# ============================================================================
# GuideAxisTheta -- angle axis for radial coordinates
# ============================================================================

class GuideAxisTheta(GuideAxis):
    """Angle axis guide for polar / radial coordinates.

    Attributes
    ----------
    _class_name : str
        ``"GuideAxisTheta"``.
    """

    _class_name: str = "GuideAxisTheta"

    params: Dict[str, Any] = {
        **GuideAxis.params,
        "name": "axis_theta",
    }

    available_aes: List[str] = ["x", "y", "theta"]

    @staticmethod
    def transform(
        params: Dict[str, Any],
        coord: Any,
        panel_params: Any,
    ) -> Dict[str, Any]:
        """Transform data for theta axis.

        Delegates to :meth:`GuideAxis.transform` and then adds
        ``theta`` column for label angle computation.

        Parameters
        ----------
        params : dict
            Guide parameters.
        coord : Coord
            Coordinate system.
        panel_params : object
            Panel parameters.

        Returns
        -------
        dict
            Transformed parameters.
        """
        params = GuideAxis.transform(params, coord, panel_params)
        key = params.get("key")
        if key is not None and not key.empty and "theta" not in key.columns:
            position = params.get("position", "bottom")
            theta_map = {
                "top": 0.0,
                "bottom": np.pi,
                "left": 1.5 * np.pi,
                "right": 0.5 * np.pi,
            }
            key = key.copy()
            key["theta"] = theta_map.get(position, 0.0)
            params["key"] = key
        return params


# ============================================================================
# GuideOld -- legacy S3 compatibility wrapper
# ============================================================================

class GuideOld(Guide):
    """Compatibility wrapper for the previous S3-based guide system.

    The old S3 methods (``guide_train``, ``guide_merge``, etc.) are
    dispatched through this class as a fallback.

    Attributes
    ----------
    _class_name : str
        ``"GuideOld"``.
    """

    _class_name: str = "GuideOld"


# ============================================================================
# new_guide() -- Guide constructor factory
# ============================================================================

def new_guide(
    *,
    available_aes: Union[str, List[str]] = "any",
    super: type = Guide,  # noqa: A002 (shadows builtin intentionally)
    **kwargs: Any,
) -> Guide:
    """Construct a Guide instance with validated parameters.

    Parameters
    ----------
    available_aes : str or list of str
        Aesthetics supported by this guide.  ``"any"`` matches all
        non-position aesthetics.
    super : type
        The Guide (sub)class to instantiate.
    **kwargs : Any
        Parameter overrides.  Must be a subset of ``super.params`` keys.

    Returns
    -------
    Guide
        A new guide instance.

    Raises
    ------
    ValueError
        If required parameters are missing.
    """
    if isinstance(available_aes, str):
        available_aes = [available_aes]

    # Determine valid parameter names
    param_names = set(super.params.keys()) if hasattr(super, "params") else set()

    # Split into params vs extra
    params: Dict[str, Any] = {}
    extra_args: List[str] = []
    for k, v in kwargs.items():
        if k in param_names:
            params[k] = v
        else:
            extra_args.append(k)

    if extra_args:
        cls_name = snake_class(super) if hasattr(super, "_class_name") else str(super)
        cli_warn(
            f"Ignoring unknown argument(s) to {cls_name}: "
            f"{', '.join(extra_args)}."
        )

    # Fill defaults
    if hasattr(super, "params"):
        merged = dict(super.params)
        merged.update(params)
        params = merged

    # Validate required base Guide params
    required = set(Guide.params.keys())
    missing = required - set(params.keys())
    if missing:
        cli_abort(
            f"The following parameters are required for setting up a guide "
            f"but are missing: {', '.join(sorted(missing))}"
        )

    # Validate theme
    theme = params.get("theme")
    if theme is not None:
        direction = params.get("direction")
        if direction is None and hasattr(theme, "get"):
            params["direction"] = theme.get("legend.direction")

    # Ensure order is an integer
    params["order"] = int(params.get("order", 0))

    # Create instance
    instance = super()
    instance.params = params
    instance.available_aes = list(available_aes)
    return instance


# ============================================================================
# Constructor functions
# ============================================================================

def guide_none(
    title: Any = waiver(),
    position: Any = waiver(),
) -> GuideNone:
    """Create a guide that draws nothing.

    Parameters
    ----------
    title : str or Waiver
        Guide title (unused but kept for interface consistency).
    position : str or Waiver
        Position hint.

    Returns
    -------
    GuideNone
        An empty guide.
    """
    return new_guide(
        title=title,
        position=position,
        available_aes="any",
        super=GuideNone,
    )


def guide_axis(
    title: Any = waiver(),
    theme: Any = None,
    check_overlap: bool = False,
    angle: Any = waiver(),
    n_dodge: int = 1,
    minor_ticks: bool = False,
    cap: Union[str, bool] = "none",
    order: int = 0,
    position: Any = waiver(),
) -> GuideAxis:
    """Create an axis guide.

    Parameters
    ----------
    title : str or Waiver
        Axis title.
    theme : Theme, optional
        Theme overrides.
    check_overlap : bool
        Silently remove overlapping labels.
    angle : float or Waiver
        Text angle in degrees.
    n_dodge : int
        Number of rows/columns for dodging labels.
    minor_ticks : bool
        Whether to draw minor ticks.
    cap : str or bool
        Axis line capping: ``"none"``, ``"both"``, ``"upper"``,
        ``"lower"``, ``True`` (="both"), or ``False`` (="none").
    order : int
        Guide ordering priority.
    position : str or Waiver
        Where the axis is drawn.

    Returns
    -------
    GuideAxis
        An axis guide instance.
    """
    if isinstance(cap, bool):
        cap = "both" if cap else "none"
    if cap not in ("none", "both", "upper", "lower"):
        cli_abort(f"`cap` must be one of 'none', 'both', 'upper', 'lower', got {cap!r}")

    return new_guide(
        title=title,
        theme=theme,
        **{
            "check.overlap": check_overlap,
        },
        angle=angle,
        **{
            "n.dodge": n_dodge,
            "minor.ticks": minor_ticks,
        },
        cap=cap,
        order=order,
        position=position,
        available_aes=["x", "y", "r"],
        name="axis",
        super=GuideAxis,
    )


def guide_legend(
    title: Any = waiver(),
    theme: Any = None,
    position: Optional[str] = None,
    direction: Optional[str] = None,
    override_aes: Optional[Dict[str, Any]] = None,
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    reverse: bool = False,
    order: int = 0,
    **kwargs: Any,
) -> GuideLegend:
    """Create a legend guide.

    Parameters
    ----------
    title : str or Waiver
        Legend title.
    theme : Theme, optional
        Theme overrides.
    position : str, optional
        One of ``"top"``, ``"right"``, ``"bottom"``, ``"left"``, or
        ``"inside"``.
    direction : str, optional
        ``"horizontal"`` or ``"vertical"``.
    override_aes : dict, optional
        Aesthetic parameters to override in the legend keys.
    nrow : int, optional
        Number of rows.
    ncol : int, optional
        Number of columns.
    reverse : bool
        Reverse the order of keys.
    order : int
        Guide ordering priority.
    **kwargs : Any
        Ignored (for compatibility).

    Returns
    -------
    GuideLegend
        A legend guide.
    """
    if position is not None and position not in _TRBL + ["inside"]:
        cli_abort(
            f"`position` must be one of {_TRBL + ['inside']!r}, got {position!r}"
        )
    if override_aes is None:
        override_aes = {}

    return new_guide(
        title=title,
        theme=theme,
        direction=direction,
        **{"override.aes": override_aes},
        nrow=nrow,
        ncol=ncol,
        reverse=reverse,
        order=order,
        position=position,
        available_aes="any",
        name="legend",
        super=GuideLegend,
    )


def guide_colourbar(
    title: Any = waiver(),
    theme: Any = None,
    nbin: Optional[int] = None,
    display: str = "raster",
    alpha: float = float("nan"),
    draw_ulim: bool = True,
    draw_llim: bool = True,
    angle: Optional[float] = None,
    position: Optional[str] = None,
    direction: Optional[str] = None,
    reverse: bool = False,
    order: int = 0,
    available_aes: Optional[List[str]] = None,
    **kwargs: Any,
) -> GuideColourbar:
    """Create a continuous colour bar guide.

    Parameters
    ----------
    title : str or Waiver
        Guide title.
    theme : Theme, optional
        Theme overrides.
    nbin : int, optional
        Number of bins.  Defaults to 300 for raster/rectangles, 15 for
        gradient.
    display : str
        ``"raster"``, ``"rectangles"``, or ``"gradient"``.
    alpha : float
        Colour transparency (0--1).  ``NaN`` preserves encoded alpha.
    draw_ulim : bool
        Draw upper limit tick.
    draw_llim : bool
        Draw lower limit tick.
    angle : float, optional
        Label angle.
    position : str, optional
        Legend position.
    direction : str, optional
        ``"horizontal"`` or ``"vertical"``.
    reverse : bool
        Reverse colour bar direction.
    order : int
        Guide ordering priority.
    available_aes : list of str, optional
        Supported aesthetics.  Defaults to colour/color/fill.
    **kwargs : Any
        Ignored.

    Returns
    -------
    GuideColourbar
        A colour bar guide.
    """
    if display not in ("raster", "rectangles", "gradient"):
        cli_abort(f"`display` must be 'raster', 'rectangles', or 'gradient', got {display!r}")
    if nbin is None:
        nbin = 15 if display == "gradient" else 300

    if position is not None and position not in _TRBL + ["inside"]:
        cli_abort(f"`position` must be one of {_TRBL + ['inside']!r}, got {position!r}")

    if available_aes is None:
        available_aes = ["colour", "color", "fill"]

    return new_guide(
        title=title,
        theme=theme,
        nbin=nbin,
        display=display,
        alpha=alpha,
        angle=angle,
        draw_lim=[bool(draw_llim), bool(draw_ulim)],
        position=position,
        direction=direction,
        reverse=reverse,
        order=order,
        available_aes=available_aes,
        name="colourbar",
        super=GuideColourbar,
    )


# Alias
guide_colorbar = guide_colourbar
"""Alias for :func:`guide_colourbar`."""


def guide_coloursteps(
    title: Any = waiver(),
    theme: Any = None,
    alpha: float = float("nan"),
    angle: Optional[float] = None,
    even_steps: bool = True,
    show_limits: Optional[bool] = None,
    direction: Optional[str] = None,
    position: Optional[str] = None,
    reverse: bool = False,
    order: int = 0,
    available_aes: Optional[List[str]] = None,
    **kwargs: Any,
) -> GuideColoursteps:
    """Create a stepped colour bar guide.

    Parameters
    ----------
    title : str or Waiver
        Guide title.
    theme : Theme, optional
        Theme overrides.
    alpha : float
        Colour transparency.
    angle : float, optional
        Label angle.
    even_steps : bool
        Make all bins the same rendered size.
    show_limits : bool, optional
        Show scale limits.
    direction : str, optional
        ``"horizontal"`` or ``"vertical"``.
    position : str, optional
        Legend position.
    reverse : bool
        Reverse colour bar.
    order : int
        Guide ordering priority.
    available_aes : list of str, optional
        Supported aesthetics.
    **kwargs : Any
        Ignored.

    Returns
    -------
    GuideColoursteps
        A stepped colour bar guide.
    """
    if available_aes is None:
        available_aes = ["colour", "color", "fill"]

    return new_guide(
        title=title,
        theme=theme,
        alpha=alpha,
        angle=angle,
        **{
            "even.steps": even_steps,
            "show.limits": show_limits,
        },
        position=position,
        direction=direction,
        reverse=reverse,
        order=order,
        available_aes=available_aes,
        super=GuideColoursteps,
    )


# Alias
guide_colorsteps = guide_coloursteps
"""Alias for :func:`guide_coloursteps`."""


def guide_bins(
    title: Any = waiver(),
    theme: Any = None,
    angle: Optional[float] = None,
    position: Optional[str] = None,
    direction: Optional[str] = None,
    override_aes: Optional[Dict[str, Any]] = None,
    reverse: bool = False,
    order: int = 0,
    show_limits: Optional[bool] = None,
    **kwargs: Any,
) -> GuideBins:
    """Create a binned legend guide.

    Parameters
    ----------
    title : str or Waiver
        Guide title.
    theme : Theme, optional
        Theme overrides.
    angle : float, optional
        Label angle.
    position : str, optional
        Legend position.
    direction : str, optional
        ``"horizontal"`` or ``"vertical"``.
    override_aes : dict, optional
        Aesthetic overrides for keys.
    reverse : bool
        Reverse key order.
    order : int
        Guide ordering priority.
    show_limits : bool, optional
        Show scale limits.
    **kwargs : Any
        Ignored.

    Returns
    -------
    GuideBins
        A binned legend guide.
    """
    if position is not None and position not in _TRBL + ["inside"]:
        cli_abort(f"`position` must be one of {_TRBL + ['inside']!r}, got {position!r}")
    if override_aes is None:
        override_aes = {}

    return new_guide(
        title=title,
        theme=theme,
        angle=angle,
        position=position,
        direction=direction,
        **{
            "override.aes": override_aes,
            "show.limits": show_limits,
        },
        reverse=reverse,
        order=order,
        available_aes="any",
        name="bins",
        super=GuideBins,
    )


def guide_custom(
    grob: Any,
    width: Any = None,
    height: Any = None,
    title: Optional[str] = None,
    theme: Any = None,
    position: Optional[str] = None,
    order: int = 0,
) -> GuideCustom:
    """Create a custom guide displaying a user-supplied grob.

    Parameters
    ----------
    grob : grob
        The graphical object to display.
    width : unit, optional
        Allocated width.
    height : unit, optional
        Allocated height.
    title : str, optional
        Guide title.  ``None`` means no title.
    theme : Theme, optional
        Theme overrides.
    position : str, optional
        Legend position.
    order : int
        Guide ordering priority.

    Returns
    -------
    GuideCustom
        A custom guide.
    """
    return new_guide(
        grob=grob,
        width=width,
        height=height,
        title=title,
        theme=theme,
        hash=_hash_object([title, grob]),
        position=position,
        order=order,
        available_aes="any",
        super=GuideCustom,
    )


def guide_axis_logticks(
    long: float = 2.25,
    mid: float = 1.5,
    short: float = 0.75,
    prescale_base: Optional[float] = None,
    negative_small: Optional[float] = None,
    short_theme: Any = None,
    expanded: bool = True,
    cap: Union[str, bool] = "none",
    theme: Any = None,
    title: Any = waiver(),
    order: int = 0,
    position: Any = waiver(),
    **kwargs: Any,
) -> GuideAxisLogticks:
    """Create an axis guide with log-spaced tick marks.

    Parameters
    ----------
    long : float
        Relative length of long (decade) ticks.
    mid : float
        Relative length of mid ticks.
    short : float
        Relative length of short ticks.
    prescale_base : float, optional
        Log base for pre-transformed data.
    negative_small : float, optional
        Smallest absolute value ticked when 0 is included.
    short_theme : element, optional
        Theme element for shortest ticks.
    expanded : bool
        Cover expanded range.
    cap : str or bool
        Axis line capping.
    theme : Theme, optional
        Theme overrides.
    title : str or Waiver
        Axis title.
    order : int
        Guide ordering priority.
    position : str or Waiver
        Axis position.
    **kwargs : Any
        Forwarded to :func:`guide_axis`.

    Returns
    -------
    GuideAxisLogticks
        A log-tick axis guide.
    """
    if isinstance(cap, bool):
        cap = "both" if cap else "none"

    return new_guide(
        title=title,
        theme=theme,
        long=long,
        mid=mid,
        short=short,
        **{
            "prescale.base": prescale_base,
            "negative.small": negative_small,
            "short.theme": short_theme,
        },
        expanded=expanded,
        cap=cap,
        order=order,
        position=position,
        available_aes=["x", "y"],
        name="axis_logticks",
        super=GuideAxisLogticks,
    )


def guide_axis_stack(
    first: Any = "axis",
    *args: Any,
    title: Any = waiver(),
    theme: Any = None,
    spacing: Any = None,
    order: int = 0,
    position: Any = waiver(),
) -> GuideAxisStack:
    """Create a stacked axis guide.

    Parameters
    ----------
    first : str or Guide
        The innermost axis guide.
    *args : str or Guide
        Additional axis guides to stack.
    title : str or Waiver
        Axis title.
    theme : Theme, optional
        Theme overrides.
    spacing : unit, optional
        Space between stacked guides.
    order : int
        Guide ordering priority.
    position : str or Waiver
        Axis position.

    Returns
    -------
    GuideAxisStack
        A stacked axis guide.
    """
    axes = [_validate_guide(first)] + [_validate_guide(a) for a in args]
    guide_params = [dict(getattr(a, "params", {})) for a in axes]

    return new_guide(
        title=title,
        theme=theme,
        guides=axes,
        guide_params=guide_params,
        spacing=spacing,
        available_aes=["x", "y", "theta", "r"],
        order=order,
        position=position,
        name="stacked_axis",
        super=GuideAxisStack,
    )


def guide_axis_theta(
    title: Any = waiver(),
    theme: Any = None,
    angle: Any = waiver(),
    minor_ticks: bool = False,
    cap: Union[str, bool] = "none",
    order: int = 0,
    position: Any = waiver(),
) -> GuideAxisTheta:
    """Create an angle axis guide for radial coordinates.

    Parameters
    ----------
    title : str or Waiver
        Axis title.
    theme : Theme, optional
        Theme overrides.
    angle : float or Waiver
        Text angle.
    minor_ticks : bool
        Draw minor ticks.
    cap : str or bool
        Axis line capping.
    order : int
        Guide ordering priority.
    position : str or Waiver
        Axis position.

    Returns
    -------
    GuideAxisTheta
        A theta axis guide.
    """
    if isinstance(cap, bool):
        cap = "both" if cap else "none"

    return new_guide(
        title=title,
        theme=theme,
        angle=angle,
        cap=cap,
        **{"minor.ticks": minor_ticks},
        available_aes=["x", "y", "theta"],
        order=order,
        position=position,
        name="axis_theta",
        super=GuideAxisTheta,
    )


# ============================================================================
# Legacy S3 compatibility functions
# ============================================================================

def old_guide(guide: Any) -> GuideOld:
    """Wrap a legacy guide object.

    Parameters
    ----------
    guide : object
        An old-style guide.

    Returns
    -------
    GuideOld
        Wrapped guide.
    """
    instance = GuideOld()
    instance._legacy = guide
    return instance


def guide_train(guide: Any, scale: Any, aesthetic: Optional[str] = None) -> Any:
    """Legacy S3-style ``guide_train`` dispatch.

    Parameters
    ----------
    guide : Guide
        The guide to train.
    scale : Scale
        Scale to train on.
    aesthetic : str, optional
        Aesthetic name.

    Returns
    -------
    Any
        Trained guide parameters.
    """
    if hasattr(guide, "train"):
        return guide.train(params=dict(getattr(guide, "params", {})),
                           scale=scale, aesthetic=aesthetic)
    cli_abort("Guide classes have been rewritten as GGProto classes. "
              "The old S3 guide methods have been superseded.")


def guide_merge(guide: Any, new_guide: Any) -> Any:
    """Legacy S3-style ``guide_merge`` dispatch.

    Parameters
    ----------
    guide : Guide
        Primary guide.
    new_guide : Guide
        Guide to merge in.

    Returns
    -------
    Any
        Merged guide.
    """
    if hasattr(guide, "merge"):
        return guide.merge(dict(getattr(guide, "params", {})),
                           new_guide,
                           dict(getattr(new_guide, "params", {})))
    cli_abort("Guide classes have been rewritten as GGProto classes. "
              "The old S3 guide methods have been superseded.")


def guide_geom(guide: Any, layers: Any = None, default_mapping: Any = None) -> Any:
    """Legacy S3-style ``guide_geom`` dispatch.

    Parameters
    ----------
    guide : Guide
        The guide.
    layers : list, optional
        Plot layers.
    default_mapping : Mapping, optional
        Default aesthetic mapping.

    Returns
    -------
    Any
        Geom info.
    """
    if hasattr(guide, "process_layers"):
        return guide.process_layers(dict(getattr(guide, "params", {})),
                                    layers or [])
    cli_abort("Guide classes have been rewritten as GGProto classes. "
              "The old S3 guide methods have been superseded.")


def guide_transform(guide: Any, coord: Any, panel_params: Any) -> Any:
    """Legacy S3-style ``guide_transform`` dispatch.

    Parameters
    ----------
    guide : Guide
        The guide.
    coord : Coord
        Coordinate system.
    panel_params : object
        Panel parameters.

    Returns
    -------
    Any
        Transformed parameters.
    """
    if hasattr(guide, "transform"):
        return guide.transform(dict(getattr(guide, "params", {})),
                               coord, panel_params)
    cli_abort("Guide classes have been rewritten as GGProto classes. "
              "The old S3 guide methods have been superseded.")


def guide_gengrob(guide: Any, theme: Any) -> Any:
    """Legacy S3-style ``guide_gengrob`` dispatch.

    Parameters
    ----------
    guide : Guide
        The guide.
    theme : Theme
        Plot theme.

    Returns
    -------
    Any
        Generated grob.
    """
    if hasattr(guide, "draw"):
        return guide.draw(theme=theme)
    cli_abort("Guide classes have been rewritten as GGProto classes. "
              "The old S3 guide methods have been superseded.")


# ============================================================================
# Type-checking helpers
# ============================================================================

def is_guide(x: Any) -> bool:
    """Test whether *x* is a Guide.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a ``Guide`` instance or subclass.
    """
    return isinstance(x, Guide) or (isinstance(x, type) and issubclass(x, Guide))


def is_guides(x: Any) -> bool:
    """Test whether *x* is a Guides container.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a ``Guides`` instance.
    """
    return isinstance(x, Guides)


# ============================================================================
# Guides container class
# ============================================================================

class Guides:
    """Container for guide specifications by aesthetic.

    A ``Guides`` object maps aesthetic names to guide objects (or strings
    that will be resolved to guide objects later).  It manages the full
    lifecycle of merging, training, and assembling guides.

    Parameters
    ----------
    guide_map : dict, optional
        Initial mapping of aesthetic name -> guide specification.

    Attributes
    ----------
    guides : dict
        Aesthetic -> Guide mapping.
    params : list[dict]
        Parallel list of parameters for each guide.
    aesthetics : list[str]
        Parallel list of aesthetic names.
    """

    def __init__(self, guide_map: Optional[Dict[str, Any]] = None) -> None:
        self.guides: Dict[str, Any] = guide_map or {}
        self.params: List[Dict[str, Any]] = []
        self.aesthetics: List[str] = []
        self._missing: GuideNone = guide_none()

    def __repr__(self) -> str:
        keys = list(self.guides.keys())
        return f"<Guides: {keys}>"

    # -- Setters -------------------------------------------------------------

    def add(self, guides: Any) -> None:
        """Add new guides provided by the user.

        Parameters
        ----------
        guides : dict or Guides
            New guide specifications to incorporate.  Existing entries
            are kept as defaults.
        """
        if guides is None:
            return
        if isinstance(guides, Guides):
            guides = guides.guides
        self.guides = _defaults(guides, self.guides)

    def update_params(self, params: List[Optional[Dict[str, Any]]]) -> None:
        """Update guide parameters in place.

        Parameters
        ----------
        params : list of dict or None
            New parameter dicts.  ``None`` entries replace the
            corresponding guide with ``guide_none()``.
        """
        if len(params) != len(self.params):
            cli_abort(
                f"Cannot update {len(self.params)} guide(s) with a list of "
                f"{len(params)} parameter(s)."
            )
        for i, p in enumerate(params):
            if p is None:
                self.guides[i] = self._missing
            else:
                self.params[i] = p

    def subset_guides(self, mask: List[bool]) -> None:
        """Keep only guides where *mask* is ``True``.

        Parameters
        ----------
        mask : list of bool
            Boolean mask parallel to ``self.guides``.
        """
        if isinstance(self.guides, dict):
            keys = list(self.guides.keys())
            self.guides = {k: v for k, keep in zip(keys, mask)
                           for v in [self.guides[k]] if keep}
        elif isinstance(self.guides, list):
            self.guides = [g for g, keep in zip(self.guides, mask) if keep]
        self.params = [p for p, keep in zip(self.params, mask) if keep]
        self.aesthetics = [a for a, keep in zip(self.aesthetics, mask) if keep]

    # -- Getters -------------------------------------------------------------

    def get_guide(self, index: Union[int, str]) -> Optional[Any]:
        """Retrieve a guide by index or aesthetic name.

        Parameters
        ----------
        index : int or str
            Index or aesthetic name.

        Returns
        -------
        Guide or None
            The guide, or ``None`` if not found.
        """
        if isinstance(index, str):
            if isinstance(self.guides, dict):
                return self.guides.get(index)
            if index in self.aesthetics:
                idx = self.aesthetics.index(index)
                guides_list = list(self.guides.values()) if isinstance(
                    self.guides, dict) else self.guides
                return guides_list[idx]
            return None
        guides_list = list(self.guides.values()) if isinstance(
            self.guides, dict) else self.guides
        if 0 <= index < len(guides_list):
            return guides_list[index]
        return None

    def get_params(self, index: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Retrieve parameters by index or aesthetic name.

        Parameters
        ----------
        index : int or str
            Index or aesthetic name.

        Returns
        -------
        dict or None
            Parameters, or ``None`` if not found.
        """
        if isinstance(index, str):
            if index in self.aesthetics:
                idx = self.aesthetics.index(index)
                return self.params[idx]
            return None
        if 0 <= index < len(self.params):
            return self.params[index]
        return None

    # -- Building ------------------------------------------------------------

    def setup(
        self,
        scales: List[Any],
        aesthetics: Optional[List[str]] = None,
        default: Any = None,
        missing: Any = None,
    ) -> "Guides":
        """Generate a guide for every scale-aesthetic pair.

        Parameters
        ----------
        scales : list
            Scale objects.
        aesthetics : list of str, optional
            Aesthetic names parallel to *scales*.
        default : Guide, optional
            Default guide when none is specified.
        missing : Guide, optional
            Guide for unresolvable entries.

        Returns
        -------
        Guides
            A new ``Guides`` instance populated with resolved guides.
        """
        if default is None:
            default = self._missing
        if missing is None:
            missing = self._missing
        if aesthetics is None:
            aesthetics = [getattr(s, "aesthetics", ["unknown"])[0] for s in scales]

        new_guides: List[Any] = []
        for idx, scale in enumerate(scales):
            aes_name = aesthetics[idx]
            guide = self.guides.get(aes_name)

            # Fallback hierarchy
            if guide is None:
                guide = getattr(scale, "guide", None)
            if guide is None or is_waiver(guide):
                guide = default
            if guide is None:
                guide = missing

            # Resolve string names
            guide = _validate_guide(guide)

            # Check compatibility
            if not isinstance(guide, GuideNone):
                scale_aes = getattr(scale, "aesthetics", [])
                if not any(a in ("x", "y") for a in scale_aes):
                    scale_aes = list(scale_aes) + ["any"]
                available = getattr(guide, "available_aes", [])
                if not any(a in available for a in scale_aes):
                    cli_warn(
                        f"{snake_class(guide)} cannot be used for "
                        f"{', '.join(scale_aes[:4])}."
                    )
                    guide = missing

            new_guides.append(guide)

        child = Guides()
        child.guides = new_guides
        child.params = [dict(getattr(g, "params", {})) for g in new_guides]
        child.aesthetics = list(aesthetics)
        return child

    def train(self, scales: List[Any], labels: Dict[str, str]) -> None:
        """Train each guide on its paired scale.

        Parameters
        ----------
        scales : list
            Scale objects, parallel to ``self.guides``.
        labels : dict
            Aesthetic -> label mapping.
        """
        guides_list = list(self.guides) if isinstance(self.guides, dict) else self.guides
        new_params: List[Optional[Dict[str, Any]]] = []
        for i, (guide, scale) in enumerate(zip(guides_list, scales)):
            aes = self.aesthetics[i] if i < len(self.aesthetics) else ""
            p = guide.train(
                params=dict(self.params[i]) if i < len(self.params) else {},
                scale=scale,
                aesthetic=aes,
                title=labels.get(aes),
            )
            new_params.append(p)

        # Filter out None (dropped guides)
        keep = [p is not None for p in new_params]
        self.params = [p for p in new_params if p is not None]
        self.guides = [g for g, k in zip(guides_list, keep) if k]
        self.aesthetics = [a for a, k in zip(self.aesthetics, keep) if k]

        # Drop GuideNone entries
        keep_none = [not isinstance(g, GuideNone) for g in self.guides]
        self.subset_guides(keep_none)

    def merge(self) -> None:
        """Merge guides that encode the same information.

        Groups guides by ``{order}_{hash}`` and merges groups with
        more than one member.
        """
        if len(self.guides) <= 1:
            return

        guides_list = list(self.guides) if isinstance(self.guides, dict) else self.guides

        # Build hash keys
        orders = [p.get("order", 0) for p in self.params]
        orders = [99 if o == 0 else o for o in orders]
        hashes = [p.get("hash", "") for p in self.params]
        keys = [f"{o:02d}_{h}" for o, h in zip(orders, hashes)]

        # Group by key
        groups: Dict[str, List[int]] = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)

        merged_guides: List[Any] = []
        merged_params: List[Dict[str, Any]] = []
        merged_aes: List[str] = []

        for key in sorted(groups.keys()):
            indices = groups[key]
            if len(indices) == 1:
                idx = indices[0]
                merged_guides.append(guides_list[idx])
                merged_params.append(self.params[idx])
                merged_aes.append(self.aesthetics[idx])
            else:
                # Sequentially merge
                result = {
                    "guide": guides_list[indices[0]],
                    "params": dict(self.params[indices[0]]),
                }
                for idx in indices[1:]:
                    result = result["guide"].merge(
                        result["params"],
                        guides_list[idx],
                        dict(self.params[idx]),
                    )
                merged_guides.append(result["guide"])
                merged_params.append(result["params"])
                merged_aes.append(self.aesthetics[indices[0]])

        self.guides = merged_guides
        self.params = merged_params
        self.aesthetics = merged_aes

    def process_layers(
        self,
        layers: List[Any],
        data: Optional[List[Any]] = None,
        theme: Any = None,
    ) -> None:
        """Let guides extract information from layers.

        Parameters
        ----------
        layers : list
            Plot layers.
        data : list, optional
            Layer data.
        theme : Theme, optional
            Plot theme.
        """
        guides_list = list(self.guides) if isinstance(self.guides, dict) else self.guides
        new_params = []
        for guide, params in zip(guides_list, self.params):
            new_params.append(guide.process_layers(params, layers, data, theme))

        keep = [p is not None for p in new_params]
        self.params = [p for p in new_params if p is not None]
        self.guides = [g for g, k in zip(guides_list, keep) if k]
        self.aesthetics = [a for a, k in zip(self.aesthetics, keep) if k]

    def build(
        self,
        scales: Any,
        layers: List[Any],
        labels: Dict[str, str],
        layer_data: Optional[List[Any]] = None,
        theme: Any = None,
    ) -> "Guides":
        """Full guide build pipeline.

        Parameters
        ----------
        scales : ScalesList
            All scales from the plot.
        layers : list
            Plot layers.
        labels : dict
            Aesthetic -> label mapping.
        layer_data : list, optional
            Layer data.
        theme : Theme, optional
            Plot theme.

        Returns
        -------
        Guides
            Built guides ready for assembly.
        """
        # Extract non-position scales
        if hasattr(scales, "non_position_scales"):
            scale_list = scales.non_position_scales()
            if hasattr(scale_list, "scales"):
                scale_list = scale_list.scales
        else:
            scale_list = scales if isinstance(scales, list) else []

        if not scale_list:
            return Guides()

        # Flatten aesthetics
        flat_scales = []
        flat_aes = []
        for s in scale_list:
            aes_names = getattr(s, "aesthetics", ["unknown"])
            for a in aes_names:
                flat_scales.append(s)
                flat_aes.append(a)

        guides = self.setup(flat_scales, aesthetics=flat_aes)
        guides.train(flat_scales, labels)

        if not guides.guides:
            return Guides()

        guides.merge()
        guides.process_layers(layers, layer_data, theme)
        return guides

    def draw(
        self,
        theme: Any,
        positions: List[str],
        direction: Optional[str] = None,
    ) -> List[Any]:
        """Render guides into grobs.

        Parameters
        ----------
        theme : Theme
            Plot theme.
        positions : list of str
            Position for each guide.
        direction : str, optional
            Default direction.

        Returns
        -------
        list
            Rendered grobs.
        """
        guides_list = list(self.guides) if isinstance(self.guides, dict) else self.guides
        directions = [direction or "vertical"] * len(positions)
        for i, pos in enumerate(positions):
            if direction is None and pos in ("top", "bottom"):
                directions[i] = "horizontal"

        grobs = []
        for i, guide in enumerate(guides_list):
            g = guide.draw(
                theme=theme,
                position=positions[i],
                direction=directions[i],
                params=self.params[i] if i < len(self.params) else None,
            )
            grobs.append(g)
        return grobs

    def assemble(self, theme: Any) -> Any:
        """Assemble all guides into positioned guide boxes.

        Parameters
        ----------
        theme : Theme
            Plot theme.

        Returns
        -------
        dict
            Mapping of position -> grob/gtable.
        """
        if not self.guides:
            return None

        default_position = "right"
        if hasattr(theme, "__getitem__"):
            try:
                default_position = theme["legend.position"] or "right"
            except (KeyError, TypeError):
                pass
        elif hasattr(theme, "legend_position"):
            default_position = theme.legend_position or "right"

        positions = []
        for p in self.params:
            pos = p.get("position") or default_position
            if is_waiver(pos):
                pos = default_position
            positions.append(pos)

        grobs = self.draw(theme, positions)
        return {pos: grob for pos, grob in zip(positions, grobs)}


# ============================================================================
# guides() -- user-facing function
# ============================================================================

def guides(**kwargs: Any) -> Optional[Guides]:
    """Set guides for each scale.

    Parameters
    ----------
    **kwargs : str or Guide
        Mapping of aesthetic name to guide specification.  Values can
        be guide objects, constructor calls, or strings like
        ``"legend"`` or ``"none"``.

    Returns
    -------
    Guides or None
        A ``Guides`` container, or ``None`` if no guides were given.

    Examples
    --------
    >>> guides(colour=guide_legend(), size="none")
    >>> guides(colour=guide_colourbar(nbin=50), shape=guide_legend(nrow=2))
    """
    if not kwargs:
        return None

    # Standardise aesthetic names
    standardised: Dict[str, Any] = {}
    for k, v in kwargs.items():
        names = standardise_aes_names([k])
        new_key = names[0] if names else k
        if v is False:
            warnings.warn(
                "Setting a guide to `False` is deprecated. Use 'none' instead.",
                FutureWarning,
                stacklevel=2,
            )
            v = "none"
        standardised[new_key] = v

    return Guides(standardised)

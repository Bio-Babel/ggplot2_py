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


def _parse_binned_breaks(
    scale: Any,
    breaks: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Parse breaks / limits of a binned scale into (breaks, limits, bin_at).

    Faithful port of R's ``parse_binned_breaks`` (guide-bins.R:342-382).

    When ``scale$labels`` is a waiver or callable, NA-valued breaks are
    stripped. For numeric breaks the result covers ``[limits[0], limits[1]]``
    with any user breaks in between; midpoints become ``bin_at``. For
    character-coded bins of the form ``"(lo, hi]"`` the endpoints are
    parsed out of the string.

    Parameters
    ----------
    scale : Scale
        Trained binned scale.
    breaks : array-like, optional
        Pre-extracted breaks; default uses ``scale.get_breaks()``.

    Returns
    -------
    dict or None
        ``{"breaks": array, "limits": (lo, hi), "bin_at": array}`` or
        ``None`` if the scale has no breaks.
    """
    import re

    if breaks is None:
        breaks = getattr(scale, "get_breaks", lambda: None)()
    if breaks is None:
        return None
    breaks = np.asarray(breaks, dtype=object)

    scale_labels = getattr(scale, "labels", None)
    labels_is_waiver_or_fn = is_waiver(scale_labels) or callable(scale_labels)
    if labels_is_waiver_or_fn:
        mask = np.array([not _is_na(b) for b in breaks])
        breaks = breaks[mask]

    if breaks.size == 0:
        return None

    # Numeric branch
    if all(isinstance(b, (int, float, np.integer, np.floating)) and not _is_na(b)
           for b in breaks) or _is_numeric_breaks(breaks):
        breaks = np.asarray([float(b) if not _is_na(b) else np.nan for b in breaks],
                            dtype=float)
        limits = getattr(scale, "get_limits", lambda: (np.nanmin(breaks),
                                                       np.nanmax(breaks)))()
        limits = tuple(float(v) for v in limits)
        scale_breaks = getattr(scale, "breaks", None)
        if not (isinstance(scale_breaks, (list, tuple, np.ndarray))
                and all(isinstance(b, (int, float, np.integer, np.floating))
                        for b in scale_breaks)):
            # R: breaks[breaks %in% limits] <- NA
            breaks = np.array([
                np.nan if (not np.isnan(b) and b in limits) else b
                for b in breaks
            ], dtype=float)

        # oob_censor
        try:
            from scales import oob_censor as _oob_censor
            breaks = np.asarray(_oob_censor(breaks, limits), dtype=float)
        except Exception:
            breaks = np.where(
                (breaks < limits[0]) | (breaks > limits[1]),
                np.nan, breaks,
            )

        # unique0(c(limits[1], breaks, limits[2]))
        combined = np.concatenate(([limits[0]], breaks, [limits[1]]))
        # Preserve order while removing duplicates (like R's unique0)
        seen: set = set()
        all_breaks: List[float] = []
        for v in combined:
            key = "NA" if (isinstance(v, float) and np.isnan(v)) else v
            if key not in seen:
                seen.add(key)
                all_breaks.append(v)
        # sort na.last = NA → drop NAs
        all_breaks = sorted(b for b in all_breaks
                            if not (isinstance(b, float) and np.isnan(b)))
        all_breaks_arr = np.asarray(all_breaks, dtype=float)
        # bin_at = all_breaks[-1] - diff(all_breaks) / 2
        # R: all_breaks[-1] drops first element
        tail = all_breaks_arr[1:]
        diffs = np.diff(all_breaks_arr)
        bin_at = tail - diffs / 2.0
        return {"breaks": breaks, "limits": limits, "bin_at": bin_at}

    # Character / interval branch: "(lo, hi]" or "[lo, hi)"
    nums: List[float] = []
    for b in breaks:
        s = str(b)
        # strip brackets
        cleaned = re.sub(r"[\(\)\[\]]", "", s)
        parts = re.split(r",\s?", cleaned)
        for p in parts:
            try:
                nums.append(float(p))
            except (TypeError, ValueError):
                nums.append(np.nan)
    nums_arr = np.asarray(nums, dtype=float)
    if np.any(np.isnan(nums_arr)):
        cli_abort(
            "Breaks are not formatted correctly for a bin legend. "
            "Use `(<lower>, <upper>]` format to indicate bins."
        )
    bin_at = breaks.copy()
    # R: all_breaks <- nums[c(1, seq_along(breaks) * 2)]
    idx = [0] + [i * 2 + 1 for i in range(len(breaks))]
    all_breaks = nums_arr[idx]
    limits = (all_breaks[0], all_breaks[-1])
    # R: breaks <- all_breaks[-c(1, length(all_breaks))]
    inner = all_breaks[1:-1]
    return {"breaks": inner, "limits": limits, "bin_at": bin_at}


def _is_na(x: Any) -> bool:
    """Test whether ``x`` is R-style NA / Python None / float NaN."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    try:
        if np.isnan(x):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _is_numeric_breaks(breaks: Any) -> bool:
    """Return True if every non-NA element of *breaks* is numeric."""
    for b in breaks:
        if _is_na(b):
            continue
        if not isinstance(b, (int, float, np.integer, np.floating)):
            return False
    return True


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

        # Mirror R's Guide$extract_key (guide-.R:285-290):
        #   if (is.numeric(breaks)) {
        #     range <- scale$continuous_range %||% scale$get_limits()
        #     key  <- vec_slice(key, is.finite(oob_censor_any(breaks, range)))
        #   }
        breaks_arr = np.asarray(breaks)
        if breaks_arr.dtype.kind in ("i", "u", "f"):
            range_ = getattr(scale, "continuous_range", None)
            if range_ is None and hasattr(scale, "get_limits"):
                range_ = scale.get_limits()
            if range_ is not None:
                from scales import oob_censor_any as _oob_censor_any
                censored = np.asarray(
                    _oob_censor_any(breaks_arr.astype(float), range=np.asarray(range_, dtype=float)),
                    dtype=float,
                )
                mask = np.isfinite(censored)
                key = key.loc[mask].reset_index(drop=True)

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
        safe = {k: v for k, v in params.items() if k not in ("key", "decor")}
        try:
            key = self.extract_key(scale, **safe)
        except TypeError:
            # Fallback: pass only aesthetic
            key = self.extract_key(scale, aesthetic=aesthetic)
        if key is not None and hasattr(key, "empty") and key.empty:
            return None
        params["key"] = key

        # Extract decor
        try:
            params["decor"] = self.extract_decor(scale, aesthetic=aesthetic)
        except Exception:
            params["decor"] = None

        # Post-process
        params = self.extract_params(scale, params)

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
    a smooth gradient.  R source: ``ggplot2/R/guide-colorsteps.R``.

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

    # -- extract_key (guide-colorsteps.R:98-135) -----------------------------

    @staticmethod
    def extract_key(
        scale: Any,
        aesthetic: str,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """Extract key for a coloursteps guide.

        Mirrors R's ``GuideColoursteps$extract_key``: falls back to
        :meth:`Guide.extract_key` for non-numeric breaks when
        ``even.steps`` is FALSE; otherwise parses breaks via
        :func:`_parse_binned_breaks` and builds a key with ``.value`` set
        to sequential integers (even-steps mode) or to the breaks.

        Parameters
        ----------
        scale : Scale
            Trained binned colour / fill scale.
        aesthetic : str
            Aesthetic name (column to populate with mapped colours).
        **kwargs : Any
            Accepts ``even_steps`` / ``even.steps`` (R-style), ignored
            otherwise.

        Returns
        -------
        pd.DataFrame or None
            Key with columns ``<aesthetic>``, ``.value``, ``.label``.
            ``None`` if the scale has no breaks.
        """
        even_steps = kwargs.get("even.steps", kwargs.get("even_steps", True))
        breaks = getattr(scale, "get_breaks", lambda: None)()
        if breaks is None:
            return None
        breaks_arr = np.asarray(breaks, dtype=object)

        if not (even_steps or not _is_numeric_breaks(breaks_arr)):
            return Guide.extract_key(scale, aesthetic)

        parsed = _parse_binned_breaks(scale, breaks_arr)
        if parsed is None:
            return None
        limits = parsed["limits"]
        inner_breaks = parsed["breaks"]
        mapped = getattr(scale, "map", lambda x: x)(inner_breaks)
        if isinstance(mapped, np.ndarray):
            mapped = mapped.tolist()

        key = pd.DataFrame({aesthetic: mapped})
        if even_steps:
            key[".value"] = np.arange(1, len(inner_breaks) + 1, dtype=float)
        else:
            key[".value"] = np.asarray(inner_breaks, dtype=float)
        key[".label"] = getattr(scale, "get_labels", lambda b: [str(v) for v in b])(inner_breaks)

        # vec_slice(key, !is.na(breaks)) — inner_breaks has no NAs post-oob_censor
        mask = ~np.isnan(np.asarray(inner_breaks, dtype=float))
        key = key.loc[mask].reset_index(drop=True)

        # Bottom / top limit handling
        if len(inner_breaks) > 0 and inner_breaks[0] in limits:
            key[".value"] = key[".value"] - 1
            key.iat[0, 0] = np.nan
        if len(inner_breaks) > 0 and inner_breaks[-1] in limits:
            key.iat[len(key) - 1, 0] = np.nan

        # Stash parsed for downstream methods (R uses attributes; we use
        # a DataFrame attribute via .attrs).
        key.attrs["parsed"] = parsed
        return key

    # -- extract_decor (guide-colorsteps.R:137-159) --------------------------

    @staticmethod
    def extract_decor(
        scale: Any,
        aesthetic: str,
        key: Optional[pd.DataFrame] = None,
        reverse: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Extract discrete decor bands for the stepped bar.

        Mirrors ``GuideColoursteps$extract_decor``.  Builds a DataFrame
        with one row per bin: ``colour`` (mapped colour at the bin
        midpoint), ``min``, ``max`` (bin boundaries in even-step or
        data-space coordinates), plus ``.size`` bookkeeping.

        Parameters
        ----------
        scale : Scale
            Trained binned scale.
        aesthetic : str
            Aesthetic name (unused; kept for signature parity).
        key : pd.DataFrame, optional
            Previously extracted key; may carry a ``parsed`` attribute
            to avoid recomputation.
        reverse : bool
            Unused here (reverse is applied later in extract_params).
        **kwargs : Any
            Accepts ``even.steps`` / ``even_steps``, ``alpha``, ``nbin``.

        Returns
        -------
        pd.DataFrame
            ``colour``, ``min``, ``max``, ``.size``.
        """
        even_steps = kwargs.get("even.steps", kwargs.get("even_steps", True))
        alpha = kwargs.get("alpha", np.nan)
        parsed = None
        if key is not None and hasattr(key, "attrs"):
            parsed = key.attrs.get("parsed")

        if parsed is not None:
            breaks = parsed["breaks"]
            limits = parsed["limits"]
            bin_at = parsed["bin_at"]
        else:
            breaks = np.asarray(getattr(scale, "get_breaks", lambda: [])(),
                                dtype=float)
            limits = tuple(getattr(scale, "get_limits", lambda: (np.nanmin(breaks),
                                                                 np.nanmax(breaks)))())
            bin_at = None

        # sort unique0(c(limits, breaks))
        combined = np.concatenate(([limits[0], limits[1]],
                                   np.asarray(breaks, dtype=float)))
        breaks_all = np.unique(combined[~np.isnan(combined)])
        n = len(breaks_all)
        if bin_at is None:
            bin_at = (breaks_all[1:] + breaks_all[:-1]) / 2.0

        if even_steps:
            breaks_used = np.arange(n, dtype=float)  # 0..n-1
        else:
            breaks_used = breaks_all

        colours = getattr(scale, "map", lambda x: x)(bin_at)
        if isinstance(colours, np.ndarray):
            colours = colours.tolist()
        # alpha
        if alpha is not None and not (isinstance(alpha, float) and np.isnan(alpha)):
            try:
                from scales import alpha as _scales_alpha
                colours = [_scales_alpha(c, alpha) for c in colours]
            except ImportError:
                pass

        n_bins = len(bin_at)
        return pd.DataFrame({
            "colour": colours,
            "min": breaks_used[:-1],
            "max": breaks_used[1:],
            ".size": [n_bins] * n_bins,
        })

    # -- extract_params (guide-colorsteps.R:161-206) -------------------------

    @staticmethod
    def extract_params(
        scale: Any,
        params: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Post-process coloursteps parameters.

        Propagates ``show.limits``, appends limit rows to the key when
        requested, and rescales ``key$.value`` / ``decor$min`` /
        ``decor$max`` onto [0, 1].

        Parameters
        ----------
        scale : Scale
            Trained scale.
        params : dict
            Guide parameters with ``key`` and ``decor`` populated.
        **kwargs : Any
            Accepts ``title``.

        Returns
        -------
        dict
            Updated parameters.
        """
        show_limits = params.get("show.limits")
        if show_limits is None:
            show_limits = getattr(scale, "show_limits", None) or False

        scale_labels = getattr(scale, "labels", None)
        if show_limits and (isinstance(scale_labels, (list, np.ndarray))):
            cli_warn(
                "`show.limits` is ignored when `labels` are given as a "
                "character vector.  Either add the limits to `breaks` or "
                "provide a function for `labels`."
            )
            show_limits = False

        key = params.get("key")
        decor = params.get("decor")
        if show_limits and key is not None and not key.empty and decor is not None:
            parsed = key.attrs.get("parsed") if hasattr(key, "attrs") else None
            limits = (parsed["limits"] if parsed is not None
                      else getattr(scale, "get_limits", lambda: (0, 1))())
            # Prepend/append NA-only rows
            na_row = {c: np.nan for c in key.columns}
            new_rows = [na_row] + [r for r in key.to_dict(orient="records")] + [na_row]
            key = pd.DataFrame(new_rows).reset_index(drop=True)
            n = len(key)
            # Fill in value/label at extremes
            value_range = (float(np.min(decor["min"])),
                           float(np.max(decor["max"])))
            key.at[0, ".value"] = value_range[0]
            key.at[n - 1, ".value"] = value_range[1]
            lim_labels = getattr(scale, "get_labels", lambda l: [str(v) for v in l])(limits)
            key.at[0, ".label"] = lim_labels[0]
            key.at[n - 1, ".label"] = lim_labels[-1]
            # drop redundant endpoints when duplicated
            if key.at[0, ".value"] == key.at[1, ".value"]:
                key = key.iloc[1:].reset_index(drop=True)
                n -= 1
            if key.at[n - 2, ".value"] == key.at[n - 1, ".value"]:
                key = key.iloc[:-1].reset_index(drop=True)
            params["key"] = key

        # title
        title = kwargs.get("title")
        scale_name = getattr(scale, "name", None)
        if is_waiver(params.get("title")):
            if title is not None and not is_waiver(title):
                params["title"] = title
            elif scale_name is not None:
                params["title"] = scale_name

        # rescale values onto [0, 1] using decor min/max as the full range
        if decor is not None and len(decor) > 0:
            limits_range = (float(decor["min"].iloc[0]),
                            float(decor["max"].iloc[-1]))
            if params.get("reverse"):
                limits_range = (limits_range[1], limits_range[0])
            from scales import rescale as _rescale
            if key is not None and not key.empty:
                key[".value"] = _rescale(key[".value"].to_numpy(),
                                          from_range=limits_range)
                # oob-censor
                key = key.loc[
                    ~((key[".value"] < 0) | (key[".value"] > 1))
                ].reset_index(drop=True)
                params["key"] = key
            decor = decor.copy()
            decor["min"] = _rescale(decor["min"].to_numpy(),
                                     from_range=limits_range)
            decor["max"] = _rescale(decor["max"].to_numpy(),
                                     from_range=limits_range)
            params["decor"] = decor
        return params

    # -- build_decor (guide-colorsteps.R:208-229) ----------------------------

    @staticmethod
    def build_decor(
        decor: Any,
        grobs: Any,
        elements: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build stepped-rectangle bar grobs.

        Mirrors ``GuideColoursteps$build_decor``. Returns a dict with
        ``bar`` (a rect_grob covering each band) and ``frame`` (an
        optional border), matching the structure expected by the
        render pipeline in ``plot_render.py``.

        Parameters
        ----------
        decor : pd.DataFrame
            Decor frame produced by :meth:`extract_decor`.
        grobs : dict
            Currently built grobs (``ticks`` may be referenced).
        elements : dict
            Resolved theme elements.
        params : dict
            Guide parameters (``direction`` is read).

        Returns
        -------
        dict
            ``{"bar": grob, "frame": grob, "ticks": grob}``.
        """
        direction = params.get("direction", "vertical") or "vertical"
        if decor is None or len(decor) == 0:
            return {"bar": None, "frame": None, "ticks": grobs.get("ticks")
                    if isinstance(grobs, dict) else None}

        from grid_py import Gpar, rect_grob, null_grob
        sizes = (decor["max"].to_numpy(dtype=float)
                 - decor["min"].to_numpy(dtype=float))
        just = (decor["min"].to_numpy(dtype=float)
                > decor["max"].to_numpy(dtype=float)).astype(float)
        colours = list(decor["colour"])
        n = len(colours)
        gp = Gpar(col=None, fill=colours)

        if direction == "vertical":
            bar = rect_grob(
                x=[0.0] * n,
                y=list(decor["min"].to_numpy(dtype=float)),
                width=[1.0] * n,
                height=list(np.abs(sizes)),
                just=("left", "bottom"),
                default_units="npc",
                gp=gp,
                name="coloursteps.bar",
            )
        else:
            bar = rect_grob(
                x=list(decor["min"].to_numpy(dtype=float)),
                y=[0.0] * n,
                width=list(np.abs(sizes)),
                height=[1.0] * n,
                just=("left", "bottom"),
                default_units="npc",
                gp=gp,
                name="coloursteps.bar",
            )

        frame = None
        frame_el = elements.get("frame") if isinstance(elements, dict) else None
        if frame_el is not None:
            frame = rect_grob(
                gp=Gpar(col="grey50", fill=None, lwd=0.5),
                name="coloursteps.frame",
            )

        return {
            "bar": bar,
            "frame": frame,
            "ticks": grobs.get("ticks") if isinstance(grobs, dict) else None,
        }


# ============================================================================
# GuideBins -- binned legend guide
# ============================================================================

class GuideBins(GuideLegend):
    """Binned legend guide.

    A version of the legend guide for binned scales.  Places ticks
    between keys and optionally shows a small axis.  R source:
    ``ggplot2/R/guide-bins.R``.

    Attributes
    ----------
    _class_name : str
        ``"GuideBins"``.
    """

    _class_name: str = "GuideBins"

    params: Dict[str, Any] = {
        **GuideLegend.params,
        # Defaults mirror R's GuideBins$params (guide-bins.R:116-135)
        "default_axis": None,
        "default_ticks": None,
        "angle": None,
        "override.aes": {},
        "reverse": False,
        "order": 0,
        "show.limits": False,
        "name": "bins",
    }

    available_aes: List[str] = ["any"]

    elements: Dict[str, str] = {
        **GuideLegend.elements,
        "axis_line": "legend.axis.line",
        "ticks_length": "legend.ticks.length",
        "ticks": "legend.ticks",
    }

    # -- extract_key (guide-bins.R:146-187) ----------------------------------

    @staticmethod
    def extract_key(
        scale: Any,
        aesthetic: str,
        show_limits: bool = False,
        reverse: bool = False,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """Extract key for a binned legend.

        Produces one row per bin + one trailing NA row used to anchor
        the final tick.  Each row carries ``.value`` in [0, 1], ``.show``
        indicating whether the limit tick is forced on, and ``.label``
        derived from the scale's bin labels.

        Parameters
        ----------
        scale : Scale
            Trained binned scale.
        aesthetic : str
            Aesthetic name (the mapped-value column).
        show_limits : bool
            Unused here (extract_params applies it).  Accepted for
            signature parity with R.
        reverse : bool
            Unused here; extract_params handles reversal.
        **kwargs : Any
            Accepts ``show.limits`` (R-style) as an alias.

        Returns
        -------
        pd.DataFrame or None
            Key frame or ``None`` if no breaks.
        """
        if "show.limits" in kwargs:
            show_limits = bool(kwargs["show.limits"])
        breaks = getattr(scale, "get_breaks", lambda: None)()
        parsed = _parse_binned_breaks(scale, breaks)
        if parsed is None:
            return None
        limits = parsed["limits"]
        inner_breaks = parsed["breaks"]
        bin_at = parsed["bin_at"]

        # key: one row per bin midpoint + one trailing NA row
        mapped_mid = getattr(scale, "map", lambda x: x)(bin_at)
        if isinstance(mapped_mid, np.ndarray):
            mapped_mid = mapped_mid.tolist()
        values = list(mapped_mid) + [np.nan]
        n_rows = len(values)

        # .value = seq_along - 1 / (n - 1)
        if n_rows > 1:
            value_seq = np.arange(n_rows, dtype=float) / (n_rows - 1)
        else:
            value_seq = np.zeros(n_rows, dtype=float)

        key = pd.DataFrame({
            aesthetic: values,
            ".value": value_seq,
            ".show": [np.nan] * n_rows,
        })

        labels = getattr(scale, "get_labels", lambda b: [str(v) for v in b])(inner_breaks)
        if labels is None:
            labels = [str(v) for v in inner_breaks]
        labels = list(labels)
        # labels <- labels[!is.na(breaks)]; breaks <- breaks[!is.na(breaks)]
        if isinstance(inner_breaks, np.ndarray):
            mask = ~np.isnan(inner_breaks)
            labels = [lbl for lbl, m in zip(labels, mask) if m]
            inner_breaks = inner_breaks[mask]

        # Scale-level label-override check
        scale_labels = getattr(scale, "labels", None)
        if isinstance(scale_labels, (list, tuple, np.ndarray)) and \
                not callable(scale_labels) and not is_waiver(scale_labels):
            limit_lab = [np.nan, np.nan]
        else:
            limit_lab = list(getattr(scale, "get_labels",
                                     lambda l: [str(v) for v in l])(limits))
            if len(limit_lab) < 2:
                limit_lab = limit_lab + [np.nan] * (2 - len(limit_lab))

        # Prepend / append limit labels or force .show = TRUE
        if len(inner_breaks) > 0:
            if inner_breaks[0] not in limits:
                labels = [limit_lab[0]] + labels
            else:
                key.at[0, ".show"] = True
            if inner_breaks[-1] not in limits:
                labels = labels + [limit_lab[1]]
            else:
                key.at[len(key) - 1, ".show"] = True

        # Pad labels to match rows
        if len(labels) < n_rows:
            labels = labels + [np.nan] * (n_rows - len(labels))
        elif len(labels) > n_rows:
            labels = labels[:n_rows]
        key[".label"] = labels

        # oob_censor_any(key$.value) — drop rows with NaN .value
        key = key.loc[~key[".value"].isna()].reset_index(drop=True)
        return key

    # -- extract_params (guide-bins.R:189-224) -------------------------------

    @staticmethod
    def extract_params(
        scale: Any,
        params: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Finalise bins-guide parameters after key extraction.

        Resolves ``show.limits``, strips the ``.show`` helper column,
        reverses the key if requested, and resolves the title.

        Parameters
        ----------
        scale : Scale
            Trained scale.
        params : dict
            Current parameters (``key`` present).
        **kwargs : Any
            Accepts ``title``.

        Returns
        -------
        dict
            Updated parameters.
        """
        show_limits = params.get("show.limits")
        if show_limits is None:
            show_limits = getattr(scale, "show_limits", None)
        if show_limits is None:
            show_limits = False

        scale_labels = getattr(scale, "labels", None)
        if show_limits and isinstance(scale_labels, (list, np.ndarray)) \
                and not callable(scale_labels) and not is_waiver(scale_labels):
            cli_warn(
                "`show.limits` is ignored when `labels` are given as a "
                "character vector. Either add the limits to `breaks` or "
                "provide a function for `labels`."
            )
            show_limits = False

        # rep_len(show_limits, 2)
        show_limits_pair = [bool(show_limits), bool(show_limits)]

        key = params.get("key")
        if key is not None and len(key) > 0 and ".show" in key.columns:
            first_show = key[".show"].iloc[0]
            last_show = key[".show"].iloc[-1]
            # R: show.limits <- ifelse(is.na(show), show.limits, show)
            if not _is_na(first_show):
                show_limits_pair[0] = bool(first_show)
            if not _is_na(last_show):
                show_limits_pair[1] = bool(last_show)
            key = key.drop(columns=[".show"])
        params["show.limits"] = show_limits_pair

        aesthetic = params.get("aesthetic", "")
        if params.get("reverse") and key is not None and not key.empty:
            ord_ = np.arange(len(key))
            key = key.iloc[ord_[::-1]].reset_index(drop=True)
            # Rotate aesthetic column — R: c(ord[-1], ord[1])
            # Drop first, append original first: yields rotated indexing
            if aesthetic in key.columns and len(key) > 1:
                rotated_idx = list(range(1, len(key))) + [0]
                key[aesthetic] = key[aesthetic].iloc[rotated_idx].values
            key[".value"] = 1.0 - key[".value"].astype(float)

        params["key"] = key
        # Title resolution (identical to Guide$extract_params)
        title = kwargs.get("title")
        scale_name = getattr(scale, "name", None)
        if is_waiver(params.get("title")):
            if title is not None and not is_waiver(title):
                params["title"] = title
            elif scale_name is not None:
                params["title"] = scale_name
        return params

    # -- setup_params (guide-bins.R:226-230) ---------------------------------

    @staticmethod
    def setup_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Force nrow/ncol/n_breaks/n_key_layers to 1.

        Bins guide is single-row / single-column — it draws one strip
        of keys along the chosen direction.
        """
        params = dict(params)
        params["nrow"] = 1
        params["ncol"] = 1
        params["n_breaks"] = 1
        params["n_key_layers"] = 1
        return params

    # -- setup_elements (guide-bins.R:232-259) -------------------------------

    def setup_elements(
        self,
        params: Dict[str, Any],
        elements: Optional[Dict[str, str]] = None,
        theme: Any = None,
    ) -> Dict[str, Any]:
        """Populate text_position / axis line / tick defaults.

        Validates that ``legend.text.position`` agrees with
        ``params["direction"]`` — horizontal direction must use top /
        bottom, vertical must use left / right.
        """
        if elements is None:
            elements = dict(self.elements)
        direction = params.get("direction") or "horizontal"
        valid_position = {
            "horizontal": ["bottom", "top"],
            "vertical": ["right", "left"],
        }.get(direction, ["bottom", "top"])

        # R: replace_null(theme, legend.text.position, axis_line, ticks)
        if isinstance(theme, dict):
            theme = dict(theme)
            if theme.get("legend.text.position") is None:
                theme["legend.text.position"] = valid_position[0]

        # Let the parent resolve; then validate text_position
        elems = super().setup_elements(params, elements, theme)

        tp = elems.get("text_position") if isinstance(elems, dict) else None
        if tp is not None and tp not in valid_position:
            cli_abort(
                f"When `direction` is {direction!r}, `legend.text.position` "
                f"must be one of {valid_position!r}, not {tp!r}."
            )
        return elems

    # -- build_labels (guide-bins.R:261-280) ---------------------------------

    @staticmethod
    def build_labels(
        key: pd.DataFrame,
        elements: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build label grobs; blanks out non-limit ends per show.limits.

        Returns a dict with a ``labels`` entry so downstream callers can
        treat label output uniformly.
        """
        if key is None or len(key) == 0:
            return {"labels": None}
        n_labels = len(key[".label"]) if ".label" in key.columns else 0
        if n_labels < 1:
            return {"labels": None}

        labels = key[".label"].astype(object).tolist()
        show_limits = params.get("show.limits", [True, True])
        if isinstance(show_limits, bool):
            show_limits = [show_limits, show_limits]
        # R: key$.label[c(1, n_labels)[!params$show.limits]] <- ""
        if n_labels >= 1 and not show_limits[0]:
            labels[0] = ""
        if n_labels >= 1 and not show_limits[-1]:
            labels[-1] = ""

        values = key[".value"].astype(float).to_numpy()
        if params.get("direction") == "vertical":
            values = 1.0 - values
        return {
            "labels": {"text": labels, "x": values,
                       "flip": params.get("direction") == "vertical"},
        }

    # -- build_ticks (guide-bins.R:282-288) ----------------------------------

    @staticmethod
    def build_ticks(
        key: pd.DataFrame,
        elements: Dict[str, Any],
        params: Dict[str, Any],
        position: Any = None,
    ) -> Dict[str, Any]:
        """Build tick mark specs; NA out non-limit ends per show.limits.

        Delegates to :meth:`Guide.build_ticks`. Returns the value array
        used by the downstream tick drawer.
        """
        if key is None or len(key) == 0:
            return {"ticks": None}
        values = key[".value"].astype(float).to_numpy().copy()
        if params.get("direction") == "vertical":
            values = 1.0 - values
        show_limits = params.get("show.limits", [True, True])
        if isinstance(show_limits, bool):
            show_limits = [show_limits, show_limits]
        if len(values) >= 1 and not show_limits[0]:
            values[0] = np.nan
        if len(values) >= 1 and not show_limits[-1]:
            values[-1] = np.nan
        return {"ticks": values}

    # -- build_decor (guide-bins.R:290-331) ----------------------------------

    @staticmethod
    def build_decor(
        decor: Any,
        grobs: Any,
        elements: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble key strip + axis line + tick marks.

        Delegates key building to :meth:`GuideLegend.build_decor` (or
        the imperative legend helpers in :mod:`ggplot2_py.guide_legend`
        when they exist) and positions an axis line alongside based on
        ``text_position``.

        Returns
        -------
        dict
            ``{"keys": gtable-like, "axis_line": spec, "ticks": ticks}``.
        """
        n_keys = (len(params["key"]) - 1) if params.get("key") is not None \
            and len(params["key"]) > 0 else 0
        params = dict(params)
        params["n_breaks"] = n_keys

        # Delegate key-strip to parent build_decor if available
        keys_grob = None
        parent_build = getattr(GuideLegend, "build_decor", None)
        if parent_build is not None:
            keys_grob = parent_build(decor, grobs, elements, params)

        # Axis line position: determined by text_position
        tp = elements.get("text_position") if isinstance(elements, dict) else None
        axis_spec = {
            "top":    {"x": [0.0, 1.0], "y": [1.0, 1.0]},
            "bottom": {"x": [0.0, 1.0], "y": [0.0, 0.0]},
            "left":   {"x": [0.0, 0.0], "y": [0.0, 1.0]},
            "right":  {"x": [1.0, 1.0], "y": [0.0, 1.0]},
        }.get(tp, {"x": [0.0, 1.0], "y": [0.0, 0.0]})

        ticks = grobs.get("ticks") if isinstance(grobs, dict) else None
        return {"keys": keys_grob, "axis_line": axis_spec, "ticks": ticks}

    # -- measure_grobs (guide-bins.R:333-339) --------------------------------

    @staticmethod
    def measure_grobs(
        grobs: Dict[str, Any],
        params: Dict[str, Any],
        elements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute key-strip widths / heights (in cm).

        Mirrors R's: sums the decor key sizes, then delegates to
        :meth:`GuideLegend.measure_grobs` for overall measurement.
        """
        decor = grobs.get("decor") if isinstance(grobs, dict) else None
        params = dict(params)
        if isinstance(decor, dict):
            keys = decor.get("keys")
        else:
            keys = None
        from ggplot2_py._utils import width_cm, height_cm
        try:
            w = float(np.sum(width_cm(keys))) if keys is not None else 0.0
        except (TypeError, ValueError):
            w = 0.0
        try:
            h = float(np.sum(height_cm(keys))) if keys is not None else 0.0
        except (TypeError, ValueError):
            h = 0.0
        params["sizes"] = {"widths": w, "heights": h}

        parent_measure = getattr(GuideLegend, "measure_grobs", None)
        if parent_measure is not None and parent_measure is not Guide.measure_grobs:
            return parent_measure(grobs, params, elements)
        return {"width": w, "height": h}


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

    Replaces standard tick placement with ticks at log10-spaced
    intervals (decade subdivisions).  R source:
    ``ggplot2/R/guide-axis-logticks.R``.

    Attributes
    ----------
    _class_name : str
        ``"GuideAxisLogticks"``.
    """

    _class_name: str = "GuideAxisLogticks"

    params: Dict[str, Any] = {
        **GuideAxis.params,
        # Defaults mirror R's GuideAxisLogticks$params (guide-axis-logticks.R:153-165)
        "prescale_base": None,
        "negative_small": 0.1,
        "minor.ticks": True,  # for spacing calculation
        "long": 2.25,
        "mid": 1.5,
        "short": 0.75,
        "expanded": True,
        "short_theme": None,
        "name": "axis_logticks",
    }

    available_aes: List[str] = ["x", "y"]

    # -- extract_params (guide-axis-logticks.R:168-215) ----------------------

    @staticmethod
    def extract_params(
        scale: Any,
        params: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build a log-tick shadow key.

        For a log-scaled axis, computes tick positions at
        ``1, 2, ..., 9 * 10^k``, reconstructs their type (major = 1,
        mid = 2, short = 3) and stashes the result in
        ``params["logkey"]``.  Also adjusts cap positions if ``cap`` is
        set, and appends the aesthetic to the guide name.

        Parameters
        ----------
        scale : Scale
            Trained continuous (usually log-transformed) scale.
        params : dict
            Current parameters.
        **kwargs : Any
            Accepts ``title``.

        Returns
        -------
        dict
            Parameters with added ``logkey`` and updated ``name``.
        """
        if getattr(scale, "is_discrete", lambda: False)():
            cli_abort("Cannot calculate logarithmic ticks for discrete scales.")

        aesthetic = params.get("aesthetic", "x")
        params = dict(params)
        params["name"] = f"{params.get('name', 'axis_logticks')}_{aesthetic}"

        get_trans = getattr(scale, "get_transformation",
                            lambda: _IdentityTrans())
        prescale_base = params.get("prescale_base") or params.get("prescale.base")
        if prescale_base is not None:
            trans_name = getattr(get_trans(), "name", "identity")
            if trans_name != "identity":
                cli_warn(
                    f"The `prescale.base` argument will override the scale's "
                    f"`{trans_name}` transformation in log-tick positioning."
                )
            from scales import transform_log
            transformation = transform_log(base=prescale_base)
        else:
            transformation = get_trans()

        # Reconstruct original range: transformation$inverse(scale$get_limits())
        scale_limits = getattr(scale, "get_limits", lambda: (1.0, 10.0))()
        try:
            limits = transformation.inverse(np.asarray(scale_limits, dtype=float))
        except (AttributeError, TypeError):
            limits = np.asarray(scale_limits, dtype=float)

        negative_small = (params.get("negative_small")
                          or params.get("negative.small"))
        from scales import minor_breaks_log as _mbl
        ticks_fn = _mbl(smallest=negative_small)
        ticks = np.asarray(ticks_fn(np.asarray(limits, dtype=float)),
                           dtype=float)
        # Classify tick type (1 = major, 2 = mid, 3 = short) based on the
        # leading digit of |tick|/10^floor(log10(|tick|)).
        # R stores `attr(ticks, 'detail')` as c(10, 5, 1) indices; we
        # reproduce via abs-magnitude.
        with np.errstate(divide="ignore", invalid="ignore"):
            exps = np.floor(np.log10(np.where(ticks > 0, ticks, np.nan)))
            leading = ticks / np.power(10.0, exps)
        tick_type = np.full(len(ticks), 3, dtype=int)  # short
        tick_type[np.isclose(leading, 5.0, atol=1e-6)] = 2  # mid
        tick_type[np.isclose(leading, 1.0, atol=1e-6)] = 1  # major

        # Transform back into axis space
        try:
            ticks_axis = transformation.transform(ticks)
        except (AttributeError, TypeError):
            ticks_axis = ticks

        logkey = pd.DataFrame({
            aesthetic: ticks_axis,
            ".type": tick_type,
        })

        # Discard out-of-bounds ticks using expanded or literal range
        if params.get("expanded", True):
            range_ = getattr(scale, "continuous_range", None) \
                or getattr(scale, "get_limits", lambda: (-np.inf, np.inf))()
        else:
            range_ = getattr(scale, "get_limits", lambda: (-np.inf, np.inf))()
        r0, r1 = float(range_[0]), float(range_[1])
        mask = (logkey[aesthetic] >= r0) & (logkey[aesthetic] <= r1)
        logkey = logkey.loc[mask].reset_index(drop=True)

        # Cap adjustment
        decor = params.get("decor")
        cap = params.get("cap", "none")
        if cap in ("both", "upper") and decor is not None and not logkey.empty \
                and aesthetic in decor.columns and len(decor) >= 2:
            decor = decor.copy()
            decor.iloc[1, decor.columns.get_loc(aesthetic)] = \
                float(logkey[aesthetic].max())
            params["decor"] = decor
        if cap in ("both", "lower") and decor is not None and not logkey.empty \
                and aesthetic in decor.columns and len(decor) >= 1:
            decor = decor.copy() if not decor.flags.writeable else decor
            decor.iloc[0, decor.columns.get_loc(aesthetic)] = \
                float(logkey[aesthetic].min())
            params["decor"] = decor

        params["logkey"] = logkey

        # Title (parent behaviour)
        title = kwargs.get("title")
        scale_name = getattr(scale, "name", None)
        if is_waiver(params.get("title")):
            if title is not None and not is_waiver(title):
                params["title"] = title
            elif scale_name is not None:
                params["title"] = scale_name
        return params

    # -- transform (guide-axis-logticks.R:217-222) ---------------------------

    @staticmethod
    def transform(
        params: Dict[str, Any],
        coord: Any,
        panel_params: Any = None,
    ) -> Dict[str, Any]:
        """Transform both the standard key and the logkey shadow.

        The logkey requires the same coord transform as the main key so
        that tick x/y coordinates line up with the data space.
        """
        params = GuideAxis.transform(params, coord, panel_params)
        logkey = params.get("logkey")
        if logkey is not None and not logkey.empty and hasattr(coord, "transform"):
            aesthetic = params.get("aesthetic", "x")
            ortho = "y" if aesthetic == "x" else "x"
            position = params.get("position")
            override = -np.inf if position in ("bottom", "left") else np.inf
            if ortho not in logkey.columns:
                logkey = logkey.copy()
                logkey[ortho] = override
            params["logkey"] = coord.transform(logkey, panel_params)
        return params

    # -- override_elements (guide-axis-logticks.R:224-242) -------------------

    @staticmethod
    def override_elements(
        params: Dict[str, Any],
        elements: Dict[str, Any],
        theme: Any,
    ) -> Dict[str, Any]:
        """Resolve three tick lengths (long / mid / short).

        Multiplies ``rel()`` values by the theme's
        ``axis.ticks.length``; fixed :class:`~grid_py.Unit` values pass
        through untouched.  Records ``major_length`` / ``minor_length``
        so the legend-spacing calculation still works.
        """
        elements = GuideAxis.override_elements(params, elements, theme)
        if not isinstance(elements, dict):
            elements = dict(elements) if elements is not None else {}
        major_length = elements.get("major_length", 1.0)
        # Inherit short ticks from minor ticks element
        elements["short"] = params.get("short_theme") or elements.get("minor")
        # Multiply rel units with theme's tick length
        lengths: Dict[str, Any] = {}
        for key in ("long", "mid", "short"):
            val = params.get(key)
            if val is None:
                continue
            # Is this a unit object (has a value + type) or a bare number?
            is_unit = hasattr(val, "value") and hasattr(val, "units")
            if is_unit:
                lengths[key] = val
            else:
                try:
                    lengths[key] = float(val) * float(major_length)
                except (TypeError, ValueError):
                    lengths[key] = major_length
        elements["tick_length"] = lengths
        numeric_lengths = [v for v in lengths.values()
                           if isinstance(v, (int, float, np.integer, np.floating))]
        if numeric_lengths:
            elements["major_length"] = float(max(numeric_lengths))
            elements["minor_length"] = float(min(numeric_lengths))
        return elements

    # -- build_ticks (guide-axis-logticks.R:244-265) -------------------------

    @staticmethod
    def build_ticks(
        key: pd.DataFrame,
        elements: Dict[str, Any],
        params: Dict[str, Any],
        position: Any = None,
    ) -> Dict[str, Any]:
        """Build three tick-length groups from the logkey.

        Returns a dict with separate ``long``, ``mid``, ``short`` keys —
        each a subset of the logkey filtered by ``.type``. The render
        pipeline draws them at the corresponding length.
        """
        logkey = params.get("logkey")
        if logkey is None or logkey.empty:
            return {"long": None, "mid": None, "short": None}
        if position is None:
            position = params.get("opposite") or params.get("position")
        tick_lengths = elements.get("tick_length", {}) if isinstance(elements, dict) else {}
        long = logkey.loc[logkey[".type"] == 1].reset_index(drop=True)
        mid = logkey.loc[logkey[".type"] == 2].reset_index(drop=True)
        short = logkey.loc[logkey[".type"] == 3].reset_index(drop=True)
        return {
            "long": {"key": long, "length": tick_lengths.get("long")},
            "mid":  {"key": mid,  "length": tick_lengths.get("mid")},
            "short":{"key": short, "length": tick_lengths.get("short")},
        }


class _IdentityTrans:
    """Lightweight identity transformation used when a scale is
    missing :meth:`get_transformation` or it returns ``None``."""

    name = "identity"

    @staticmethod
    def transform(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    @staticmethod
    def inverse(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)


# ============================================================================
# GuideAxisStack -- stacked axis guides
# ============================================================================

class GuideAxisStack(GuideAxis):
    """Stacked axis guide combining multiple axis guides.

    R source: ``ggplot2/R/guide-axis-stack.R``.

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

    # hashables: title, class-names of child guides, name
    hashables: List[str] = ["title", "guides", "name"]

    # -- train (guide-axis-stack.R:108-123) ----------------------------------

    def train(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale: Any = None,
        aesthetic: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Train each child guide on the shared scale / aesthetic.

        Position / angle are propagated from the stack's own parameters
        so children inherit them when they don't set an override.
        """
        if params is None:
            params = dict(self.params)
        position = params.get("position")
        valid = list(_TRBL) + ["theta", "theta.sec"]
        if position is not None and not is_waiver(position) and position not in valid:
            cli_abort(
                f"`position` must be one of {valid!r}, not {position!r}."
            )

        guides = params.get("guides") or []
        guide_params = params.get("guide_params") or []
        while len(guide_params) < len(guides):
            guide_params.append({})
        stack_angle = params.get("angle")

        for i, child in enumerate(guides):
            child_params = dict(guide_params[i])
            if position is not None:
                child_params["position"] = position
            # R: params$guide_params[[i]]$angle <- angle %|W|% stack_angle
            child_angle = child_params.get("angle")
            if child_angle is None or is_waiver(child_angle):
                child_params["angle"] = stack_angle
            trained = child.train(
                params=child_params,
                scale=scale,
                aesthetic=aesthetic,
                **kwargs,
            )
            guide_params[i] = trained if trained is not None else child_params

        params["guide_params"] = guide_params
        return params

    # -- transform (guide-axis-stack.R:125-134) ------------------------------

    @staticmethod
    def transform(
        params: Dict[str, Any],
        coord: Any,
        panel_params: Any = None,
    ) -> Dict[str, Any]:
        """Propagate :meth:`transform` to each child guide."""
        params = dict(params)
        guides = params.get("guides") or []
        guide_params = params.get("guide_params") or []
        for i, child in enumerate(guides):
            if i >= len(guide_params):
                break
            transformed = child.transform(
                params=guide_params[i],
                coord=coord,
                panel_params=panel_params,
            )
            guide_params[i] = transformed
        params["guide_params"] = guide_params
        return params

    # -- get_layer_key (guide-axis-stack.R:137-145) --------------------------

    def get_layer_key(
        self,
        params: Dict[str, Any],
        layers: List[Any],
        data: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Forward get_layer_key to every child guide."""
        params = dict(params)
        guides = params.get("guides") or []
        guide_params = params.get("guide_params") or []
        for i, child in enumerate(guides):
            if i >= len(guide_params):
                break
            guide_params[i] = child.get_layer_key(
                params=guide_params[i],
                layers=layers,
                data=data,
            )
        params["guide_params"] = guide_params
        return params

    # -- draw (guide-axis-stack.R:147-253) -----------------------------------

    def draw(
        self,
        theme: Any = None,
        position: Optional[str] = None,
        direction: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Stack each child guide's drawing along the axis direction.

        For Cartesian positions (top/right/bottom/left) the children
        are placed in a gtable with the stack's spacing between them;
        for theta positions each child accumulates a radial offset.
        Returns ``None`` if every child produced an empty grob.
        """
        if params is None:
            params = dict(self.params)
        pos = params.get("position") or position
        dir_ = params.get("direction") or direction
        draw_label = params.get("draw_label", True)
        guides = params.get("guides") or []
        guide_params = params.get("guide_params") or []
        idx_range = range(len(guides)) if draw_label else range(min(1, len(guides)))

        # Default spacing: axis.ticks.length from theme when not provided
        spacing = params.get("spacing")

        # Theta-style stacking
        if pos in ("theta", "theta.sec"):
            grobs: List[Any] = []
            offset = 0.0
            default_spacing_cm = spacing if spacing is not None else 0.079  # 2.25pt
            for i in idx_range:
                pars = dict(guide_params[i])
                pars["stack_offset"] = offset
                grob_i = guides[i].draw(
                    theme=theme, position=pos, direction=dir_, params=pars,
                )
                grobs.append(grob_i)
                if grob_i is not None and hasattr(grob_i, "offset"):
                    offset += float(default_spacing_cm) + float(grob_i.offset)
            return {"children": grobs, "stack": "theta"}

        # Cartesian stacking
        grobs = []
        for i in idx_range:
            pars = dict(guide_params[i])
            pars["draw_label"] = draw_label
            grobs.append(
                guides[i].draw(
                    theme=theme, position=pos, direction=dir_, params=pars,
                )
            )

        # Remove empty grobs
        def _is_zero(g: Any) -> bool:
            return (g is None
                    or (hasattr(g, "is_zero") and bool(g.is_zero))
                    or (hasattr(g, "__len__") and len(g) == 0))
        grobs = [g for g in grobs if not _is_zero(g)]
        if not grobs:
            return None

        # For Cartesian stack: along-axis dimension is rows for top/bottom,
        # cols for left/right
        if pos in ("top", "left"):
            grobs = list(reversed(grobs))

        layout = "rows" if pos in ("top", "bottom") else "cols"
        return {
            "children": grobs,
            "stack": "cartesian",
            "position": pos,
            "layout": layout,
            "spacing": spacing,
        }


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

    # ``plot.update_ggplot`` duck-types on this flag to route a ``Guides``
    # added via ``ggplot() + guides(...)`` onto ``plot.guides`` (avoids a
    # circular import from registering on the singledispatch generic).
    _is_guides: bool = True

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

    def get_position(
        self,
        position: str,
        theme: Any = None,
        direction: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return guides registered at *position*.

        Mirrors R ``Guides$get_position`` (guides-.R:211-238).

        Filters ``self.guides`` / ``self.params`` by the ``position``
        entry in each params dict (matching ``position[1]`` in R for
        multi-value positions).  When more than one guide matches the
        position, they are merged in ``order`` order using the guide's
        own ``merge`` method — exactly what R does.

        Parameters
        ----------
        position : str
            One of ``"top"``, ``"right"``, ``"bottom"``, ``"left"``,
            ``"inside"``.
        theme, direction : optional
            Accepted for API symmetry with R (which does not use them
            in the core logic, but downstream calls inspect them).

        Returns
        -------
        dict or None
            ``{"guide": Guide, "params": dict}`` for the resolved guide,
            or a ``{"guide": GuideNone, "params": ...}`` sentinel when
            no guide matches.  Returns ``None`` only when the container
            is empty.
        """
        if not isinstance(position, str):
            cli_abort("`position` must be a string.")

        guides_list = (
            list(self.guides.values())
            if isinstance(self.guides, dict)
            else self.guides
        )
        params_list = self.params

        # Extract each guide's position (first element if multi-valued).
        guide_positions: List[Optional[str]] = []
        for p in params_list:
            pos = p.get("position") if isinstance(p, dict) else None
            if isinstance(pos, (list, tuple, np.ndarray)):
                pos = pos[0] if len(pos) > 0 else None
            guide_positions.append(pos)

        idx = [i for i, pos in enumerate(guide_positions) if pos == position]

        if len(idx) < 1:
            return {
                "guide": self._missing,
                "params": dict(getattr(self._missing, "params", {})),
            }
        if len(idx) == 1:
            i = idx[0]
            return {
                "guide": guides_list[i],
                "params": dict(params_list[i]),
            }

        # Multiple guides at the same position — merge in ``order`` order.
        pairs = [
            {"guide": guides_list[i], "params": dict(params_list[i])}
            for i in idx
        ]
        order_key = [
            float(pairs[j]["params"].get("order", 0) or 0)
            for j in range(len(pairs))
        ]
        perm = sorted(range(len(pairs)), key=lambda j: order_key[j])

        result = pairs[perm[0]]
        for j in perm[1:]:
            nxt = pairs[j]
            result = result["guide"].merge(
                result["params"], nxt["guide"], nxt["params"]
            )
        return result

    def get_custom(self, position: Optional[str] = None) -> "Guides":
        """Return only the ``GuideCustom`` entries as a new ``Guides``.

        Mirrors R ``Guides$get_custom`` (guides-.R:240-250).

        Parameters
        ----------
        position : str, optional
            Unused by R; accepted for forward-compatibility when a
            caller wants to filter customs by position.

        Returns
        -------
        Guides
            A fresh ``Guides`` container holding only the ``GuideCustom``
            entries from ``self``.  When none match, returns an empty
            ``Guides``.
        """
        guides_list = (
            list(self.guides.values())
            if isinstance(self.guides, dict)
            else self.guides
        )
        mask = [isinstance(g, GuideCustom) for g in guides_list]
        n_custom = sum(mask)
        if n_custom < 1:
            return Guides()

        custom = Guides()
        custom.guides = [g for g, m in zip(guides_list, mask) if m]
        # R: custom$params <- lapply(custom$guides, `[[`, "params")
        custom.params = [dict(getattr(g, "params", {})) for g in custom.guides]
        if self.aesthetics:
            custom.aesthetics = [
                a for a, m in zip(self.aesthetics, mask) if m
            ]
        else:
            custom.aesthetics = [""] * n_custom

        if position is not None:
            # Optional forward-compatibility filter.
            keep = []
            for p in custom.params:
                pos = p.get("position") if isinstance(p, dict) else None
                if isinstance(pos, (list, tuple, np.ndarray)):
                    pos = pos[0] if len(pos) > 0 else None
                keep.append(pos == position)
            custom.guides = [g for g, k in zip(custom.guides, keep) if k]
            custom.params = [p for p, k in zip(custom.params, keep) if k]
            custom.aesthetics = [
                a for a, k in zip(custom.aesthetics, keep) if k
            ]
        return custom

    def package_box(
        self,
        grobs: List[Any],
        position: str,
        theme: Any,
        direction: Optional[str] = None,
    ) -> Any:
        """Arrange guide grobs into a single guide-box gtable.

        Mirrors R ``Guides$package_box`` (guides-.R:592-757).

        For ``position`` in ``("top", "bottom")`` the legends are laid
        out in a single row (horizontal); otherwise they are stacked
        into a column (vertical).  Spacing is taken from theme
        entries ``legend.spacing.x`` / ``legend.spacing.y`` and the
        outer box padding from ``legend.box.margin``.  A
        ``legend.box.background`` rectangle is inserted at the bottom
        of the z-stack.

        Parameters
        ----------
        grobs : list
            Rendered guide grobs (one per guide at this position).
        position : str
            Position of the guide-box; drives default direction.
        theme : Theme or dict
            Plot theme.  ``legend.box.*`` entries are consulted.
        direction : str, optional
            Override direction; when ``None`` it is derived from
            ``position`` (R behaviour).

        Returns
        -------
        Gtable
            The packaged guide-box (``name="guide-box"``).
        """
        from gtable_py import (
            Gtable,
            gtable_add_col_space,
            gtable_add_grob,
            gtable_add_padding,
            gtable_add_row_space,
            gtable_col,
            gtable_row,
        )
        from grid_py import Unit

        # Filter zero grobs (R uses is_zero()).
        def _is_zero(g: Any) -> bool:
            if g is None:
                return True
            cls = getattr(g, "_grid_class", "")
            name = getattr(g, "_name", getattr(g, "name", ""))
            return cls == "null" or (
                isinstance(name, str)
                and ("zero" in name.lower() or name == "NULL")
            )

        grobs = [g for g in (grobs or []) if not _is_zero(g)]
        if not grobs:
            return Gtable(name="guide-box")

        if direction is None:
            direction = (
                "horizontal" if position in ("top", "bottom") else "vertical"
            )

        # Read theme values via calc_element when available, else bare dict.
        def _theme_get(key: str, default: Any = None) -> Any:
            if theme is None:
                return default
            if hasattr(theme, "__getitem__"):
                try:
                    v = theme[key]
                    if v is not None:
                        return v
                except (KeyError, TypeError):
                    pass
            return default

        def _calc(key: str, default: Any = None) -> Any:
            try:
                from ggplot2_py.theme_elements import calc_element
                v = calc_element(key, theme) if theme is not None else None
                if v is not None:
                    return v
            except Exception:
                pass
            return _theme_get(key, default)

        def _unit_to_cm(u: Any, fallback_cm: float = 0.0) -> float:
            if u is None:
                return fallback_cm
            if isinstance(u, (int, float)):
                return float(u)
            try:
                from grid_py import convert_unit
                cm = convert_unit(u, "cm", valueOnly=True)
                return float(np.sum(cm))
            except Exception:
                return fallback_cm

        # R: box spacing defaults
        spacing_x_cm = _unit_to_cm(
            _calc("legend.spacing.x") or _calc("legend.spacing"),
            fallback_cm=0.2,
        )
        spacing_y_cm = _unit_to_cm(
            _calc("legend.spacing.y") or _calc("legend.spacing"),
            fallback_cm=0.2,
        )

        # Measure each grob. The R code uses the raw widths/heights
        # units; we resolve them to cm via the gtable helpers.
        widths_cm: List[float] = []
        heights_cm: List[float] = []
        for g in grobs:
            w_u = getattr(g, "widths", None) or getattr(g, "_widths", None)
            h_u = getattr(g, "heights", None) or getattr(g, "_heights", None)
            widths_cm.append(_unit_to_cm(w_u))
            heights_cm.append(_unit_to_cm(h_u))

        if direction == "horizontal":
            total_height = max(heights_cm) if heights_cm else 0.0
            guides = gtable_row(
                name="guides",
                grobs=grobs,
                height=Unit([total_height], ["cm"]),
                widths=Unit(widths_cm, ["cm"] * len(widths_cm)),
            )
            guides = gtable_add_col_space(guides, Unit([spacing_x_cm], ["cm"]))
        else:
            total_width = max(widths_cm) if widths_cm else 0.0
            guides = gtable_col(
                name="guides",
                grobs=grobs,
                width=Unit([total_width], ["cm"]),
                heights=Unit(heights_cm, ["cm"] * len(heights_cm)),
            )
            guides = gtable_add_row_space(guides, Unit([spacing_y_cm], ["cm"]))

        # Outer margin (legend.box.margin). R default: margin()
        box_margin = _calc("legend.box.margin")
        if box_margin is not None:
            # margin elements are expected to be a 4-unit vector;
            # fall back to zero padding if we cannot interpret it.
            try:
                guides = gtable_add_padding(guides, box_margin)
            except Exception:
                pass

        # Background rectangle (legend.box.background).
        bg = _calc("legend.box.background")
        if bg is not None:
            try:
                from ggplot2_py.theme_elements import element_grob
                bg_grob = element_grob(bg)
                if bg_grob is not None:
                    guides = gtable_add_grob(
                        guides,
                        bg_grob,
                        t=1, l=1, b=-1, r=-1,
                        z=-np.inf,
                        clip="off",
                        name="legend.box.background",
                    )
            except Exception:
                pass

        guides.name = "guide-box"
        return guides

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

        # Record suppressed aesthetics before dropping GuideNone entries.
        # Downstream renderers (``plot_render._table_add_legends``) check
        # this set to skip scale-derived legends for aesthetics the user
        # explicitly blanked via ``guides(<aes>='none')``.
        existing = getattr(self, "suppressed_aesthetics", set()) or set()
        self.suppressed_aesthetics = set(existing) | {
            self.aesthetics[i]
            for i, g in enumerate(self.guides)
            if isinstance(g, GuideNone) and i < len(self.aesthetics)
        }

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
            # Preserve the ``suppressed_aesthetics`` set that
            # ``Guides.train`` records before dropping GuideNone entries —
            # downstream ``plot_render._table_add_legends`` reads it to
            # skip scale-derived legends the user disabled via
            # ``guides(<aes>='none')``.
            empty = Guides()
            empty.suppressed_aesthetics = getattr(guides, "suppressed_aesthetics", set())
            return empty

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

        Mirrors R ``Guides$assemble`` (guides-.R:474-568): draw each
        guide, group by position, then package each group into a
        guide-box via ``package_box``.

        Parameters
        ----------
        theme : Theme
            Plot theme.

        Returns
        -------
        dict
            Mapping of position -> gtable (``"top"``, ``"right"``,
            ``"bottom"``, ``"left"``, ``"inside"``).  Missing positions
            are set to ``None`` (R uses ``zeroGrob()``).  Returns
            ``None`` when ``theme$legend.position == "none"`` or when
            no guides have been registered.
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

        # Numeric 2-vector -> inside placement (R: guides-.R:481-483).
        if isinstance(default_position, (list, tuple, np.ndarray)) and len(
            default_position
        ) == 2:
            default_position = "inside"

        # R: legend.position="none" suppresses all legends.
        if default_position == "none":
            return None
        if default_position not in ("top", "right", "bottom", "left", "inside"):
            return None

        positions = []
        for p in self.params:
            pos = p.get("position") or default_position
            if is_waiver(pos):
                pos = default_position
            positions.append(pos)

        grobs = self.draw(theme, positions)

        # Group by position and package each group via ``package_box``
        # (mirrors the vec_group_loc + package_box loop in R).
        groups: Dict[str, List[Any]] = {}
        for pos, g in zip(positions, grobs):
            if g is None:
                continue
            groups.setdefault(pos, []).append(g)

        # Emit all five R slot keys; missing positions -> None (caller
        # may treat as zeroGrob, matching R's
        # ``grobs[setdiff(c(.trbl, "inside"), names(grobs))] <- list(zeroGrob())``).
        out: Dict[str, Any] = {k: None for k in ("top", "right", "bottom", "left", "inside")}
        for pos, group_grobs in groups.items():
            out[pos] = self.package_box(group_grobs, pos, theme)
        return out


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


# ============================================================================
# Register Guides with update_ggplot (R plot-construction.R:180-187)
# ============================================================================
#
# R:
#     S7::method(update_ggplot, list(class_guides, class_ggplot)) <-
#       function(object, plot, ...) {
#         old <- plot@guides
#         new <- ggproto(NULL, old)   # clone before mutating
#         new$add(object)
#         plot@guides <- new
#         plot
#       }
#
# ggplot2_py mirrors this: allocate a fresh Guides, copy old fields into it,
# then apply the user-supplied Guides. The copy is deliberate — ``GGPlot._clone``
# is shallow and does not clone ``guides``, so mutating ``plot.guides`` in
# place would bleed across ``p1`` / ``p2 = p1 + guides(...)``.

from ggplot2_py.plot import update_ggplot as _update_ggplot


def _clone_guides(old: "Guides") -> "Guides":
    """Return a new Guides whose mutable fields are independent of *old*.

    Mirrors R ``ggproto(NULL, old)``'s copy-on-write semantics for the
    guides / params / aesthetics collections.
    """
    new = Guides()
    new.guides = dict(old.guides)
    new.params = list(old.params)
    new.aesthetics = list(old.aesthetics)
    new._missing = old._missing  # shared; set once at init, never mutated
    return new


@_update_ggplot.register(Guides)
def _update_ggplot_guides(obj: Guides, plot: Any, object_name: str = "") -> Any:
    old = plot.guides if plot.guides is not None else Guides()
    new = _clone_guides(old)
    new.add(obj)
    plot.guides = new
    return plot

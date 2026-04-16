"""
Faceting system for ggplot2.

Facets control how data is split into subsets and displayed as a matrix
of panels. The base :class:`Facet` class defines the interface; concrete
implementations include :class:`FacetNull` (no faceting),
:class:`FacetGrid` (rows x columns grid), and :class:`FacetWrap`
(1-d ribbon wrapped into 2-d).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn
from ggplot2_py.ggproto import GGProto, ggproto
from ggplot2_py._utils import snake_class, compact, modify_list, empty


def _is_null_grob(grob: Any) -> bool:
    """Check if a grob is a null grob (R semantics: zeroGrob / nullGrob)."""
    if grob is None:
        return True
    cls = getattr(grob, "_grid_class", "")
    name = getattr(grob, "_name", getattr(grob, "name", ""))
    return cls == "null" or "null" in str(name).lower() or "zero" in str(name).lower()


def _axis_width_cm(ax: Any) -> float:
    """Measure axis grob width in cm.

    R measures axis width via ``gtable_width(gt)`` + ``convertUnit(..., "cm")``.
    No fallback — if measurement fails, let it surface.
    """
    from gtable_py import Gtable, gtable_width
    from grid_py import convert_width
    if isinstance(ax, Gtable):
        w = gtable_width(ax)
        result = convert_width(w, "cm", valueOnly=True)
        return float(np.sum(result))
    # _AbsoluteAxisGrob path
    val = getattr(ax, "_width_cm", None)
    if val is not None:
        return val
    # width_details path
    if hasattr(ax, "width_details"):
        from ggplot2_py.guide_axis import _width_cm
        return _width_cm(ax)
    raise ValueError(f"Cannot measure width of {type(ax).__name__}")


def _axis_height_cm(ax: Any) -> float:
    """Measure axis grob height in cm.

    R measures axis height via ``gtable_height(gt)`` + ``convertUnit(..., "cm")``.
    No fallback — if measurement fails, let it surface.
    """
    from gtable_py import Gtable, gtable_height
    from grid_py import convert_height
    if isinstance(ax, Gtable):
        h = gtable_height(ax)
        result = convert_height(h, "cm", valueOnly=True)
        return float(np.sum(result))
    val = getattr(ax, "_height_cm", None)
    if val is not None:
        return val
    if hasattr(ax, "height_details"):
        from ggplot2_py.guide_axis import _height_cm
        return _height_cm(ax)
    raise ValueError(f"Cannot measure height of {type(ax).__name__}")

__all__ = [
    "Facet",
    "FacetNull",
    "FacetGrid",
    "FacetWrap",
    "facet_null",
    "facet_grid",
    "facet_wrap",
    "is_facet",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layout_null() -> pd.DataFrame:
    """Return a single-panel layout.

    Returns
    -------
    pd.DataFrame
        One-row layout with columns ``PANEL``, ``ROW``, ``COL``,
        ``SCALE_X``, ``SCALE_Y``.
    """
    return pd.DataFrame({
        "PANEL": pd.Categorical([1]),
        "ROW": [1],
        "COL": [1],
        "SCALE_X": [1],
        "SCALE_Y": [1],
    })


def _wrap_dims(n: int, nrow: Optional[int] = None, ncol: Optional[int] = None) -> Tuple[int, int]:
    """Compute grid dimensions for *n* panels.

    Parameters
    ----------
    n : int
        Number of panels.
    nrow, ncol : int or None

    Returns
    -------
    tuple of (nrow, ncol)

    Raises
    ------
    ValueError
        If the grid is too small for *n* panels.
    """
    if nrow is None and ncol is None:
        ncol = math.ceil(math.sqrt(n))
        nrow = math.ceil(n / ncol)
    elif ncol is None:
        ncol = math.ceil(n / nrow)
    elif nrow is None:
        nrow = math.ceil(n / ncol)

    if nrow * ncol < n:
        cli_abort(
            f"Need {n} panels, but nrow*ncol = {nrow * ncol}. "
            "Increase nrow and/or ncol."
        )
    return nrow, ncol


def _resolve_facet_vars(facets: Any) -> List[str]:
    """Resolve *facets* specification to a list of column-name strings.

    Parameters
    ----------
    facets : str, list, tuple, or None
        Faceting variable specification.

    Returns
    -------
    list of str
    """
    if facets is None:
        return []
    if isinstance(facets, str):
        # Could be formula-like "a + b" or simple name
        parts = [s.strip() for s in facets.replace("~", " ").replace("+", " ").split()]
        return [p for p in parts if p and p != "."]
    if isinstance(facets, (list, tuple)):
        result = []
        for f in facets:
            if isinstance(f, str):
                result.append(f)
            else:
                result.append(str(f))
        return result
    if isinstance(facets, dict):
        return list(facets.keys())
    return []


def _combine_vars(
    data_list: List[pd.DataFrame],
    vars_: List[str],
    drop: bool = True,
) -> pd.DataFrame:
    """Combine the unique values of *vars_* across all datasets.

    Parameters
    ----------
    data_list : list of DataFrame
    vars_ : list of str
    drop : bool

    Returns
    -------
    pd.DataFrame
        Unique combinations of the faceting variables.
    """
    if not vars_:
        return pd.DataFrame()

    frames = []
    for df in data_list:
        if df is None or (isinstance(df, pd.DataFrame) and len(df) == 0):
            continue
        cols = [c for c in vars_ if c in df.columns]
        if cols:
            frames.append(df[cols].drop_duplicates())

    if not frames:
        return pd.DataFrame({v: pd.Series(dtype=object) for v in vars_})

    combined = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    # Fill missing columns
    for v in vars_:
        if v not in combined.columns:
            combined[v] = "(all)"
    return combined[vars_].reset_index(drop=True)


def _map_facet_data(
    data: pd.DataFrame,
    layout: pd.DataFrame,
    params: Dict[str, Any],
    facet_vars: List[str],
) -> pd.DataFrame:
    """Map data rows to panels.

    Parameters
    ----------
    data : pd.DataFrame
        Layer data.
    layout : pd.DataFrame
        Layout with faceting variable columns and ``PANEL``.
    params : dict
    facet_vars : list of str

    Returns
    -------
    pd.DataFrame
        Data with a ``PANEL`` column.
    """
    if data is None or (isinstance(data, pd.DataFrame) and len(data) == 0):
        return pd.DataFrame({"PANEL": pd.Categorical([])})

    if is_waiver(data):
        return pd.DataFrame({"PANEL": pd.Categorical([])})

    data = data.copy()
    if not facet_vars:
        data["PANEL"] = pd.Categorical([1] * len(data))
        return data

    # Match data to layout on facet vars
    present = [v for v in facet_vars if v in data.columns and v in layout.columns]
    if not present:
        # No matching vars: repeat across all panels
        data["PANEL"] = pd.Categorical([1] * len(data))
        return data

    # Merge to get PANEL assignment
    merged = data.merge(
        layout[present + ["PANEL"]],
        on=present,
        how="left",
    )
    # Rows that didn't match any panel get dropped
    merged = merged.dropna(subset=["PANEL"]).reset_index(drop=True)
    merged["PANEL"] = pd.Categorical(merged["PANEL"])
    return merged


# ---------------------------------------------------------------------------
# Base Facet
# ---------------------------------------------------------------------------

class Facet(GGProto):
    """Base facet class.

    Attributes
    ----------
    shrink : bool
        Whether to shrink scales to fit stat output.
    params : dict
        Faceting parameters (populated by the constructor).
    """

    # --- Auto-registration registry (Python-exclusive) -------------------
    _registry: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if name.startswith("Facet") and len(name) > 5:
            key = name[5:]
            Facet._registry[key] = cls
            Facet._registry[key.lower()] = cls

    shrink: bool = False
    params: Dict[str, Any] = {}

    def setup_params(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and modify faceting parameters.

        Parameters
        ----------
        data : list of DataFrame
            Global + layer data.
        params : dict

        Returns
        -------
        dict
        """
        all_cols: List[str] = []
        for df in data:
            if isinstance(df, pd.DataFrame):
                all_cols.extend(df.columns.tolist())
        params["_possible_columns"] = list(set(all_cols))
        return params

    def setup_data(
        self, data: List[pd.DataFrame], params: Dict[str, Any]
    ) -> List[pd.DataFrame]:
        """Modify data before processing.

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        list of DataFrame
        """
        return data

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Create the panel layout table.

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
            Must have ``PANEL``, ``ROW``, ``COL``, ``SCALE_X``, ``SCALE_Y``.

        Raises
        ------
        NotImplementedError
            In the base class.
        """
        cli_abort("compute_layout() is not implemented in the base Facet class.")

    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Assign data rows to panels via the ``PANEL`` column.

        Parameters
        ----------
        data : pd.DataFrame
        layout : pd.DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        cli_abort("map_data() is not implemented in the base Facet class.")

    def init_scales(
        self,
        layout: pd.DataFrame,
        x_scale: Any = None,
        y_scale: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, list]:
        """Initialise per-panel scales.

        Parameters
        ----------
        layout : pd.DataFrame
        x_scale, y_scale : Scale or None
            Prototype scales.
        params : dict

        Returns
        -------
        dict
            ``{"x": [scales...], "y": [scales...]}``.
        """
        scales: Dict[str, list] = {}
        if x_scale is not None:
            n_x = int(layout["SCALE_X"].max())
            scales["x"] = [x_scale] * n_x
        if y_scale is not None:
            n_y = int(layout["SCALE_Y"].max())
            scales["y"] = [y_scale] * n_y
        return scales

    def train_scales(
        self,
        x_scales: list,
        y_scales: list,
        layout: pd.DataFrame,
        data: List[pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train per-panel scales on data.

        Parameters
        ----------
        x_scales, y_scales : list
        layout : pd.DataFrame
        data : list of DataFrame
        params : dict
        """
        for layer_data in data:
            if layer_data is None or (hasattr(layer_data, "empty") and layer_data.empty):
                continue
            if "PANEL" not in layer_data.columns:
                continue
            for _, row in layout.iterrows():
                panel_id = row["PANEL"]
                sx_idx = int(row["SCALE_X"]) - 1
                sy_idx = int(row["SCALE_Y"]) - 1
                mask = layer_data["PANEL"] == panel_id
                panel_data = layer_data.loc[mask]
                if panel_data.empty:
                    continue
                if x_scales and sx_idx < len(x_scales):
                    x_scales[sx_idx].train_df(panel_data)
                if y_scales and sy_idx < len(y_scales):
                    y_scales[sy_idx].train_df(panel_data)

    def finish_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Final data adjustments.

        Parameters
        ----------
        data : pd.DataFrame
        layout : pd.DataFrame
        x_scales, y_scales : list
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        return data

    def draw_panels(
        self,
        panels: list,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        ranges: list,
        coord: Any,
        data: Any,
        theme: Any,
        params: Dict[str, Any],
    ) -> Any:
        """Assemble panels into a gtable with background, axes, and labels.

        Mirrors R's ``Facet$draw_panels`` → ``init_gtable`` → ``attach_axes``
        pipeline (facet-.R:501-532):

        1. Create panel-only gtable with null units (init_gtable)
        2. Decorate each panel with coord background + foreground
        3. Render axis grobs, measure them, and attach as new rows/columns

        Parameters
        ----------
        panels : list of grobs (per-layer, each containing per-panel grobs)
        layout : pd.DataFrame
        x_scales, y_scales : list
        ranges : list of panel_params dicts
        coord : Coord
        data : list
        theme : Theme
        params : dict

        Returns
        -------
        gtable
        """
        from grid_py import GTree, GList, null_grob, Viewport
        from gtable_py import Gtable, gtable_add_grob, gtable_add_rows, gtable_add_cols
        from grid_py import Unit as unit

        nrow = int(layout["ROW"].max()) if len(layout) > 0 else 1
        ncol = int(layout["COL"].max()) if len(layout) > 0 else 1

        # ── Step 1: init_gtable — panel-only matrix (R: facet-.R:562-612)
        # Panel sizes use "null" units (flexible, fill available space).
        # Aspect ratio from coord is encoded in the null-unit ratio.
        aspect_ratio = None
        if hasattr(coord, "aspect") and ranges:
            aspect_ratio = coord.aspect(ranges[0])

        panel_h = abs(aspect_ratio) if aspect_ratio is not None else 1.0
        widths = unit([1] * ncol, "null")
        heights = unit([panel_h] * nrow, "null")
        gt = Gtable(widths=widths, heights=heights, name="layout")

        # Mark respect flag for aspect ratio (R: facet-.R:592)
        if aspect_ratio is not None:
            gt._respect = True

        # ── Step 2: Place decorated panels into the gtable
        for _, row_info in layout.iterrows():
            panel_id = int(row_info["PANEL"])
            r = int(row_info["ROW"])
            c = int(row_info["COL"])
            panel_idx = panel_id - 1
            pp = ranges[panel_idx] if panel_idx < len(ranges) else {}

            # Collect geom grobs for this panel
            panel_grobs = []
            for layer_grobs in panels:
                if isinstance(layer_grobs, list) and panel_idx < len(layer_grobs):
                    panel_grobs.append(layer_grobs[panel_idx])
                elif not isinstance(layer_grobs, list) and layer_grobs is not None:
                    panel_grobs.append(layer_grobs)

            # Decorate panel with coord background + foreground
            if hasattr(coord, "draw_panel"):
                decorated = coord.draw_panel(panel_grobs, pp, theme)
            else:
                decorated = GTree(
                    children=GList(*panel_grobs),
                    name=f"panel-{panel_id}",
                )

            gt = gtable_add_grob(
                gt, decorated, t=r, l=c, name=f"panel-{r}-{c}",
            )

        # ── Step 3: Render axes and attach with measured sizes
        # (R: facet-.R attach_axes → seam_table → max_height/max_width)
        #
        # Render axis grobs for each unique scale, then attach as new
        # rows/columns with sizes from grob._height_cm / _width_cm.

        # Collect axis grobs across all panels to find max sizes
        left_axes = []
        bottom_axes = []
        top_axes = []
        right_axes = []

        for _, row_info in layout.iterrows():
            panel_idx = int(row_info["PANEL"]) - 1
            r = int(row_info["ROW"])
            c = int(row_info["COL"])
            pp = ranges[panel_idx] if panel_idx < len(ranges) else {}

            if hasattr(coord, "render_axis_v"):
                axes_v = coord.render_axis_v(pp, theme)
                if c == 1:
                    left_ax = axes_v.get("left")
                    if left_ax is not None and not _is_null_grob(left_ax):
                        left_axes.append((r, left_ax))
                if c == ncol:
                    right_ax = axes_v.get("right")
                    if right_ax is not None and not _is_null_grob(right_ax):
                        right_axes.append((r, right_ax))

            if hasattr(coord, "render_axis_h"):
                axes_h = coord.render_axis_h(pp, theme)
                if r == nrow:
                    bottom_ax = axes_h.get("bottom")
                    if bottom_ax is not None and not _is_null_grob(bottom_ax):
                        bottom_axes.append((c, bottom_ax))
                if r == 1:
                    top_ax = axes_h.get("top")
                    if top_ax is not None and not _is_null_grob(top_ax):
                        top_axes.append((c, top_ax))

        # Track column offset from left-axis insertion
        col_offset = 0

        # ── Attach left axis (R: seam_table side="left")
        if left_axes:
            max_w = max(_axis_width_cm(ax) for _, ax in left_axes)
            gt = gtable_add_cols(gt, unit([max_w], "cm"), pos=0)
            col_offset = 1
            for r, ax in left_axes:
                gt = gtable_add_grob(
                    gt, ax, t=r, l=1, clip="off", name=f"axis-l-{r}",
                )

        # ── Attach right axis (R: seam_table side="right")
        if right_axes:
            max_w = max(_axis_width_cm(ax) for _, ax in right_axes)
            gt = gtable_add_cols(gt, unit([max_w], "cm"), pos=-1)
            ncol_now = len(gt._widths)
            for r, ax in right_axes:
                gt = gtable_add_grob(
                    gt, ax, t=r, l=ncol_now, clip="off", name=f"axis-r-{r}",
                )

        # ── Attach bottom axis (R: seam_table side="bottom")
        if bottom_axes:
            max_h = max(_axis_height_cm(ax) for _, ax in bottom_axes)
            gt = gtable_add_rows(gt, unit([max_h], "cm"), pos=-1)
            nrow_now = len(gt._heights)
            for c, ax in bottom_axes:
                gt = gtable_add_grob(
                    gt, ax, t=nrow_now, l=c + col_offset, clip="off",
                    name=f"axis-b-{c}",
                )

        # ── Attach top axis (R: seam_table side="top")
        if top_axes:
            max_h = max(_axis_height_cm(ax) for _, ax in top_axes)
            gt = gtable_add_rows(gt, unit([max_h], "cm"), pos=0)
            for c, ax in top_axes:
                gt = gtable_add_grob(
                    gt, ax, t=1, l=c + col_offset, clip="off",
                    name=f"axis-t-{c}",
                )

        # ── Strip labels — pass col_offset so strips align with panels
        gt = self._add_strip_labels(
            gt, layout, nrow, ncol, params, theme, col_offset=col_offset,
        )

        return gt

    def _add_strip_labels(
        self,
        gt: Any,
        layout: pd.DataFrame,
        nrow: int,
        ncol: int,
        params: Dict[str, Any],
        theme: Any = None,
        col_offset: int = 0,
    ) -> Any:
        """Add facet strip text labels to the gtable.

        All visual properties resolved from theme via ``calc_element()``
        for strip.background.x/y and strip.text.x/y.
        """
        from gtable_py import gtable_add_grob, gtable_add_rows, gtable_add_cols
        from grid_py import Unit as unit, text_grob, Gpar, rect_grob
        from grid_py._grob import grob_tree, GList
        from ggplot2_py.theme_elements import calc_element as _calc_el

        # R always has a complete theme at this point.  If Python's theme
        # is None, fall back to theme_grey() to surface real bugs rather
        # than None-attribute errors.
        if theme is None:
            from ggplot2_py.theme_defaults import theme_grey
            theme = theme_grey()

        meta_cols = {"PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y", "COORD"}
        facet_vars = [c for c in layout.columns if c not in meta_cols]
        if not facet_vars:
            return gt

        col_vars = _resolve_facet_vars(params.get("cols"))
        row_vars = _resolve_facet_vars(params.get("rows"))
        wrap_vars = _resolve_facet_vars(params.get("facets"))

        # Resolve labeller function
        from ggplot2_py.labeller import as_labeller, label_value
        labeller_spec = params.get("labeller", "label_value")
        try:
            labeller_fn = as_labeller(labeller_spec)
        except (ValueError, TypeError):
            labeller_fn = label_value

        # Resolve strip theme elements via calc_element (proper inheritance).
        # R always has a complete theme with strip elements defined.
        # If calc_element returns None, the element tree is incomplete
        # — reset it and retry with a guaranteed-complete theme.
        from ggplot2_py.theme_elements import ElementBlank as _EB
        strip_txt_x_el = _calc_el("strip.text.x", theme)
        if strip_txt_x_el is None:
            from ggplot2_py.theme_elements import reset_theme_settings
            reset_theme_settings()
            from ggplot2_py.theme_defaults import theme_grey
            theme = theme_grey()
            strip_txt_x_el = _calc_el("strip.text.x", theme)
        strip_bg_x_el = _calc_el("strip.background.x", theme)
        strip_txt_y_el = _calc_el("strip.text.y", theme)
        strip_bg_y_el = _calc_el("strip.background.y", theme)

        def _props(el, attrs):
            """Extract attrs from an element, returning None for ElementBlank.

            R: when a strip element is ``element_blank()``, the
            corresponding grob is simply a ``zeroGrob()``.  We surface
            this by mapping each attr to ``None``.
            """
            if el is None or isinstance(el, _EB):
                return {k: None for k in attrs}
            return {k: getattr(el, k, None) for k in attrs}

        strip_txt_x = _props(strip_txt_x_el, ["colour", "size", "angle"])
        strip_bg_x  = _props(strip_bg_x_el,  ["fill", "colour"])
        strip_txt_y = _props(strip_txt_y_el, ["colour", "size", "angle"])
        strip_bg_y  = _props(strip_bg_y_el,  ["fill", "colour"])

        _txt_blank_x = isinstance(strip_txt_x_el, _EB)
        _bg_blank_x  = isinstance(strip_bg_x_el,  _EB)
        _txt_blank_y = isinstance(strip_txt_y_el, _EB)
        _bg_blank_y  = isinstance(strip_bg_y_el,  _EB)

        def _make_strip(label_text, bg_el, txt_el, rot, name,
                         bg_blank=False, txt_blank=False):
            """Compose a strip: optional rect bg + optional text.

            R: ElementBlank → zeroGrob, omitted from the output.
            """
            from grid_py import null_grob as _null
            bg = (
                _null()
                if bg_blank or bg_el.get("fill") is None and bg_el.get("colour") is None
                else rect_grob(
                    x=0.5, y=0.5, width=1, height=1,
                    gp=Gpar(fill=bg_el.get("fill"), col=bg_el.get("colour")),
                    name=f"strip.bg.{name}",
                )
            )
            txt = (
                _null()
                if txt_blank or txt_el.get("size") is None
                else text_grob(
                    label=label_text, x=0.5, y=0.5, rot=rot, just="centre",
                    gp=Gpar(fontsize=float(txt_el["size"]),
                            col=txt_el.get("colour")),
                    name=f"strip.text.{name}",
                )
            )
            return grob_tree(bg, txt, name=f"strip-{name}")

        def _get_strip_text(vars_list, row_info):
            """Get formatted strip label text using the labeller."""
            lab_dict = {v: [str(row_info.get(v, ""))] for v in vars_list}
            result = labeller_fn(lab_dict)
            return result[0] if result else ""

        # Helper: measure strip text height/width in cm
        # R: assemble_strips → max_height(grobs) / max_width(grobs)
        from grid_py._size import calc_string_metric

        def _strip_height_cm(labels, txt_el):
            """Max height of strip labels (R: max_height(strip_grobs))."""
            fs = float(txt_el.get("size") or 8)
            max_h = 0.0
            for lbl in labels:
                m = calc_string_metric(str(lbl), Gpar(fontsize=fs))
                max_h = max(max_h, (m["ascent"] + m["descent"]) * 2.54)
            # Add small padding for strip background
            return max(max_h + 0.1, 0.2)

        def _strip_width_cm(labels, txt_el):
            """Max width of strip labels (R: max_width(strip_grobs))."""
            fs = float(txt_el.get("size") or 8)
            max_w = 0.0
            for lbl in labels:
                m = calc_string_metric(str(lbl), Gpar(fontsize=fs))
                max_w = max(max_w, (m["ascent"] + m["descent"]) * 2.54)
            return max(max_w + 0.1, 0.2)

        # --- facet_wrap ---
        if wrap_vars and not col_vars and not row_vars:
            # Collect all wrap labels to measure max height
            all_wrap_labels = []
            for _, row_info in layout.iterrows():
                all_wrap_labels.append(_get_strip_text(wrap_vars, row_info))
            strip_h = _strip_height_cm(all_wrap_labels, strip_txt_x)

            for r in range(nrow, 0, -1):
                gt = gtable_add_rows(gt, unit([strip_h], "cm"), pos=r - 1)
                panels_in_row = layout[layout["ROW"] == r]
                for _, row_info in panels_in_row.iterrows():
                    c = int(row_info["COL"])
                    label_text = _get_strip_text(wrap_vars, row_info)
                    strip = _make_strip(label_text, strip_bg_x, strip_txt_x, 0, f"w-{r}-{c}",
                                         bg_blank=_bg_blank_x, txt_blank=_txt_blank_x)
                    gt = gtable_add_grob(gt, strip, t=r, l=c + col_offset,
                                         clip="off", name=f"strip-w-{r}-{c}")
            return gt

        # --- Top strip (col vars) ---
        if col_vars:
            # Measure col strip labels
            col_labels = []
            for c in range(1, ncol + 1):
                panel_row = layout[layout["COL"] == c].iloc[0]
                col_labels.append(_get_strip_text(col_vars, panel_row))
            strip_h = _strip_height_cm(col_labels, strip_txt_x) if not _txt_blank_x else 0.2

            gt = gtable_add_rows(gt, unit([strip_h], "cm"), pos=0)
            for c in range(1, ncol + 1):
                panel_row = layout[layout["COL"] == c].iloc[0]
                label_text = _get_strip_text(col_vars, panel_row)
                strip = _make_strip(label_text, strip_bg_x, strip_txt_x, 0, f"t-{c}",
                                     bg_blank=_bg_blank_x, txt_blank=_txt_blank_x)
                gt = gtable_add_grob(gt, strip, t=1, l=c + col_offset,
                                     clip="off", name=f"strip-t-{c}")

        # --- Right strip (row vars) ---
        if row_vars:
            # Measure row strip labels (rotated text — width = text height)
            row_labels = []
            for r in range(1, nrow + 1):
                panel_row = layout[layout["ROW"] == r].iloc[0]
                row_labels.append(_get_strip_text(row_vars, panel_row))
            strip_w = _strip_width_cm(row_labels, strip_txt_y) if not _txt_blank_y else 0.2

            gt = gtable_add_cols(gt, unit([strip_w], "cm"), pos=-1)
            ncol_now = len(gt._widths)
            row_offset = 1 if col_vars else 0
            for r in range(1, nrow + 1):
                panel_row = layout[layout["ROW"] == r].iloc[0]
                label_text = _get_strip_text(row_vars, panel_row)
                rot = float(strip_txt_y.get("angle") or 0)
                strip = _make_strip(label_text, strip_bg_y, strip_txt_y, rot, f"r-{r}",
                                     bg_blank=_bg_blank_y, txt_blank=_txt_blank_y)
                gt = gtable_add_grob(gt, strip, t=r + row_offset, l=ncol_now,
                                     clip="off", name=f"strip-r-{r}")

        return gt

    def draw_labels(
        self,
        panels: Any,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        ranges: list,
        coord: Any,
        data: Any,
        theme: Any,
        labels: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Any:
        """Add axis title labels (xlab/ylab) to the panel table.

        Mirrors R's ``Facet$draw_labels``: adds a bottom row for the
        x-axis title and a left column for the y-axis title.

        Parameters
        ----------
        panels : gtable
        labels : dict
            Rendered label grobs keyed by ``"x"`` / ``"y"``, each a
            two-element list ``[primary, secondary]``.
        """
        from gtable_py import gtable_add_grob, gtable_add_rows, gtable_add_cols
        from grid_py import Unit as unit, text_grob, Gpar, null_grob

        from grid_py import grob_height, grob_width

        gt = panels

        # --- x-axis title (bottom) ---
        x_label = None
        if "x" in labels:
            pair = labels["x"]
            if isinstance(pair, list) and len(pair) > 0:
                x_label = pair[0]  # primary

        if x_label is not None and not _is_null_grob(x_label):
            # R: gtable_add_rows(table, grobHeight(xlab), pos=-1)
            xlab_h = grob_height(x_label)
            gt = gtable_add_rows(gt, xlab_h, pos=-1)
            nrow = len(gt._heights)
            ncol = len(gt._widths)
            gt = gtable_add_grob(
                gt, x_label, t=nrow, l=1, r=ncol,
                clip="off", name="xlab",
            )

        # --- y-axis title (left) ---
        y_label = None
        if "y" in labels:
            pair = labels["y"]
            if isinstance(pair, list) and len(pair) > 0:
                y_label = pair[0]  # primary

        if y_label is not None and not _is_null_grob(y_label):
            # R: gtable_add_cols(table, grobWidth(ylab), pos=0)
            ylab_w = grob_width(y_label)
            gt = gtable_add_cols(gt, ylab_w, pos=0)
            nrow = len(gt._heights)
            gt = gtable_add_grob(
                gt, y_label, t=1, b=nrow, l=1,
                clip="off", name="ylab",
            )

        return gt

    def vars(self) -> List[str]:
        """Return the faceting variable names.

        Returns
        -------
        list of str
        """
        return []


# ---------------------------------------------------------------------------
# FacetNull
# ---------------------------------------------------------------------------

class FacetNull(Facet):
    """Single-panel facet (no faceting).

    This is the default when no faceting is specified.
    """

    shrink: bool = True

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        return _layout_null()

    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        if is_waiver(data):
            return pd.DataFrame({"PANEL": pd.Categorical([])})
        if isinstance(data, pd.DataFrame) and len(data) == 0:
            df = data.copy()
            df["PANEL"] = pd.Categorical([])
            return df
        data = data.copy()
        data["PANEL"] = pd.Categorical([1] * len(data))
        return data

    def draw_panels(
        self,
        panels: list,
        layout: pd.DataFrame,
        x_scales: list,
        y_scales: list,
        ranges: list,
        coord: Any,
        data: Any,
        theme: Any,
        params: Dict[str, Any],
    ) -> Any:
        """Build a single-panel gtable with background, axes, and geom content.

        Delegates to the base ``Facet.draw_panels`` which handles coord
        decoration and axis rendering.
        """
        return super().draw_panels(
            panels, layout, x_scales, y_scales, ranges,
            coord, data, theme, params,
        )


# ---------------------------------------------------------------------------
# FacetGrid
# ---------------------------------------------------------------------------

class FacetGrid(Facet):
    """Grid facet: panels arranged in a row x column matrix.

    Attributes
    ----------
    shrink : bool
    params : dict
        Contains ``rows``, ``cols``, ``scales``, ``space``, ``labeller``,
        ``as_table``, ``switch``, ``drop``, ``margins``, ``free``,
        ``space_free``, ``draw_axes``, ``axis_labels``.
    """

    shrink: bool = True

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Compute a grid layout from data and parameters.

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        row_vars = _resolve_facet_vars(params.get("rows"))
        col_vars = _resolve_facet_vars(params.get("cols"))
        drop = params.get("drop", True)
        free = params.get("free", {"x": False, "y": False})

        base_rows = _combine_vars(data, row_vars, drop=drop) if row_vars else pd.DataFrame()
        base_cols = _combine_vars(data, col_vars, drop=drop) if col_vars else pd.DataFrame()

        # Cross-product
        if len(base_rows) > 0 and len(base_cols) > 0:
            base_rows["_key_"] = 1
            base_cols["_key_"] = 1
            base = base_rows.merge(base_cols, on="_key_").drop("_key_", axis=1)
        elif len(base_rows) > 0:
            base = base_rows.copy()
        elif len(base_cols) > 0:
            base = base_cols.copy()
        else:
            return _layout_null()

        if len(base) == 0:
            return _layout_null()

        base = base.drop_duplicates().reset_index(drop=True)

        # Assign PANEL
        n = len(base)
        base["PANEL"] = pd.Categorical(range(1, n + 1))

        # ROW / COL identifiers
        if row_vars and any(v in base.columns for v in row_vars):
            present_rows = [v for v in row_vars if v in base.columns]
            row_ids = base[present_rows].apply(
                lambda r: "|".join(str(v) for v in r), axis=1
            )
            base["ROW"] = pd.Categorical(row_ids).codes + 1
        else:
            base["ROW"] = 1

        if col_vars and any(v in base.columns for v in col_vars):
            present_cols = [v for v in col_vars if v in base.columns]
            col_ids = base[present_cols].apply(
                lambda r: "|".join(str(v) for v in r), axis=1
            )
            base["COL"] = pd.Categorical(col_ids).codes + 1
        else:
            base["COL"] = 1

        # Scale identifiers
        base["SCALE_X"] = base["COL"] if free.get("x", False) else 1
        base["SCALE_Y"] = base["ROW"] if free.get("y", False) else 1

        base = base.sort_values("PANEL").reset_index(drop=True)
        return base

    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        row_vars = _resolve_facet_vars(params.get("rows"))
        col_vars = _resolve_facet_vars(params.get("cols"))
        all_vars = row_vars + col_vars
        return _map_facet_data(data, layout, params, all_vars)

    def vars(self) -> List[str]:
        row_vars = _resolve_facet_vars(self.params.get("rows"))
        col_vars = _resolve_facet_vars(self.params.get("cols"))
        return row_vars + col_vars


# ---------------------------------------------------------------------------
# FacetWrap
# ---------------------------------------------------------------------------

class FacetWrap(Facet):
    """Wrap facet: 1-d ribbon of panels wrapped into 2-d.

    Attributes
    ----------
    shrink : bool
    params : dict
        Contains ``facets``, ``nrow``, ``ncol``, ``scales``, ``free``,
        ``space_free``, ``labeller``, ``strip_position``, ``dir``,
        ``drop``, ``draw_axes``, ``axis_labels``.
    """

    shrink: bool = True

    def compute_layout(
        self,
        data: List[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Compute a wrapped layout.

        Parameters
        ----------
        data : list of DataFrame
        params : dict

        Returns
        -------
        pd.DataFrame
        """
        facet_vars = _resolve_facet_vars(params.get("facets"))
        drop = params.get("drop", True)
        free = params.get("free", {"x": False, "y": False})
        nrow = params.get("nrow")
        ncol = params.get("ncol")
        dir_ = params.get("dir", "lt")

        if not facet_vars:
            return _layout_null()

        base = _combine_vars(data, facet_vars, drop=drop)
        if len(base) == 0:
            return _layout_null()

        base = base.drop_duplicates().reset_index(drop=True)
        n = len(base)
        dims = _wrap_dims(n, nrow, ncol)

        # Assign PANEL, ROW, COL
        ids = np.arange(1, n + 1)
        base["PANEL"] = pd.Categorical(ids)

        # Determine layout direction
        if len(dir_) == 2:
            row_vals, col_vals = _wrap_layout(ids, dims, dir_)
        else:
            # Fallback
            row_vals = (ids - 1) // dims[1] + 1
            col_vals = (ids - 1) % dims[1] + 1

        base["ROW"] = row_vals.astype(int)
        base["COL"] = col_vals.astype(int)

        # Scale identifiers
        base["SCALE_X"] = ids if free.get("x", False) else 1
        base["SCALE_Y"] = ids if free.get("y", False) else 1

        base = base.sort_values("PANEL").reset_index(drop=True)
        return base

    def map_data(
        self,
        data: pd.DataFrame,
        layout: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        facet_vars = _resolve_facet_vars(params.get("facets"))
        return _map_facet_data(data, layout, params, facet_vars)

    def vars(self) -> List[str]:
        return _resolve_facet_vars(self.params.get("facets"))


def _wrap_layout(
    ids: np.ndarray,
    dims: Tuple[int, int],
    dir_: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ROW and COL for wrapped layout.

    Parameters
    ----------
    ids : np.ndarray
        1-based panel IDs.
    dims : tuple of (nrow, ncol)
    dir_ : str
        Two-letter direction code.

    Returns
    -------
    tuple of (ROW, COL) arrays
    """
    nrow, ncol = dims
    ids0 = ids - 1  # 0-based

    if dir_ in ("lt", "lb"):
        row = ids0 // ncol
        col = ids0 % ncol
    elif dir_ in ("tl", "bl"):
        row = ids0 % nrow
        col = ids0 // nrow
    elif dir_ in ("rt", "rb"):
        row = ids0 // ncol
        col = ncol - 1 - ids0 % ncol
    elif dir_ in ("tr", "br"):
        row = ids0 % nrow
        col = ncol - 1 - ids0 // nrow
    else:
        row = ids0 // ncol
        col = ids0 % ncol

    # Handle bottom-start directions
    if dir_ in ("lb", "bl", "rb", "br"):
        row = nrow - 1 - row

    return row + 1, col + 1


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------

def facet_null(shrink: bool = True) -> FacetNull:
    """Create a null facet (single panel).

    Parameters
    ----------
    shrink : bool

    Returns
    -------
    FacetNull
    """
    obj = FacetNull()
    obj.shrink = shrink
    return obj


def facet_grid(
    rows: Any = None,
    cols: Any = None,
    scales: str = "fixed",
    space: str = "fixed",
    shrink: bool = True,
    labeller: Any = "label_value",
    as_table: bool = True,
    switch: Optional[str] = None,
    drop: bool = True,
    margins: Union[bool, List[str]] = False,
    axes: str = "margins",
    axis_labels: str = "all",
) -> FacetGrid:
    """Create a grid facet.

    Parameters
    ----------
    rows, cols : str, list, or None
        Faceting variables for rows and columns.
    scales : str
        ``"fixed"``, ``"free_x"``, ``"free_y"``, or ``"free"``.
    space : str
        ``"fixed"``, ``"free_x"``, ``"free_y"``, or ``"free"``.
    shrink : bool
    labeller : callable or str
    as_table : bool
    switch : str or None
        ``"x"``, ``"y"``, ``"both"``, or None.
    drop : bool
    margins : bool or list of str
    axes : str
        ``"margins"``, ``"all_x"``, ``"all_y"``, or ``"all"``.
    axis_labels : str
        ``"margins"``, ``"all_x"``, ``"all_y"``, or ``"all"``.

    Returns
    -------
    FacetGrid
    """
    free = {
        "x": scales in ("free_x", "free"),
        "y": scales in ("free_y", "free"),
    }
    space_free = {
        "x": space in ("free_x", "free"),
        "y": space in ("free_y", "free"),
    }
    draw_axes_ = {
        "x": axes in ("all_x", "all"),
        "y": axes in ("all_y", "all"),
    }
    axis_labels_ = {
        "x": not draw_axes_["x"] or axis_labels in ("all_x", "all"),
        "y": not draw_axes_["y"] or axis_labels in ("all_y", "all"),
    }

    obj = FacetGrid()
    obj.shrink = shrink
    obj.params = {
        "rows": rows,
        "cols": cols,
        "margins": margins,
        "free": free,
        "space_free": space_free,
        "labeller": labeller,
        "as_table": as_table,
        "switch": switch,
        "drop": drop,
        "draw_axes": draw_axes_,
        "axis_labels": axis_labels_,
    }
    return obj


def facet_wrap(
    facets: Any,
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    scales: str = "fixed",
    space: str = "fixed",
    shrink: bool = True,
    labeller: Any = "label_value",
    as_table: bool = True,
    drop: bool = True,
    dir: str = "h",
    strip_position: str = "top",
    axes: str = "margins",
    axis_labels: str = "all",
) -> FacetWrap:
    """Create a wrap facet.

    Parameters
    ----------
    facets : str, list, or dict
        Faceting variables.
    nrow, ncol : int or None
    scales : str
        ``"fixed"``, ``"free_x"``, ``"free_y"``, or ``"free"``.
    space : str
    shrink : bool
    labeller : callable or str
    as_table : bool
    drop : bool
    dir : str
        Direction: ``"h"`` or ``"v"``, or a two-letter code.
    strip_position : str
        ``"top"``, ``"bottom"``, ``"left"``, or ``"right"``.
    axes : str
    axis_labels : str

    Returns
    -------
    FacetWrap
    """
    free = {
        "x": scales in ("free_x", "free"),
        "y": scales in ("free_y", "free"),
    }
    space_free = {
        "x": space == "free_x",
        "y": space == "free_y",
    }
    draw_axes_ = {
        "x": free["x"] or axes in ("all_x", "all"),
        "y": free["y"] or axes in ("all_y", "all"),
    }
    axis_labels_ = {
        "x": free["x"] or not draw_axes_["x"] or axis_labels in ("all_x", "all"),
        "y": free["y"] or not draw_axes_["y"] or axis_labels in ("all_y", "all"),
    }

    # Resolve direction
    if len(dir) == 1:
        if dir == "h":
            dir = "lt" if as_table else "lb"
        elif dir == "v":
            dir = "tl" if as_table else "tr"

    if strip_position not in ("top", "bottom", "left", "right"):
        cli_abort("strip_position must be 'top', 'bottom', 'left', or 'right'.")

    obj = FacetWrap()
    obj.shrink = shrink
    obj.params = {
        "facets": facets,
        "nrow": nrow,
        "ncol": ncol,
        "free": free,
        "space_free": space_free,
        "labeller": labeller,
        "dir": dir,
        "strip_position": strip_position,
        "drop": drop,
        "draw_axes": draw_axes_,
        "axis_labels": axis_labels_,
    }
    return obj


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def is_facet(x: Any) -> bool:
    """Test whether *x* is a Facet.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
    """
    return isinstance(x, Facet)

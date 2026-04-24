"""
ggplot2_py — Python port of the R ggplot2 package.

A grammar of graphics implementation for Python, providing a layered
approach to creating statistical visualizations.
"""

from __future__ import annotations

__version__ = "4.0.2.9000.post1"
__r_commit__ = "c02c05a"

# ---------------------------------------------------------------------------
# Core infrastructure
# ---------------------------------------------------------------------------
from ggplot2_py._compat import (
    Waiver,
    waiver,
    is_waiver,
)
from ggplot2_py.ggproto import (
    GGProto,
    ggproto,
    ggproto_parent,
    is_ggproto,
)

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------
from ggplot2_py.aes import (
    aes,
    after_stat,
    after_scale,
    stage,
    vars,
    is_mapping,
    standardise_aes_names,
    Mapping,
    eval_aes_value,
)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
from ggplot2_py import datasets

# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------
from ggplot2_py.layer import Layer, layer, is_layer

# ---------------------------------------------------------------------------
# Plot core
# ---------------------------------------------------------------------------
from ggplot2_py.plot import (
    GGPlot,
    ggplot,
    is_ggplot,
    ggplot_build,
    ggplot_gtable,
    ggplotGrob,
    ggplot_add,
    add_gg,
    get_last_plot,
    set_last_plot,
    last_plot,
    get_alt_text,
    update_ggplot,
    update_labels,
    by_layer,
    BuildStage,
    ggplot_defaults,
    get_layer_data,
    get_layer_grob,
    get_panel_scales,
    get_guide_data,
    get_strip_labels,
    get_labs,
    layer_data,
    layer_grob,
    layer_scales,
    summarise_plot,
    summarise_coord,
    summarise_layers,
    summarise_layout,
    find_panel,
    panel_rows,
    panel_cols,
    print_plot,
)

# ---------------------------------------------------------------------------
# Protocols (structural typing contracts — Python-exclusive)
# ---------------------------------------------------------------------------
from ggplot2_py.protocols import (
    GeomProtocol,
    StatProtocol,
    ScaleProtocol,
    CoordProtocol,
    FacetProtocol,
    PositionProtocol,
)

# ---------------------------------------------------------------------------
# Geoms
# ---------------------------------------------------------------------------
from ggplot2_py.geom import (
    Geom,
    GeomPoint,
    GeomPath,
    GeomLine,
    GeomStep,
    GeomBar,
    GeomCol,
    GeomRect,
    GeomTile,
    GeomRaster,
    GeomText,
    GeomLabel,
    GeomBoxplot,
    GeomViolin,
    GeomDotplot,
    GeomRibbon,
    GeomArea,
    GeomSmooth,
    GeomPolygon,
    GeomErrorbar,
    GeomErrorbarh,
    GeomCrossbar,
    GeomLinerange,
    GeomPointrange,
    GeomSegment,
    GeomCurve,
    GeomSpoke,
    GeomDensity,
    GeomDensity2d,
    GeomDensity2dFilled,
    GeomContour,
    GeomContourFilled,
    GeomHex,
    GeomBin2d,
    GeomAbline,
    GeomHline,
    GeomVline,
    GeomRug,
    GeomBlank,
    GeomFunction,
    GeomFreqpoly,
    GeomHistogram,
    GeomCount,
    GeomMap,
    GeomQuantile,
    GeomJitter,
    GeomSf,
    geom_point,
    geom_path,
    geom_line,
    geom_step,
    geom_bar,
    geom_col,
    geom_rect,
    geom_tile,
    geom_raster,
    geom_text,
    geom_label,
    geom_boxplot,
    geom_violin,
    geom_dotplot,
    geom_ribbon,
    geom_area,
    geom_smooth,
    geom_polygon,
    geom_errorbar,
    geom_errorbarh,
    geom_crossbar,
    geom_linerange,
    geom_pointrange,
    geom_segment,
    geom_curve,
    geom_spoke,
    geom_density,
    geom_density2d,
    geom_density2d_filled,
    geom_density_2d,
    geom_density_2d_filled,
    geom_contour,
    geom_contour_filled,
    geom_hex,
    geom_bin2d,
    geom_bin_2d,
    geom_abline,
    geom_hline,
    geom_vline,
    geom_rug,
    geom_blank,
    geom_function,
    geom_histogram,
    geom_freqpoly,
    geom_count,
    geom_jitter,
    geom_map,
    geom_quantile,
    geom_sf,
    geom_sf_label,
    geom_sf_text,
    geom_qq,
    geom_qq_line,
    is_geom,
)

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
from ggplot2_py.stat import (
    Stat,
    StatIdentity,
    StatBin,
    StatCount,
    StatDensity,
    StatSmooth,
    StatBoxplot,
    StatSummary,
    StatSummaryBin,
    StatSummary2d,
    StatSummaryHex,
    StatFunction,
    StatEcdf,
    StatQq,
    StatQqLine,
    StatBin2d,
    StatBinhex,
    StatContour,
    StatContourFilled,
    StatDensity2d,
    StatDensity2dFilled,
    StatEllipse,
    StatUnique,
    StatSum,
    StatYdensity,
    StatBindot,
    StatAlign,
    StatConnect,
    StatManual,
    StatQuantile,
    stat_identity,
    stat_bin,
    stat_count,
    stat_density,
    stat_smooth,
    stat_boxplot,
    stat_summary,
    stat_summary_bin,
    stat_summary2d,
    stat_summary_2d,
    stat_summary_hex,
    stat_function,
    stat_ecdf,
    stat_qq,
    stat_qq_line,
    stat_bin2d,
    stat_bin_2d,
    stat_bin_hex,
    stat_binhex,
    stat_contour,
    stat_contour_filled,
    stat_density2d,
    stat_density2d_filled,
    stat_density_2d,
    stat_density_2d_filled,
    stat_ellipse,
    stat_unique,
    stat_sum,
    stat_ydensity,
    stat_align,
    stat_connect,
    stat_manual,
    stat_quantile,
    stat_spoke,
    is_stat,
    mean_se,
    mean_cl_boot,
    mean_cl_normal,
    mean_sdl,
    median_hilow,
)

# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------
from ggplot2_py.scale import (
    Scale,
    ScaleContinuous,
    ScaleDiscrete,
    ScaleBinned,
    ScaleContinuousPosition,
    ScaleDiscretePosition,
    ScaleBinnedPosition,
    ScaleContinuousIdentity,
    ScaleDiscreteIdentity,
    ScaleContinuousDate,
    ScaleContinuousDatetime,
    ScalesList,
    AxisSecondary,
    continuous_scale,
    discrete_scale,
    binned_scale,
    sec_axis,
    dup_axis,
    expansion,
    expand_scale,
    find_scale,
    scale_type,
    is_scale,
)
from ggplot2_py.scales import (
    scale_x_continuous,
    scale_y_continuous,
    scale_x_discrete,
    scale_y_discrete,
    scale_x_log10,
    scale_y_log10,
    scale_x_sqrt,
    scale_y_sqrt,
    scale_x_reverse,
    scale_y_reverse,
    scale_x_binned,
    scale_y_binned,
    scale_x_date,
    scale_y_date,
    scale_x_datetime,
    scale_y_datetime,
    scale_x_time,
    scale_y_time,
    scale_colour_continuous,
    scale_colour_discrete,
    scale_colour_gradient,
    scale_colour_gradient2,
    scale_colour_gradientn,
    scale_colour_hue,
    scale_colour_brewer,
    scale_colour_distiller,
    scale_colour_fermenter,
    scale_colour_viridis_c,
    scale_colour_viridis_d,
    scale_colour_viridis_b,
    scale_colour_grey,
    scale_colour_identity,
    scale_colour_manual,
    scale_colour_binned,
    scale_colour_steps,
    scale_colour_steps2,
    scale_colour_stepsn,
    scale_colour_date,
    scale_colour_datetime,
    scale_colour_ordinal,
    scale_fill_continuous,
    scale_fill_discrete,
    scale_fill_gradient,
    scale_fill_gradient2,
    scale_fill_gradientn,
    scale_fill_hue,
    scale_fill_brewer,
    scale_fill_distiller,
    scale_fill_fermenter,
    scale_fill_viridis_c,
    scale_fill_viridis_d,
    scale_fill_viridis_b,
    scale_fill_grey,
    scale_fill_identity,
    scale_fill_manual,
    scale_fill_binned,
    scale_fill_steps,
    scale_fill_steps2,
    scale_fill_stepsn,
    scale_fill_date,
    scale_fill_datetime,
    scale_fill_ordinal,
    scale_color_continuous,
    scale_color_discrete,
    scale_color_gradient,
    scale_color_gradient2,
    scale_color_gradientn,
    scale_color_hue,
    scale_color_brewer,
    scale_color_distiller,
    scale_color_fermenter,
    scale_color_viridis_c,
    scale_color_viridis_d,
    scale_color_viridis_b,
    scale_color_grey,
    scale_color_identity,
    scale_color_manual,
    scale_color_binned,
    scale_color_steps,
    scale_color_steps2,
    scale_color_stepsn,
    scale_color_date,
    scale_color_datetime,
    scale_color_ordinal,
    scale_alpha,
    scale_alpha_continuous,
    scale_alpha_discrete,
    scale_alpha_binned,
    scale_alpha_identity,
    scale_alpha_manual,
    scale_alpha_ordinal,
    scale_alpha_date,
    scale_alpha_datetime,
    scale_size,
    scale_size_continuous,
    scale_size_discrete,
    scale_size_binned,
    scale_size_area,
    scale_size_binned_area,
    scale_size_identity,
    scale_size_manual,
    scale_size_ordinal,
    scale_size_date,
    scale_size_datetime,
    scale_radius,
    scale_shape,
    scale_shape_discrete,
    scale_shape_binned,
    scale_shape_identity,
    scale_shape_manual,
    scale_shape_ordinal,
    scale_linetype,
    scale_linetype_discrete,
    scale_linetype_binned,
    scale_linetype_identity,
    scale_linetype_manual,
    scale_linewidth,
    scale_linewidth_continuous,
    scale_linewidth_discrete,
    scale_linewidth_binned,
    scale_linewidth_identity,
    scale_linewidth_manual,
    scale_linewidth_ordinal,
    scale_linewidth_date,
    scale_linewidth_datetime,
    scale_stroke,
    scale_stroke_continuous,
    scale_stroke_discrete,
    scale_stroke_binned,
    scale_stroke_identity,
    scale_stroke_manual,
    scale_stroke_ordinal,
    scale_continuous_identity,
    scale_discrete_identity,
    scale_discrete_manual,
)

# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
from ggplot2_py.coord import (
    Coord,
    CoordCartesian,
    CoordFixed,
    CoordFlip,
    CoordPolar,
    CoordQuickmap,
    CoordRadial,
    CoordSf,
    CoordTrans,
    CoordTransform,
    coord_cartesian,
    coord_equal,
    coord_fixed,
    coord_flip,
    coord_polar,
    coord_quickmap,
    coord_radial,
    coord_sf,
    coord_trans,
    coord_transform,
    coord_munch,
    is_coord,
    sf_transform_xy,
)

# ---------------------------------------------------------------------------
# Faceting
# ---------------------------------------------------------------------------
from ggplot2_py.facet import (
    Facet,
    FacetNull,
    FacetGrid,
    FacetWrap,
    facet_null,
    facet_grid,
    facet_wrap,
    is_facet,
)

# ---------------------------------------------------------------------------
# Position adjustments
# ---------------------------------------------------------------------------
from ggplot2_py.position import (
    Position,
    PositionIdentity,
    PositionDodge,
    PositionDodge2,
    PositionJitter,
    PositionJitterdodge,
    PositionNudge,
    PositionStack,
    PositionFill,
    position_identity,
    position_dodge,
    position_dodge2,
    position_jitter,
    position_jitterdodge,
    position_nudge,
    position_stack,
    position_fill,
    is_position,
)

# ---------------------------------------------------------------------------
# Guides
# ---------------------------------------------------------------------------
from ggplot2_py.guide import (
    Guide,
    GuideAxis,
    GuideAxisLogticks,
    GuideAxisStack,
    GuideAxisTheta,
    GuideBins,
    GuideColourbar,
    GuideColoursteps,
    GuideCustom,
    GuideLegend,
    GuideNone,
    guide_axis,
    guide_axis_logticks,
    guide_axis_stack,
    guide_axis_theta,
    guide_bins,
    guide_colourbar,
    guide_colorbar,
    guide_old_colourbar,
    guide_old_colorbar,
    guide_coloursteps,
    guide_colorsteps,
    guide_custom,
    guide_legend,
    guide_none,
    guides,
    is_guide,
)

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
from ggplot2_py.theme import (
    theme,
    is_theme,
    complete_theme,
    theme_get,
    theme_set,
    theme_update,
    theme_replace,
    set_theme,
    get_theme,
    reset_theme_settings,
    update_theme,
    replace_theme,
)
from ggplot2_py.theme_elements import (
    Element,
    element_blank,
    element_line,
    element_rect,
    element_text,
    element_point,
    element_polygon,
    element_geom,
    element_grob,
    element_render,
    el_def,
    merge_element,
    is_theme_element,
    Margin,
    margin,
    margin_auto,
    margin_part,
    Rel,
    rel,
    calc_element,
    get_element_tree,
    register_theme_elements,
)
from ggplot2_py.theme_defaults import (
    theme_grey,
    theme_gray,
    theme_bw,
    theme_linedraw,
    theme_light,
    theme_dark,
    theme_minimal,
    theme_classic,
    theme_void,
    theme_test,
)

# ---------------------------------------------------------------------------
# Labels, limits, annotations
# ---------------------------------------------------------------------------
from ggplot2_py.labels import labs, xlab, ylab, ggtitle
from ggplot2_py.limits import lims, xlim, ylim, expand_limits
from ggplot2_py.annotation import (
    annotate,
    annotation_custom,
    annotation_raster,
    annotation_logticks,
)

# ---------------------------------------------------------------------------
# Draw keys
# ---------------------------------------------------------------------------
from ggplot2_py.draw_key import (
    draw_key_point,
    draw_key_path,
    draw_key_rect,
    draw_key_polygon,
    draw_key_blank,
    draw_key_boxplot,
    draw_key_crossbar,
    draw_key_dotplot,
    draw_key_label,
    draw_key_linerange,
    draw_key_pointrange,
    draw_key_smooth,
    draw_key_text,
    draw_key_abline,
    draw_key_vline,
    draw_key_timeseries,
    draw_key_vpath,
)

# ---------------------------------------------------------------------------
# Save, fortify, qplot
# ---------------------------------------------------------------------------
from ggplot2_py.save import ggsave, check_device
from ggplot2_py.fortify import (
    fortify,
    fortify_lm,
    fortify_dispatch,
    fortify_glht,
    fortify_confint_glht,
    fortify_summary_glht,
    fortify_cld,
)
from ggplot2_py.qplot import qplot, quickplot
from ggplot2_py.labeller import (
    label_value,
    label_both,
    label_context,
    label_parsed,
    label_bquote,
    label_wrap_gen,
    as_labeller,
)
from ggplot2_py.autoplot import autoplot, autolayer

# ---------------------------------------------------------------------------
# Utility re-exports (matching R namespace)
# ---------------------------------------------------------------------------
from ggplot2_py._utils import resolution, remove_missing

# grid re-exports (matching R: importFrom(grid, unit, arrow))
from grid_py import Unit as unit, arrow
from scales import alpha

# Constants
PT = 72.27 / 25.4  # points per mm
STROKE = 96 / 25.4  # pixels per mm

# Derived aliases
derive = None  # placeholder for axis derivation sentinel
flip_data = None  # placeholder
flipped_names = None  # placeholder
has_flipped_aes = None  # placeholder

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
__all__ = [
    # Version
    "__version__",
    # Core
    "GGProto", "ggproto", "ggproto_parent", "is_ggproto",
    "Waiver", "waiver", "is_waiver",
    # Aesthetics
    "aes", "after_stat", "after_scale", "stage", "vars",
    "is_mapping", "standardise_aes_names",
    # Datasets
    "datasets",
    # Layer
    "Layer", "layer", "is_layer",
    # Plot
    "ggplot", "is_ggplot", "ggplot_build", "ggplot_gtable", "ggplotGrob",
    "ggplot_add", "add_gg", "get_last_plot", "set_last_plot", "last_plot",
    "print_plot", "get_alt_text", "update_ggplot",
    # Introspection
    "get_layer_data", "get_layer_grob", "get_panel_scales",
    "get_guide_data", "get_strip_labels", "get_labs",
    "layer_data", "layer_grob", "layer_scales",
    "summarise_plot", "summarise_coord", "summarise_layers", "summarise_layout",
    "find_panel", "panel_rows", "panel_cols",
    # Geom classes
    "Geom", "GeomPoint", "GeomPath", "GeomLine", "GeomStep",
    "GeomBar", "GeomCol", "GeomRect", "GeomTile", "GeomRaster",
    "GeomText", "GeomLabel", "GeomBoxplot", "GeomViolin", "GeomDotplot",
    "GeomRibbon", "GeomArea", "GeomSmooth", "GeomPolygon",
    "GeomErrorbar", "GeomErrorbarh", "GeomCrossbar", "GeomLinerange", "GeomPointrange",
    "GeomSegment", "GeomCurve", "GeomSpoke",
    "GeomDensity", "GeomDensity2d", "GeomDensity2dFilled",
    "GeomContour", "GeomContourFilled",
    "GeomHex", "GeomBin2d", "GeomAbline", "GeomHline", "GeomVline",
    "GeomRug", "GeomBlank", "GeomFunction", "GeomSf",
    # Geom constructors
    "geom_point", "geom_path", "geom_line", "geom_step",
    "geom_bar", "geom_col", "geom_rect", "geom_tile", "geom_raster",
    "geom_text", "geom_label", "geom_boxplot", "geom_violin", "geom_dotplot",
    "geom_ribbon", "geom_area", "geom_smooth", "geom_polygon",
    "geom_errorbar", "geom_errorbarh", "geom_crossbar", "geom_linerange", "geom_pointrange",
    "geom_segment", "geom_curve", "geom_spoke",
    "geom_density", "geom_density2d", "geom_density2d_filled",
    "geom_density_2d", "geom_density_2d_filled",
    "geom_contour", "geom_contour_filled",
    "geom_hex", "geom_bin2d", "geom_bin_2d",
    "geom_abline", "geom_hline", "geom_vline",
    "geom_rug", "geom_blank", "geom_function",
    "geom_histogram", "geom_freqpoly", "geom_count", "geom_jitter",
    "geom_map", "geom_quantile",
    "geom_sf", "geom_sf_label", "geom_sf_text",
    "geom_qq", "geom_qq_line",
    "is_geom",
    # Stat classes
    "Stat", "StatIdentity", "StatBin", "StatCount", "StatDensity",
    "StatSmooth", "StatBoxplot", "StatSummary", "StatSummaryBin",
    "StatSummary2d", "StatSummaryHex", "StatFunction", "StatEcdf",
    "StatQq", "StatQqLine", "StatBin2d", "StatBinhex",
    "StatContour", "StatContourFilled",
    "StatDensity2d", "StatDensity2dFilled",
    "StatEllipse", "StatUnique", "StatSum", "StatYdensity",
    "StatBindot", "StatAlign", "StatConnect", "StatManual", "StatQuantile",
    # Stat constructors
    "stat_identity", "stat_bin", "stat_count", "stat_density",
    "stat_smooth", "stat_boxplot", "stat_summary", "stat_summary_bin",
    "stat_summary2d", "stat_summary_2d", "stat_summary_hex",
    "stat_function", "stat_ecdf",
    "stat_qq", "stat_qq_line",
    "stat_bin2d", "stat_bin_2d", "stat_bin_hex", "stat_binhex",
    "stat_contour", "stat_contour_filled",
    "stat_density2d", "stat_density2d_filled",
    "stat_density_2d", "stat_density_2d_filled",
    "stat_ellipse", "stat_unique", "stat_sum", "stat_ydensity",
    "stat_align", "stat_connect", "stat_manual", "stat_quantile", "stat_spoke",
    "is_stat",
    "mean_se", "mean_cl_boot", "mean_cl_normal", "mean_sdl", "median_hilow",
    # Scale classes
    "Scale", "ScaleContinuous", "ScaleDiscrete", "ScaleBinned",
    "ScaleContinuousPosition", "ScaleDiscretePosition", "ScaleBinnedPosition",
    "ScaleContinuousIdentity", "ScaleDiscreteIdentity",
    "ScaleContinuousDate", "ScaleContinuousDatetime",
    "ScalesList", "AxisSecondary",
    "continuous_scale", "discrete_scale", "binned_scale",
    "sec_axis", "dup_axis", "expansion", "expand_scale",
    "find_scale", "scale_type", "is_scale",
    # Scale constructors (selected)
    "scale_x_continuous", "scale_y_continuous",
    "scale_x_discrete", "scale_y_discrete",
    "scale_x_log10", "scale_y_log10",
    "scale_x_sqrt", "scale_y_sqrt",
    "scale_x_reverse", "scale_y_reverse",
    "scale_x_binned", "scale_y_binned",
    "scale_x_date", "scale_y_date",
    "scale_x_datetime", "scale_y_datetime",
    "scale_colour_continuous", "scale_colour_discrete",
    "scale_colour_gradient", "scale_colour_gradient2", "scale_colour_gradientn",
    "scale_colour_hue", "scale_colour_brewer",
    "scale_colour_viridis_c", "scale_colour_viridis_d",
    "scale_colour_grey", "scale_colour_identity", "scale_colour_manual",
    "scale_fill_continuous", "scale_fill_discrete",
    "scale_fill_gradient", "scale_fill_gradient2", "scale_fill_gradientn",
    "scale_fill_hue", "scale_fill_brewer",
    "scale_fill_viridis_c", "scale_fill_viridis_d",
    "scale_fill_grey", "scale_fill_identity", "scale_fill_manual",
    "scale_color_continuous", "scale_color_discrete",
    "scale_color_gradient", "scale_color_gradient2", "scale_color_gradientn",
    "scale_color_hue", "scale_color_brewer",
    "scale_color_viridis_c", "scale_color_viridis_d",
    "scale_color_grey", "scale_color_identity", "scale_color_manual",
    "scale_alpha", "scale_alpha_continuous", "scale_alpha_discrete",
    "scale_size", "scale_size_continuous", "scale_size_area",
    "scale_shape", "scale_shape_discrete",
    "scale_linetype", "scale_linetype_discrete",
    "scale_linewidth", "scale_linewidth_continuous",
    "scale_stroke", "scale_stroke_continuous",
    # Coords
    "Coord", "CoordCartesian", "CoordFixed", "CoordFlip",
    "CoordPolar", "CoordQuickmap", "CoordRadial", "CoordTrans", "CoordTransform",
    "coord_cartesian", "coord_equal", "coord_fixed", "coord_flip",
    "coord_polar", "coord_quickmap", "coord_radial", "coord_sf", "coord_trans", "coord_transform",
    "CoordSf",
    "coord_munch", "is_coord",
    # Facets
    "Facet", "FacetNull", "FacetGrid", "FacetWrap",
    "facet_null", "facet_grid", "facet_wrap", "is_facet",
    # Positions
    "Position", "PositionIdentity", "PositionDodge", "PositionDodge2",
    "PositionJitter", "PositionJitterdodge", "PositionNudge",
    "PositionStack", "PositionFill",
    "position_identity", "position_dodge", "position_dodge2",
    "position_jitter", "position_jitterdodge", "position_nudge",
    "position_stack", "position_fill", "is_position",
    # Guides
    "Guide", "GuideAxis", "GuideAxisLogticks", "GuideAxisStack",
    "GuideAxisTheta", "GuideBins", "GuideColourbar", "GuideColoursteps",
    "GuideCustom", "GuideLegend", "GuideNone",
    "guide_axis", "guide_legend", "guide_colourbar", "guide_colorbar",
    "guide_old_colourbar", "guide_old_colorbar",
    "guide_coloursteps", "guide_colorsteps", "guide_bins",
    "guide_custom", "guide_none", "guides", "is_guide",
    # Themes
    "theme", "is_theme", "complete_theme",
    "theme_get", "theme_set", "theme_update", "theme_replace",
    "set_theme", "get_theme", "reset_theme_settings",
    "Element", "element_blank", "element_line", "element_rect",
    "element_text", "element_point", "element_polygon", "element_geom",
    "element_grob", "element_render", "merge_element",
    "Margin", "margin", "Rel", "rel",
    "calc_element", "get_element_tree", "register_theme_elements",
    "theme_grey", "theme_gray", "theme_bw", "theme_linedraw",
    "theme_light", "theme_dark", "theme_minimal", "theme_classic",
    "theme_void", "theme_test",
    # Labels, limits
    "labs", "xlab", "ylab", "ggtitle",
    "lims", "xlim", "ylim", "expand_limits",
    # Annotations
    "annotate", "annotation_custom", "annotation_raster", "annotation_logticks",
    # Draw keys
    "draw_key_point", "draw_key_path", "draw_key_rect", "draw_key_polygon",
    "draw_key_blank", "draw_key_boxplot", "draw_key_crossbar",
    "draw_key_dotplot", "draw_key_label", "draw_key_linerange",
    "draw_key_pointrange", "draw_key_smooth", "draw_key_text",
    "draw_key_abline", "draw_key_vline", "draw_key_timeseries", "draw_key_vpath",
    # Save, fortify, qplot
    "ggsave", "check_device",
    "fortify", "fortify_lm", "fortify_dispatch",
    "fortify_glht", "fortify_confint_glht", "fortify_summary_glht", "fortify_cld",
    "qplot", "quickplot",
    # Labellers
    "label_value", "label_both", "label_context", "label_parsed",
    "label_bquote", "label_wrap_gen", "as_labeller",
    # autoplot / autolayer generics
    "autoplot", "autolayer",
    # Utilities
    "resolution", "remove_missing",
    "unit", "arrow", "alpha",
    "PT", "STROKE",
    # Plugin discovery
    "discover_extensions", "list_extensions",
]

# ---------------------------------------------------------------------------
# Initialize the global theme state — mirrors R's on_load block in
# ``theme-elements.R:727-733`` which calls ``reset_theme_settings()`` to
# establish ``theme_default = theme_grey()`` and ``theme_current`` before
# any plot is built.  Without this, ``complete_theme`` cannot fall back
# to a valid default, and all downstream ``calc_element`` resolutions
# return ``None``.
# ---------------------------------------------------------------------------
reset_theme_settings()

# ---------------------------------------------------------------------------
# Entry-point plugin discovery — scan installed packages for extensions.
# This runs once at import time. Extensions that fail to load emit a
# warning but do not block import.
# ---------------------------------------------------------------------------
from ggplot2_py._plugins import discover_extensions, list_extensions

discover_extensions()

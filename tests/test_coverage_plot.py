"""Tests to improve coverage for plot.py."""

import pytest
import pandas as pd
import numpy as np

from ggplot2_py.plot import (
    ggplot,
    GGPlot,
    BuiltGGPlot,
    is_ggplot,
    ggplot_build,
    ggplot_add,
    update_ggplot,
    add_gg,
    get_last_plot,
    set_last_plot,
    last_plot,
    get_alt_text,
    get_layer_data,
    get_panel_scales,
    get_guide_data,
    get_strip_labels,
    get_labs,
    summarise_plot,
    summarise_coord,
    summarise_layers,
    summarise_layout,
    find_panel,
    panel_rows,
    panel_cols,
    _setup_plot_labels,
)
from ggplot2_py.aes import aes, Mapping
from ggplot2_py.geom import geom_point
from ggplot2_py.labels import labs, Labels
from ggplot2_py.theme import Theme, theme
from ggplot2_py._compat import waiver


# =====================================================================
# GGPlot constructor tests
# =====================================================================

class TestGGPlotConstructor:
    def test_basic_creation(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p = ggplot(df, aes(x="x", y="y"))
        assert isinstance(p, GGPlot)
        assert isinstance(p.data, pd.DataFrame)
        assert "x" in p.mapping

    def test_no_data(self):
        p = ggplot()
        assert isinstance(p, GGPlot)

    def test_dict_data(self):
        p = ggplot({"x": [1, 2], "y": [3, 4]})
        assert isinstance(p.data, pd.DataFrame)

    def test_none_data(self):
        p = ggplot(None)
        assert isinstance(p, GGPlot)

    def test_no_mapping(self):
        df = pd.DataFrame({"x": [1, 2]})
        p = ggplot(df)
        assert isinstance(p.mapping, Mapping)

    def test_callable_data_raises(self):
        with pytest.raises(TypeError, match="cannot be a function"):
            ggplot(lambda: None)

    def test_mapping_swap_heuristic(self):
        df = pd.DataFrame({"x": [1]})
        m = aes(x="x")
        # Swap: mapping=df, data=mapping -- data and mapping get swapped
        p = ggplot(data=m, mapping=df)
        assert isinstance(p.data, pd.DataFrame)

    def test_sets_defaults(self):
        p = ggplot()
        assert p.coordinates is not None
        assert p.facet is not None

    def test_labels_from_mapping(self):
        # R parity: ``plot$labels`` stores ONLY user-set labels at
        # construction time; aes-derived defaults are filled at render
        # time inside ``_setup_plot_labels``. Build first, then assert.
        from ggplot2_py import ggplot_build
        p = ggplot(pd.DataFrame({"x": [1], "y": [2]}), aes(x="x", y="y"))
        assert "x" not in p.labels  # not yet auto-derived
        built = ggplot_build(p + geom_point())
        assert "x" in built.plot.labels  # filled by _setup_plot_labels

    def test_labels_default_from_geom(self):
        # Sanity: needs a layer for _setup_plot_labels to do anything
        from ggplot2_py import ggplot_build
        p = ggplot(pd.DataFrame({"x": [1], "y": [2]}), aes(x="x", y="y")) + geom_point()
        built = ggplot_build(p)
        assert built.plot.labels.get("x") == "x"
        assert built.plot.labels.get("y") == "y"


class TestGGPlotClass:
    def test_repr_with_data(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        p = ggplot(df)
        r = repr(p)
        assert "GGPlot" in r
        assert "3x1" in r

    def test_repr_no_data(self):
        p = ggplot()
        r = repr(p)
        assert "GGPlot" in r

    def test_clone(self):
        df = pd.DataFrame({"x": [1]})
        p = ggplot(df, aes(x="x"))
        p2 = p._clone()
        assert p2 is not p
        assert p2.data is p.data  # shallow copy

    def test_add_none(self):
        p = ggplot()
        p2 = p + None
        assert isinstance(p2, GGPlot)

    def test_radd_none(self):
        p = ggplot()
        p2 = None + p
        assert isinstance(p2, GGPlot)

    def test_radd_zero(self):
        p = ggplot()
        p2 = 0 + p
        assert isinstance(p2, GGPlot)

    def test_iadd(self):
        p = ggplot()
        p += labs(title="test")
        assert "title" in p.labels

    def test_meta_attribute(self):
        p = ggplot()
        p.custom_attr = "hello"
        assert p.custom_attr == "hello"

    def test_missing_attr_raises(self):
        p = ggplot()
        with pytest.raises(AttributeError):
            _ = p.nonexistent_attr

    def test_private_attr_raises(self):
        p = ggplot()
        with pytest.raises(AttributeError):
            _ = p._nonexistent

    def test_summary(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes(x="x", y="y"))
        s = p.summary()
        assert "data" in s
        assert "mapping" in s


# =====================================================================
# is_ggplot tests
# =====================================================================

class TestIsGGPlot:
    def test_positive(self):
        assert is_ggplot(ggplot())

    def test_negative(self):
        assert not is_ggplot("hello")
        assert not is_ggplot(None)


# =====================================================================
# last_plot bookkeeping
# =====================================================================

class TestLastPlot:
    def test_set_and_get(self):
        p = ggplot()
        set_last_plot(p)
        assert get_last_plot() is p

    def test_last_plot_alias(self):
        p = ggplot()
        set_last_plot(p)
        assert last_plot() is p


# =====================================================================
# BuiltGGPlot
# =====================================================================

class TestBuiltGGPlot:
    def test_repr(self):
        built = BuiltGGPlot(data=[pd.DataFrame()], layout=None, plot=ggplot())
        assert "BuiltGGPlot" in repr(built)

    def test_already_built_returns_self(self):
        built = BuiltGGPlot(data=[pd.DataFrame()], layout=None, plot=ggplot())
        result = ggplot_build(built)
        assert result is built


# =====================================================================
# ggplot_add / update_ggplot tests
# =====================================================================

class TestGGPlotAdd:
    def test_add_none(self):
        p = ggplot()
        result = ggplot_add(None, p)
        assert result is p

    def test_add_labels(self):
        p = ggplot()
        result = ggplot_add(labs(title="Title"), p)
        assert result.labels["title"] == "Title"

    def test_add_mapping(self):
        p = ggplot()
        result = ggplot_add(aes(x="x"), p)
        assert "x" in result.mapping

    def test_add_theme(self):
        from ggplot2_py.theme import Theme
        p = ggplot()
        # Set the plot's theme to a Theme object first (default is dict)
        p.theme = Theme()
        t = theme(plot_title=None)
        result = ggplot_add(t, p)
        assert isinstance(result, GGPlot)

    def test_add_list(self):
        p = ggplot()
        result = ggplot_add([labs(x="X"), labs(y="Y")], p)
        assert result.labels.get("x") == "X"
        assert result.labels.get("y") == "Y"

    def test_add_dataframe(self):
        p = ggplot()
        df = pd.DataFrame({"a": [1]})
        result = ggplot_add(df, p)
        assert isinstance(result.data, pd.DataFrame)

    def test_add_callable_raises(self):
        p = ggplot()
        with pytest.raises(ValueError, match="forget to add parentheses"):
            ggplot_add(lambda: None, p)

    def test_add_ggproto_raises(self):
        from ggplot2_py.ggproto import GGProto
        p = ggplot()
        with pytest.raises(ValueError, match="ggproto"):
            ggplot_add(GGProto(), p)

    def test_add_unsupported_raises(self):
        p = ggplot()
        with pytest.raises(ValueError):
            ggplot_add(42, p)


# =====================================================================
# add_gg tests
# =====================================================================

class TestAddGg:
    def test_theme_plus_theme(self):
        t1 = Theme(elements={"a": 1})
        t2 = Theme(elements={"b": 2})
        result = add_gg(t1, t2)
        assert "a" in result
        assert "b" in result

    def test_ggplot_plus_labels(self):
        p = ggplot()
        result = add_gg(p, labs(title="T"))
        assert "title" in result.labels

    def test_ggproto_raises(self):
        from ggplot2_py.ggproto import GGProto
        with pytest.raises(ValueError):
            add_gg(GGProto(), labs())

    def test_other_raises(self):
        with pytest.raises(ValueError):
            add_gg(42, labs())


# =====================================================================
# get_alt_text tests
# =====================================================================

class TestGetAltText:
    def test_ggplot_no_alt(self):
        p = ggplot()
        assert get_alt_text(p) == ""

    def test_ggplot_with_alt(self):
        p = ggplot()
        p.labels["alt"] = "Alt text"
        assert get_alt_text(p) == "Alt text"

    def test_built_ggplot(self):
        p = ggplot()
        p.labels["alt"] = "Built alt"
        built = BuiltGGPlot(data=[], layout=None, plot=p)
        assert get_alt_text(built) == "Built alt"

    def test_callable_alt_in_ggplot(self):
        p = ggplot()
        p.labels["alt"] = lambda plot: "dynamic"
        assert get_alt_text(p) == ""  # not built, returns empty

    def test_callable_alt_in_built(self):
        p = ggplot()
        p.labels["alt"] = lambda plot: "dynamic alt"
        built = BuiltGGPlot(data=[], layout=None, plot=p)
        assert get_alt_text(built) == "dynamic alt"

    def test_gtable_with_alt(self):
        class FakeTable:
            _alt_label = "table alt"
        assert get_alt_text(FakeTable()) == "table alt"

    def test_gtable_without_alt(self):
        class FakeTable:
            pass
        assert get_alt_text(FakeTable()) == ""


# =====================================================================
# Introspection helpers
# =====================================================================

class TestGetGuideData:
    def test_returns_none(self):
        assert get_guide_data() is None


class TestGetStripLabels:
    def test_returns_none(self):
        assert get_strip_labels() is None


class TestGetLabs:
    def test_returns_labels(self):
        p = ggplot()
        p.labels["x"] = "X axis"
        set_last_plot(p)
        result = get_labs()
        assert isinstance(result, Labels)


# =====================================================================
# Summarise functions
# =====================================================================

class TestSummarise:
    def test_summarise_plot(self):
        df = pd.DataFrame({"x": [1, 2]})
        p = ggplot(df, aes(x="x"))
        result = summarise_plot(p)
        assert result["n_layers"] == 0
        assert "x" in result["mapping"]

    def test_summarise_coord(self):
        p = ggplot()
        result = summarise_coord(p)
        assert "class" in result

    def test_summarise_coord_none(self):
        p = ggplot()
        p.coordinates = None
        result = summarise_coord(p)
        assert result == {}

    def test_summarise_layers_empty(self):
        p = ggplot()
        result = summarise_layers(p)
        assert result == []

    def test_summarise_layout(self):
        p = ggplot()
        result = summarise_layout(p)
        assert "class" in result

    def test_summarise_layout_no_facet(self):
        p = ggplot()
        p.facet = None
        result = summarise_layout(p)
        assert result == {}


# =====================================================================
# Panel helpers
# =====================================================================

class TestPanelHelpers:
    def test_find_panel_with_layout(self):
        layout_df = pd.DataFrame({
            "name": ["panel-1", "axis"],
            "t": [2, 1],
            "l": [2, 1],
            "b": [4, 1],
            "r": [4, 1],
        })
        class FakeTable:
            layout = layout_df
        result = find_panel(FakeTable())
        assert result["t"] == 2
        assert result["b"] == 4

    def test_find_panel_no_layout(self):
        class FakeTable:
            pass
        result = find_panel(FakeTable())
        assert result == {"t": 1, "l": 1, "b": 1, "r": 1}

    def test_find_panel_no_panel_names(self):
        layout_df = pd.DataFrame({
            "name": ["axis-t", "axis-b"],
            "t": [1, 3],
            "l": [1, 1],
            "b": [1, 3],
            "r": [1, 1],
        })
        class FakeTable:
            layout = layout_df
        result = find_panel(FakeTable())
        assert result == {"t": 1, "l": 1, "b": 1, "r": 1}

    def test_panel_rows(self):
        layout_df = pd.DataFrame({
            "name": ["panel-1"],
            "t": [2], "l": [2], "b": [4], "r": [4],
        })
        class FakeTable:
            layout = layout_df
        result = panel_rows(FakeTable())
        assert result == {"t": 2, "b": 4}

    def test_panel_cols(self):
        layout_df = pd.DataFrame({
            "name": ["panel-1"],
            "t": [2], "l": [2], "b": [4], "r": [4],
        })
        class FakeTable:
            layout = layout_df
        result = panel_cols(FakeTable())
        assert result == {"l": 2, "r": 4}


# =====================================================================
# _setup_plot_labels tests
# =====================================================================

class TestSetupPlotLabels:
    def test_merges_layer_labels(self):
        p = ggplot(pd.DataFrame({"x": [1]}), aes(x="x"))
        # Create a mock layer with mapping
        class FakeLayer:
            computed_mapping = Mapping(y="y_col")
            mapping = Mapping(y="y_col")
            stat = None
        p.layers = [FakeLayer()]
        _setup_plot_labels(p, p.layers, [pd.DataFrame()])
        assert "y" in p.labels


# =====================================================================
# Add coord/facet tests
# =====================================================================

class TestUpdateGGPlotCoordFacet:
    def test_add_coord(self):
        from ggplot2_py.coord import CoordCartesian
        p = ggplot()
        coord = CoordCartesian()
        result = ggplot_add(coord, p)
        assert result.coordinates is coord

    def test_add_coord_default_skip(self):
        from ggplot2_py.coord import CoordCartesian
        p = ggplot()
        # Set a non-default coord first
        non_default = CoordCartesian()
        non_default.default = False
        p.coordinates = non_default
        # Try adding a default coord -- should be skipped
        default_coord = CoordCartesian()
        default_coord.default = True
        result = ggplot_add(default_coord, p)
        assert result.coordinates is non_default

    def test_add_facet(self):
        from ggplot2_py.facet import FacetNull
        p = ggplot()
        facet = FacetNull()
        result = ggplot_add(facet, p)
        assert result.facet is facet

    def test_add_layer(self):
        from ggplot2_py.layer import layer
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
        lyr = layer(geom="point", stat="identity")
        result = p + lyr
        assert len(result.layers) == 1

    def test_add_scale(self):
        from ggplot2_py.scales import scale_x_continuous
        p = ggplot()
        result = p + scale_x_continuous()
        assert result.scales.n() > 0


class TestGGPlotBuildIntegration:
    def test_build_empty_plot(self):
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        # Build should add a blank layer and complete
        built = ggplot_build(p)
        assert isinstance(built, BuiltGGPlot)
        assert len(built.data) >= 1

    def test_build_with_layer(self):
        from ggplot2_py.layer import layer
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        p = p + layer(geom="point", stat="identity")
        built = ggplot_build(p)
        assert len(built.data) == 1

    def test_summary_with_layer(self):
        from ggplot2_py.layer import layer
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        p = p + layer(geom="point", stat="identity")
        s = p.summary()
        assert "geom" in s or "point" in s.lower()

    def test_summarise_layers(self):
        from ggplot2_py.layer import layer
        p = ggplot(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), aes(x="x", y="y"))
        p = p + layer(geom="point", stat="identity")
        result = summarise_layers(p)
        assert len(result) == 1
        assert "geom" in result[0]


class TestSetattr:
    def test_direct_attrs(self):
        p = ggplot()
        p.data = pd.DataFrame()
        p.mapping = aes()
        p.layers = []
        assert isinstance(p.data, pd.DataFrame)

    def test_meta_attrs(self):
        p = ggplot()
        p.custom = "test"
        assert p.custom == "test"

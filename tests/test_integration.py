"""Integration tests for ggplot2_py — full plot construction pipelines."""

import pytest
import pandas as pd
from ggplot2_py import (
    ggplot,
    aes,
    is_ggplot,
    geom_point,
    geom_bar,
    geom_boxplot,
    geom_histogram,
    geom_line,
    geom_smooth,
    scale_x_continuous,
    scale_colour_viridis_c,
    coord_flip,
    facet_wrap,
    theme_bw,
    labs,
    Layer,
)
from ggplot2_py.plot import GGPlot
from ggplot2_py.facet import FacetWrap
from ggplot2_py.theme import Theme
from ggplot2_py.coord import CoordFlip


class TestScatterPlot:
    """Test scatter plot construction."""

    def test_basic_scatter(self, mpg):
        p = ggplot(mpg, aes("displ", "hwy")) + geom_point()
        assert is_ggplot(p)
        assert len(p.layers) == 1
        assert isinstance(p.layers[0], Layer)

    def test_scatter_with_color(self, mpg):
        p = ggplot(mpg, aes("displ", "hwy", colour="class")) + geom_point()
        assert len(p.layers) == 1


class TestBarPlot:
    """Test bar plot construction."""

    def test_basic_bar(self, mpg):
        from ggplot2_py.stat import StatCount
        p = ggplot(mpg, aes("class")) + geom_bar()
        assert is_ggplot(p)
        assert len(p.layers) == 1
        assert isinstance(p.layers[0].stat, StatCount)


class TestBoxplot:
    """Test boxplot construction."""

    def test_basic_boxplot(self, mpg):
        from ggplot2_py.stat import StatBoxplot
        p = ggplot(mpg, aes("class", "hwy")) + geom_boxplot()
        assert is_ggplot(p)
        assert len(p.layers) == 1
        assert isinstance(p.layers[0].stat, StatBoxplot)


class TestHistogram:
    """Test histogram construction."""

    def test_basic_histogram(self, mpg):
        from ggplot2_py.stat import StatBin
        p = ggplot(mpg, aes("hwy")) + geom_histogram(bins=30)
        assert is_ggplot(p)
        assert len(p.layers) == 1
        assert isinstance(p.layers[0].stat, StatBin)


class TestMultiLayer:
    """Test multi-layer plot construction."""

    def test_scatter_plus_smooth(self, mpg):
        p = (
            ggplot(mpg, aes("displ", "hwy"))
            + geom_point()
            + geom_smooth()
        )
        assert is_ggplot(p)
        assert len(p.layers) == 2

    def test_scatter_plus_line(self, mpg):
        p = (
            ggplot(mpg, aes("displ", "hwy"))
            + geom_point()
            + geom_line()
        )
        assert len(p.layers) == 2

    def test_layer_types(self, mpg):
        p = (
            ggplot(mpg, aes("displ", "hwy"))
            + geom_point()
            + geom_smooth()
        )
        assert all(isinstance(layer, Layer) for layer in p.layers)


class TestFacetedPlot:
    """Test faceted plot construction."""

    def test_scatter_with_facet_wrap(self, mpg):
        p = (
            ggplot(mpg, aes("displ", "hwy"))
            + geom_point()
            + facet_wrap("class")
        )
        assert is_ggplot(p)
        assert isinstance(p.facet, FacetWrap)
        assert len(p.layers) == 1


class TestThemedPlot:
    """Test themed plot construction."""

    def test_scatter_with_theme_bw(self, mpg):
        p = (
            ggplot(mpg, aes("displ", "hwy"))
            + geom_point()
            + theme_bw()
        )
        assert is_ggplot(p)
        assert isinstance(p.theme, Theme)
        assert p.theme.complete is True


class TestScaledPlot:
    """Test plot with custom scales."""

    def test_scatter_with_viridis(self, mpg):
        p = (
            ggplot(mpg, aes("displ", "hwy"))
            + geom_point()
            + scale_colour_viridis_c()
        )
        assert is_ggplot(p)
        assert len(p.layers) == 1


class TestFullPipeline:
    """Test combining many components."""

    def test_full_pipeline(self, mpg):
        p = (
            ggplot(mpg, aes("displ", "hwy"))
            + geom_point()
            + scale_x_continuous()
            + coord_flip()
            + facet_wrap("class")
            + theme_bw()
            + labs(title="Test", x="Displacement", y="Highway MPG")
        )
        assert is_ggplot(p)
        assert len(p.layers) == 1
        assert isinstance(p.coordinates, CoordFlip)
        assert isinstance(p.facet, FacetWrap)
        assert isinstance(p.theme, Theme)
        assert p.labels["title"] == "Test"

    def test_data_preserved(self, mpg):
        p = ggplot(mpg, aes("displ", "hwy")) + geom_point()
        assert isinstance(p.data, pd.DataFrame)
        assert p.data.shape == mpg.shape

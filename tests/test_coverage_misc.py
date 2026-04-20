"""Tests for annotation.py, draw_key.py, fortify.py, labels.py, limits.py,
ggproto.py, aes.py, and save.py."""

import warnings
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# =====================================================================
# annotation.py
# =====================================================================

from ggplot2_py.annotation import (
    annotate,
    annotation_custom,
    annotation_raster,
    annotation_logticks,
    annotation_map,
    annotation_borders,
    borders,
)
from ggplot2_py.layer import Layer, is_layer


class TestAnnotate:
    def test_basic_text(self):
        lyr = annotate("text", x=4, y=25, label="hello")
        assert is_layer(lyr)

    def test_basic_rect(self):
        lyr = annotate("rect", xmin=1, xmax=2, ymin=3, ymax=4, alpha=0.2)
        assert is_layer(lyr)

    def test_segment(self):
        lyr = annotate("segment", x=1, xend=2, y=1, yend=2)
        assert is_layer(lyr)

    def test_array_positions(self):
        lyr = annotate("point", x=[1, 2, 3], y=[4, 5, 6])
        assert is_layer(lyr)

    def test_vline_warns(self):
        with pytest.warns(UserWarning, match="geom_vline"):
            lyr = annotate("vline", x=1)

    def test_hline_warns(self):
        with pytest.warns(UserWarning, match="geom_hline"):
            lyr = annotate("hline", y=1)

    def test_abline_warns(self):
        with pytest.warns(UserWarning, match="geom_abline"):
            lyr = annotate("abline", x=1)

    def test_unequal_lengths_error(self):
        with pytest.raises(ValueError, match="Unequal"):
            annotate("point", x=[1, 2], y=[1, 2, 3])

    def test_position_param_warns(self):
        with pytest.warns(UserWarning, match="position"):
            annotate("point", x=1, y=1, position="dodge")

    def test_stat_param_warns(self):
        with pytest.warns(UserWarning, match="stat"):
            annotate("point", x=1, y=1, stat="identity")


class TestAnnotationCustom:
    def test_returns_layer(self):
        from grid_py import null_grob
        lyr = annotation_custom(null_grob(), xmin=0, xmax=1, ymin=0, ymax=1)
        assert is_layer(lyr)


class TestAnnotationRaster:
    def test_returns_layer(self):
        raster = np.zeros((10, 10, 3))
        lyr = annotation_raster(raster, xmin=0, xmax=1, ymin=0, ymax=1)
        assert is_layer(lyr)


class TestAnnotationLogticks:
    def test_returns_layer(self):
        lyr = annotation_logticks()
        assert is_layer(lyr)

    def test_color_alias(self):
        lyr = annotation_logticks(color="red")
        assert is_layer(lyr)

    def test_params(self):
        lyr = annotation_logticks(base=10, sides="bl", outside=True, scaled=False)
        assert is_layer(lyr)


class TestAnnotationStubs:
    def test_annotation_map_raises(self):
        with pytest.raises(NotImplementedError):
            annotation_map(pd.DataFrame())

    def test_annotation_borders_raises(self):
        with pytest.raises(NotImplementedError):
            annotation_borders()

    def test_borders_raises(self):
        with pytest.raises(NotImplementedError):
            borders()


# =====================================================================
# draw_key.py
# =====================================================================

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
    _get,
    _alpha,
    _fill_alpha,
)


class TestDrawKeyHelpers:
    def test_get_dict(self):
        assert _get({"a": 1}, "a") == 1
        assert _get({"a": 1}, "b", 99) == 99

    def test_get_object(self):
        class Obj:
            a = 1
        assert _get(Obj(), "a") == 1
        assert _get(Obj(), "b", 42) == 42

    def test_alpha(self):
        result = _alpha("red", 0.5)
        assert result is not None

    def test_alpha_fallback(self):
        result = _alpha(None, None)
        assert result is None

    def test_fill_alpha_none(self):
        assert _fill_alpha(None, 0.5) is None

    def test_fill_alpha_valid(self):
        result = _fill_alpha("blue", 0.5)
        assert result is not None


class TestDrawKeyFunctions:
    def _data(self, **kw):
        d = {
            "colour": "black", "fill": "grey", "size": 1.5,
            "linewidth": 0.5, "linetype": 1, "alpha": 1.0,
            "shape": 19, "stroke": 0.5,
        }
        d.update(kw)
        return d

    def _params(self, **kw):
        p = {"lineend": "butt", "linejoin": "mitre"}
        p.update(kw)
        return p

    def test_draw_key_point(self):
        result = draw_key_point(self._data(), self._params())
        assert result is not None

    def test_draw_key_path(self):
        result = draw_key_path(self._data(), self._params())
        assert result is not None

    def test_draw_key_path_no_linetype(self):
        d = self._data(linetype=None)
        result = draw_key_path(d, self._params())
        assert result is not None

    def test_draw_key_rect(self):
        result = draw_key_rect(self._data(), self._params())
        assert result is not None

    def test_draw_key_rect_no_fill(self):
        d = self._data(fill=None)
        result = draw_key_rect(d, self._params())
        assert result is not None

    def test_draw_key_polygon(self):
        result = draw_key_polygon(self._data(), self._params())
        assert result is not None

    def test_draw_key_blank(self):
        result = draw_key_blank(self._data(), self._params())
        assert result is not None

    def test_draw_key_boxplot(self):
        result = draw_key_boxplot(self._data(), self._params())
        assert result is not None

    def test_draw_key_crossbar(self):
        result = draw_key_crossbar(self._data(), self._params())
        assert result is not None

    def test_draw_key_dotplot(self):
        result = draw_key_dotplot(self._data(), self._params())
        assert result is not None

    def test_draw_key_label(self):
        result = draw_key_label(self._data(label="A", linewidth=0.25), self._params())
        assert result is not None

    def test_draw_key_label_no_linewidth(self):
        result = draw_key_label(self._data(label="B", linewidth=0), self._params())
        assert result is not None

    def test_draw_key_linerange_horizontal(self):
        result = draw_key_linerange(self._data(), self._params(flipped_aes=True))
        assert result is not None

    def test_draw_key_linerange_vertical(self):
        result = draw_key_linerange(self._data(), self._params(flipped_aes=False))
        assert result is not None

    def test_draw_key_pointrange(self):
        result = draw_key_pointrange(self._data(), self._params())
        assert result is not None

    def test_draw_key_smooth_no_se(self):
        result = draw_key_smooth(self._data(), self._params(se=False))
        assert result is not None

    def test_draw_key_smooth_with_se(self):
        result = draw_key_smooth(self._data(), self._params(se=True))
        assert result is not None

    def test_draw_key_text(self):
        result = draw_key_text(self._data(label="X"), self._params())
        assert result is not None

    def test_draw_key_abline(self):
        result = draw_key_abline(self._data(), self._params())
        assert result is not None

    def test_draw_key_vline(self):
        result = draw_key_vline(self._data(), self._params())
        assert result is not None

    def test_draw_key_vpath(self):
        result = draw_key_vpath(self._data(), self._params())
        assert result is not None

    def test_draw_key_timeseries(self):
        result = draw_key_timeseries(self._data(), self._params())
        assert result is not None

    def test_draw_key_timeseries_no_linetype(self):
        result = draw_key_timeseries(self._data(linetype=None), self._params())
        assert result is not None


# =====================================================================
# fortify.py
# =====================================================================

from ggplot2_py.fortify import fortify
from ggplot2_py._compat import Waiver, is_waiver


class TestFortify:
    def test_none(self):
        result = fortify(None)
        assert is_waiver(result)

    def test_dataframe(self):
        df = pd.DataFrame({"x": [1]})
        assert fortify(df) is df

    def test_waiver_passthrough(self):
        from ggplot2_py._compat import waiver
        w = waiver()
        assert fortify(w) is w

    def test_callable(self):
        fn = lambda: None
        assert fortify(fn) is fn

    def test_dict(self):
        result = fortify({"x": [1, 2], "y": [3, 4]})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_numpy_array(self):
        result = fortify(np.array([[1, 2], [3, 4]]))
        assert isinstance(result, pd.DataFrame)

    def test_to_pandas_protocol(self):
        class HasToPandas:
            def to_pandas(self):
                return pd.DataFrame({"x": [1]})
        result = fortify(HasToPandas())
        assert isinstance(result, pd.DataFrame)

    def test_invalid_type_raises(self):
        class BadObj:
            pass
        with pytest.raises(TypeError, match="must be a DataFrame"):
            fortify(BadObj())


# =====================================================================
# labels.py
# =====================================================================

from ggplot2_py.labels import (
    labs,
    xlab,
    ylab,
    ggtitle,
    Labels,
    is_labels,
    make_labels,
    update_labels,
    get_labs,
)
from ggplot2_py.aes import aes, Mapping, AfterStat, AfterScale, Stage


class TestLabels:
    def test_labels_class(self):
        l = Labels(x="X axis")
        assert isinstance(l, dict)
        assert "Labels" in repr(l)

    def test_is_labels(self):
        assert is_labels(Labels())
        assert not is_labels({})

    def test_labs_basic(self):
        l = labs(title="T", x="X")
        assert l["title"] == "T"
        assert l["x"] == "X"

    def test_labs_all_special(self):
        l = labs(title="T", subtitle="S", caption="C", tag="A",
                 alt="Alt", alt_insight="Insight", dictionary={"a": "b"})
        assert l["title"] == "T"
        assert l["subtitle"] == "S"
        assert l["caption"] == "C"
        assert l["tag"] == "A"
        assert l["alt"] == "Alt"
        assert l["alt_insight"] == "Insight"

    def test_labs_aesthetic_alias(self):
        l = labs(color="Color")
        assert l["colour"] == "Color"

    def test_xlab(self):
        l = xlab("Foo")
        assert l["x"] == "Foo"

    def test_xlab_none(self):
        l = xlab(None)
        # xlab(None) passes x=None to labs, which includes it
        assert "x" in l
        assert l["x"] is None

    def test_ylab(self):
        l = ylab("Bar")
        assert l["y"] == "Bar"

    def test_ggtitle(self):
        l = ggtitle("Title", subtitle="Sub")
        assert l["title"] == "Title"
        assert l["subtitle"] == "Sub"

    def test_ggtitle_no_subtitle(self):
        l = ggtitle("Title")
        assert l["title"] == "Title"


class TestMakeLabels:
    def test_string_mapping(self):
        m = Mapping(x="displ", y="hwy")
        result = make_labels(m)
        assert result["x"] == "displ"
        assert result["y"] == "hwy"

    def test_after_stat(self):
        m = Mapping(y=AfterStat("count"))
        result = make_labels(m)
        assert result["y"] == "count"

    def test_after_scale(self):
        m = Mapping(colour=AfterScale("fill"))
        result = make_labels(m)
        assert result["colour"] == "fill"

    def test_stage(self):
        m = Mapping(colour=Stage(start="class"))
        result = make_labels(m)
        assert result["colour"] == "class"

    def test_stage_after_stat(self):
        m = Mapping(y=Stage(after_stat="count"))
        result = make_labels(m)
        # Stage.after_stat is AfterStat, str(AfterStat('count')) = "AfterStat('count')"
        assert "count" in result["y"]

    def test_stage_after_scale(self):
        m = Mapping(colour=Stage(after_scale="fill"))
        result = make_labels(m)
        assert "fill" in result["colour"]

    def test_stage_empty(self):
        m = Mapping(colour=Stage())
        result = make_labels(m)
        assert result["colour"] == "colour"

    def test_none_value(self):
        m = Mapping(x=None)
        result = make_labels(m)
        assert result["x"] == "x"

    def test_plain_dict_accepted(self):
        # Plain dicts come through Stat/Geom default_aes; they're handled
        # the same way as Mapping instances (both are just named key/value
        # pairs).  Atomic values become their string form — consistent with
        # ``test_numeric_value`` above.
        result = make_labels({"x": "a"})
        assert result == {"x": "a"}

    def test_numeric_value(self):
        m = Mapping(x=42)
        result = make_labels(m)
        assert result["x"] == "42"


class TestUpdateLabels:
    def test_basic(self):
        from ggplot2_py.plot import ggplot
        p = ggplot()
        result = update_labels(p, Labels(title="New"))
        assert result.labels["title"] == "New"


class TestGetLabs:
    def test_none_plot(self):
        result = get_labs(None)
        assert isinstance(result, Labels)

    def test_ggplot(self):
        from ggplot2_py.plot import ggplot
        p = ggplot()
        p.labels["x"] = "test"
        result = get_labs(p)
        assert "x" in result


# =====================================================================
# limits.py
# =====================================================================

from ggplot2_py.limits import xlim, ylim, lims, expand_limits


class TestLimits:
    def test_xlim_two_args(self):
        result = xlim(0, 10)
        assert result is not None

    def test_xlim_sequence(self):
        result = xlim([0, 10])
        assert result is not None

    def test_ylim_two_args(self):
        result = ylim(0, 50)
        assert result is not None

    def test_ylim_sequence(self):
        result = ylim([0, 50])
        assert result is not None

    def test_xlim_reversed(self):
        result = xlim(10, 0)
        assert result is not None

    def test_xlim_none_lower(self):
        result = xlim(None, 10)
        assert result is not None

    def test_lims_numeric(self):
        result = lims(x=(0, 10))
        assert len(result) == 1

    def test_lims_character(self):
        result = lims(colour=["red", "blue", "green"])
        assert len(result) == 1

    def test_lims_non_list_raises(self):
        with pytest.raises(ValueError, match="list or tuple"):
            lims(x=42)

    def test_xlim_wrong_length_raises(self):
        with pytest.raises(ValueError, match="two-element"):
            xlim(1, 2, 3)

    def test_empty_lims_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            xlim([])

    def test_lims_numpy_array(self):
        result = xlim(np.array([0, 10]))
        assert result is not None

    def test_expand_limits(self):
        lyr = expand_limits(x=0, y=[0, 100])
        assert is_layer(lyr)

    def test_expand_limits_scalar(self):
        lyr = expand_limits(x=5)
        assert is_layer(lyr)

    def test_expand_limits_series(self):
        lyr = expand_limits(x=pd.Series([1, 2, 3]))
        assert is_layer(lyr)

    def test_expand_limits_uneven(self):
        lyr = expand_limits(x=[1, 2, 3], y=[10])
        assert is_layer(lyr)


# =====================================================================
# ggproto.py
# =====================================================================

from ggplot2_py.ggproto import (
    GGProto,
    ggproto,
    ggproto_parent,
    is_ggproto,
    fetch_ggproto,
    GGProtoMeta,
)


class TestGGProto:
    def test_metaclass_repr(self):
        r = repr(GGProto)
        assert "ggproto class" in r

    def test_instance_repr(self):
        obj = GGProto()
        assert "ggproto object" in repr(obj)

    def test_instance_repr_with_name(self):
        obj = GGProto()
        obj._class_name = "MyObj"
        assert "MyObj" in repr(obj)

    def test_set(self):
        obj = GGProto()
        obj._set(custom=42)
        assert obj.custom == 42

    def test_auto_bind_self(self):
        cls = ggproto("TestCls", GGProto, greet=lambda self: "hello")
        obj = cls()
        assert obj.greet() == "hello"

    def test_ggproto_factory(self):
        MyClass = ggproto("MyClass", GGProto, value=10)
        assert MyClass._class_name == "MyClass"
        assert MyClass.value == 10

    def test_ggproto_inheritance(self):
        Parent = ggproto("Parent", GGProto, x=1)
        Child = ggproto("Child", Parent, y=2)
        obj = Child()
        assert obj.x == 1
        assert obj.y == 2

    def test_ggproto_default_parent(self):
        cls = ggproto("NoPar")
        assert issubclass(cls, GGProto)


class TestGGProtoParent:
    def test_basic(self):
        Parent = ggproto("Parent", GGProto, greet=lambda self: "parent")
        Child = ggproto("Child", Parent, greet=lambda self: "child")
        obj = Child()
        proxy = ggproto_parent(Parent, obj)
        assert proxy.greet() == "parent"

    def test_repr(self):
        proxy = ggproto_parent(GGProto, GGProto())
        assert "parent proxy" in repr(proxy)

    def test_missing_attr_raises(self):
        proxy = ggproto_parent(GGProto, GGProto())
        with pytest.raises(AttributeError, match="no member"):
            proxy.nonexistent_xyz


class TestIsGGProto:
    def test_instance(self):
        assert is_ggproto(GGProto())

    def test_class(self):
        assert is_ggproto(GGProto)

    def test_negative(self):
        assert not is_ggproto("hello")
        assert not is_ggproto(42)


class TestFetchGGProto:
    def test_basic(self):
        cls = ggproto("Test", GGProto, val=42)
        assert fetch_ggproto(cls, "val") == 42

    def test_non_ggproto_raises(self):
        with pytest.raises(TypeError, match="Expected a GGProto"):
            fetch_ggproto("bad", "val")


# =====================================================================
# aes.py
# =====================================================================

from ggplot2_py.aes import (
    aes as aes_fn,
    after_stat,
    after_scale,
    stage,
    vars as aes_vars,
    is_mapping,
    standardise_aes_names,
    rename_aes,
    Mapping as AesMapping,
    AfterStat as AesAfterStat,
    AfterScale as AesAfterScale,
    Stage as AesStage,
    AESTHETIC_ALIASES,
)


class TestAes:
    def test_basic(self):
        m = aes_fn(x="a", y="b")
        assert m["x"] == "a"
        assert m["y"] == "b"

    def test_color_alias(self):
        m = aes_fn(color="class")
        assert "colour" in m
        assert m["colour"] == "class"

    def test_repr(self):
        m = aes_fn(x="a")
        assert "aes(" in repr(m)

    def test_attribute_access(self):
        m = aes_fn(x="a")
        assert m.x == "a"

    def test_attribute_missing_raises(self):
        m = aes_fn(x="a")
        with pytest.raises(AttributeError):
            _ = m.nonexistent

    def test_no_args(self):
        m = aes_fn()
        assert len(m) == 0


class TestAfterStat:
    def test_creation(self):
        a = after_stat("count")
        assert isinstance(a, AesAfterStat)
        assert a.x == "count"

    def test_repr(self):
        assert "count" in repr(after_stat("count"))

    def test_eq(self):
        assert after_stat("count") == after_stat("count")
        assert after_stat("count") != after_stat("density")

    def test_eq_other_type(self):
        assert after_stat("count").__eq__("count") is NotImplemented

    def test_hash(self):
        s = {after_stat("count"), after_stat("count")}
        assert len(s) == 1

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            AesAfterStat(42)


class TestAfterScale:
    def test_creation(self):
        a = after_scale("fill")
        assert isinstance(a, AesAfterScale)
        assert a.x == "fill"

    def test_repr(self):
        assert "fill" in repr(after_scale("fill"))

    def test_eq(self):
        assert after_scale("fill") == after_scale("fill")
        assert after_scale("fill") != after_scale("colour")

    def test_eq_other_type(self):
        assert after_scale("fill").__eq__("fill") is NotImplemented

    def test_hash(self):
        s = {after_scale("fill"), after_scale("fill")}
        assert len(s) == 1

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            AesAfterScale(42)


class TestStage:
    def test_creation(self):
        s = stage(start="class", after_scale="fill")
        assert s.start == "class"
        assert isinstance(s.after_scale, AesAfterScale)

    def test_after_stat_string(self):
        s = stage(after_stat="count")
        assert isinstance(s.after_stat, AesAfterStat)

    def test_repr(self):
        s = stage(start="x")
        assert "Stage" in repr(s)

    def test_eq(self):
        s1 = stage(start="x")
        s2 = stage(start="x")
        assert s1 == s2

    def test_eq_different(self):
        assert stage(start="x") != stage(start="y")

    def test_eq_other_type(self):
        assert stage(start="x").__eq__("x") is NotImplemented

    def test_hash(self):
        s = {stage(start="x"), stage(start="x")}
        assert len(s) == 1


class TestVars:
    def test_positional(self):
        assert aes_vars("cyl", "drv") == ["cyl", "drv"]

    def test_keyword(self):
        result = aes_vars(rows="cyl")
        assert result == ["cyl"]

    def test_mixed(self):
        result = aes_vars("a", b="c")
        assert result == ["a", "c"]


class TestStandardise:
    def test_basic(self):
        result = standardise_aes_names(["color", "lwd", "x"])
        assert result == ["colour", "linewidth", "x"]


class TestRenameAes:
    def test_basic(self):
        m = AesMapping(color="class")
        result = rename_aes(m)
        assert "colour" in result
        assert "color" not in result


class TestIsMapping:
    def test_positive(self):
        assert is_mapping(aes_fn(x="a"))

    def test_negative(self):
        assert not is_mapping({"x": "a"})


# =====================================================================
# save.py
# =====================================================================

from ggplot2_py.save import check_device, _parse_dpi, _to_inches


class TestCheckDevice:
    def test_explicit_device(self):
        assert check_device("png", "test.pdf") == "png"

    def test_from_extension(self):
        assert check_device(None, "test.pdf") == "pdf"
        assert check_device(None, "test.PNG") == "png"

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            check_device("xyz", "test.xyz")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot determine"):
            check_device(None, "noext")

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            check_device(None, "test.xyz123")


class TestParseDpi:
    def test_numeric(self):
        assert _parse_dpi(300) == 300.0
        assert _parse_dpi(72.0) == 72.0

    def test_presets(self):
        assert _parse_dpi("screen") == 72.0
        assert _parse_dpi("print") == 300.0
        assert _parse_dpi("retina") == 320.0

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown DPI"):
            _parse_dpi("ultra")


class TestToInches:
    def test_none(self):
        assert _to_inches(None, "in", 300) is None

    def test_inches(self):
        assert _to_inches(5.0, "in", 300) == 5.0

    def test_cm(self):
        result = _to_inches(2.54, "cm", 300)
        assert abs(result - 1.0) < 0.001

    def test_mm(self):
        result = _to_inches(25.4, "mm", 300)
        assert abs(result - 1.0) < 0.001

    def test_px(self):
        result = _to_inches(300, "px", 300)
        assert abs(result - 1.0) < 0.001

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            _to_inches(1.0, "furlongs", 300)

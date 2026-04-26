"""Boilerplate constructor generator — port of R ``make-constructor.R``.

Lets extension authors generate user-facing ``geom_*()`` / ``stat_*()``
layer constructors directly from a :class:`Geom` / :class:`Stat`
subclass, mirroring R's :func:`make_constructor`. The returned
function carries a real :class:`inspect.Signature` so IDEs, ``help()``,
and runtime introspection see the right parameter list and defaults.

Idiomatic Python adaptation of R's quotation-and-eval design:

* R uses ``rlang::enexprs`` / ``pairlist2`` / ``call2`` / ``new_function``
  to splice an AST and synthesize a function with non-standard
  evaluation. We use :class:`inspect.Signature` and a closure — the
  result has the same observable signature without runtime AST
  manipulation.
* R's ``checks = rlang::exprs(...)`` (a list of unevaluated calls) maps
  to ``checks = [callable, ...]`` here, where each callable receives
  the bound argument dict and may mutate it in place or raise. This
  swap keeps the boilerplate-injection feature without leaning on
  ``eval``.

R reference: ``ggplot2/R/make-constructor.R`` (lines 59-235).
"""
from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping as TypingMapping,
    Optional,
    Tuple,
    Type,
)

from ggplot2_py._compat import cli_abort, cli_warn

__all__ = ["make_constructor"]


# Fixed signature slots — match R's ``fixed_fmls_names``. The Python
# canonical form uses underscores (``na_rm`` not ``na.rm``); R's dotted
# names get translated by ``layer()`` upstream so this is purely a
# user-facing surface choice.
_GEOM_FIXED: Tuple[str, ...] = (
    "mapping", "data", "stat", "position",
    "na_rm", "show_legend", "inherit_aes",
)
_STAT_FIXED: Tuple[str, ...] = (
    "mapping", "data", "geom", "position",
    "na_rm", "show_legend", "inherit_aes",
)

# ``flipped_aes`` is auto-injected by orientation-aware geoms; never
# treat it as a constructor-extras parameter (R: same exclusion).
_FLIPPED_AES: Tuple[str, ...] = ("flipped_aes",)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_constructor(
    cls: Type,
    *,
    checks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
    omit: Tuple[str, ...] = (),
    **extras: Any,
) -> Callable[..., Any]:
    """Synthesize a ``geom_*()`` / ``stat_*()`` constructor from a class.

    Mirrors R :func:`make_constructor` (R/make-constructor.R:59-235) —
    inspects ``cls.draw_panel`` / ``cls.draw_group`` (Geom) or
    ``cls.compute_panel`` / ``cls.compute_group`` (Stat) to derive the
    layer-parameter list, blends in user-supplied *extras*, and returns
    a callable that delegates to :func:`layer`.

    Parameters
    ----------
    cls : type
        A subclass of :class:`Geom` or :class:`Stat`.
    checks : list of callable, optional
        Functions ``f(params: dict) -> None``. Run before the
        :func:`layer` call; may raise to abort, or mutate ``params``
        in place to coerce values. Mirrors R's
        ``checks = rlang::exprs(...)`` slot.
    omit : tuple of str, optional
        Names of automatically-retrieved parameters to exclude from
        the generated signature (e.g. internally-computed variables).
    **extras
        Named defaults to add to the constructor's signature.
        ``geom`` (when *cls* is a Geom) and ``stat`` (when *cls* is a
        Stat) are reserved.

    Returns
    -------
    callable
        A function with R-faithful signature: positional ``mapping``,
        ``data``, ``stat``/``geom``, ``position``; keyword-only
        *extras*, ``na_rm``, ``show_legend``, ``inherit_aes``; trailing
        ``**kwargs`` for unknown layer parameters. Calling it returns
        a :class:`Layer` ready to be added to a :class:`GGPlot`.

    Examples
    --------
    >>> from ggplot2_py import Geom
    >>> class GeomFoo(Geom):
    ...     def draw_panel(self, data, panel_params, coord, alpha=0.5):
    ...         pass
    >>> geom_foo = make_constructor(GeomFoo, na_rm=False)
    >>> import inspect
    >>> "alpha" in inspect.signature(geom_foo).parameters
    True
    """
    from ggplot2_py.geom import Geom
    from ggplot2_py.stat import Stat

    if isinstance(cls, type) and issubclass(cls, Geom):
        return _make_layer_constructor(
            cls,
            partner_kind="geom",
            fixed=_GEOM_FIXED,
            introspect_methods=("draw_panel", "draw_group"),
            base_class=Geom,
            checks=checks,
            omit=tuple(omit),
            extras=dict(extras),
        )
    if isinstance(cls, type) and issubclass(cls, Stat):
        return _make_layer_constructor(
            cls,
            partner_kind="stat",
            fixed=_STAT_FIXED,
            introspect_methods=("compute_panel", "compute_group"),
            base_class=Stat,
            checks=checks,
            omit=tuple(omit),
            extras=dict(extras),
        )
    cli_abort(
        f"`make_constructor` does not have a method for "
        f"{type(cls).__name__!s}; expected a Geom or Stat subclass."
    )


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _short_name(cls: Type, prefix: str) -> str:
    """``GeomPoint`` → ``"point"``; ``StatBin2d`` → ``"bin2d"``.

    Mirrors R ``snake_class(x)`` minus the ``geom_`` / ``stat_`` prefix.
    """
    name = cls.__name__
    if name.startswith(prefix):
        name = name[len(prefix):]
    out: List[str] = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(c.lower())
    return "".join(out) or name.lower()


def _named_params(fn: Callable[..., Any]) -> Dict[str, inspect.Parameter]:
    """Return ordered named parameters of *fn*, skipping self/var-args."""
    sig = inspect.signature(fn)
    out: Dict[str, inspect.Parameter] = {}
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                       inspect.Parameter.VAR_KEYWORD):
            continue
        out[p.name] = p
    return out


def _has_var_kw(fn: Callable[..., Any]) -> bool:
    sig = inspect.signature(fn)
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


def _make_layer_constructor(
    cls: Type,
    *,
    partner_kind: str,           # "geom" or "stat"
    fixed: Tuple[str, ...],
    introspect_methods: Tuple[str, str],
    base_class: Type,
    checks: Optional[List[Callable[[Dict[str, Any]], None]]],
    omit: Tuple[str, ...],
    extras: Dict[str, Any],
) -> Callable[..., Any]:
    """Build a constructor for either Geom or Stat dispatch."""

    reserved = "geom" if partner_kind == "geom" else "stat"
    if reserved in extras:
        cli_abort(f"`{reserved}` is a reserved argument.")

    short_name = _short_name(cls, base_class.__name__)
    warn_label = f"{partner_kind}_{short_name}"

    # The class needs to be instantiated to call ``aesthetics()`` /
    # ``parameters()``. Builtin Geom / Stat subclasses are zero-arg.
    instance = cls()

    # ----- Split user kwargs (R: ``setdiff(names(args), fixed_fmls_names)``)
    # User-supplied kwargs that name a fixed slot (``stat``, ``position``,
    # ``na_rm``, ...) override the slot's default value rather than adding
    # a new parameter; everything else becomes a real extras parameter.
    fixed_overrides: Dict[str, Any] = {}
    real_extras: Dict[str, Any] = {}
    for k, v in extras.items():
        if k in fixed:
            fixed_overrides[k] = v
        else:
            real_extras[k] = v

    aes_set = set(instance.aesthetics())
    parameters = list(instance.parameters())

    known = (
        set(real_extras)
        | set(fixed)
        | set(_FLIPPED_AES)
        | aes_set
        | set(omit)
    )
    missing = [p for p in parameters if p not in known]

    auto_extras: Dict[str, Any] = dict(real_extras)
    if missing:
        # R parity (make-constructor.R:86-89): inspect ``draw_panel``
        # (or ``compute_panel``) first; if it's a thin ``*args/**kwargs``
        # forwarder, fall back to ``draw_group`` / ``compute_group``.
        # Python's port quirk: many overrides keep an explicit
        # ``**params`` to swallow unused kwargs WHILE still defining
        # real parameters with defaults — so we union both methods'
        # named params and let the panel-level method win on conflicts,
        # which matches R's intent (collect all introspectable defaults).
        primary = getattr(cls, introspect_methods[0])
        secondary = getattr(cls, introspect_methods[1])
        primary_args = _named_params(primary)
        secondary_args = _named_params(secondary)

        merged_args = dict(secondary_args)
        merged_args.update(primary_args)  # primary (panel) wins

        for param_name in missing:
            if param_name in merged_args:
                p = merged_args[param_name]
                if p.default is not inspect.Parameter.empty:
                    auto_extras[param_name] = p.default
                # else: parameter has no default — leave it out so the
                # ``cli_warn`` below fires (matches R's behaviour).

        still_missing = [p for p in missing if p not in auto_extras]
        if still_missing:
            cli_warn(
                f"In `{warn_label}()`: please consider providing default "
                f"values for: {', '.join(still_missing)}."
            )

    return _synthesize_constructor(
        cls,
        partner_kind=partner_kind,
        partner_name=short_name,
        extras=auto_extras,
        fixed_overrides=fixed_overrides,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# Function synthesis
# ---------------------------------------------------------------------------


_FIXED_DEFAULTS: Dict[str, Any] = {
    "mapping": None,
    "data": None,
    "stat": "identity",
    "position": "identity",
    "na_rm": False,
    "show_legend": None,
    "inherit_aes": True,
}


class _RequiredSentinel:
    """Marker for required keyword args (no default).

    Allows the synthesized signature to expose the param without a
    default; the closure raises an R-faithful message at call time
    when the user omits it.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<required>"


_REQUIRED = _RequiredSentinel()


def _synthesize_constructor(
    cls: Type,
    *,
    partner_kind: str,
    partner_name: str,
    extras: Dict[str, Any],
    fixed_overrides: Dict[str, Any],
    checks: Optional[List[Callable[[Dict[str, Any]], None]]],
) -> Callable[..., Any]:
    """Materialize the callable with its full :class:`inspect.Signature`."""

    def _slot_default(slot: str) -> Any:
        if slot in fixed_overrides:
            return fixed_overrides[slot]
        return _FIXED_DEFAULTS[slot]

    params: List[inspect.Parameter] = [
        inspect.Parameter("mapping", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          default=_slot_default("mapping")),
        inspect.Parameter("data", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          default=_slot_default("data")),
    ]
    if partner_kind == "geom":
        params.append(inspect.Parameter(
            "stat", inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=_slot_default("stat"),
        ))
    else:
        # Stat: ``geom`` is required unless the caller passed an
        # explicit default via ``make_constructor(StatX, geom="...")``.
        # R uses an eager ``args$geom %||% cli_abort(...)`` default at
        # pairlist2 time, which is actually a bug — it aborts at
        # construction when no default is provided. We use the
        # _REQUIRED sentinel so the abort fires at *call* time
        # (matching the documented intent), and accept an explicit
        # geom kwarg to suppress it.
        geom_default = fixed_overrides.get("geom", _REQUIRED)
        params.append(inspect.Parameter(
            "geom", inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=geom_default,
        ))
    params.append(inspect.Parameter(
        "position", inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=_slot_default("position"),
    ))

    # Auto-extras and user-supplied extras — keyword-only to match R's
    # ``pairlist2(..., extras...)`` placement after ``...``.
    for ename, edefault in extras.items():
        params.append(inspect.Parameter(
            ename, inspect.Parameter.KEYWORD_ONLY,
            default=edefault,
        ))

    for fname in ("na_rm", "show_legend", "inherit_aes"):
        params.append(inspect.Parameter(
            fname, inspect.Parameter.KEYWORD_ONLY,
            default=_slot_default(fname),
        ))

    # Trailing ``**kwargs`` so the user can pass arbitrary fixed
    # aesthetics / unknown layer params without explicit declaration —
    # ``layer()`` warns about anything truly unknown.
    params.append(inspect.Parameter(
        "kwargs", inspect.Parameter.VAR_KEYWORD,
    ))

    sig = inspect.Signature(params)
    extras_keys = tuple(extras.keys())

    def constructor(*args: Any, **kw: Any) -> Any:
        try:
            bound = sig.bind(*args, **kw)
        except TypeError as exc:
            cli_abort(str(exc))
        bound.apply_defaults()
        a = bound.arguments

        # User-injected validation hooks (R: ``checks = exprs(...)``).
        if checks:
            for chk in checks:
                chk(a)

        # ``geom`` is required for Stat dispatch — R aborts at the
        # default-evaluation site; we abort here at call time with the
        # same message.
        if partner_kind == "stat":
            geom_v = a.get("geom")
            if isinstance(geom_v, _RequiredSentinel) or geom_v is None:
                cli_abort("`geom` is required.")

        # Build the params dict in R's order: na_rm first, then the
        # auto-extras / user-extras, then any **kwargs spillover.
        params_dict: Dict[str, Any] = {"na_rm": a["na_rm"]}
        for ek in extras_keys:
            params_dict[ek] = a[ek]
        params_dict.update(a.get("kwargs", {}))

        from ggplot2_py.layer import layer as _layer

        layer_kwargs: Dict[str, Any] = dict(
            mapping=a["mapping"],
            data=a["data"],
            position=a["position"],
            show_legend=a["show_legend"],
            inherit_aes=a["inherit_aes"],
            params=params_dict,
        )
        if partner_kind == "geom":
            layer_kwargs["geom"] = cls
            layer_kwargs["stat"] = a["stat"]
        else:
            layer_kwargs["stat"] = cls
            layer_kwargs["geom"] = a["geom"]
        return _layer(**layer_kwargs)

    constructor.__signature__ = sig
    constructor.__name__ = f"{partner_kind}_{partner_name}"
    constructor.__qualname__ = constructor.__name__
    constructor.__doc__ = (
        f"Auto-generated layer constructor for "
        f"`{cls.__name__}` (via :func:`make_constructor`).\n\n"
        f"Returns a Layer ready to be added to a ``ggplot()`` plot."
    )
    return constructor

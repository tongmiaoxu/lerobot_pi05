"""Sim-to-real observation tooling (lazy-load heavy deps until needed)."""

__all__ = ["SimToRealTranslator"]


def __getattr__(name: str):
    if name == "SimToRealTranslator":
        from .translator import SimToRealTranslator as _SimToRealTranslator

        return _SimToRealTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

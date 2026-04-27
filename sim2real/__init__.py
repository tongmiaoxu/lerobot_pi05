"""Sim-to-real observation tooling (lazy-load heavy deps until needed)."""

__all__ = ["GPTImageTranslator", "SimToRealTranslator", "load_calibration_pairs_pil"]


def __getattr__(name: str):
    if name == "GPTImageTranslator":
        from .gpt_image_translator import GPTImageTranslator as _GPTImageTranslator

        return _GPTImageTranslator
    if name == "load_calibration_pairs_pil":
        from .gpt_image_translator import load_calibration_pairs_pil as _load_calibration_pairs_pil

        return _load_calibration_pairs_pil
    if name == "SimToRealTranslator":
        from .translator import SimToRealTranslator as _SimToRealTranslator

        return _SimToRealTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

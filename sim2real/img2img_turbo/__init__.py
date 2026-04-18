# Vendored pix2pix-turbo core (see pix2pix_turbo.py header for upstream URL + commit).
# Lazy exports so `import sim2real.img2img_turbo.my_utils.training_utils` does not pull diffusers/xformers.

__all__ = ["Pix2Pix_Turbo"]


def __getattr__(name: str):
    if name == "Pix2Pix_Turbo":
        from .pix2pix_turbo import Pix2Pix_Turbo as _Pix2Pix_Turbo

        return _Pix2Pix_Turbo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

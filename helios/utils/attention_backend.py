"""Attention backend selection with a safe native fallback.

The hub-backed flash-attention kernels are resolved lazily by diffusers and
their wrapper names can change independently of a Helios release.  Keep the
hardware-specific preference in one place and fall back to PyTorch native
attention when a backend is unavailable or incompatible.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence


FLASH3_HUB = "_flash_3_hub"
FLASH2_HUB = "flash_hub"
NATIVE = "native"

_BACKEND_MARKERS = (
    "attention backend",
    "flash attention",
    "flash_attn",
    "flash-hub",
    "flash_hub",
    "kernel module",
    "kernel package",
    "kernels package",
    "compute capability",
    "not usable",
    "missing package",
    "does not define attribute",
    "cudnn_status",
    "cudnn",
)


def is_backend_compatibility_error(error: BaseException) -> bool:
    """Return whether *error* indicates an unavailable attention backend.

    Backend setup happens next to model construction, so broad exception
    handling would hide unrelated configuration and programming errors.  The
    messages below cover the errors emitted by diffusers/kernels for missing,
    incompatible, or stale flash-attention implementations.
    """

    message = str(error).lower()
    if not message:
        return False

    if isinstance(error, (ImportError, ModuleNotFoundError, AttributeError)):
        return any(marker in message for marker in _BACKEND_MARKERS)

    if isinstance(error, (RuntimeError, ValueError)):
        if any(marker in message for marker in _BACKEND_MARKERS):
            return True
        # Hardware checks in different diffusers releases use either
        # "unsupported" or "not supported" alongside the capability name.
        return "compute capability" in message and (
            "unsupported" in message or "not supported" in message
        )

    return False


def _default_logger() -> logging.Logger:
    return logging.getLogger(__name__)


def configure_attention_backend(
    transformer: Any,
    compute_capability: Sequence[int] | None = None,
    logger: Any | None = None,
) -> str:
    """Select the fastest available attention backend for ``transformer``.

    Hopper-class devices try FA3 first and then FA2.  Older CUDA devices try
    FA2 directly.  If hub kernel resolution fails (for example after a
    wrapper rename in the kernels package), native PyTorch SDPA is selected.
    Errors unrelated to backend availability are re-raised so that model or
    configuration failures remain visible to callers.

    Args:
        transformer: A diffusers model implementing ``set_attention_backend``.
        compute_capability: Optional ``(major, minor)`` tuple.  Supplying it is
            useful for callers that already queried CUDA and for unit tests.
        logger: Optional logger exposing ``warning``.  The standard library
            logger is used when omitted.

    Returns:
        The backend name selected, including ``"native"`` when the fallback
        path is used.
    """

    if compute_capability is None:
        import torch

        compute_capability = torch.cuda.get_device_capability()

    if not compute_capability:
        raise ValueError("compute_capability must contain a CUDA major version")

    log = logger or _default_logger()
    major = int(compute_capability[0])
    candidates = [FLASH3_HUB, FLASH2_HUB] if major >= 9 else [FLASH2_HUB]
    failures: list[tuple[str, BaseException]] = []

    for backend in candidates:
        try:
            transformer.set_attention_backend(backend)
        except Exception as error:
            if not is_backend_compatibility_error(error):
                raise
            failures.append((backend, error))
            log.warning(
                "Attention backend %s is unavailable; trying a compatible fallback: %s",
                backend,
                error,
            )
        else:
            return backend

    # ``native`` is the public diffusers name for torch SDPA.  Calling the
    # setter explicitly also clears a previously selected hub backend in the
    # global attention registry; reset_attention_backend() alone did not do so
    # in older diffusers releases.
    try:
        transformer.set_attention_backend(NATIVE)
    except Exception as error:
        # A failure to configure native attention is not a backend capability
        # problem anymore, so preserve the original exception and context.
        attempted = ", ".join(name for name, _ in failures) or "flash attention"
        raise RuntimeError(
            f"Unable to configure attention backend after trying {attempted} and {NATIVE}"
        ) from error

    log.warning("Falling back to native PyTorch scaled dot-product attention")
    return NATIVE

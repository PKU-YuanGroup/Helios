import logging

import pytest

from helios.utils.attention_backend import (
    FLASH2_HUB,
    FLASH3_HUB,
    NATIVE,
    configure_attention_backend,
    is_backend_compatibility_error,
)


class FakeTransformer:
    def __init__(self, failures=None):
        self.calls = []
        self.failures = failures or {}

    def set_attention_backend(self, backend):
        self.calls.append(backend)
        failure = self.failures.get(backend)
        if failure is not None:
            raise failure


def test_hopper_prefers_flash3():
    transformer = FakeTransformer()

    selected = configure_attention_backend(transformer, compute_capability=(9, 0))

    assert selected == FLASH3_HUB
    assert transformer.calls == [FLASH3_HUB]


def test_pre_hopper_uses_flash2():
    transformer = FakeTransformer()

    selected = configure_attention_backend(transformer, compute_capability=(8, 9))

    assert selected == FLASH2_HUB
    assert transformer.calls == [FLASH2_HUB]


def test_stale_hub_wrapper_falls_back_to_flash2():
    transformer = FakeTransformer(
        {FLASH3_HUB: AttributeError("Kernel module does not define attribute path")}
    )

    selected = configure_attention_backend(transformer, compute_capability=(9, 0))

    assert selected == FLASH2_HUB
    assert transformer.calls == [FLASH3_HUB, FLASH2_HUB]


def test_unavailable_flash2_falls_back_to_native(caplog):
    transformer = FakeTransformer(
        {
            FLASH2_HUB: RuntimeError(
                "attention backend is not usable: kernels package is missing"
            )
        }
    )

    with caplog.at_level(logging.WARNING):
        selected = configure_attention_backend(transformer, compute_capability=(8, 9))

    assert selected == NATIVE
    assert transformer.calls == [FLASH2_HUB, NATIVE]
    assert "native" in caplog.text


def test_unrelated_model_errors_are_not_swallowed():
    transformer = FakeTransformer(
        {FLASH2_HUB: RuntimeError("model tensor has an invalid shape")}
    )

    with pytest.raises(RuntimeError, match="invalid shape"):
        configure_attention_backend(transformer, compute_capability=(8, 9))

    assert transformer.calls == [FLASH2_HUB]


def test_native_configuration_failure_preserves_context():
    transformer = FakeTransformer(
        {
            FLASH2_HUB: ImportError("kernels package is unavailable"),
            NATIVE: RuntimeError("native backend registration failed"),
        }
    )

    with pytest.raises(
        RuntimeError, match="Unable to configure attention backend"
    ) as exc_info:
        configure_attention_backend(transformer, compute_capability=(8, 9))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "native backend registration failed" in str(exc_info.value.__cause__)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (AttributeError("flash_attn wrapper is missing"), True),
        (RuntimeError("compute capability is unsupported"), True),
        (ImportError("optional tokenizer package is missing"), False),
        (RuntimeError("CUDA kernel launch failed"), False),
        (RuntimeError("CUDA out of memory"), False),
        (ValueError("model tensor has an invalid shape"), False),
    ],
)
def test_backend_error_classifier(error, expected):
    assert is_backend_compatibility_error(error) is expected

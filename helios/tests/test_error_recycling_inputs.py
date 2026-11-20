"""Error injection writes into the injected tensors in place; the original
clean inputs must stay untouched so the flow target is not corrupted."""

import torch

from utils.utils_helios_base import clone_error_inputs


def test_error_inputs_are_independent_copies():
    noise = torch.ones(2, 3)
    model_input = torch.zeros(2, 3)
    noise_w_error, model_input_w_error = clone_error_inputs(noise, model_input)

    mask = torch.tensor([True, False])
    noise_w_error[mask] = 99.0
    model_input_w_error[mask] = -1.0

    assert torch.equal(noise, torch.ones(2, 3))
    assert torch.equal(model_input, torch.zeros(2, 3))

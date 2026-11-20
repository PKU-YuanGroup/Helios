"""Scheduler.step() with the default return_dict=True constructs an output
with only prev_sample; the remaining fields must have defaults."""

import torch

from scheduler.scheduling_helios import HeliosScheduler, HeliosSchedulerOutput


def test_output_allows_only_prev_sample():
    sample = torch.zeros(1, 2)
    out = HeliosSchedulerOutput(prev_sample=sample)
    assert out.prev_sample is sample
    assert out.model_outputs is None
    assert out.last_sample is None
    assert out.this_order is None


def test_step_returns_output_by_default():
    scheduler = HeliosScheduler()
    scheduler.set_timesteps(4, 0)
    sample = torch.zeros(1, 3, 4, 4)
    out = scheduler.step(torch.zeros_like(sample), scheduler.timesteps[0], sample)
    assert isinstance(out, HeliosSchedulerOutput)
    assert out.prev_sample.shape == sample.shape

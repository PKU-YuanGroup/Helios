"""Regression test: repeated stage-2 sampling must not compound input corruption."""

from types import SimpleNamespace

import torch

import helios.utils.utils_helios_base as m


def test_stage2_repeated_samples_use_clean_endpoint(monkeypatch):
    # Fixed density and a deterministic corruption so the only difference
    # between samples is whether corruption was applied more than once.
    monkeypatch.setattr(
        m,
        "compute_density_for_timestep_sampling",
        lambda **kw: torch.full((kw["batch_size"],), 0.5),
    )
    monkeypatch.setattr(m, "corrupt_model_input", lambda x, **kw: x * 0.5)

    cfg = SimpleNamespace(
        weighting_scheme="uniform",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.0,
        corrupt_model_input=True,
        use_dynamic_shifting=False,
        corrupt_mode_model_input="noise",
        corrupt_mode_prob_model_input=0.9,
        is_frame_independent_corrupt_model_input=False,
        is_chunk_independent_corrupt_model_input=False,
        noise_corrupt_ratio_model_input=1 / 3,
        noise_corrupt_clean_prob_model_input=0.1,
        downsample_min_corrupt_ratio_model_input=0.9,
        downsample_max_corrupt_ratio_model_input=1.0,
    )

    class Sched:
        class C:
            num_train_timesteps = 100

        config = C()
        start_sigmas = [1.0, 0.5, 0.1]
        end_sigmas = [0.5, 0.1, 0.0]
        timesteps_per_stage = [torch.linspace(100, 50, 100), torch.linspace(50, 10, 100), torch.linspace(10, 0, 100)]
        sigmas_per_stage = [torch.linspace(1.0, 0.5, 100), torch.linspace(0.5, 0.1, 100), torch.linspace(0.1, 0.0, 100)]

    latents = torch.randn(1, 2, 4, 8, 8)
    args = SimpleNamespace(training_config=cfg)
    _, _, _, targets = m.prepare_stage2_clean_input(
        args, Sched(), latents, pyramid_stage_num=3, stage2_sample_ratios=[1, 2, 1]
    )

    # The middle stage samples twice at the same sigma. Both targets must
    # come from the same clean endpoint, not a twice-corrupted one.
    assert torch.allclose(targets[1], targets[2])

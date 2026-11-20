"""Precomputed prompt_embeds must not leave the attention mask unbound."""

import torch

from utils.utils_base import encode_prompt


def test_precomputed_prompt_embeds_return_none_mask():
    precomputed = torch.zeros(1, 4, 16)
    embeds, mask = encode_prompt(
        tokenizer=None,
        text_encoder=None,
        prompt=["x"],
        prompt_embeds=precomputed,
    )
    assert embeds is precomputed
    assert mask is None

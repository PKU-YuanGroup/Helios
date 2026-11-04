# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapter copyright 2026 Ruixing Zhang.
"""Incremental outer loop around the pinned official Helios pipeline.

Sampling, scheduler, preprocessing, latent encoding and per-chunk VAE decoding
remain upstream calls. The live prompt transition uses its three-chunk lerp.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from helios_model import HEIGHT, WIDTH, HeliosInput, Settings

logger = logging.getLogger(__name__)


@dataclass
class Run:
    generator: Any
    history: Any
    anchor: Any
    mean: Any
    std: Any
    indices: tuple
    prompt: str
    embeddings: Any
    transition: list
    index: int = 0


class HeliosBackend:
    def __init__(self, settings: Settings):
        import torch

        sys.path.insert(0, str(settings.source_path))
        from diffusers import AutoencoderKLWan
        from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
        from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
        from helios.diffusers_version.transformer_helios_diffusers import (
            HeliosTransformer3DModel,
        )

        self.run: Run | None = None
        checkpoint = str(settings.checkpoint_path)
        transformer = HeliosTransformer3DModel.from_pretrained(
            checkpoint,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        # Same attention path as infer_helios.py --disable_flash_attention.
        # The pinned Diffusers flash_hub loader requires wrapped symbols absent
        # from the published kernel. Native SDPA supports the validated B200.
        backend = "native"
        transformer.set_attention_backend(backend)
        vae = AutoencoderKLWan.from_pretrained(
            checkpoint, subfolder="vae", torch_dtype=torch.float32
        )
        scheduler = HeliosScheduler.from_pretrained(checkpoint, subfolder="scheduler")
        self.pipe = HeliosPipeline.from_pretrained(
            checkpoint,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        if settings.compile:
            torch.backends.cudnn.benchmark = True
            for component in (
                self.pipe.text_encoder,
                self.pipe.vae,
                self.pipe.transformer,
            ):
                component.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        logger.info(
            "Helios loaded; attention=%s, compile=%s", backend, settings.compile
        )
        if settings.warmup_chunks:
            self.start(
                HeliosInput(
                    world_id=0,
                    prompt="A colorful fish swimming through a coral reef.",
                    seed=42,
                    mode="text",
                    max_chunks=settings.warmup_chunks,
                )
            )
            for _ in range(settings.warmup_chunks):
                self.step(self.run.prompt)
            # Precompile the reference encoders too. Readiness must include
            # image/video shapes, not defer their first compile to Start.
            pixels = np.full((HEIGHT, WIDTH, 3), 127, dtype=np.uint8)
            for mode in ("image", "video"):
                self.start(
                    HeliosInput(
                        world_id=0,
                        prompt="A colorful fish swimming through a coral reef.",
                        seed=42,
                        mode=mode,
                        max_chunks=1,
                        image=pixels if mode == "image" else None,
                        video=np.repeat(pixels[None], 33, axis=0)
                        if mode == "video"
                        else None,
                    )
                )
                self.step(self.run.prompt)
            self.reset()

    def reset(self) -> None:
        self.run = None

    def start(self, input: HeliosInput) -> None:
        import torch

        with torch.no_grad():
            self._start(input)

    def _start(self, input: HeliosInput) -> None:
        import torch
        from diffusers.utils.torch_utils import randn_tensor
        from PIL import Image

        pipe = self.pipe
        device = pipe._execution_device
        generator = torch.Generator(device=device).manual_seed(input.seed)
        pipe._guidance_scale = 1.0
        pipe._attention_kwargs = None
        pipe._current_timestep = None
        pipe._interrupt = False
        mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(device, pipe.vae.dtype)
        )
        std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(device, pipe.vae.dtype)
        embeddings = self._encode(input.prompt)
        anchor = fake = video_latents = None
        if input.mode == "image":
            image = pipe.video_processor.preprocess(
                Image.fromarray(input.image), height=HEIGHT, width=WIDTH
            )
            anchor, fake = pipe.prepare_image_latents(
                image,
                latents_mean=mean,
                latents_std=std,
                num_latent_frames_per_chunk=9,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            sigma = (
                torch.rand(1, device=device, generator=generator) * (0.135 - 0.111)
                + 0.111
            )
            anchor = (
                sigma * randn_tensor(anchor.shape, generator=generator, device=device)
                + (1 - sigma) * anchor
            )
            sigma = (
                torch.rand(1, device=device, generator=generator) * (0.135 - 0.111)
                + 0.111
            )
            fake = (
                sigma * randn_tensor(fake.shape, generator=generator, device=device)
                + (1 - sigma) * fake
            )
        if input.mode == "video":
            video = pipe.video_processor.preprocess_video(
                [Image.fromarray(frame) for frame in input.video],
                height=HEIGHT,
                width=WIDTH,
            )
            anchor, video_latents = pipe.prepare_video_latents(
                video,
                latents_mean=mean,
                latents_std=std,
                num_latent_frames_per_chunk=9,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            sigma = (
                torch.rand(1, device=device, generator=generator) * (0.135 - 0.111)
                + 0.111
            )
            anchor = (
                sigma * randn_tensor(anchor.shape, generator=generator, device=device)
                + (1 - sigma) * anchor
            )
            noisy = []
            for chunk in video_latents.split(9, dim=2):
                sigmas = (
                    torch.rand(chunk.shape[2], device=device, generator=generator)
                    * (0.135 - 0.111)
                    + 0.111
                )
                sigmas = sigmas.view(1, 1, chunk.shape[2], 1, 1)
                noisy.append(
                    sigmas
                    * randn_tensor(chunk.shape, generator=generator, device=device)
                    + (1 - sigmas) * chunk
                )
            video_latents = torch.cat(noisy, dim=2)
        history = torch.zeros(
            1,
            pipe.transformer.config.in_channels,
            19,
            HEIGHT // pipe.vae_scale_factor_spatial,
            WIDTH // pipe.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        if fake is not None:
            history = torch.cat([history[:, :, :-1], fake], dim=2)
        if video_latents is not None:
            if video_latents.shape[2] < 19:
                history = torch.cat(
                    [history[:, :, : 19 - video_latents.shape[2]], video_latents], dim=2
                )
            else:
                history = video_latents
        prefix, long, mid, short, current = torch.arange(29).split([1, 16, 2, 1, 9])
        indices = (
            current.unsqueeze(0),
            torch.cat([prefix, short]).unsqueeze(0),
            mid.unsqueeze(0),
            long.unsqueeze(0),
        )
        self.run = Run(
            generator=generator,
            history=history,
            anchor=anchor,
            mean=mean,
            std=std,
            indices=indices,
            prompt=input.prompt,
            embeddings=embeddings,
            transition=[],
        )

    def _encode(self, prompt: str):
        embeddings, _ = self.pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=self.pipe._execution_device,
        )
        return embeddings.to(self.pipe.transformer.dtype)

    def step(self, prompt: str) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self._step(prompt)

    def _step(self, prompt: str) -> np.ndarray:
        import torch

        run, pipe = self.run, self.pipe
        if run is None:
            raise RuntimeError("No active Helios rollout")
        if prompt != run.prompt:
            target = self._encode(prompt)
            run.transition = pipe.interpolate_prompt_embeds(
                run.embeddings, target, interpolation_steps=3
            )
            run.prompt = prompt
        if run.transition:
            run.embeddings = run.transition.pop(0)
        long, mid, short = run.history[:, :, -19:].split([16, 2, 1], dim=2)
        prefix = torch.zeros_like(short) if run.anchor is None else run.anchor
        short = torch.cat([prefix, short], dim=2)
        latents = pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels,
            HEIGHT,
            WIDTH,
            33,
            dtype=torch.float32,
            device=pipe._execution_device,
            generator=run.generator,
        )
        current_i, short_i, mid_i, long_i = run.indices
        with pipe.progress_bar(total=12 if run.index == 0 else 6) as progress:
            # Upstream __call__ deliberately does not pass its generator to
            # stage2_sample. Its block-noise fallback has a separate fixed RNG;
            # forwarding run.generator here changes the native trajectory.
            latents = pipe.stage2_sample(
                latents=latents,
                pyramid_num_stages=3,
                pyramid_num_inference_steps_list=[2, 2, 2],
                prompt_embeds=run.embeddings,
                negative_prompt_embeds=None,
                guidance_scale=1.0,
                indices_hidden_states=current_i,
                indices_latents_history_short=short_i,
                indices_latents_history_mid=mid_i,
                indices_latents_history_long=long_i,
                latents_history_short=short,
                latents_history_mid=mid,
                latents_history_long=long,
                attention_kwargs=None,
                device=pipe._execution_device,
                transformer_dtype=pipe.transformer.dtype,
                use_zero_init=False,
                zero_steps=1,
                is_amplify_first_chunk=run.index == 0,
                progress_bar=progress,
            )
        if run.anchor is None:
            run.anchor = latents[:, :, :1]
        # Only these 19 frames are read by the next native step. clone releases
        # the unused storage; the persistent first-frame anchor stays separate.
        run.history = torch.cat([run.history[:, :, -19:], latents], dim=2)[
            :, :, -19:
        ].clone()
        current = latents.to(pipe.vae.dtype) / run.std + run.mean
        decoded = pipe.vae.decode(current, return_dict=False)[0]
        frames = pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
        run.index += 1
        pipe._current_timestep = None
        # Match diffusers export_to_video's truncating conversion.
        return np.ascontiguousarray((frames * 255).astype(np.uint8))

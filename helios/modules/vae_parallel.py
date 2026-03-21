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

"""
VAE Parallelism for Helios.

This module implements temporal axis parallelism for the VAE (Variational Autoencoder)
used in video generation. Unlike the transformer's sequence parallelism which uses
ring/ulysses attention patterns, VAE parallelism uses simple tensor splitting along
the temporal dimension.

The key insight is that the VAE's encode/decode operations can be split along the
temporal axis:
- Encoder: [B, C, T, H, W] -> [B, C', T', H', W'] where T' = T / temporal_scale_factor
- Decoder: [B, C', T', H', W'] -> [B, C, T*temporal_scale_factor, H*tpatial_scale_factor, W*tpatial_scale_factor]

By splitting the temporal dimension, each GPU processes a subset of frames independently.
"""

import torch
import torch.distributed as dist


class VAEParallelConfig:
    """
    Configuration for VAE parallelism.

    Args:
        vae_temporal_chunk_size: Number of latent frames per temporal chunk.
            When None, uses the full tensor on each GPU.
        vae_temporal_split_mode: How to split temporal dimension.
            - "chunk": Split into contiguous chunks (default, best for VAE locality)
            - "interleave": Interleave frames across GPUs (better for communication overlap)
    """

    def __init__(
        self,
        vae_temporal_chunk_size: int | None = None,
        vae_temporal_split_mode: str = "chunk",
    ):
        self.vae_temporal_chunk_size = vae_temporal_chunk_size
        self.vae_temporal_split_mode = vae_temporal_split_mode


def split_temporal(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    Split a [B, C, T, H, W] tensor along the temporal (T) dimension.

    Args:
        tensor: Input tensor of shape [B, C, T, H, W]
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Sub-tensor for this rank of shape [B, C, T//world_size, H, W]
    """
    if world_size == 1:
        return tensor

    B, C, T, H, W = tensor.shape
    assert T % world_size == 0, (
        f"Temporal dimension {T} is not divisible by world_size {world_size}. "
        f"Consider padding or using a different chunk size."
    )

    T_per_rank = T // world_size
    start_idx = rank * T_per_rank
    return tensor[:, :, start_idx:start_idx + T_per_rank, :, :]


def gather_temporal(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    Gather temporal chunks from all ranks back into a single tensor.

    Args:
        tensor: Input tensor of shape [B, C, T//world_size, H, W]
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Full tensor of shape [B, C, T, H, W]
    """
    if world_size == 1:
        return tensor

    B, C, T_chunk, H, W = tensor.shape
    T_full = T_chunk * world_size

    if tensor.is_cuda:
        # GPU tensor: use all_gather
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=2)  # Concatenate along temporal dim
        return output
    else:
        # CPU tensor: gather manually
        if rank == 0:
            output = [torch.empty_like(tensor) for _ in range(world_size)]
        else:
            output = None
        dist.gather(tensor, output if rank == 0 else None, dst=0)
        if rank == 0:
            return torch.cat(output, dim=2)
        return tensor


def vae_encode_parallel(
    vae,
    video: torch.Tensor,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    """
    VAE encode with temporal parallelism.

    Splits the input video along temporal dimension, encodes each chunk,
    and gathers the latent chunks.

    Args:
        vae: The VAE model (AutoencoderKLWan)
        video: Input video tensor [B, C, T, H, W]
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Encoded latent tensor (on rank 0, None on other ranks)
    """
    if world_size == 1:
        return vae.encode(video).latent_dist.sample()

    # Split video along temporal dimension
    video_chunk = split_temporal(video, rank, world_size)

    # Encode local chunk
    with torch.no_grad():
        latent_chunk = vae.encode(video_chunk).latent_dist.sample()

    # Gather all latent chunks
    latent_full = gather_temporal(latent_chunk, rank, world_size)
    return latent_full


def vae_decode_parallel(
    vae,
    latent: torch.Tensor,
    rank: int = 0,
    world_size: int = 1,
    overlap_frames: int = 0,
) -> torch.Tensor:
    """
    VAE decode with temporal parallelism.

    Splits the latent tensor along temporal dimension, decodes each chunk,
    and gathers the decoded video chunks.

    The overlap_frames parameter handles boundary artifacts by decoding
    overlapping segments and blending them. This is useful because VAE
    decoders may produce artifacts at temporal boundaries.

    Args:
        vae: The VAE model (AutoencoderKLWan)
        latent: Input latent tensor [B, C, T, H, W]
        rank: Current process rank
        world_size: Total number of processes
        overlap_frames: Number of overlapping frames at chunk boundaries.
            Each chunk includes `overlap_frames` extra frames from the next chunk
            for blending. Default is 0 (no overlap, may have boundary artifacts).

    Returns:
        Decoded video tensor (on rank 0, None on other ranks)
    """
    if world_size == 1 or overlap_frames == 0:
        # No overlap mode: simple split and decode
        if world_size == 1:
            with torch.no_grad():
                return vae.decode(latent, return_dict=False)[0]

        latent_chunk = split_temporal(latent, rank, world_size)
        with torch.no_grad():
            video_chunk = vae.decode(latent_chunk, return_dict=False)[0]
        return gather_temporal(video_chunk, rank, world_size)

    # Overlap mode: decode with overlapping chunks and blend
    # Each GPU decodes its chunk plus `overlap_frames` from the next chunk
    B, C, T, H, W = latent.shape
    T_per_rank = T // world_size

    local_start = rank * T_per_rank
    local_end = (rank + 1) * T_per_rank

    # For non-last ranks, include overlap from next chunk
    if rank < world_size - 1:
        local_end = min(local_end + overlap_frames, T)
    else:
        local_end = T

    latent_chunk = latent[:, :, local_start:local_end, :, :]

    with torch.no_grad():
        video_chunk = vae.decode(latent_chunk, return_dict=False)[0]

    # video_chunk has shape [B, C_out, T_decoded, H_out, W_out]
    # Calculate the overlap in the decoded space
    temporal_scale = video_chunk.shape[2] // (local_end - local_start)
    decoded_overlap = overlap_frames * temporal_scale

    if rank == 0:
        # First chunk: no blending needed on the left side
        output = [video_chunk]
        gathered = [torch.empty_like(video_chunk) for _ in range(world_size)]
    elif rank < world_size - 1:
        # Middle chunks: blend with previous
        # Send our chunk to previous rank and receive previous chunk
        my_output_frames = video_chunk.shape[2]
        my_local_frames = T_per_rank * temporal_scale

        # Gather all chunks first
        gathered = [torch.empty_like(video_chunk) for _ in range(world_size)]
        dist.all_gather(gathered, video_chunk)

        # Then blend: for rank > 0, blend the first `decoded_overlap` frames
        # with the last `decoded_overlap` frames of the previous chunk
        prev_chunk = gathered[rank - 1]
        curr_chunk = gathered[rank]

        # Blend overlap region (simple linear blending)
        blend_start = my_local_frames - decoded_overlap
        if blend_start >= 0:
            weights = torch.linspace(0, 1, decoded_overlap, device=curr_chunk.device)
            weights = weights.view(1, 1, -1, 1, 1)
            blended = (
                prev_chunk[:, :, -decoded_overlap:, :, :] * (1 - weights)
                + curr_chunk[:, :, :decoded_overlap, :, :] * weights
            )
            output_chunk = torch.cat([curr_chunk[:, :, :blend_start, :, :], blended], dim=2)
        else:
            output_chunk = curr_chunk
        output = [output_chunk]
    else:
        # Last chunk: no blending needed on the right side
        gathered = [torch.empty_like(video_chunk) for _ in range(world_size)]
        dist.all_gather(gathered, video_chunk)
        output = [gathered[rank]]  # Just return our chunk (already gathered)

    # Concatenate all outputs along temporal dimension
    if rank == 0:
        return torch.cat(output, dim=2)
    return None

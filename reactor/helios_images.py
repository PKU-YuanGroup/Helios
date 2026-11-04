"""Bounded CPU upload decoding with the official entrypoint's resize policy."""

import io

import av
import numpy as np
from PIL import Image, ImageOps

from helios_model import HEIGHT, WIDTH

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_PIXELS = 32_000_000
MAX_VIDEO_FRAMES = 330


class InvalidMedia(ValueError):
    """An uploaded reference exceeds a bound or cannot be decoded."""


def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InvalidMedia("Choose an image of at most 20 MiB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if (
                image.format not in ("PNG", "JPEG", "WEBP")
                or image.width * image.height > MAX_PIXELS
            ):
                raise InvalidMedia("Use PNG, JPEG or WebP with at most 32 megapixels.")
            # Match diffusers.load_image (EXIF + RGB), then the entrypoint's
            # PIL.resize before VideoProcessor.
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(image.resize((WIDTH, HEIGHT)), dtype=np.uint8).copy()
    except InvalidMedia:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidMedia(
            "The file is not a decodable PNG, JPEG or WebP image."
        ) from exc


def decode_video(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_VIDEO_BYTES:
        raise InvalidMedia("Choose a video of at most 100 MiB.")
    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.video:
                raise InvalidMedia("The file contains no video track.")
            stream = container.streams.video[0]
            if stream.width * stream.height > MAX_PIXELS:
                raise InvalidMedia("Video frames must have at most 32 megapixels.")
            frames = []
            for frame in container.decode(stream):
                if len(frames) >= MAX_VIDEO_FRAMES:
                    raise InvalidMedia("Use a reference video with 33–330 frames.")
                if frame.width * frame.height > MAX_PIXELS:
                    raise InvalidMedia("Video frames must have at most 32 megapixels.")
                image = frame.to_image().resize(
                    (WIDTH, HEIGHT), Image.Resampling.LANCZOS
                )
                frames.append(np.asarray(image, dtype=np.uint8))
    except InvalidMedia:
        raise
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise InvalidMedia("The file is not a decodable video.") from exc
    if len(frames) < 33:
        raise InvalidMedia("Video continuation requires at least 33 frames.")
    return np.stack(frames)

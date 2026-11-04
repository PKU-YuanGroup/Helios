"""Runtime-independent contracts and Helios rollout ownership."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from helios_backend import HeliosBackend

WIDTH, HEIGHT, FRAMES_PER_CHUNK = 640, 384, 33
Mode = Literal["text", "image", "video"]


@dataclass(frozen=True)
class Settings:
    """Resolved, picklable worker settings; checkpoints remain resident."""

    source_path: Path
    checkpoint_path: Path
    compile: bool = True
    warmup_chunks: int = 2


@dataclass(frozen=True)
class HeliosInput:
    """One chunk's conditioning. Anchors are CPU RGB uint8 arrays.

    ``image`` has shape (384,640,3); ``video`` has shape (T,384,640,3).
    They travel only until the model acknowledges ``world_id``. Text changes
    retain video history and interpolate embeddings across three native chunks.
    """

    world_id: int
    prompt: str
    seed: int
    mode: Mode
    max_chunks: int
    image: np.ndarray | None = None
    video: np.ndarray | None = None


@dataclass(frozen=True)
class HeliosResult:
    """A completed chunk: CPU RGB uint8 (T,384,640,3), T <= 33."""

    world_id: int
    chunk_index: int
    frames: np.ndarray
    complete: bool


class InvalidWorld(ValueError):
    """An input cannot initialize or continue the requested world."""


class RolloutExhausted(RuntimeError):
    """The requested finite video has already finished."""


class HeliosModel:
    """Own one finite video, starting fresh when the input world ID changes.

    The first successful result acknowledges the anchor; chunk_index counts
    generated chunks from zero. InvalidWorld rejects incomplete new inputs,
    and RolloutExhausted rejects a step beyond the selected video length.
    """

    def __init__(self) -> None:
        self._backend: HeliosBackend | None = None
        self._world_id: int | None = None
        self._index = 0
        self._max_chunks = 0

    def load(self, settings: Settings) -> None:
        """Load resident weights and warm up from already resolved paths."""
        from helios_backend import HeliosBackend

        self._backend = HeliosBackend(settings)

    def generate(self, input: HeliosInput) -> HeliosResult:
        """Produce one chunk, retaining native history across prompt changes."""
        if self._backend is None:
            raise RuntimeError("load() must complete before generate()")
        if input.world_id != self._world_id:
            if not input.prompt.strip() or not 1 <= input.max_chunks <= 132:
                raise InvalidWorld("A new world needs a prompt and 1–132 chunks")
            if input.mode not in ("text", "image", "video"):
                raise InvalidWorld("Unknown input mode")
            if input.mode == "image" and input.image is None:
                raise InvalidWorld("An image world needs an anchor")
            if input.mode == "video" and input.video is None:
                raise InvalidWorld("A video world needs an anchor")
            # A preparation failure is fatal to this run. No old identity may
            # survive partially initialized native state.
            self.reset()
            self._backend.start(input)
            self._world_id = input.world_id
            self._max_chunks = input.max_chunks
        if self._index >= self._max_chunks:
            raise RolloutExhausted("Start or reset to generate another video")
        frames = self._backend.step(input.prompt)
        index = self._index
        complete = index + 1 == self._max_chunks
        if complete:
            # Match the native final (F-1)//4*4+1 crop without buffering history.
            trim = (FRAMES_PER_CHUNK * self._max_chunks - 1) % 4
            if trim:
                frames = frames[:-trim]
        self._index += 1
        return HeliosResult(
            world_id=input.world_id,
            chunk_index=index,
            frames=frames,
            complete=complete,
        )

    def reset(self) -> None:
        """Drop the current video's history and identity; retain loaded weights."""
        if self._backend is not None:
            self._backend.reset()
        self._world_id = None
        self._index = 0
        self._max_chunks = 0

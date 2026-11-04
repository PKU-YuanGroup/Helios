"""Helios client state, replies, broadcasts and silent video track."""

import numpy as np
from reactor_runtime import InputState, MessageField, ModelMessage, Output, Video

from helios_model import Mode


class HeliosState(InputState):
    """Atomic start/update commands own the accepted shared selection."""

    _prompt: str = ""
    _seed: int = 42
    _mode: Mode = "text"
    _max_chunks: int = 44
    _image: np.ndarray | None = None
    _video: np.ndarray | None = None
    _reference_name: str = ""
    _world_id: int = 0
    _applied_id: int | None = None
    _active: bool = False
    _paused: bool = False
    _chunks: int = 0
    _frames: int = 0
    _complete: bool = False
    _last_seconds: float | None = None


class HeliosOutput(Output):
    """Generated silent video with adaptive Runtime pacing."""

    main_video: Video


class CommandAccepted(ModelMessage):
    """Reply to the caller after a command has changed the shared state."""

    world_id: int = MessageField(
        description="World identity after the accepted command."
    )


class StateUpdate(ModelMessage):
    """Shared state, sent to a joining viewer and broadcast after changes."""

    world_id: int = MessageField(
        description="Identity of the selected video; changes on Start and Reset."
    )
    prompt: str = MessageField(
        description="Accepted prompt for the next chunk transition."
    )
    seed: int = MessageField(description="Random seed of the selected video.")
    mode: Mode = MessageField(description="Text, image or video starting condition.")
    reference_name: str = MessageField(
        description="Uploaded filename, empty for text-only generation."
    )
    active: bool = MessageField(
        description="Whether a video has been started and not stopped."
    )
    paused: bool = MessageField(
        description="Whether generation is paused; buffered frames may still play."
    )
    limit_reached: bool = MessageField(
        description="The selected video reached its chunk limit."
    )
    completed_chunks: int = MessageField(
        description="Successfully generated chunks in this video."
    )
    output_frames: int = MessageField(
        description="Frames emitted for the current video."
    )
    max_chunks: int = MessageField(
        description="Selected finite video length in 33-frame chunks."
    )
    last_chunk_seconds: float | None = MessageField(
        description="Last generation time, null before the first result."
    )

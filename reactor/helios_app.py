"""The Helios application: accepted inputs, lifecycle and streamed output."""

import asyncio
from pathlib import Path

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    UploadedFile,
    connected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from helios_assets import prepare
from helios_images import InvalidMedia, decode_image, decode_video
from helios_model import HeliosInput, HeliosModel, HeliosResult
from helios_types import CommandAccepted, HeliosOutput, HeliosState, StateUpdate


class HeliosApp(ReactorApp):
    """Generate Helios-Distilled video from text, an image or a video, with live prompts."""

    state: HeliosState
    buffer_size = 33

    def load(self, config_path: Path | None) -> None:
        settings = prepare(config_path, get_weights_path())
        self.engine = DistributedRunner(
            HeliosModel,
            world_size=1,
            load_kwargs={"settings": settings},
            start_timeout=3600,
            call_timeout=300,
        )
        self.engine.start()

    def _snapshot(self) -> StateUpdate:
        s = self.state
        return StateUpdate(
            world_id=s._world_id,
            prompt=s._prompt,
            seed=s._seed,
            mode=s._mode,
            reference_name=s._reference_name,
            active=s._active,
            paused=s._paused,
            limit_reached=s._complete,
            completed_chunks=s._chunks,
            output_frames=s._frames,
            max_chunks=s._max_chunks,
            last_chunk_seconds=s._last_seconds,
        )

    async def _accepted(self) -> CommandAccepted:
        await self.send(self._snapshot())
        return CommandAccepted(world_id=self.state._world_id)

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._snapshot())

    @session_ended
    def on_session_ended(self) -> None:
        if self.engine.healthy:
            self.engine.reset()
        self.output.flush()

    @event(
        name="start",
        description=(
            "Start a fresh video with the selected prompt, seed and optional uploaded reference. "
            "Replaces the current video and clears buffered output. Broadcasts `state_update`. "
            "Fails with `invalid_prompt` for blank text, `invalid_reference` for a missing, "
            "unexpected, oversized or undecodable reference."
        ),
    )
    async def start(
        self,
        prompt: str = InputField(
            moderate=True,
            max_length=4096,
            description="Describe the scene, subjects and action.",
        ),
        mode: str = InputField(
            default="text",
            choices=["text", "image", "video"],
            description="Starting condition.",
        ),
        seed: int = InputField(
            default=42, ge=0, le=2147483647, description="Seed used for a fresh video."
        ),
        chunks: int = InputField(
            default=44,
            ge=1,
            le=132,
            description="33-frame chunks; the final 0–3 frames are trimmed.",
        ),
        reference: UploadedFile | None = InputField(  # noqa: B008 - Runtime schema declaration
            default=None,
            moderate=True,
            description="Image (20 MiB, 32 MP) or video (100 MiB, 33–330 frames), resized to 640×384; omit for text.",
        ),
    ) -> CommandAccepted:
        text = self._check_prompt(prompt)
        if (mode == "text") != (reference is None):
            raise CommandError(
                "invalid_reference",
                "Choose a reference for image/video, or omit it for text.",
            )
        image = video = None
        if reference is not None:
            try:
                pixels = await asyncio.to_thread(
                    decode_image if mode == "image" else decode_video, reference.data
                )
            except InvalidMedia as exc:
                raise CommandError("invalid_reference", str(exc)) from exc
            if mode == "image":
                image = pixels
            else:
                video = pixels
        s = self.state
        s._prompt, s._seed, s._mode, s._max_chunks = text, seed, mode, chunks
        s._image, s._video = image, video
        s._reference_name = reference.name if reference else ""
        self._new_world()
        return await self._accepted()

    @event(
        name="update_prompt",
        description=(
            "Update the live video without resetting its history. Uses a three-chunk text "
            "transition; queued frames keep playing. A paused video stays paused. Broadcasts "
            "`state_update`. Fails with `invalid_prompt` for blank text or `not_generating` "
            "when no unfinished video is active."
        ),
    )
    async def update_prompt(
        self,
        prompt: str = InputField(
            moderate=True, max_length=4096, description="The next scene or action."
        ),
    ) -> CommandAccepted:
        text = self._check_prompt(prompt)
        self._require_active()
        self.state._prompt = text
        return await self._accepted()

    @event(
        name="pause",
        description="Pause generation after the current chunk. Broadcasts `state_update`. Fails with `not_generating` without an unfinished video.",
    )
    async def pause(self) -> CommandAccepted:
        self._require_active()
        self.state._paused = True
        return await self._accepted()

    @event(
        name="resume",
        description="Continue the paused video's history. Broadcasts `state_update`. Fails with `not_generating` without an unfinished video.",
    )
    async def resume(self) -> CommandAccepted:
        self._require_active()
        self.state._paused = False
        return await self._accepted()

    @event(
        name="reset",
        description="Restart from the accepted reference and current prompt, optionally with a new seed. Broadcasts `state_update`. Fails with `no_video` before Start.",
    )
    async def reset(
        self,
        seed: int | None = InputField(
            default=None,
            ge=0,
            le=2147483647,
            description="New random seed; omit to reuse the accepted seed.",
        ),
    ) -> CommandAccepted:
        if not self.state._prompt:
            raise CommandError("no_video", "Start a video before resetting.")
        if seed is not None:
            self.state._seed = seed
        self._new_world()
        return await self._accepted()

    @event(
        name="stop",
        description="End generation and clear playback while retaining inputs for Reset. Broadcasts `state_update`.",
    )
    async def stop_video(self) -> CommandAccepted:
        self.state._active = False
        self.state._paused = False
        self.engine.reset()
        self.output.flush()
        return await self._accepted()

    def _new_world(self) -> None:
        s = self.state
        s._world_id += 1
        s._applied_id = None
        s._chunks = s._frames = 0
        s._complete = s._paused = False
        s._active = True
        s._last_seconds = None
        self.output.flush()

    @staticmethod
    def _check_prompt(prompt: str) -> str:
        if not prompt.strip():
            raise CommandError("invalid_prompt", "Enter a scene prompt.")
        return prompt.strip()

    def _require_active(self) -> None:
        if not self.state._active or self.state._complete:
            raise CommandError("not_generating", "Start or reset a video first.")

    async def process_input(self) -> HeliosInput:
        s = self.state
        if not s._active or s._complete:
            raise ApplicationError("Waiting for Start or Reset")
        if s._paused:
            raise ApplicationError("Paused")
        new = s._applied_id != s._world_id
        return HeliosInput(
            world_id=s._world_id,
            prompt=s._prompt,
            seed=s._seed,
            mode=s._mode,
            max_chunks=s._max_chunks,
            image=s._image if new else None,
            video=s._video if new else None,
        )

    def generate(self, input: HeliosInput) -> HeliosResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> HeliosOutput | None:
        if outcome.error is not None:
            # Native partial-step/CUDA failures have no demonstrated recovery;
            # fail the session instead of resuming an uncertain history.
            raise outcome.error
        result: HeliosResult = outcome.result
        s = self.state
        if result.world_id != s._world_id or not s._active:
            return None
        s._applied_id = result.world_id
        s._chunks = result.chunk_index + 1
        s._frames += len(result.frames)
        s._complete = result.complete
        s._last_seconds = outcome.elapsed
        await self.send(self._snapshot())
        return HeliosOutput(
            main_video=TrackPayload(
                result.frames,
                metadata=[
                    {"world_id": result.world_id, "chunk_index": result.chunk_index}
                ]
                * len(result.frames),
            )
        )

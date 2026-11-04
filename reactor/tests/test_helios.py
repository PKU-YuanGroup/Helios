"""Drive actual application hooks and the model half with CPU fake compute."""

import asyncio
import io
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

from helios_app import HeliosApp
from helios_images import InvalidMedia, decode_image, decode_video
from helios_model import (
    HeliosInput,
    HeliosModel,
    HeliosResult,
    InvalidWorld,
    RolloutExhausted,
)
from helios_types import HeliosState


class Compute:
    def __init__(self):
        self.starts, self.prompts, self.resets = [], [], 0
        self.failure = None

    def reset(self):
        self.resets += 1

    def start(self, value):
        if self.failure:
            raise self.failure
        self.starts.append(value)

    def step(self, prompt):
        if self.failure:
            raise self.failure
        self.prompts.append(prompt)
        return np.zeros((33, 8, 8, 3), dtype=np.uint8)


def make_app():
    app = HeliosApp()
    app.state = HeliosState()
    app.engine = HeliosModel()
    app.engine._backend = Compute()
    app.output = SimpleNamespace(flush=Mock())
    app.send = AsyncMock()
    return app


async def start(app, **kwargs):
    return await app.start(
        **{
            "prompt": "A fish swims.",
            "mode": "text",
            "seed": 42,
            "chunks": 10,
            "reference": None,
        }
        | kwargs
    )


async def step(app):
    value = await app.process_input()
    try:
        outcome = StepOutcome(result=app.generate(value), elapsed=0.25)
    except RuntimeError as exc:
        outcome = StepOutcome(error=exc, elapsed=0.25)
    return await app.process_output(outcome)


def test_refused_step_never_runs_model():
    async def run():
        app = make_app()
        with pytest.raises(ApplicationError):
            await step(app)
        await start(app)
        await app.pause()
        with pytest.raises(ApplicationError):
            await step(app)
        assert app.engine._backend.starts == []

    asyncio.run(run())


def test_generate_reads_only_passed_input():
    app = make_app()
    app.state = None
    value = HeliosInput(
        world_id=9, prompt="A forest.", seed=7, mode="text", max_chunks=2
    )
    result = app.generate(value)
    assert result.world_id == 9
    assert app.engine._backend.starts == [value]


def test_live_prompts_and_pause_preserve_history():
    async def run():
        app = make_app()
        await start(app)
        for _ in range(3):
            await step(app)
        await app.update_prompt("A fish turns left.")
        for _ in range(3):
            await step(app)
        await app.pause()
        await app.update_prompt("A fish swims upward.")
        assert app.state._paused
        await app.resume()
        for _ in range(4):
            await step(app)
        assert app.state._chunks == 10 and app.state._frames == 329
        assert app.state._complete and app.state._world_id == 1
        assert len(app.engine._backend.starts) == 1
        assert (
            app.engine._backend.prompts
            == ["A fish swims."] * 3
            + ["A fish turns left."] * 3
            + ["A fish swims upward."] * 4
        )
        with pytest.raises(ApplicationError):
            await step(app)

    asyncio.run(run())


def test_anchor_is_sent_until_result_acknowledges_and_again_on_reset():
    async def run():
        app = make_app()
        await start(app)
        app.state._mode = "image"
        app.state._image = np.zeros((384, 640, 3), np.uint8)
        first = await app.process_input()
        assert first.image is app.state._image
        assert (await app.process_input()).image is first.image
        await step(app)
        assert (await app.process_input()).image is None
        await app.reset(seed=None)
        assert (await app.process_input()).image is first.image
        await step(app)
        assert app.state._chunks == 1 and app.state._world_id == 2
        assert app.engine._backend.starts[-1].seed == 42

    asyncio.run(run())


@pytest.mark.parametrize("complete", [False, True])
def test_old_world_output_cannot_acknowledge_or_complete_new_world(complete):
    async def run():
        app = make_app()
        await start(app)
        await app.reset(seed=None)
        calls = app.send.await_count
        result = HeliosResult(
            world_id=1,
            chunk_index=9,
            frames=np.zeros((33, 8, 8, 3), np.uint8),
            complete=complete,
        )
        assert await app.process_output(StepOutcome(result=result, elapsed=3)) is None
        assert (
            app.state._chunks == 0
            and app.state._applied_id is None
            and not app.state._complete
        )
        assert app.send.await_count == calls

    asyncio.run(run())


def test_stop_suppresses_inflight_output_and_retains_reference():
    async def run():
        app = make_app()
        await start(app)
        result = app.generate(await app.process_input())
        await app.stop_video()
        assert await app.process_output(StepOutcome(result=result, elapsed=1)) is None
        assert not app.state._active and app.state._prompt
        await app.reset(seed=None)
        await step(app)
        assert app.state._active and app.state._chunks == 1

    asyncio.run(run())


def test_model_failure_propagates_without_completion_or_success():
    async def run():
        app = make_app()
        await start(app)
        await step(app)
        app.send.reset_mock()
        app.engine._backend.failure = RuntimeError("compute failed")
        with pytest.raises(RuntimeError, match="compute failed"):
            await step(app)
        assert app.state._chunks == 1 and not app.state._complete
        app.send.assert_not_awaited()

    asyncio.run(run())


def test_failed_preparation_does_not_retain_old_world_identity():
    model = HeliosModel()
    model._backend = Compute()
    value = HeliosInput(
        world_id=1, prompt="A fish.", seed=42, mode="text", max_chunks=4
    )
    model.generate(value)
    model._backend.failure = ValueError("encoding failed")
    with pytest.raises(ValueError, match="encoding failed"):
        model.generate(replace(value, world_id=2))
    assert model._world_id is None
    model._backend.failure = None
    assert model.generate(value).chunk_index == 0


@pytest.mark.parametrize("chunks", [1, 2, 3, 4, 10, 44])
def test_final_frame_count_matches_native_rounding(chunks):
    model = HeliosModel()
    model._backend = Compute()
    value = HeliosInput(
        world_id=1, prompt="A fish.", seed=42, mode="text", max_chunks=chunks
    )
    results = [model.generate(value) for _ in range(chunks)]
    assert sum(len(r.frames) for r in results) == (chunks * 33 - 1) // 4 * 4 + 1
    assert results[-1].complete
    with pytest.raises(RolloutExhausted):
        model.generate(value)


@pytest.mark.parametrize("mode", ["image", "video"])
def test_missing_anchor_is_model_error(mode):
    model = HeliosModel()
    model._backend = Compute()
    with pytest.raises(InvalidWorld):
        model.generate(
            HeliosInput(world_id=1, prompt="A fish.", seed=42, mode=mode, max_chunks=4)
        )
    assert not model._backend.starts


@pytest.mark.parametrize("command", ["pause", "resume", "update_prompt"])
def test_inactive_commands_refuse_without_broadcast(command):
    async def run():
        app = make_app()
        with pytest.raises(CommandError) as err:
            if command == "update_prompt":
                await app.update_prompt("A bird.")
            else:
                await getattr(app, command)()
        assert err.value.code == "not_generating"
        app.send.assert_not_awaited()

    asyncio.run(run())


def test_invalid_replacement_preserves_accepted_video_and_pause():
    async def run():
        app = make_app()
        await start(app)
        await step(app)
        await app.pause()
        previous = app._snapshot()
        app.send.reset_mock()
        bad = UploadedFile(name="bad.png", mime_type="image/png", data=b"not a picture")
        with pytest.raises(CommandError) as err:
            await start(app, mode="image", reference=bad, prompt="Another scene")
        assert err.value.code == "invalid_reference"
        assert app._snapshot() == previous
        app.send.assert_not_awaited()
        with pytest.raises(CommandError):
            await app.update_prompt("   ")
        assert app._snapshot() == previous

    asyncio.run(run())


def test_join_snapshot_is_addressed_and_session_state_is_fresh():
    async def run():
        app = make_app()
        await start(app, seed=123)
        await app.pause()
        app.send.reset_mock()
        client = SimpleNamespace(send=AsyncMock())
        await app.on_connected(client)
        client.send.assert_awaited_once()
        assert client.send.call_args.args[0].paused
        app.send.assert_not_awaited()
        assert HeliosState()._seed == 42 and HeliosState()._image is None
        assert not HeliosState()._active

    asyncio.run(run())


@pytest.mark.parametrize("data", [b"", b"bad", b"x" * (20 * 1024 * 1024 + 1)])
def test_bad_images_are_bounded(data):
    with pytest.raises(InvalidMedia):
        decode_image(data)


def test_image_resize_matches_native_rgb_pil_resize():
    image = Image.new("RGB", (701, 433), (10, 30, 80))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    assert np.array_equal(
        decode_image(buf.getvalue()), np.asarray(image.resize((640, 384)))
    )


@pytest.mark.parametrize("data", [b"", b"bad"])
def test_bad_video_is_refused(data):
    with pytest.raises(InvalidMedia):
        decode_video(data)


def test_runtime_loop_drives_real_hooks_and_recreates_session_state():
    from reactor_runtime.core.model import ClientConnected, SessionStarted
    from reactor_runtime.core.values import ConnId

    async def run():
        app = HeliosApp()
        app.engine = HeliosModel()
        app.engine._backend = Compute()
        app._on_loop_ready()
        messages, media = [], []
        app.bind_output(
            broadcast=messages.append,
            addressed=lambda *args: None,
            media=lambda chunk: None,
        )

        async def emit(output, **kwargs):
            media.append(output)
            await asyncio.sleep(0)

        app.emit = emit
        await app._dispatch_reactor_event(SessionStarted("test"))
        await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
        task = asyncio.create_task(app.run())
        try:
            await asyncio.sleep(0.02)
            assert not app.engine._backend.starts
            await start(app, chunks=2)
            for _ in range(100):
                if app.state._complete:
                    break
                await asyncio.sleep(0.01)
            assert app.state._complete and app.state._frames == 65
            assert len(media) == 2
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await app._dispatch_reactor_event(SessionStarted("second"))
        assert not app.state._active and app.state._seed == 42

    asyncio.run(run())


def test_model_imports_without_runtime_or_torch():
    script = """
import importlib.abc, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'reactor_runtime', 'torch'}:
            raise ImportError('Forbidden model import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
import helios_model, helios_backend
assert helios_model.HeliosModel()._world_id is None
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
    )


def test_service_stop_is_not_overridden_by_video_command():
    from reactor_runtime import ReactorApp

    assert HeliosApp.stop is ReactorApp.stop


def test_reset_can_select_seed_and_keeps_reference():
    async def run():
        app = make_app()
        await start(app, seed=17)
        await step(app)
        await app.reset(seed=99)
        value = await app.process_input()
        assert value.seed == 99 and value.world_id == 2
        assert app.state._chunks == 0
        await step(app)
        await app.reset(seed=None)
        assert (await app.process_input()).seed == 99

    asyncio.run(run())


@pytest.mark.parametrize("decoder", [decode_image, decode_video])
def test_decoder_refusal_does_not_expose_library_error(decoder):
    with pytest.raises(InvalidMedia, match="The file is not a decodable") as error:
        decoder(b"not media")
    assert "BytesIO" not in str(error.value)

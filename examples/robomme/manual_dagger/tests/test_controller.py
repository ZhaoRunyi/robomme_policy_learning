"""tests / test_controller: shared behavior owned by this layer."""

from __future__ import annotations

import asyncio
import time
import numpy as np
from manual_dagger.controller import ManualDaggerController
from manual_dagger.controller import FakeBackendFactory
from manual_dagger.state import Command, CommandRequest, EpisodeSpec, Phase, command_from_key
from manual_dagger.state import CollectionProgress
from manual_dagger.state import StepResult
import json
from pathlib import Path
import threading
from manual_dagger.state import ControlArtifact, ControlCache
from manual_dagger.state import (
    PreparationCancelled,
)


def test_partial_expert_pauses_and_allows_repeated_takeovers() -> None:
    async def scenario() -> None:
        async def ignore(*_values):
            pass

        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=100)
        controller = ManualDaggerController(
            factory, [EpisodeSpec("BinFill", 0)], ignore, ignore, default_fps=30,
        )
        await controller.start()
        try:
            await _wait_phase(controller, Phase.READY)
            episode = next(item for item in factory.episodes if item.role == "enabled")
            calls = []

            def intervene(_abort):
                calls.append(episode.step)
                episode.step += 2
                return StepResult("subtask_completed", episode.step)

            episode.intervene = intervene
            await controller.submit(CommandRequest(Command.START, "test", time.monotonic_ns()))
            await _wait_phase(controller, Phase.POLICY)
            for expected in (1, 2):
                await controller.submit(CommandRequest(Command.INTERVENE, "test", time.monotonic_ns()))
                await _wait_until(lambda: len(calls) == expected)
                await _wait_phase(controller, Phase.PAUSED)
                previous_actions = episode.policy_actions
                await asyncio.sleep(0.07)
                assert episode.policy_actions == previous_actions
                assert not controller._takeover_event.is_set()
            await controller.submit(CommandRequest(Command.PAUSE, "test", time.monotonic_ns()))
            await _wait_until(lambda: episode.policy_actions > previous_actions)
        finally:
            await controller.stop()

    asyncio.run(scenario())


async def _wait_phase(controller: ManualDaggerController, phase: Phase) -> None:
    deadline = time.monotonic() + 3.0
    while controller.state.phase != phase:
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {phase}; got {controller.state.phase}")
        await asyncio.sleep(0.005)


async def _wait_until(predicate) -> None:
    deadline = time.monotonic() + 3.0
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


def test_intervention_latches_during_inference_before_policy_action() -> None:
    async def scenario() -> None:
        states = []
        frames: list[np.ndarray] = []
        factory = FakeBackendFactory(inference_delay=0.15, terminal_step=10)

        async def publish_state(state):
            states.append(state)

        async def publish_frame(stream, frame, step):
            del stream, step
            frames.append(frame)

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("BinFill", 0)],
            publish_state,
            publish_frame,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await controller.submit(CommandRequest(Command.START, "start", time.monotonic_ns()))
        await _wait_phase(controller, Phase.POLICY)
        await asyncio.sleep(0.05)
        started = time.monotonic()
        accepted, _ = await controller.submit(CommandRequest(Command.INTERVENE, "intervene", time.monotonic_ns()))
        submit_latency = time.monotonic() - started
        assert accepted
        assert submit_latency < 0.03
        await _wait_phase(controller, Phase.FINISHED)
        episode = next(item for item in factory.episodes if item.role == "enabled")
        assert episode.policy_actions == 0
        assert controller.state.intervened
        assert controller.state.first_expert_action_latency_ms is not None
        await controller.submit(CommandRequest(Command.DISCARD, "discard", time.monotonic_ns()))
        await _wait_phase(controller, Phase.COMPLETE)
        assert episode.closed_with is False
        await controller.stop()

    asyncio.run(scenario())


def test_abort_discards_without_completion_prompt() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.1, terminal_step=10)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("RouteStick", 2)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await controller.submit(CommandRequest(Command.START, "start", time.monotonic_ns()))
        await _wait_phase(controller, Phase.POLICY)
        accepted, _ = await controller.submit(CommandRequest(Command.ABORT, "abort", time.monotonic_ns()))
        assert accepted
        await _wait_phase(controller, Phase.COMPLETE)
        assert factory.episodes[0].closed_with is False
        await controller.stop()

    asyncio.run(scenario())


def test_next_discards_unfinished_pair_and_advances_immediately() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=50)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("BinFill", 0), EpisodeSpec("BinFill", 1)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        assert command_from_key("N") == Command.NEXT_EPISODE

        await controller.submit(CommandRequest(Command.START, "start", time.monotonic_ns()))
        await _wait_until(lambda: controller.state.step >= 2)
        accepted, _ = await controller.submit(
            CommandRequest(Command.NEXT_EPISODE, "next", time.monotonic_ns())
        )
        assert accepted
        await _wait_until(
            lambda: controller.state.phase == Phase.READY and controller.state.episode == 1
        )

        first_enabled = next(
            item for item in factory.episodes if item.role == "enabled" and item.episode == 0
        )
        first_control = next(
            item for item in factory.episodes if item.role == "control" and item.episode == 0
        )
        second_enabled = next(
            item for item in factory.episodes if item.role == "enabled" and item.episode == 1
        )
        assert first_enabled.closed_with is False
        assert first_control.closed_with is False
        assert second_enabled.closed_with is None

        await controller.submit(
            CommandRequest(Command.NEXT_EPISODE, "next-ready", time.monotonic_ns())
        )
        await _wait_phase(controller, Phase.COMPLETE)
        await controller.stop()

    asyncio.run(scenario())


def test_abort_during_takeover_does_not_discard_next_episode() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.1, terminal_step=10)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("BinFill", 0), EpisodeSpec("BinFill", 1)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await controller.submit(CommandRequest(Command.START, "start", time.monotonic_ns()))
        await _wait_phase(controller, Phase.POLICY)
        await controller.submit(CommandRequest(Command.INTERVENE, "intervene", time.monotonic_ns()))
        await controller.submit(CommandRequest(Command.ABORT, "abort", time.monotonic_ns()))
        await _wait_phase(controller, Phase.READY)
        assert controller.state.episode == 1
        first_enabled = next(
            item for item in factory.episodes if item.role == "enabled" and item.episode == 0
        )
        second_enabled = next(
            item for item in factory.episodes if item.role == "enabled" and item.episode == 1
        )
        assert first_enabled.closed_with is False
        assert second_enabled.closed_with is None
        await asyncio.sleep(0.05)
        assert controller.state.phase == Phase.READY
        await controller.submit(CommandRequest(Command.ABORT, "abort-next", time.monotonic_ns()))
        await _wait_phase(controller, Phase.COMPLETE)
        await controller.stop()

    asyncio.run(scenario())


def test_pause_only_stops_enabled_while_control_continues() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=40)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("MoveCube", 0)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await controller.submit(CommandRequest(Command.START, "start", time.monotonic_ns()))
        await _wait_phase(controller, Phase.POLICY)
        await asyncio.sleep(0.08)
        await controller.submit(CommandRequest(Command.PAUSE, "pause", time.monotonic_ns()))
        await _wait_phase(controller, Phase.PAUSED)
        enabled_step = controller.state.step
        control_step = controller.state.control_step
        await asyncio.sleep(0.12)
        assert controller.state.step == enabled_step
        assert controller.state.control_step > control_step
        await controller.submit(CommandRequest(Command.PAUSE, "resume", time.monotonic_ns()))
        await _wait_phase(controller, Phase.POLICY)
        await asyncio.sleep(0.08)
        assert controller.state.step > enabled_step
        await controller.submit(CommandRequest(Command.ABORT, "abort", time.monotonic_ns()))
        await _wait_phase(controller, Phase.COMPLETE)
        await controller.stop()

    asyncio.run(scenario())


def test_control_starts_before_enabled_start_command() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=40)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("VideoPlaceOrder", 0)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await _wait_until(lambda: controller.state.control_step >= 2)
        enabled = next(item for item in factory.episodes if item.role == "enabled")
        control = next(item for item in factory.episodes if item.role == "control")
        assert controller.state.phase == Phase.READY
        assert controller.state.control_phase == Phase.POLICY.value
        assert enabled.policy_actions == 0
        assert control.policy_actions >= 2
        await controller.submit(
            CommandRequest(Command.ABORT, "abort", time.monotonic_ns())
        )
        await _wait_phase(controller, Phase.COMPLETE)
        assert enabled.closed_with is False
        assert control.closed_with is False
        await controller.stop()

    asyncio.run(scenario())


def test_start_uses_prefetched_first_chunk_without_another_inference_delay() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.15, terminal_step=20)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("VideoPlaceOrder", 0)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await _wait_until(
            lambda: controller.state.policy_prefetch_ready
            and controller.state.control_prefetch_ready
        )
        started = time.monotonic()
        await controller.submit(CommandRequest(Command.START, "start", time.monotonic_ns()))
        await _wait_until(lambda: controller.state.step >= 1)
        assert time.monotonic() - started < 0.08
        await controller.submit(CommandRequest(Command.ABORT, "abort", time.monotonic_ns()))
        await _wait_phase(controller, Phase.COMPLETE)
        await controller.stop()

    asyncio.run(scenario())


def test_reset_recreates_only_enabled_on_the_same_seed() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=35)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("BinFill", 4, 1)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        generation = controller.state.enabled_generation
        await controller.submit(CommandRequest(Command.START, "start", time.monotonic_ns()))
        await _wait_until(lambda: controller.state.step >= 2)
        control = next(item for item in factory.episodes if item.role == "control")
        control_step = control.step
        accepted, _ = await controller.submit(
            CommandRequest(Command.RESET, "reset", time.monotonic_ns())
        )
        assert accepted
        await _wait_until(
            lambda: controller.state.phase == Phase.READY
            and controller.state.enabled_generation > generation
        )
        enabled = [item for item in factory.episodes if item.role == "enabled"]
        assert len(enabled) == 2
        assert enabled[0].closed_with is False
        assert enabled[0].seed == enabled[1].seed == 10_004
        assert not enabled[0].replayed_initialization
        assert enabled[1].replayed_initialization
        assert control.closed_with is None
        await _wait_until(lambda: control.step > control_step)
        assert control.step > control_step
        await controller.submit(CommandRequest(Command.ABORT, "abort", time.monotonic_ns()))
        await _wait_phase(controller, Phase.COMPLETE)
        await controller.stop()

    asyncio.run(scenario())


def test_discarded_enabled_can_keep_completed_control() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=2)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("MoveCube", 0, 1)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await controller.submit(CommandRequest(Command.START, "start", time.monotonic_ns()))
        await _wait_phase(controller, Phase.FINISHED)
        await controller.submit(CommandRequest(Command.DISCARD, "discard", time.monotonic_ns()))
        await _wait_phase(controller, Phase.COMPLETE)
        enabled = next(item for item in factory.episodes if item.role == "enabled")
        control = next(item for item in factory.episodes if item.role == "control")
        assert enabled.closed_with is False
        assert control.closed_with is True
        await controller.stop()

    asyncio.run(scenario())


def test_collection_target_and_manual_seed_drive_next_pair(tmp_path) -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=1)
        progress = CollectionProgress(
            tmp_path,
            ("TaskA", "TaskB"),
            {"TaskA": 100, "TaskB": 100},
            target_per_task=1,
        )

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("TaskA", 0)],
            ignore,
            ignore,
            default_fps=30,
            collection=progress,
        )
        await controller.start()
        try:
            await _wait_until(
                lambda: controller.state.phase == Phase.READY
                and controller.state.task == "TaskA"
            )
            await controller.submit(
                CommandRequest(Command.START, "start", time.monotonic_ns())
            )
            await _wait_phase(controller, Phase.FINISHED)
            await controller.submit(
                CommandRequest(Command.KEEP, "keep", time.monotonic_ns())
            )
            await _wait_until(
                lambda: controller.state.phase == Phase.READY
                and controller.state.task == "TaskB"
            )
            assert controller.state.progress_counts["TaskA"] == 1
            accepted, _ = await controller.submit(
                CommandRequest(
                    Command.SELECT_NEXT,
                    "select",
                    time.monotonic_ns(),
                    {"task": "TaskA", "seed": "10007"},
                )
            )
            assert accepted
            await controller.submit(
                CommandRequest(Command.ABORT, "skip", time.monotonic_ns())
            )
            await _wait_until(
                lambda: controller.state.phase == Phase.READY
                and controller.state.task == "TaskA"
                and controller.state.episode == 7
            )
            await _wait_until(
                lambda: controller.state.policy_prefetch_ready
                and controller.state.control_prefetch_ready
            )
        finally:
            await controller.stop()

    asyncio.run(scenario())


def test_control_frame_schedules_same_episode_replay_and_takeover() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=30)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("BinFill", 3)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await controller.submit(
            CommandRequest(Command.START, "start", time.monotonic_ns())
        )
        await _wait_until(lambda: controller.state.control_step >= 4)
        generation = controller.state.enabled_generation
        accepted, _ = await controller.submit(
            CommandRequest(
                Command.SCHEDULE_INTERVENTION,
                "schedule",
                time.monotonic_ns(),
                {"step": 3},
            )
        )
        assert accepted
        await _wait_until(
            lambda: controller.state.enabled_generation > generation
            and controller.state.phase in {Phase.POLICY, Phase.INTERVENING, Phase.FINISHED}
        )
        await _wait_phase(controller, Phase.FINISHED)
        enabled = [item for item in factory.episodes if item.role == "enabled"]
        control = [item for item in factory.episodes if item.role == "control"]
        assert len(enabled) == 2
        assert len(control) == 1
        assert enabled[0].closed_with is False
        assert enabled[1].replayed_initialization
        assert enabled[1].policy_actions == 3
        assert controller.state.scheduled_intervention_fired
        assert controller.state.scheduled_intervention_step == 3
        await asyncio.sleep(0.05)
        assert controller.state.phase == Phase.FINISHED
        assert enabled[1].closed_with is None
        assert "explicit operator review required" in controller.state.message
        await controller.submit(
            CommandRequest(Command.DISCARD, "discard", time.monotonic_ns())
        )
        await _wait_phase(controller, Phase.COMPLETE)
        await controller.stop()

    asyncio.run(scenario())


def test_manual_intervention_overrides_later_scheduled_takeover() -> None:
    async def scenario() -> None:
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=40)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [EpisodeSpec("MoveCube", 2)],
            ignore,
            ignore,
            default_fps=30,
        )
        await controller.start()
        await _wait_phase(controller, Phase.READY)
        await controller.submit(
            CommandRequest(Command.START, "start", time.monotonic_ns())
        )
        await _wait_until(lambda: controller.state.control_step >= 12)
        generation = controller.state.enabled_generation
        await controller.submit(
            CommandRequest(
                Command.SCHEDULE_INTERVENTION,
                "schedule",
                time.monotonic_ns(),
                {"step": 12},
            )
        )
        await _wait_until(
            lambda: controller.state.enabled_generation > generation
            and controller.state.phase == Phase.POLICY
            and controller.state.step >= 2
        )
        accepted, _ = await controller.submit(
            CommandRequest(Command.INTERVENE, "manual", time.monotonic_ns())
        )
        assert accepted
        await _wait_phase(controller, Phase.FINISHED)
        replay = [item for item in factory.episodes if item.role == "enabled"][-1]
        assert replay.policy_actions < 12
        assert controller.state.intervened
        assert controller.state.scheduled_intervention_step is None
        assert not controller.state.scheduled_intervention_fired
        await controller.submit(
            CommandRequest(Command.DISCARD, "discard", time.monotonic_ns())
        )
        await _wait_phase(controller, Phase.COMPLETE)
        await controller.stop()

    asyncio.run(scenario())


async def control_cache__wait_until(predicate) -> None:
    deadline = time.monotonic() + 3.0
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


class MemoryControlCache:
    def __init__(self, specs: tuple[EpisodeSpec, ...], *, ready: bool) -> None:
        self.specs = specs
        self.publish_on_failure_lookup = not ready
        self.artifacts: dict[tuple[str, int], ControlArtifact] = {}
        if ready:
            for spec in specs:
                self._record(spec)

    def _record(self, spec: EpisodeSpec) -> ControlArtifact:
        artifact = ControlArtifact(
            {
                "role": "control",
                "kept": True,
                "task": spec.task,
                "episode": spec.episode,
                "seed": 10_000 + spec.episode,
                "steps": 3,
                "outcome": "fail",
                "video_frames": 3,
                "video_fps": 30,
            },
            Path("/tmp/fake-control.mp4"),
        )
        self.artifacts[(spec.task, spec.episode)] = artifact
        return artifact

    def lookup(self, task: str, episode: int, seed: int | None = None):
        artifact = self.artifacts.get((task, episode))
        if artifact is None or (
            seed is not None and artifact.metadata["seed"] != seed
        ):
            return None
        return artifact

    def lookup_spec(self, spec: EpisodeSpec):
        return self.lookup(spec.task, spec.episode)

    def lookup_failure(self, task: str, episode: int, seed: int | None = None):
        return self.lookup(task, episode, seed)

    def lookup_failure_spec(self, spec: EpisodeSpec):
        artifact = self.lookup_spec(spec)
        if artifact is None and self.publish_on_failure_lookup:
            self.publish_on_failure_lookup = False
            artifact = self._record(spec)
        return artifact

    def screened(self, spec: EpisodeSpec) -> bool:
        return self.lookup_spec(spec) is not None

    def failure_episodes_by_task(self):
        tasks = dict.fromkeys(spec.task for spec in self.specs)
        return {
            task: tuple(
                sorted(episode for item_task, episode in self.artifacts if item_task == task)
            )
            for task in tasks
        }

    @staticmethod
    def is_failure_outcome(outcome):
        return outcome != "success"

    def wait_for_failure(self, spec: EpisodeSpec, timeout: float = 30.0):
        del timeout
        return self._record(spec)

    def record_success(self, spec: EpisodeSpec, seed: int) -> None:
        del spec, seed

    def abandon(self, spec: EpisodeSpec, reason: str) -> None:
        del spec, reason

    def prune_unreferenced_successes(self) -> int:
        return 0

    def ready_count(self) -> int:
        return len(self.artifacts)


def test_cached_control_avoids_live_control_environment() -> None:
    async def scenario() -> None:
        spec = EpisodeSpec("MoveCube", 0)
        cache = MemoryControlCache((spec,), ready=True)
        factory = FakeBackendFactory(inference_delay=0.01, terminal_step=2)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [spec],
            ignore,
            ignore,
            default_fps=30,
            control_cache=cache,
        )
        await controller.start()
        await control_cache__wait_until(lambda: controller.state.phase == Phase.READY)
        assert controller.state.control_cached
        assert controller.state.control_source == "CACHE"
        assert [item.role for item in factory.episodes] == ["enabled"]
        await controller.submit(
            CommandRequest(Command.START, "start", time.monotonic_ns())
        )
        await control_cache__wait_until(lambda: controller.state.phase == Phase.FINISHED)
        await controller.submit(
            CommandRequest(Command.DISCARD, "discard", time.monotonic_ns())
        )
        await control_cache__wait_until(lambda: controller.state.phase == Phase.COMPLETE)
        await asyncio.wait_for(controller.stop(), timeout=1.0)

    asyncio.run(scenario())


def test_cached_control_is_published_while_enabled_is_still_preparing() -> None:
    class SlowEnabledFactory(FakeBackendFactory):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def prepare(self, spec, *args, **kwargs):
            role = kwargs.get("role", args[-1])
            if role == "enabled":
                self.entered.set()
                assert self.release.wait(timeout=3.0)
            return super().prepare(spec, *args, **kwargs)

    async def scenario() -> None:
        spec = EpisodeSpec("MoveCube", 3)
        cache = MemoryControlCache((spec,), ready=True)
        factory = SlowEnabledFactory()
        published = []

        async def publish_state(state):
            published.append(state)

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [spec],
            publish_state,
            ignore,
            control_cache=cache,
        )
        await controller.start()
        await control_cache__wait_until(factory.entered.is_set)
        assert controller.state.phase == Phase.PREPARING
        assert controller.state.seed == 10_003
        assert controller.state.control_cached
        assert controller.state.control_video_frames == 3
        assert any(state["control_cached"] for state in published)
        accepted, _ = await controller.submit(
            CommandRequest(
                Command.SCHEDULE_INTERVENTION,
                "schedule-while-preparing",
                time.monotonic_ns(),
                {"step": 1},
            )
        )
        assert accepted
        assert controller.state.scheduled_intervention_pending
        factory.release.set()
        await control_cache__wait_until(lambda: controller.state.phase == Phase.FINISHED)
        assert controller.state.scheduled_intervention_fired
        await asyncio.wait_for(controller.stop(), timeout=1.0)

    asyncio.run(scenario())


def test_select_next_cancels_active_preparation_and_switches_task(tmp_path) -> None:
    class CancellableFactory(FakeBackendFactory):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()

        def prepare(self, spec, *args, **kwargs):
            role = kwargs.get("role", args[-1])
            abort_event = args[1]
            if role == "enabled" and spec.task == "MoveCube":
                self.entered.set()
                while not abort_event.wait(0.01):
                    pass
                raise PreparationCancelled()
            return super().prepare(spec, *args, **kwargs)

    async def scenario() -> None:
        first = EpisodeSpec("MoveCube", 3)
        selected = EpisodeSpec("BinFill", 4)
        cache = MemoryControlCache((first, selected), ready=True)
        factory = CancellableFactory()
        collection = CollectionProgress(
            tmp_path / "progress.json",
            ("MoveCube", "BinFill"),
            {"MoveCube": 100, "BinFill": 100},
            target_per_task=100,
        )

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [first, selected],
            ignore,
            ignore,
            collection=collection,
            control_cache=cache,
        )
        await controller.start()
        await control_cache__wait_until(factory.entered.is_set)
        accepted, _ = await controller.submit(
            CommandRequest(
                Command.SELECT_NEXT,
                "select-binfill",
                time.monotonic_ns(),
                {"task": "BinFill", "seed": "10004"},
            )
        )
        assert accepted
        await control_cache__wait_until(
            lambda: controller.state.phase == Phase.READY
            and controller.state.task == "BinFill"
            and controller.state.episode == 4
        )
        await asyncio.wait_for(controller.stop(), timeout=1.0)

    asyncio.run(scenario())


def test_missing_control_is_generated_in_background_without_blocking_enabled() -> None:
    async def scenario() -> None:
        spec = EpisodeSpec("BinFill", 0)
        cache = MemoryControlCache((spec,), ready=False)
        factory = FakeBackendFactory(
            inference_delay=0.01,
            terminal_step=3,
            control_outcome="fail",
        )

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [spec],
            ignore,
            ignore,
            default_fps=30,
            control_cache=cache,
        )
        await controller.start()
        await control_cache__wait_until(lambda: controller.state.phase == Phase.READY)
        enabled = next(item for item in factory.episodes if item.role == "enabled")
        assert enabled.closed_with is None
        await control_cache__wait_until(lambda: controller.state.control_cached)
        controls = [item for item in factory.episodes if item.role == "control"]
        assert len(controls) == 1
        assert controls[0].closed_with is True
        assert controller.state.control_cache_ready == 1
        await controller.submit(
            CommandRequest(Command.NEXT_EPISODE, "next", time.monotonic_ns())
        )
        await control_cache__wait_until(lambda: controller.state.phase == Phase.COMPLETE)
        await asyncio.wait_for(controller.stop(), timeout=1.0)

    asyncio.run(scenario())


def test_empty_failure_pool_is_generated_before_collection_allocation(tmp_path) -> None:
    async def scenario() -> None:
        spec = EpisodeSpec("BinFill", 2)
        cache = MemoryControlCache((spec,), ready=False)
        factory = FakeBackendFactory(
            inference_delay=0.01,
            terminal_step=3,
            control_outcome="fail",
        )
        progress = CollectionProgress(
            tmp_path,
            ("BinFill",),
            {"BinFill": 100},
            target_per_task=1,
        )

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            [spec],
            ignore,
            ignore,
            default_fps=30,
            collection=progress,
            control_cache=cache,
        )
        await controller.start()
        await control_cache__wait_until(lambda: controller.state.phase == Phase.READY)
        assert controller.state.task == "BinFill"
        assert not controller.state.control_cached
        await control_cache__wait_until(lambda: controller.state.control_cached)
        assert controller.state.control_cached
        assert controller.state.control_cache_ready == 1
        await asyncio.wait_for(controller.stop(), timeout=1.0)

    asyncio.run(scenario())


def _write_control(
    root: Path,
    spec: EpisodeSpec,
    *,
    seed: int,
    outcome: str,
) -> None:
    task_root = root / spec.task
    task_root.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.task}_ep{spec.episode}_seed{seed}_take0000_control"
    (task_root / f"{stem}.mp4").write_bytes(b"video")
    (task_root / f"{stem}.json").write_text(
        json.dumps(
            {
                "role": "control",
                "kept": True,
                "task": spec.task,
                "episode": spec.episode,
                "seed": seed,
                "outcome": outcome,
                "control_cache": True,
            }
        ),
        encoding="utf-8",
    )


def test_control_cache_exposes_only_failures_and_remembers_removed_success(tmp_path) -> None:
    failed = EpisodeSpec("TaskA", 0)
    succeeded = EpisodeSpec("TaskA", 1)
    _write_control(tmp_path, failed, seed=100, outcome="fail")
    _write_control(tmp_path, succeeded, seed=101, outcome="success")

    cache = ControlCache(tmp_path, (failed, succeeded))

    assert cache.lookup_failure_spec(failed) is not None
    assert cache.lookup_failure_spec(succeeded) is None
    assert cache.failure_episodes("TaskA") == (0,)
    assert cache.screened(succeeded)
    assert cache.prune_unreferenced_successes() == 1
    assert cache.lookup_spec(succeeded) is None
    assert ControlCache(tmp_path, (failed, succeeded)).screened(succeeded)


def test_scene_generation_error_skips_one_candidate_without_stopping_service(tmp_path) -> None:
    class SceneGenerationError(RuntimeError):
        pass

    class RecoveringFactory(FakeBackendFactory):
        def prepare(self, spec, *args, **kwargs):
            if spec.episode == 0 and kwargs.get("role", args[-1]) == "enabled":
                raise SceneGenerationError("bad randomized scene")
            return super().prepare(spec, *args, **kwargs)

    async def scenario() -> None:
        specs = (EpisodeSpec("VideoPlaceOrder", 0), EpisodeSpec("VideoPlaceOrder", 1))
        cache = MemoryControlCache(specs, ready=True)
        progress = CollectionProgress(
            tmp_path,
            ("VideoPlaceOrder",),
            {"VideoPlaceOrder": 100},
        )
        factory = RecoveringFactory()

        async def ignore(*_values):
            return None

        controller = ManualDaggerController(
            factory,
            specs,
            ignore,
            ignore,
            collection=progress,
            control_cache=cache,
        )
        await controller.start()
        await control_cache__wait_until(
            lambda: controller.state.phase == Phase.READY
            and controller.state.episode == 1
        )
        assert controller.state.phase != Phase.ERROR
        await asyncio.wait_for(controller.stop(), timeout=2.0)

    asyncio.run(scenario())


def test_target_guides_next_task_but_does_not_cap_collection(tmp_path) -> None:
    progress = CollectionProgress(
        tmp_path,
        ("TaskA", "TaskB"),
        {"TaskA": 2, "TaskB": 2},
        target_per_task=1,
    )
    first = progress.allocate(None)
    assert (first.task, first.episode, first.take) == ("TaskA", 0, 1)
    progress.record_keep(first)
    second = progress.allocate(first.task)
    assert second.task == "TaskB"
    progress.record_keep(second)
    extra = progress.allocate(second.task)
    assert extra.task == "TaskA"
    assert progress.snapshot(extra.task)["progress_total_kept"] == 2


def test_manual_selection_and_blocked_episode_persist(tmp_path) -> None:
    progress = CollectionProgress(
        tmp_path,
        ("TaskA", "TaskB"),
        {"TaskA": 3, "TaskB": 3},
    )
    blocked = progress.allocate(None)
    progress.block(blocked, 101)
    progress.select("TaskB", 2)
    selected = progress.allocate("TaskA")
    assert (selected.task, selected.episode) == ("TaskB", 2)
    restored = CollectionProgress(
        tmp_path,
        ("TaskA", "TaskB"),
        {"TaskA": 3, "TaskB": 3},
    )
    assert restored.blocked["TaskA"] == {blocked.episode: 101}


def test_restart_recounts_only_published_enabled_artifacts(tmp_path) -> None:
    task_root = tmp_path / "TaskA"
    task_root.mkdir()
    h5_path = task_root / "kept.h5"
    h5_path.write_bytes(b"published")
    (task_root / "kept.json").write_text(
        json.dumps(
            {
                "episode_id": "kept-enabled",
                "role": "enabled",
                    "kept": True,
                    "take": 7,
                    "training_h5": {"num_steps": 3, "expert_steps": 1},
                    "artifacts": {"h5": [str(h5_path)]},
            }
        ),
        encoding="utf-8",
    )
    (task_root / "control.json").write_text(
        json.dumps(
            {
                "episode_id": "kept-control",
                "role": "control",
                "kept": True,
                "take": 8,
            }
        ),
        encoding="utf-8",
    )
    progress = CollectionProgress(tmp_path, ("TaskA",), {"TaskA": 100})
    assert progress.counts == {"TaskA": 1}
    assert progress.next_take == {"TaskA": 8}


def test_restart_merges_training_manifest_without_counting_live_copy_twice(
    tmp_path,
) -> None:
    primary = tmp_path / "primary"
    training = tmp_path / "training"
    for root in (primary, training):
        (root / "TaskA").mkdir(parents=True)

    archived_h5 = training / "TaskA/archived.h5"
    copied_live_h5 = training / "TaskA/copied-live.h5"
    archived_h5.write_bytes(b"archived")
    copied_live_h5.write_bytes(b"copied-live")
    (training / "training_manifest.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_name": "TaskA",
                        "episode": 1,
                        "seed": 101,
                        "take": 2,
                        "h5_path": str(archived_h5),
                    },
                    {
                        "task_name": "TaskA",
                        "episode": 2,
                        "seed": 102,
                        "take": 3,
                        "h5_path": str(copied_live_h5),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    live_h5 = primary / "TaskA/live.h5"
    live_h5.write_bytes(b"live")
    (primary / "TaskA/live.json").write_text(
        json.dumps(
            {
                "episode_id": "live-enabled",
                "role": "enabled",
                "kept": True,
                "episode": 2,
                "seed": 102,
                "take": 3,
                "training_h5": {"num_steps": 3, "expert_steps": 1},
                "artifacts": {"h5": [str(live_h5)]},
            }
        ),
        encoding="utf-8",
    )

    progress = CollectionProgress(
        primary,
        ("TaskA",),
        {"TaskA": 100},
        training_root=training,
    )

    assert progress.counts == {"TaskA": 2}
    assert progress.next_take == {"TaskA": 4}


def test_control_specs_follow_cursor_and_exclude_blocked_episodes(tmp_path) -> None:
    progress = CollectionProgress(
        tmp_path,
        ("TaskA", "TaskB"),
        {"TaskA": 5, "TaskB": 3},
        target_per_task=3,
        episode_start=2,
    )
    progress.blocked["TaskA"] = {3: 103}

    specs = progress.control_specs()

    assert [(item.task, item.episode, item.take) for item in specs] == [
        ("TaskA", 2, 0),
        ("TaskA", 4, 0),
        ("TaskA", 0, 0),
        ("TaskB", 2, 0),
        ("TaskB", 0, 0),
        ("TaskB", 1, 0),
    ]


def test_failure_allocation_cycles_only_failed_candidates(tmp_path) -> None:
    progress = CollectionProgress(
        tmp_path,
        ("VideoPlaceOrder", "TaskB"),
        {"VideoPlaceOrder": 100, "TaskB": 100},
        target_per_task=3,
    )
    candidates = {"VideoPlaceOrder": (2, 7), "TaskB": (1,)}

    first = progress.allocate_failure(None, candidates)
    second = progress.allocate_failure("VideoPlaceOrder", candidates)
    third = progress.allocate_failure("VideoPlaceOrder", candidates)

    assert first is not None and first.episode == 2
    assert second is not None and second.episode == 7
    assert third is not None and third.episode == 2
    assert progress.failure_attempts["VideoPlaceOrder"] == {2: 2, 7: 1}

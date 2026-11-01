"""manual_dagger / controller: shared behavior owned by this layer."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import itertools
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol
import numpy as np
from manual_dagger.state import (
    Command,
    CommandRequest,
    EpisodeSpec,
    Phase,
    PreparationCancelled,
    SessionState,
    StepResult,
)
from manual_dagger.state import CollectionProgress
from manual_dagger.state import ControlArtifact, ControlCache


StatePublisher = Callable[[dict[str, Any]], Awaitable[None]]
FramePublisher = Callable[[str, np.ndarray, int], Awaitable[None]]


class EpisodeRuntime(Protocol):
    task: str
    episode: int
    seed: int
    difficulty: str
    prompt: str
    current_subtask: str
    step: int

    def initial_frame(self) -> np.ndarray: ...

    def initialization_replay(self) -> Any: ...

    def prime_policy(self) -> None: ...

    def policy_step(
        self,
        takeover_event: threading.Event,
        abort_event: threading.Event,
        pause_event: threading.Event,
    ) -> StepResult: ...

    def intervene(self, abort_event: threading.Event) -> StepResult: ...

    def close(self, keep: bool, metadata: dict[str, Any]) -> dict[str, Any] | None: ...


class BackendFactory(Protocol):
    def prepare(
        self,
        spec: EpisodeSpec,
        frame_callback: Callable[[np.ndarray, str], None],
        abort_event: threading.Event,
        fps_provider: Callable[[], int],
        role: str,
        initialization_replay: Any | None = None,
    ) -> EpisodeRuntime: ...

    def close(self) -> None: ...

    def episode_count(self, task: str) -> int: ...

    def episode_for_seed(self, task: str, seed: int) -> int: ...


class ManualDaggerController:
    """Serialize episodes while WebSocket handlers latch i/p immediately."""

    def __init__(
        self,
        factory: BackendFactory,
        episodes: Sequence[EpisodeSpec],
        publish_state: StatePublisher,
        publish_frame: FramePublisher,
        *,
        default_fps: int = 5,
        minimum_fps: int = 1,
        maximum_fps: int = 30,
        collection: CollectionProgress | None = None,
        control_cache: ControlCache | None = None,
    ) -> None:
        if not episodes:
            raise ValueError("episode queue must not be empty")
        if not minimum_fps <= default_fps <= maximum_fps:
            raise ValueError("default FPS is outside configured bounds")
        self.factory = factory
        self.episodes = tuple(episodes)
        self.publish_state = publish_state
        self.publish_frame = publish_frame
        self.minimum_fps = minimum_fps
        self.maximum_fps = maximum_fps
        self.collection = collection
        self.control_cache = control_cache
        self.state = SessionState(
            fps=default_fps,
            queue_total=len(self.episodes),
        )
        self._commands: asyncio.Queue[CommandRequest] = asyncio.Queue()
        self._takeover_event = threading.Event()
        self._abort_event = threading.Event()
        self._pause_event = threading.Event()
        self._control_abort_event = threading.Event()
        self._control_pause_event = threading.Event()
        self._never_takeover_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runtime: EpisodeRuntime | None = None
        self._control_runtime: EpisodeRuntime | None = None
        self._control_cache_runtime: EpisodeRuntime | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._control_task: asyncio.Task[None] | None = None
        self._control_cache_task: asyncio.Task[None] | None = None
        self._control_cleanup_thread: threading.Thread | None = None
        self._control_cache_queue: asyncio.PriorityQueue[
            tuple[int, int, EpisodeSpec]
        ] = asyncio.PriorityQueue()
        self._control_cache_order = itertools.count()
        self._queued_control_keys: set[tuple[str, int]] = set()
        self._prime_tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        self._last_frame_publish = {"enabled": 0.0, "control": 0.0}
        self._pending_frames: dict[str, np.ndarray | None] = {
            "enabled": None,
            "control": None,
        }
        self._intervention_receive_ns: int | None = None
        self._first_expert_action_ns: int | None = None
        self._current_spec: EpisodeSpec | None = None
        self._reset_requested = False
        self._next_requested = False
        self._block_requested = False
        self._scheduled_autostart = False
        # Each independent environment stays on its own dedicated OS thread.
        # This preserves simulator thread affinity without blocking WebSockets.
        self._simulator_executors = {
            role: ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"manual-dagger-{role}",
            )
            for role in ("enabled", "control")
        }
        self._sync_collection_state()
        if self.control_cache is not None:
            self.state.control_cache_target = len(self.control_cache.specs)
            self.state.control_cache_ready = self.control_cache.ready_count()
            for spec in self.control_cache.specs:
                self._queue_control(spec, priority=1)

    async def start(self) -> None:
        if self._run_task is not None:
            return
        self._loop = asyncio.get_running_loop()
        if self.control_cache is not None:
            self._control_cleanup_thread = threading.Thread(
                target=self.control_cache.prune_unreferenced_successes,
                name="manual-dagger-control-cleanup",
                daemon=True,
            )
            self._control_cleanup_thread.start()
        self._run_task = asyncio.create_task(self._run(), name="manual-dagger")
        if self.control_cache is not None:
            self._control_cache_task = asyncio.create_task(
                self._run_control_cache(),
                name="manual-dagger-control-cache",
            )

    async def stop(self) -> None:
        self._closed = True
        self._abort_event.set()
        self._control_abort_event.set()
        if self.control_cache is not None:
            self._control_cache_queue.put_nowait(
                (-1, next(self._control_cache_order), EpisodeSpec("", 0))
            )
        if self._run_task is not None:
            if not self._run_task.done():
                self._commands.put_nowait(
                    CommandRequest(Command.ABORT, "shutdown", time.monotonic_ns())
                )
                deadline = time.monotonic() + 30.0
                while not self._run_task.done() and time.monotonic() < deadline:
                    # Periodic wakeups also drain executor completion callbacks
                    # on CCI kernels that occasionally coalesce eventfd signals.
                    await asyncio.sleep(0.05)
            if not self._run_task.done():
                self._run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(self._run_task, timeout=5.0)
            self._run_task = None
        for task in self._prime_tasks.values():
            if not task.done():
                task.cancel()
        if self._control_task is not None and not self._control_task.done():
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(self._control_task, timeout=5.0)
        if self._control_cache_task is not None and not self._control_cache_task.done():
            self._control_cache_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._control_cache_task
        self._control_cache_task = None
        self._control_cleanup_thread = None
        if self._runtime is not None:
            await self._run_blocking_on(
                "enabled",
                self._runtime.close,
                False,
                self._metadata("enabled"),
            )
            self._runtime = None
        if self._control_runtime is not None:
            await self._run_blocking_on(
                "control",
                self._control_runtime.close,
                False,
                self._metadata("control"),
            )
            self._control_runtime = None
        if self._control_cache_runtime is not None:
            await self._run_blocking_on(
                "control",
                self._control_cache_runtime.close,
                False,
                self._control_cache_metadata(self._control_cache_runtime),
            )
            self._control_cache_runtime = None
        # Shutdown is already terminal for the web service.  Call this directly
        # so event-loop wakeups are not required while the storage process joins.
        self.factory.close()
        for executor in self._simulator_executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    async def submit(self, request: CommandRequest) -> tuple[bool, str]:
        command = request.command
        if command == Command.SELECT_NEXT:
            if self.collection is None:
                return False, "task selection is unavailable for a finite queue"
            value = request.value if isinstance(request.value, dict) else {}
            task = str(value.get("task", ""))
            raw_seed = value.get("seed")
            try:
                episode = (
                    None
                    if raw_seed in {None, ""}
                    else self.factory.episode_for_seed(task, int(raw_seed))
                )
                if (
                    episode is not None
                    and self.control_cache is not None
                    and self.control_cache.lookup_failure(task, episode, int(raw_seed))
                    is None
                ):
                    raise ValueError(
                        f"seed {int(raw_seed)} for {task} has no failed control"
                    )
                self.collection.select(task, episode)
            except (TypeError, ValueError) as error:
                return False, str(error)
            switch_request = CommandRequest(
                Command.NEXT_EPISODE,
                f"{request.request_id}:switch",
                request.server_receive_ns,
            )
            if self.state.phase == Phase.PREPARING:
                self._next_requested = True
                self._abort_event.set()
            elif self.state.phase in {Phase.POLICY, Phase.INTERVENING}:
                self._next_requested = True
                self._abort_event.set()
            elif self.state.phase in {Phase.READY, Phase.PAUSED, Phase.FINISHED}:
                await self._commands.put(switch_request)
            self._sync_collection_state()
            self.state.message = (
                f"Switching to task: {task}"
                if raw_seed in {None, ""}
                else f"Switching to task: {task}, seed {int(raw_seed)}"
            )
            self.state.command_sequence += 1
            self.state.last_command = command.value
            self.state.last_command_server_ns = request.server_receive_ns
            await self._publish_state()
            return True, "accepted"
        if command == Command.SCHEDULE_INTERVENTION:
            if self.state.phase not in {
                Phase.PREPARING,
                Phase.READY,
                Phase.POLICY,
                Phase.PAUSED,
                Phase.INTERVENING,
                Phase.FINISHED,
            }:
                return False, (
                    "control-frame intervention is unavailable during "
                    f"{self.state.phase.value}"
                )
            if self.state.phase == Phase.PREPARING and not self.state.control_cached:
                return False, "control-frame intervention requires a ready control cache"
            value = request.value if isinstance(request.value, dict) else {}
            try:
                target_step = int(value["step"])
            except (KeyError, TypeError, ValueError):
                return False, "selected control frame has no valid action step"
            if not 0 <= target_step <= self.state.control_step:
                return False, (
                    f"control step {target_step} is outside the available "
                    f"range 0..{self.state.control_step}"
                )
            self.state.scheduled_intervention_step = target_step
            self.state.scheduled_intervention_pending = True
            self.state.scheduled_intervention_fired = False
            self._scheduled_autostart = True
            if self.state.phase == Phase.PREPARING:
                self.state.message = (
                    f"Expert scheduled at control step {target_step}; "
                    "waiting for enabled preparation"
                )
                self.state.command_sequence += 1
                self.state.last_command = command.value
                self.state.last_command_server_ns = request.server_receive_ns
                await self._publish_state()
                return True, "accepted"
            self._reset_requested = True
            self._abort_event.set()
            self.state.message = (
                f"Resetting enabled; expert is scheduled at control step {target_step}"
            )
        if command == Command.INTERVENE:
            if self.state.phase not in {Phase.POLICY, Phase.PAUSED}:
                return False, f"i is only valid during policy rollout ({self.state.phase.value})"
            self._cancel_scheduled_intervention(clear_abort=True)
            self._intervention_receive_ns = request.server_receive_ns
            self._takeover_event.set()
            self.state.source = "EXPERT_PENDING"
            self.state.message = "Intervention accepted; policy cutoff is latched"
        elif command == Command.ABORT:
            if self.state.phase not in {
                Phase.READY,
                Phase.POLICY,
                Phase.PAUSED,
                Phase.INTERVENING,
            }:
                return False, f"p is not valid during {self.state.phase.value}"
            self._cancel_scheduled_intervention()
            self._abort_event.set()
            if self.control_cache is None:
                self._control_abort_event.set()
            self.state.message = "Discard requested"
        elif command == Command.PAUSE:
            if self.state.phase == Phase.POLICY:
                self._cancel_scheduled_intervention(clear_abort=True)
                self._pause_event.set()
                self.state.phase = Phase.PAUSED
                self.state.source = "PAUSED"
                self.state.message = "Enabled paused; press z to resume"
            elif self.state.phase == Phase.PAUSED:
                self._cancel_scheduled_intervention(clear_abort=True)
                self._pause_event.clear()
                self.state.phase = Phase.POLICY
                self.state.source = "POLICY"
                self.state.message = "Enabled resumed"
            else:
                return False, f"z is only valid during enabled rollout ({self.state.phase.value})"
        elif command == Command.RESET:
            if self.state.phase not in {
                Phase.POLICY,
                Phase.PAUSED,
                Phase.INTERVENING,
                Phase.FINISHED,
            }:
                return False, f"r is only valid after enabled starts ({self.state.phase.value})"
            self._cancel_scheduled_intervention()
            self._reset_requested = True
            self._abort_event.set()
            self.state.message = "Reset requested for enabled; control continues"
        elif command == Command.NEXT_EPISODE:
            if self.state.phase not in {
                Phase.PREPARING,
                Phase.READY,
                Phase.POLICY,
                Phase.PAUSED,
                Phase.INTERVENING,
                Phase.FINISHED,
            }:
                return False, f"n is not valid during {self.state.phase.value}"
            self._cancel_scheduled_intervention()
            self._next_requested = True
            self._abort_event.set()
            if self.control_cache is None:
                self._control_abort_event.set()
            self.state.message = "Discarding this episode and moving to the next"
        elif command == Command.BLOCK_SEED:
            if self.collection is None:
                return False, "seed blocking is unavailable for a finite queue"
            if self.state.phase not in {
                Phase.READY,
                Phase.POLICY,
                Phase.PAUSED,
                Phase.INTERVENING,
                Phase.FINISHED,
            }:
                return False, f"b is not valid during {self.state.phase.value}"
            self._cancel_scheduled_intervention()
            self._block_requested = True
            self._abort_event.set()
            if self.control_cache is None:
                self._control_abort_event.set()
            self.state.message = "Blocking this seed and moving to the next legal seed"
        elif command == Command.START:
            if self.state.phase != Phase.READY:
                return False, f"s is only valid while ready ({self.state.phase.value})"
            self._cancel_scheduled_intervention(clear_abort=True)
        elif command in {Command.KEEP, Command.DISCARD}:
            if self.state.phase != Phase.FINISHED:
                return False, (
                    f"{command.value} is only valid after an episode ends "
                    f"({self.state.phase.value})"
                )
            # Preserve the selected/fired step in the final audit metadata while
            # preventing a queued schedule command from restarting the episode.
            self.state.scheduled_intervention_pending = False
            self._scheduled_autostart = False
            self._reset_requested = False
        elif command in {Command.FPS_UP, Command.FPS_DOWN} and self.state.phase in {
            Phase.COMPLETE,
            Phase.ERROR,
        }:
            return False, f"FPS cannot change during {self.state.phase.value}"
        self.state.command_sequence += 1
        self.state.last_command = command.value
        self.state.last_command_server_ns = request.server_receive_ns
        await self._commands.put(request)
        await self._publish_state()
        return True, "accepted"

    async def snapshot(self) -> dict[str, Any]:
        return self.state.as_dict()

    def _queue_control(self, spec: EpisodeSpec, *, priority: int) -> None:
        if self.control_cache is None:
            return
        key = (spec.task, spec.episode)
        if self.control_cache.screened(spec):
            return
        if key in self._queued_control_keys and priority > 0:
            return
        self._queued_control_keys.add(key)
        self._control_cache_queue.put_nowait(
            (priority, next(self._control_cache_order), EpisodeSpec(spec.task, spec.episode, 0))
        )

    def _backend_frame(self, stream: str, frame: np.ndarray, source: str) -> None:
        if self._loop is None or self._closed:
            return
        copied = np.asarray(frame, dtype=np.uint8).copy()
        self._loop.call_soon_threadsafe(self._offer_frame, stream, copied, source)

    def _offer_frame(self, stream: str, frame: np.ndarray, source: str) -> None:
        self._pending_frames[stream] = frame
        if stream == "enabled" and source == "EXPERT" and self._first_expert_action_ns is None:
            self._first_expert_action_ns = time.monotonic_ns()
            if self._intervention_receive_ns is not None:
                self.state.first_expert_action_latency_ms = (
                    self._first_expert_action_ns - self._intervention_receive_ns
                ) / 1_000_000.0
        now = time.monotonic()
        if now - self._last_frame_publish[stream] >= 1.0 / self.state.fps:
            asyncio.create_task(self._flush_frame(stream))

    async def _flush_frame(self, stream: str, force: bool = False) -> None:
        if self._pending_frames[stream] is None:
            return
        now = time.monotonic()
        if not force and now - self._last_frame_publish[stream] < 1.0 / self.state.fps:
            return
        frame = self._pending_frames[stream]
        self._pending_frames[stream] = None
        self._last_frame_publish[stream] = now
        if stream == "enabled":
            self.state.frame_sequence += 1
        else:
            self.state.control_frame_sequence += 1
        assert frame is not None
        step = self.state.control_step if stream == "control" else self.state.step
        await self.publish_frame(stream, frame, step)
        await self._publish_state()

    async def _publish_state(self) -> None:
        self._sync_collection_state()
        await self.publish_state(self.state.as_dict())

    def _sync_collection_state(self) -> None:
        if self.collection is None:
            return
        snapshot = self.collection.snapshot(self.state.task or None)
        for key, value in snapshot.items():
            setattr(self.state, key, value)

    async def _run(self) -> None:
        try:
            if self.collection is None:
                for index, spec in enumerate(self.episodes):
                    if self._closed:
                        break
                    self._current_spec = spec
                    await self._prepare(index, spec)
                    await self._episode_loop()
            else:
                current_task: str | None = None
                while not self._closed:
                    if self.control_cache is None:
                        spec = self.collection.allocate(current_task)
                    else:
                        spec = self.collection.allocate_failure(
                            current_task,
                            self.control_cache.failure_episodes_by_task(),
                        )
                        if spec is None:
                            # A failed-control pool improves review and seed
                            # selection, but must not gate live collection.
                            # Screen controls in the background while the
                            # enabled stream starts from the normal allocator.
                            spec = self.collection.allocate(current_task)
                    current_task = spec.task
                    self._current_spec = spec
                    try:
                        await self._prepare(self.state.progress_total_kept, spec)
                    except PreparationCancelled:
                        if not self._next_requested:
                            raise
                        self.state.phase = Phase.PREPARING
                        self.state.source = "IDLE"
                        self.state.message = "Previous preparation cancelled; switching task"
                        await self._publish_state()
                        continue
                    except Exception as error:
                        if not self._recoverable_episode_error(error):
                            raise
                        self.state.phase = Phase.PREPARING
                        self.state.source = "IDLE"
                        self.state.message = (
                            f"Skipped one bad scene: {type(error).__name__}: {error}"
                        )
                        await self._publish_state()
                        continue
                    await self._episode_loop()
            self.state.phase = Phase.COMPLETE
            self.state.source = "IDLE"
            self.state.message = "Episode queue complete"
            await self._publish_state()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.state.phase = Phase.ERROR
            self.state.message = f"{type(error).__name__}: {error}"
            await self._publish_state()

    @staticmethod
    def _recoverable_episode_error(error: BaseException) -> bool:
        current: BaseException | None = error
        while current is not None:
            if type(current).__name__ == "SceneGenerationError":
                return True
            current = current.__cause__ or current.__context__
        return False

    async def _prepare(self, index: int, spec: EpisodeSpec) -> None:
        self.state.episode_generation += 1
        self._takeover_event.clear()
        self._abort_event.clear()
        self._pause_event.clear()
        self._control_abort_event.clear()
        self._control_pause_event.clear()
        self._intervention_receive_ns = None
        self._first_expert_action_ns = None
        self._reset_requested = False
        self._next_requested = False
        self._block_requested = False
        self._scheduled_autostart = False
        self.state.phase = Phase.PREPARING
        self.state.task = spec.task
        self.state.episode = spec.episode
        self.state.seed = None
        self.state.prompt = ""
        self.state.current_subtask = ""
        self.state.step = 0
        self.state.outcome = ""
        self.state.eval_success = None
        self.state.intervened = False
        self.state.source = "IDLE"
        self.state.message = "Resetting environment and policy"
        self.state.queue_index = index + 1
        self.state.policy_cutoff_latency_ms = None
        self.state.first_expert_action_latency_ms = None
        self.state.policy_prefetch_ready = False
        self.state.control_phase = Phase.PREPARING.value
        self.state.control_source = "IDLE"
        self.state.control_step = 0
        self.state.control_outcome = ""
        self.state.control_eval_success = None
        self.state.control_message = "Preparing independent control environment"
        self.state.control_prefetch_ready = False
        self.state.control_cached = False
        self.state.control_video_frames = 0
        self.state.control_video_fps = 0
        self.state.scheduled_intervention_step = None
        self.state.scheduled_intervention_pending = False
        self.state.scheduled_intervention_fired = False
        self.state.enabled_generation += 1
        cached_control: ControlArtifact | None = None
        if self.control_cache is not None:
            cached_control = self.control_cache.lookup_failure(spec.task, spec.episode)
            if cached_control is not None:
                metadata = cached_control.metadata
                self.state.seed = int(metadata["seed"])
                self.state.difficulty = str(metadata.get("difficulty", ""))
                self.state.prompt = str(metadata.get("prompt", ""))
                self.state.current_subtask = str(
                    metadata.get("current_subtask", "")
                )
            self._apply_cached_control(cached_control)
            self._queue_control(spec, priority=0)
        await self._publish_state()
        enabled_prepare = self._run_blocking_on(
            "enabled",
            self.factory.prepare,
            spec,
            functools.partial(self._backend_frame, "enabled"),
            self._abort_event,
            lambda: self.state.fps,
            "enabled",
        )
        if self.control_cache is None:
            control_prepare = self._run_blocking_on(
                "control",
                self.factory.prepare,
                spec,
                functools.partial(self._backend_frame, "control"),
                self._control_abort_event,
                lambda: self.state.fps,
                "control",
            )
            self._runtime, self._control_runtime = await asyncio.gather(
                enabled_prepare,
                control_prepare,
            )
            if (
                self._runtime.seed != self._control_runtime.seed
                or self._runtime.prompt != self._control_runtime.prompt
            ):
                raise RuntimeError(
                    "enabled and control environments resolved different episode identities"
                )
        else:
            self._runtime = await enabled_prepare
            self._control_runtime = None
            if (
                cached_control is not None
                and self._runtime.seed != int(cached_control.metadata["seed"])
            ):
                raise RuntimeError(
                    "enabled environment and cached control resolved different seeds"
                )
        self.state.seed = self._runtime.seed
        self.state.difficulty = self._runtime.difficulty
        self.state.prompt = self._runtime.prompt
        self.state.current_subtask = self._runtime.current_subtask
        self.state.step = self._runtime.step
        self.state.phase = Phase.READY
        self.state.message = "Press s to start"
        if self.control_cache is None:
            assert self._control_runtime is not None
            self.state.control_step = self._control_runtime.step
            self.state.control_phase = Phase.POLICY.value
            self.state.control_source = "POLICY"
            self.state.control_message = "Independent control rollout is starting"
            await self.publish_frame("control", self._control_runtime.initial_frame(), 0)
            self.state.control_frame_sequence += 1
        else:
            self._apply_cached_control(
                self.control_cache.lookup_failure(
                    spec.task,
                    spec.episode,
                    self._runtime.seed,
                )
            )
        await self.publish_frame("enabled", self._runtime.initial_frame(), 0)
        self.state.frame_sequence += 1
        self._prime_tasks = {
            "enabled": asyncio.create_task(
                self._prime_runtime("enabled", self._runtime),
                name="manual-dagger-prime-enabled",
            )
        }
        if self.control_cache is None:
            assert self._control_runtime is not None
            self._prime_tasks["control"] = asyncio.create_task(
                self._prime_runtime("control", self._control_runtime),
                name="manual-dagger-prime-control",
            )
            self._control_task = asyncio.create_task(
                self._run_control(),
                name="manual-dagger-control",
            )
        await self._publish_state()

    async def _episode_loop(self) -> None:
        while await self._episode_attempt():
            await self._reset_enabled()

    async def _episode_attempt(self) -> bool:
        assert self._runtime is not None
        if self.control_cache is None:
            assert self._control_runtime is not None
        if self._scheduled_autostart and self.state.scheduled_intervention_pending:
            self._scheduled_autostart = False
            await self._start_rollouts(
                f"Scheduled replay; expert at step {self.state.scheduled_intervention_step}"
            )
        else:
            while True:
                request = await self._commands.get()
                if request.command in {Command.FPS_UP, Command.FPS_DOWN}:
                    await self._adjust_fps(request.command)
                    continue
                if request.command == Command.SCHEDULE_INTERVENTION:
                    if self.state.scheduled_intervention_pending:
                        return True
                    continue
                if request.command == Command.ABORT:
                    await self._close_episode(
                        False,
                        "Discarded before start",
                        keep_control=False,
                    )
                    return False
                if request.command == Command.NEXT_EPISODE:
                    await self._close_episode(
                        False,
                        "Skipped by n",
                        keep_control=False,
                    )
                    return False
                if request.command == Command.BLOCK_SEED:
                    await self._block_current_seed()
                    return False
                if request.command == Command.START:
                    await self._start_rollouts("Policy rollout")
                    break

        first_tick = True
        while self.state.phase in {Phase.POLICY, Phase.PAUSED}:
            if self.state.phase == Phase.PAUSED:
                request = await self._commands.get()
            elif first_tick:
                request = None
                first_tick = False
            else:
                request = await self._wait_for_command_or_tick()
            if request is not None:
                if request.command in {Command.FPS_UP, Command.FPS_DOWN}:
                    await self._adjust_fps(request.command)
                    continue
                if request.command == Command.PAUSE:
                    continue
                if request.command == Command.SCHEDULE_INTERVENTION:
                    if self.state.scheduled_intervention_pending:
                        return True
                    continue
                if request.command == Command.INTERVENE:
                    transition = await self._run_intervention()
                    if transition == "reset":
                        return True
                    if transition == "closed":
                        return False
                    if transition == "paused":
                        continue
                    break
                if request.command == Command.RESET:
                    return True
                if request.command == Command.NEXT_EPISODE:
                    await self._close_episode(
                        False,
                        "Skipped by n",
                        keep_control=False,
                    )
                    return False
                if request.command == Command.BLOCK_SEED:
                    await self._block_current_seed()
                    return False
                if request.command == Command.ABORT:
                    await self._close_episode(False, "Discarded by p", keep_control=False)
                    return False
                continue
            if self.state.phase == Phase.PAUSED:
                continue
            if self._scheduled_intervention_due():
                transition = await self._run_intervention(scheduled=True)
                if transition == "reset":
                    return True
                if transition == "closed":
                    return False
                if transition == "paused":
                    continue
                break
            await self._await_prime("enabled")
            result = await self._run_blocking_on(
                "enabled",
                self._runtime.policy_step,
                self._takeover_event,
                self._abort_event,
                self._pause_event,
            )
            self.state.step = result.step
            self.state.current_subtask = self._runtime.current_subtask
            if result.status == "takeover_requested":
                transition = await self._run_intervention()
                if transition == "reset":
                    return True
                if transition == "closed":
                    return False
                if transition == "paused":
                    continue
                break
            if result.status == "aborted":
                if self._reset_requested:
                    return True
                if self._next_requested:
                    await self._close_episode(
                        False,
                        "Skipped by n",
                        keep_control=False,
                    )
                    return False
                if self._block_requested:
                    await self._block_current_seed()
                    return False
                await self._close_episode(False, "Discarded by p", keep_control=False)
                return False
            if result.status == "paused":
                self.state.phase = Phase.PAUSED
                self.state.source = "PAUSED"
                self.state.message = "Enabled paused; press z to resume"
                await self._publish_state()
                continue
            if result.status == "terminal":
                await self._finish(result)
                break
            await self._flush_frame("enabled", force=True)

        if self.state.phase == Phase.INTERVENING:
            return False
        if self.state.phase != Phase.FINISHED:
            return False
        while True:
            request = await self._commands.get()
            if request.command in {Command.FPS_UP, Command.FPS_DOWN}:
                await self._adjust_fps(request.command)
                continue
            if request.command == Command.SCHEDULE_INTERVENTION:
                if self.state.scheduled_intervention_pending:
                    return True
                continue
            if request.command == Command.KEEP:
                if self.state.eval_success is not True:
                    self.state.message = (
                        "Cannot keep: enabled trajectory is not an official eval success"
                    )
                    await self._publish_state()
                    continue
                await self._close_episode(True, "Kept")
                assert self._current_spec is not None
                if self.collection is not None:
                    self.collection.record_keep(self._current_spec)
                return False
            if request.command == Command.DISCARD:
                await self._close_episode(False, "Discarded")
                return False
            if request.command == Command.NEXT_EPISODE:
                await self._close_episode(
                    False,
                    "Skipped by n",
                    keep_control=False,
                )
                return False
            if request.command == Command.ABORT:
                await self._close_episode(False, "Discarded during shutdown", keep_control=False)
                return False
            if request.command == Command.RESET:
                return True
            if request.command == Command.BLOCK_SEED:
                await self._block_current_seed()
                return False

    async def _start_rollouts(self, message: str) -> None:
        self.state.phase = Phase.POLICY
        self.state.source = "POLICY"
        self.state.message = message
        if self.control_cache is None and self._control_task is None:
            self.state.control_phase = Phase.POLICY.value
            self.state.control_source = "POLICY"
            self.state.control_message = "Independent control rollout"
        await self._publish_state()
        if self.control_cache is None and self._control_task is None:
            self._control_task = asyncio.create_task(
                self._run_control(),
                name="manual-dagger-control",
            )

    def _apply_cached_control(self, artifact: ControlArtifact | None) -> None:
        if artifact is None:
            self.state.control_phase = "caching"
            self.state.control_source = "BACKGROUND"
            self.state.control_step = 0
            self.state.control_outcome = ""
            self.state.control_eval_success = None
            self.state.control_message = "Control is queued for background generation"
            self.state.control_prefetch_ready = False
            self.state.control_cached = False
            self.state.control_video_frames = 0
            self.state.control_video_fps = 0
            return
        metadata = artifact.metadata
        self.state.control_phase = Phase.FINISHED.value
        self.state.control_source = "CACHE"
        self.state.control_step = int(metadata.get("steps", 0))
        self.state.control_outcome = str(metadata.get("outcome", ""))
        self.state.control_eval_success = bool(
            metadata.get(
                "eval_success",
                self.state.control_outcome == "success",
            )
        )
        self.state.control_message = "Persistent control cache is ready"
        self.state.control_prefetch_ready = True
        self.state.control_cached = True
        self.state.control_video_frames = int(metadata.get("video_frames", 0))
        self.state.control_video_fps = int(metadata.get("video_fps", 0))

    def _control_cache_metadata(self, runtime: EpisodeRuntime) -> dict[str, Any]:
        return {
            "role": "control",
            "task": runtime.task,
            "episode": runtime.episode,
            "take": 0,
            "seed": runtime.seed,
            "difficulty": runtime.difficulty,
            "prompt": runtime.prompt,
            "current_subtask": runtime.current_subtask,
            "outcome": "",
            "eval_success": None,
            "intervened": False,
            "scheduled_intervention_step": None,
            "scheduled_intervention_fired": False,
            "intervention_server_receive_ns": None,
            "first_expert_action_ns": None,
            "policy_cutoff_latency_ms": None,
            "first_expert_action_latency_ms": None,
            "fps": self.state.fps,
            "steps": runtime.step,
            "control_cache": True,
            "paired_enabled_kept": False,
        }

    async def _run_control_cache(self) -> None:
        assert self.control_cache is not None
        while not self._closed:
            _, _, spec = await self._control_cache_queue.get()
            if self._closed:
                return
            key = (spec.task, spec.episode)
            self._queued_control_keys.discard(key)
            if self.control_cache.screened(spec):
                self.state.control_cache_ready = self.control_cache.ready_count()
                continue
            runtime: EpisodeRuntime | None = None
            try:
                await asyncio.sleep(0)
                while (
                    self.state.phase
                    in {
                        Phase.PREPARING,
                        Phase.POLICY,
                        Phase.PAUSED,
                        Phase.INTERVENING,
                    }
                    or not self.state.policy_prefetch_ready
                ):
                    await asyncio.sleep(0.05)
                    if self._closed or self._control_abort_event.is_set():
                        return
                runtime = await self._run_blocking_on(
                    "control",
                    self.factory.prepare,
                    spec,
                    lambda _frame, _source: None,
                    self._control_abort_event,
                    lambda: self.state.fps,
                    "control",
                )
                self._control_cache_runtime = runtime
                await self._run_blocking_on("control", runtime.prime_policy)
                result = StepResult("running", runtime.step)
                while not self._control_abort_event.is_set():
                    while self.state.phase in {
                        Phase.PREPARING,
                        Phase.POLICY,
                        Phase.PAUSED,
                        Phase.INTERVENING,
                    }:
                        await asyncio.sleep(0.05)
                        if self._closed or self._control_abort_event.is_set():
                            break
                    if self._closed or self._control_abort_event.is_set():
                        break
                    result = await self._run_blocking_on(
                        "control",
                        runtime.policy_step,
                        self._never_takeover_event,
                        self._control_abort_event,
                        self._control_pause_event,
                    )
                    if result.status in {"terminal", "aborted"}:
                        break
                    await asyncio.sleep(0)
                metadata = self._control_cache_metadata(runtime)
                metadata["outcome"] = result.outcome
                metadata["eval_success"] = result.outcome == "success"
                metadata["steps"] = result.step
                keep = (
                    result.status == "terminal"
                    and self.control_cache.is_failure_outcome(result.outcome)
                )
                await self._run_blocking_on(
                    "control",
                    runtime.close,
                    keep,
                    metadata,
                )
                self._control_cache_runtime = None
                if keep:
                    artifact = await self._wait_for_cached_failure(spec)
                    if artifact is not None:
                        self.state.control_cache_ready = self.control_cache.ready_count()
                        if (
                            self._current_spec is not None
                            and self._current_spec.task == spec.task
                            and self._current_spec.episode == spec.episode
                        ):
                            self._apply_cached_control(artifact)
                        await self._publish_state()
                elif result.status == "terminal":
                    self.control_cache.record_success(spec, runtime.seed)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if runtime is not None:
                    with contextlib.suppress(Exception):
                        await self._run_blocking_on(
                            "control",
                            runtime.close,
                            False,
                            self._control_cache_metadata(runtime),
                        )
                self._control_cache_runtime = None
                self.control_cache.abandon(
                    spec,
                    f"{type(error).__name__}: {error}",
                )
                if (
                    self._current_spec is not None
                    and self._current_spec.task == spec.task
                    and self._current_spec.episode == spec.episode
                ):
                    self.state.control_phase = Phase.ERROR.value
                    self.state.control_message = (
                        f"Background control failed: {type(error).__name__}: {error}"
                    )
                    await self._publish_state()

    async def _wait_for_cached_failure(
        self,
        spec: EpisodeSpec,
        timeout: float = 30.0,
    ) -> ControlArtifact | None:
        assert self.control_cache is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            artifact = self.control_cache.lookup_failure_spec(spec)
            if artifact is not None:
                return artifact
            await asyncio.sleep(0.1)
        return self.control_cache.lookup_failure_spec(spec)

    async def _run_control(self) -> None:
        assert self._control_runtime is not None
        await self._await_prime("control")
        while not self._control_abort_event.is_set():
            started = time.monotonic()
            result = await self._run_blocking_on(
                "control",
                self._control_runtime.policy_step,
                self._never_takeover_event,
                self._control_abort_event,
                self._control_pause_event,
            )
            self.state.control_step = result.step
            await self._flush_frame("control", force=True)
            if result.status == "terminal":
                self.state.control_phase = Phase.FINISHED.value
                self.state.control_outcome = result.outcome
                self.state.control_eval_success = result.outcome == "success"
                self.state.control_message = "Control rollout finished"
                await self._publish_state()
                return
            if result.status == "aborted":
                self.state.control_phase = Phase.FINISHED.value
                self.state.control_message = "Control rollout stopped"
                await self._publish_state()
                return
            remaining = 1.0 / self.state.fps - (time.monotonic() - started)
            if remaining > 0:
                await asyncio.sleep(remaining)

    async def _run_intervention(self, *, scheduled: bool = False) -> str:
        assert self._runtime is not None
        # A policy step can observe the latch before the matching queued key is
        # consumed. Consume that key here so it cannot start a second subtask
        # automatically after the expert pauses.
        retained_commands = []
        while not self._commands.empty():
            request = self._commands.get_nowait()
            if request.command != Command.INTERVENE:
                retained_commands.append(request)
        for request in retained_commands:
            self._commands.put_nowait(request)
        if scheduled:
            self.state.scheduled_intervention_pending = False
            self.state.scheduled_intervention_fired = True
            self._intervention_receive_ns = time.monotonic_ns()
        self.state.phase = Phase.INTERVENING
        self.state.source = "EXPERT"
        self.state.intervened = True
        now_ns = time.monotonic_ns()
        if self._intervention_receive_ns is not None:
            self.state.policy_cutoff_latency_ms = (now_ns - self._intervention_receive_ns) / 1_000_000.0
        self.state.message = (
            f"Scheduled control-step {self.state.scheduled_intervention_step} takeover"
            if scheduled
            else "Policy stopped; official expert is taking over"
        )
        await self._publish_state()
        result = await self._run_blocking_on(
            "enabled",
            self._runtime.intervene,
            self._abort_event,
        )
        self.state.step = result.step
        self.state.current_subtask = self._runtime.current_subtask
        await self._flush_frame("enabled", force=True)
        if result.status == "aborted":
            if self._reset_requested:
                return "reset"
            if self._next_requested:
                await self._close_episode(
                    False,
                    "Skipped by n during intervention",
                    keep_control=False,
                )
                return "closed"
            if self._block_requested:
                await self._block_current_seed()
                return "closed"
            await self._close_episode(
                False,
                "Discarded by p during intervention",
                keep_control=False,
            )
            return "closed"
        if result.status in {"subtask_completed", "subtask_incomplete"}:
            self._takeover_event.clear()
            self._pause_event.set()
            self.state.phase = Phase.PAUSED
            self.state.source = "PAUSED"
            detail = (
                "Subtask completed" if result.status == "subtask_completed"
                else "Expert stopped without completing the subtask"
            )
            self.state.message = f"{detail}; z: policy rollout, i: one expert subtask"
            await self._publish_state()
            return "paused"
        await self._finish(result)
        return "finished"

    async def _finish(self, result: StepResult) -> None:
        self.state.phase = Phase.FINISHED
        self.state.outcome = result.outcome
        self.state.eval_success = result.outcome == "success"
        self.state.message = (
            "Episode ended: explicit operator review required; "
            "press c to keep or d to discard"
        )
        await self._publish_state()

    async def _close_episode(
        self,
        keep: bool,
        message: str,
        *,
        keep_control: bool = True,
    ) -> dict[str, Any] | None:
        assert self._runtime is not None
        if self.control_cache is None:
            assert self._control_runtime is not None
        self._drain_closed_episode_commands()
        self.state.phase = Phase.FINALIZING
        self.state.source = "IDLE"
        self.state.message = message
        if (
            self.control_cache is None
            and self._control_task is not None
            and not self._control_task.done()
        ):
            self.state.control_message = "Control continues to its own terminal state"
        await self._publish_state()
        await self._await_prime("enabled")
        close_enabled = self._run_blocking_on(
            "enabled",
            self._runtime.close,
            keep,
            self._metadata("enabled"),
        )
        if self.control_cache is None:
            await self._await_prime("control")
            if self._control_task is not None:
                await self._control_task
                self._control_task = None
            assert self._control_runtime is not None
            results = await asyncio.gather(
                close_enabled,
                self._run_blocking_on(
                    "control",
                    self._control_runtime.close,
                    keep_control,
                    {
                        **self._metadata("control"),
                        "paired_enabled_kept": keep,
                    },
                ),
            )
            enabled_result = results[0]
        else:
            enabled_result = await close_enabled
        self._runtime = None
        self._control_runtime = None
        self._prime_tasks = {}
        return enabled_result

    async def _block_current_seed(self) -> None:
        assert self.collection is not None
        assert self._current_spec is not None
        assert self.state.seed is not None
        self.collection.block(self._current_spec, self.state.seed)
        await self._close_episode(
            False,
            f"Blocked seed {self.state.seed}",
            keep_control=False,
        )

    async def _reset_enabled(self) -> None:
        assert self._runtime is not None
        if self.control_cache is None:
            assert self._control_runtime is not None
        assert self._current_spec is not None
        self.state.phase = Phase.FINALIZING
        self.state.source = "IDLE"
        self.state.message = "Resetting enabled on the same seed; control continues"
        await self._publish_state()
        await self._await_prime("enabled")
        initialization_replay = self._runtime.initialization_replay()
        await self._run_blocking_on(
            "enabled",
            self._runtime.close,
            False,
            {**self._metadata("enabled"), "reset_discarded": True},
        )
        self._drain_closed_episode_commands()
        self._takeover_event.clear()
        self._abort_event.clear()
        self._pause_event.clear()
        self._reset_requested = False
        self._next_requested = False
        self._block_requested = False
        self._intervention_receive_ns = None
        self._first_expert_action_ns = None
        self.state.step = 0
        self.state.outcome = ""
        self.state.eval_success = None
        self.state.intervened = False
        self.state.policy_cutoff_latency_ms = None
        self.state.first_expert_action_latency_ms = None
        self.state.policy_prefetch_ready = False
        self.state.enabled_generation += 1
        self._runtime = await self._run_blocking_on(
            "enabled",
            self.factory.prepare,
            self._current_spec,
            functools.partial(self._backend_frame, "enabled"),
            self._abort_event,
            lambda: self.state.fps,
            "enabled",
            initialization_replay,
        )
        if self.control_cache is None:
            assert self._control_runtime is not None
            if (
                self._runtime.seed != self._control_runtime.seed
                or self._runtime.prompt != self._control_runtime.prompt
            ):
                raise RuntimeError("enabled reset changed episode identity")
        self.state.step = self._runtime.step
        self.state.current_subtask = self._runtime.current_subtask
        self.state.phase = Phase.READY
        self.state.source = "IDLE"
        self.state.message = (
            "Enabled reset; scheduled replay starts automatically"
            if self.state.scheduled_intervention_pending
            else "Enabled reset on the same seed; press s to restart"
        )
        await self.publish_frame("enabled", self._runtime.initial_frame(), 0)
        self.state.frame_sequence += 1
        await self._publish_state()
        self._prime_tasks["enabled"] = asyncio.create_task(
            self._prime_runtime("enabled", self._runtime),
            name="manual-dagger-prime-enabled",
        )

    def _drain_closed_episode_commands(self) -> None:
        while True:
            try:
                request = self._commands.get_nowait()
            except asyncio.QueueEmpty:
                return
            if request.command == Command.FPS_UP:
                self.state.fps = min(self.maximum_fps, self.state.fps + 1)
            elif request.command == Command.FPS_DOWN:
                self.state.fps = max(self.minimum_fps, self.state.fps - 1)

    def _scheduled_intervention_due(self) -> bool:
        target = self.state.scheduled_intervention_step
        return (
            self.state.scheduled_intervention_pending
            and target is not None
            and self.state.step >= target
        )

    def _cancel_scheduled_intervention(self, *, clear_abort: bool = False) -> None:
        self.state.scheduled_intervention_step = None
        self.state.scheduled_intervention_pending = False
        self.state.scheduled_intervention_fired = False
        self._scheduled_autostart = False
        self._reset_requested = False
        if clear_abort:
            self._abort_event.clear()

    async def _adjust_fps(self, command: Command) -> None:
        delta = 1 if command == Command.FPS_UP else -1
        self.state.fps = min(
            self.maximum_fps,
            max(self.minimum_fps, self.state.fps + delta),
        )
        self.state.message = f"Playback rate: {self.state.fps} FPS"
        await self._publish_state()

    async def _wait_for_command_or_tick(self) -> CommandRequest | None:
        try:
            return await asyncio.wait_for(
                self._commands.get(),
                timeout=1.0 / self.state.fps,
            )
        except TimeoutError:
            return None

    async def _prime_runtime(self, role: str, runtime: EpisodeRuntime) -> None:
        await self._run_blocking_on(role, runtime.prime_policy)
        if role == "enabled":
            self.state.policy_prefetch_ready = True
        else:
            self.state.control_prefetch_ready = True
        await self._publish_state()

    async def _await_prime(self, role: str) -> None:
        task = self._prime_tasks.get(role)
        if task is not None:
            await task

    async def _run_blocking_on(
        self,
        role: str,
        function: Callable[..., Any],
        *args: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        call = functools.partial(function, *args)
        return await loop.run_in_executor(self._simulator_executors[role], call)

    def _metadata(self, role: str) -> dict[str, Any]:
        control = role == "control"
        return {
            "role": role,
            "task": self.state.task,
            "episode": self.state.episode,
            "take": (0 if self._current_spec is None else self._current_spec.take),
            "seed": self.state.seed,
            "difficulty": self.state.difficulty,
            "prompt": self.state.prompt,
            "current_subtask": self.state.current_subtask,
            "outcome": (self.state.control_outcome if control else self.state.outcome),
            "eval_success": (
                self.state.control_eval_success if control else self.state.eval_success
            ),
            "intervened": (False if control else self.state.intervened),
            "scheduled_intervention_step": (
                None if control else self.state.scheduled_intervention_step
            ),
            "scheduled_intervention_fired": (
                False if control else self.state.scheduled_intervention_fired
            ),
            "intervention_server_receive_ns": (None if control else self._intervention_receive_ns),
            "first_expert_action_ns": (None if control else self._first_expert_action_ns),
            "policy_cutoff_latency_ms": (None if control else self.state.policy_cutoff_latency_ms),
            "first_expert_action_latency_ms": (
                None if control else self.state.first_expert_action_latency_ms
            ),
            "fps": self.state.fps,
            "steps": (self.state.control_step if control else self.state.step),
        }


class FakeEpisode:
    def __init__(
        self,
        spec: EpisodeSpec,
        frame_callback: Any,
        *,
        inference_delay: float,
        terminal_step: int,
        role: str,
        control_outcome: str,
    ) -> None:
        self.task = spec.task
        self.episode = spec.episode
        self.take = spec.take
        self.seed = 10_000 + spec.episode
        self.difficulty = "simulation"
        self.prompt = (
            "This is the complete simulated prompt. Judge the policy trajectory "
            "and press i exactly when the official expert should take over."
        )
        self.step = 0
        self.frame_callback = frame_callback
        self.inference_delay = inference_delay
        self.terminal_step = terminal_step
        self.closed_with: bool | None = None
        self.policy_actions = 0
        self.role = role
        self.control_outcome = control_outcome
        self._prefetched = False
        self.replayed_initialization = False

    def _frame(self, source: str) -> np.ndarray:
        frame = np.full((240, 640, 3), 245, dtype=np.uint8)
        color = (220, 40, 40) if source == "EXPERT" else (40, 160, 70)
        frame[:, :320] = color
        frame[:, 320:] = np.roll(color, 1)
        return frame

    def initial_frame(self) -> np.ndarray:
        return self._frame("POLICY")

    def initialization_replay(self) -> object:
        return (self.task, self.episode)

    @property
    def current_subtask(self) -> str:
        return f"Simulated subtask at step {self.step}"

    def prime_policy(self) -> None:
        time.sleep(self.inference_delay)
        self._prefetched = True

    def policy_step(
        self,
        takeover_event: threading.Event,
        abort_event: threading.Event,
        pause_event: threading.Event,
    ) -> StepResult:
        if self._prefetched:
            self._prefetched = False
        else:
            time.sleep(self.inference_delay)
        if takeover_event.is_set():
            return StepResult("takeover_requested", self.step)
        if abort_event.is_set():
            return StepResult("aborted", self.step)
        if pause_event.is_set():
            return StepResult("paused", self.step)
        self.policy_actions += 1
        self.step += 1
        self.frame_callback(self._frame("POLICY"), "POLICY")
        if self.step >= self.terminal_step:
            outcome = self.control_outcome if self.role == "control" else "success"
            return StepResult("terminal", self.step, outcome)
        return StepResult("running", self.step)

    def intervene(self, abort_event: threading.Event) -> StepResult:
        for _ in range(3):
            if abort_event.is_set():
                return StepResult("aborted", self.step)
            self.step += 1
            self.frame_callback(self._frame("EXPERT"), "EXPERT")
            time.sleep(0.01)
        return StepResult("terminal", self.step, "success")

    def close(self, keep: bool, metadata: dict[str, Any]) -> dict[str, Any]:
        del metadata
        self.closed_with = keep
        return {"status": "published" if keep else "discarded"}


class FakeBackendFactory:
    def __init__(
        self,
        *,
        inference_delay: float = 0.01,
        terminal_step: int = 8,
        control_outcome: str = "success",
    ):
        self.inference_delay = inference_delay
        self.terminal_step = terminal_step
        self.control_outcome = control_outcome
        self.episodes: list[FakeEpisode] = []

    def prepare(
        self,
        spec: EpisodeSpec,
        frame_callback: Any,
        abort_event: threading.Event,
        fps_provider: Any,
        role: str = "enabled",
        initialization_replay: object | None = None,
    ) -> FakeEpisode:
        del abort_event, fps_provider
        episode = FakeEpisode(
            spec,
            frame_callback,
            inference_delay=self.inference_delay,
            terminal_step=self.terminal_step,
            role=role,
            control_outcome=self.control_outcome,
        )
        episode.replayed_initialization = initialization_replay is not None
        self.episodes.append(episode)
        return episode

    def close(self) -> None:
        return None

    def episode_count(self, task: str) -> int:
        del task
        return 100

    def episode_for_seed(self, task: str, seed: int) -> int:
        del task
        episode = seed - 10_000
        if not 0 <= episode < 100:
            raise ValueError(f"seed {seed} is not legal for simulated task")
        return episode

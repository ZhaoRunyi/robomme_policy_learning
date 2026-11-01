"""manual_dagger / backend: shared behavior owned by this layer."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np
import json
import contextlib
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
import h5py
from manual_dagger.state import (  # noqa: E402
    EpisodeSpec,
    PreparationCancelled,
    StepResult,
)
from manual_dagger.storage import StorageProcess  # noqa: E402


ROBOMME_EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
POLICY_REPO = ROBOMME_EXAMPLE_ROOT.parents[1]
for import_path in (
    ROBOMME_EXAMPLE_ROOT,
    POLICY_REPO / "src",
    POLICY_REPO / "packages/openpi-client/src",
    POLICY_REPO / "third_party/robomme_benchmark/src",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))




import env_runner as official_env_runner  # noqa: E402
from eval import Args
from robottt_eval import RoboTTTEpisodeEvaluator  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402
from utils import EpisodeState  # noqa: E402
from dagger_review.env_runner import DAggerEnvRunner, DAggerRecordWrapper
from manual_dagger.web import overlay_target_markers


class EpisodeAbort(BaseException):
    """Leave native planner stacks immediately after p is received."""


@dataclass(frozen=True)
class _InitializationEvent:
    kind: str
    demonstration: bool
    action: np.ndarray | None = None
    side_effects: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class _InitializationReplay:
    task: str
    episode: int
    prompt: str
    pretrajectory: dict[str, Any]
    events: tuple[_InitializationEvent, ...]
    side_effects: tuple[tuple[str, Any], ...] = ()
    policy_action_plan: tuple[np.ndarray, ...] = ()
    subgoal: str | None = None
    last_subgoal: str | None = None


def _camera_row(base: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    base_array = np.asarray(base, dtype=np.uint8)
    wrist_array = np.asarray(wrist, dtype=np.uint8)
    if wrist_array.shape[:2] != base_array.shape[:2]:
        wrist_array = cv2.resize(
            wrist_array,
            (base_array.shape[1], base_array.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.concatenate([base_array, wrist_array], axis=1)


def _copy_replay_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if hasattr(value, "clone"):
        return value.clone()
    return np.asarray(value).copy()


_INITIALIZATION_SIDE_EFFECTS = (
    "swing_qpos",
    "after_demo",
    "reset_in_proecess",
)


def _capture_initialization_side_effects(environment: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, _copy_replay_value(getattr(environment, name)))
        for name in _INITIALIZATION_SIDE_EFFECTS
        if hasattr(environment, name)
    )


def _restore_initialization_side_effects(
    environment: Any,
    side_effects: tuple[tuple[str, Any], ...],
) -> None:
    for name, value in side_effects:
        setattr(environment, name, _copy_replay_value(value))


class InteractiveDAggerRecordWrapper(DAggerRecordWrapper):
    """Keep native HDF5 semantics; stream a clean two-camera video separately."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_video = False
        self.abort_event: threading.Event | None = None
        self.frame_sink: Any = None
        self.first_expert_action_callback: Any = None
        self.expert_observation_callback: Any = None
        self.fps_provider: Any = None
        self._expert_action_seen = False
        self.review_stage = "POLICY"
        self.review_decision: Any | None = None

    def _video_should_record(self, current_task_name: str) -> bool:
        # RobommeRecordWrapper currently gates both HDF5 buffering and its legacy
        # video renderer on this method. Keep HDF5 capture enabled while the
        # manual UI owns video rendering through frame_sink.
        return current_task_name != "NO RECORD"

    def _video_append_step_frame(self, frame: Any, no_object_flag: bool) -> None:
        del frame, no_object_flag

    def set_expert(self, decision: Any, stage: str) -> None:
        self.action_source = 1
        self.review_stage = stage
        self.review_decision = decision

    def step(self, action: Any) -> Any:
        if self.abort_event is not None and self.abort_event.is_set():
            raise EpisodeAbort()
        result = super().step(action)
        observation = result[0]
        base = observation["sensor_data"]["base_camera"]["rgb"][0]
        wrist = observation["sensor_data"]["hand_camera"]["rgb"][0]
        if hasattr(base, "detach"):
            base = base.detach().cpu().numpy()
        if hasattr(wrist, "detach"):
            wrist = wrist.detach().cpu().numpy()
        segmentation = observation["sensor_data"]["base_camera"]["segmentation"]
        if hasattr(segmentation, "detach"):
            segmentation = segmentation.detach().cpu().numpy()
        base = overlay_target_markers(
            base,
            segmentation,
            self,
            self.env_id,
        )
        source = "EXPERT" if self.action_source else "POLICY"
        if source == "EXPERT" and not self._expert_action_seen:
            self._expert_action_seen = True
            if self.first_expert_action_callback is not None:
                self.first_expert_action_callback(time.monotonic_ns())
        if self.frame_sink is not None:
            self.frame_sink(_camera_row(base, wrist), source)
        if source == "EXPERT" and self.expert_observation_callback is not None:
            self.expert_observation_callback(observation, action)
        if source == "EXPERT" and self.fps_provider is not None:
            fps = max(1, int(self.fps_provider()))
            if self.abort_event is not None and self.abort_event.wait(1.0 / fps):
                raise EpisodeAbort()
        return result


class InteractiveDAggerEnvRunner(DAggerEnvRunner):
    record_wrapper_class = InteractiveDAggerRecordWrapper


class RoboTTTEpisode:
    def __init__(
        self,
        factory: "RoboTTTBackendFactory",
        spec: EpisodeSpec,
        frame_callback: Any,
        abort_event: threading.Event,
        fps_provider: Any,
        role: str,
        initialization_replay: _InitializationReplay | None = None,
    ) -> None:
        self.factory = factory
        self.task = spec.task
        self.episode = spec.episode
        self.take = spec.take
        self.abort_event = abort_event
        self.frame_callback = frame_callback
        self.role = role
        self.episode_id = f"{self.task}-ep{self.episode}-{role}-{uuid.uuid4().hex[:10]}"
        self.staging_root = factory.staging_root / self.episode_id
        self.staging_root.mkdir(parents=True, exist_ok=False)
        self.factory.storage.begin(self.episode_id, self.staging_root)
        self.env_runner = InteractiveDAggerEnvRunner(
            self.task,
            str(self.staging_root / "unused-native-videos"),
            max_steps=factory.args.max_steps,
            dagger_dir=str(self.staging_root / "dataset"),
            dataset="train",
        )
        self.env_runner.make_env(self.episode)
        self.recorder: InteractiveDAggerRecordWrapper = self.env_runner.recorder
        self.state = EpisodeState()
        self.frame_steps: list[int] = []
        self.first_expert_action_ns: int | None = None
        self.recorder.abort_event = abort_event
        self.recorder.frame_sink = self._record_frame
        self.recorder.first_expert_action_callback = self._first_expert_action
        self.recorder.fps_provider = fps_provider
        self.seed, _ = self.env_runner.env_builder.resolve_episode(self.episode)
        if self.seed is None:
            raise RuntimeError(f"episode {self.episode} has no seed for {self.task}")
        self.seed = int(self.seed)
        self.difficulty = str(self.env_runner.difficulty)
        self.client = factory.client(role)
        try:
            if initialization_replay is None:
                response = self.client.reset(factory.args.robottt_mode)
                deadline = time.monotonic() + 300.0
                while not response.get("reset_finished", False):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("policy reset did not finish within 300 seconds")
                    time.sleep(0.05)
            else:
                response = self.client.restore_reset_point()
                if not response.get("restore_reset_point_finished", False):
                    raise RuntimeError("policy reset point restore did not finish")
            if initialization_replay is None:
                pretrajectory, initialization_events = self._capture_initialization()
                self._initialization_replay = _InitializationReplay(
                    task=self.task,
                    episode=self.episode,
                    prompt=str(pretrajectory["task_goal"]),
                    pretrajectory=pretrajectory,
                    events=initialization_events,
                    side_effects=_capture_initialization_side_effects(
                        self.env_runner.env.unwrapped
                    ),
                )
            else:
                if (
                    initialization_replay.task != self.task
                    or initialization_replay.episode != self.episode
                ):
                    raise RuntimeError("initialization replay changed episode identity")
                pretrajectory = self._replay_initialization(initialization_replay)
                self._initialization_replay = initialization_replay
        except BaseException:
            with contextlib.suppress(Exception):
                self.env_runner.close_env()
            with contextlib.suppress(Exception):
                self.factory.storage.finish(
                    self.episode_id,
                    False,
                    [],
                    {"episode_id": self.episode_id},
                )
            raise
        self.prompt = str(pretrajectory["task_goal"])
        self.state.image_buffer.extend(pretrajectory["images"])
        self.state.wrist_image_buffer.extend(pretrajectory["wrist_images"])
        self.state.state_buffer.extend(pretrajectory["states"])
        self.state.action_buffer.extend(pretrajectory["actions"])
        self.state.exec_start_idx = len(self.state.image_buffer) - 1
        self.image, self.wrist_image, self.robot_state = self.state.get_current_obs()
        initial_segmentation = None
        if self.recorder.buffer:
            initial_segmentation = self.recorder.buffer[-1]["obs"].get(
                "front_camera_segmentation"
            )
        initial_base = overlay_target_markers(
            self.image,
            initial_segmentation,
            self.recorder,
            self.task,
        )
        self._initial_frame = _camera_row(initial_base, self.wrist_image)
        self.subgoal = (
            None if initialization_replay is None else initialization_replay.subgoal
        )
        self.last_subgoal = (
            None if initialization_replay is None else initialization_replay.last_subgoal
        )
        self._initial_policy_action_plan: tuple[np.ndarray, ...] | None = None
        self._initial_subgoal: str | None = None
        self._initial_last_subgoal: str | None = None
        if initialization_replay is not None:
            if not initialization_replay.policy_action_plan:
                raise RuntimeError("initialization replay has no prefetched action chunk")
            self._initial_policy_action_plan = initialization_replay.policy_action_plan
            self._initial_subgoal = initialization_replay.subgoal
            self._initial_last_subgoal = initialization_replay.last_subgoal
            self.state.action_plan.extend(
                action.copy() for action in initialization_replay.policy_action_plan
            )
        self.outcome = "unknown"
        self.closed = False
        self.evaluator = factory.evaluator(role)
        self.evaluator.episode_metrics = {
            "reset_time_ms": response.get("reset_time_ms", 0),
            "video_blocks_seen": len(
                range(
                    self.state.exec_start_idx % 16,
                    self.state.exec_start_idx,
                    16,
                )
            ),
            "prefill_time_ms": 0,
            "infer_time_ms": [],
        }

    def initialization_replay(self) -> _InitializationReplay:
        if self._initial_policy_action_plan is None:
            raise RuntimeError("policy reset point is not ready")
        return replace(
            self._initialization_replay,
            policy_action_plan=self._initial_policy_action_plan,
            subgoal=self._initial_subgoal,
            last_subgoal=self._initial_last_subgoal,
        )

    def _capture_initialization(
        self,
    ) -> tuple[dict[str, Any], tuple[_InitializationEvent, ...]]:
        demonstration_wrapper = self.env_runner.env.env
        original_step_batch = demonstration_wrapper._step_batch
        original_evaluate = demonstration_wrapper.evaluate
        events: list[_InitializationEvent] = []
        step_depth = 0

        def traced_step_batch(action: Any) -> Any:
            nonlocal step_depth
            demonstration = bool(
                getattr(
                    demonstration_wrapper.unwrapped,
                    "demonstration_record_traj",
                    False,
                )
            )
            step_depth += 1
            try:
                result = original_step_batch(action)
            finally:
                step_depth -= 1
            events.append(
                _InitializationEvent(
                    "action",
                    demonstration,
                    np.asarray(action).copy(),
                    _capture_initialization_side_effects(
                        demonstration_wrapper.unwrapped
                    ),
                )
            )
            return result

        def traced_evaluate(*args: Any, **kwargs: Any) -> Any:
            result = original_evaluate(*args, **kwargs)
            if step_depth == 0 and kwargs.get("solve_complete_eval", False):
                events.append(
                    _InitializationEvent(
                        "evaluate",
                        bool(
                            getattr(
                                demonstration_wrapper.unwrapped,
                                "demonstration_record_traj",
                                False,
                            )
                        ),
                        side_effects=_capture_initialization_side_effects(
                            demonstration_wrapper.unwrapped
                        ),
                    )
                )
            return result

        demonstration_wrapper._step_batch = traced_step_batch
        demonstration_wrapper.evaluate = traced_evaluate
        original_frame_sink = self.recorder.frame_sink
        if self.role == "enabled":
            self.recorder.frame_sink = None
        try:
            pretrajectory = self.env_runner.get_init_obs()
        finally:
            demonstration_wrapper._step_batch = original_step_batch
            demonstration_wrapper.evaluate = original_evaluate
            self.recorder.frame_sink = original_frame_sink
        return pretrajectory, tuple(events)

    def _replay_initialization(
        self,
        replay: _InitializationReplay,
    ) -> dict[str, Any]:
        demonstration_wrapper = self.env_runner.env.env
        for name, value in {
            "last_subgoal_segment": None,
            "latched_replacements": None,
            "_failed_match_save_count": 0,
            "steps_without_demonstration": 0,
            "_prev_ee_quat_wxyz": None,
            "_prev_ee_rpy_xyz": None,
        }.items():
            setattr(demonstration_wrapper, name, value)
        demonstration_wrapper.episode_success = False
        self.recorder.frame_sink = None
        self.recorder.reset()
        last_batch = None
        try:
            for event in replay.events:
                demonstration_wrapper.unwrapped.demonstration_record_traj = (
                    event.demonstration
                )
                if event.kind == "evaluate":
                    demonstration_wrapper.evaluate(solve_complete_eval=True)
                    _restore_initialization_side_effects(
                        demonstration_wrapper.unwrapped,
                        event.side_effects,
                    )
                    continue
                if event.kind != "action" or event.action is None:
                    raise RuntimeError(f"invalid initialization event {event.kind!r}")
                last_batch = demonstration_wrapper._step_batch(event.action.copy())
                _restore_initialization_side_effects(
                    demonstration_wrapper.unwrapped,
                    event.side_effects,
                )
        finally:
            demonstration_wrapper.unwrapped.demonstration_record_traj = False
            self.recorder.frame_sink = self._record_frame
        if last_batch is None:
            raise RuntimeError("initialization replay contains no actions")
        info_batch = last_batch[4]
        self.env_runner.info = demonstration_wrapper._flatten_info_batch(info_batch)
        if self.env_runner.info.get("status") == "error":
            raise RuntimeError(
                "initialization replay failed: "
                f"{self.env_runner.info.get('error_message', 'unknown error')}"
            )
        _restore_initialization_side_effects(
            demonstration_wrapper.unwrapped,
            replay.side_effects,
        )
        self.env_runner.task_goal = replay.prompt
        return replay.pretrajectory

    @property
    def step(self) -> int:
        return int(self.state.count)

    @property
    def current_subtask(self) -> str:
        environment = self.env_runner.env.unwrapped
        task_list = getattr(environment, "task_list", ())
        index = max(0, int(getattr(environment, "current_task_index", 0)))
        if index >= len(task_list):
            return "Complete"
        task = task_list[index]
        if isinstance(task, dict):
            return str(task.get("name") or task.get("description") or f"Subtask {index + 1}")
        return str(task)

    def initial_frame(self) -> np.ndarray:
        return self._initial_frame.copy()

    def _first_expert_action(self, timestamp_ns: int) -> None:
        self.first_expert_action_ns = timestamp_ns

    def _record_frame(self, frame: np.ndarray, source: str) -> None:
        self.frame_steps.append(self.state.count + 1)
        self.factory.storage.frame(self.episode_id, frame)
        self.frame_callback(frame, source)

    def _fill_action_plan(self) -> bool:
        if self.state.action_plan:
            return True
        self.subgoal = None
        actions = self.evaluator.get_action_chunk(
            self.client,
            self.state,
            self.image,
            self.wrist_image,
            self.robot_state,
            self.prompt,
            self.subgoal,
            exec_horizon=self.factory.args.obs_horizon,
        )
        self.state.action_plan.extend(actions)
        self.state.clear_buffers()
        self.last_subgoal = self.subgoal
        return True

    def prime_policy(self) -> None:
        if not self.state.action_plan and not self._fill_action_plan():
            raise RuntimeError("first policy action chunk could not be prefetched")
        if self._initial_policy_action_plan is None:
            response = self.client.save_reset_point()
            if not response.get("save_reset_point_finished", False):
                raise RuntimeError("policy reset point save did not finish")
            self._initial_policy_action_plan = tuple(
                np.asarray(action).copy() for action in self.state.action_plan
            )
            self._initial_subgoal = self.subgoal
            self._initial_last_subgoal = self.last_subgoal

    def policy_step(
        self,
        takeover_event: threading.Event,
        abort_event: threading.Event,
        pause_event: threading.Event,
    ) -> StepResult:
        if abort_event.is_set():
            return StepResult("aborted", self.step)
        if takeover_event.is_set():
            return StepResult("takeover_requested", self.step)
        if pause_event.is_set():
            return StepResult("paused", self.step)
        if not self._fill_action_plan():
            return StepResult("terminal", self.step, self.outcome)
        # This check is deliberately after inference: an i received while the
        # policy RPC is in flight prevents the inferred action from executing.
        if takeover_event.is_set():
            self.state.action_plan.clear()
            return StepResult("takeover_requested", self.step)
        if abort_event.is_set():
            self.state.action_plan.clear()
            return StepResult("aborted", self.step)
        if pause_event.is_set():
            return StepResult("paused", self.step)
        action = np.asarray(self.state.action_plan.popleft())
        self.recorder.action_source = 0
        try:
            observation, stop, self.outcome = self.env_runner.step(action)
        except EpisodeAbort:
            return StepResult("aborted", self.step)
        self.state.count += 1
        if stop:
            return StepResult("terminal", self.step, self.outcome)
        self.image, self.wrist_image, self.robot_state = observation
        self.state.add_observation(
            self.image,
            self.wrist_image,
            self.robot_state,
            action,
        )
        if self.state.count > self.factory.args.max_steps:
            self.outcome = "timeout"
            return StepResult("terminal", self.step, self.outcome)
        return StepResult("running", self.step)

    def intervene(self, abort_event: threading.Event) -> StepResult:
        self.state.action_plan.clear()
        if abort_event.is_set():
            return StepResult("aborted", self.step)
        self.recorder.action_source = 1
        self.recorder.review_stage = "native_expert_subtask"
        self.recorder.review_decision = None
        self.recorder._expert_action_seen = False
        environment = self.env_runner.env.unwrapped
        start_subtask = int(np.asarray(environment.timestep).item())
        self.recorder.expert_observation_callback = self._observe_expert_action
        try:
            success = bool(self.env_runner.finish_with_expert(max_subtasks=1))
        except EpisodeAbort:
            return StepResult("aborted", self.step)
        finally:
            self.recorder.expert_observation_callback = None
            self.recorder.action_source = 0
        if success:
            self.outcome = "success"
            return StepResult("terminal", self.step, self.outcome)
        if self.state.count >= self.factory.args.max_steps:
            self.outcome = "timeout"
            return StepResult("terminal", self.step, self.outcome)
        advanced = int(np.asarray(environment.timestep).item()) > start_subtask
        return StepResult("subtask_completed" if advanced else "subtask_incomplete", self.step)

    def _observe_expert_action(self, observation: Any, action: Any) -> None:
        def as_array(value: Any) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        self.image = as_array(observation["sensor_data"]["base_camera"]["rgb"])[0]
        self.wrist_image = as_array(observation["sensor_data"]["hand_camera"]["rgb"])[0]
        joints = as_array(self.env_runner.env.unwrapped.agent.robot.qpos).reshape(-1)
        gripper = joints[7:9] if joints.size >= 9 else np.zeros(2)
        self.robot_state = official_env_runner.pack_state(joints[:7], gripper)
        action = as_array(action).reshape(-1)
        if action.size == 7:
            action = np.r_[action, -1.0]
        self.state.add_observation(self.image, self.wrist_image, self.robot_state, action)
        self.state.count += 1

    def close(self, keep: bool, metadata: dict[str, Any]) -> dict[str, Any]:
        if self.closed:
            return
        self.closed = True
        self.recorder.episode_success = bool(keep and metadata.get("eval_success") is True)
        dataset_path = Path(self.recorder.dataset_path)
        self.env_runner.close_env()
        h5_paths: list[Path] = []
        if keep and dataset_path.is_file():
            h5_paths.append(dataset_path)
            with h5py.File(dataset_path, "a") as output:
                episode_group = output[f"episode_{self.episode}"]
                setup = episode_group["setup"]
                for key, value in {
                    "human_intervened": bool(metadata["intervened"]),
                    "official_eval_success": bool(metadata["eval_success"]),
                    "human_intervention_server_receive_ns": (
                        -1
                        if metadata["intervention_server_receive_ns"] is None
                        else int(metadata["intervention_server_receive_ns"])
                    ),
                    "human_first_expert_action_ns": (
                        -1 if self.first_expert_action_ns is None else int(self.first_expert_action_ns)
                    ),
                }.items():
                    if key in setup:
                        del setup[key]
                    setup.create_dataset(key, data=value)
        artifact_stem = (
            f"{self.task}_ep{self.episode}_seed{self.seed}"
            f"_take{self.take:04d}_{self.role}"
        )
        payload = {
            **metadata,
            "episode_id": self.episode_id,
            "artifact_stem": artifact_stem,
            "take": self.take,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "outcome": self.outcome,
            "first_expert_action_ns": self.first_expert_action_ns,
            "frame_steps": self.frame_steps,
            "kept": keep,
        }
        (self.staging_root / "episode.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.factory.storage.finish(
            self.episode_id,
            keep,
            h5_paths,
            payload,
        )


class RoboTTTBackendFactory:
    def __init__(
        self,
        storage: StorageProcess,
        staging_root: Path,
        *,
        policy_host: str,
        policy_port: int,
        max_steps: int = 1300,
    ) -> None:
        self.storage = storage
        self.staging_root = staging_root
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self._episode_metadata: dict[str, list[tuple[int, int]]] = {}
        self.args = Args(
            host=policy_host,
            port=policy_port,
            obs_horizon=16,
            max_steps=max_steps,
            save_dir=str(staging_root),
            overwrite=True,
            use_history=True,
            policy_name="robottt_manual_dagger",
            model_seed=7,
            model_ckpt_id=60000,
            robottt_mode="normal",
            dagger_dir=str(staging_root),
            dagger_dataset="train",
        )
        self._evaluators = {
            role: RoboTTTEpisodeEvaluator(self.args, staging_root)
            for role in ("enabled", "control")
        }
        self._clients: dict[str, Any] = {}

    def _metadata_for_task(self, task: str) -> list[tuple[int, int]]:
        if task not in self._episode_metadata:
            builder = official_env_runner.BenchmarkEnvBuilder(
                env_id=task,
                dataset="train",
                action_space="joint_angle",
                gui_render=False,
                max_steps=self.args.max_steps,
            )
            entries = []
            for episode in range(builder.get_episode_num()):
                seed, _ = builder.resolve_episode(episode)
                if seed is not None:
                    entries.append((episode, int(seed)))
            self._episode_metadata[task] = entries
        return self._episode_metadata[task]

    def episode_count(self, task: str) -> int:
        return len(self._metadata_for_task(task))

    def episode_for_seed(self, task: str, seed: int) -> int:
        for episode, candidate in self._metadata_for_task(task):
            if candidate == seed:
                return episode
        raise ValueError(f"seed {seed} is not legal for {task}")

    def client(self, role: str) -> Any:
        if role not in self._clients:
            client = websocket_client_policy.MMEVLAWebsocketClientPolicy(
                self.args.host,
                self.args.port,
            )
            self._clients[role] = client
            self._evaluators[role].client = client
        return self._clients[role]

    def evaluator(self, role: str) -> Any:
        return self._evaluators[role]

    def prepare(
        self,
        spec: EpisodeSpec,
        frame_callback: Any,
        abort_event: threading.Event,
        fps_provider: Any,
        role: str = "enabled",
        initialization_replay: _InitializationReplay | None = None,
    ) -> RoboTTTEpisode:
        try:
            return RoboTTTEpisode(
                self,
                spec,
                frame_callback,
                abort_event,
                fps_provider,
                role,
                initialization_replay,
            )
        except EpisodeAbort as error:
            raise PreparationCancelled() from error

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
        self.storage.close()

"""dagger_review / signals: shared behavior owned by this layer."""

from __future__ import annotations

import dataclasses
import enum
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
import numpy as np


Array = np.ndarray


class AdapterContractError(ValueError):
    """Raised when a task adapter did not provide required semantic evidence."""


class SignalClass(str, enum.Enum):
    SEMANTIC = "semantic"
    FAILSAFE = "failsafe"
    DIAGNOSTIC = "diagnostic"


class RecoveryMode(str, enum.Enum):
    NONE = "none"
    OPEN_AND_RETRACT = "open_and_retract"
    CANCEL_FORBIDDEN_PUSH_PICKUP = "cancel_forbidden_push_pickup"
    RETURN_WRONG_OBJECT = "return_wrong_object"
    LIFT_HOLD_THEN_PLACE = "lift_hold_then_place"
    HOLD_POSE = "hold_pose"
    RETURN_TO_SAFE_ANCHOR = "return_to_safe_anchor"
    EXPERT_CONTINUE = "expert_continue"


def _position(value: Any, field_name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise AdapterContractError(
            f"{field_name} must be a three-dimensional position, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise AdapterContractError(f"{field_name} contains a non-finite value")
    return array


def _json_value(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


@dataclasses.dataclass(frozen=True)
class ActorState:
    name: str
    kind: str
    position: Array
    grasped: bool = False
    attributes: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _position(self.position, self.name))


@dataclasses.dataclass(frozen=True)
class SubtaskState:
    index: int
    name: str
    phase: str
    expected_actor: str | None = None
    expected_target: str | None = None


@dataclasses.dataclass(frozen=True)
class SceneState:
    """Semantic state read by a task-specific adapter before one policy action.

    ``actors`` contains manipulable objects and containers. ``landmarks``
    contains target poses, buttons, route waypoints, bin centers, and any other
    named three-dimensional point. ``points`` contains controlled points such
    as the TCP or the RouteStick tip. ``metrics`` is for explicitly named task
    values such as ``seconds_to_target_crossing`` or
    ``pickup_attempt_complete``.
    """

    step: int
    elapsed_seconds: float
    task_name: str
    subtask: SubtaskState
    tcp_position: Array
    gripper_opening: float
    actors: Mapping[str, ActorState]
    landmarks: Mapping[str, Array]
    points: Mapping[str, Array] = dataclasses.field(default_factory=dict)
    held_actor: str | None = None
    metrics: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    official_failure: bool = False
    terminal: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tcp_position",
            _position(self.tcp_position, "tcp_position"),
        )
        normalized_landmarks = {
            name: _position(value, f"landmarks[{name!r}]")
            for name, value in self.landmarks.items()
        }
        normalized_points = {
            name: _position(value, f"points[{name!r}]")
            for name, value in self.points.items()
        }
        object.__setattr__(self, "landmarks", normalized_landmarks)
        object.__setattr__(self, "points", normalized_points)


@dataclasses.dataclass(frozen=True)
class PredictedAction:
    """One-step action consequence supplied by the environment adapter.

    Joint-angle actions cannot be interpreted geometrically without the active
    controller and robot model.  Therefore the shared framework never guesses
    FK from the raw action.  The adapter must supply ``next_tcp_position`` and,
    where needed, predicted controlled/object points.
    """

    next_tcp_position: Array
    next_gripper_opening: float
    predicted_actor_positions: Mapping[str, Array] = dataclasses.field(
        default_factory=dict
    )
    predicted_points: Mapping[str, Array] = dataclasses.field(default_factory=dict)
    intents: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "next_tcp_position",
            _position(self.next_tcp_position, "next_tcp_position"),
        )
        actor_positions = {
            name: _position(value, f"predicted_actor_positions[{name!r}]")
            for name, value in self.predicted_actor_positions.items()
        }
        predicted_points = {
            name: _position(value, f"predicted_points[{name!r}]")
            for name, value in self.predicted_points.items()
        }
        object.__setattr__(self, "predicted_actor_positions", actor_positions)
        object.__setattr__(self, "predicted_points", predicted_points)


class TaskAdapter(Protocol):
    """The only RoboMME-specific boundary required by the shared framework."""

    @property
    def task_name(self) -> str:
        ...

    def read_scene(self) -> SceneState:
        ...

    def predict_action(
        self,
        action: Array,
        scene: SceneState,
    ) -> PredictedAction:
        ...


@dataclasses.dataclass(frozen=True)
class HistorySample:
    step: int
    elapsed_seconds: float
    subtask_index: int
    tcp_position: Array
    actor_positions: Mapping[str, Array]
    landmarks: Mapping[str, Array]
    points: Mapping[str, Array]
    metrics: Mapping[str, Any]


class SceneHistory:
    """Bounded semantic history used for trends and diagnostics."""

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 2:
            raise ValueError("history capacity must be at least two")
        self._samples: deque[HistorySample] = deque(maxlen=capacity)

    def append(self, scene: SceneState) -> None:
        sample = HistorySample(
            step=scene.step,
            elapsed_seconds=scene.elapsed_seconds,
            subtask_index=scene.subtask.index,
            tcp_position=scene.tcp_position.copy(),
            actor_positions={
                name: actor.position.copy() for name, actor in scene.actors.items()
            },
            landmarks={
                name: position.copy() for name, position in scene.landmarks.items()
            },
            points={name: position.copy() for name, position in scene.points.items()},
            metrics=dict(scene.metrics),
        )
        if self._samples and scene.step < self._samples[-1].step:
            raise ValueError("scene steps must be non-decreasing")
        if self._samples and scene.step == self._samples[-1].step:
            self._samples[-1] = sample
        else:
            self._samples.append(sample)

    def samples(
        self,
        window: int | None = None,
        current_subtask_only: bool = True,
    ) -> list[HistorySample]:
        values = list(self._samples)
        if current_subtask_only and values:
            subtask_index = values[-1].subtask_index
            values = [
                sample for sample in values if sample.subtask_index == subtask_index
            ]
        if window is not None:
            values = values[-window:]
        return values

    def task_index_unchanged_steps(self) -> int:
        values = list(self._samples)
        if not values:
            return 0
        current_index = values[-1].subtask_index
        count = 0
        for sample in reversed(values):
            if sample.subtask_index != current_index:
                break
            count += 1
        return max(0, count - 1)

    def tcp_motion(self, window: int = 128) -> dict[str, float]:
        values = self.samples(window=window, current_subtask_only=True)
        if len(values) < 2:
            return {"path_length": 0.0, "net_displacement": 0.0}
        positions = np.stack([sample.tcp_position for sample in values])
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        return {
            "path_length": path_length,
            "net_displacement": net_displacement,
        }

    def metric_values(
        self,
        metric_name: str,
        window: int,
    ) -> tuple[Array, Array]:
        samples = self.samples(window=window, current_subtask_only=True)
        times = []
        values = []
        for sample in samples:
            if metric_name not in sample.metrics:
                continue
            value = sample.metrics[metric_name]
            if not isinstance(value, (int, float, np.number)):
                continue
            if not math.isfinite(float(value)):
                continue
            times.append(sample.elapsed_seconds)
            values.append(float(value))
        return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)

    def point_distance_values(
        self,
        point_name: str,
        landmark_name: str,
        window: int,
    ) -> tuple[Array, Array]:
        samples = self.samples(window=window, current_subtask_only=True)
        times = []
        distances = []
        for sample in samples:
            if point_name not in sample.points:
                continue
            if landmark_name not in sample.landmarks:
                continue
            times.append(sample.elapsed_seconds)
            distances.append(
                euclidean(
                    sample.points[point_name],
                    sample.landmarks[landmark_name],
                )
            )
        return np.asarray(times, dtype=np.float64), np.asarray(
            distances,
            dtype=np.float64,
        )


def linear_trend(times: Array, values: Array) -> float | None:
    """Least-squares value slope per second, or ``None`` without enough data."""

    if len(times) < 2 or len(values) != len(times):
        return None
    centered_time = times - times.mean()
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 1e-12:
        return None
    centered_value = values - values.mean()
    return float(np.dot(centered_time, centered_value) / denominator)


def euclidean(first: Array, second: Array) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def point_to_segment_distance(point: Array, start: Array, end: Array) -> float:
    point = _position(point, "point")
    start = _position(start, "segment_start")
    end = _position(end, "segment_end")
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return euclidean(point, start)
    fraction = float(np.dot(point - start, segment) / denominator)
    fraction = min(1.0, max(0.0, fraction))
    projection = start + fraction * segment
    return euclidean(point, projection)


def closest_position(
    query: Array,
    positions: Mapping[str, Array],
    names: Sequence[str],
) -> tuple[str | None, float]:
    if not names:
        return None, math.inf
    missing = [name for name in names if name not in positions]
    if missing:
        raise AdapterContractError(f"missing named positions: {missing}")
    distances = [(name, euclidean(query, positions[name])) for name in names]
    return min(distances, key=lambda item: item[1])


def require_metric(scene: SceneState, metric_name: str) -> Any:
    if metric_name not in scene.metrics:
        raise AdapterContractError(
            f"task adapter for {scene.task_name} must provide metric {metric_name!r}"
        )
    return scene.metrics[metric_name]


@dataclasses.dataclass(frozen=True)
class CandidateSignal:
    active: bool
    reason_code: str
    signal_class: SignalClass
    recovery_mode: RecoveryMode
    expected_actor: str | None
    observed_actor: str | None
    metrics: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    priority: int = 0
    activate_after: int = 1
    release_after: int = 1
    pre_failure_required: bool = True

    @property
    def key(self) -> str:
        observed = self.observed_actor or "none"
        return f"{self.reason_code}:{observed}"


@dataclasses.dataclass(frozen=True)
class InterventionDecision:
    intervene: bool
    reason_code: str
    signal_class: SignalClass | None
    task_name: str
    step: int
    subtask_index: int
    subtask_name: str
    phase: str
    expected_actor: str | None
    observed_actor: str | None
    metrics: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    recovery_mode: RecoveryMode
    diagnostics: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclasses.dataclass
class _HysteresisState:
    active_count: int = 0
    inactive_count: int = 0
    latched: bool = False


class HysteresisBook:
    """Consecutive-frame activation and release keyed by semantic reason."""

    def __init__(self) -> None:
        self._states: dict[str, _HysteresisState] = {}

    def update(self, signal: CandidateSignal) -> bool:
        if signal.activate_after < 1 or signal.release_after < 1:
            raise ValueError("hysteresis counts must be positive")
        state = self._states.setdefault(signal.key, _HysteresisState())
        if signal.active:
            state.active_count += 1
            state.inactive_count = 0
            if state.active_count >= signal.activate_after:
                state.latched = True
        else:
            state.active_count = 0
            state.inactive_count += 1
            if state.inactive_count >= signal.release_after:
                state.latched = False
        return state.latched

    def reset_absent(self, present_keys: set[str]) -> None:
        for key, state in self._states.items():
            if key in present_keys:
                continue
            state.active_count = 0
            state.inactive_count += 1
            if state.inactive_count >= 1:
                state.latched = False


@dataclasses.dataclass(frozen=True)
class PreActionContext:
    scene: SceneState
    prediction: PredictedAction
    history: SceneHistory


Rule = Callable[[PreActionContext], Sequence[CandidateSignal]]


class TaskDetector(Protocol):
    def evaluate(self, context: PreActionContext) -> Sequence[CandidateSignal]:
        ...


class CompositeTaskDetector:
    """Compose shared primitives without creating a parallel task runtime."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        self._rules = list(rules)

    def evaluate(self, context: PreActionContext) -> Sequence[CandidateSignal]:
        signals: list[CandidateSignal] = []
        for rule in self._rules:
            signals.extend(rule(context))
        return signals


class DetectorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], TaskDetector]] = {}

    def register(
        self,
        task_name: str,
        factory: Callable[[], TaskDetector],
    ) -> None:
        if task_name in self._factories:
            raise ValueError(f"detector already registered for {task_name}")
        self._factories[task_name] = factory

    def build(self, task_name: str) -> TaskDetector:
        if task_name not in self._factories:
            raise KeyError(f"no pre-action detector registered for {task_name}")
        return self._factories[task_name]()


class PreActionDetectorEngine:
    """Stateful engine called once immediately before each policy action."""

    def __init__(
        self,
        adapter: TaskAdapter,
        detector: TaskDetector,
        history_capacity: int = 256,
    ) -> None:
        self.adapter = adapter
        self.detector = detector
        self._history_capacity = history_capacity
        self.history = SceneHistory(capacity=history_capacity)
        self.hysteresis = HysteresisBook()
        self._lifecycle_closed = False
        self._lifecycle_close_reason: str | None = None

    def reset_episode(self) -> None:
        """Reset state only after the runtime has confirmed a new episode reset."""

        self.history = SceneHistory(capacity=self._history_capacity)
        self.hysteresis = HysteresisBook()
        self._lifecycle_closed = False
        self._lifecycle_close_reason = None

    def before_action(self, action: Array) -> InterventionDecision:
        scene = self.adapter.read_scene()
        if scene.task_name != self.adapter.task_name:
            raise AdapterContractError(
                f"adapter task {self.adapter.task_name!r} returned "
                f"scene for {scene.task_name!r}"
            )
        if scene.terminal and not self._lifecycle_closed:
            self._lifecycle_closed = True
            self._lifecycle_close_reason = "terminal"
        elif scene.official_failure and not self._lifecycle_closed:
            self._lifecycle_closed = True
            self._lifecycle_close_reason = "official_failure"
        prediction = self.adapter.predict_action(action, scene)
        self.history.append(scene)
        context = PreActionContext(
            scene=scene,
            prediction=prediction,
            history=self.history,
        )
        signals = list(self.detector.evaluate(context))
        present_keys = {signal.key for signal in signals}
        ready = []
        blocked_after_official_failure = 0
        blocked_after_terminal = 0
        for signal in signals:
            if signal.signal_class == SignalClass.DIAGNOSTIC:
                continue
            blocked_by_terminal = (
                self._lifecycle_closed
                and self._lifecycle_close_reason == "terminal"
            )
            blocked_by_official_failure = (
                self._lifecycle_closed
                and self._lifecycle_close_reason == "official_failure"
            )
            if blocked_by_terminal or blocked_by_official_failure:
                inactive_signal = dataclasses.replace(signal, active=False)
                self.hysteresis.update(inactive_signal)
                if blocked_by_terminal:
                    blocked_after_terminal += 1
                else:
                    blocked_after_official_failure += 1
                continue
            if self.hysteresis.update(signal):
                ready.append(signal)
        self.hysteresis.reset_absent(present_keys)

        motion = self.history.tcp_motion(window=128)
        diagnostics = {
            "official_failure_before_action": scene.official_failure,
            "terminal_before_action": scene.terminal,
            "lifecycle_latched_closed": self._lifecycle_closed,
            "lifecycle_close_reason": self._lifecycle_close_reason,
            "task_index_unchanged_steps": (
                self.history.task_index_unchanged_steps()
            ),
            "tcp_path_last_128": motion["path_length"],
            "tcp_net_last_128": motion["net_displacement"],
            "signals_blocked_after_official_failure": (
                blocked_after_official_failure
            ),
            "signals_blocked_after_terminal": blocked_after_terminal,
        }
        if not ready:
            return InterventionDecision(
                intervene=False,
                reason_code="continue",
                signal_class=None,
                task_name=scene.task_name,
                step=scene.step,
                subtask_index=scene.subtask.index,
                subtask_name=scene.subtask.name,
                phase=scene.subtask.phase,
                expected_actor=scene.subtask.expected_actor,
                observed_actor=scene.held_actor,
                metrics={},
                thresholds={},
                recovery_mode=RecoveryMode.NONE,
                diagnostics=diagnostics,
            )

        selected = max(
            ready,
            key=lambda signal: (signal.priority, signal.reason_code),
        )
        metrics = dict(selected.metrics)
        metrics["official_failure_before_action"] = scene.official_failure
        metrics["evidence_is_pre_failure"] = not scene.official_failure
        return InterventionDecision(
            intervene=True,
            reason_code=selected.reason_code,
            signal_class=selected.signal_class,
            task_name=scene.task_name,
            step=scene.step,
            subtask_index=scene.subtask.index,
            subtask_name=scene.subtask.name,
            phase=scene.subtask.phase,
            expected_actor=selected.expected_actor,
            observed_actor=selected.observed_actor,
            metrics=metrics,
            thresholds=selected.thresholds,
            recovery_mode=selected.recovery_mode,
            diagnostics=diagnostics,
        )


def _actor_positions(scene: SceneState) -> dict[str, Array]:
    return {name: actor.position for name, actor in scene.actors.items()}


def _approach_metrics(
    context: PreActionContext,
    expected_names: Sequence[str],
    forbidden_names: Sequence[str],
) -> dict[str, Any]:
    actor_positions = _actor_positions(context.scene)
    current_tcp = context.scene.tcp_position
    predicted_tcp = context.prediction.next_tcp_position
    expected_actor, expected_distance = closest_position(
        predicted_tcp,
        actor_positions,
        expected_names,
    )
    forbidden_actor, forbidden_distance = closest_position(
        predicted_tcp,
        actor_positions,
        forbidden_names,
    )
    current_forbidden_distance = math.inf
    if forbidden_actor is not None:
        current_forbidden_distance = euclidean(
            current_tcp,
            actor_positions[forbidden_actor],
        )
    return {
        "expected_actor": expected_actor,
        "expected_distance": expected_distance,
        "forbidden_actor": forbidden_actor,
        "forbidden_distance": forbidden_distance,
        "forbidden_approach_delta": (
            current_forbidden_distance - forbidden_distance
        ),
        "tcp_motion": euclidean(current_tcp, predicted_tcp),
        "tcp_vertical_delta": float(predicted_tcp[2] - current_tcp[2]),
    }


@dataclasses.dataclass(frozen=True)
class PickupIntentSpec:
    expected_actors: tuple[str, ...]
    forbidden_actors: tuple[str, ...]
    intent_radius: float
    selection_margin: float
    minimum_approach_delta: float
    activate_after: int
    missed_grasp_metric: str | None = None


def pickup_intent_signals(
    context: PreActionContext,
    spec: PickupIntentSpec,
) -> Sequence[CandidateSignal]:
    """Detect a wrong pickup target before contact/closure and a missed grasp.

    Actor identities and thresholds must come from the task adapter/config.
    No color, name, or gripper convention is inferred here.
    """

    metrics = _approach_metrics(
        context,
        spec.expected_actors,
        spec.forbidden_actors,
    )
    approaching = metrics["forbidden_approach_delta"] >= spec.minimum_approach_delta
    descending = metrics["tcp_vertical_delta"] < 0.0
    closing = (
        context.prediction.next_gripper_opening
        < context.scene.gripper_opening
    ) or "close_gripper" in context.prediction.intents
    wrong_is_closer = (
        metrics["forbidden_distance"] + spec.selection_margin
        < metrics["expected_distance"]
    )
    active = bool(
        metrics["forbidden_actor"] is not None
        and metrics["forbidden_distance"] <= spec.intent_radius
        and wrong_is_closer
        and (approaching or descending or closing)
    )
    signals = [
        CandidateSignal(
            active=active,
            reason_code="pickup.wrong_actor_precontact",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
            expected_actor=metrics["expected_actor"],
            observed_actor=metrics["forbidden_actor"],
            metrics={
                **metrics,
                "closing": closing,
                "descending": descending,
            },
            thresholds={
                "intent_radius": spec.intent_radius,
                "selection_margin": spec.selection_margin,
                "minimum_approach_delta": spec.minimum_approach_delta,
            },
            priority=80,
            activate_after=spec.activate_after,
            release_after=1,
        )
    ]
    if spec.missed_grasp_metric is not None:
        attempt_complete = bool(
            require_metric(context.scene, spec.missed_grasp_metric)
        )
        signals.append(
            CandidateSignal(
                active=attempt_complete and context.scene.held_actor is None,
                reason_code="pickup.missed_grasp",
                signal_class=SignalClass.SEMANTIC,
                recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                expected_actor=metrics["expected_actor"],
                observed_actor=None,
                metrics={"attempt_complete": attempt_complete},
                thresholds={"held_actor_required": True},
                priority=85,
                activate_after=1,
                release_after=1,
            )
        )
    return signals


@dataclasses.dataclass(frozen=True)
class PlaceIntentSpec:
    held_actor: str
    expected_targets: tuple[str, ...]
    forbidden_targets: tuple[str, ...]
    release_radius: float
    transport_radius: float
    selection_margin: float
    minimum_approach_delta: float
    activate_after: int


def place_intent_signals(
    context: PreActionContext,
    spec: PlaceIntentSpec,
) -> Sequence[CandidateSignal]:
    if context.scene.held_actor != spec.held_actor:
        return []
    if spec.held_actor not in context.prediction.predicted_actor_positions:
        raise AdapterContractError(
            "place detector requires predicted_actor_positions for the held actor"
        )
    predicted_position = context.prediction.predicted_actor_positions[spec.held_actor]
    current_position = context.scene.actors[spec.held_actor].position
    expected_target, expected_distance = closest_position(
        predicted_position,
        context.scene.landmarks,
        spec.expected_targets,
    )
    forbidden_target, forbidden_distance = closest_position(
        predicted_position,
        context.scene.landmarks,
        spec.forbidden_targets,
    )
    current_forbidden_distance = math.inf
    if forbidden_target is not None:
        current_forbidden_distance = euclidean(
            current_position,
            context.scene.landmarks[forbidden_target],
        )
    approach_delta = current_forbidden_distance - forbidden_distance
    wrong_is_closer = (
        forbidden_distance + spec.selection_margin < expected_distance
    )
    opening = (
        context.prediction.next_gripper_opening
        > context.scene.gripper_opening
    ) or "open_gripper" in context.prediction.intents
    wrong_release = bool(
        forbidden_target is not None
        and wrong_is_closer
        and opening
        and forbidden_distance <= spec.release_radius
    )
    wrong_transport = bool(
        forbidden_target is not None
        and wrong_is_closer
        and forbidden_distance <= spec.transport_radius
        and approach_delta >= spec.minimum_approach_delta
    )
    common_metrics = {
        "held_actor": spec.held_actor,
        "expected_target": expected_target,
        "expected_distance": expected_distance,
        "forbidden_target": forbidden_target,
        "forbidden_distance": forbidden_distance,
        "forbidden_approach_delta": approach_delta,
        "opening": opening,
    }
    return [
        CandidateSignal(
            active=wrong_release,
            reason_code="place.wrong_target_prerelease",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.LIFT_HOLD_THEN_PLACE,
            expected_actor=expected_target,
            observed_actor=forbidden_target,
            metrics=common_metrics,
            thresholds={"release_radius": spec.release_radius},
            priority=95,
            activate_after=1,
            release_after=1,
        ),
        CandidateSignal(
            active=wrong_transport,
            reason_code="place.wrong_target_transport",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.LIFT_HOLD_THEN_PLACE,
            expected_actor=expected_target,
            observed_actor=forbidden_target,
            metrics=common_metrics,
            thresholds={
                "transport_radius": spec.transport_radius,
                "selection_margin": spec.selection_margin,
                "minimum_approach_delta": spec.minimum_approach_delta,
            },
            priority=75,
            activate_after=spec.activate_after,
            release_after=1,
        ),
    ]


@dataclasses.dataclass(frozen=True)
class ContainerIntentSpec:
    expected_containers: tuple[str, ...]
    forbidden_containers: tuple[str, ...]
    intent_radius: float
    selection_margin: float
    minimum_approach_delta: float
    activate_after: int


def container_intent_signals(
    context: PreActionContext,
    spec: ContainerIntentSpec,
) -> Sequence[CandidateSignal]:
    metrics = _approach_metrics(
        context,
        spec.expected_containers,
        spec.forbidden_containers,
    )
    opening_intent = bool(
        "open_container" in context.prediction.intents
        or "close_gripper" in context.prediction.intents
        or metrics["tcp_vertical_delta"] < 0.0
    )
    wrong_is_closer = (
        metrics["forbidden_distance"] + spec.selection_margin
        < metrics["expected_distance"]
    )
    active = bool(
        metrics["forbidden_actor"] is not None
        and wrong_is_closer
        and metrics["forbidden_distance"] <= spec.intent_radius
        and metrics["forbidden_approach_delta"] >= spec.minimum_approach_delta
        and opening_intent
    )
    return [
        CandidateSignal(
            active=active,
            reason_code="container.wrong_container_precontact",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
            expected_actor=metrics["expected_actor"],
            observed_actor=metrics["forbidden_actor"],
            metrics={**metrics, "opening_intent": opening_intent},
            thresholds={
                "intent_radius": spec.intent_radius,
                "selection_margin": spec.selection_margin,
                "minimum_approach_delta": spec.minimum_approach_delta,
            },
            priority=85,
            activate_after=spec.activate_after,
            release_after=1,
        )
    ]


@dataclasses.dataclass(frozen=True)
class ButtonIntentSpec:
    expected_buttons: tuple[str, ...]
    forbidden_buttons: tuple[str, ...]
    intent_radius: float
    selection_margin: float
    minimum_approach_delta: float
    activate_after: int
    prerequisite_metric: str | None = None


def button_intent_signals(
    context: PreActionContext,
    spec: ButtonIntentSpec,
) -> Sequence[CandidateSignal]:
    positions = context.scene.landmarks
    predicted_tcp = context.prediction.next_tcp_position
    expected_button, expected_distance = closest_position(
        predicted_tcp,
        positions,
        spec.expected_buttons,
    )
    forbidden_button, forbidden_distance = closest_position(
        predicted_tcp,
        positions,
        spec.forbidden_buttons,
    )
    current_forbidden_distance = math.inf
    if forbidden_button is not None:
        current_forbidden_distance = euclidean(
            context.scene.tcp_position,
            positions[forbidden_button],
        )
    approach_delta = current_forbidden_distance - forbidden_distance
    vertical_delta = float(
        predicted_tcp[2] - context.scene.tcp_position[2]
    )
    press_intent = bool(
        "press" in context.prediction.intents or vertical_delta < 0.0
    )
    wrong_is_closer = forbidden_distance + spec.selection_margin < expected_distance
    wrong_button = bool(
        forbidden_button is not None
        and wrong_is_closer
        and forbidden_distance <= spec.intent_radius
        and approach_delta >= spec.minimum_approach_delta
        and press_intent
    )
    signals = [
        CandidateSignal(
            active=wrong_button,
            reason_code="button.wrong_button_prepress",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
            expected_actor=expected_button,
            observed_actor=forbidden_button,
            metrics={
                "expected_distance": expected_distance,
                "forbidden_distance": forbidden_distance,
                "forbidden_approach_delta": approach_delta,
                "vertical_delta": vertical_delta,
                "press_intent": press_intent,
            },
            thresholds={
                "intent_radius": spec.intent_radius,
                "selection_margin": spec.selection_margin,
                "minimum_approach_delta": spec.minimum_approach_delta,
            },
            priority=90,
            activate_after=spec.activate_after,
            release_after=1,
        )
    ]
    if spec.prerequisite_metric is not None:
        prerequisite_complete = bool(
            require_metric(context.scene, spec.prerequisite_metric)
        )
        expected_press = bool(
            expected_button is not None
            and expected_distance <= spec.intent_radius
            and press_intent
        )
        signals.append(
            CandidateSignal(
                active=expected_press and not prerequisite_complete,
                reason_code="button.prerequisite_incomplete_prepress",
                signal_class=SignalClass.SEMANTIC,
                recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                expected_actor=expected_button,
                observed_actor=expected_button,
                metrics={
                    "expected_distance": expected_distance,
                    "press_intent": press_intent,
                    "prerequisite_complete": prerequisite_complete,
                },
                thresholds={"prerequisite_required": True},
                priority=92,
                activate_after=spec.activate_after,
                release_after=1,
            )
        )
    return signals


@dataclasses.dataclass(frozen=True)
class RouteProgressSpec:
    tracked_point: str
    corridor_start: str
    expected_waypoint: str
    forbidden_waypoints: tuple[str, ...]
    target_radius: float
    wrong_waypoint_radius: float
    maximum_cross_track: float
    minimum_improvement_per_second: float
    trend_window: int
    activate_after: int


def route_progress_signals(
    context: PreActionContext,
    spec: RouteProgressSpec,
) -> Sequence[CandidateSignal]:
    if spec.tracked_point not in context.scene.points:
        raise AdapterContractError(
            f"route adapter must provide points[{spec.tracked_point!r}]"
        )
    if spec.tracked_point not in context.prediction.predicted_points:
        raise AdapterContractError(
            f"route adapter must predict points[{spec.tracked_point!r}]"
        )
    required_landmarks = (spec.corridor_start, spec.expected_waypoint)
    for landmark in required_landmarks:
        if landmark not in context.scene.landmarks:
            raise AdapterContractError(
                f"route adapter must provide landmarks[{landmark!r}]"
            )
    current_point = context.scene.points[spec.tracked_point]
    predicted_point = context.prediction.predicted_points[spec.tracked_point]
    expected_position = context.scene.landmarks[spec.expected_waypoint]
    current_distance = euclidean(current_point, expected_position)
    predicted_distance = euclidean(predicted_point, expected_position)
    times, distance_values = context.history.point_distance_values(
        spec.tracked_point,
        spec.expected_waypoint,
        spec.trend_window,
    )
    distance_trend = linear_trend(times, distance_values)
    cross_track = point_to_segment_distance(
        predicted_point,
        context.scene.landmarks[spec.corridor_start],
        expected_position,
    )
    forbidden_waypoint, forbidden_distance = closest_position(
        predicted_point,
        context.scene.landmarks,
        spec.forbidden_waypoints,
    )
    wrong_waypoint = bool(
        forbidden_waypoint is not None
        and forbidden_distance <= spec.wrong_waypoint_radius
        and forbidden_distance < predicted_distance
    )
    moving_away = bool(
        distance_trend is not None
        and distance_trend > -spec.minimum_improvement_per_second
        and current_distance > spec.target_radius
        and predicted_distance >= current_distance
    )
    outside_corridor = bool(
        cross_track > spec.maximum_cross_track
        and predicted_distance >= current_distance
        and current_distance > spec.target_radius
    )
    common_metrics = {
        "current_distance": current_distance,
        "predicted_distance": predicted_distance,
        "distance_trend_per_second": distance_trend,
        "cross_track_distance": cross_track,
        "forbidden_waypoint": forbidden_waypoint,
        "forbidden_distance": forbidden_distance,
    }
    common_thresholds = {
        "target_radius": spec.target_radius,
        "wrong_waypoint_radius": spec.wrong_waypoint_radius,
        "maximum_cross_track": spec.maximum_cross_track,
        "minimum_improvement_per_second": spec.minimum_improvement_per_second,
        "trend_window": spec.trend_window,
    }
    return [
        CandidateSignal(
            active=wrong_waypoint,
            reason_code="route.wrong_waypoint_approach",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
            expected_actor=spec.expected_waypoint,
            observed_actor=forbidden_waypoint,
            metrics=common_metrics,
            thresholds=common_thresholds,
            priority=92,
            activate_after=spec.activate_after,
            release_after=1,
        ),
        CandidateSignal(
            active=moving_away,
            reason_code="route.no_expected_progress",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
            expected_actor=spec.expected_waypoint,
            observed_actor=None,
            metrics=common_metrics,
            thresholds=common_thresholds,
            priority=75,
            activate_after=spec.activate_after,
            release_after=2,
        ),
        CandidateSignal(
            active=outside_corridor,
            reason_code="route.corridor_deviation",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
            expected_actor=spec.expected_waypoint,
            observed_actor=None,
            metrics=common_metrics,
            thresholds=common_thresholds,
            priority=80,
            activate_after=spec.activate_after,
            release_after=2,
        ),
    ]


@dataclasses.dataclass(frozen=True)
class TimedProgressSpec:
    seconds_to_event_metric: str
    progress_metric: str
    expert_completion_seconds: float
    early_margin_seconds: float
    late_margin_seconds: float
    minimum_ready_progress: float
    activate_after: int


def timed_progress_signals(
    context: PreActionContext,
    spec: TimedProgressSpec,
) -> Sequence[CandidateSignal]:
    """Detect premature motion or a latest-safe-time miss for timed tasks.

    Durations must be calibrated from successful policy/expert traces.  The
    primitive deliberately has no built-in StopCube constants.
    """

    seconds_to_event = float(
        require_metric(context.scene, spec.seconds_to_event_metric)
    )
    current_progress = float(require_metric(context.scene, spec.progress_metric))
    if not math.isfinite(seconds_to_event) or not math.isfinite(current_progress):
        raise AdapterContractError("timed-progress metrics must be finite")
    current_tcp = context.scene.tcp_position
    predicted_tcp = context.prediction.next_tcp_position
    descending = predicted_tcp[2] < current_tcp[2]
    press_motion = bool(
        "press" in context.prediction.intents or descending
    )
    early_boundary = (
        spec.expert_completion_seconds + spec.early_margin_seconds
    )
    latest_safe_boundary = (
        spec.expert_completion_seconds + spec.late_margin_seconds
    )
    premature = bool(press_motion and seconds_to_event > early_boundary)
    latest_safe_miss = bool(
        seconds_to_event <= latest_safe_boundary
        and current_progress < spec.minimum_ready_progress
        and not press_motion
    )
    metrics = {
        "seconds_to_event": seconds_to_event,
        "current_progress": current_progress,
        "press_motion": press_motion,
        "tcp_vertical_delta": float(predicted_tcp[2] - current_tcp[2]),
    }
    thresholds = {
        "expert_completion_seconds": spec.expert_completion_seconds,
        "early_boundary_seconds": early_boundary,
        "latest_safe_boundary_seconds": latest_safe_boundary,
        "minimum_ready_progress": spec.minimum_ready_progress,
    }
    return [
        CandidateSignal(
            active=premature,
            reason_code="timed_progress.premature_press_motion",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.HOLD_POSE,
            expected_actor=context.scene.subtask.expected_actor,
            observed_actor=context.scene.subtask.expected_actor,
            metrics=metrics,
            thresholds=thresholds,
            priority=90,
            activate_after=spec.activate_after,
            release_after=2,
        ),
        CandidateSignal(
            active=latest_safe_miss,
            reason_code="timed_progress.latest_safe_takeover",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.EXPERT_CONTINUE,
            expected_actor=context.scene.subtask.expected_actor,
            observed_actor=None,
            metrics=metrics,
            thresholds=thresholds,
            priority=95,
            activate_after=spec.activate_after,
            release_after=1,
        ),
    ]


@dataclasses.dataclass(frozen=True)
class ForbiddenPickupSpec:
    """Guard a phase in which starting another pickup is semantically invalid."""

    forbidden_actors: tuple[str, ...]
    intent_radius: float
    minimum_approach_delta: float
    activate_after: int
    prerequisite_metric: str | None = None


@dataclasses.dataclass(frozen=True)
class GraspedLiftIntentSpec:
    """Detect the action that would turn an established grasp into a pickup.

    This is deliberately separate from :class:`ForbiddenPickupSpec`.  Closing
    or descending near an actor is useful evidence for an extra pickup in
    phases where no contact is valid, but it is also the native motion for a
    closed-gripper push.  Here the actor must already be grasped and the
    predicted rigidly attached actor pose must cross the native pickup height.
    """

    candidate_actors: tuple[str, ...]
    native_pickup_height: float
    minimum_lift_delta: float
    activate_after: int = 1


def grasped_lift_intent_signals(
    context: PreActionContext,
    spec: GraspedLiftIntentSpec,
) -> Sequence[CandidateSignal]:
    """Flag a grasped object's pre-threshold lift, never a horizontal push."""

    grasped_actor_names = tuple(
        context.scene.metrics.get("grasped_actor_names", ())
    )
    grasp_identity_ambiguous = bool(
        context.scene.metrics.get("grasp_identity_ambiguous", False)
        or len(grasped_actor_names) > 1
    )
    if grasp_identity_ambiguous:
        raise AdapterContractError(
            "grasped-lift detection cannot resolve multiple grasped actors: "
            f"{grasped_actor_names}"
        )
    held_actor = context.scene.held_actor
    if held_actor is None or held_actor not in spec.candidate_actors:
        return []
    if held_actor not in context.scene.actors:
        raise AdapterContractError(
            f"held actor {held_actor!r} is missing from scene actors"
        )
    if held_actor not in context.prediction.predicted_actor_positions:
        raise AdapterContractError(
            "grasped-lift detection requires the predicted pose of held actor "
            f"{held_actor!r}"
        )
    current_position = context.scene.actors[held_actor].position
    predicted_position = context.prediction.predicted_actor_positions[held_actor]
    current_height = float(current_position[2])
    predicted_height = float(predicted_position[2])
    lift_delta = predicted_height - current_height
    crosses_native_pickup_height = bool(
        current_height <= spec.native_pickup_height
        and predicted_height > spec.native_pickup_height
    )
    meets_minimum_lift_delta = lift_delta >= spec.minimum_lift_delta
    active = crosses_native_pickup_height
    return [
        CandidateSignal(
            active=active,
            reason_code="pickup.grasped_lift_prethreshold",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.RETURN_WRONG_OBJECT,
            expected_actor=None,
            observed_actor=held_actor,
            metrics={
                "grasp_established": True,
                "current_actor_height": current_height,
                "predicted_actor_height": predicted_height,
                "lift_delta": lift_delta,
                "meets_minimum_lift_delta": meets_minimum_lift_delta,
                "crosses_native_pickup_height": crosses_native_pickup_height,
            },
            thresholds={
                "native_pickup_height": spec.native_pickup_height,
                "minimum_lift_delta": spec.minimum_lift_delta,
            },
            priority=97,
            activate_after=spec.activate_after,
            release_after=1,
        )
    ]


def forbidden_pickup_signals(
    context: PreActionContext,
    spec: ForbiddenPickupSpec,
) -> Sequence[CandidateSignal]:
    """Detect a pickup-like action after a repeat/cycle boundary.

    This is intentionally actor-identity based.  It is used for phases such as
    VideoRepick's final button phase, where approaching any cube to close the
    gripper would create an extra pickup even though no pickup actor is valid.
    """

    actor_positions = _actor_positions(context.scene)
    predicted_tcp = context.prediction.next_tcp_position
    actor_name, predicted_distance = closest_position(
        predicted_tcp,
        actor_positions,
        spec.forbidden_actors,
    )
    current_distance = math.inf
    if actor_name is not None:
        current_distance = euclidean(
            context.scene.tcp_position,
            actor_positions[actor_name],
        )
    approach_delta = current_distance - predicted_distance
    vertical_delta = float(
        predicted_tcp[2] - context.scene.tcp_position[2]
    )
    closing = bool(
        context.prediction.next_gripper_opening
        < context.scene.gripper_opening
        or "close_gripper" in context.prediction.intents
    )
    descending = vertical_delta < 0.0
    prerequisite_complete = True
    if spec.prerequisite_metric is not None:
        prerequisite_complete = bool(
            require_metric(context.scene, spec.prerequisite_metric)
        )
    active = bool(
        prerequisite_complete
        and actor_name is not None
        and predicted_distance <= spec.intent_radius
        and approach_delta >= spec.minimum_approach_delta
        and (closing or descending)
    )
    return [
        CandidateSignal(
            active=active,
            reason_code="cycle.extra_pickup_precontact",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
            expected_actor=None,
            observed_actor=actor_name,
            metrics={
                "forbidden_actor": actor_name,
                "predicted_distance": predicted_distance,
                "approach_delta": approach_delta,
                "vertical_delta": vertical_delta,
                "closing": closing,
                "descending": descending,
                "prerequisite_complete": prerequisite_complete,
            },
            thresholds={
                "intent_radius": spec.intent_radius,
                "minimum_approach_delta": spec.minimum_approach_delta,
            },
            priority=94,
            activate_after=spec.activate_after,
            release_after=1,
        )
    ]


@dataclasses.dataclass(frozen=True)
class InsertionIntentSpec:
    expected_insert_actor: str
    opposite_end_actor: str
    target_landmark: str
    expected_direction: int
    side_axis: int
    intent_radius: float
    selection_margin: float
    side_margin: float
    minimum_approach_delta: float
    activate_after: int


def insertion_intent_signals(
    context: PreActionContext,
    spec: InsertionIntentSpec,
) -> Sequence[CandidateSignal]:
    """Detect wrong-end or wrong-side insertion before entering the box.

    The adapter must predict both rigid peg endpoints.  No peg orientation or
    side convention is reconstructed here; the exact endpoint identities and
    native ``direction`` sign are part of the task-specific contract.
    """

    if spec.expected_direction not in (-1, 1):
        raise AdapterContractError("insertion expected_direction must be -1 or 1")
    if spec.side_axis not in (0, 1, 2):
        raise AdapterContractError("insertion side_axis must be 0, 1, or 2")
    for actor_name in (spec.expected_insert_actor, spec.opposite_end_actor):
        if actor_name not in context.scene.actors:
            raise AdapterContractError(
                f"insertion adapter must provide actors[{actor_name!r}]"
            )
        if actor_name not in context.prediction.predicted_actor_positions:
            raise AdapterContractError(
                "insertion detector requires predicted_actor_positions for "
                f"{actor_name!r}"
            )
    if spec.target_landmark not in context.scene.landmarks:
        raise AdapterContractError(
            "insertion adapter must provide target landmark "
            f"{spec.target_landmark!r}"
        )

    target = context.scene.landmarks[spec.target_landmark]
    current_expected = context.scene.actors[spec.expected_insert_actor].position
    predicted_expected = context.prediction.predicted_actor_positions[
        spec.expected_insert_actor
    ]
    predicted_opposite = context.prediction.predicted_actor_positions[
        spec.opposite_end_actor
    ]
    current_distance = euclidean(current_expected, target)
    expected_distance = euclidean(predicted_expected, target)
    opposite_distance = euclidean(predicted_opposite, target)
    approach_delta = current_distance - expected_distance
    wrong_end_closer = bool(
        opposite_distance + spec.selection_margin < expected_distance
    )
    near_box = min(expected_distance, opposite_distance) <= spec.intent_radius
    side_indicator = float(
        context.prediction.next_tcp_position[spec.side_axis]
        - target[spec.side_axis]
    )
    side_product = side_indicator * spec.expected_direction
    wrong_side = bool(
        expected_distance <= spec.intent_radius
        and approach_delta >= spec.minimum_approach_delta
        and side_product >= spec.side_margin
    )
    metrics = {
        "expected_insert_distance": expected_distance,
        "opposite_end_distance": opposite_distance,
        "expected_approach_delta": approach_delta,
        "side_indicator": side_indicator,
        "side_product": side_product,
    }
    thresholds = {
        "intent_radius": spec.intent_radius,
        "selection_margin": spec.selection_margin,
        "side_margin": spec.side_margin,
        "minimum_approach_delta": spec.minimum_approach_delta,
        "native_correct_side_condition": "side_indicator * direction < 0",
    }
    return [
        CandidateSignal(
            active=bool(
                near_box
                and wrong_end_closer
                and approach_delta >= spec.minimum_approach_delta
            ),
            reason_code="insertion.wrong_end_approach",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
            expected_actor=spec.expected_insert_actor,
            observed_actor=spec.opposite_end_actor,
            metrics=metrics,
            thresholds=thresholds,
            priority=97,
            activate_after=spec.activate_after,
            release_after=1,
        ),
        CandidateSignal(
            active=wrong_side,
            reason_code="insertion.wrong_side_approach",
            signal_class=SignalClass.SEMANTIC,
            recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
            expected_actor=spec.expected_insert_actor,
            observed_actor=None,
            metrics=metrics,
            thresholds=thresholds,
            priority=96,
            activate_after=spec.activate_after,
            release_after=1,
        ),
    ]


@dataclasses.dataclass(frozen=True)
class ActionIntents:
    descending: bool
    closing: bool
    opening: bool
    tcp_motion: float
    vertical_delta: float


@dataclasses.dataclass(frozen=True)
class PickupActorAssessment:
    """Geometry for one explicitly named pickup actor.

    Multiple valid objects are intentionally not represented here.  A task
    owns the OR relation between several assessments, because only the task
    knows whether those actors are interchangeable at the current stage.
    """

    actor_name: str
    actor_kind: str | None
    distance: float
    xy_distance: float
    along_distance: float
    cross_distance: float
    z_clearance: float
    center_aligned: bool
    cluster_obstructed: bool
    obstructing_actors: tuple[str, ...]
    cluster_safe_centered: bool
    graspable: bool


def action_intents(before: SceneState, after: SceneState) -> ActionIntents:
    tcp_delta = after.tcp_position - before.tcp_position
    opening_delta = after.gripper_opening - before.gripper_opening
    return ActionIntents(
        descending=bool(tcp_delta[2] < 0.0),
        closing=bool(opening_delta < -1e-4),
        opening=bool(opening_delta > 1e-4),
        tcp_motion=float(np.linalg.norm(tcp_delta)),
        vertical_delta=float(tcp_delta[2]),
    )


def _positions(scene: SceneState) -> dict[str, np.ndarray]:
    return {name: actor.position for name, actor in scene.actors.items()}


def _closest(
    query: np.ndarray,
    positions: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> tuple[str | None, float]:
    available = [(name, positions[name]) for name in names if name in positions]
    if not available:
        return None, math.inf
    return min(
        (
            (name, float(np.linalg.norm(query - position)))
            for name, position in available
        ),
        key=lambda item: item[1],
    )


def _closest_xy(
    query: np.ndarray,
    positions: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> tuple[str | None, float]:
    available = [(name, positions[name]) for name in names if name in positions]
    if not available:
        return None, math.inf
    return min(
        (
            (name, float(np.linalg.norm(query[:2] - position[:2])))
            for name, position in available
        ),
        key=lambda item: item[1],
    )


def semantic_signal(
    *,
    active: bool,
    reason_code: str,
    recovery_mode: RecoveryMode,
    expected_actor: str | None,
    observed_actor: str | None,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    priority: int,
) -> CandidateSignal:
    return CandidateSignal(
        active=active,
        reason_code=reason_code,
        signal_class=SignalClass.SEMANTIC,
        recovery_mode=recovery_mode,
        expected_actor=expected_actor,
        observed_actor=observed_actor,
        metrics=dict(metrics),
        thresholds=dict(thresholds),
        priority=priority,
        activate_after=1,
        release_after=1,
    )


def pickup_actor_assessment(
    after: SceneState,
    *,
    expected_actor: str,
    expected_actor_kind: str | None,
    expected_xy_radius: float,
    cluster_safe_xy_radius: float,
    cluster_along_radius: float,
    cluster_cross_radius: float,
    cluster_vertical_radius: float,
    expected_along_radius: float | None = None,
    expected_cross_radius: float | None = None,
    cluster_safe_along_radius: float | None = None,
    cluster_safe_cross_radius: float | None = None,
) -> PickupActorAssessment:
    actor = after.actors.get(expected_actor)
    if actor is None:
        return PickupActorAssessment(
            actor_name=expected_actor,
            actor_kind=None,
            distance=math.inf,
            xy_distance=math.inf,
            along_distance=math.inf,
            cross_distance=math.inf,
            z_clearance=math.inf,
            center_aligned=False,
            cluster_obstructed=False,
            obstructing_actors=(),
            cluster_safe_centered=False,
            graspable=False,
        )

    distance = float(np.linalg.norm(after.tcp_position - actor.position))
    xy_distance = float(
        np.linalg.norm(after.tcp_position[:2] - actor.position[:2])
    )
    z_clearance = float(after.tcp_position[2] - actor.position[2])
    kind_matches = expected_actor_kind is None or actor.kind == expected_actor_kind
    closing_axis = np.asarray(
        after.metrics.get("gripper_closing_axis_xy", (0.0, 1.0)),
        dtype=np.float64,
    ).reshape(-1)
    if closing_axis.size != 2 or not np.isfinite(closing_axis).all():
        closing_axis = np.asarray((0.0, 1.0), dtype=np.float64)
    closing_norm = float(np.linalg.norm(closing_axis))
    if closing_norm <= 1e-8:
        closing_axis = np.asarray((0.0, 1.0), dtype=np.float64)
    else:
        closing_axis = closing_axis / closing_norm
    cross_axis = np.asarray((-closing_axis[1], closing_axis[0]))
    actor_delta_xy = after.tcp_position[:2] - actor.position[:2]
    along_distance = abs(float(np.dot(actor_delta_xy, closing_axis)))
    cross_distance = abs(float(np.dot(actor_delta_xy, cross_axis)))
    if expected_along_radius is None or expected_cross_radius is None:
        center_aligned = bool(kind_matches and xy_distance <= expected_xy_radius)
    else:
        center_aligned = bool(
            kind_matches
            and along_distance <= expected_along_radius
            and cross_distance <= expected_cross_radius
        )

    obstructing_actors: list[str] = []
    if actor.kind == "cube":
        for other_name, other in after.actors.items():
            if other_name == expected_actor or other.kind != "cube":
                continue
            if abs(float(other.position[2] - actor.position[2])) > cluster_vertical_radius:
                continue
            delta_xy = other.position[:2] - actor.position[:2]
            along = abs(float(np.dot(delta_xy, closing_axis)))
            cross = abs(float(np.dot(delta_xy, cross_axis)))
            if along <= cluster_along_radius and cross <= cluster_cross_radius:
                obstructing_actors.append(other_name)

    cluster_obstructed = bool(obstructing_actors)
    if (
        cluster_safe_along_radius is None
        or cluster_safe_cross_radius is None
    ):
        cluster_safe_centered = bool(
            cluster_obstructed and xy_distance <= cluster_safe_xy_radius
        )
    else:
        cluster_safe_centered = bool(
            cluster_obstructed
            and along_distance <= cluster_safe_along_radius
            and cross_distance <= cluster_safe_cross_radius
        )
    return PickupActorAssessment(
        actor_name=expected_actor,
        actor_kind=actor.kind,
        distance=distance,
        xy_distance=xy_distance,
        along_distance=along_distance,
        cross_distance=cross_distance,
        z_clearance=z_clearance,
        center_aligned=center_aligned,
        cluster_obstructed=cluster_obstructed,
        obstructing_actors=tuple(sorted(obstructing_actors)),
        cluster_safe_centered=cluster_safe_centered,
        graspable=bool(
            center_aligned
            and (not cluster_obstructed or cluster_safe_centered)
        ),
    )


def pickup_actor_signals(
    before: SceneState,
    after: SceneState,
    *,
    expected_actor: str,
    expected_actor_kind: str | None,
    forbidden_actors: Sequence[str],
    intent_radius: float,
    selection_margin: float,
    minimum_approach_delta: float,
    missed_grasp_radius: float,
    expected_xy_radius: float,
    expected_descent_height: float,
    closed_gripper_opening: float,
    cluster_safe_xy_radius: float = 0.0,
    cluster_along_radius: float = 0.0,
    cluster_cross_radius: float = 0.0,
    cluster_vertical_radius: float = 0.0,
) -> list[CandidateSignal]:
    positions = _positions(after)
    assessment = pickup_actor_assessment(
        after,
        expected_actor=expected_actor,
        expected_actor_kind=expected_actor_kind,
        expected_xy_radius=expected_xy_radius,
        cluster_safe_xy_radius=cluster_safe_xy_radius,
        cluster_along_radius=cluster_along_radius,
        cluster_cross_radius=cluster_cross_radius,
        cluster_vertical_radius=cluster_vertical_radius,
    )
    expected_distance = assessment.distance
    forbidden_actor, forbidden_distance = _closest(
        after.tcp_position,
        positions,
        forbidden_actors,
    )
    forbidden_xy_actor, forbidden_xy_distance = _closest_xy(
        after.tcp_position,
        positions,
        forbidden_actors,
    )
    expected_xy_distance = assessment.xy_distance
    expected_z_clearance = assessment.z_clearance
    expected_graspable = assessment.graspable
    forbidden_z_clearance = math.inf
    if forbidden_xy_actor is not None and forbidden_xy_actor in positions:
        forbidden_z_clearance = float(
            after.tcp_position[2] - positions[forbidden_xy_actor][2]
        )
    before_forbidden_distance = math.inf
    if forbidden_actor is not None and forbidden_actor in before.actors:
        before_forbidden_distance = float(
            np.linalg.norm(
                before.tcp_position - before.actors[forbidden_actor].position
            )
        )
    intents = action_intents(before, after)
    forbidden_approach = before_forbidden_distance - forbidden_distance
    expected_not_graspable_descent = bool(
        after.held_actor is None
        and intents.descending
        and 0.0 <= expected_z_clearance <= expected_descent_height
        and not expected_graspable
    )
    wrong_precontact = bool(
        forbidden_xy_actor is not None
        and after.held_actor is None
        and forbidden_xy_distance <= expected_xy_radius
        and 0.0 <= forbidden_z_clearance <= expected_descent_height
        and not expected_graspable
        and intents.descending
    )
    expected_grasped = bool(
        expected_actor in after.actors and after.actors[expected_actor].grasped
    )
    gripper_is_closed = bool(
        before.gripper_opening <= closed_gripper_opening
        and after.gripper_opening <= closed_gripper_opening
    )
    missed_grasp = bool(
        gripper_is_closed
        and not intents.closing
        and not intents.descending
        and after.held_actor is None
        and not expected_grasped
        and expected_distance <= missed_grasp_radius
        and expected_graspable
    )
    common_metrics = {
        "expected_distance": expected_distance,
        "expected_xy_actor": expected_actor,
        "expected_xy_distance": expected_xy_distance,
        "expected_along_distance": assessment.along_distance,
        "expected_cross_distance": assessment.cross_distance,
        "expected_z_clearance": expected_z_clearance,
        "expected_graspable": expected_graspable,
        "expected_actor_kind": assessment.actor_kind,
        "expected_center_aligned": assessment.center_aligned,
        "expected_cluster_obstructed": assessment.cluster_obstructed,
        "expected_obstructing_actors": assessment.obstructing_actors,
        "expected_cluster_safe_centered": assessment.cluster_safe_centered,
        "forbidden_distance": forbidden_distance,
        "forbidden_xy_actor": forbidden_xy_actor,
        "forbidden_xy_distance": forbidden_xy_distance,
        "forbidden_z_clearance": forbidden_z_clearance,
        "forbidden_approach_delta": forbidden_approach,
        "descending": intents.descending,
        "closing": intents.closing,
        "before_gripper_opening": before.gripper_opening,
        "after_gripper_opening": after.gripper_opening,
        "gripper_is_closed": gripper_is_closed,
    }
    thresholds = {
        "intent_radius": intent_radius,
        "selection_margin": selection_margin,
        "minimum_approach_delta": minimum_approach_delta,
        "missed_grasp_radius": missed_grasp_radius,
        "expected_xy_radius": expected_xy_radius,
        "cluster_safe_xy_radius": cluster_safe_xy_radius,
        "expected_descent_height": expected_descent_height,
        "closed_gripper_opening": closed_gripper_opening,
        "cluster_along_radius": cluster_along_radius,
        "cluster_cross_radius": cluster_cross_radius,
        "cluster_vertical_radius": cluster_vertical_radius,
    }
    return [
        semantic_signal(
            active=expected_not_graspable_descent,
            reason_code="pickup.expected_not_graspable_descent",
            recovery_mode=RecoveryMode.EXPERT_CONTINUE,
            expected_actor=expected_actor,
            observed_actor=None,
            metrics=common_metrics,
            thresholds=thresholds,
            priority=94,
        ),
        semantic_signal(
            active=wrong_precontact,
            reason_code="pickup.wrong_actor_precontact",
            recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
            expected_actor=expected_actor,
            observed_actor=forbidden_xy_actor,
            metrics=common_metrics,
            thresholds=thresholds,
            priority=95,
        ),
        semantic_signal(
            active=missed_grasp,
            reason_code="pickup.missed_grasp",
            recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
            expected_actor=expected_actor,
            observed_actor=None,
            metrics=common_metrics,
            thresholds=thresholds,
            priority=92,
        ),
    ]


def held_identity_signal(
    after: SceneState,
    *,
    expected_actors: Sequence[str],
    reason_code: str = "identity.wrong_actor_held",
) -> CandidateSignal:
    held_actor = after.held_actor
    wrong_actor = held_actor is not None and held_actor not in expected_actors
    return semantic_signal(
        active=wrong_actor,
        reason_code=reason_code,
        recovery_mode=RecoveryMode.RETURN_WRONG_OBJECT,
        expected_actor=expected_actors[0] if len(expected_actors) == 1 else None,
        observed_actor=held_actor,
        metrics={
            "held_actor": held_actor,
            "expected_actor_names": tuple(expected_actors),
        },
        thresholds={"exact_actor_identity_required": True},
        priority=99,
    )


def required_held_signal(
    before: SceneState,
    after: SceneState,
    *,
    expected_actors: Sequence[str],
) -> CandidateSignal:
    intents = action_intents(before, after)
    released_expected_actor = bool(
        before.held_actor in expected_actors
        and after.held_actor is None
        and intents.opening
    )
    return semantic_signal(
        active=after.held_actor is None and not released_expected_actor,
        reason_code="identity.required_actor_not_held",
        recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
        expected_actor=expected_actors[0] if len(expected_actors) == 1 else None,
        observed_actor=None,
        metrics={
            "expected_actor_names": tuple(expected_actors),
            "released_expected_actor": released_expected_actor,
        },
        thresholds={"held_actor_required": True},
        priority=97,
    )


def place_signals(
    before: SceneState,
    after: SceneState,
    *,
    expected_actors: Sequence[str],
    expected_targets: Sequence[str],
    forbidden_targets: Sequence[str],
    release_radius: float,
    transport_radius: float,
    selection_margin: float,
    minimum_motion: float,
    minimum_improvement: float,
    horizontal_target_distance: bool = False,
) -> list[CandidateSignal]:
    signals = [
        held_identity_signal(after, expected_actors=expected_actors),
        required_held_signal(before, after, expected_actors=expected_actors),
    ]
    intents = action_intents(before, after)
    held_actor = after.held_actor
    manipulated_actor = held_actor
    if (
        manipulated_actor is None
        and intents.opening
        and before.held_actor in expected_actors
    ):
        manipulated_actor = before.held_actor
    if manipulated_actor is None or manipulated_actor not in expected_actors:
        return signals
    if manipulated_actor not in before.actors or manipulated_actor not in after.actors:
        return signals
    after_position = after.actors[manipulated_actor].position
    before_position = before.actors[manipulated_actor].position
    closest = _closest_xy if horizontal_target_distance else _closest
    expected_target, expected_distance = closest(
        after_position, after.landmarks, expected_targets
    )
    forbidden_target, forbidden_distance = closest(
        after_position, after.landmarks, forbidden_targets
    )
    before_expected_distance = math.inf
    if expected_target is not None and expected_target in before.landmarks:
        expected_delta = before_position - before.landmarks[expected_target]
        if horizontal_target_distance:
            expected_delta = expected_delta[:2]
        before_expected_distance = float(np.linalg.norm(expected_delta))
    before_forbidden_distance = math.inf
    if forbidden_target is not None and forbidden_target in before.landmarks:
        forbidden_delta = before_position - before.landmarks[forbidden_target]
        if horizontal_target_distance:
            forbidden_delta = forbidden_delta[:2]
        before_forbidden_distance = float(np.linalg.norm(forbidden_delta))
    actor_motion = float(np.linalg.norm(after_position - before_position))
    improvement = before_expected_distance - expected_distance
    forbidden_improvement = before_forbidden_distance - forbidden_distance
    wrong_target_closer = bool(
        forbidden_target is not None
        and forbidden_distance + selection_margin < expected_distance
        and forbidden_distance <= transport_radius
    )
    release_outside = bool(intents.opening and expected_distance > release_radius)
    wrong_transport = bool(
        actor_motion >= minimum_motion
        and improvement <= -minimum_improvement
        and forbidden_improvement >= minimum_improvement
        and wrong_target_closer
        and expected_distance > release_radius
    )
    metrics = {
        "held_actor": held_actor,
        "manipulated_actor": manipulated_actor,
        "expected_target": expected_target,
        "expected_distance": expected_distance,
        "forbidden_target": forbidden_target,
        "forbidden_distance": forbidden_distance,
        "actor_motion": actor_motion,
        "expected_improvement": improvement,
        "forbidden_improvement": forbidden_improvement,
        "opening": intents.opening,
        "horizontal_target_distance": horizontal_target_distance,
    }
    thresholds = {
        "release_radius": release_radius,
        "transport_radius": transport_radius,
        "selection_margin": selection_margin,
        "minimum_motion": minimum_motion,
        "minimum_improvement": minimum_improvement,
        "horizontal_target_distance": horizontal_target_distance,
    }
    signals.append(
        semantic_signal(
            active=release_outside or wrong_transport,
            reason_code="place.wrong_target_or_release",
            recovery_mode=RecoveryMode.LIFT_HOLD_THEN_PLACE,
            expected_actor=expected_target,
            observed_actor=forbidden_target if wrong_transport else None,
            metrics=metrics,
            thresholds=thresholds,
            priority=95 if release_outside else 88,
        )
    )
    return signals


def button_signals(
    before: SceneState,
    after: SceneState,
    *,
    expected_buttons: Sequence[str],
    forbidden_buttons: Sequence[str],
    intent_radius: float,
    selection_margin: float,
    minimum_approach_delta: float,
    descent_height: float,
    prerequisite_complete: bool = True,
) -> list[CandidateSignal]:
    expected_button, expected_distance = _closest(
        after.tcp_position,
        after.landmarks,
        expected_buttons,
    )
    forbidden_button, forbidden_distance = _closest(
        after.tcp_position,
        after.landmarks,
        forbidden_buttons,
    )
    before_forbidden_distance = math.inf
    if forbidden_button is not None and forbidden_button in before.landmarks:
        before_forbidden_distance = float(
            np.linalg.norm(before.tcp_position - before.landmarks[forbidden_button])
        )
    intents = action_intents(before, after)
    approach = before_forbidden_distance - forbidden_distance
    wrong_is_closer = bool(forbidden_distance + selection_margin < expected_distance)
    forbidden_xy_distance = math.inf
    forbidden_z_clearance = math.inf
    if forbidden_button is not None and forbidden_button in after.landmarks:
        forbidden_position = after.landmarks[forbidden_button]
        forbidden_xy_distance = float(
            np.linalg.norm(after.tcp_position[:2] - forbidden_position[:2])
        )
        forbidden_z_clearance = float(
            after.tcp_position[2] - forbidden_position[2]
        )
    approaching_button = bool(
        forbidden_xy_distance <= intent_radius
        and abs(forbidden_z_clearance) <= descent_height
        and intents.descending
    )
    active = bool(
        forbidden_button is not None and wrong_is_closer and approaching_button
    )
    if not prerequisite_complete and "button" in after.landmarks:
        before_button_distance = float(
            np.linalg.norm(before.tcp_position - before.landmarks["button"])
        )
        button_distance = float(
            np.linalg.norm(after.tcp_position - after.landmarks["button"])
        )
        before_button_xy_distance = float(
            np.linalg.norm(before.tcp_position[:2] - before.landmarks["button"][:2])
        )
        button_xy_distance = float(
            np.linalg.norm(after.tcp_position[:2] - after.landmarks["button"][:2])
        )
        button_z_clearance = float(
            after.tcp_position[2] - after.landmarks["button"][2]
        )
        active = active or bool(
            button_xy_distance <= intent_radius
            and abs(button_z_clearance) <= descent_height
            and intents.descending
        )
        forbidden_button = "button" if active else forbidden_button
        forbidden_distance = button_distance
    return [
        semantic_signal(
            active=active,
            reason_code=(
                "button.prerequisite_incomplete"
                if not prerequisite_complete
                else "button.wrong_button_precontact"
            ),
            recovery_mode=(
                RecoveryMode.EXPERT_CONTINUE
                if not prerequisite_complete
                else RecoveryMode.OPEN_AND_RETRACT
            ),
            expected_actor=expected_button,
            observed_actor=forbidden_button,
            metrics={
                "expected_distance": expected_distance,
                "forbidden_distance": forbidden_distance,
                "descending": intents.descending,
                "prerequisite_complete": prerequisite_complete,
                "button_xy_distance": (
                    button_xy_distance
                    if not prerequisite_complete and "button" in after.landmarks
                    else math.inf
                ),
                "button_z_clearance": (
                    button_z_clearance
                    if not prerequisite_complete and "button" in after.landmarks
                    else forbidden_z_clearance
                ),
            },
            thresholds={
                "intent_radius": intent_radius,
                "selection_margin": selection_margin,
                "minimum_approach_delta": minimum_approach_delta,
                "button_xy_intent_radius": intent_radius,
                "descent_height": descent_height,
            },
            priority=96,
        )
    ]


def route_signals(
    before: SceneState,
    after: SceneState,
    *,
    tracked_point: str,
    corridor_start: str,
    expected_waypoint: str,
    forbidden_waypoints: Sequence[str],
    target_radius: float,
    wrong_waypoint_radius: float,
    maximum_cross_track: float,
    minimum_motion: float,
) -> list[CandidateSignal]:
    if tracked_point not in before.points or tracked_point not in after.points:
        return []
    if (
        corridor_start not in after.landmarks
        or expected_waypoint not in after.landmarks
    ):
        return []
    before_point = before.points[tracked_point]
    after_point = after.points[tracked_point]
    start = after.landmarks[corridor_start]
    target = after.landmarks[expected_waypoint]
    motion = after_point - before_point
    motion_distance = float(np.linalg.norm(motion))
    before_distance = float(np.linalg.norm(before_point - target))
    after_distance = float(np.linalg.norm(after_point - target))
    improvement = before_distance - after_distance
    target_direction = target - before_point
    target_direction_norm = float(np.linalg.norm(target_direction))
    direction_cosine = 1.0
    if motion_distance > 1e-12 and target_direction_norm > 1e-12:
        direction_cosine = float(
            np.dot(motion, target_direction) / (motion_distance * target_direction_norm)
        )
    cross_track = point_to_segment_distance(after_point, start, target)
    wrong_waypoint, wrong_distance = _closest(
        after_point,
        after.landmarks,
        forbidden_waypoints,
    )
    meaningful_motion = motion_distance >= minimum_motion
    moving_away = bool(
        meaningful_motion
        and after_distance > target_radius
        and improvement <= 0.0
        and direction_cosine <= 0.0
    )
    outside_corridor = bool(meaningful_motion and cross_track > maximum_cross_track)
    wrong_waypoint_risk = bool(
        meaningful_motion
        and wrong_waypoint is not None
        and wrong_distance <= wrong_waypoint_radius
        and wrong_distance < after_distance
    )
    return [
        semantic_signal(
            active=moving_away or outside_corridor or wrong_waypoint_risk,
            reason_code=(
                "route.wrong_waypoint_risk"
                if wrong_waypoint_risk
                else (
                    "route.outside_corridor"
                    if outside_corridor
                    else "route.meaningful_motion_away"
                )
            ),
            recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
            expected_actor=expected_waypoint,
            observed_actor=wrong_waypoint if wrong_waypoint_risk else None,
            metrics={
                "motion": motion_distance,
                "before_distance": before_distance,
                "after_distance": after_distance,
                "improvement": improvement,
                "direction_cosine": direction_cosine,
                "cross_track": cross_track,
                "wrong_waypoint_distance": wrong_distance,
            },
            thresholds={
                "target_radius": target_radius,
                "wrong_waypoint_radius": wrong_waypoint_radius,
                "maximum_cross_track": maximum_cross_track,
                "minimum_motion": minimum_motion,
                "maximum_wrong_direction_cosine": 0.0,
            },
            priority=94 if wrong_waypoint_risk else 87,
        )
    ]

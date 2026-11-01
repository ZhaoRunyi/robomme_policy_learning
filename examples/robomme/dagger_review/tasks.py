"""dagger_review / tasks: shared behavior owned by this layer."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any
import numpy as np
from .signals import (
    ActorState,
    AdapterContractError,
    PredictedAction,
    SceneState,
    SubtaskState,
)
from dataclasses import dataclass
from .signals import (
    action_intents,
    button_signals,
    held_identity_signal,
    pickup_actor_assessment,
    pickup_actor_signals,
    place_signals,
    required_held_signal,
    semantic_signal,
)
from .signals import (
    CandidateSignal,
    InterventionDecision,
    RecoveryMode,
    SignalClass,
)


ActionPredictor = Callable[[np.ndarray, SceneState], PredictedAction | Mapping[str, Any]]
TimingMetricsProvider = Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value: Any, field_name: str) -> Any:
    array = _to_numpy(value)
    if array.size != 1:
        raise AdapterContractError(
            f"{field_name} must contain exactly one value, got shape {array.shape}"
        )
    return array.reshape(-1)[0].item()


def _scalar_int(value: Any, field_name: str) -> int:
    scalar = _scalar(value, field_name)
    if isinstance(scalar, (bool, np.bool_)):
        raise AdapterContractError(f"{field_name} must be an integer, not bool")
    integer = int(scalar)
    if float(scalar) != float(integer):
        raise AdapterContractError(f"{field_name} is not integral: {scalar!r}")
    return integer


def _scalar_float(value: Any, field_name: str) -> float:
    scalar = float(_scalar(value, field_name))
    if not math.isfinite(scalar):
        raise AdapterContractError(f"{field_name} must be finite")
    return scalar


def _scalar_bool(value: Any, field_name: str) -> bool:
    scalar = _scalar(value, field_name)
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, np.integer)) and scalar in (0, 1):
        return bool(scalar)
    raise AdapterContractError(f"{field_name} must be a scalar bool, got {scalar!r}")


def _position(value: Any, field_name: str) -> np.ndarray:
    array = _to_numpy(value).astype(np.float64, copy=False)
    if array.shape == (1, 3):
        array = array[0]
    if array.shape != (3,):
        raise AdapterContractError(
            f"{field_name} must have shape (3,) or (1, 3), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise AdapterContractError(f"{field_name} contains non-finite values")
    return np.array(array, dtype=np.float64, copy=True)


def _require_attr(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise AdapterContractError(
            f"{type(value).__name__} is missing required runtime attribute {name!r}"
        )
    return getattr(value, name)


def _require_mapping_value(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise AdapterContractError(f"task entry is missing required field {name!r}")
    return mapping[name]


def _require_sequence(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AdapterContractError(f"{field_name} must be a sequence")
    return list(value)


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterContractError(f"{field_name} must be a mapping")
    return value


def _actor_name(actor: Any, field_name: str) -> str:
    name = _require_attr(actor, "name")
    if not isinstance(name, str) or not name:
        raise AdapterContractError(f"{field_name}.name must be a non-empty string")
    return name


def _actor_position(actor: Any, field_name: str) -> np.ndarray:
    pose = _require_attr(actor, "pose")
    return _position(_require_attr(pose, "p"), f"{field_name}.pose.p")


def _same_actor(first: Any, second: Any) -> bool:
    return first is second


def _actor_in(actor: Any, actors: Sequence[Any]) -> bool:
    return any(_same_actor(actor, candidate) for candidate in actors)


def _actor_index(actor: Any, actors: Sequence[Any], field_name: str) -> int:
    matches = [index for index, candidate in enumerate(actors) if actor is candidate]
    if len(matches) != 1:
        raise AdapterContractError(
            f"{field_name} must occur exactly once by identity, got {len(matches)}"
        )
    return matches[0]


def _segment_objects(task_entry: Mapping[str, Any]) -> list[Any]:
    segment = task_entry.get("segment")
    if segment is None:
        return []
    if isinstance(segment, (list, tuple)):
        return list(segment)
    return [segment]


def _require_exact_segment(
    task_entry: Mapping[str, Any],
    expected_objects: Sequence[Any],
    field_name: str,
) -> None:
    observed_objects = _segment_objects(task_entry)
    expected_objects = list(expected_objects)
    if len(observed_objects) != len(expected_objects) or any(
        observed is not expected
        for observed, expected in zip(observed_objects, expected_objects)
    ):
        raise AdapterContractError(
            f"{field_name} must match the runtime objects exactly by identity"
        )


def _require_one_segment_actor(
    task_entry: Mapping[str, Any],
    candidates: Sequence[Any],
    field_name: str,
) -> Any:
    segment = _segment_objects(task_entry)
    if len(segment) != 1 or not _actor_in(segment[0], candidates):
        raise AdapterContractError(
            f"{field_name} must contain exactly one candidate actor by identity"
        )
    return segment[0]


def _policy_entries_before(task_list: Sequence[Any], task_index: int) -> list[Any]:
    return [
        entry
        for entry in task_list[:task_index]
        if isinstance(entry, Mapping) and entry.get("demonstration") is False
    ]


def _identity_complement(
    candidates: Sequence[Any],
    selected: Sequence[Any],
) -> list[Any]:
    return [candidate for candidate in candidates if not _actor_in(candidate, selected)]


def _unique_identity_sequence(value: Any, field_name: str) -> list[Any]:
    sequence = _require_sequence(value, field_name)
    identities = [id(item) for item in sequence]
    if len(set(identities)) != len(identities):
        raise AdapterContractError(f"{field_name} contains duplicate actor identities")
    return sequence


class RoboMMETaskAdapter:
    """Strict common boundary shared by task-specific adapters."""

    canonical_task_name = "RoboMME"

    def __init__(
        self,
        raw_env: Any,
        action_predictor: ActionPredictor,
        step_duration_seconds: float,
    ) -> None:
        if not callable(action_predictor):
            raise AdapterContractError("action_predictor must be callable")
        if not math.isfinite(step_duration_seconds) or step_duration_seconds <= 0.0:
            raise AdapterContractError(
                "step_duration_seconds must be an explicit positive finite value"
            )
        self.raw_env = raw_env
        self._action_predictor = action_predictor
        self._step_duration_seconds = float(step_duration_seconds)

    @property
    def task_name(self) -> str:
        return self.canonical_task_name

    def predict_action(
        self,
        action: np.ndarray,
        scene: SceneState,
    ) -> PredictedAction:
        prediction = self._action_predictor(action, scene)
        if isinstance(prediction, PredictedAction):
            return prediction
        if not isinstance(prediction, Mapping):
            raise AdapterContractError(
                "action_predictor must return PredictedAction or a mapping"
            )
        required = ("next_tcp_position", "next_gripper_opening")
        missing = [name for name in required if name not in prediction]
        if missing:
            raise AdapterContractError(
                f"action prediction is missing required fields: {missing}"
            )
        return PredictedAction(
            next_tcp_position=prediction["next_tcp_position"],
            next_gripper_opening=float(prediction["next_gripper_opening"]),
            predicted_actor_positions=prediction.get(
                "predicted_actor_positions",
                {},
            ),
            predicted_points=prediction.get("predicted_points", {}),
            intents=frozenset(prediction.get("intents", ())),
            metadata=prediction.get("metadata", {}),
        )

    def _task_entry(self) -> tuple[int, Mapping[str, Any], int | None]:
        task_list = _require_sequence(
            _require_attr(self.raw_env, "task_list"),
            "raw_env.task_list",
        )
        task_index = _scalar_int(
            _require_attr(self.raw_env, "timestep"),
            "raw_env.timestep",
        )
        if task_index < 0 or task_index >= len(task_list):
            raise AdapterContractError(
                f"timestep {task_index} is outside task_list of length "
                f"{len(task_list)}; detector must not run before/after an episode"
            )
        task_entry = _require_mapping(
            task_list[task_index],
            f"raw_env.task_list[{task_index}]",
        )
        demonstration = _require_mapping_value(task_entry, "demonstration")
        if not isinstance(demonstration, (bool, np.bool_)):
            raise AdapterContractError("task entry demonstration must be bool")
        if bool(demonstration):
            raise AdapterContractError(
                "pre-action policy detector cannot run on a demonstration task entry"
            )
        reported_task_index = None
        if hasattr(self.raw_env, "current_task_index"):
            reported_task_index = _scalar_int(
                self.raw_env.current_task_index,
                "raw_env.current_task_index",
            )
        return task_index, task_entry, reported_task_index

    def _step(self) -> int:
        return _scalar_int(
            _require_attr(self.raw_env, "elapsed_steps"),
            "raw_env.elapsed_steps",
        )

    def _tcp_position(self) -> np.ndarray:
        agent = _require_attr(self.raw_env, "agent")
        tcp = _require_attr(agent, "tcp")
        return _actor_position(tcp, "raw_env.agent.tcp")

    def _gripper_closing_axis_xy(self) -> tuple[float, float]:
        agent = _require_attr(self.raw_env, "agent")
        tcp = _require_attr(agent, "tcp")
        pose = _require_attr(tcp, "pose")
        to_matrix = getattr(pose, "to_transformation_matrix", None)
        if to_matrix is None:
            return (0.0, 1.0)
        if not callable(to_matrix):
            raise AdapterContractError(
                "raw_env.agent.tcp.pose.to_transformation_matrix must be callable"
            )
        matrix = _to_numpy(to_matrix()).astype(np.float64, copy=False)
        if matrix.ndim == 3 and matrix.shape[0] == 1:
            matrix = matrix[0]
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise AdapterContractError(
                "TCP transformation matrix must be finite with shape (4, 4)"
            )
        axis = matrix[:2, 1]
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-8:
            return (0.0, 1.0)
        axis = axis / norm
        return float(axis[0]), float(axis[1])

    def _gripper_opening(self) -> tuple[float, bool]:
        agent = _require_attr(self.raw_env, "agent")
        robot = _require_attr(agent, "robot")
        get_qpos = _require_attr(robot, "get_qpos")
        if not callable(get_qpos):
            raise AdapterContractError("raw_env.agent.robot.get_qpos must be callable")
        qpos = _to_numpy(get_qpos()).astype(np.float64, copy=False)
        if qpos.ndim == 2 and qpos.shape[0] == 1:
            qpos = qpos[0]
        if qpos.ndim != 1 or qpos.size < 7:
            raise AdapterContractError(
                f"robot qpos must have shape (>=7,) or (1, >=7), got {qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise AdapterContractError("robot qpos contains non-finite values")
        if qpos.size == 7:
            return 0.0, False
        if qpos.size < 9:
            raise AdapterContractError(
                "robot qpos has an ambiguous gripper layout; expected 7 or at least 9"
            )
        return float(qpos[-2:].sum()), True

    def _actor_states(
        self,
        actor_kinds: Sequence[tuple[Any, str]],
        *,
        semantic_names_by_identity: Mapping[int, str] | None = None,
        non_graspable_identities: frozenset[int] = frozenset(),
    ) -> tuple[dict[str, ActorState], tuple[str, ...], str | None]:
        agent = _require_attr(self.raw_env, "agent")
        is_grasping = _require_attr(agent, "is_grasping")
        if not callable(is_grasping):
            raise AdapterContractError("raw_env.agent.is_grasping must be callable")
        states: dict[str, ActorState] = {}
        grasped_names: list[str] = []
        seen_identities: set[int] = set()
        for actor, kind in actor_kinds:
            identity = id(actor)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            native_name = _actor_name(actor, "actor")
            if semantic_names_by_identity is None:
                name = native_name
            else:
                name = semantic_names_by_identity.get(identity, native_name)
            if not isinstance(name, str) or not name:
                raise AdapterContractError(
                    "semantic actor names must be non-empty strings"
                )
            if name in states:
                raise AdapterContractError(
                    f"actor names must be unique, duplicate {name!r}"
                )
            if identity in non_graspable_identities:
                grasped = False
            else:
                try:
                    grasped = _scalar_bool(
                        is_grasping(actor),
                        f"is_grasping({name})",
                    )
                except AdapterContractError:
                    raise
                except Exception as error:
                    raise AdapterContractError(
                        f"is_grasping({name}) failed: "
                        f"{type(error).__name__}: {error}"
                    ) from error
            attributes = {}
            if name != native_name:
                attributes["native_name"] = native_name
            states[name] = ActorState(
                name=name,
                kind=kind,
                position=_actor_position(actor, f"actor[{name!r}]"),
                grasped=grasped,
                attributes=attributes,
            )
            if grasped:
                grasped_names.append(name)
        held_actor = grasped_names[0] if len(grasped_names) == 1 else None
        return states, tuple(grasped_names), held_actor

    def _failure_state(self) -> tuple[bool, bool]:
        current_failure = False
        if hasattr(self.raw_env, "current_task_failure"):
            current_failure = _scalar_bool(
                self.raw_env.current_task_failure,
                "raw_env.current_task_failure",
            )
        episode_failure = False
        if hasattr(self.raw_env, "failureflag"):
            episode_failure = _scalar_bool(
                self.raw_env.failureflag,
                "raw_env.failureflag",
            )
        success = False
        if hasattr(self.raw_env, "successflag"):
            success = _scalar_bool(
                self.raw_env.successflag,
                "raw_env.successflag",
            )
        official_failure = current_failure or episode_failure
        return official_failure, official_failure or success

    def _scene(
        self,
        task_index: int,
        task_entry: Mapping[str, Any],
        phase: str,
        expected_actor: str | None,
        expected_target: str | None,
        actors: Mapping[str, ActorState],
        landmarks: Mapping[str, np.ndarray],
        held_actor: str | None,
        metrics: Mapping[str, Any],
        points: Mapping[str, np.ndarray] | None = None,
    ) -> SceneState:
        task_name = _require_mapping_value(task_entry, "name")
        if not isinstance(task_name, str) or not task_name:
            raise AdapterContractError("task entry name must be a non-empty string")
        opening, has_gripper = self._gripper_opening()
        official_failure, terminal = self._failure_state()
        step = self._step()
        scene_metrics = dict(metrics)
        for actor_key, actor_state in actors.items():
            if actor_key != actor_state.name:
                raise AdapterContractError(
                    f"actors key {actor_key!r} does not match ActorState.name "
                    f"{actor_state.name!r}"
                )
        if held_actor is not None and held_actor not in actors:
            raise AdapterContractError(
                f"held_actor {held_actor!r} is absent from actors"
            )
        if expected_actor is not None and expected_actor not in actors:
            raise AdapterContractError(
                f"expected_actor {expected_actor!r} is absent from actors"
            )
        if expected_target is not None and expected_target not in landmarks:
            raise AdapterContractError(
                f"expected_target {expected_target!r} is absent from landmarks"
            )
        scene_metrics["gripper_control_available"] = has_gripper
        scene_metrics["gripper_closing_axis_xy"] = self._gripper_closing_axis_xy()
        scene_metrics["task_solver_callable"] = callable(task_entry.get("solve"))
        scene_metrics["failure_func_callable"] = callable(
            task_entry.get("failure_func")
        )
        return SceneState(
            step=step,
            elapsed_seconds=step * self._step_duration_seconds,
            task_name=self.task_name,
            subtask=SubtaskState(
                index=task_index,
                name=task_name,
                phase=phase,
                expected_actor=expected_actor,
                expected_target=expected_target,
            ),
            tcp_position=self._tcp_position(),
            gripper_opening=opening,
            actors=actors,
            landmarks=landmarks,
            points=points or {},
            held_actor=held_actor,
            metrics=scene_metrics,
            official_failure=official_failure,
            terminal=terminal,
        )


def _binfill_valid_actor_names(
    cube_colors: Mapping[str, str],
    remaining_by_color: Mapping[str, int],
) -> tuple[str, ...]:
    return tuple(
        name
        for name, color in cube_colors.items()
        if remaining_by_color[color] > 0
    )


class BinFillTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "BinFill"

    _PHASES = {
        "pick up the cube": "pickup",
        "put it into the bin": "place_in_bin",
        "press the button": "press_button",
    }

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label not in self._PHASES:
            raise AdapterContractError(
                f"unsupported BinFill choice_label {choice_label!r}"
            )
        phase = self._PHASES[choice_label]
        all_cubes = _require_sequence(
            _require_attr(self.raw_env, "all_cubes"),
            "raw_env.all_cubes",
        )
        color_lists = {
            "red": _require_sequence(
                _require_attr(self.raw_env, "red_cubes"),
                "raw_env.red_cubes",
            ),
            "blue": _require_sequence(
                _require_attr(self.raw_env, "blue_cubes"),
                "raw_env.blue_cubes",
            ),
            "green": _require_sequence(
                _require_attr(self.raw_env, "green_cubes"),
                "raw_env.green_cubes",
            ),
        }
        actor_kinds = [(cube, "cube") for cube in all_cubes]
        actors, grasped_names, held_actor = self._actor_states(actor_kinds)
        cube_colors: dict[str, str] = {}
        for cube in all_cubes:
            memberships = [
                color for color, cubes in color_lists.items() if _actor_in(cube, cubes)
            ]
            if len(memberships) != 1:
                raise AdapterContractError(
                    f"BinFill cube {_actor_name(cube, 'cube')!r} must belong to "
                    f"exactly one color list, got {memberships}"
                )
            cube_colors[_actor_name(cube, "cube")] = memberships[0]

        expected_actor: str | None = None
        if phase == "pickup":
            segment = _segment_objects(task_entry)
            if len(segment) != 1 or not _actor_in(segment[0], all_cubes):
                raise AdapterContractError(
                    "BinFill pickup segment must contain the exact scheduled cube"
                )
            expected_actor = _actor_name(segment[0], "task_entry.segment[0]")
        elif phase == "place_in_bin":
            if task_index < 1:
                raise AdapterContractError("BinFill place task has no preceding pickup")
            previous_entry = _require_mapping(
                self.raw_env.task_list[task_index - 1],
                "preceding BinFill pickup task",
            )
            previous_segment = _segment_objects(previous_entry)
            if len(previous_segment) != 1 or not _actor_in(
                previous_segment[0],
                all_cubes,
            ):
                raise AdapterContractError(
                    "preceding BinFill pickup task does not identify one cube"
                )
            expected_actor = _actor_name(
                previous_segment[0],
                "preceding pickup segment",
            )

        in_bin = {
            color: _scalar_int(
                _require_attr(self.raw_env, f"{color}_cubes_in_bin"),
                f"raw_env.{color}_cubes_in_bin",
            )
            for color in color_lists
        }
        target = {
            color: _scalar_int(
                _require_attr(self.raw_env, f"{color}_cubes_target_number"),
                f"raw_env.{color}_cubes_target_number",
            )
            for color in color_lists
        }
        remaining = {color: target[color] - in_bin[color] for color in color_lists}
        objects_complete = all(in_bin[color] == target[color] for color in color_lists)
        board = _require_attr(self.raw_env, "board_with_hole")
        button = _require_attr(self.raw_env, "button")
        landmarks = {
            "bin": _actor_position(board, "raw_env.board_with_hole"),
            "button": _actor_position(button, "raw_env.button"),
        }
        scheduled_actor = expected_actor
        color_valid_actor_names = _binfill_valid_actor_names(cube_colors, remaining)
        removed_actor_names = tuple(
            name
            for name, actor in actors.items()
            if float(np.linalg.norm(actor.position[:2])) >= 2.0
        )
        valid_actor_names = tuple(
            name
            for name in color_valid_actor_names
            if name not in removed_actor_names
        )
        if phase == "place_in_bin" and held_actor in valid_actor_names:
            expected_actor = held_actor
        forbidden_names = tuple(
            name for name in actors if name not in valid_actor_names
        )
        metrics = {
            "expected_actor_names": valid_actor_names,
            "forbidden_actor_names": forbidden_names,
            "expected_target_names": (
                ("bin",) if phase == "place_in_bin" else ()
            ),
            "forbidden_target_names": (
                ("button",) if phase == "place_in_bin" else ()
            ),
            "scheduled_actor_name": scheduled_actor,
            "valid_actor_selection": "any_cube_of_any_still_needed_color",
            "removed_actor_names": removed_actor_names,
            "cube_colors": cube_colors,
            "in_bin_by_color": in_bin,
            "target_by_color": target,
            "remaining_by_color": remaining,
            "cube_half_size": float(
                _scalar(
                    _require_attr(self.raw_env, "cube_half_size"),
                    "raw_env.cube_half_size",
                )
            ),
            "binfill_delay_correct_ungraspable": bool(
                _scalar_int(_require_attr(self.raw_env, "seed"), "raw_env.seed")
                % 20
                == 7
            ),
            "objects_complete": objects_complete,
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "native_success_uses_all_cubes": True,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        expected_target = "bin" if phase == "place_in_bin" else None
        if phase == "press_button":
            expected_target = "button"
        return self._scene(
            task_index,
            task_entry,
            phase,
            expected_actor,
            expected_target,
            actors,
            landmarks,
            held_actor,
            metrics,
        )


class PickHighlightTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "PickHighlight"

    _PHASES = {
        "press button": "press_button",
        "pick up the highlighted cube": "pickup",
        "place the cube onto the table": "putdown",
    }

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label not in self._PHASES:
            raise AdapterContractError(
                f"unsupported PickHighlight choice_label {choice_label!r}"
            )
        phase = self._PHASES[choice_label]
        all_cubes = _require_sequence(
            _require_attr(self.raw_env, "all_cubes"),
            "raw_env.all_cubes",
        )
        target_cubes = _require_sequence(
            _require_attr(self.raw_env, "target_cubes"),
            "raw_env.target_cubes",
        )
        actors, grasped_names, held_actor = self._actor_states(
            [(cube, "cube") for cube in all_cubes]
        )
        target_names = tuple(_actor_name(cube, "target_cube") for cube in target_cubes)
        if len(set(target_names)) != len(target_names):
            raise AdapterContractError("PickHighlight target cube names are not unique")
        counts = _require_mapping(
            _require_attr(self.raw_env, "target_cube_pickup_counts"),
            "raw_env.target_cube_pickup_counts",
        )
        if set(counts) != set(target_names):
            raise AdapterContractError(
                "target_cube_pickup_counts keys must exactly match target cube names"
            )
        pickup_counts = {
            name: _scalar_int(counts[name], f"pickup_count[{name!r}]")
            for name in target_names
        }
        remaining_names = tuple(
            name for name in target_names if pickup_counts[name] < 1
        )
        non_target_names = tuple(
            _actor_name(cube, "non_target_cube")
            for cube in all_cubes
            if not _actor_in(cube, target_cubes)
        )
        ordered_segment_name = None
        segment = _segment_objects(task_entry)
        if phase == "pickup":
            if len(segment) != 1 or not _actor_in(segment[0], target_cubes):
                raise AdapterContractError(
                    "PickHighlight pickup segment must contain one target cube"
                )
            ordered_segment_name = _actor_name(segment[0], "task_entry.segment")
        expected_actor = remaining_names[0] if len(remaining_names) == 1 else None
        if phase == "putdown":
            current_pickup = getattr(self.raw_env, "currentpickup", None)
            if current_pickup is not None and _actor_in(current_pickup, target_cubes):
                expected_actor = _actor_name(current_pickup, "raw_env.currentpickup")
            elif held_actor in target_names:
                expected_actor = held_actor
            else:
                expected_actor = None
        if phase == "press_button":
            expected_actor = None
        button = _require_attr(self.raw_env, "button")
        metrics = {
            "expected_actor_names": remaining_names if phase == "pickup" else (
                () if expected_actor is None else (expected_actor,)
            ),
            "forbidden_actor_names": non_target_names,
            "target_actor_names": target_names,
            "pickup_counts": pickup_counts,
            "objects_complete": len(remaining_names) == 0,
            "button_prerequisite_complete": phase == "press_button",
            "ordered_task_actor": ordered_segment_name,
            "semantic_order_mode": "remaining_set",
            "native_task_list_order_is_stricter_than_prompt": True,
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        return self._scene(
            task_index,
            task_entry,
            phase,
            expected_actor,
            "button" if phase == "press_button" else None,
            actors,
            {"button": _actor_position(button, "raw_env.button")},
            held_actor,
            metrics,
        )


class MoveCubeTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "MoveCube"

    _PHASES = {
        "pick up the peg": "pickup_tool",
        "hook the cube to the target with the peg": "peg_push",
        "close gripper and push the cube to the target": "gripper_push",
        "pick up the cube": "pickup_cube",
        "place the cube onto the target": "place_cube",
    }

    _WAY_PHASES = {
        "peg_push": {"pickup_tool", "peg_push"},
        "gripper_push": {"gripper_push"},
        "grasp_putdown": {"pickup_cube", "place_cube"},
    }

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label not in self._PHASES:
            raise AdapterContractError(
                f"unsupported MoveCube choice_label {choice_label!r}"
            )
        phase = self._PHASES[choice_label]
        way = _require_attr(self.raw_env, "way")
        if way not in self._WAY_PHASES or phase not in self._WAY_PHASES[way]:
            raise AdapterContractError(
                f"MoveCube way/phase mismatch: way={way!r}, phase={phase!r}"
            )
        cube = _require_attr(self.raw_env, "cube")
        correct_tool = _require_attr(self.raw_env, "grasp_target")
        wrong_tool = _require_attr(self.raw_env, "grasp_target_false")
        actor_kinds = [
            (cube, "cube"),
            (correct_tool, "peg_end"),
            (wrong_tool, "peg_end"),
        ]
        actors, grasped_names, held_actor = self._actor_states(actor_kinds)
        cube_name = _actor_name(cube, "raw_env.cube")
        correct_tool_name = _actor_name(correct_tool, "raw_env.grasp_target")
        wrong_tool_name = _actor_name(wrong_tool, "raw_env.grasp_target_false")
        segment = _segment_objects(task_entry)
        expected_segment: tuple[Any, ...]
        if phase == "pickup_tool":
            expected_segment = (correct_tool,)
        elif phase in ("peg_push", "gripper_push"):
            expected_segment = (
                cube,
                _require_attr(self.raw_env, "goal_site"),
            )
        elif phase == "pickup_cube":
            expected_segment = (cube,)
        else:
            expected_segment = (_require_attr(self.raw_env, "goal_site"),)
        if len(segment) != len(expected_segment) or any(
            observed is not expected
            for observed, expected in zip(segment, expected_segment)
        ):
            raise AdapterContractError(
                f"MoveCube {phase} segment does not match runtime actor identities"
            )
        expected_actor = None
        expected_target = None
        expected_actor_names: tuple[str, ...] = ()
        forbidden_actor_names: tuple[str, ...] = ()
        if phase == "pickup_tool":
            expected_actor = correct_tool_name
            expected_actor_names = (correct_tool_name,)
            forbidden_actor_names = (wrong_tool_name, cube_name)
        elif phase == "peg_push":
            expected_actor = correct_tool_name
            expected_actor_names = (correct_tool_name,)
            expected_target = "goal"
            forbidden_actor_names = (wrong_tool_name, cube_name)
        elif phase == "gripper_push":
            expected_target = "goal"
            forbidden_actor_names = (cube_name, correct_tool_name, wrong_tool_name)
        elif phase == "pickup_cube":
            expected_actor = cube_name
            expected_actor_names = (cube_name,)
            forbidden_actor_names = (correct_tool_name, wrong_tool_name)
        elif phase == "place_cube":
            expected_actor = cube_name
            expected_actor_names = (cube_name,)
            expected_target = "goal"
            forbidden_actor_names = (correct_tool_name, wrong_tool_name)
        goal = _require_attr(self.raw_env, "goal_site")
        goal_position = _actor_position(goal, "raw_env.goal_site")
        cube_position = actors[cube_name].position
        metrics = {
            "manner": way,
            "expected_actor_names": expected_actor_names,
            "forbidden_actor_names": forbidden_actor_names,
            "correct_tool_actor": correct_tool_name,
            "wrong_tool_actor": wrong_tool_name,
            "correct_tool_grasped": correct_tool_name in grasped_names,
            "wrong_tool_grasped": wrong_tool_name in grasped_names,
            "cube_grasped": cube_name in grasped_names,
            "cube_actor": cube_name,
            "peg_actors": (correct_tool_name, wrong_tool_name),
            "cube_goal_distance": float(np.linalg.norm(cube_position - goal_position)),
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "native_peg_pickup_accepts_either_end": way == "peg_push",
            "native_peg_push_failure_func_missing": phase == "peg_push",
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        return self._scene(
            task_index,
            task_entry,
            phase,
            expected_actor,
            expected_target,
            actors,
            {"goal": goal_position},
            held_actor,
            metrics,
        )


class RouteStickTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "RouteStick"

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        task_list = _require_sequence(self.raw_env.task_list, "raw_env.task_list")
        policy_ordinal = sum(
            1
            for entry in task_list[:task_index]
            if isinstance(entry, Mapping) and entry.get("demonstration") is False
        )
        selected_buttons = _require_sequence(
            _require_attr(self.raw_env, "selected_buttons"),
            "raw_env.selected_buttons",
        )
        directions = _require_sequence(
            _require_attr(self.raw_env, "swing_directions"),
            "raw_env.swing_directions",
        )
        if len(selected_buttons) < 2 or len(directions) != len(selected_buttons) - 1:
            raise AdapterContractError(
                "RouteStick requires len(swing_directions) == "
                "len(selected_buttons) - 1"
            )
        if policy_ordinal >= len(directions):
            raise AdapterContractError(
                f"RouteStick policy ordinal {policy_ordinal} exceeds route length"
            )
        buttons_grid = _require_sequence(
            _require_attr(self.raw_env, "buttons_grid"),
            "raw_env.buttons_grid",
        )
        previous_button = selected_buttons[policy_ordinal]
        expected_button = selected_buttons[policy_ordinal + 1]
        previous_grid_index = _actor_index(
            previous_button,
            buttons_grid,
            "RouteStick previous waypoint",
        )
        expected_grid_index = _actor_index(
            expected_button,
            buttons_grid,
            "RouteStick expected waypoint",
        )
        expected_direction = directions[policy_ordinal]
        if expected_direction not in ("clockwise", "counterclockwise"):
            raise AdapterContractError(
                f"unsupported RouteStick direction {expected_direction!r}"
            )
        landmarks = {
            f"grid_waypoint_{index}": _actor_position(
                button,
                f"raw_env.buttons_grid[{index}]",
            )
            for index, button in enumerate(buttons_grid)
        }
        previous_name = f"grid_waypoint_{previous_grid_index}"
        expected_name = f"grid_waypoint_{expected_grid_index}"
        forbidden_names = tuple(
            name
            for name in landmarks
            if name not in (previous_name, expected_name)
        )
        target_cubes = _require_mapping(
            _require_attr(self.raw_env, "target_cubes"),
            "raw_env.target_cubes",
        )
        obstacle_actors = list(target_cubes.values())
        actors, grasped_names, held_actor = self._actor_states(
            [(actor, "route_obstacle") for actor in obstacle_actors],
            non_graspable_identities=frozenset(
                id(actor) for actor in obstacle_actors
            ),
        )
        tcp_position = self._tcp_position()
        metrics = {
            "policy_ordinal": policy_ordinal,
            "expected_direction": expected_direction,
            "expected_waypoint_names": (expected_name,),
            "forbidden_waypoint_names": forbidden_names,
            "corridor_start": previous_name,
            "native_route_control_lateral_offset": 0.2,
            "native_route_peak_lateral_offset": 0.5 * 0.2,
            "official_wrong_waypoint_xy_radius": 0.01,
            "official_route_z_threshold": 0.1,
            "grasped_actor_names": grasped_names,
            "native_direction_failure_only_sets_episode_failure": True,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if not isinstance(choice_label, str) or not choice_label:
            raise AdapterContractError("RouteStick choice_label must be non-empty")
        return self._scene(
            task_index,
            task_entry,
            "route_segment",
            None,
            expected_name,
            actors,
            landmarks,
            held_actor,
            metrics,
            points={"stick_tip": tcp_position},
        )


class StopCubeTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "StopCube"

    _PHASES = {
        "move to the top of the button to prepare": "button_ready",
        "remain static": "timed_hold",
        "press button to stop the cube": "timed_press",
    }

    def __init__(
        self,
        raw_env: Any,
        action_predictor: ActionPredictor,
        step_duration_seconds: float,
        timing_metrics_provider: TimingMetricsProvider | None = None,
    ) -> None:
        super().__init__(raw_env, action_predictor, step_duration_seconds)
        if timing_metrics_provider is not None and not callable(timing_metrics_provider):
            raise AdapterContractError("timing_metrics_provider must be callable")
        self._timing_metrics_provider = timing_metrics_provider

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label not in self._PHASES:
            raise AdapterContractError(
                f"unsupported StopCube choice_label {choice_label!r}"
            )
        phase = self._PHASES[choice_label]
        cube = _require_attr(self.raw_env, "cube")
        actors, grasped_names, held_actor = self._actor_states([(cube, "cube")])
        step = self._step()
        steps_press = _scalar_float(
            _require_attr(self.raw_env, "steps_press"),
            "raw_env.steps_press",
        )
        move_interval = _scalar_int(
            _require_attr(self.raw_env, "move_interval"),
            "raw_env.move_interval",
        )
        stop_time = _scalar_int(
            _require_attr(self.raw_env, "stop_time"),
            "raw_env.stop_time",
        )
        stop_range = _require_sequence(
            _require_attr(self.raw_env, "stop_time_range"),
            "raw_env.stop_time_range",
        )
        if len(stop_range) != 2:
            raise AdapterContractError("StopCube stop_time_range must contain two steps")
        stop_range_steps = tuple(
            _scalar_int(value, f"stop_time_range[{index}]")
            for index, value in enumerate(stop_range)
        )
        metrics: dict[str, Any] = {
            "steps_to_nominal_press": steps_press - step,
            "seconds_to_event": (steps_press - step) * self._step_duration_seconds,
            "move_interval_steps": move_interval,
            "stop_time": stop_time,
            "stop_time_range_steps": stop_range_steps,
            "stop_latched": _scalar_bool(
                _require_attr(self.raw_env, "stop"),
                "raw_env.stop",
            ),
            "timing_detector_ready": False,
            "timing_diagnostic": "missing_calibrated_press_progress_provider",
            "grasped_actor_names": grasped_names,
            "native_failure_func_missing": True,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        if self._timing_metrics_provider is not None:
            provided = self._timing_metrics_provider(self.raw_env, task_entry)
            provided = _require_mapping(provided, "timing_metrics_provider result")
            if "press_progress" not in provided:
                raise AdapterContractError(
                    "timing_metrics_provider must provide press_progress"
                )
            press_progress = _scalar_float(
                provided["press_progress"],
                "timing_metrics_provider.press_progress",
            )
            metrics.update(provided)
            metrics["press_progress"] = press_progress
            metrics["timing_detector_ready"] = True
            metrics.pop("timing_diagnostic", None)
        button = _require_attr(self.raw_env, "button")
        target = _require_attr(self.raw_env, "target")
        return self._scene(
            task_index,
            task_entry,
            phase,
            None,
            "button",
            actors,
            {
                "button": _actor_position(button, "raw_env.button"),
                "cube_target": _actor_position(target, "raw_env.target"),
            },
            held_actor,
            metrics,
        )


class SwingXtimesTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "SwingXtimes"

    _PHASES = {
        "pick up the cube": "pickup",
        "move to the top of the target": "swing",
        "put the cube on the table": "putdown",
        "press the button": "press_button",
    }

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label not in self._PHASES:
            raise AdapterContractError(
                f"unsupported SwingXtimes choice_label {choice_label!r}"
            )
        phase = self._PHASES[choice_label]
        all_cubes = _require_sequence(
            _require_attr(self.raw_env, "all_cubes"),
            "raw_env.all_cubes",
        )
        target_cube = _require_attr(self.raw_env, "target_cube")
        if not _actor_in(target_cube, all_cubes):
            raise AdapterContractError("SwingXtimes target_cube is not in all_cubes")
        non_target_cubes = _require_sequence(
            _require_attr(self.raw_env, "non_target_cubes"),
            "raw_env.non_target_cubes",
        )
        actors, grasped_names, held_actor = self._actor_states(
            [(cube, "cube") for cube in all_cubes]
        )
        target_cube_name = _actor_name(target_cube, "raw_env.target_cube")
        non_target_names = tuple(
            _actor_name(cube, "non_target_cube") for cube in non_target_cubes
        )
        right = _require_attr(self.raw_env, "target_right")
        left = _require_attr(self.raw_env, "target_left")
        button = _require_attr(self.raw_env, "button")
        cap_links = _require_sequence(
            _require_attr(self.raw_env, "cap_link"),
            "raw_env.cap_link",
        )
        if len(cap_links) != 1:
            raise AdapterContractError("SwingXtimes cap_link must contain one link")
        button_cap_position = _actor_position(
            cap_links[0],
            "raw_env.cap_link[0]",
        )
        button_depth = -_scalar_float(
            _require_attr(button, "get_qpos")(),
            "raw_env.button.get_qpos()",
        )
        cube_half_size = _scalar_float(
            _require_attr(self.raw_env, "cube_half_size"),
            "raw_env.cube_half_size",
        )
        native_button_scale = 1.5
        native_cap_radius = 0.015 * native_button_scale
        native_cap_half_length = 0.006 * native_button_scale
        landmarks = {
            "target_right": _actor_position(right, "raw_env.target_right"),
            "target_left": _actor_position(left, "raw_env.target_left"),
            "button": _actor_position(button, "raw_env.button"),
            "button_cap": button_cap_position,
        }
        expected_target = None
        if phase == "swing":
            segment = _segment_objects(task_entry)
            if len(segment) != 1:
                raise AdapterContractError(
                    "SwingXtimes swing segment must contain one target"
                )
            if segment[0] is right:
                expected_target = "target_right"
            elif segment[0] is left:
                expected_target = "target_left"
            else:
                raise AdapterContractError(
                    "SwingXtimes swing segment is neither target_right nor target_left"
                )
        elif phase == "press_button":
            expected_target = "button"
        expected_actor = None if phase == "press_button" else target_cube_name
        swing_count = _scalar_int(
            _require_attr(self.raw_env, "swing_count"),
            "raw_env.swing_count",
        )
        maximum_swings = _scalar_int(
            _require_attr(self.raw_env, "max_swings"),
            "raw_env.max_swings",
        )
        metrics = {
            "expected_actor_names": () if expected_actor is None else (expected_actor,),
            "forbidden_actor_names": non_target_names,
            "expected_target_names": () if expected_target is None else (expected_target,),
            "target_cube_grasped": target_cube_name in grasped_names,
            "swing_count": swing_count,
            "maximum_swings": maximum_swings,
            "swing_sequence_complete": swing_count >= maximum_swings,
            "button_depth": button_depth,
            "official_button_depth_threshold": 0.005,
            "button_contact_xy_radius": native_cap_radius + cube_half_size,
            "button_contact_center_height": (
                button_cap_position[2]
                + 2.0 * native_cap_half_length
                + cube_half_size
                + cube_half_size
            ),
            "swing_over_limit": _scalar_bool(
                _require_attr(self.raw_env, "swing_over_limit"),
                "raw_env.swing_over_limit",
            ),
            "on_right_latched": _scalar_bool(
                _require_attr(self.raw_env, "_was_on_right"),
                "raw_env._was_on_right",
            ),
            "on_left_latched": _scalar_bool(
                _require_attr(self.raw_env, "_was_on_left"),
                "raw_env._was_on_left",
            ),
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "native_swing_milestone_requires_held_actor": False,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        return self._scene(
            task_index,
            task_entry,
            phase,
            expected_actor,
            expected_target,
            actors,
            landmarks,
            held_actor,
            metrics,
        )


class _ContainerSequenceTaskAdapter(RoboMMETaskAdapter):
    """Shared strict mapping for Button/Video unmask container stages."""

    button_choice_labels: frozenset[str] = frozenset()

    def _selected_containers(self) -> list[Any]:
        raise NotImplementedError

    def _button_definitions(self) -> Mapping[str, tuple[Any, Sequence[Any]]]:
        return {}

    def _allowed_button_actors(self) -> list[Any]:
        return [definition[0] for definition in self._button_definitions().values()]

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label in self.button_choice_labels:
            phase = "press_button"
        elif choice_label == "pick up the container":
            phase = "pickup_container"
        elif choice_label == "put down the container":
            phase = "putdown_container"
        else:
            raise AdapterContractError(
                f"unsupported {self.task_name} choice_label {choice_label!r}"
            )

        spawned_bins = _unique_identity_sequence(
            _require_attr(self.raw_env, "spawned_bins"),
            "raw_env.spawned_bins",
        )
        if not spawned_bins:
            raise AdapterContractError(f"{self.task_name} has no spawned containers")
        selected_containers = _unique_identity_sequence(
            self._selected_containers(),
            f"{self.task_name}.selected_containers",
        )
        if not selected_containers or any(
            not _actor_in(container, spawned_bins)
            for container in selected_containers
        ):
            raise AdapterContractError(
                f"{self.task_name} selected containers must be spawned containers"
            )
        actors, grasped_names, held_actor = self._actor_states(
            [(container, "container") for container in spawned_bins]
        )
        selected_names = tuple(
            _actor_name(container, "selected_container")
            for container in selected_containers
        )

        button_definitions = dict(self._button_definitions())
        button_actor_ids: set[int] = set()
        landmarks: dict[str, np.ndarray] = {}
        segment_button_name = None
        for button_name, definition in button_definitions.items():
            if not isinstance(button_name, str) or not button_name:
                raise AdapterContractError("button semantic names must be non-empty")
            if len(definition) != 2:
                raise AdapterContractError(
                    f"button definition {button_name!r} must contain actor and links"
                )
            button_actor, button_links = definition
            if id(button_actor) in button_actor_ids:
                raise AdapterContractError("button actors must be unique by identity")
            button_actor_ids.add(id(button_actor))
            links = _require_sequence(
                button_links,
                f"button_definitions[{button_name!r}].links",
            )
            if not links:
                raise AdapterContractError(
                    f"button definition {button_name!r} has no cap link"
                )
            landmarks[button_name] = _actor_position(
                button_actor,
                f"button_definitions[{button_name!r}].actor",
            )
            observed_segment = _segment_objects(task_entry)
            if (
                phase == "press_button"
                and len(observed_segment) == len(links)
                and all(
                    observed is expected
                    for observed, expected in zip(observed_segment, links)
                )
            ):
                segment_button_name = button_name

        expected_actor = None
        expected_target = None
        expected_actor_names: tuple[str, ...] = ()
        forbidden_actor_names: tuple[str, ...]
        pickup_ordinal = sum(
            1
            for entry in _policy_entries_before(self.raw_env.task_list, task_index)
            if entry.get("choice_label") == "pick up the container"
        )
        if phase == "pickup_container":
            expected_container = _require_one_segment_actor(
                task_entry,
                selected_containers,
                f"{self.task_name} pickup segment",
            )
            expected_actor = _actor_name(expected_container, "pickup segment")
            expected_actor_names = (expected_actor,)
            forbidden_actor_names = tuple(
                _actor_name(container, "forbidden_container")
                for container in spawned_bins
                if container is not expected_container
            )
        elif phase == "putdown_container":
            if task_index == 0:
                raise AdapterContractError(
                    f"{self.task_name} putdown has no preceding pickup"
                )
            preceding_entry = _require_mapping(
                self.raw_env.task_list[task_index - 1],
                f"{self.task_name} preceding pickup entry",
            )
            if preceding_entry.get("choice_label") != "pick up the container":
                raise AdapterContractError(
                    f"{self.task_name} putdown must immediately follow pickup"
                )
            expected_container = _require_one_segment_actor(
                preceding_entry,
                selected_containers,
                f"{self.task_name} preceding pickup segment",
            )
            expected_actor = _actor_name(expected_container, "preceding pickup")
            expected_actor_names = (expected_actor,)
            forbidden_actor_names = tuple(
                _actor_name(container, "forbidden_container")
                for container in spawned_bins
                if container is not expected_container
            )
        else:
            if not button_definitions or segment_button_name is None:
                raise AdapterContractError(
                    f"{self.task_name} button segment must exactly match a cap link"
                )
            allowed_button_actors = _unique_identity_sequence(
                self._allowed_button_actors(),
                f"{self.task_name}.allowed_button_actors",
            )
            allowed_button_names = tuple(
                name
                for name, (actor, _) in button_definitions.items()
                if _actor_in(actor, allowed_button_actors)
            )
            if len(allowed_button_names) != len(allowed_button_actors):
                raise AdapterContractError(
                    f"{self.task_name} allowed button is not a known button actor"
                )
            if not allowed_button_names:
                raise AdapterContractError(
                    f"{self.task_name} has no semantically valid remaining button"
                )
            expected_target = (
                allowed_button_names[0]
                if len(allowed_button_names) == 1
                else None
            )
            expected_actor_names = ()
            forbidden_actor_names = tuple(actors)

        remaining_button_names = tuple(
            name
            for name, (actor, _) in button_definitions.items()
            if _actor_in(actor, self._allowed_button_actors())
        )
        allowed_button_names = (
            (segment_button_name,)
            if phase == "press_button" and segment_button_name is not None
            else remaining_button_names
        )
        metrics = {
            "expected_actor_names": expected_actor_names,
            "forbidden_actor_names": forbidden_actor_names,
            "selected_container_names": selected_names,
            "pickup_ordinal": pickup_ordinal,
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "allowed_button_names": allowed_button_names,
            "remaining_button_names": remaining_button_names,
            "task_solver_button_name": segment_button_name,
            "task_solver_semantically_valid": (
                segment_button_name in allowed_button_names
                if phase == "press_button"
                else True
            ),
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        return self._scene(
            task_index,
            task_entry,
            phase,
            expected_actor,
            expected_target,
            actors,
            landmarks,
            held_actor,
            metrics,
        )


class ButtonUnmaskTaskAdapter(_ContainerSequenceTaskAdapter):
    canonical_task_name = "ButtonUnmask"
    button_choice_labels = frozenset({"press the button"})

    def _selected_containers(self) -> list[Any]:
        selected = [_require_attr(self.raw_env, "bin_0")]
        if hasattr(self.raw_env, "bin_1"):
            selected.append(self.raw_env.bin_1)
        return selected

    def _button_definitions(self) -> Mapping[str, tuple[Any, Sequence[Any]]]:
        return {
            "button_left": (
                _require_attr(self.raw_env, "button_left"),
                _require_attr(self.raw_env, "cap_link"),
            )
        }


class ButtonUnmaskSwapTaskAdapter(_ContainerSequenceTaskAdapter):
    canonical_task_name = "ButtonUnmaskSwap"
    button_choice_labels = frozenset(
        {"press the first button", "press the second button"}
    )

    def _selected_containers(self) -> list[Any]:
        return _require_sequence(
            _require_attr(self.raw_env, "selected_bins"),
            "raw_env.selected_bins",
        )

    def _button_definitions(self) -> Mapping[str, tuple[Any, Sequence[Any]]]:
        cap_links = _require_mapping(
            _require_attr(self.raw_env, "cap_links"),
            "raw_env.cap_links",
        )
        return {
            "button_left": (
                _require_attr(self.raw_env, "button_left"),
                _require_mapping_value(cap_links, "button_left"),
            ),
            "button_right": (
                _require_attr(self.raw_env, "button_right"),
                _require_mapping_value(cap_links, "button_right"),
            ),
        }

    def _allowed_button_actors(self) -> list[Any]:
        return _require_sequence(
            _require_attr(self.raw_env, "button_list"),
            "raw_env.button_list",
        )


class VideoUnmaskTaskAdapter(_ContainerSequenceTaskAdapter):
    canonical_task_name = "VideoUnmask"

    def _selected_containers(self) -> list[Any]:
        selected = [_require_attr(self.raw_env, "bin_0")]
        if hasattr(self.raw_env, "bin_1"):
            selected.append(self.raw_env.bin_1)
        return selected


class VideoUnmaskSwapTaskAdapter(_ContainerSequenceTaskAdapter):
    canonical_task_name = "VideoUnmaskSwap"

    def _selected_containers(self) -> list[Any]:
        return _require_sequence(
            _require_attr(self.raw_env, "selected_bins"),
            "raw_env.selected_bins",
        )


class InsertPegTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "InsertPeg"

    _PICKUP_LABEL = "pick up the peg by grasping one end"

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        insert_way = _require_attr(self.raw_env, "insert_way")
        if choice_label == self._PICKUP_LABEL:
            phase = "pickup_peg_end"
        elif choice_label == f"insert the peg from the {insert_way} side":
            phase = "insert_peg"
        else:
            raise AdapterContractError(
                f"unsupported InsertPeg choice_label {choice_label!r}"
            )

        pegs = _unique_identity_sequence(
            _require_attr(self.raw_env, "pegs"),
            "raw_env.pegs",
        )
        peg_heads = _unique_identity_sequence(
            _require_attr(self.raw_env, "peg_heads"),
            "raw_env.peg_heads",
        )
        peg_tails = _unique_identity_sequence(
            _require_attr(self.raw_env, "peg_tails"),
            "raw_env.peg_tails",
        )
        peg_init_poses = _require_sequence(
            _require_attr(self.raw_env, "peg_init_poses"),
            "raw_env.peg_init_poses",
        )
        if not pegs or not (
            len(pegs) == len(peg_heads) == len(peg_tails) == len(peg_init_poses)
        ):
            raise AdapterContractError(
                "InsertPeg pegs, endpoints, and reset poses must have equal length"
            )
        selected_peg = _require_attr(self.raw_env, "peg")
        selected_index = _actor_index(selected_peg, pegs, "InsertPeg selected peg")
        grasp_target = _require_attr(self.raw_env, "grasp_target")
        insert_target = _require_attr(self.raw_env, "insert_target")
        selected_head = peg_heads[selected_index]
        selected_tail = peg_tails[selected_index]
        if {id(grasp_target), id(insert_target)} != {
            id(selected_head),
            id(selected_tail),
        }:
            raise AdapterContractError(
                "InsertPeg grasp/insert endpoints must be opposite ends of selected peg"
            )
        obj_flag = _scalar_int(
            _require_attr(self.raw_env, "obj_flag"),
            "raw_env.obj_flag",
        )
        if obj_flag not in (-1, 1):
            raise AdapterContractError("InsertPeg obj_flag must be -1 or 1")
        expected_grasp = selected_head if obj_flag == -1 else selected_tail
        expected_insert = selected_tail if obj_flag == -1 else selected_head
        if grasp_target is not expected_grasp or insert_target is not expected_insert:
            raise AdapterContractError(
                "InsertPeg endpoint identities disagree with obj_flag"
            )
        direction = _scalar_int(
            _require_attr(self.raw_env, "direction"),
            "raw_env.direction",
        )
        if direction not in (-1, 1):
            raise AdapterContractError("InsertPeg direction must be -1 or 1")
        expected_way = "left" if direction == -1 else "right"
        if insert_way != expected_way:
            raise AdapterContractError(
                "InsertPeg insert_way disagrees with native direction sign"
            )

        endpoint_actors = [*peg_heads, *peg_tails]
        endpoint_names_by_identity = {
            id(endpoint): f"peg_{peg_index}_{endpoint_kind}"
            for endpoint_kind, endpoints in (
                ("head", peg_heads),
                ("tail", peg_tails),
            )
            for peg_index, endpoint in enumerate(endpoints)
        }
        actors, grasped_names, held_actor = self._actor_states(
            [(endpoint, "peg_end") for endpoint in endpoint_actors],
            semantic_names_by_identity=endpoint_names_by_identity,
        )
        grasp_name = endpoint_names_by_identity[id(grasp_target)]
        insert_name = endpoint_names_by_identity[id(insert_target)]
        box = _require_attr(self.raw_env, "box")
        box_position = _actor_position(box, "raw_env.box")
        if phase == "pickup_peg_end":
            _require_exact_segment(
                task_entry,
                [grasp_target],
                "InsertPeg pickup segment",
            )
            expected_actor = grasp_name
            expected_target = None
        else:
            _require_exact_segment(task_entry, [box], "InsertPeg insertion segment")
            expected_actor = insert_name
            expected_target = "box"

        forbidden_names = tuple(
            endpoint_names_by_identity[id(endpoint)]
            for endpoint in endpoint_actors
            if endpoint is not grasp_target
        )
        insert_position = actors[insert_name].position
        grasp_position = actors[grasp_name].position
        tcp_position = self._tcp_position()
        side_indicator = float(tcp_position[1] - box_position[1])
        metrics = {
            "expected_actor_names": (expected_actor,),
            "forbidden_actor_names": forbidden_names,
            "selected_peg_index": selected_index,
            "grasp_target_actor": grasp_name,
            "insert_target_actor": insert_name,
            "opposite_end_actor": grasp_name,
            "obj_flag": obj_flag,
            "insert_direction": direction,
            "insert_way": insert_way,
            "insert_side_axis": 1,
            "native_correct_side_condition": "side_indicator * direction < 0",
            "insert_side_indicator": side_indicator,
            "insert_side_product": side_indicator * direction,
            "insert_target_box_distance": float(
                np.linalg.norm(insert_position - box_position)
            ),
            "opposite_end_box_distance": float(
                np.linalg.norm(grasp_position - box_position)
            ),
            "peg_axis_vector": insert_position - grasp_position,
            "selected_peg_endpoint_separation": float(
                np.linalg.norm(insert_position - grasp_position)
            ),
            "grasp_target_grasped": grasp_name in grasped_names,
            "insert_target_grasped": insert_name in grasped_names,
            "selected_peg_grasped": (
                grasp_name in grasped_names or insert_name in grasped_names
            ),
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "reset_pose_available": True,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        return self._scene(
            task_index,
            task_entry,
            phase,
            expected_actor,
            expected_target,
            actors,
            {"box": box_position},
            held_actor,
            metrics,
        )


class PatternLockTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "PatternLock"

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if not isinstance(choice_label, str) or not choice_label.startswith("move "):
            raise AdapterContractError(
                f"unsupported PatternLock choice_label {choice_label!r}"
            )
        if not _scalar_bool(
            _require_attr(self.raw_env, "after_demo"),
            "raw_env.after_demo",
        ):
            raise AdapterContractError(
                "PatternLock policy task is active before after_demo is latched"
            )
        task_list = _require_sequence(self.raw_env.task_list, "raw_env.task_list")
        policy_ordinal = len(_policy_entries_before(task_list, task_index))
        selected_buttons = _unique_identity_sequence(
            _require_attr(self.raw_env, "selected_buttons"),
            "raw_env.selected_buttons",
        )
        if len(selected_buttons) < 2 or policy_ordinal >= len(selected_buttons) - 1:
            raise AdapterContractError(
                "PatternLock policy ordinal is outside selected path"
            )
        buttons_grid = _unique_identity_sequence(
            _require_attr(self.raw_env, "buttons_grid"),
            "raw_env.buttons_grid",
        )
        previous_button = selected_buttons[policy_ordinal]
        expected_button = selected_buttons[policy_ordinal + 1]
        previous_grid_index = _actor_index(
            previous_button,
            buttons_grid,
            "PatternLock previous waypoint",
        )
        expected_grid_index = _actor_index(
            expected_button,
            buttons_grid,
            "PatternLock expected waypoint",
        )
        landmarks = {
            f"grid_waypoint_{index}": _actor_position(
                button,
                f"raw_env.buttons_grid[{index}]",
            )
            for index, button in enumerate(buttons_grid)
        }
        previous_name = f"grid_waypoint_{previous_grid_index}"
        expected_name = f"grid_waypoint_{expected_grid_index}"
        achieved_list = _require_sequence(
            _require_attr(self.raw_env, "achieved_list"),
            "raw_env.achieved_list",
        )
        if any(not _actor_in(button, buttons_grid) for button in achieved_list):
            raise AdapterContractError(
                "PatternLock achieved_list contains an unknown waypoint actor"
            )
        forbidden_names = tuple(
            name
            for name in landmarks
            if name not in (previous_name, expected_name)
        )
        metrics = {
            "policy_ordinal": policy_ordinal,
            "expected_waypoint_names": (expected_name,),
            "forbidden_waypoint_names": forbidden_names,
            "corridor_start": previous_name,
            "achieved_waypoint_names": tuple(
                f"grid_waypoint_{_actor_index(button, buttons_grid, 'achieved')}"
                for button in achieved_list
            ),
            "sequence_match_latched": _scalar_bool(
                _require_attr(self.raw_env, "match"),
                "raw_env.match",
            ),
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        tcp_position = self._tcp_position()
        return self._scene(
            task_index,
            task_entry,
            "pattern_segment",
            None,
            expected_name,
            {},
            landmarks,
            None,
            metrics,
            points={"swing_tip": tcp_position},
        )


class PickXtimesTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "PickXtimes"

    _PHASES = {
        "pick up the cube": "pickup",
        "place the cube onto the target": "place",
        "press the button to stop": "press_button",
    }

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label not in self._PHASES:
            raise AdapterContractError(
                f"unsupported PickXtimes choice_label {choice_label!r}"
            )
        phase = self._PHASES[choice_label]
        all_cubes = _unique_identity_sequence(
            _require_attr(self.raw_env, "all_cubes"),
            "raw_env.all_cubes",
        )
        target_cube = _require_attr(self.raw_env, "target_cube")
        target_cube_index = _actor_index(
            target_cube,
            all_cubes,
            "PickXtimes target_cube",
        )
        non_target_cubes = _unique_identity_sequence(
            _require_attr(self.raw_env, "non_target_cubes"),
            "raw_env.non_target_cubes",
        )
        expected_non_targets = _identity_complement(all_cubes, [target_cube])
        if {id(actor) for actor in non_target_cubes} != {
            id(actor) for actor in expected_non_targets
        }:
            raise AdapterContractError(
                "PickXtimes non_target_cubes is not the identity complement"
            )
        actors, grasped_names, held_actor = self._actor_states(
            [(cube, "cube") for cube in all_cubes]
        )
        target_cube_name = _actor_name(target_cube, "raw_env.target_cube")
        target = _require_attr(self.raw_env, "target")
        button = _require_attr(self.raw_env, "button")
        landmarks = {
            "place_target": _actor_position(target, "raw_env.target"),
            "button": _actor_position(button, "raw_env.button"),
        }
        if phase == "pickup":
            _require_exact_segment(
                task_entry,
                [target_cube],
                "PickXtimes pickup segment",
            )
        elif phase == "place":
            _require_exact_segment(
                task_entry,
                [target],
                "PickXtimes place segment",
            )
        else:
            _require_exact_segment(
                task_entry,
                _require_sequence(
                    _require_attr(self.raw_env, "cap_link"),
                    "raw_env.cap_link",
                ),
                "PickXtimes button segment",
            )
        repeat_count = _scalar_int(
            _require_attr(self.raw_env, "num_repeats"),
            "raw_env.num_repeats",
        )
        if repeat_count < 1:
            raise AdapterContractError("PickXtimes num_repeats must be positive")
        entries_before = _policy_entries_before(self.raw_env.task_list, task_index)
        pickups_before = sum(
            entry.get("choice_label") == "pick up the cube"
            for entry in entries_before
        )
        places_before = sum(
            entry.get("choice_label") == "place the cube onto the target"
            for entry in entries_before
        )
        if pickups_before > repeat_count or places_before > repeat_count:
            raise AdapterContractError("PickXtimes cycle ordinal exceeds repeat count")
        cycle_index = pickups_before if phase == "pickup" else places_before
        expected_actor = None if phase == "press_button" else target_cube_name
        expected_target = None
        if phase == "place":
            expected_target = "place_target"
        elif phase == "press_button":
            expected_target = "button"
        metrics = {
            "expected_actor_names": (
                () if expected_actor is None else (expected_actor,)
            ),
            "forbidden_actor_names": tuple(
                _actor_name(cube, "non_target_cube")
                for cube in non_target_cubes
            ),
            "expected_target_names": (
                () if expected_target is None else (expected_target,)
            ),
            "forbidden_target_names": (),
            "target_cube_index": target_cube_index,
            "repeat_count": repeat_count,
            "cycle_index": cycle_index,
            "completed_cycles": places_before,
            "cycles_complete": places_before >= repeat_count,
            "button_prerequisite_complete": places_before >= repeat_count,
            "extra_pickup_forbidden": phase == "press_button",
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        return self._scene(
            task_index,
            task_entry,
            phase,
            expected_actor,
            expected_target,
            actors,
            landmarks,
            held_actor,
            metrics,
        )


class VideoRepickTaskAdapter(RoboMMETaskAdapter):
    canonical_task_name = "VideoRepick"

    _PHASES = {
        "pick up the cube": "pickup",
        "put it down": "putdown",
        "press the button to finish": "press_button",
    }

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label not in self._PHASES:
            raise AdapterContractError(
                f"unsupported VideoRepick choice_label {choice_label!r}"
            )
        phase = self._PHASES[choice_label]
        spawned_cubes = _unique_identity_sequence(
            _require_attr(self.raw_env, "spawned_cubes"),
            "raw_env.spawned_cubes",
        )
        target_cube = _require_attr(self.raw_env, "target_cube_1")
        _actor_index(target_cube, spawned_cubes, "VideoRepick target_cube_1")
        actors, grasped_names, held_actor = self._actor_states(
            [(cube, "cube") for cube in spawned_cubes]
        )
        target_cube_name = _actor_name(target_cube, "raw_env.target_cube_1")
        if phase == "pickup":
            _require_exact_segment(
                task_entry,
                [target_cube],
                "VideoRepick pickup segment",
            )
        elif phase == "putdown":
            if _segment_objects(task_entry):
                raise AdapterContractError(
                    "VideoRepick putdown entry must not invent a target segment"
                )
        else:
            _require_exact_segment(
                task_entry,
                _require_sequence(
                    _require_attr(self.raw_env, "cap_link"),
                    "raw_env.cap_link",
                ),
                "VideoRepick button segment",
            )
        repeat_count = _scalar_int(
            _require_attr(self.raw_env, "num_repeats"),
            "raw_env.num_repeats",
        )
        if repeat_count < 1:
            raise AdapterContractError("VideoRepick num_repeats must be positive")
        entries_before = _policy_entries_before(self.raw_env.task_list, task_index)
        pickups_before = sum(
            entry.get("choice_label") == "pick up the cube"
            for entry in entries_before
        )
        putdowns_before = sum(
            entry.get("choice_label") == "put it down"
            for entry in entries_before
        )
        if pickups_before > repeat_count or putdowns_before > repeat_count:
            raise AdapterContractError("VideoRepick cycle ordinal exceeds repeat count")
        cycle_index = pickups_before if phase == "pickup" else putdowns_before
        button = _require_attr(self.raw_env, "button_left")
        expected_actor = None if phase == "press_button" else target_cube_name
        expected_target = "button" if phase == "press_button" else None
        non_target_names = tuple(
            _actor_name(cube, "non_target_cube")
            for cube in spawned_cubes
            if cube is not target_cube
        )
        metrics = {
            "expected_actor_names": (
                () if expected_actor is None else (expected_actor,)
            ),
            "forbidden_actor_names": non_target_names,
            "all_pickup_actor_names": tuple(actors),
            "repeat_count": repeat_count,
            "cycle_index": cycle_index,
            "completed_cycles": putdowns_before,
            "cycles_complete": putdowns_before >= repeat_count,
            "button_prerequisite_complete": putdowns_before >= repeat_count,
            "extra_pickup_forbidden": phase == "press_button",
            "native_timer_ids_reused_across_cycles": True,
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
        }
        return self._scene(
            task_index,
            task_entry,
            phase,
            expected_actor,
            expected_target,
            actors,
            {"button": _actor_position(button, "raw_env.button_left")},
            held_actor,
            metrics,
        )


class _VideoPlaceTaskAdapter(RoboMMETaskAdapter):
    """Shared strict actor/target mapping for the two video-place tasks."""

    def _video_metrics(
        self,
        target_target: Any,
        targets: Sequence[Any],
    ) -> Mapping[str, Any]:
        return {}

    def read_scene(self) -> SceneState:
        task_index, task_entry, reported_task_index = self._task_entry()
        choice_label = _require_mapping_value(task_entry, "choice_label")
        if choice_label == "pick up the cube":
            phase = "pickup"
        elif choice_label == "drop onto":
            phase = "place"
        else:
            raise AdapterContractError(
                f"unsupported {self.task_name} choice_label {choice_label!r}"
            )
        all_cubes = _unique_identity_sequence(
            _require_attr(self.raw_env, "all_cubes"),
            "raw_env.all_cubes",
        )
        target_cube = _require_attr(self.raw_env, "target_cube")
        _actor_index(target_cube, all_cubes, f"{self.task_name} target_cube")
        non_target_cubes = _unique_identity_sequence(
            _require_attr(self.raw_env, "non_target_cubes"),
            "raw_env.non_target_cubes",
        )
        if {id(actor) for actor in non_target_cubes} != {
            id(actor) for actor in _identity_complement(all_cubes, [target_cube])
        }:
            raise AdapterContractError(
                f"{self.task_name} non_target_cubes is not the identity complement"
            )
        targets = _unique_identity_sequence(
            _require_attr(self.raw_env, "targets"),
            "raw_env.targets",
        )
        target_target = _require_attr(self.raw_env, "target_target")
        target_index = _actor_index(
            target_target,
            targets,
            f"{self.task_name} target_target",
        )
        targets_not_true = _unique_identity_sequence(
            _require_attr(self.raw_env, "targets_not_true"),
            "raw_env.targets_not_true",
        )
        if {id(actor) for actor in targets_not_true} != {
            id(actor) for actor in _identity_complement(targets, [target_target])
        }:
            raise AdapterContractError(
                f"{self.task_name} targets_not_true is not the identity complement"
            )
        actors, grasped_names, held_actor = self._actor_states(
            [(cube, "cube") for cube in all_cubes]
        )
        target_cube_name = _actor_name(target_cube, "raw_env.target_cube")
        landmarks = {
            f"place_target_{index}": _actor_position(
                target,
                f"raw_env.targets[{index}]",
            )
            for index, target in enumerate(targets)
        }
        expected_target_name = f"place_target_{target_index}"
        if phase == "pickup":
            _require_exact_segment(
                task_entry,
                [target_cube],
                f"{self.task_name} pickup segment",
            )
            expected_target = None
        else:
            _require_exact_segment(
                task_entry,
                [target_target],
                f"{self.task_name} place segment",
            )
            expected_target = expected_target_name
        metrics = {
            "expected_actor_names": (target_cube_name,),
            "forbidden_actor_names": tuple(
                _actor_name(cube, "non_target_cube")
                for cube in non_target_cubes
            ),
            "expected_target_names": (
                () if expected_target is None else (expected_target,)
            ),
            "forbidden_target_names": tuple(
                name for name in landmarks if name != expected_target_name
            ),
            "target_target_index": target_index,
            "target_cube_grasped": target_cube_name in grasped_names,
            "grasped_actor_names": grasped_names,
            "grasp_identity_ambiguous": len(grasped_names) > 1,
            "reported_current_task_index": reported_task_index,
            "reported_task_index_lag": (
                None
                if reported_task_index is None
                else task_index - reported_task_index
            ),
            **self._video_metrics(target_target, targets),
        }
        return self._scene(
            task_index,
            task_entry,
            phase,
            target_cube_name,
            expected_target,
            actors,
            landmarks,
            held_actor,
            metrics,
        )


class VideoPlaceButtonTaskAdapter(_VideoPlaceTaskAdapter):
    canonical_task_name = "VideoPlaceButton"

    def _video_metrics(
        self,
        target_target: Any,
        targets: Sequence[Any],
    ) -> Mapping[str, Any]:
        language = _require_attr(self.raw_env, "target_target_language")
        if language not in ("before", "after"):
            raise AdapterContractError(
                "VideoPlaceButton target_target_language must be before or after"
            )
        expected_actor = (
            _require_attr(self.raw_env, "target_0")
            if language == "before"
            else _require_attr(self.raw_env, "target_1")
        )
        if target_target is not expected_actor:
            raise AdapterContractError(
                "VideoPlaceButton target_target identity disagrees with video relation"
            )
        return {
            "video_relation": language,
            "task_flag": _scalar_bool(
                _require_attr(self.raw_env, "task_flag"),
                "raw_env.task_flag",
            ),
            "pre_flag": _scalar_bool(
                _require_attr(self.raw_env, "pre_flag"),
                "raw_env.pre_flag",
            ),
            "post_flag": _scalar_bool(
                _require_attr(self.raw_env, "post_flag"),
                "raw_env.post_flag",
            ),
        }


class VideoPlaceOrderTaskAdapter(_VideoPlaceTaskAdapter):
    canonical_task_name = "VideoPlaceOrder"

    def _video_metrics(
        self,
        target_target: Any,
        targets: Sequence[Any],
    ) -> Mapping[str, Any]:
        ordered_targets = _require_sequence(
            _require_attr(self.raw_env, "which_targets_to_pick"),
            "raw_env.which_targets_to_pick",
        )
        if not ordered_targets or any(
            not _actor_in(target, targets) for target in ordered_targets
        ):
            raise AdapterContractError(
                "VideoPlaceOrder ordered targets must reference runtime targets"
            )
        ordinal = _scalar_int(
            _require_attr(self.raw_env, "which_in_subset"),
            "raw_env.which_in_subset",
        )
        if ordinal < 1 or ordinal > len(ordered_targets):
            raise AdapterContractError(
                "VideoPlaceOrder which_in_subset is outside demonstrated order"
            )
        if target_target is not ordered_targets[ordinal - 1]:
            raise AdapterContractError(
                "VideoPlaceOrder target_target must be selected by demo order identity"
            )
        return {
            "video_target_ordinal_one_based": ordinal,
            "video_order_target_indices": tuple(
                _actor_index(target, targets, "video ordered target")
                for target in ordered_targets
            ),
            "button_task_index": _scalar_int(
                _require_attr(self.raw_env, "button_task_index"),
                "raw_env.button_task_index",
            ),
        }


_ADAPTER_TYPES = {
    "BinFill": BinFillTaskAdapter,
    "ButtonUnmask": ButtonUnmaskTaskAdapter,
    "ButtonUnmaskSwap": ButtonUnmaskSwapTaskAdapter,
    "InsertPeg": InsertPegTaskAdapter,
    "PatternLock": PatternLockTaskAdapter,
    "PickHighlight": PickHighlightTaskAdapter,
    "PickXtimes": PickXtimesTaskAdapter,
    "MoveCube": MoveCubeTaskAdapter,
    "RouteStick": RouteStickTaskAdapter,
    "StopCube": StopCubeTaskAdapter,
    "SwingXtimes": SwingXtimesTaskAdapter,
    "VideoPlaceButton": VideoPlaceButtonTaskAdapter,
    "VideoPlaceOrder": VideoPlaceOrderTaskAdapter,
    "VideoRepick": VideoRepickTaskAdapter,
    "VideoUnmask": VideoUnmaskTaskAdapter,
    "VideoUnmaskSwap": VideoUnmaskSwapTaskAdapter,
}


def build_robomme_task_adapter(
    task_name: str,
    raw_env: Any,
    action_predictor: ActionPredictor,
    step_duration_seconds: float,
    **adapter_options: Any,
) -> RoboMMETaskAdapter:
    """Build one strict adapter without deriving task identity from language."""

    if task_name not in _ADAPTER_TYPES:
        raise AdapterContractError(
            f"no repo-external RoboMME adapter registered for {task_name!r}"
        )
    adapter_type = _ADAPTER_TYPES[task_name]
    return adapter_type(
        raw_env=raw_env,
        action_predictor=action_predictor,
        step_duration_seconds=step_duration_seconds,
        **adapter_options,
    )


def _runtime_scale(raw_env: Any) -> float:
    for field_name in ("cube_half_size", "object_half_size"):
        value = getattr(raw_env, field_name, None)
        if value is not None:
            scale = float(np.asarray(value).reshape(-1)[0])
            if math.isfinite(scale) and scale > 0.0:
                return scale
    return 0.02


def _continue(after: SceneState, reason: str) -> InterventionDecision:
    return InterventionDecision(
        intervene=False,
        reason_code=reason,
        signal_class=None,
        task_name=after.task_name,
        step=after.step,
        subtask_index=after.subtask.index,
        subtask_name=after.subtask.name,
        phase=after.subtask.phase,
        expected_actor=after.subtask.expected_actor,
        observed_actor=after.held_actor,
        metrics={},
        thresholds={},
        recovery_mode=RecoveryMode.NONE,
        diagnostics={"integration": "direct_post_action"},
    )


class TaskPostActionDetector:
    task_name = "RoboMME"

    def __init__(self, raw_env: Any) -> None:
        self.raw_env = raw_env
        self.scale = _runtime_scale(raw_env)
        # This is the same Panda gripper boundary used by the existing native
        # action/recovery path.  A decreasing opening is only "closing"; it is
        # not a missed grasp until the fingers are already closed on a later
        # post-action observation.
        self.closed_gripper_opening = 0.02
        self.pickup_xy_alignment_radius = 2.5 * self.scale
        self.pickup_descent_height = 4.0 * self.scale

    def evaluate(
        self,
        before: SceneState,
        action: np.ndarray,
        after: SceneState,
    ) -> Sequence[CandidateSignal]:
        raise NotImplementedError

    def evaluate_transition(
        self,
        before: SceneState,
        action: np.ndarray,
        after: SceneState,
    ) -> Sequence[CandidateSignal]:
        del before, action, after
        return ()

    def observe(
        self,
        before: SceneState,
        action: np.ndarray,
        after: SceneState,
    ) -> InterventionDecision:
        if before.task_name != self.task_name or after.task_name != self.task_name:
            raise ValueError(f"{self.task_name} detector received another task")
        if after.official_failure or after.terminal:
            return _continue(after, "terminal_owned_by_caller")
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size < 8 or not np.isfinite(action_array).all():
            raise ValueError("policy action must contain eight finite values")
        subtask_advanced = before.subtask.index != after.subtask.index
        candidate_signals = (
            self.evaluate_transition(before, action_array, after)
            if subtask_advanced
            else self.evaluate(before, action_array, after)
        )
        active = [
            signal
            for signal in candidate_signals
            if signal.active and signal.signal_class == SignalClass.SEMANTIC
        ]
        if not active:
            return _continue(
                after,
                "subtask_advanced" if subtask_advanced else "continue",
            )
        selected = max(active, key=lambda signal: (signal.priority, signal.reason_code))
        return InterventionDecision(
            intervene=True,
            reason_code=selected.reason_code,
            signal_class=selected.signal_class,
            task_name=after.task_name,
            step=after.step,
            subtask_index=after.subtask.index,
            subtask_name=after.subtask.name,
            phase=after.subtask.phase,
            expected_actor=selected.expected_actor,
            observed_actor=selected.observed_actor,
            metrics=dict(selected.metrics),
            thresholds=dict(selected.thresholds),
            recovery_mode=selected.recovery_mode,
            diagnostics={
                "integration": "direct_post_action",
                "actual_action": action_array.tolist(),
            },
        )

    def pickup(
        self,
        before: SceneState,
        after: SceneState,
        *,
        intent_radius: float,
        selection_margin: float,
        minimum_approach: float,
        expected_xy_radius: float | None = None,
        expected_descent_height: float | None = None,
    ) -> list[CandidateSignal]:
        expected_actors = tuple(after.metrics.get("expected_actor_names", ()))
        if not expected_actors:
            return []
        if len(expected_actors) != 1:
            raise ValueError(
                f"{self.task_name} must define its own OR over multiple pickup actors"
            )
        expected_actor = expected_actors[0]
        actor = after.actors.get(expected_actor)
        expected_actor_kind = None if actor is None else actor.kind
        return pickup_actor_signals(
            before,
            after,
            expected_actor=expected_actor,
            expected_actor_kind=expected_actor_kind,
            forbidden_actors=tuple(after.metrics.get("forbidden_actor_names", ())),
            intent_radius=intent_radius,
            selection_margin=selection_margin,
            minimum_approach_delta=minimum_approach,
            missed_grasp_radius=intent_radius,
            expected_xy_radius=(
                self.pickup_xy_alignment_radius
                if expected_xy_radius is None
                else expected_xy_radius
            ),
            expected_descent_height=(
                self.pickup_descent_height
                if expected_descent_height is None
                else expected_descent_height
            ),
            closed_gripper_opening=self.closed_gripper_opening,
        )

    def place(
        self,
        before: SceneState,
        after: SceneState,
        *,
        release_radius: float,
        transport_radius: float,
        selection_margin: float,
        minimum_motion: float,
        minimum_improvement: float,
        horizontal_target_distance: bool = False,
    ) -> list[CandidateSignal]:
        return place_signals(
            before,
            after,
            expected_actors=tuple(after.metrics.get("expected_actor_names", ())),
            expected_targets=tuple(after.metrics.get("expected_target_names", ())),
            forbidden_targets=tuple(after.metrics.get("forbidden_target_names", ())),
            release_radius=release_radius,
            transport_radius=transport_radius,
            selection_margin=selection_margin,
            minimum_motion=minimum_motion,
            minimum_improvement=minimum_improvement,
            horizontal_target_distance=horizontal_target_distance,
        )

    def premature_button(
        self,
        before: SceneState,
        after: SceneState,
        *,
        intent_radius: float,
        minimum_approach: float,
        descent_height: float,
    ) -> list[CandidateSignal]:
        if "button" not in after.landmarks:
            return []
        return button_signals(
            before,
            after,
            expected_buttons=(),
            forbidden_buttons=("button",),
            intent_radius=intent_radius,
            selection_margin=0.0,
            minimum_approach_delta=minimum_approach,
            descent_height=descent_height,
            prerequisite_complete=False,
        )


@dataclass
class _ButtonProgressState:
    phase_key: tuple[int, str] | None = None
    phase_start: np.ndarray | None = None
    trace_motion: float = 0.0
    best_distance: float = math.inf
    motion_without_improvement: float = 0.0
    misaligned_descent_active: bool = False
    misaligned_descent_episodes: int = 0
    misaligned_descent_steps: int = 0
    last_misaligned_endpoint_xy: np.ndarray | None = None
    repeated_misaligned_landings: int = 0


def _expected_button_phase_signals(
    before: SceneState,
    after: SceneState,
    *,
    state: _ButtonProgressState,
    button_radius: float,
    button_alignment_radius: float | None = None,
    button_descent_height: float,
    selection_margin: float,
    minimum_approach: float,
    wrong_path_fraction: float,
    prerequisite_complete: bool,
    descent_retry_limit: int | None = None,
    far_descent_radius: float | None = None,
    far_descent_retry_limit: int | None = None,
    far_descent_hysteresis: float = 0.0,
    descent_confirmation_steps: int = 1,
    far_descent_confirmation_steps: int | None = None,
    generic_retry_exhaustion: bool = True,
) -> list[CandidateSignal]:
    if button_alignment_radius is None:
        button_alignment_radius = button_radius
    signals = button_signals(
        before,
        after,
        expected_buttons=("button",),
        forbidden_buttons=(),
        intent_radius=button_radius,
        selection_margin=selection_margin,
        minimum_approach_delta=minimum_approach,
        descent_height=button_descent_height,
        prerequisite_complete=prerequisite_complete,
    )
    if not prerequisite_complete or "button" not in after.landmarks:
        return signals

    button_position = after.landmarks["button"]
    phase_key = (after.subtask.index, after.subtask.phase)
    if phase_key != state.phase_key:
        state.phase_key = phase_key
        state.phase_start = before.tcp_position.copy()
        state.trace_motion = 0.0
        state.best_distance = float(
            np.linalg.norm(before.tcp_position - button_position)
        )
        state.motion_without_improvement = 0.0
        state.misaligned_descent_active = False
        state.misaligned_descent_episodes = 0
        state.misaligned_descent_steps = 0
        state.last_misaligned_endpoint_xy = None
        state.repeated_misaligned_landings = 0
    action_motion = float(
        np.linalg.norm(after.tcp_position - before.tcp_position)
    )
    state.trace_motion += action_motion

    assert state.phase_start is not None
    segment_length = float(
        np.linalg.norm(button_position - state.phase_start)
    )
    current_distance = float(
        np.linalg.norm(after.tcp_position - button_position)
    )
    minimum_trace_motion = wrong_path_fraction * segment_length
    meaningful_improvement = bool(
        state.best_distance - current_distance >= minimum_approach
    )
    if meaningful_improvement:
        state.best_distance = current_distance
        state.motion_without_improvement = 0.0
    else:
        state.motion_without_improvement += action_motion
    wrong_path = bool(
        segment_length > 0.0
        and state.motion_without_improvement >= minimum_trace_motion
    )

    intents = action_intents(before, after)
    button_xy_distance = float(
        np.linalg.norm(after.tcp_position[:2] - button_position[:2])
    )
    button_z_clearance = float(
        after.tcp_position[2] - button_position[2]
    )
    button_corridor_radius = 2.0 * button_radius
    repeated_landing_corridor_radius = (
        button_corridor_radius + 2.0 * selection_margin
    )
    low_descent = bool(
        intents.descending
        and abs(button_z_clearance) <= button_descent_height
    )
    inside_button_corridor = bool(
        button_xy_distance <= button_corridor_radius
    )
    misaligned_descent = bool(
        low_descent and button_xy_distance > button_alignment_radius
    )
    if misaligned_descent:
        if not state.misaligned_descent_active:
            state.misaligned_descent_episodes += 1
            state.misaligned_descent_steps = 0
            if (
                state.last_misaligned_endpoint_xy is not None
                and float(
                    np.linalg.norm(
                        after.tcp_position[:2]
                        - state.last_misaligned_endpoint_xy
                    )
                )
                <= selection_margin
            ):
                state.repeated_misaligned_landings += 1
            else:
                state.repeated_misaligned_landings = 0
        state.misaligned_descent_active = True
        state.misaligned_descent_steps += 1
        state.last_misaligned_endpoint_xy = after.tcp_position[:2].copy()
    else:
        state.misaligned_descent_active = False
        state.misaligned_descent_steps = 0
        if low_descent and button_xy_distance <= button_alignment_radius:
            state.last_misaligned_endpoint_xy = None
            state.repeated_misaligned_landings = 0
    wrong_path_low_descent = bool(
        wrong_path and low_descent and inside_button_corridor
    )
    retry_gated = descent_retry_limit is not None
    effective_far_descent_radius = (
        far_descent_radius + far_descent_hysteresis
        if far_descent_radius is not None
        else None
    )
    far_misaligned_descent = bool(
        effective_far_descent_radius is not None
        and button_xy_distance > effective_far_descent_radius
    )
    active_retry_limit = descent_retry_limit
    active_confirmation_steps = descent_confirmation_steps
    if far_misaligned_descent and far_descent_retry_limit is not None:
        active_retry_limit = far_descent_retry_limit
        if far_descent_confirmation_steps is not None:
            active_confirmation_steps = far_descent_confirmation_steps
    retry_exhausted = bool(
        retry_gated
        and active_retry_limit is not None
        and state.misaligned_descent_episodes > active_retry_limit
        and state.misaligned_descent_steps >= active_confirmation_steps
        and wrong_path
        and (generic_retry_exhaustion or far_misaligned_descent)
    )
    repeated_landing_failure = bool(
        retry_gated
        and active_retry_limit is not None
        and state.misaligned_descent_episodes > active_retry_limit
        and state.repeated_misaligned_landings >= active_retry_limit
        and misaligned_descent
        and not far_misaligned_descent
        and button_xy_distance <= repeated_landing_corridor_radius
    )
    immediate_failure = bool(
        not retry_gated
        and (
            wrong_path_low_descent
            or (misaligned_descent and inside_button_corridor)
        )
    )
    signals.append(
        semantic_signal(
            active=(
                retry_exhausted
                or repeated_landing_failure
                or immediate_failure
            ),
            reason_code=(
                "button.expected_repeated_misaligned_landing"
                if repeated_landing_failure
                else (
                    "button.expected_repeated_far_descent"
                    if retry_exhausted and far_misaligned_descent
                    else (
                        "button.expected_descent_retry_exhausted"
                        if retry_exhausted
                        else (
                            "button.expected_path_no_progress_after_quarter"
                            if wrong_path_low_descent
                            else "button.expected_not_aligned_descent"
                        )
                    )
                )
            ),
            recovery_mode=RecoveryMode.EXPERT_CONTINUE,
            expected_actor="button",
            observed_actor=None,
            metrics={
                "button_distance": current_distance,
                "button_xy_distance": button_xy_distance,
                "button_z_clearance": button_z_clearance,
                "trace_motion": state.trace_motion,
                "best_button_distance": state.best_distance,
                "motion_without_improvement": (
                    state.motion_without_improvement
                ),
                "descending": intents.descending,
                "low_descent": low_descent,
                "inside_button_corridor": inside_button_corridor,
                "wrong_path_pending": wrong_path,
                "misaligned_descent_episodes": (
                    state.misaligned_descent_episodes
                ),
                "misaligned_descent_steps": (
                    state.misaligned_descent_steps
                ),
                "repeated_misaligned_landings": (
                    state.repeated_misaligned_landings
                ),
                "far_misaligned_descent": far_misaligned_descent,
            },
            thresholds={
                "button_radius": button_alignment_radius,
                "button_corridor_radius": button_corridor_radius,
                "repeated_landing_corridor_radius": (
                    repeated_landing_corridor_radius
                ),
                "button_descent_height": button_descent_height,
                "button_wrong_path_fraction": wrong_path_fraction,
                "minimum_trace_motion": minimum_trace_motion,
                "descent_retry_limit": descent_retry_limit,
                "far_descent_radius": far_descent_radius,
                "far_descent_hysteresis": far_descent_hysteresis,
                "effective_far_descent_radius": (
                    effective_far_descent_radius
                ),
                "far_descent_retry_limit": far_descent_retry_limit,
                "descent_confirmation_steps": descent_confirmation_steps,
                "far_descent_confirmation_steps": (
                    far_descent_confirmation_steps
                ),
                "repeated_landing_radius": selection_margin,
            },
            priority=97,
        )
    )
    return signals


class BinFillDetector(TaskPostActionDetector):
    task_name = "BinFill"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self.pickup_radius = 4.0 * self.scale
        self.pickup_center_radius = 1.4 * self.scale
        self.cluster_safe_center_radius = 0.85 * self.scale
        self.cube_grasp_along_radius = 1.25 * self.scale
        self.cube_grasp_cross_radius = 0.85 * self.scale
        self.cluster_safe_along_radius = 0.75 * self.scale
        self.cluster_safe_cross_radius = 0.4 * self.scale
        self.pickup_contact_clearance = 1.25 * self.scale
        self.pickup_intervention_min_clearance = 2.5 * self.scale
        self.pickup_descent_height = 4.0 * self.scale
        self.pickup_closing_height = 3.0 * self.scale
        self.pickup_closing_min_height = 0.75 * self.scale
        self.pickup_minimum_closing_delta = 0.25 * self.scale
        self.pickup_projection_steps = 6.0
        self.pickup_descent_confirmation_steps = 2
        self.cluster_along_radius = 3.5 * self.scale
        self.cluster_cross_radius = 1.25 * self.scale
        self.cluster_vertical_radius = 1.5 * self.scale
        self.selection_margin = 0.75 * self.scale
        self.minimum_approach = 0.05 * self.scale
        self.release_radius = 5.0 * self.scale
        self.transport_radius = 10.0 * self.scale
        self.minimum_place_motion = 0.05 * self.scale
        self.minimum_place_improvement = 0.25 * self.scale
        self.button_radius = 4.0 * self.scale
        self.button_alignment_radius = 2.125 * self.scale
        self.button_descent_height = 4.0 * self.scale
        self.button_wrong_path_fraction = 0.25
        self.button_descent_retry_limit = 2
        self.button_far_descent_radius = 3.0 * self.button_radius
        self.button_far_descent_retry_limit = 2
        self.button_descent_confirmation_steps = 2
        self.button_far_descent_confirmation_steps = 1
        self.place_release_grace_steps = 60
        self._button_progress_state = _ButtonProgressState()
        self._pickup_candidate_subtask: int | None = None
        self._pickup_candidate_steps = 0
        self._wrong_subtask_candidate_subtask: int | None = None
        self._wrong_subtask_candidate_steps = 0
        self._place_release_grace_subtask: int | None = None
        self._place_release_grace_until_step: int | None = None
        self.audit_samples: list[dict[str, Any]] = []

    def _pickup_signals(
        self,
        before: SceneState,
        after: SceneState,
    ) -> list[CandidateSignal]:
        valid_names = tuple(after.metrics.get("expected_actor_names", ()))
        forbidden_names = tuple(after.metrics.get("forbidden_actor_names", ()))
        assessments = [
            pickup_actor_assessment(
                after,
                expected_actor=name,
                expected_actor_kind="cube",
                expected_xy_radius=self.pickup_center_radius,
                cluster_safe_xy_radius=self.cluster_safe_center_radius,
                cluster_along_radius=self.cluster_along_radius,
                cluster_cross_radius=self.cluster_cross_radius,
                cluster_vertical_radius=self.cluster_vertical_radius,
                expected_along_radius=self.cube_grasp_along_radius,
                expected_cross_radius=self.cube_grasp_cross_radius,
                cluster_safe_along_radius=self.cluster_safe_along_radius,
                cluster_safe_cross_radius=self.cluster_safe_cross_radius,
            )
            for name in valid_names
        ]
        if not assessments:
            return []

        nearest = min(assessments, key=lambda value: value.xy_distance)
        intents = action_intents(before, after)
        vertical_drop = max(
            0.0,
            float(before.tcp_position[2] - after.tcp_position[2]),
        )
        projected_tcp_xy = after.tcp_position[:2]
        projection_steps = 0.0
        if intents.descending and vertical_drop > 1e-6:
            projection_steps = min(
                self.pickup_projection_steps,
                max(
                    0.0,
                    nearest.z_clearance - self.pickup_contact_clearance,
                )
                / vertical_drop,
            )
            projected_tcp_xy = (
                after.tcp_position[:2]
                + (after.tcp_position[:2] - before.tcp_position[:2])
                * projection_steps
            )
        projected_xy_distances = {
            value.actor_name: float(
                np.linalg.norm(
                    projected_tcp_xy - after.actors[value.actor_name].position[:2]
                )
            )
            for value in assessments
            if value.actor_name in after.actors
        }
        closing_axis = np.asarray(
            after.metrics.get("gripper_closing_axis_xy", (0.0, 1.0)),
            dtype=np.float64,
        ).reshape(-1)
        closing_axis_norm = float(np.linalg.norm(closing_axis))
        if closing_axis.size != 2 or closing_axis_norm <= 1e-8:
            closing_axis = np.asarray((0.0, 1.0), dtype=np.float64)
        else:
            closing_axis = closing_axis / closing_axis_norm
        cross_axis = np.asarray((-closing_axis[1], closing_axis[0]))
        projected_axis_distances: dict[str, tuple[float, float]] = {}
        for value in assessments:
            if value.actor_name not in after.actors:
                continue
            projected_delta = (
                projected_tcp_xy - after.actors[value.actor_name].position[:2]
            )
            projected_axis_distances[value.actor_name] = (
                abs(float(np.dot(projected_delta, closing_axis))),
                abs(float(np.dot(projected_delta, cross_axis))),
            )
        graspable = []
        for value in assessments:
            along_radius = (
                self.cluster_safe_along_radius
                if value.cluster_obstructed
                else self.cube_grasp_along_radius
            )
            cross_radius = (
                self.cluster_safe_cross_radius
                if value.cluster_obstructed
                else self.cube_grasp_cross_radius
            )
            projected_along, projected_cross = projected_axis_distances.get(
                value.actor_name,
                (math.inf, math.inf),
            )
            current_aligned = bool(
                value.along_distance <= along_radius
                and value.cross_distance <= cross_radius
            )
            projected_aligned = bool(
                projected_along <= along_radius
                and projected_cross <= cross_radius
            )
            if (
                value.actor_kind == "cube"
                and (current_aligned or projected_aligned)
            ):
                graspable.append(value)
        best_graspable = min(graspable, key=lambda value: value.xy_distance, default=None)
        recovery_actor = min(
            assessments,
            key=lambda value: (len(value.obstructing_actors), value.xy_distance),
        )
        any_graspable = best_graspable is not None
        low_descent = bool(
            intents.descending
            and self.pickup_intervention_min_clearance
            <= nearest.z_clearance
            <= self.pickup_descent_height
        )

        forbidden_candidates = [
            after.actors[name]
            for name in forbidden_names
            if name in after.actors
        ]
        nearest_forbidden = min(
            forbidden_candidates,
            key=lambda actor: float(
                np.linalg.norm(after.tcp_position[:2] - actor.position[:2])
            ),
            default=None,
        )
        forbidden_xy_distance = math.inf
        forbidden_z_clearance = math.inf
        if nearest_forbidden is not None:
            forbidden_xy_distance = float(
                np.linalg.norm(
                    after.tcp_position[:2] - nearest_forbidden.position[:2]
                )
            )
            forbidden_z_clearance = float(
                after.tcp_position[2] - nearest_forbidden.position[2]
            )
        wrong_actor_corridor_height = (
            self.pickup_closing_height
            if intents.closing
            else self.pickup_descent_height
        )
        wrong_actor_path = bool(
            nearest_forbidden is not None
            and forbidden_xy_distance <= self.pickup_radius
            and forbidden_xy_distance + self.selection_margin < nearest.xy_distance
            and self.pickup_intervention_min_clearance
            <= forbidden_z_clearance
            <= wrong_actor_corridor_height
        )

        correct_actor_intent = bool(nearest.xy_distance <= self.pickup_radius)
        task_delays_correct_miss = bool(
            after.metrics.get("binfill_delay_correct_ungraspable", False)
            and correct_actor_intent
            and not wrong_actor_path
        )
        closing_delta = float(
            before.gripper_opening - after.gripper_opening
        )
        closing_committed = bool(
            intents.closing
            and closing_delta >= self.pickup_minimum_closing_delta
        )
        closing_without_graspable = bool(
            closing_committed
            and self.pickup_intervention_min_clearance
            <= nearest.z_clearance
            <= self.pickup_closing_height
            and after.held_actor is None
            and not any_graspable
        )
        ungraspable_descent = bool(
            low_descent
            and after.held_actor is None
            and not any_graspable
        )
        if (
            ungraspable_descent
            and self._pickup_candidate_subtask == after.subtask.index
        ):
            self._pickup_candidate_steps += 1
        elif ungraspable_descent:
            self._pickup_candidate_subtask = after.subtask.index
            self._pickup_candidate_steps = 1
        else:
            self._pickup_candidate_subtask = after.subtask.index
            self._pickup_candidate_steps = 0
        gripper_closed = bool(
            before.gripper_opening <= self.closed_gripper_opening
            and after.gripper_opening <= self.closed_gripper_opening
        )
        confirmed_low_descent = bool(
            ungraspable_descent
            and self._pickup_candidate_steps
            >= self.pickup_descent_confirmation_steps
            and not task_delays_correct_miss
        )
        closed_empty_descent = bool(
            ungraspable_descent
            and gripper_closed
            and not task_delays_correct_miss
        )
        half_air_failure = bool(
            confirmed_low_descent
            or closed_empty_descent
            or (
                not gripper_closed
                and (
                    closing_without_graspable
                    and not task_delays_correct_miss
                )
            )
        )
        delayed_miss = bool(
            task_delays_correct_miss
            and gripper_closed
            and not intents.closing
            and after.held_actor is None
            and not any(value.actor_name == after.held_actor for value in assessments)
        )

        reason_code = "binfill.pickup_no_graspable_actor_half_air"
        if wrong_actor_path:
            reason_code = "binfill.pickup_wrong_actor_half_air"
        metrics = {
            "valid_actor_names": valid_names,
            "nearest_valid_actor": nearest.actor_name,
            "nearest_valid_xy_distance": nearest.xy_distance,
            "nearest_valid_along_distance": nearest.along_distance,
            "nearest_valid_cross_distance": nearest.cross_distance,
            "nearest_valid_z_clearance": nearest.z_clearance,
            "nearest_valid_center_aligned": nearest.center_aligned,
            "nearest_valid_cluster_obstructed": nearest.cluster_obstructed,
            "nearest_valid_obstructing_actors": nearest.obstructing_actors,
            "nearest_valid_cluster_safe_centered": (
                nearest.cluster_safe_centered
            ),
            "graspable_actor_names": tuple(value.actor_name for value in graspable),
            "projected_xy_distances": projected_xy_distances,
            "projected_axis_distances": projected_axis_distances,
            "projected_tcp_xy": projected_tcp_xy.tolist(),
            "projection_steps": projection_steps,
            "pickup_candidate_steps": self._pickup_candidate_steps,
            "held_actor": after.held_actor,
            "vertical_drop": vertical_drop,
            "best_graspable_actor": (
                None if best_graspable is None else best_graspable.actor_name
            ),
            "recovery_actor": recovery_actor.actor_name,
            "nearest_forbidden_actor": (
                None if nearest_forbidden is None else nearest_forbidden.name
            ),
            "nearest_forbidden_xy_distance": forbidden_xy_distance,
            "nearest_forbidden_z_clearance": forbidden_z_clearance,
            "wrong_actor_path": wrong_actor_path,
            "correct_actor_intent": correct_actor_intent,
            "task_delays_correct_miss": task_delays_correct_miss,
            "low_descent": low_descent,
            "ungraspable_descent": ungraspable_descent,
            "closed_empty_descent": closed_empty_descent,
            "closing": intents.closing,
            "closing_committed": closing_committed,
            "closing_delta": closing_delta,
            "closing_low": closing_without_graspable,
            "gripper_closed": gripper_closed,
            "before_gripper_opening": before.gripper_opening,
            "after_gripper_opening": after.gripper_opening,
        }
        thresholds = {
            "pickup_intent_radius": self.pickup_radius,
            "pickup_center_radius": self.pickup_center_radius,
            "cluster_safe_center_radius": self.cluster_safe_center_radius,
            "cube_grasp_along_radius": self.cube_grasp_along_radius,
            "cube_grasp_cross_radius": self.cube_grasp_cross_radius,
            "cluster_safe_along_radius": self.cluster_safe_along_radius,
            "cluster_safe_cross_radius": self.cluster_safe_cross_radius,
            "pickup_descent_height": self.pickup_descent_height,
            "pickup_contact_clearance": self.pickup_contact_clearance,
            "pickup_intervention_min_clearance": (
                self.pickup_intervention_min_clearance
            ),
            "pickup_closing_height": self.pickup_closing_height,
            "pickup_closing_min_height": self.pickup_closing_min_height,
            "pickup_minimum_closing_delta": (
                self.pickup_minimum_closing_delta
            ),
            "pickup_projection_steps": self.pickup_projection_steps,
            "pickup_descent_confirmation_steps": (
                self.pickup_descent_confirmation_steps
            ),
            "cluster_along_radius": self.cluster_along_radius,
            "cluster_cross_radius": self.cluster_cross_radius,
            "cluster_vertical_radius": self.cluster_vertical_radius,
            "selection_margin": self.selection_margin,
            "closed_gripper_opening": self.closed_gripper_opening,
            "task_delayed_miss_probability": 0.05,
        }
        descent_audit = bool(
            intents.descending
            and nearest.z_clearance <= self.pickup_descent_height
        )
        if intents.closing or descent_audit:
            self.audit_samples.append(
                {
                    "step": after.step,
                    "subtask_index": after.subtask.index,
                    "subtask_name": after.subtask.name,
                    "active_half_air_failure": half_air_failure,
                    "active_delayed_miss": delayed_miss,
                    "metrics": metrics,
                    "thresholds": thresholds,
                }
            )
            self.audit_samples = self.audit_samples[-2048:]
        return [
            semantic_signal(
                active=half_air_failure,
                reason_code=reason_code,
                recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                expected_actor=recovery_actor.actor_name,
                observed_actor=(
                    nearest_forbidden.name if wrong_actor_path else None
                ),
                metrics=metrics,
                thresholds=thresholds,
                priority=97 if wrong_actor_path else 96,
            ),
            semantic_signal(
                active=delayed_miss,
                reason_code="binfill.pickup_correct_actor_delayed_miss",
                recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                expected_actor=nearest.actor_name,
                observed_actor=None,
                metrics=metrics,
                thresholds=thresholds,
                priority=95,
            ),
        ]

    def _wrong_subtask_pickup_signals(
        self,
        before: SceneState,
        after: SceneState,
    ) -> list[CandidateSignal]:
        cubes = [
            actor
            for actor in after.actors.values()
            if actor.kind == "cube"
            and float(np.linalg.norm(actor.position[:2])) < 2.0
        ]
        if not cubes or "button" not in after.landmarks:
            return []
        nearest = min(
            cubes,
            key=lambda actor: float(
                np.linalg.norm(after.tcp_position[:2] - actor.position[:2])
            ),
        )
        intents = action_intents(before, after)
        vertical_drop = max(
            0.0,
            float(before.tcp_position[2] - after.tcp_position[2]),
        )
        cube_xy_distance = float(
            np.linalg.norm(after.tcp_position[:2] - nearest.position[:2])
        )
        cube_z_clearance = float(after.tcp_position[2] - nearest.position[2])
        projected_tcp_xy = after.tcp_position[:2]
        if intents.descending and vertical_drop > 1e-6:
            projection_steps = min(
                self.pickup_projection_steps,
                max(
                    0.0,
                    cube_z_clearance - self.pickup_contact_clearance,
                )
                / vertical_drop,
            )
            projected_tcp_xy = (
                after.tcp_position[:2]
                + (after.tcp_position[:2] - before.tcp_position[:2])
                * projection_steps
            )
        projected_cube_xy_distance = float(
            np.linalg.norm(projected_tcp_xy - nearest.position[:2])
        )
        button_xy_distance = float(
            np.linalg.norm(
                after.tcp_position[:2] - after.landmarks["button"][:2]
            )
        )
        pickup_descent = bool(
            after.metrics.get("objects_complete", False)
            and intents.descending
            and self.pickup_intervention_min_clearance
            <= cube_z_clearance
            <= self.pickup_descent_height
            and min(cube_xy_distance, projected_cube_xy_distance)
            <= self.pickup_radius
            and button_xy_distance > self.button_alignment_radius
        )
        if (
            pickup_descent
            and self._wrong_subtask_candidate_subtask == after.subtask.index
        ):
            self._wrong_subtask_candidate_steps += 1
        elif pickup_descent:
            self._wrong_subtask_candidate_subtask = after.subtask.index
            self._wrong_subtask_candidate_steps = 1
        else:
            self._wrong_subtask_candidate_subtask = after.subtask.index
            self._wrong_subtask_candidate_steps = 0
        metrics = {
            "expected_phase": "press_button",
            "observed_motion_primitive": "pickup_descent",
            "nearest_cube": nearest.name,
            "nearest_cube_xy_distance": cube_xy_distance,
            "nearest_cube_z_clearance": cube_z_clearance,
            "projected_cube_xy_distance": projected_cube_xy_distance,
            "button_xy_distance": button_xy_distance,
            "pickup_descent": pickup_descent,
            "wrong_subtask_candidate_steps": (
                self._wrong_subtask_candidate_steps
            ),
        }
        thresholds = {
            "pickup_intent_radius": self.pickup_radius,
            "pickup_contact_clearance": self.pickup_contact_clearance,
            "pickup_intervention_min_clearance": (
                self.pickup_intervention_min_clearance
            ),
            "pickup_descent_height": self.pickup_descent_height,
            "button_alignment_radius": self.button_alignment_radius,
            "confirmation_steps": self.pickup_descent_confirmation_steps,
        }
        if intents.descending and cube_z_clearance <= self.pickup_descent_height:
            self.audit_samples.append(
                {
                    "step": after.step,
                    "subtask_index": after.subtask.index,
                    "subtask_name": after.subtask.name,
                    "active_wrong_subtask_failure": bool(
                        pickup_descent
                        and self._wrong_subtask_candidate_steps
                        >= self.pickup_descent_confirmation_steps
                    ),
                    "metrics": metrics,
                    "thresholds": thresholds,
                }
            )
            self.audit_samples = self.audit_samples[-2048:]
        return [
            semantic_signal(
                active=bool(
                    pickup_descent
                    and self._wrong_subtask_candidate_steps
                    >= self.pickup_descent_confirmation_steps
                ),
                reason_code="binfill.press_button_pickup_descent",
                recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                expected_actor=None,
                observed_actor=nearest.name,
                metrics=metrics,
                thresholds=thresholds,
                priority=99,
            )
        ]

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        phase = after.subtask.phase
        if phase == "pickup":
            self._place_release_grace_subtask = None
            self._place_release_grace_until_step = None
            return self._pickup_signals(before, after)
        if phase == "place_in_bin":
            if after.metrics.get("objects_complete", False):
                return []
            entered_without_held_actor = bool(
                before.subtask.index != after.subtask.index
                and after.held_actor is None
            )
            if entered_without_held_actor:
                self._place_release_grace_subtask = after.subtask.index
                self._place_release_grace_until_step = (
                    after.step + self.place_release_grace_steps
                )
            expected_before = set(
                before.metrics.get("expected_actor_names", ())
            )
            expected_after = set(
                after.metrics.get("expected_actor_names", ())
            )
            intents = action_intents(before, after)
            legal_release = bool(
                before.held_actor is not None
                and before.held_actor in expected_before | expected_after
                and after.held_actor is None
                and intents.opening
            )
            if legal_release:
                self._place_release_grace_subtask = after.subtask.index
                self._place_release_grace_until_step = (
                    after.step + self.place_release_grace_steps
                )
            signals = self.place(
                before,
                after,
                release_radius=self.release_radius,
                transport_radius=self.transport_radius,
                selection_margin=self.selection_margin,
                minimum_motion=self.minimum_place_motion,
                minimum_improvement=self.minimum_place_improvement,
                horizontal_target_distance=True,
            )
            release_grace_active = bool(
                self._place_release_grace_subtask == after.subtask.index
                and self._place_release_grace_until_step is not None
                and after.step <= self._place_release_grace_until_step
            )
            if release_grace_active:
                signals = [
                    signal
                    for signal in signals
                    if signal.reason_code != "identity.required_actor_not_held"
                ]
            return signals
        if phase == "press_button":
            signals = self._wrong_subtask_pickup_signals(before, after)
            signals.extend(_expected_button_phase_signals(
                before,
                after,
                state=self._button_progress_state,
                button_radius=self.button_radius,
                button_alignment_radius=self.button_alignment_radius,
                button_descent_height=self.button_descent_height,
                selection_margin=self.selection_margin,
                minimum_approach=self.minimum_approach,
                wrong_path_fraction=self.button_wrong_path_fraction,
                prerequisite_complete=bool(
                    after.metrics.get("objects_complete", False)
                ),
                descent_retry_limit=self.button_descent_retry_limit,
                far_descent_radius=self.button_far_descent_radius,
                far_descent_retry_limit=self.button_far_descent_retry_limit,
                far_descent_hysteresis=self.selection_margin,
                descent_confirmation_steps=(
                    self.button_descent_confirmation_steps
                ),
                far_descent_confirmation_steps=(
                    self.button_far_descent_confirmation_steps
                ),
                generic_retry_exhaustion=False,
            ))
            for signal in signals:
                if (
                    signal.reason_code.startswith("button.expected_")
                    and signal.metrics.get("low_descent", False)
                ):
                    self.audit_samples.append(
                        {
                            "step": after.step,
                            "subtask_index": after.subtask.index,
                            "subtask_name": after.subtask.name,
                            "active_button_failure": signal.active,
                            "metrics": dict(signal.metrics),
                            "thresholds": dict(signal.thresholds),
                        }
                    )
                    self.audit_samples = self.audit_samples[-2048:]
            return signals
        return []


class _ContainerDetector(TaskPostActionDetector):
    def _initialize_container_thresholds(self) -> None:
        self.container_radius = 5.0 * self.scale
        self.pickup_xy_alignment_radius = 3.0 * self.scale
        self.pickup_descent_height = 4.0 * self.scale
        self.container_selection_margin = 0.75 * self.scale
        self.container_minimum_approach = 0.05 * self.scale
        self.button_radius = 4.0 * self.scale
        self.button_selection_margin = 0.75 * self.scale
        self.button_descent_height = 4.0 * self.scale

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        expected = tuple(after.metrics.get("expected_actor_names", ()))
        phase = after.subtask.phase
        if phase == "pickup_container":
            signals = self.pickup(
                before,
                after,
                intent_radius=self.container_radius,
                selection_margin=self.container_selection_margin,
                minimum_approach=self.container_minimum_approach,
            )
            signals.append(held_identity_signal(after, expected_actors=expected))
            return signals
        if phase == "putdown_container":
            return [
                held_identity_signal(after, expected_actors=expected),
                required_held_signal(before, after, expected_actors=expected),
            ]
        if phase == "press_button":
            allowed = tuple(after.metrics.get("allowed_button_names", ()))
            forbidden = tuple(name for name in after.landmarks if name not in allowed)
            return button_signals(
                before,
                after,
                expected_buttons=allowed,
                forbidden_buttons=forbidden,
                intent_radius=self.button_radius,
                selection_margin=self.button_selection_margin,
                minimum_approach_delta=self.container_minimum_approach,
                descent_height=self.button_descent_height,
            )
        return []


class ButtonUnmaskDetector(_ContainerDetector):
    task_name = "ButtonUnmask"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_container_thresholds()


class ButtonUnmaskSwapDetector(_ContainerDetector):
    task_name = "ButtonUnmaskSwap"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_container_thresholds()


class VideoUnmaskDetector(_ContainerDetector):
    task_name = "VideoUnmask"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_container_thresholds()


class VideoUnmaskSwapDetector(_ContainerDetector):
    task_name = "VideoUnmaskSwap"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_container_thresholds()


class InsertPegDetector(TaskPostActionDetector):
    task_name = "InsertPeg"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self.endpoint_radius = 4.0 * self.scale
        self.selection_margin = 0.75 * self.scale
        self.minimum_approach = 0.05 * self.scale
        self.insertion_radius = 4.0 * self.scale
        self.side_margin = 0.5 * self.scale
        self.pickup_xy_alignment_radius = self.endpoint_radius
        self.pickup_descent_height = 8.0 * self.scale

    def _wrong_endpoint_precontact_signal(
        self,
        before: SceneState,
        after: SceneState,
    ) -> CandidateSignal:
        expected_names = tuple(after.metrics.get("expected_actor_names", ()))
        forbidden_names = tuple(after.metrics.get("forbidden_actor_names", ()))
        native_expected_name = after.metrics.get("grasp_target_actor")
        expected_name = (
            str(native_expected_name)
            if native_expected_name in after.actors
            else expected_names[0] if len(expected_names) == 1 else None
        )
        expected_position = (
            after.actors[expected_name].position
            if expected_name in after.actors
            else None
        )
        forbidden_candidates = [
            name for name in forbidden_names if name in after.actors
        ]
        native_opposite_name = after.metrics.get("insert_target_actor")
        if native_opposite_name in after.actors:
            forbidden_name = str(native_opposite_name)
        else:
            forbidden_name = min(
                forbidden_candidates,
                key=lambda name: float(
                    np.linalg.norm(
                        after.tcp_position[:2] - after.actors[name].position[:2]
                    )
                ),
                default=None,
            )
        expected_xy_distance = math.inf
        if expected_position is not None:
            expected_xy_distance = float(
                np.linalg.norm(after.tcp_position[:2] - expected_position[:2])
            )
        forbidden_xy_distance = math.inf
        forbidden_z_clearance = math.inf
        before_forbidden_xy_distance = math.inf
        if forbidden_name is not None:
            forbidden_position = after.actors[forbidden_name].position
            forbidden_xy_distance = float(
                np.linalg.norm(after.tcp_position[:2] - forbidden_position[:2])
            )
            forbidden_z_clearance = float(
                after.tcp_position[2] - forbidden_position[2]
            )
            if forbidden_name in before.actors:
                before_forbidden_xy_distance = float(
                    np.linalg.norm(
                        before.tcp_position[:2]
                        - before.actors[forbidden_name].position[:2]
                    )
                )
        endpoint_separation = float(
            after.metrics.get(
                "selected_peg_endpoint_separation",
                2.0 * self.endpoint_radius,
            )
        )
        endpoint_xy_radius = self.endpoint_radius
        endpoint_descent_height = self.pickup_descent_height
        forbidden_xy_approach = (
            before_forbidden_xy_distance - forbidden_xy_distance
        )
        intents = action_intents(before, after)
        active = bool(
            forbidden_name is not None
            and min(forbidden_xy_distance, expected_xy_distance)
            <= endpoint_xy_radius
            and 0.0 <= forbidden_z_clearance <= endpoint_descent_height
            and forbidden_xy_distance <= expected_xy_distance
            and (
                intents.closing
                or intents.descending
                or forbidden_xy_approach >= self.minimum_approach
            )
        )
        return semantic_signal(
            active=active,
            reason_code="insertpeg.wrong_endpoint_precontact",
            recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
            expected_actor=expected_name,
            observed_actor=forbidden_name,
            metrics={
                "expected_xy_distance": expected_xy_distance,
                "forbidden_xy_distance": forbidden_xy_distance,
                "forbidden_z_clearance": forbidden_z_clearance,
                "forbidden_xy_approach_delta": forbidden_xy_approach,
                "descending": intents.descending,
                "closing": intents.closing,
                "selected_peg_grasped": after.metrics.get(
                    "selected_peg_grasped",
                    False,
                ),
                "held_actor": after.held_actor,
            },
            thresholds={
                "endpoint_xy_radius": endpoint_xy_radius,
                "endpoint_descent_height": endpoint_descent_height,
                "native_closer_margin": 0.0,
                "minimum_approach": self.minimum_approach,
                "endpoint_separation": endpoint_separation,
            },
            priority=98,
        )

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        if after.subtask.phase == "pickup_peg_end":
            endpoint_separation = float(
                after.metrics.get(
                    "selected_peg_endpoint_separation",
                    2.0 * self.endpoint_radius,
                )
            )
            return [
                self._wrong_endpoint_precontact_signal(before, after),
                *self.pickup(
                    before,
                    after,
                    intent_radius=max(
                        self.endpoint_radius,
                        endpoint_separation,
                    ),
                    selection_margin=self.selection_margin,
                    minimum_approach=self.minimum_approach,
                    expected_xy_radius=self.endpoint_radius,
                ),
            ]
        expected_insert = str(after.metrics["insert_target_actor"])
        opposite_end = str(after.metrics["opposite_end_actor"])
        box = after.landmarks["box"]
        insert_position = after.actors[expected_insert].position
        opposite_position = after.actors[opposite_end].position
        before_insert_position = before.actors[expected_insert].position
        insert_distance = float(np.linalg.norm(insert_position - box))
        opposite_distance = float(np.linalg.norm(opposite_position - box))
        before_distance = float(np.linalg.norm(before_insert_position - box))
        approach = before_distance - insert_distance
        direction = int(after.metrics["insert_direction"])
        side_axis = int(after.metrics["insert_side_axis"])
        side_product = float((insert_position[side_axis] - box[side_axis]) * direction)
        wrong_end_closer = bool(
            opposite_distance + self.selection_margin < insert_distance
            and opposite_distance <= self.insertion_radius
        )
        wrong_side = bool(
            insert_distance <= self.insertion_radius
            and side_product >= self.side_margin
        )
        approaching = (
            approach >= self.minimum_approach
            or action_intents(before, after).descending
        )
        return [
            semantic_signal(
                active=not bool(after.metrics["selected_peg_grasped"]),
                reason_code="insert.selected_peg_not_held",
                recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                expected_actor=str(after.metrics["grasp_target_actor"]),
                observed_actor=None,
                metrics={
                    "grasp_target_grasped": after.metrics[
                        "grasp_target_grasped"
                    ],
                    "insert_target_grasped": after.metrics[
                        "insert_target_grasped"
                    ],
                },
                thresholds={"selected_peg_must_remain_grasped": True},
                priority=99,
            ),
            semantic_signal(
                active=approaching and (wrong_end_closer or wrong_side),
                reason_code=(
                    "insert.wrong_endpoint_alignment"
                    if wrong_end_closer
                    else "insert.wrong_side_alignment"
                ),
                recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
                expected_actor=expected_insert,
                observed_actor=opposite_end if wrong_end_closer else None,
                metrics={
                    "insert_distance": insert_distance,
                    "opposite_distance": opposite_distance,
                    "approach_delta": approach,
                    "side_product": side_product,
                    "direction": direction,
                },
                thresholds={
                    "insertion_radius": self.insertion_radius,
                    "selection_margin": self.selection_margin,
                    "minimum_approach": self.minimum_approach,
                    "side_margin": self.side_margin,
                },
                priority=98,
            )
        ]


class MoveCubeDetector(TaskPostActionDetector):
    task_name = "MoveCube"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self.pickup_radius = 4.0 * self.scale
        self.selection_margin = 0.75 * self.scale
        self.minimum_approach = 0.05 * self.scale
        self.minimum_goal_improvement = 0.05 * self.scale
        self.native_pickup_height = 0.04
        self.push_pickup_xy_radius = 2.0 * self.scale
        self.push_pickup_descent_height = 6.0 * self.scale
        self.push_contact_xy_radius = 5.0 * self.scale
        self.push_minimum_side_cosine = 0.25

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        phase = after.subtask.phase
        expected = tuple(after.metrics.get("expected_actor_names", ()))
        if phase in {"pickup_tool", "pickup_cube"}:
            return self.pickup(
                before,
                after,
                intent_radius=self.pickup_radius,
                selection_margin=self.selection_margin,
                minimum_approach=self.minimum_approach,
            )
        if phase == "peg_push":
            return [
                held_identity_signal(
                    after,
                    expected_actors=expected,
                    reason_code="movecube.wrong_tool_held_before_hook",
                ),
                required_held_signal(before, after, expected_actors=expected),
            ]
        if phase == "gripper_push":
            intents = action_intents(before, after)
            cube_name = str(after.metrics.get("cube_actor"))
            cube = after.actors.get(cube_name)
            before_cube = before.actors.get(cube_name)
            goal = after.landmarks.get("goal")
            cube_xy_distance = math.inf
            cube_z_clearance = math.inf
            gripper_pickup_risk = False
            forbidden_xy_distance = math.inf
            forbidden_z_clearance = math.inf
            forbidden_pickup_actor = None
            forbidden_pickup_risk = False
            if cube is not None:
                cube_xy_distance = float(
                    np.linalg.norm(after.tcp_position[:2] - cube.position[:2])
                )
                cube_z_clearance = float(after.tcp_position[2] - cube.position[2])
                gripper_pickup_risk = bool(
                    after.held_actor is None
                    and not cube.grasped
                    and cube_xy_distance <= self.push_pickup_xy_radius
                    and cube_z_clearance <= self.push_pickup_descent_height
                    and (
                        intents.descending
                        or intents.closing
                        or after.gripper_opening <= self.closed_gripper_opening
                    )
                )
            for actor_name in after.metrics.get("forbidden_actor_names", ()):
                if actor_name == cube_name:
                    continue
                actor = after.actors.get(str(actor_name))
                if actor is None:
                    continue
                actor_xy_distance = float(
                    np.linalg.norm(after.tcp_position[:2] - actor.position[:2])
                )
                actor_z_clearance = float(after.tcp_position[2] - actor.position[2])
                if actor_xy_distance >= forbidden_xy_distance:
                    continue
                forbidden_xy_distance = actor_xy_distance
                forbidden_z_clearance = actor_z_clearance
                forbidden_pickup_actor = str(actor_name)
            forbidden_pickup_risk = bool(
                after.held_actor is None
                and forbidden_pickup_actor is not None
                and forbidden_xy_distance <= self.push_pickup_xy_radius
                and forbidden_z_clearance <= self.push_pickup_descent_height
                and (
                    intents.descending
                    or intents.closing
                    or after.gripper_opening <= self.closed_gripper_opening
                )
            )
            lifted = bool(
                cube is not None
                and before_cube is not None
                and cube.grasped
                and cube.position[2] >= self.native_pickup_height
                and cube.position[2] - before_cube.position[2] >= self.minimum_approach
            )
            cube_motion = 0.0
            goal_improvement = math.inf
            after_goal_distance = math.inf
            push_side_cosine = math.inf
            wrong_push_side = False
            if cube is not None and before_cube is not None and goal is not None:
                cube_motion = float(np.linalg.norm(cube.position - before_cube.position))
                before_goal_distance = float(np.linalg.norm(before_cube.position - goal))
                after_goal_distance = float(np.linalg.norm(cube.position - goal))
                goal_improvement = before_goal_distance - after_goal_distance
                goal_vector = goal[:2] - cube.position[:2]
                tcp_to_cube = cube.position[:2] - after.tcp_position[:2]
                goal_norm = float(np.linalg.norm(goal_vector))
                tcp_to_cube_norm = float(np.linalg.norm(tcp_to_cube))
                if goal_norm > 1e-12 and tcp_to_cube_norm > 1e-12:
                    push_side_cosine = float(
                        np.dot(tcp_to_cube, goal_vector)
                        / (tcp_to_cube_norm * goal_norm)
                    )
                wrong_push_side = bool(
                    after.held_actor is None
                    and not cube.grasped
                    and cube_xy_distance <= self.push_contact_xy_radius
                    and cube_z_clearance <= self.push_pickup_descent_height
                    and after_goal_distance > self.pickup_radius
                    and push_side_cosine <= self.push_minimum_side_cosine
                    and (
                        intents.descending
                        or intents.closing
                        or after.gripper_opening <= self.closed_gripper_opening
                    )
                )
            moving_away_from_goal = bool(
                cube_motion >= self.minimum_approach
                and after_goal_distance > self.pickup_radius
                and goal_improvement <= -self.minimum_goal_improvement
            )
            return [
                semantic_signal(
                    active=(
                        forbidden_pickup_risk
                        or gripper_pickup_risk
                        or wrong_push_side
                        or lifted
                        or moving_away_from_goal
                    ),
                    reason_code=(
                        "movecube.gripper_push_wrong_actor_pickup_risk"
                        if forbidden_pickup_risk
                        else
                        "movecube.gripper_push_pickup_risk"
                        if gripper_pickup_risk
                        else
                        "movecube.gripper_push_wrong_side_contact"
                        if wrong_push_side
                        else
                        "movecube.forbidden_pickup_before_push"
                        if lifted
                        else "movecube.gripper_push_moving_away"
                    ),
                    recovery_mode=(
                        RecoveryMode.OPEN_AND_RETRACT
                        if (
                            forbidden_pickup_risk
                            or gripper_pickup_risk
                            or wrong_push_side
                        )
                        else
                        RecoveryMode.CANCEL_FORBIDDEN_PUSH_PICKUP
                        if lifted
                        else RecoveryMode.EXPERT_CONTINUE
                    ),
                    expected_actor="cube_push_contact",
                    observed_actor=(
                        forbidden_pickup_actor
                        if forbidden_pickup_risk
                        else cube_name
                    ),
                    metrics={
                        "cube_height": None if cube is None else float(cube.position[2]),
                        "cube_xy_distance": cube_xy_distance,
                        "cube_z_clearance": cube_z_clearance,
                        "forbidden_xy_distance": forbidden_xy_distance,
                        "forbidden_z_clearance": forbidden_z_clearance,
                        "cube_motion": cube_motion,
                        "goal_improvement": goal_improvement,
                        "after_goal_distance": after_goal_distance,
                        "push_side_cosine": push_side_cosine,
                        "descending": intents.descending,
                        "closing": intents.closing,
                    },
                    thresholds={
                        "native_pickup_height": self.native_pickup_height,
                        "push_pickup_xy_radius": self.push_pickup_xy_radius,
                        "push_pickup_descent_height": (
                            self.push_pickup_descent_height
                        ),
                        "closed_gripper_opening": self.closed_gripper_opening,
                        "minimum_goal_improvement": self.minimum_goal_improvement,
                        "push_contact_xy_radius": self.push_contact_xy_radius,
                        "push_minimum_side_cosine": (
                            self.push_minimum_side_cosine
                        ),
                    },
                    priority=99,
                )
            ]
        if phase == "place_cube":
            return place_signals(
                before,
                after,
                expected_actors=expected,
                expected_targets=("goal",),
                forbidden_targets=(),
                release_radius=self.pickup_radius,
                transport_radius=2.0 * self.pickup_radius,
                selection_margin=self.selection_margin,
                minimum_motion=self.minimum_approach,
                minimum_improvement=self.minimum_approach,
            )
        return []


class PatternLockDetector(TaskPostActionDetector):
    task_name = "PatternLock"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self.target_radius = 2.5 * self.scale
        self.wrong_waypoint_radius = 3.0 * self.scale
        self.minimum_motion = 0.25 * self.scale
        self.wrong_direction_progress = 0.25

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        expected = tuple(after.metrics.get("expected_waypoint_names", ()))
        if len(expected) != 1:
            return []
        tracked_point = "swing_tip"
        start_name = str(after.metrics["corridor_start"])
        target_name = expected[0]
        if (
            tracked_point not in before.points
            or tracked_point not in after.points
            or start_name not in after.landmarks
            or target_name not in after.landmarks
        ):
            return []
        before_xy = before.points[tracked_point][:2]
        after_xy = after.points[tracked_point][:2]
        start_xy = after.landmarks[start_name][:2]
        target_xy = after.landmarks[target_name][:2]
        motion = after_xy - before_xy
        motion_distance = float(np.linalg.norm(motion))
        before_distance = float(np.linalg.norm(before_xy - target_xy))
        after_distance = float(np.linalg.norm(after_xy - target_xy))
        route_vector = target_xy - start_xy
        segment_length = float(np.linalg.norm(target_xy - start_xy))
        progress = 0.0
        forward_distance = 0.0
        lateral_distance = 0.0
        distance_from_start = float(np.linalg.norm(after_xy - start_xy))
        if segment_length > 1e-12:
            route_unit = route_vector / segment_length
            displacement_from_start = after_xy - start_xy
            forward_distance = float(np.dot(displacement_from_start, route_unit))
            lateral_vector = displacement_from_start - forward_distance * route_unit
            lateral_distance = float(np.linalg.norm(lateral_vector))
            progress = forward_distance / segment_length
        forbidden_names = tuple(
            after.metrics.get("forbidden_waypoint_names", ())
        )
        wrong_name = None
        wrong_distance = math.inf
        for name in forbidden_names:
            if name not in after.landmarks:
                continue
            distance = float(
                np.linalg.norm(after_xy - after.landmarks[name][:2])
            )
            if distance < wrong_distance:
                wrong_name = name
                wrong_distance = distance
        wrong_direction_distance = (
            self.wrong_direction_progress * segment_length
        )
        wrong_direction_from_start = bool(
            segment_length > 1e-12
            and distance_from_start >= wrong_direction_distance
            and motion_distance >= self.minimum_motion
            and (
                forward_distance < wrong_direction_distance
                or lateral_distance >= wrong_direction_distance
            )
        )
        wrong_waypoint_risk = bool(
            progress >= 0.5
            and wrong_name is not None
            and wrong_distance <= self.wrong_waypoint_radius
            and wrong_distance < after_distance
        )
        return [
            semantic_signal(
                active=wrong_direction_from_start or wrong_waypoint_risk,
                reason_code=(
                    "pattern.wrong_waypoint_risk"
                    if wrong_waypoint_risk
                    else "pattern.wrong_direction_after_quarter"
                ),
                recovery_mode=RecoveryMode.EXPERT_CONTINUE,
                expected_actor=target_name,
                observed_actor=wrong_name if wrong_waypoint_risk else None,
                metrics={
                    "before_target_distance_xy": before_distance,
                    "after_target_distance_xy": after_distance,
                    "motion_xy": motion_distance,
                    "distance_from_start_xy": distance_from_start,
                    "forward_distance_from_start_xy": forward_distance,
                    "lateral_distance_from_segment_xy": lateral_distance,
                    "segment_progress": progress,
                    "wrong_waypoint_distance_xy": wrong_distance,
                },
                thresholds={
                    "minimum_motion": self.minimum_motion,
                    "wrong_direction_progress": self.wrong_direction_progress,
                    "wrong_direction_distance": wrong_direction_distance,
                    "wrong_waypoint_radius": self.wrong_waypoint_radius,
                    "wrong_waypoint_minimum_progress": 0.5,
                },
                priority=96,
            )
        ]


class RouteStickDetector(TaskPostActionDetector):
    task_name = "RouteStick"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self.minimum_motion = 0.25 * self.scale
        self.direction_side_fraction = 0.25
        self.minimum_trace_progress = 0.25
        self.target_radius = 1.5 * self.scale
        self._segment_key: tuple[int, str, str, str] | None = None
        self._cross_sum = 0.0
        self._cross_count = 0
        self._trace_motion = 0.0

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        expected = tuple(after.metrics.get("expected_waypoint_names", ()))
        if len(expected) != 1:
            return []
        signals: list[CandidateSignal] = []
        point = after.points.get("stick_tip")
        start = after.landmarks.get(str(after.metrics["corridor_start"]))
        end = after.landmarks.get(expected[0])
        if point is None or start is None or end is None:
            return signals
        route_vector = end[:2] - start[:2]
        route_norm = float(np.linalg.norm(route_vector))
        if route_norm <= 1e-12:
            return signals
        segment_key = (
            after.subtask.index,
            str(after.metrics["corridor_start"]),
            expected[0],
            str(after.metrics.get("expected_direction")),
        )
        if segment_key != self._segment_key:
            self._segment_key = segment_key
            self._cross_sum = 0.0
            self._cross_count = 0
            self._trace_motion = 0.0
        point_vector = point[:2] - start[:2]
        signed_cross = float(route_vector[0] * point_vector[1] - route_vector[1] * point_vector[0])
        signed_side = signed_cross / route_norm
        observed_direction = "clockwise" if signed_side > 0.0 else "counterclockwise"
        expected_direction = str(after.metrics.get("expected_direction"))
        motion = float(
            np.linalg.norm(
                after.points["stick_tip"][:2] - before.points["stick_tip"][:2]
            )
        )
        if motion >= self.minimum_motion:
            self._cross_sum += signed_cross
            self._cross_count += 1
            self._trace_motion += motion
        average_signed_side = 0.0
        if self._cross_count:
            average_signed_side = (self._cross_sum / self._cross_count) / route_norm
        expected_sign = 1.0 if expected_direction == "clockwise" else -1.0
        native_peak_lateral_offset = float(
            after.metrics["native_route_peak_lateral_offset"]
        )
        direction_side_margin = max(
            self.minimum_motion,
            self.direction_side_fraction * native_peak_lateral_offset,
        )
        instant_wrong_direction = bool(
            motion >= self.minimum_motion
            and abs(signed_side) >= direction_side_margin
            and signed_side * expected_sign < 0.0
        )
        trace_wrong_direction = bool(
            self._trace_motion >= self.minimum_trace_progress * route_norm
            and abs(average_signed_side) >= direction_side_margin
            and average_signed_side * expected_sign < 0.0
        )
        wrong_direction = instant_wrong_direction or trace_wrong_direction
        official_wrong_waypoint_radius = float(
            after.metrics["official_wrong_waypoint_xy_radius"]
        )
        wrong_waypoint_radius = 3.0 * official_wrong_waypoint_radius
        official_route_z_threshold = float(
            after.metrics["official_route_z_threshold"]
        )
        wrong_waypoint_name = None
        wrong_waypoint_distance = math.inf
        for name in after.metrics.get("forbidden_waypoint_names", ()):
            if name not in after.landmarks:
                continue
            distance = float(
                np.linalg.norm(point[:2] - after.landmarks[name][:2])
            )
            if distance < wrong_waypoint_distance:
                wrong_waypoint_name = name
                wrong_waypoint_distance = distance
        expected_waypoint_distance = float(np.linalg.norm(point[:2] - end[:2]))
        wrong_waypoint_risk = bool(
            motion >= self.minimum_motion
            and wrong_waypoint_name is not None
            and wrong_waypoint_distance <= wrong_waypoint_radius
            and wrong_waypoint_distance < expected_waypoint_distance
            and point[2] < official_route_z_threshold
        )
        signals.append(
            semantic_signal(
                active=wrong_waypoint_risk,
                reason_code="route.wrong_waypoint_risk",
                recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
                expected_actor=expected[0],
                observed_actor=wrong_waypoint_name,
                metrics={
                    "wrong_waypoint_distance_xy": wrong_waypoint_distance,
                    "expected_waypoint_distance_xy": expected_waypoint_distance,
                    "stick_tip_z": float(point[2]),
                },
                thresholds={
                    "official_wrong_waypoint_xy_radius": (
                        official_wrong_waypoint_radius
                    ),
                    "warning_wrong_waypoint_xy_radius": wrong_waypoint_radius,
                    "official_route_z_threshold": official_route_z_threshold,
                },
                priority=97,
            )
        )
        safe_anchor = 0.5 * (start + end)
        route_unit = route_vector / route_norm
        route_perpendicular = np.asarray(
            [-route_unit[1], route_unit[0]],
            dtype=np.float64,
        )
        safe_anchor[:2] += (
            expected_sign * native_peak_lateral_offset * route_perpendicular
        )
        signals.append(
            semantic_signal(
                active=wrong_direction,
                reason_code=(
                    "route.direction_trace_mismatch"
                    if trace_wrong_direction
                    else "route.direction_side_mismatch"
                ),
                recovery_mode=RecoveryMode.EXPERT_CONTINUE,
                expected_actor=expected[0],
                observed_actor=None,
                metrics={
                    "expected_direction": expected_direction,
                    "observed_direction": observed_direction,
                    "signed_side_distance": signed_side,
                    "average_signed_side_distance": average_signed_side,
                    "motion": motion,
                    "trace_motion": self._trace_motion,
                    "route_segment_length": route_norm,
                    "safe_anchor": safe_anchor,
                },
                thresholds={
                    "direction_side_fraction": self.direction_side_fraction,
                    "direction_side_margin": direction_side_margin,
                    "native_route_peak_lateral_offset": (
                        native_peak_lateral_offset
                    ),
                    "minimum_motion": self.minimum_motion,
                    "minimum_trace_progress": self.minimum_trace_progress,
                },
                priority=96,
            )
        )
        return signals


class StopCubeDetector(TaskPostActionDetector):
    task_name = "StopCube"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self.minimum_ready_progress = 0.25
        self.early_descent_actions = 45
        self.latest_start_actions = 60
        # The native press needs enough runway after takeover.  The r07
        # videos still had late terminal misses, so keep the original
        # early-descent guard but hand over to the expert press routine
        # earlier before the terminal window collapses.
        self.expert_press_actions = 70
        self.step_seconds = 1.0 / 30.0

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        if not bool(after.metrics.get("timing_detector_ready", False)):
            return []
        progress = float(after.metrics["press_progress"])
        seconds_to_event = float(after.metrics["seconds_to_event"])
        early_seconds = self.early_descent_actions * self.step_seconds
        latest_start_seconds = self.latest_start_actions * self.step_seconds
        expert_seconds = self.expert_press_actions * self.step_seconds
        intents = action_intents(before, after)
        premature = bool(
            intents.descending
            and progress >= self.minimum_ready_progress
            and seconds_to_event > early_seconds
        )
        too_late = bool(
            seconds_to_event < latest_start_seconds
            and not intents.descending
        )
        return [
            semantic_signal(
                active=premature or too_late,
                reason_code=(
                    "stopcube.premature_descent"
                    if premature
                    else "stopcube.latest_start_reached"
                ),
                recovery_mode=RecoveryMode.HOLD_POSE,
                expected_actor="button",
                observed_actor=None,
                metrics={
                    "press_progress": progress,
                    "seconds_to_event": seconds_to_event,
                    "descending": intents.descending,
                },
                thresholds={
                    "minimum_ready_progress": self.minimum_ready_progress,
                    "early_descent_actions": self.early_descent_actions,
                    "early_descent_seconds": early_seconds,
                    "latest_start_actions": self.latest_start_actions,
                    "latest_start_seconds": latest_start_seconds,
                    "expert_press_actions": self.expert_press_actions,
                    "expert_press_seconds": expert_seconds,
                },
                priority=100,
            )
        ]


class _PickupPlaceButtonDetector(TaskPostActionDetector):
    def _initialize_pickup_place_thresholds(self) -> None:
        self.pickup_radius = 4.0 * self.scale
        self.pickup_xy_alignment_radius = 2.5 * self.scale
        self.pickup_descent_height = 4.0 * self.scale
        self.pickup_selection_margin = 0.75 * self.scale
        self.minimum_approach = 0.05 * self.scale
        self.release_radius = 5.0 * self.scale
        self.release_xy_radius = 5.0 * self.scale
        self.transport_radius = 10.0 * self.scale
        self.place_selection_margin = 0.75 * self.scale
        self.minimum_improvement = 0.25 * self.scale
        self.button_radius = 4.0 * self.scale
        self.button_descent_height = 4.0 * self.scale
        self.button_wrong_path_fraction = 0.25
        self._button_progress_state = _ButtonProgressState()

    def _button_phase(
        self,
        before: SceneState,
        after: SceneState,
    ) -> list[CandidateSignal]:
        return _expected_button_phase_signals(
            before,
            after,
            state=self._button_progress_state,
            button_radius=self.button_radius,
            button_descent_height=self.button_descent_height,
            selection_margin=self.pickup_selection_margin,
            minimum_approach=self.minimum_approach,
            wrong_path_fraction=self.button_wrong_path_fraction,
            prerequisite_complete=bool(
                after.metrics.get("button_prerequisite_complete", False)
            ),
        )

    def _pickup_phase(
        self, before: SceneState, after: SceneState, guard_button: bool
    ) -> list[CandidateSignal]:
        signals = self.pickup(
            before,
            after,
            intent_radius=self.pickup_radius,
            selection_margin=self.pickup_selection_margin,
            minimum_approach=self.minimum_approach,
        )
        if guard_button:
            signals.extend(
                self.premature_button(
                    before,
                    after,
                    intent_radius=self.button_radius,
                    minimum_approach=self.minimum_approach,
                    descent_height=self.button_descent_height,
                )
            )
        return signals

    def _video_pickup_phase(
        self,
        before: SceneState,
        after: SceneState,
    ) -> list[CandidateSignal]:
        return self._pickup_phase(before, after, False)

    def _place_phase(
        self, before: SceneState, after: SceneState, guard_button: bool
    ) -> list[CandidateSignal]:
        signals = self.place(
            before,
            after,
            release_radius=self.release_radius,
            transport_radius=self.transport_radius,
            selection_margin=self.place_selection_margin,
            minimum_motion=self.minimum_approach,
            minimum_improvement=self.minimum_improvement,
        )
        if guard_button:
            signals.extend(
                self.premature_button(
                    before,
                    after,
                    intent_radius=self.button_radius,
                    minimum_approach=self.minimum_approach,
                    descent_height=self.button_descent_height,
                )
            )
        return signals

    def _putdown_phase(
        self,
        before: SceneState,
        after: SceneState,
        guard_button: bool,
        detect_incomplete_release: bool = True,
        detect_held_loss: bool = True,
    ) -> list[CandidateSignal]:
        expected = tuple(after.metrics.get("expected_actor_names", ()))
        signals: list[CandidateSignal] = []
        if guard_button:
            signals.extend(
                self.premature_button(
                    before,
                    after,
                    intent_radius=self.button_radius,
                    minimum_approach=self.minimum_approach,
                    descent_height=self.button_descent_height,
                )
            )
        if detect_held_loss:
            signals.extend(
                (
                    held_identity_signal(after, expected_actors=expected),
                    required_held_signal(before, after, expected_actors=expected),
                )
            )
        intents = action_intents(before, after)
        released_without_completion = bool(
            detect_incomplete_release
            and before.held_actor in expected
            and after.held_actor is None
            and intents.opening
        )
        signals.append(
            semantic_signal(
                active=released_without_completion,
                reason_code="putdown.release_without_subtask_completion",
                recovery_mode=RecoveryMode.EXPERT_CONTINUE,
                expected_actor=before.held_actor,
                observed_actor=None,
                metrics={
                    "released_actor": before.held_actor,
                    "subtask_index": after.subtask.index,
                    "opening": intents.opening,
                },
                thresholds={"subtask_must_advance_after_release": True},
                priority=93,
            )
        )
        return signals

    def _video_place_phase(
        self,
        before: SceneState,
        after: SceneState,
    ) -> list[CandidateSignal]:
        signals = self._place_phase(before, after, False)
        expected_actors = tuple(after.metrics.get("expected_actor_names", ()))
        expected_targets = tuple(after.metrics.get("expected_target_names", ()))
        held_actor = after.held_actor
        if (
            held_actor not in expected_actors
            or held_actor not in after.actors
            or len(expected_targets) != 1
            or expected_targets[0] not in after.landmarks
        ):
            return signals
        intents = action_intents(before, after)
        actor_position = after.actors[held_actor].position
        target_name = expected_targets[0]
        target_position = after.landmarks[target_name]
        target_xy_distance = float(
            np.linalg.norm(actor_position[:2] - target_position[:2])
        )
        target_z_clearance = float(actor_position[2] - target_position[2])
        video_release_xy_radius = min(
            self.release_xy_radius,
            3.0 * self.scale,
        )
        misaligned_low_descent = bool(
            intents.descending
            and target_xy_distance > video_release_xy_radius
            and target_z_clearance <= self.pickup_descent_height
        )
        signals.append(
            semantic_signal(
                active=misaligned_low_descent,
                reason_code="place.expected_target_not_aligned_descent",
                recovery_mode=RecoveryMode.LIFT_HOLD_THEN_PLACE,
                expected_actor=held_actor,
                observed_actor=target_name,
                metrics={
                    "held_actor": held_actor,
                    "expected_target": target_name,
                    "target_xy_distance": target_xy_distance,
                    "target_z_clearance": target_z_clearance,
                    "descending": intents.descending,
                },
                thresholds={
                    "release_xy_radius": video_release_xy_radius,
                    "place_descent_height": self.pickup_descent_height,
                },
                priority=96,
            )
        )
        return signals


class PickHighlightDetector(_PickupPlaceButtonDetector):
    task_name = "PickHighlight"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_pickup_place_thresholds()

    def _pickup_remaining_target(
        self,
        before: SceneState,
        after: SceneState,
    ) -> list[CandidateSignal]:
        """Task-owned OR: any not-yet-picked highlighted cube is correct."""
        remaining = [
            str(name)
            for name in after.metrics.get("expected_actor_names", ())
            if str(name) in after.actors
        ]
        if not remaining:
            return []
        expected_actor = min(
            remaining,
            key=lambda name: float(
                np.linalg.norm(
                    after.tcp_position[:2] - after.actors[name].position[:2]
                )
            ),
        )
        return pickup_actor_signals(
            before,
            after,
            expected_actor=expected_actor,
            expected_actor_kind="cube",
            forbidden_actors=tuple(
                after.metrics.get("forbidden_actor_names", ())
            ),
            intent_radius=self.pickup_radius,
            selection_margin=self.pickup_selection_margin,
            minimum_approach_delta=self.minimum_approach,
            missed_grasp_radius=self.pickup_radius,
            expected_xy_radius=self.pickup_xy_alignment_radius,
            expected_descent_height=self.pickup_descent_height,
            closed_gripper_opening=self.closed_gripper_opening,
        )

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        phase = after.subtask.phase
        expected = tuple(after.metrics.get("expected_actor_names", ()))
        if phase == "press_button":
            return self._button_phase(before, after)
        if phase == "pickup":
            return [
                *self.premature_button(
                    before,
                    after,
                    intent_radius=self.button_radius,
                    minimum_approach=self.minimum_approach,
                    descent_height=self.button_descent_height,
                ),
                *self._pickup_remaining_target(before, after),
                held_identity_signal(after, expected_actors=expected),
            ]
        if phase == "putdown":
            return [
                *self.premature_button(
                    before,
                    after,
                    intent_radius=self.button_radius,
                    minimum_approach=self.minimum_approach,
                    descent_height=self.button_descent_height,
                ),
                held_identity_signal(after, expected_actors=expected),
                required_held_signal(before, after, expected_actors=expected),
            ]
        return []


class SwingXtimesDetector(_PickupPlaceButtonDetector):
    task_name = "SwingXtimes"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_pickup_place_thresholds()

    def _extra_landing_signals(
        self,
        before: SceneState,
        after: SceneState,
        target_names: Sequence[str],
    ) -> list[CandidateSignal]:
        expected_actors = tuple(after.metrics.get("expected_actor_names", ()))
        if len(expected_actors) != 1:
            return []
        actor_name = expected_actors[0]
        if actor_name not in before.actors or actor_name not in after.actors:
            return []
        native_target_xy_radius = 1.5 * self.scale
        native_target_z_height = 6.0 * self.scale
        warning_xy_radius = 2.0 * native_target_xy_radius
        warning_z_height = native_target_z_height + warning_xy_radius
        signals: list[CandidateSignal] = []
        for target_name in target_names:
            if target_name not in after.landmarks:
                continue
            target = after.landmarks[target_name]
            before_xy_distance = float(
                np.linalg.norm(
                    before.actors[actor_name].position[:2] - target[:2]
                )
            )
            after_position = after.actors[actor_name].position
            after_xy_distance = float(
                np.linalg.norm(after_position[:2] - target[:2])
            )
            target_latched = bool(
                after.metrics["on_right_latched"]
                if target_name == "target_right"
                else after.metrics["on_left_latched"]
            )
            approaching = bool(
                int(after.metrics["swing_count"])
                >= int(after.metrics["maximum_swings"])
                and after.held_actor == actor_name
                and not target_latched
                and after_position[2] <= warning_z_height
                and after_xy_distance <= warning_xy_radius
                and after_xy_distance < before_xy_distance
            )
            signals.append(
                semantic_signal(
                    active=approaching,
                    reason_code="swing.extra_landing_risk",
                    recovery_mode=RecoveryMode.LIFT_HOLD_THEN_PLACE,
                    expected_actor=actor_name,
                    observed_actor=target_name,
                    metrics={
                        "swing_count": int(after.metrics["swing_count"]),
                        "maximum_swings": int(after.metrics["maximum_swings"]),
                        "before_target_xy_distance": before_xy_distance,
                        "after_target_xy_distance": after_xy_distance,
                        "actor_z_height": float(after_position[2]),
                        "target_latched": target_latched,
                    },
                    thresholds={
                        "native_target_xy_radius": native_target_xy_radius,
                        "native_target_z_height": native_target_z_height,
                        "warning_xy_radius": warning_xy_radius,
                        "warning_z_height": warning_z_height,
                    },
                    priority=98,
                )
            )
        return signals

    def _button_contact_signals(
        self,
        before: SceneState,
        after: SceneState,
    ) -> list[CandidateSignal]:
        actor_name = after.held_actor
        if (
            actor_name is None
            or actor_name not in before.actors
            or actor_name not in after.actors
            or "button_cap" not in after.landmarks
        ):
            return []
        before_position = before.actors[actor_name].position
        after_position = after.actors[actor_name].position
        cap_position = after.landmarks["button_cap"]
        xy_distance = float(np.linalg.norm(after_position[:2] - cap_position[:2]))
        contact_xy_radius = float(after.metrics["button_contact_xy_radius"])
        contact_center_height = float(
            after.metrics["button_contact_center_height"]
        )
        descending = bool(after_position[2] < before_position[2])
        low_contact_corridor = bool(
            descending
            and xy_distance <= contact_xy_radius
            and after_position[2] <= contact_center_height
        )
        return [
            semantic_signal(
                active=low_contact_corridor,
                reason_code="button.prerequisite_incomplete",
                recovery_mode=RecoveryMode.LIFT_HOLD_THEN_PLACE,
                expected_actor="button",
                observed_actor=actor_name,
                metrics={
                    "held_actor": actor_name,
                    "button_xy_distance": xy_distance,
                    "actor_center_z": float(after_position[2]),
                    "button_depth": float(after.metrics["button_depth"]),
                    "descending": descending,
                },
                thresholds={
                    "button_contact_xy_radius": contact_xy_radius,
                    "button_contact_center_height": contact_center_height,
                    "official_button_depth_threshold": float(
                        after.metrics["official_button_depth_threshold"]
                    ),
                },
                priority=99,
            )
        ]

    def evaluate_transition(
        self,
        before: SceneState,
        action: np.ndarray,
        after: SceneState,
    ) -> Sequence[CandidateSignal]:
        del action
        return [
            *self._extra_landing_signals(
                before,
                after,
                ("target_right", "target_left"),
            ),
            *self._button_contact_signals(before, after),
        ]

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        phase = after.subtask.phase
        if phase == "pickup":
            return self._pickup_phase(before, after, True)
        if phase == "swing":
            return [
                *self._place_phase(before, after, False),
                *self._extra_landing_signals(
                    before,
                    after,
                    tuple(after.metrics.get("expected_target_names", ())),
                ),
                *self._button_contact_signals(before, after),
            ]
        if phase == "putdown":
            return [
                *self._putdown_phase(
                    before,
                    after,
                    False,
                    detect_incomplete_release=False,
                    detect_held_loss=False,
                ),
                *self._extra_landing_signals(
                    before,
                    after,
                    ("target_right", "target_left"),
                ),
                *self._button_contact_signals(before, after),
            ]
        return []


class PickXtimesDetector(_PickupPlaceButtonDetector):
    task_name = "PickXtimes"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_pickup_place_thresholds()

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        phase = after.subtask.phase
        if phase == "pickup":
            return self._pickup_phase(before, after, True)
        if phase == "place":
            return self._place_phase(before, after, True)
        if phase == "press_button":
            return self._button_phase(before, after)
        return []


class VideoRepickDetector(_PickupPlaceButtonDetector):
    task_name = "VideoRepick"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_pickup_place_thresholds()

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        phase = after.subtask.phase
        if phase == "pickup":
            return self._pickup_phase(before, after, True)
        if phase == "putdown":
            return self._putdown_phase(before, after, True)
        if phase == "press_button":
            return self._button_phase(before, after)
        return []


class VideoPlaceButtonDetector(_PickupPlaceButtonDetector):
    task_name = "VideoPlaceButton"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_pickup_place_thresholds()

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        if after.subtask.phase == "pickup":
            return self._video_pickup_phase(before, after)
        if after.subtask.phase == "place":
            return self._video_place_phase(before, after)
        return []


class VideoPlaceOrderDetector(_PickupPlaceButtonDetector):
    task_name = "VideoPlaceOrder"

    def __init__(self, raw_env: Any) -> None:
        super().__init__(raw_env)
        self._initialize_pickup_place_thresholds()

    def evaluate(
        self, before: SceneState, action: np.ndarray, after: SceneState
    ) -> Sequence[CandidateSignal]:
        del action
        if after.subtask.phase == "pickup":
            return self._video_pickup_phase(before, after)
        if after.subtask.phase == "place":
            return self._video_place_phase(before, after)
        return []


TASK_DETECTORS = {
    detector.task_name: detector
    for detector in (
        BinFillDetector,
        ButtonUnmaskDetector,
        ButtonUnmaskSwapDetector,
        InsertPegDetector,
        MoveCubeDetector,
        PatternLockDetector,
        PickHighlightDetector,
        PickXtimesDetector,
        RouteStickDetector,
        StopCubeDetector,
        SwingXtimesDetector,
        VideoPlaceButtonDetector,
        VideoPlaceOrderDetector,
        VideoRepickDetector,
        VideoUnmaskDetector,
        VideoUnmaskSwapDetector,
    )
}


def build_task_detector(task_name: str, raw_env: Any) -> TaskPostActionDetector:
    if task_name not in TASK_DETECTORS:
        raise KeyError(f"no task-owned post-action detector for {task_name}")
    return TASK_DETECTORS[task_name](raw_env)

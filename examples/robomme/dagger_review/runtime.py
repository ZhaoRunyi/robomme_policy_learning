"""dagger_review / runtime: shared behavior owned by this layer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
import numpy as np
from .live import (
    LiveDaggerController,
    LiveHarnessContractError,
    PolicyContinuationBridge,
    PolicyContinuationSnapshot,
    ReasonPersistenceBridge,
    SolverActionRouter,
    build_live_controller,
)
from .live import CounterfactualGateResult
from .signals import (
    AdapterContractError,
    ButtonIntentSpec,
    CandidateSignal,
    ContainerIntentSpec,
    ForbiddenPickupSpec,
    GraspedLiftIntentSpec,
    InsertionIntentSpec,
    InterventionDecision,
    PickupIntentSpec,
    PlaceIntentSpec,
    PreActionContext,
    PredictedAction,
    RecoveryMode,
    RouteProgressSpec,
    SignalClass,
    TimedProgressSpec,
    button_intent_signals,
    container_intent_signals,
    forbidden_pickup_signals,
    grasped_lift_intent_signals,
    insertion_intent_signals,
    pickup_intent_signals,
    place_intent_signals,
    route_progress_signals,
    timed_progress_signals,
)
from .recovery import ExpertSolverRole, LifecycleState, RecoveryContext
from .tasks import (
    RoboMMETaskAdapter,
    build_robomme_task_adapter,
)
from .signals import SceneState
from .live import _json_value
from .tasks import build_task_detector


Array = np.ndarray
_PLANNER_EVENT_LOCK = threading.Lock()
# RoboMME ``is_obj_pickup`` uses ``obj.pose.p[:, 2] > 0.05`` for every
# MoveCube pickup success/failure predicate.
MOVECUBE_NATIVE_PICKUP_HEIGHT_METERS = 0.05


class ComponentFactoryContractError(LiveHarnessContractError):
    """Raised when live factory dependencies are absent or ambiguous."""


class StrictCounterfactualProvider(Protocol):
    """Own all normal three-lane steps and evaluate the frozen branch pair."""

    def advance_common_policy_action(
        self,
        action: Array,
    ) -> tuple[Any, bool, str]:
        ...

    def evaluate(
        self,
        decision: InterventionDecision,
        action: Array,
        policy_continuation: PolicyContinuationSnapshot,
    ) -> CounterfactualGateResult:
        ...


@dataclasses.dataclass(frozen=True)
class RuntimeCalibration:
    """Named smoke calibration derived from native geometry and solvers.

    Detector distances are scaled from ``raw_env.cube_half_size``.  Recovery
    height 0.20 m matches the native pickup/peg solvers.  StopCube timing uses
    the measured p95 visible descent of 20 frames at 30 Hz; it must still pass
    the strict live-pair gate before a trajectory is accepted.
    """

    step_duration_seconds: float = 1.0 / 30.0
    pickup_intent_radius_half_sizes: float = 4.0
    pickup_selection_margin_half_sizes: float = 0.75
    minimum_approach_half_sizes: float = 0.05
    insertion_intent_radius_half_sizes: float = 5.0
    insertion_side_margin_half_sizes: float = 0.5
    safe_clearance_z: float = 0.20
    detector_activate_after: int = 2
    open_gripper_steps: int = 10
    maximum_prefix_path_points: int = 160
    stopcube_expert_completion_seconds: float = 20.0 / 30.0
    stopcube_early_margin_seconds: float = 3.0 / 30.0
    stopcube_late_margin_seconds: float = 0.0
    stopcube_minimum_ready_progress: float = 0.05

    # Optional path assessment and its diagnostic output.
    reject_paths: bool = False
    planner_telemetry: Path | None = None

    def __post_init__(self) -> None:
        finite_positive = (
            self.step_duration_seconds,
            self.pickup_intent_radius_half_sizes,
            self.pickup_selection_margin_half_sizes,
            self.minimum_approach_half_sizes,
            self.insertion_intent_radius_half_sizes,
            self.insertion_side_margin_half_sizes,
            self.safe_clearance_z,
            self.stopcube_expert_completion_seconds,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
            raise ComponentFactoryContractError(
                "runtime calibration values must be positive and finite"
            )
        if self.detector_activate_after < 1:
            raise ComponentFactoryContractError(
                "detector_activate_after must be positive"
            )
        if self.open_gripper_steps < 1:
            raise ComponentFactoryContractError(
                "open_gripper_steps must be positive"
            )
        if self.maximum_prefix_path_points < 1:
            raise ComponentFactoryContractError(
                "maximum_prefix_path_points must be positive"
            )


@dataclasses.dataclass(frozen=True)
class TaskDetectorCalibration:
    """Task-geometry thresholds measured in metres, never guessed from prompts.

    The original four representative tasks retain their native
    ``cube_half_size`` calibration.  Every additional task must receive this
    explicit structure from a live calibration artifact.  Only fields used by
    that task are required; absent or non-positive required values fail during
    factory construction rather than silently falling back to cube geometry.
    """

    pickup_intent_radius: float | None = None
    pickup_selection_margin: float | None = None
    minimum_approach_delta: float | None = None
    container_intent_radius: float | None = None
    container_selection_margin: float | None = None
    button_intent_radius: float | None = None
    button_selection_margin: float | None = None
    place_release_radius: float | None = None
    place_transport_radius: float | None = None
    place_selection_margin: float | None = None
    place_minimum_improvement: float | None = None
    route_target_radius: float | None = None
    route_wrong_waypoint_radius: float | None = None
    route_maximum_cross_track: float | None = None
    route_minimum_improvement_per_second: float | None = None
    route_trend_window: int | None = None
    route_direction_side_margin: float | None = None
    route_positive_cross_direction: str | None = None


@dataclasses.dataclass(frozen=True)
class TaskEvidenceCapabilities:
    """Explicit launcher-installed semantic evidence used by detector rules.

    These are capabilities, not defaults.  A task that needs missed-grasp or
    semantic-order evidence cannot be collected until the live adapter/overlay
    emits the named metric on every relevant phase.
    """

    pickup_attempt_complete_metric: str | None = None
    remaining_set_failure_overlay_metric: str | None = None
    semantic_solver_valid_metric: str | None = None
    pickup_attempt_complete_provider: Callable[[Any, Any], bool] | None = None
    remaining_set_failure_overlay_provider: Callable[[Any, Any], bool] | None = None
    semantic_solver_valid_provider: Callable[[Any, Any], bool] | None = None


@dataclasses.dataclass(frozen=True)
class TriggerContract:
    reason_code: str
    recovery_mode: RecoveryMode
    expected_official_failure_codes: frozenset[str]


@dataclasses.dataclass(frozen=True)
class TaskInterventionContract:
    triggers: tuple[TriggerContract, ...]
    required_live_calibration: tuple[str, ...]
    required_live_evidence: tuple[str, ...] = ()
    unresolved_live_evidence: tuple[str, ...] = ()


_PICKUP_FIELDS = (
    "pickup_intent_radius",
    "pickup_selection_margin",
    "minimum_approach_delta",
)
_CONTAINER_FIELDS = (
    "container_intent_radius",
    "container_selection_margin",
    "minimum_approach_delta",
)
_BUTTON_FIELDS = (
    "button_intent_radius",
    "button_selection_margin",
    "minimum_approach_delta",
)
_PLACE_FIELDS = (
    "place_release_radius",
    "place_transport_radius",
    "place_selection_margin",
    "place_minimum_improvement",
    "minimum_approach_delta",
)
_ROUTE_FIELDS = (
    "route_target_radius",
    "route_wrong_waypoint_radius",
    "route_maximum_cross_track",
    "route_minimum_improvement_per_second",
    "route_trend_window",
)


def _trigger(
    reason_code: str,
    recovery_mode: RecoveryMode,
    *failure_codes: str,
) -> TriggerContract:
    return TriggerContract(
        reason_code=reason_code,
        recovery_mode=recovery_mode,
        expected_official_failure_codes=frozenset(failure_codes),
    )


_WRONG_PICKUP = _trigger(
    "pickup.wrong_actor_precontact",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.subtask.non_target_pickup",
)
_WRONG_HELD = _trigger(
    "identity.wrong_actor_held",
    RecoveryMode.RETURN_WRONG_OBJECT,
    "robomme.subtask.non_target_pickup",
)
_WRONG_CONTAINER = _trigger(
    "container.wrong_container_precontact",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.subtask.wrong_container_pickup",
)
_WRONG_CONTAINER_HELD = _trigger(
    "identity.wrong_actor_held",
    RecoveryMode.RETURN_WRONG_OBJECT,
    "robomme.subtask.wrong_container_pickup",
)
_WRONG_PLACE_RELEASE = _trigger(
    "place.wrong_target_prerelease",
    RecoveryMode.LIFT_HOLD_THEN_PLACE,
    "robomme.subtask.wrong_target_drop",
)
_WRONG_PLACE_TRANSPORT = _trigger(
    "place.wrong_target_transport",
    RecoveryMode.LIFT_HOLD_THEN_PLACE,
    "robomme.subtask.wrong_target_drop",
)
_PLACE_NOT_PROGRESSING = _trigger(
    "place.not_approaching_expected_target",
    RecoveryMode.LIFT_HOLD_THEN_PLACE,
    "robomme.subtask.wrong_target_drop",
    "robomme.episode.timeout",
)
_PREMATURE_BUTTON = _trigger(
    "button.wrong_button_prepress",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.subtask.premature_button",
)
_BUTTON_PREREQUISITE = _trigger(
    "button.prerequisite_incomplete_prepress",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.subtask.premature_button",
)
_EXTRA_PICKUP = _trigger(
    "cycle.extra_pickup_precontact",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.subtask.extra_pickup",
    "robomme.episode.timeout",
)
_REQUIRED_ACTOR_MISSING = _trigger(
    "identity.required_actor_not_held",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.subtask.missed_pickup",
    "robomme.episode.timeout",
)
_MISSED_PICKUP = _trigger(
    "pickup.missed_grasp",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.subtask.missed_pickup",
    "robomme.episode.timeout",
)
_MISSED_CONTAINER_PICKUP = _trigger(
    "container.missed_pickup",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.subtask.missed_pickup",
    "robomme.episode.timeout",
)
_ROUTE_WRONG_WAYPOINT = _trigger(
    "route.wrong_waypoint_approach",
    RecoveryMode.RETURN_TO_SAFE_ANCHOR,
    "robomme.route.wrong_waypoint",
)
_ROUTE_NO_PROGRESS = _trigger(
    "route.no_expected_progress",
    RecoveryMode.RETURN_TO_SAFE_ANCHOR,
    "robomme.route.wrong_waypoint",
    "robomme.route.direction_mismatch",
    "robomme.episode.timeout",
)
_ROUTE_CORRIDOR = _trigger(
    "route.corridor_deviation",
    RecoveryMode.RETURN_TO_SAFE_ANCHOR,
    "robomme.route.wrong_waypoint",
    "robomme.route.direction_mismatch",
)
_ROUTE_DIRECTION = _trigger(
    "route.direction_side_mismatch",
    RecoveryMode.RETURN_TO_SAFE_ANCHOR,
    "robomme.route.direction_mismatch",
)

_MOVECUBE_WRONG_PICKUP = _trigger(
    "pickup.wrong_actor_precontact",
    RecoveryMode.OPEN_AND_RETRACT,
    "robomme.movecube.peg_push.cube_pickup",
    "robomme.movecube.grasp_putdown.peg_pickup",
)
_MOVECUBE_FORBIDDEN_PUSH_PICKUP = _trigger(
    "movecube.forbidden_pickup_before_push",
    RecoveryMode.CANCEL_FORBIDDEN_PUSH_PICKUP,
    "robomme.movecube.gripper_push.cube_pickup",
    "robomme.movecube.gripper_push.peg_pickup",
)


def _contract(
    triggers: Sequence[TriggerContract],
    required_fields: Sequence[str],
    required_evidence: Sequence[str] = (),
    unresolved: Sequence[str] = (),
) -> TaskInterventionContract:
    return TaskInterventionContract(
        triggers=tuple(triggers),
        required_live_calibration=tuple(dict.fromkeys(required_fields)),
        required_live_evidence=tuple(dict.fromkeys(required_evidence)),
        unresolved_live_evidence=tuple(unresolved),
    )


TASK_INTERVENTION_CONTRACTS: Mapping[str, TaskInterventionContract] = {
    "MoveCube": _contract(
        (
            _MOVECUBE_WRONG_PICKUP,
            _MOVECUBE_FORBIDDEN_PUSH_PICKUP,
            _trigger(
                "movecube.wrong_tool_held_before_hook",
                RecoveryMode.RETURN_WRONG_OBJECT,
                "robomme.movecube.peg_push.cube_pickup",
            ),
            _trigger(
                "movecube.required_tool_not_held",
                RecoveryMode.OPEN_AND_RETRACT,
                "robomme.episode.timeout",
            ),
            _trigger(
                "movecube.wrong_actor_held_before_place",
                RecoveryMode.RETURN_WRONG_OBJECT,
                "robomme.movecube.grasp_putdown.peg_pickup",
            ),
            _trigger(
                "place.wrong_target_prerelease",
                RecoveryMode.LIFT_HOLD_THEN_PLACE,
                "robomme.episode.timeout",
            ),
            _trigger(
                "place.wrong_target_transport",
                RecoveryMode.LIFT_HOLD_THEN_PLACE,
                "robomme.episode.timeout",
            ),
        ),
        (),
    ),
    "InsertPeg": _contract(
        (
            _WRONG_PICKUP,
            _trigger(
                "insertion.wrong_end_approach",
                RecoveryMode.RETURN_TO_SAFE_ANCHOR,
                "robomme.subtask.wrong_peg_end",
            ),
            _trigger(
                "insertion.wrong_side_approach",
                RecoveryMode.RETURN_TO_SAFE_ANCHOR,
                "robomme.subtask.wrong_insert_side",
            ),
        ),
        (),
    ),
    "BinFill": _contract(
        (
            _WRONG_PICKUP,
            _trigger(
                "binfill.wrong_cube_held_before_bin_drop",
                RecoveryMode.RETURN_WRONG_OBJECT,
                "robomme.subtask.wrong_bin_contents",
            ),
            _BUTTON_PREREQUISITE,
        ),
        (),
    ),
    "StopCube": _contract(
        (
            _trigger(
                "timed_progress.premature_press_motion",
                RecoveryMode.HOLD_POSE,
                "robomme.subtask.premature_button",
            ),
            _trigger(
                "timed_progress.latest_safe_takeover",
                RecoveryMode.EXPERT_CONTINUE,
                "robomme.subtask.missed_timed_press",
            ),
        ),
        (),
    ),
    "PickHighlight": _contract(
        (
            _WRONG_PICKUP,
            _trigger(
                "pickup.missed_grasp",
                RecoveryMode.OPEN_AND_RETRACT,
                "robomme.subtask.highlight_count_mismatch",
                "robomme.episode.timeout",
            ),
            _WRONG_HELD,
            _trigger(
                "identity.required_actor_not_held",
                RecoveryMode.OPEN_AND_RETRACT,
                "robomme.subtask.highlight_count_mismatch",
                "robomme.episode.timeout",
            ),
            _PREMATURE_BUTTON,
            _EXTRA_PICKUP,
        ),
        (*_PICKUP_FIELDS, *_BUTTON_FIELDS),
        (
            "pickup_attempt_complete_metric",
            "remaining_set_failure_overlay_metric",
            "semantic_solver_valid_metric",
        ),
    ),
    "ButtonUnmask": _contract(
        (
            _WRONG_CONTAINER,
            _MISSED_CONTAINER_PICKUP,
            _WRONG_CONTAINER_HELD,
            _REQUIRED_ACTOR_MISSING,
            _PREMATURE_BUTTON,
            _EXTRA_PICKUP,
        ),
        (*_CONTAINER_FIELDS, *_BUTTON_FIELDS),
        ("pickup_attempt_complete_metric",),
    ),
    "ButtonUnmaskSwap": _contract(
        (
            _WRONG_CONTAINER,
            _MISSED_CONTAINER_PICKUP,
            _WRONG_CONTAINER_HELD,
            _REQUIRED_ACTOR_MISSING,
            _PREMATURE_BUTTON,
            _EXTRA_PICKUP,
        ),
        (*_CONTAINER_FIELDS, *_BUTTON_FIELDS),
        (
            "pickup_attempt_complete_metric",
            "semantic_solver_valid_metric",
        ),
    ),
    "VideoUnmask": _contract(
        (
            _WRONG_CONTAINER,
            _MISSED_CONTAINER_PICKUP,
            _WRONG_CONTAINER_HELD,
            _REQUIRED_ACTOR_MISSING,
        ),
        _CONTAINER_FIELDS,
        ("pickup_attempt_complete_metric",),
    ),
    "VideoUnmaskSwap": _contract(
        (
            _WRONG_CONTAINER,
            _MISSED_CONTAINER_PICKUP,
            _WRONG_CONTAINER_HELD,
            _REQUIRED_ACTOR_MISSING,
        ),
        _CONTAINER_FIELDS,
        ("pickup_attempt_complete_metric",),
    ),
    "PatternLock": _contract(
        (
            _trigger(
                "route.wrong_waypoint_approach",
                RecoveryMode.RETURN_TO_SAFE_ANCHOR,
                "robomme.route.wrong_waypoint",
                "robomme.route.sequence_mismatch",
            ),
            _trigger(
                "route.no_expected_progress",
                RecoveryMode.RETURN_TO_SAFE_ANCHOR,
                "robomme.route.wrong_waypoint",
                "robomme.route.sequence_mismatch",
                "robomme.episode.timeout",
            ),
            _trigger(
                "route.corridor_deviation",
                RecoveryMode.RETURN_TO_SAFE_ANCHOR,
                "robomme.route.wrong_waypoint",
                "robomme.route.sequence_mismatch",
            ),
        ),
        _ROUTE_FIELDS,
    ),
    "RouteStick": _contract(
        (
            _ROUTE_WRONG_WAYPOINT,
            _ROUTE_NO_PROGRESS,
            _ROUTE_CORRIDOR,
            _ROUTE_DIRECTION,
        ),
        (
            *_ROUTE_FIELDS,
            "route_direction_side_margin",
            "route_positive_cross_direction",
        ),
    ),
    "SwingXtimes": _contract(
        (
            _WRONG_PICKUP,
            _MISSED_PICKUP,
            _WRONG_HELD,
            _REQUIRED_ACTOR_MISSING,
            _WRONG_PLACE_RELEASE,
            _WRONG_PLACE_TRANSPORT,
            _PLACE_NOT_PROGRESSING,
            _PREMATURE_BUTTON,
        ),
        (*_PICKUP_FIELDS, *_PLACE_FIELDS, *_BUTTON_FIELDS),
        ("pickup_attempt_complete_metric",),
    ),
    "PickXtimes": _contract(
        (
            _WRONG_PICKUP,
            _MISSED_PICKUP,
            _WRONG_HELD,
            _REQUIRED_ACTOR_MISSING,
            _WRONG_PLACE_RELEASE,
            _PLACE_NOT_PROGRESSING,
            _PREMATURE_BUTTON,
            _BUTTON_PREREQUISITE,
            _EXTRA_PICKUP,
        ),
        (*_PICKUP_FIELDS, *_PLACE_FIELDS, *_BUTTON_FIELDS),
        ("pickup_attempt_complete_metric",),
    ),
    "VideoRepick": _contract(
        (
            _WRONG_PICKUP,
            _MISSED_PICKUP,
            _WRONG_HELD,
            _REQUIRED_ACTOR_MISSING,
            _PREMATURE_BUTTON,
            _BUTTON_PREREQUISITE,
            _EXTRA_PICKUP,
        ),
        (*_PICKUP_FIELDS, *_BUTTON_FIELDS),
        ("pickup_attempt_complete_metric",),
    ),
    "VideoPlaceButton": _contract(
        (
            _WRONG_PICKUP,
            _MISSED_PICKUP,
            _WRONG_HELD,
            _REQUIRED_ACTOR_MISSING,
            _WRONG_PLACE_RELEASE,
            _WRONG_PLACE_TRANSPORT,
            _PLACE_NOT_PROGRESSING,
        ),
        (*_PICKUP_FIELDS, *_PLACE_FIELDS),
        ("pickup_attempt_complete_metric",),
    ),
    "VideoPlaceOrder": _contract(
        (
            _WRONG_PICKUP,
            _MISSED_PICKUP,
            _WRONG_HELD,
            _REQUIRED_ACTOR_MISSING,
            _WRONG_PLACE_RELEASE,
            _WRONG_PLACE_TRANSPORT,
            _PLACE_NOT_PROGRESSING,
        ),
        (*_PICKUP_FIELDS, *_PLACE_FIELDS),
        ("pickup_attempt_complete_metric",),
    ),
}


CANONICAL_FAILURE_SOURCES: Mapping[str, str] = {
    "robomme.movecube.peg_push.cube_pickup": (
        "MoveCube.py peg_push pickup failure_func[0]: is_obj_pickup(cube)"
    ),
    "robomme.movecube.peg_push.cube_already_at_goal": (
        "MoveCube.py peg_push pickup failure_func[1]: "
        "is_obj_pushed_onto(cube, goal_site)"
    ),
    "robomme.movecube.gripper_push.cube_pickup": (
        "MoveCube.py gripper_push failure_func[0]: is_obj_pickup(cube)"
    ),
    "robomme.movecube.gripper_push.peg_pickup": (
        "MoveCube.py gripper_push failure_func[1:3]: is_obj_pickup(peg end)"
    ),
    "robomme.movecube.grasp_putdown.cube_already_at_goal": (
        "MoveCube.py grasp_putdown pickup failure_func[0]: "
        "is_obj_pushed_onto(cube, goal_site)"
    ),
    "robomme.movecube.grasp_putdown.peg_pickup": (
        "MoveCube.py grasp_putdown pickup failure_func[1:3]: "
        "is_obj_pickup(peg end)"
    ),
    "robomme.subtask.non_target_pickup": (
        "native failure_func identity predicate is_any_obj_pickup"
    ),
    "robomme.subtask.wrong_container_pickup": (
        "native failure_func identity predicate is_any_bin_pickup"
    ),
    "robomme.subtask.wrong_target_drop": (
        "native failure_func identity predicate is_obj_dropped_onto_any"
    ),
    "robomme.subtask.premature_button": (
        "native phase failure_func predicate is_button_pressed"
    ),
    "robomme.subtask.extra_pickup": (
        "native final-button failure_func predicate is_any_obj_pickup"
    ),
    "robomme.subtask.missed_pickup": (
        "instrumented attempt boundary followed by native subtask noncompletion"
    ),
    "robomme.subtask.wrong_bin_contents": (
        "BinFill native bin-content prerequisite/final evaluator"
    ),
    "robomme.subtask.unexpected_extra_bin_drop": (
        "BinFill.py final-button failure_func[1]: one side-effectful fresh "
        "drop remained to be consumed at the button phase"
    ),
    "robomme.subtask.highlight_count_mismatch": (
        "PickHighlight.py terminal target_cube_pickup_counts verification"
    ),
    "robomme.subtask.wrong_peg_end": (
        "InsertPeg native endpoint identity evaluator"
    ),
    "robomme.subtask.wrong_insert_side": (
        "InsertPeg native direction/side evaluator"
    ),
    "robomme.subtask.missed_timed_press": (
        "StopCube native stop-window evaluator"
    ),
    "robomme.subtask.extra_swing": (
        "SwingXtimes native too_many_swings predicate"
    ),
    "robomme.route.wrong_waypoint": (
        "PatternLock/RouteStick native _wrong_button_touch predicate"
    ),
    "robomme.route.direction_mismatch": (
        "RouteStick native direction_fail trace predicate"
    ),
    "robomme.route.direction_unverifiable": (
        "RouteStick direction_fail malformed/degenerate/empty/on-line evidence"
    ),
    "robomme.route.sequence_mismatch": (
        "PatternLock.py terminal achieved_list versus selected_buttons verifier"
    ),
    "robomme.episode.timeout": (
        "official episode truncation while the named subtask remains incomplete"
    ),
}


@dataclasses.dataclass(frozen=True)
class SemanticFailureBinding:
    """One source-positioned element of a native ``failure_func`` result."""

    task_name: str
    predicate_index: int
    failure_code: str
    choice_label: str | None = None
    choice_label_prefix: str | None = None
    way: str | None = None
    demonstration: bool | None = None

    def matches(
        self,
        task_name: str,
        raw_env: Any,
        task_entry: Mapping[str, Any],
        predicate_index: int,
    ) -> bool:
        if self.task_name != task_name or self.predicate_index != predicate_index:
            return False
        choice_label = task_entry.get("choice_label")
        if self.choice_label is not None and choice_label != self.choice_label:
            return False
        if self.choice_label_prefix is not None and (
            not isinstance(choice_label, str)
            or not choice_label.startswith(self.choice_label_prefix)
        ):
            return False
        if self.way is not None and getattr(raw_env, "way", None) != self.way:
            return False
        if self.demonstration is not None and (
            bool(task_entry.get("demonstration")) != self.demonstration
        ):
            return False
        return True


def _failure_binding(
    task_name: str,
    choice_label: str | None,
    predicate_index: int,
    failure_code: str,
    *,
    choice_label_prefix: str | None = None,
    way: str | None = None,
    demonstration: bool | None = None,
) -> SemanticFailureBinding:
    return SemanticFailureBinding(
        task_name=task_name,
        choice_label=choice_label,
        choice_label_prefix=choice_label_prefix,
        predicate_index=predicate_index,
        failure_code=failure_code,
        way=way,
        demonstration=demonstration,
    )


SEMANTIC_FAILURE_BINDINGS: tuple[SemanticFailureBinding, ...] = (
    # MoveCube.evaluate() reconstructs these lambdas on every call.  The list
    # indices below are the source order in MoveCube.py, not inferred labels.
    _failure_binding(
        "MoveCube",
        "pick up the peg",
        0,
        "robomme.movecube.peg_push.cube_pickup",
        way="peg_push",
    ),
    _failure_binding(
        "MoveCube",
        "pick up the peg",
        1,
        "robomme.movecube.peg_push.cube_already_at_goal",
        way="peg_push",
    ),
    _failure_binding(
        "MoveCube",
        "close gripper and push the cube to the target",
        0,
        "robomme.movecube.gripper_push.cube_pickup",
        way="gripper_push",
    ),
    _failure_binding(
        "MoveCube",
        "close gripper and push the cube to the target",
        1,
        "robomme.movecube.gripper_push.peg_pickup",
        way="gripper_push",
    ),
    _failure_binding(
        "MoveCube",
        "close gripper and push the cube to the target",
        2,
        "robomme.movecube.gripper_push.peg_pickup",
        way="gripper_push",
    ),
    _failure_binding(
        "MoveCube",
        "pick up the cube",
        0,
        "robomme.movecube.grasp_putdown.cube_already_at_goal",
        way="grasp_putdown",
    ),
    _failure_binding(
        "MoveCube",
        "pick up the cube",
        1,
        "robomme.movecube.grasp_putdown.peg_pickup",
        way="grasp_putdown",
    ),
    _failure_binding(
        "MoveCube",
        "pick up the cube",
        2,
        "robomme.movecube.grasp_putdown.peg_pickup",
        way="grasp_putdown",
    ),
    _failure_binding(
        "BinFill",
        "pick up the cube",
        0,
        "robomme.subtask.premature_button",
    ),
    _failure_binding(
        "BinFill",
        "put it into the bin",
        0,
        "robomme.subtask.premature_button",
    ),
    _failure_binding(
        "BinFill",
        "press the button",
        0,
        "robomme.subtask.wrong_bin_contents",
    ),
    _failure_binding(
        "BinFill",
        "press the button",
        1,
        "robomme.subtask.unexpected_extra_bin_drop",
    ),
    _failure_binding(
        "InsertPeg",
        "pick up the peg by grasping one end",
        0,
        "robomme.subtask.wrong_peg_end",
    ),
    _failure_binding(
        "InsertPeg",
        "pick up the peg by grasping one end",
        1,
        "robomme.subtask.non_target_pickup",
        demonstration=False,
    ),
    _failure_binding(
        "InsertPeg",
        None,
        0,
        "robomme.subtask.wrong_peg_end",
        choice_label_prefix="insert the peg from the ",
        demonstration=False,
    ),
    _failure_binding(
        "InsertPeg",
        None,
        1,
        "robomme.subtask.wrong_insert_side",
        choice_label_prefix="insert the peg from the ",
        demonstration=False,
    ),
    _failure_binding(
        "InsertPeg",
        None,
        2,
        "robomme.subtask.non_target_pickup",
        choice_label_prefix="insert the peg from the ",
        demonstration=False,
    ),
    _failure_binding(
        "PickHighlight",
        "press button",
        0,
        "robomme.subtask.non_target_pickup",
    ),
    _failure_binding(
        "PickHighlight",
        "pick up the highlighted cube",
        0,
        "robomme.subtask.non_target_pickup",
    ),
    _failure_binding(
        "PickHighlight",
        "place the cube onto the table",
        0,
        "robomme.subtask.non_target_pickup",
    ),
    *(
        _failure_binding(
            task_name,
            choice_label,
            0,
            "robomme.subtask.wrong_container_pickup",
        )
        for task_name in (
            "ButtonUnmask",
            "ButtonUnmaskSwap",
            "VideoUnmask",
            "VideoUnmaskSwap",
        )
        for choice_label in (
            "pick up the container",
            "put down the container",
        )
    ),
    _failure_binding(
        "PatternLock",
        None,
        0,
        "robomme.route.wrong_waypoint",
    ),
    _failure_binding(
        "RouteStick",
        None,
        0,
        "robomme.route.wrong_waypoint",
    ),
    *(
        _failure_binding(
            "SwingXtimes",
            choice_label,
            0,
            "robomme.subtask.non_target_pickup",
        )
        for choice_label in (
            "pick up the cube",
            "move to the top of the target",
            "put the cube on the table",
            "press the button",
        )
    ),
    *(
        _failure_binding(
            "SwingXtimes",
            choice_label,
            1,
            "robomme.subtask.premature_button",
        )
        for choice_label in (
            "pick up the cube",
            "move to the top of the target",
            "put the cube on the table",
        )
    ),
    *(
        _failure_binding(
            "SwingXtimes",
            choice_label,
            2,
            "robomme.subtask.extra_swing",
        )
        for choice_label in (
            "pick up the cube",
            "move to the top of the target",
            "put the cube on the table",
        )
    ),
    _failure_binding(
        "SwingXtimes",
        "press the button",
        1,
        "robomme.subtask.extra_swing",
    ),
    *(
        _failure_binding(
            "PickXtimes",
            choice_label,
            0,
            "robomme.subtask.non_target_pickup",
        )
        for choice_label in (
            "pick up the cube",
            "place the cube onto the target",
        )
    ),
    *(
        _failure_binding(
            "PickXtimes",
            choice_label,
            1,
            "robomme.subtask.premature_button",
        )
        for choice_label in (
            "pick up the cube",
            "place the cube onto the target",
        )
    ),
    _failure_binding(
        "PickXtimes",
        "press the button to stop",
        0,
        "robomme.subtask.extra_pickup",
    ),
    *(
        _failure_binding(
            task_name,
            "pick up the cube",
            0,
            "robomme.subtask.non_target_pickup",
        )
        for task_name in ("VideoPlaceButton", "VideoPlaceOrder")
    ),
    *(
        _failure_binding(
            task_name,
            "drop onto",
            0,
            "robomme.subtask.wrong_target_drop",
        )
        for task_name in ("VideoPlaceButton", "VideoPlaceOrder")
    ),
    _failure_binding(
        "VideoRepick",
        "pick up the cube",
        0,
        "robomme.subtask.non_target_pickup",
    ),
    _failure_binding(
        "VideoRepick",
        "pick up the cube",
        1,
        "robomme.subtask.premature_button",
    ),
    _failure_binding(
        "VideoRepick",
        "put it down",
        0,
        "robomme.subtask.non_target_pickup",
    ),
    _failure_binding(
        "VideoRepick",
        "put it down",
        1,
        "robomme.subtask.premature_button",
    ),
    _failure_binding(
        "VideoRepick",
        "press the button to finish",
        0,
        "robomme.subtask.extra_pickup",
    ),
)


TASK_ADDITIONAL_OBSERVED_FAILURE_CODES: Mapping[str, frozenset[str]] = {
    "MoveCube": frozenset(
        {
            "robomme.movecube.peg_push.cube_already_at_goal",
            "robomme.movecube.grasp_putdown.cube_already_at_goal",
        }
    ),
    "BinFill": frozenset({"robomme.subtask.unexpected_extra_bin_drop"}),
    "RouteStick": frozenset({"robomme.route.direction_unverifiable"}),
    "SwingXtimes": frozenset({"robomme.subtask.extra_swing"}),
}


def _registered_task_failure_codes(task_name: str) -> frozenset[str]:
    contract = TASK_INTERVENTION_CONTRACTS.get(task_name)
    if contract is None:
        raise ComponentFactoryContractError(
            f"no typed intervention contract for {task_name!r}"
        )
    return frozenset(
        failure_code
        for trigger in contract.triggers
        for failure_code in trigger.expected_official_failure_codes
    )


def _known_task_failure_codes(task_name: str) -> frozenset[str]:
    return (
        _registered_task_failure_codes(task_name)
        | TASK_ADDITIONAL_OBSERVED_FAILURE_CODES.get(task_name, frozenset())
    )


def expected_official_failure_codes(
    task_name: str,
    reason_code: str,
    decision: InterventionDecision | None = None,
) -> frozenset[str]:
    """Return the source-audited native failures named by one early trigger."""

    contract = TASK_INTERVENTION_CONTRACTS.get(task_name)
    if contract is None:
        raise ComponentFactoryContractError(
            f"no typed intervention contract for {task_name!r}"
        )
    matching = [
        trigger
        for trigger in contract.triggers
        if trigger.reason_code == reason_code
    ]
    if len(matching) != 1:
        raise ComponentFactoryContractError(
            f"reason {reason_code!r} is not uniquely registered for {task_name!r}"
        )
    registered_codes = matching[0].expected_official_failure_codes
    if task_name != "MoveCube" or decision is None:
        return registered_codes
    manner = decision.metrics.get("manner")
    if reason_code == "pickup.wrong_actor_precontact":
        expected_by_manner = {
            "peg_push": "robomme.movecube.peg_push.cube_pickup",
            "grasp_putdown": "robomme.movecube.grasp_putdown.peg_pickup",
        }
        if manner not in expected_by_manner:
            raise ComponentFactoryContractError(
                "MoveCube wrong-pickup evidence lacks an applicable live manner: "
                f"{manner!r}"
            )
        return frozenset({expected_by_manner[str(manner)]})
    if reason_code == "movecube.forbidden_pickup_before_push":
        if manner != "gripper_push":
            raise ComponentFactoryContractError(
                "MoveCube forbidden-pickup evidence is outside gripper_push: "
                f"{manner!r}"
            )
        observed_actor = decision.observed_actor
        cube_actor = decision.metrics.get("cube_actor")
        peg_actors = tuple(decision.metrics.get("peg_actors", ()))
        if observed_actor == cube_actor:
            return frozenset(
                {"robomme.movecube.gripper_push.cube_pickup"}
            )
        if observed_actor in peg_actors:
            return frozenset(
                {"robomme.movecube.gripper_push.peg_pickup"}
            )
        raise ComponentFactoryContractError(
            "MoveCube forbidden-pickup evidence has no exact observed actor "
            f"binding: {observed_actor!r}"
        )
    return registered_codes


def canonicalize_observed_lifecycle_failure(
    task_name: str,
    lifecycle: LifecycleState,
) -> str:
    """Convert instrumented native failure evidence to one canonical code.

    A boolean failure flag is deliberately insufficient.  The process lane
    must preserve the native failure kind and the exact failed subtask so the
    counterfactual gate can distinguish the predicted error from an unrelated
    terminal event.
    """

    if not lifecycle.official_failure:
        raise ComponentFactoryContractError(
            "cannot canonicalize a lifecycle without official failure"
        )
    if not lifecycle.failure_kind:
        raise ComponentFactoryContractError(
            "official failure lacks instrumented failure_kind"
        )
    if lifecycle.failed_subtask_index is None or not lifecycle.failed_subtask_name:
        raise ComponentFactoryContractError(
            "official failure lacks failed_subtask_index/name"
        )
    failure_kind = lifecycle.failure_kind
    if not failure_kind.startswith("robomme."):
        raise ComponentFactoryContractError(
            "generic failure_kind is not source-attributed; the runtime overlay "
            "must emit a canonical robomme.* semantic id"
        )
    if failure_kind not in CANONICAL_FAILURE_SOURCES:
        raise ComponentFactoryContractError(
            f"failure_kind {failure_kind!r} has no audited source contract"
        )
    if failure_kind not in _known_task_failure_codes(task_name):
        raise ComponentFactoryContractError(
            f"failure_kind {failure_kind!r} is not source-bound for {task_name!r}"
        )
    return failure_kind


def canonical_failure_evidence(
    task_name: str,
    lifecycle: LifecycleState,
) -> Mapping[str, Any]:
    return {
        "failure_code": canonicalize_observed_lifecycle_failure(
            task_name,
            lifecycle,
        ),
        "failure_kind": lifecycle.failure_kind,
        "failed_subtask_index": lifecycle.failed_subtask_index,
        "failed_subtask_name": lifecycle.failed_subtask_name,
        "failure_reason": lifecycle.failure_reason,
        "source_contract": CANONICAL_FAILURE_SOURCES[lifecycle.failure_kind],
    }


@dataclasses.dataclass
class _SemanticFailureOverlayInstallation:
    raw_env: Any
    task_name: str
    reason_bridge: Any | None


@dataclasses.dataclass
class _SemanticFailureModuleRegistry:
    module: Any
    original_sequential_task_check: Callable[..., Any]
    wrapped_sequential_task_check: Callable[..., Any]
    installations: dict[int, _SemanticFailureOverlayInstallation]


_SEMANTIC_FAILURE_MODULE_REGISTRIES: dict[
    int,
    _SemanticFailureModuleRegistry,
] = {}
_MISSING_INSTANCE_ATTRIBUTE = object()


@dataclasses.dataclass
class SemanticFailureOverlayHandle:
    """Episode-scoped handle for a module-global RoboMME failure hook."""

    module_key: int
    environment_key: int
    raw_env: Any
    original_instance_attributes: dict[str, Any]
    _uninstalled: bool = False

    def uninstall(self) -> None:
        if self._uninstalled:
            return
        for attribute_name, original_value in (
            self.original_instance_attributes.items()
        ):
            if original_value is _MISSING_INSTANCE_ATTRIBUTE:
                if attribute_name in vars(self.raw_env):
                    delattr(self.raw_env, attribute_name)
            else:
                setattr(self.raw_env, attribute_name, original_value)
        registry = _SEMANTIC_FAILURE_MODULE_REGISTRIES.get(self.module_key)
        if registry is None:
            self._uninstalled = True
            return
        registry.installations.pop(self.environment_key, None)
        if not registry.installations:
            current = getattr(registry.module, "sequential_task_check", None)
            if current is registry.wrapped_sequential_task_check:
                registry.module.sequential_task_check = (
                    registry.original_sequential_task_check
                )
            _SEMANTIC_FAILURE_MODULE_REGISTRIES.pop(self.module_key, None)
        self._uninstalled = True


def semantic_failure_code(
    task_name: str,
    raw_env: Any,
    task_entry: Mapping[str, Any],
    predicate_index: int,
) -> str:
    """Resolve one true native predicate position to one canonical source id."""

    matches = [
        binding.failure_code
        for binding in SEMANTIC_FAILURE_BINDINGS
        if binding.matches(
            task_name,
            raw_env,
            task_entry,
            predicate_index,
        )
    ]
    if len(matches) != 1:
        raise ComponentFactoryContractError(
            "native failure predicate has no unique semantic source binding: "
            f"task={task_name!r}, way={getattr(raw_env, 'way', None)!r}, "
            f"choice_label={task_entry.get('choice_label')!r}, "
            f"name={task_entry.get('name')!r}, "
            f"demonstration={bool(task_entry.get('demonstration'))!r}, "
            f"predicate_index={predicate_index}, matches={matches}"
        )
    failure_code = matches[0]
    if failure_code not in CANONICAL_FAILURE_SOURCES:
        raise ComponentFactoryContractError(
            f"semantic binding {failure_code!r} lacks a source contract"
        )
    if failure_code not in _known_task_failure_codes(task_name):
        raise ComponentFactoryContractError(
            f"semantic binding {failure_code!r} is not source-bound for "
            f"{task_name!r}"
        )
    return failure_code


def _coerce_failure_leaf(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_coerce_failure_leaf(item) for item in value)
    if isinstance(value, Mapping):
        return any(_coerce_failure_leaf(item) for item in value.values())
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    if isinstance(converted, np.ndarray):
        return bool(np.any(converted)) if converted.size else False
    try:
        return bool(converted)
    except ValueError as error:
        raise ComponentFactoryContractError(
            "native failure predicate result cannot be reduced to one boolean"
        ) from error


def _true_failure_codes(
    installation: _SemanticFailureOverlayInstallation,
    task_entry: Mapping[str, Any],
    result: Any,
) -> tuple[str, ...]:
    if isinstance(result, (list, tuple)):
        indexed_values = tuple(enumerate(result))
    elif isinstance(result, (set, Mapping)):
        raise ComponentFactoryContractError(
            "unordered native failure result cannot have source-stable indices"
        )
    else:
        indexed_values = ((0, result),)
    codes: list[str] = []
    for predicate_index, value in indexed_values:
        if not _coerce_failure_leaf(value):
            continue
        codes.append(
            semantic_failure_code(
                installation.task_name,
                installation.raw_env,
                task_entry,
                predicate_index,
            )
        )
    return tuple(codes)


def _latch_semantic_failure(
    installation: _SemanticFailureOverlayInstallation,
    task_index: int,
    task_entry: Mapping[str, Any],
    failure_codes: Sequence[str],
) -> None:
    if not failure_codes:
        return
    raw_env = installation.raw_env
    existing = getattr(raw_env, "_robottt_dagger_v2_failure_kind", None)
    if isinstance(existing, str) and existing.startswith("robomme."):
        return
    task_name = task_entry.get("name") or task_entry.get("task_name") or "Unknown"
    primary_code = failure_codes[0]
    raw_env._robottt_dagger_v2_failure_kind = primary_code
    raw_env._robottt_dagger_v2_failure_subtask_index = task_index
    raw_env._robottt_dagger_v2_failure_subtask_name = str(task_name)
    raw_env._robottt_dagger_v2_failure_codes = tuple(failure_codes)
    reason_bridge = installation.reason_bridge
    if reason_bridge is not None:
        record_event = getattr(reason_bridge, "record_event", None)
        if not callable(record_event):
            raise ComponentFactoryContractError(
                "semantic failure reason bridge lacks record_event()"
            )
        record_event(
            "semantic_failure_predicate_true",
            {
                "failure_code": primary_code,
                "simultaneous_failure_codes": tuple(failure_codes),
                "failed_subtask_index": task_index,
                "failed_subtask_name": str(task_name),
                "choice_label": task_entry.get("choice_label"),
                "way": getattr(raw_env, "way", None),
                "source_contract": CANONICAL_FAILURE_SOURCES[primary_code],
            },
        )


def _instrument_semantic_failure_tasks(
    installation: _SemanticFailureOverlayInstallation,
    tasks: Sequence[Any],
) -> list[Any]:
    instrumented: list[Any] = []
    current_index = getattr(installation.raw_env, "timestep", None)
    for task_index, task_entry in enumerate(tasks):
        if not isinstance(task_entry, Mapping):
            instrumented.append(task_entry)
            continue
        copied_entry = dict(task_entry)
        failure_function = copied_entry.get("failure_func")
        if failure_function is None:
            failure_function = copied_entry.get("failure")
        if callable(failure_function):

            def wrapped_failure(
                *args: Any,
                _failure_function: Callable[..., Any] = failure_function,
                _task_index: int = task_index,
                _task_entry: Mapping[str, Any] = copied_entry,
                **kwargs: Any,
            ) -> Any:
                result = _failure_function(*args, **kwargs)
                failure_codes = _true_failure_codes(
                    installation,
                    _task_entry,
                    result,
                )
                _latch_semantic_failure(
                    installation,
                    _task_index,
                    _task_entry,
                    failure_codes,
                )
                return result

            wrapped_failure._robottt_dagger_v2_semantic_failure_probe = True
            copied_entry["failure_func"] = wrapped_failure
        elif failure_function is not None and task_index == current_index:
            failure_codes = _true_failure_codes(
                installation,
                copied_entry,
                failure_function,
            )
            _latch_semantic_failure(
                installation,
                task_index,
                copied_entry,
                failure_codes,
            )
        instrumented.append(copied_entry)
    return instrumented


def _integer_state(value: Any, field_name: str) -> int:
    array = _to_numpy(value, field_name).reshape(-1)
    if array.size != 1:
        raise ComponentFactoryContractError(
            f"{field_name} must contain one integer, got {array.shape}"
        )
    scalar = float(array[0])
    if not scalar.is_integer():
        raise ComponentFactoryContractError(
            f"{field_name} must be integral, got {scalar}"
        )
    return int(scalar)


def _task_name_at(raw_env: Any, task_index: int, fallback: str) -> str:
    task_list = getattr(raw_env, "task_list", ())
    if isinstance(task_list, Sequence) and 0 <= task_index < len(task_list):
        entry = task_list[task_index]
        if isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("task_name")
            if isinstance(name, str) and name:
                return name
    return fallback


def _install_episode_evaluate_hook(
    installation: _SemanticFailureOverlayInstallation,
    original_instance_attributes: dict[str, Any],
) -> None:
    raw_env = installation.raw_env
    task_name = installation.task_name
    if task_name not in {"PatternLock", "PickHighlight", "StopCube"}:
        return
    original_instance_attributes["evaluate"] = vars(raw_env).get(
        "evaluate",
        _MISSING_INSTANCE_ATTRIBUTE,
    )
    original_evaluate = raw_env.evaluate

    def wrapped_evaluate(
        environment: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_evaluate(*args, **kwargs)
        if not isinstance(result, Mapping) or "fail" not in result:
            raise ComponentFactoryContractError(
                f"{task_name}.evaluate did not return a fail field"
            )
        if not _coerce_failure_leaf(result["fail"]):
            return result
        task_list = getattr(environment, "task_list", ())
        if not isinstance(task_list, Sequence):
            raise ComponentFactoryContractError(
                f"{task_name}.task_list is unavailable after evaluate"
            )
        timestep = _integer_state(
            getattr(environment, "timestep", -1),
            f"{task_name}.timestep",
        )
        all_tasks_completed = timestep >= len(task_list)
        failure_codes: tuple[str, ...] = ()
        failed_subtask_index = min(max(timestep, 0), max(len(task_list) - 1, 0))
        failed_subtask_name = _task_name_at(
            environment,
            failed_subtask_index,
            "Unknown",
        )
        if task_name == "PatternLock":
            if all_tasks_completed and getattr(environment, "match", None) is False:
                failure_codes = ("robomme.route.sequence_mismatch",)
                failed_subtask_index = len(task_list)
                failed_subtask_name = "route sequence verification"
        elif task_name == "PickHighlight":
            pickup_counts = getattr(
                environment,
                "target_cube_pickup_counts",
                None,
            )
            if not isinstance(pickup_counts, Mapping):
                raise ComponentFactoryContractError(
                    "PickHighlight target_cube_pickup_counts is unavailable"
                )
            counts_satisfied = bool(
                pickup_counts
                and all(int(count) >= 1 for count in pickup_counts.values())
            )
            if all_tasks_completed and not counts_satisfied:
                failure_codes = ("robomme.subtask.highlight_count_mismatch",)
                failed_subtask_index = len(task_list)
                failed_subtask_name = "highlight pickup count verification"
        else:
            stop_timestep = getattr(environment, "stop_timestep", None)
            stop_time_range = getattr(environment, "stop_time_range", None)
            if (
                not isinstance(stop_time_range, Sequence)
                or len(stop_time_range) != 2
            ):
                raise ComponentFactoryContractError(
                    "StopCube stop_time_range is unavailable"
                )
            minimum_step = _integer_state(
                stop_time_range[0],
                "StopCube.stop_time_range[0]",
            )
            maximum_step = _integer_state(
                stop_time_range[1],
                "StopCube.stop_time_range[1]",
            )
            stop_step = (
                None
                if stop_timestep is None
                else _integer_state(stop_timestep, "StopCube.stop_timestep")
            )
            current_step = _integer_state(
                getattr(environment, "elapsed_steps", -1),
                "StopCube.elapsed_steps",
            )
            deadline = _integer_state(
                getattr(environment, "move_interval"),
                "StopCube.move_interval",
            ) * _integer_state(
                getattr(environment, "stop_time"),
                "StopCube.stop_time",
            )
            codes: list[str] = []
            if all_tasks_completed and (
                stop_step is None
                or not minimum_step <= stop_step <= maximum_step
            ):
                codes.append("robomme.subtask.missed_timed_press")
            if bool(getattr(environment, "stop", False)) and stop_step is None:
                codes.append("robomme.subtask.premature_button")
            if current_step > deadline and not all_tasks_completed:
                codes.append("robomme.subtask.missed_timed_press")
            failure_codes = tuple(dict.fromkeys(codes))
        if failure_codes:
            _latch_semantic_failure(
                installation,
                failed_subtask_index,
                {"name": failed_subtask_name},
                failure_codes,
            )
        return result

    raw_env.evaluate = types.MethodType(wrapped_evaluate, raw_env)


def _direction_diagnostic(
    raw_env: Any,
    judge_direction_list: Any,
) -> tuple[str, Mapping[str, Any]]:
    try:
        current_target, previous_target, expected_direction = judge_direction_list
    except Exception:
        return "robomme.route.direction_unverifiable", {
            "unverifiable_reason": "malformed_judge_direction_list"
        }
    trace = getattr(raw_env, "_gripper_xy_trace", None)
    try:
        previous_xy = _to_numpy(
            previous_target.pose.p,
            "RouteStick.previous_target.pose.p",
        ).reshape(-1, 3)[0, :2]
        current_xy = _to_numpy(
            current_target.pose.p,
            "RouteStick.current_target.pose.p",
        ).reshape(-1, 3)[0, :2]
    except Exception as error:
        return "robomme.route.direction_unverifiable", {
            "unverifiable_reason": "invalid_target_geometry",
            "geometry_error": f"{type(error).__name__}: {error}",
        }
    line = current_xy - previous_xy
    line_norm = float(np.linalg.norm(line))
    if line_norm < 1e-6:
        return "robomme.route.direction_unverifiable", {
            "unverifiable_reason": "degenerate_target_line",
            "line_norm": line_norm,
        }
    if not isinstance(trace, Sequence) or len(trace) == 0:
        return "robomme.route.direction_unverifiable", {
            "unverifiable_reason": "empty_gripper_trace",
            "line_norm": line_norm,
        }
    cross_values: list[float] = []
    try:
        for _, xy in trace:
            point_xy = _to_numpy(xy, "RouteStick.trace.xy").reshape(-1)[:2]
            relative = point_xy - previous_xy
            cross_values.append(float(line[0] * relative[1] - line[1] * relative[0]))
    except Exception as error:
        return "robomme.route.direction_unverifiable", {
            "unverifiable_reason": "invalid_gripper_trace",
            "trace_error": f"{type(error).__name__}: {error}",
        }
    if not cross_values:
        return "robomme.route.direction_unverifiable", {
            "unverifiable_reason": "empty_cross_product_trace"
        }
    average_cross = float(sum(cross_values) / len(cross_values))
    observed_direction = (
        "clockwise"
        if average_cross > 0.0
        else "counterclockwise"
        if average_cross < 0.0
        else "on the line"
    )
    expected = str(expected_direction).lower()
    diagnostics = {
        "average_cross": average_cross,
        "observed_direction": observed_direction,
        "expected_direction": expected,
        "trace_length": len(cross_values),
        "positive_cross_direction": "clockwise",
    }
    if observed_direction == "on the line":
        return "robomme.route.direction_unverifiable", {
            **diagnostics,
            "unverifiable_reason": "on_target_line",
        }
    return "robomme.route.direction_mismatch", diagnostics


def _install_route_direction_hook(
    installation: _SemanticFailureOverlayInstallation,
    original_instance_attributes: dict[str, Any],
) -> None:
    if installation.task_name != "RouteStick":
        return
    raw_env = installation.raw_env
    original_instance_attributes["direction_fail"] = vars(raw_env).get(
        "direction_fail",
        _MISSING_INSTANCE_ATTRIBUTE,
    )
    original_direction_fail = raw_env.direction_fail

    def wrapped_direction_fail(
        environment: Any,
        judge_direction_list: Any = None,
    ) -> Any:
        failure_code, diagnostics = _direction_diagnostic(
            environment,
            judge_direction_list,
        )
        result = original_direction_fail(
            judge_direction_list=judge_direction_list
        )
        if not _coerce_failure_leaf(result):
            task_list = getattr(environment, "task_list", ())
            if not isinstance(task_list, Sequence):
                raise ComponentFactoryContractError(
                    "RouteStick.task_list is unavailable at direction failure"
                )
            task_index = _integer_state(
                getattr(environment, "timestep", -1),
                "RouteStick.timestep",
            )
            task_name = _task_name_at(
                environment,
                task_index,
                "route direction verification",
            )
            _latch_semantic_failure(
                installation,
                task_index,
                {"name": task_name},
                (failure_code,),
            )
            if installation.reason_bridge is not None:
                installation.reason_bridge.record_event(
                    "route_direction_diagnostics",
                    diagnostics,
                )
        return result

    raw_env.direction_fail = types.MethodType(wrapped_direction_fail, raw_env)


def _record_pickhighlight_eager_failure_snapshot(
    installation: _SemanticFailureOverlayInstallation,
) -> None:
    if installation.task_name != "PickHighlight":
        return
    task_list = getattr(installation.raw_env, "task_list", ())
    if not isinstance(task_list, Sequence) or not task_list:
        raise ComponentFactoryContractError(
            "PickHighlight task_list is unavailable for eager failure audit"
        )
    entry = task_list[0]
    if not isinstance(entry, Mapping):
        raise ComponentFactoryContractError(
            "PickHighlight initial task entry is not a mapping"
        )
    frozen_value = entry.get("failure_func")
    if callable(frozen_value):
        raise ComponentFactoryContractError(
            "PickHighlight initial failure_func no longer has the audited eager bug"
        )
    reason_bridge = installation.reason_bridge
    if reason_bridge is not None:
        reason_bridge.record_event(
            "pickhighlight_eager_failure_snapshot",
            {
                "failed_subtask_index": 0,
                "failed_subtask_name": entry.get("name"),
                "value_type": type(frozen_value).__name__,
                "frozen_value_true": _coerce_failure_leaf(frozen_value),
                "source_contract": (
                    "PickHighlight.py assigns is_any_obj_pickup(...) eagerly "
                    "while constructing the initial button task"
                ),
            },
        )


def install_semantic_failure_overlay(
    raw_env: Any,
    task_name: str,
    reason_bridge: Any | None = None,
) -> SemanticFailureOverlayHandle:
    """Install episode-scoped semantic failure capture at the native call edge.

    RoboMME environment modules import ``sequential_task_check`` into module
    globals.  MoveCube additionally reconstructs its task dictionaries during
    every ``evaluate`` call.  Hooking this call edge therefore observes both
    stable reset-time lists and MoveCube's fresh lambdas without editing the
    benchmark or changing predicate return values.
    """

    if task_name not in TASK_INTERVENTION_CONTRACTS:
        raise ComponentFactoryContractError(
            f"cannot install semantic failures for unsupported {task_name!r}"
        )
    module_name = type(raw_env).__module__
    module = sys.modules.get(module_name)
    if module is None:
        raise ComponentFactoryContractError(
            f"raw environment module {module_name!r} is not loaded"
        )
    module_key = id(module)
    environment_key = id(raw_env)
    registry = _SEMANTIC_FAILURE_MODULE_REGISTRIES.get(module_key)
    if registry is None:
        original = getattr(module, "sequential_task_check", None)
        if not callable(original):
            raise ComponentFactoryContractError(
                f"{module_name}.sequential_task_check is unavailable"
            )
        installations: dict[int, _SemanticFailureOverlayInstallation] = {}

        def wrapped_sequential_task_check(
            environment: Any,
            tasks: Sequence[Any],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            active = installations.get(id(environment))
            if active is None or active.raw_env is not environment:
                return original(environment, tasks, *args, **kwargs)
            instrumented = _instrument_semantic_failure_tasks(active, tasks)
            return original(environment, instrumented, *args, **kwargs)

        registry = _SemanticFailureModuleRegistry(
            module=module,
            original_sequential_task_check=original,
            wrapped_sequential_task_check=wrapped_sequential_task_check,
            installations=installations,
        )
        _SEMANTIC_FAILURE_MODULE_REGISTRIES[module_key] = registry
        module.sequential_task_check = wrapped_sequential_task_check
    elif getattr(module, "sequential_task_check", None) is not (
        registry.wrapped_sequential_task_check
    ):
        raise ComponentFactoryContractError(
            f"{module_name}.sequential_task_check changed after overlay install"
        )
    if environment_key in registry.installations:
        raise ComponentFactoryContractError(
            "semantic failure overlay is already installed for this environment"
        )
    installation = _SemanticFailureOverlayInstallation(
        raw_env=raw_env,
        task_name=task_name,
        reason_bridge=reason_bridge,
    )
    registry.installations[environment_key] = installation
    original_instance_attributes: dict[str, Any] = {}
    handle = SemanticFailureOverlayHandle(
        module_key=module_key,
        environment_key=environment_key,
        raw_env=raw_env,
        original_instance_attributes=original_instance_attributes,
    )
    try:
        _install_episode_evaluate_hook(
            installation,
            original_instance_attributes,
        )
        _install_route_direction_hook(
            installation,
            original_instance_attributes,
        )
        _record_pickhighlight_eager_failure_snapshot(installation)
    except Exception:
        handle.uninstall()
        raise
    return handle


def _to_numpy(value: Any, field_name: str) -> Array:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ComponentFactoryContractError(f"{field_name} contains non-finite data")
    return array


def _position(value: Any, field_name: str) -> Array:
    array = _to_numpy(value, field_name).astype(np.float64, copy=False)
    if array.shape == (1, 3):
        array = array[0]
    if array.shape != (3,):
        raise ComponentFactoryContractError(
            f"{field_name} must have shape (3,) or (1, 3), got {array.shape}"
        )
    return np.array(array, copy=True)


def _quaternion(value: Any, field_name: str) -> Array:
    array = _to_numpy(value, field_name).astype(np.float64, copy=False).reshape(-1)
    if array.shape != (4,):
        raise ComponentFactoryContractError(
            f"{field_name} must have four quaternion values, got {array.shape}"
        )
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ComponentFactoryContractError(f"{field_name} has zero norm")
    return np.array(array / norm, copy=True)


def _positive_scalar(value: Any, field_name: str) -> float:
    array = _to_numpy(value, field_name).reshape(-1)
    if array.size != 1:
        raise ComponentFactoryContractError(
            f"{field_name} must contain one scalar, got {array.shape}"
        )
    scalar = float(array[0])
    if scalar <= 0.0:
        raise ComponentFactoryContractError(f"{field_name} must be positive")
    return scalar


def _runtime_scale(raw_env: Any) -> float:
    if not hasattr(raw_env, "cube_half_size"):
        raise ComponentFactoryContractError(
            "raw_env.cube_half_size is required for detector calibration"
        )
    return _positive_scalar(raw_env.cube_half_size, "raw_env.cube_half_size")


def _semantic_signal(
    *,
    active: bool,
    reason_code: str,
    recovery_mode: RecoveryMode,
    expected_actor: str | None,
    observed_actor: str | None,
    metrics: Mapping[str, Any],
    priority: int,
    activate_after: int = 1,
) -> CandidateSignal:
    return CandidateSignal(
        active=active,
        reason_code=reason_code,
        signal_class=SignalClass.SEMANTIC,
        recovery_mode=recovery_mode,
        expected_actor=expected_actor,
        observed_actor=observed_actor,
        metrics=dict(metrics),
        thresholds={"identity_source": "live_actor_instance"},
        priority=priority,
        activate_after=activate_after,
        release_after=1,
    )


class RoboMMEFactoryDetector:
    """Compose shared primitives from adapter-supplied live identities."""

    def __init__(
        self,
        task_name: str,
        object_half_size: float | None,
        calibration: RuntimeCalibration,
        task_calibration: TaskDetectorCalibration | None = None,
        evidence_capabilities: TaskEvidenceCapabilities | None = None,
    ) -> None:
        self.task_name = task_name
        self.object_half_size = object_half_size
        self.calibration = calibration
        self.task_calibration = task_calibration
        self.evidence_capabilities = evidence_capabilities
        contract = TASK_INTERVENTION_CONTRACTS.get(task_name)
        if contract is not None and contract.required_live_calibration:
            if task_calibration is None:
                raise ComponentFactoryContractError(
                    f"{task_name} requires an explicit live task calibration"
                )
            self._validate_task_calibration(contract)
            self._validate_evidence_capabilities(contract)
        elif contract is None and object_half_size is None:
            raise ComponentFactoryContractError(
                f"{task_name} requires native object scale calibration"
            )

    def _validate_evidence_capabilities(
        self,
        contract: TaskInterventionContract,
    ) -> None:
        if not contract.required_live_evidence:
            return
        if self.evidence_capabilities is None:
            raise ComponentFactoryContractError(
                f"{self.task_name} requires explicit live evidence capabilities: "
                f"{list(contract.required_live_evidence)}"
            )
        missing: list[str] = []
        for field_name in contract.required_live_evidence:
            metric_name = getattr(self.evidence_capabilities, field_name)
            provider_name = field_name.replace("_metric", "_provider")
            provider = getattr(self.evidence_capabilities, provider_name)
            if not isinstance(metric_name, str) or not metric_name:
                missing.append(field_name)
            if not callable(provider):
                missing.append(provider_name)
        if missing:
            raise ComponentFactoryContractError(
                f"{self.task_name} live evidence capability is missing {missing}"
            )

    def _evidence_metric_name(self, capability_field: str) -> str:
        if self.evidence_capabilities is None:
            raise ComponentFactoryContractError(
                f"{self.task_name} has no live evidence capabilities"
            )
        metric_name = getattr(self.evidence_capabilities, capability_field)
        if not isinstance(metric_name, str) or not metric_name:
            raise ComponentFactoryContractError(
                f"{self.task_name}.{capability_field} is unavailable"
            )
        return metric_name

    def _require_true_evidence_metric(
        self,
        context: PreActionContext,
        capability_field: str,
    ) -> None:
        metric_name = self._evidence_metric_name(capability_field)
        if metric_name not in context.scene.metrics:
            raise AdapterContractError(
                f"{self.task_name} live adapter/overlay did not emit {metric_name!r}"
            )
        if context.scene.metrics[metric_name] is not True:
            raise AdapterContractError(
                f"{self.task_name} live evidence {metric_name!r} is not verified true"
            )

    def evaluate(self, context: PreActionContext) -> Sequence[CandidateSignal]:
        if context.scene.task_name != self.task_name:
            raise AdapterContractError(
                f"detector for {self.task_name!r} received "
                f"{context.scene.task_name!r}"
            )
        rules = {
            "MoveCube": self._movecube,
            "InsertPeg": self._insertpeg,
            "BinFill": self._binfill,
            "StopCube": self._stopcube,
            "PickHighlight": self._pickhighlight,
            "ButtonUnmask": self._container_sequence,
            "ButtonUnmaskSwap": self._container_sequence,
            "VideoUnmask": self._container_sequence,
            "VideoUnmaskSwap": self._container_sequence,
            "PatternLock": self._route,
            "RouteStick": self._route,
            "SwingXtimes": self._swingxtimes,
            "PickXtimes": self._pickxtimes,
            "VideoRepick": self._videorepick,
            "VideoPlaceButton": self._video_place,
            "VideoPlaceOrder": self._video_place,
        }
        if self.task_name not in rules:
            raise AdapterContractError(
                f"live factory detector is not implemented for {self.task_name!r}"
            )
        return rules[self.task_name](context)

    def _validate_task_calibration(
        self,
        contract: TaskInterventionContract,
    ) -> None:
        assert self.task_calibration is not None
        missing: list[str] = []
        invalid: list[str] = []
        for field_name in contract.required_live_calibration:
            value = getattr(self.task_calibration, field_name)
            if value is None:
                missing.append(field_name)
                continue
            if field_name == "route_positive_cross_direction":
                if value not in ("clockwise", "counterclockwise"):
                    invalid.append(field_name)
                continue
            if field_name == "route_trend_window":
                if not isinstance(value, int) or value < 2:
                    invalid.append(field_name)
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                invalid.append(field_name)
            elif float(value) <= 0.0:
                invalid.append(field_name)
        if missing or invalid:
            raise ComponentFactoryContractError(
                f"{self.task_name} live calibration is incomplete: "
                f"missing={missing}, invalid={invalid}"
            )

    def _task_value(self, field_name: str) -> float:
        if self.task_calibration is None:
            raise ComponentFactoryContractError(
                f"{self.task_name} lacks task calibration for {field_name}"
            )
        value = getattr(self.task_calibration, field_name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ComponentFactoryContractError(
                f"{self.task_name}.{field_name} is not calibrated"
            )
        return float(value)

    @property
    def _pickup_radius(self) -> float:
        if self.task_calibration is not None:
            return self._task_value("pickup_intent_radius")
        assert self.object_half_size is not None
        return (
            self.object_half_size
            * self.calibration.pickup_intent_radius_half_sizes
        )

    @property
    def _selection_margin(self) -> float:
        if self.task_calibration is not None:
            return self._task_value("pickup_selection_margin")
        assert self.object_half_size is not None
        return (
            self.object_half_size
            * self.calibration.pickup_selection_margin_half_sizes
        )

    @property
    def _minimum_approach(self) -> float:
        if self.task_calibration is not None:
            return self._task_value("minimum_approach_delta")
        assert self.object_half_size is not None
        return (
            self.object_half_size
            * self.calibration.minimum_approach_half_sizes
        )

    def _pickup(self, context: PreActionContext) -> list[CandidateSignal]:
        expected = tuple(context.scene.metrics.get("expected_actor_names", ()))
        forbidden = tuple(context.scene.metrics.get("forbidden_actor_names", ()))
        if not expected:
            raise AdapterContractError(
                f"{self.task_name} pickup phase lacks exact expected actor identities"
            )
        missed_grasp_metric = None
        contract = TASK_INTERVENTION_CONTRACTS[self.task_name]
        if "pickup_attempt_complete_metric" in contract.required_live_evidence:
            missed_grasp_metric = self._evidence_metric_name(
                "pickup_attempt_complete_metric"
            )
            if missed_grasp_metric not in context.scene.metrics:
                raise AdapterContractError(
                    f"{self.task_name} pickup phase did not emit "
                    f"{missed_grasp_metric!r}"
                )
        return list(
            pickup_intent_signals(
                context,
                PickupIntentSpec(
                    expected_actors=expected,
                    forbidden_actors=forbidden,
                    intent_radius=self._pickup_radius,
                    selection_margin=self._selection_margin,
                    minimum_approach_delta=self._minimum_approach,
                    activate_after=self.calibration.detector_activate_after,
                    missed_grasp_metric=missed_grasp_metric,
                ),
            )
        )

    def _container_pickup(
        self,
        context: PreActionContext,
    ) -> list[CandidateSignal]:
        expected = tuple(context.scene.metrics.get("expected_actor_names", ()))
        forbidden = tuple(context.scene.metrics.get("forbidden_actor_names", ()))
        if not expected:
            raise AdapterContractError(
                f"{self.task_name} container phase lacks exact expected identity"
            )
        signals = list(
            container_intent_signals(
                context,
                ContainerIntentSpec(
                    expected_containers=expected,
                    forbidden_containers=forbidden,
                    intent_radius=self._task_value("container_intent_radius"),
                    selection_margin=self._task_value(
                        "container_selection_margin"
                    ),
                    minimum_approach_delta=self._minimum_approach,
                    activate_after=self.calibration.detector_activate_after,
                ),
            )
        )
        missed_grasp_metric = self._evidence_metric_name(
            "pickup_attempt_complete_metric"
        )
        if missed_grasp_metric not in context.scene.metrics:
            raise AdapterContractError(
                f"{self.task_name} container phase did not emit "
                f"{missed_grasp_metric!r}"
            )
        attempt_complete = bool(context.scene.metrics[missed_grasp_metric])
        signals.append(
            _semantic_signal(
                active=attempt_complete and context.scene.held_actor is None,
                reason_code="container.missed_pickup",
                recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                expected_actor=expected[0] if len(expected) == 1 else None,
                observed_actor=None,
                metrics={
                    "attempt_complete": attempt_complete,
                    "evidence_metric": missed_grasp_metric,
                },
                priority=86,
            )
        )
        return signals

    def _button(
        self,
        context: PreActionContext,
        *,
        expected: Sequence[str],
        forbidden: Sequence[str],
        prerequisite_metric: str | None = None,
    ) -> list[CandidateSignal]:
        if not expected and not forbidden:
            raise AdapterContractError("button detector has no named button evidence")
        return list(
            button_intent_signals(
                context,
                ButtonIntentSpec(
                    expected_buttons=tuple(expected),
                    forbidden_buttons=tuple(forbidden),
                    intent_radius=self._task_value("button_intent_radius"),
                    selection_margin=self._task_value("button_selection_margin"),
                    minimum_approach_delta=self._minimum_approach,
                    activate_after=self.calibration.detector_activate_after,
                    prerequisite_metric=prerequisite_metric,
                ),
            )
        )

    def _guard_button_before_phase_completion(
        self,
        context: PreActionContext,
    ) -> list[CandidateSignal]:
        if "button" not in context.scene.landmarks:
            return []
        return self._button(context, expected=(), forbidden=("button",))

    def _place(self, context: PreActionContext) -> list[CandidateSignal]:
        expected_actor_names = tuple(
            context.scene.metrics.get("expected_actor_names", ())
        )
        held_actor = context.scene.held_actor
        signals = self._held_identity_signal(
            context,
            "identity.wrong_actor_held",
        )
        if held_actor is None:
            signals.extend(self._required_held_signal(context))
            return signals
        if held_actor not in expected_actor_names:
            return signals
        expected_targets = tuple(
            context.scene.metrics.get("expected_target_names", ())
        )
        forbidden_targets = tuple(
            context.scene.metrics.get("forbidden_target_names", ())
        )
        if self.task_name == "SwingXtimes" and not forbidden_targets:
            forbidden_targets = tuple(
                name
                for name in context.scene.landmarks
                if name.startswith("target_") and name not in expected_targets
            )
        if not expected_targets:
            raise AdapterContractError(
                f"{self.task_name} place phase lacks exact expected target"
            )
        signals.extend(
            place_intent_signals(
                context,
                PlaceIntentSpec(
                    held_actor=held_actor,
                    expected_targets=expected_targets,
                    forbidden_targets=forbidden_targets,
                    release_radius=self._task_value("place_release_radius"),
                    transport_radius=self._task_value("place_transport_radius"),
                    selection_margin=self._task_value("place_selection_margin"),
                    minimum_approach_delta=self._minimum_approach,
                    activate_after=self.calibration.detector_activate_after,
                ),
            )
        )
        signals.extend(
            self._expected_place_progress_signal(
                context,
                held_actor,
                expected_targets,
            )
        )
        return signals

    def _expected_place_progress_signal(
        self,
        context: PreActionContext,
        held_actor: str,
        expected_targets: Sequence[str],
    ) -> list[CandidateSignal]:
        if held_actor not in context.prediction.predicted_actor_positions:
            raise AdapterContractError(
                "place progress requires a predicted held-actor position"
            )
        current_position = context.scene.actors[held_actor].position
        predicted_position = context.prediction.predicted_actor_positions[held_actor]
        target_distances = {
            name: float(
                np.linalg.norm(predicted_position - context.scene.landmarks[name])
            )
            for name in expected_targets
        }
        expected_target = min(target_distances, key=target_distances.get)
        predicted_distance = target_distances[expected_target]
        current_distance = float(
            np.linalg.norm(
                current_position - context.scene.landmarks[expected_target]
            )
        )
        actor_motion = float(np.linalg.norm(predicted_position - current_position))
        improvement = current_distance - predicted_distance
        opening = bool(
            context.prediction.next_gripper_opening > context.scene.gripper_opening
            or "open_gripper" in context.prediction.intents
        )
        release_outside = bool(
            opening
            and predicted_distance > self._task_value("place_release_radius")
        )
        not_progressing = bool(
            actor_motion >= self._minimum_approach
            and improvement
            < self._task_value("place_minimum_improvement")
            and predicted_distance > self._task_value("place_release_radius")
        )
        return [
            _semantic_signal(
                active=release_outside or not_progressing,
                reason_code="place.not_approaching_expected_target",
                recovery_mode=RecoveryMode.LIFT_HOLD_THEN_PLACE,
                expected_actor=expected_target,
                observed_actor=None,
                metrics={
                    "held_actor": held_actor,
                    "expected_target": expected_target,
                    "current_distance": current_distance,
                    "predicted_distance": predicted_distance,
                    "expected_approach_improvement": improvement,
                    "actor_motion": actor_motion,
                    "opening": opening,
                    "release_outside": release_outside,
                },
                priority=93 if release_outside else 74,
                activate_after=(
                    1
                    if release_outside
                    else self.calibration.detector_activate_after
                ),
            )
        ]

    def _required_held_signal(
        self,
        context: PreActionContext,
    ) -> list[CandidateSignal]:
        expected = tuple(context.scene.metrics.get("expected_actor_names", ()))
        if not expected:
            raise AdapterContractError(
                "required-held recovery lacks an expected actor identity"
            )
        return [
            _semantic_signal(
                active=context.scene.held_actor is None,
                reason_code="identity.required_actor_not_held",
                recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                expected_actor=expected[0] if len(expected) == 1 else None,
                observed_actor=None,
                metrics={
                    "expected_actor_names": expected,
                    "phase_requires_held_actor": True,
                },
                priority=97,
                activate_after=self.calibration.detector_activate_after,
            )
        ]

    def _forbid_extra_pickup(
        self,
        context: PreActionContext,
        *,
        intent_radius: float | None = None,
    ) -> list[CandidateSignal]:
        actor_names = tuple(context.scene.actors)
        if not actor_names:
            return []
        return list(
            forbidden_pickup_signals(
                context,
                ForbiddenPickupSpec(
                    forbidden_actors=actor_names,
                    intent_radius=(
                        self._task_value("pickup_intent_radius")
                        if intent_radius is None
                        else intent_radius
                    ),
                    minimum_approach_delta=self._minimum_approach,
                    activate_after=self.calibration.detector_activate_after,
                    prerequisite_metric=None,
                ),
            )
        )

    @staticmethod
    def _held_identity_signal(
        context: PreActionContext,
        reason_code: str,
    ) -> list[CandidateSignal]:
        expected = tuple(context.scene.metrics.get("expected_actor_names", ()))
        held = context.scene.held_actor
        if held is None or held in expected:
            return []
        if not expected:
            raise AdapterContractError(
                "held-identity recovery requires exact expected actor identities"
            )
        return [
            _semantic_signal(
                active=True,
                reason_code=reason_code,
                recovery_mode=RecoveryMode.RETURN_WRONG_OBJECT,
                expected_actor=expected[0] if len(expected) == 1 else None,
                observed_actor=held,
                metrics={
                    "held_actor": held,
                    "expected_actor_names": expected,
                    "evidence_timing": "after_grasp_before_official_terminal",
                },
                priority=99,
            )
        ]

    def _movecube(self, context: PreActionContext) -> list[CandidateSignal]:
        phase = context.scene.subtask.phase
        signals: list[CandidateSignal] = []
        if phase in {"pickup_tool", "pickup_cube"}:
            signals.extend(
                dataclasses.replace(
                    signal,
                    metrics={
                        **signal.metrics,
                        "manner": context.scene.metrics.get("manner"),
                    },
                )
                for signal in self._pickup(context)
            )
        if phase == "gripper_push":
            forbidden = tuple(
                context.scene.metrics.get("forbidden_actor_names", ())
            )
            if not forbidden:
                raise AdapterContractError(
                    "MoveCube gripper_push lacks exact forbidden actor identities"
                )
            push_pickup_signals = grasped_lift_intent_signals(
                context,
                GraspedLiftIntentSpec(
                    candidate_actors=forbidden,
                    native_pickup_height=MOVECUBE_NATIVE_PICKUP_HEIGHT_METERS,
                    minimum_lift_delta=self._minimum_approach,
                    activate_after=1,
                ),
            )
            signals.extend(
                dataclasses.replace(
                    signal,
                    reason_code="movecube.forbidden_pickup_before_push",
                    recovery_mode=RecoveryMode.CANCEL_FORBIDDEN_PUSH_PICKUP,
                    metrics={
                        **signal.metrics,
                        "manner": context.scene.metrics.get("manner"),
                        "cube_actor": context.scene.metrics.get("cube_actor"),
                        "peg_actors": tuple(
                            context.scene.metrics.get("peg_actors", ())
                        ),
                    },
                )
                for signal in push_pickup_signals
            )
        if phase == "peg_push":
            signals.extend(
                self._held_identity_signal(
                    context,
                    "movecube.wrong_tool_held_before_hook",
                )
            )
            expected = tuple(
                context.scene.metrics.get("expected_actor_names", ())
            )
            if expected and context.scene.held_actor is None:
                signals.append(
                    _semantic_signal(
                        active=True,
                        reason_code="movecube.required_tool_not_held",
                        recovery_mode=RecoveryMode.OPEN_AND_RETRACT,
                        expected_actor=expected[0],
                        observed_actor=None,
                        metrics={"expected_actor_names": expected},
                        priority=98,
                    )
                )
        if phase == "place_cube":
            signals.extend(
                self._held_identity_signal(
                    context,
                    "movecube.wrong_actor_held_before_place",
                )
            )
            held = context.scene.held_actor
            if held == context.scene.subtask.expected_actor:
                signals.extend(
                    place_intent_signals(
                        context,
                        PlaceIntentSpec(
                            held_actor=held,
                            expected_targets=("goal",),
                            forbidden_targets=(),
                            release_radius=self._pickup_radius,
                            transport_radius=self._pickup_radius * 2.0,
                            selection_margin=self._selection_margin,
                            minimum_approach_delta=self._minimum_approach,
                            activate_after=self.calibration.detector_activate_after,
                        ),
                    )
                )
        return signals

    def _insertpeg(self, context: PreActionContext) -> list[CandidateSignal]:
        phase = context.scene.subtask.phase
        if phase == "pickup_peg_end":
            return self._pickup(context)
        if phase != "insert_peg":
            raise AdapterContractError(f"unsupported InsertPeg phase {phase!r}")
        metrics = context.scene.metrics
        return list(
            insertion_intent_signals(
                context,
                InsertionIntentSpec(
                    expected_insert_actor=str(metrics["insert_target_actor"]),
                    opposite_end_actor=str(metrics["opposite_end_actor"]),
                    target_landmark="box",
                    expected_direction=int(metrics["insert_direction"]),
                    side_axis=int(metrics["insert_side_axis"]),
                    intent_radius=(
                        self.object_half_size
                        * self.calibration.insertion_intent_radius_half_sizes
                    ),
                    selection_margin=self._selection_margin,
                    side_margin=(
                        self.object_half_size
                        * self.calibration.insertion_side_margin_half_sizes
                    ),
                    minimum_approach_delta=self._minimum_approach,
                    activate_after=self.calibration.detector_activate_after,
                ),
            )
        )

    def _binfill(self, context: PreActionContext) -> list[CandidateSignal]:
        phase = context.scene.subtask.phase
        if phase == "pickup":
            return self._pickup(context)
        if phase == "place_in_bin":
            return self._held_identity_signal(
                context,
                "binfill.wrong_cube_held_before_bin_drop",
            )
        if phase == "press_button":
            return list(
                button_intent_signals(
                    context,
                    ButtonIntentSpec(
                        expected_buttons=("button",),
                        forbidden_buttons=(),
                        intent_radius=self._pickup_radius,
                        selection_margin=self._selection_margin,
                        minimum_approach_delta=self._minimum_approach,
                        activate_after=self.calibration.detector_activate_after,
                        prerequisite_metric="objects_complete",
                    ),
                )
            )
        raise AdapterContractError(f"unsupported BinFill phase {phase!r}")

    def _stopcube(self, context: PreActionContext) -> list[CandidateSignal]:
        if not bool(context.scene.metrics.get("timing_detector_ready", False)):
            raise AdapterContractError(
                "StopCube timing detector lacks calibrated live press progress"
            )
        return list(
            timed_progress_signals(
                context,
                TimedProgressSpec(
                    seconds_to_event_metric="seconds_to_event",
                    progress_metric="press_progress",
                    expert_completion_seconds=(
                        self.calibration.stopcube_expert_completion_seconds
                    ),
                    early_margin_seconds=(
                        self.calibration.stopcube_early_margin_seconds
                    ),
                    late_margin_seconds=(
                        self.calibration.stopcube_late_margin_seconds
                    ),
                    minimum_ready_progress=(
                        self.calibration.stopcube_minimum_ready_progress
                    ),
                    activate_after=self.calibration.detector_activate_after,
                ),
            )
        )

    def _pickhighlight(self, context: PreActionContext) -> list[CandidateSignal]:
        self._require_true_evidence_metric(
            context,
            "remaining_set_failure_overlay_metric",
        )
        self._require_true_evidence_metric(
            context,
            "semantic_solver_valid_metric",
        )
        phase = context.scene.subtask.phase
        if phase == "press_button":
            signals = self._button(context, expected=("button",), forbidden=())
            signals.extend(self._forbid_extra_pickup(context))
            return signals
        signals = self._guard_button_before_phase_completion(context)
        if phase == "pickup":
            signals.extend(self._pickup(context))
            signals.extend(
                self._held_identity_signal(
                    context,
                    "identity.wrong_actor_held",
                )
            )
            return signals
        if phase == "putdown":
            signals.extend(
                self._held_identity_signal(
                    context,
                    "identity.wrong_actor_held",
                )
            )
            if context.scene.held_actor is None:
                signals.extend(self._required_held_signal(context))
            return signals
        raise AdapterContractError(f"unsupported PickHighlight phase {phase!r}")

    def _container_sequence(
        self,
        context: PreActionContext,
    ) -> list[CandidateSignal]:
        phase = context.scene.subtask.phase
        if phase == "pickup_container":
            signals = self._container_pickup(context)
            signals.extend(
                self._held_identity_signal(
                    context,
                    "identity.wrong_actor_held",
                )
            )
            return signals
        if phase == "putdown_container":
            signals = self._held_identity_signal(
                context,
                "identity.wrong_actor_held",
            )
            if context.scene.held_actor is None:
                signals.extend(self._required_held_signal(context))
            return signals
        if phase == "press_button":
            solver_metric = "task_solver_semantically_valid"
            if "semantic_solver_valid_metric" in (
                TASK_INTERVENTION_CONTRACTS[
                    self.task_name
                ].required_live_evidence
            ):
                solver_metric = self._evidence_metric_name(
                    "semantic_solver_valid_metric"
                )
            if context.scene.metrics.get(solver_metric) is not True:
                raise AdapterContractError(
                    f"{self.task_name} native button solver is not semantically valid"
                )
            expected = tuple(
                context.scene.metrics.get("allowed_button_names", ())
            )
            forbidden = tuple(
                name for name in context.scene.landmarks if name not in expected
            )
            signals = self._button(
                context,
                expected=expected,
                forbidden=forbidden,
            )
            signals.extend(
                self._forbid_extra_pickup(
                    context,
                    intent_radius=self._task_value("container_intent_radius"),
                )
            )
            return signals
        raise AdapterContractError(
            f"unsupported {self.task_name} phase {phase!r}"
        )

    def _route(self, context: PreActionContext) -> list[CandidateSignal]:
        metrics = context.scene.metrics
        expected_waypoints = tuple(metrics.get("expected_waypoint_names", ()))
        if len(expected_waypoints) != 1:
            raise AdapterContractError(
                f"{self.task_name} requires one exact expected waypoint"
            )
        corridor_start = metrics.get("corridor_start")
        if not isinstance(corridor_start, str):
            raise AdapterContractError(
                f"{self.task_name} lacks a semantic corridor_start"
            )
        tracked_point = (
            "stick_tip" if self.task_name == "RouteStick" else "swing_tip"
        )
        signals = list(
            route_progress_signals(
                context,
                RouteProgressSpec(
                    tracked_point=tracked_point,
                    corridor_start=corridor_start,
                    expected_waypoint=expected_waypoints[0],
                    forbidden_waypoints=tuple(
                        metrics.get("forbidden_waypoint_names", ())
                    ),
                    target_radius=self._task_value("route_target_radius"),
                    wrong_waypoint_radius=self._task_value(
                        "route_wrong_waypoint_radius"
                    ),
                    maximum_cross_track=self._task_value(
                        "route_maximum_cross_track"
                    ),
                    minimum_improvement_per_second=self._task_value(
                        "route_minimum_improvement_per_second"
                    ),
                    trend_window=int(
                        self._task_value("route_trend_window")
                    ),
                    activate_after=self.calibration.detector_activate_after,
                ),
            )
        )
        if self.task_name == "RouteStick":
            signals.extend(
                self._route_direction_signal(
                    context,
                    tracked_point,
                    corridor_start,
                    expected_waypoints[0],
                )
            )
        return signals

    def _route_direction_signal(
        self,
        context: PreActionContext,
        tracked_point: str,
        corridor_start: str,
        expected_waypoint: str,
    ) -> list[CandidateSignal]:
        assert self.task_calibration is not None
        if tracked_point not in context.prediction.predicted_points:
            raise AdapterContractError(
                f"RouteStick predictor lacks {tracked_point!r}"
            )
        expected_direction = context.scene.metrics.get("expected_direction")
        if expected_direction not in ("clockwise", "counterclockwise"):
            raise AdapterContractError("RouteStick expected_direction is invalid")
        positive_direction = self.task_calibration.route_positive_cross_direction
        if positive_direction not in ("clockwise", "counterclockwise"):
            raise ComponentFactoryContractError(
                "RouteStick positive cross direction is not calibrated"
            )
        start = context.scene.landmarks[corridor_start]
        end = context.scene.landmarks[expected_waypoint]
        predicted = context.prediction.predicted_points[tracked_point]
        route_vector = end[:2] - start[:2]
        point_vector = predicted[:2] - start[:2]
        signed_cross = float(
            route_vector[0] * point_vector[1]
            - route_vector[1] * point_vector[0]
        )
        route_norm = float(np.linalg.norm(route_vector))
        if route_norm <= 1e-12:
            raise AdapterContractError("RouteStick route segment has zero length")
        signed_side_distance = signed_cross / route_norm
        observed_direction = (
            positive_direction
            if signed_side_distance > 0.0
            else (
                "counterclockwise"
                if positive_direction == "clockwise"
                else "clockwise"
            )
        )
        side_margin = self._task_value("route_direction_side_margin")
        active = bool(
            abs(signed_side_distance) >= side_margin
            and observed_direction != expected_direction
        )
        return [
            _semantic_signal(
                active=active,
                reason_code="route.direction_side_mismatch",
                recovery_mode=RecoveryMode.RETURN_TO_SAFE_ANCHOR,
                expected_actor=expected_waypoint,
                observed_actor=None,
                metrics={
                    "expected_direction": expected_direction,
                    "observed_direction": observed_direction,
                    "signed_side_distance": signed_side_distance,
                    "positive_cross_direction": positive_direction,
                },
                priority=91,
                activate_after=self.calibration.detector_activate_after,
            )
        ]

    def _swingxtimes(self, context: PreActionContext) -> list[CandidateSignal]:
        phase = context.scene.subtask.phase
        if phase == "pickup":
            signals = self._guard_button_before_phase_completion(context)
            signals.extend(self._pickup(context))
            return signals
        if phase == "swing":
            signals = self._guard_button_before_phase_completion(context)
            signals.extend(self._place(context))
            return signals
        if phase == "putdown":
            signals = self._guard_button_before_phase_completion(context)
            signals.extend(
                self._held_identity_signal(
                    context,
                    "identity.wrong_actor_held",
                )
            )
            if context.scene.held_actor is None:
                signals.extend(self._required_held_signal(context))
            return signals
        if phase == "press_button":
            signals = self._button(context, expected=("button",), forbidden=())
            signals.extend(self._forbid_extra_pickup(context))
            return signals
        raise AdapterContractError(f"unsupported SwingXtimes phase {phase!r}")

    def _pickxtimes(self, context: PreActionContext) -> list[CandidateSignal]:
        phase = context.scene.subtask.phase
        if phase == "pickup":
            signals = self._guard_button_before_phase_completion(context)
            signals.extend(self._pickup(context))
            return signals
        if phase == "place":
            signals = self._guard_button_before_phase_completion(context)
            signals.extend(self._place(context))
            return signals
        if phase == "press_button":
            signals = self._button(
                context,
                expected=("button",),
                forbidden=(),
                prerequisite_metric="button_prerequisite_complete",
            )
            signals.extend(self._forbid_extra_pickup(context))
            return signals
        raise AdapterContractError(f"unsupported PickXtimes phase {phase!r}")

    def _videorepick(self, context: PreActionContext) -> list[CandidateSignal]:
        phase = context.scene.subtask.phase
        if phase == "pickup":
            signals = self._guard_button_before_phase_completion(context)
            signals.extend(self._pickup(context))
            return signals
        if phase == "putdown":
            signals = self._guard_button_before_phase_completion(context)
            signals.extend(
                self._held_identity_signal(
                    context,
                    "identity.wrong_actor_held",
                )
            )
            if context.scene.held_actor is None:
                signals.extend(self._required_held_signal(context))
            return signals
        if phase == "press_button":
            signals = self._button(
                context,
                expected=("button",),
                forbidden=(),
                prerequisite_metric="button_prerequisite_complete",
            )
            signals.extend(self._forbid_extra_pickup(context))
            return signals
        raise AdapterContractError(f"unsupported VideoRepick phase {phase!r}")

    def _video_place(self, context: PreActionContext) -> list[CandidateSignal]:
        phase = context.scene.subtask.phase
        if phase == "pickup":
            return self._pickup(context)
        if phase == "place":
            return self._place(context)
        raise AdapterContractError(
            f"unsupported {self.task_name} phase {phase!r}"
        )


class PureMplibRecoveryActionPlanners:
    """Plan joint actions without calling ``env.step``.

    The recorder already creates the exact Panda/Panda-stick mplib model used
    for action FK.  This class calls only its low-level planning API and turns
    returned qpos waypoints into joint actions.  Any absent planner field,
    planning error, oversized path, or non-finite output fails closed before a
    recovery action is applied.
    """

    def __init__(
        self,
        recorder: Any,
        raw_env: Any,
        actor_objects: Mapping[str, Any],
        calibration: RuntimeCalibration,
    ) -> None:
        self.recorder = recorder
        self.raw_env = raw_env
        self.actor_objects = dict(actor_objects)
        self.calibration = calibration
        self.solver = getattr(recorder, "planner", None)
        self.planner = getattr(recorder, "_mplib_planner", None)
        if self.solver is None or self.planner is None:
            raise ComponentFactoryContractError(
                "RecordWrapper mplib planner is unavailable after reset"
            )
        required = (
            "transform_goal_to_wrt_base",
            "plan_screw",
            "plan_qpos_to_pose",
        )
        missing = [name for name in required if not callable(getattr(self.planner, name, None))]
        if missing:
            raise ComponentFactoryContractError(
                f"RecordWrapper mplib planner lacks pure APIs: {missing}"
            )

    def vertical_lift(
        self,
        clearance_z: float,
        keep_grasp: bool,
    ) -> Sequence[Array]:
        target = self._tcp_position()
        target[2] = max(target[2], float(clearance_z))
        gripper = -1.0 if keep_grasp else self._current_gripper_command()
        return self._plan_position(target, gripper)

    def return_held_actor_to_initial(
        self,
        actor_name: str,
        initial_position: Array,
        clearance_z: float,
    ) -> Sequence[Array]:
        if actor_name not in self.actor_objects:
            raise ComponentFactoryContractError(
                f"no live actor object registered for {actor_name!r}"
            )
        actor = self.actor_objects[actor_name]
        current_actor = _position(actor.pose.p, f"actor[{actor_name!r}].pose.p")
        initial_actor = _position(initial_position, f"initial[{actor_name!r}]")
        target = self._tcp_position() + (initial_actor - current_actor)
        target[2] = max(target[2], float(clearance_z))
        return self._plan_position(target, -1.0)

    def open_gripper(self) -> Sequence[Array]:
        hold = self._hold_action(1.0)
        return tuple(
            hold.copy() for _ in range(self.calibration.open_gripper_steps)
        )

    def vertical_retract(self, clearance_z: float) -> Sequence[Array]:
        current = self._tcp_position()
        if current[2] >= float(clearance_z):
            return ()
        target = current.copy()
        target[2] = max(target[2], float(clearance_z))
        return self._plan_position(target, 1.0)

    def return_to_safe_anchor(
        self,
        anchor: Array,
        clearance_z: float,
    ) -> Sequence[Array]:
        target = _position(anchor, "last_safe_anchor")
        if target[2] + 1e-9 < clearance_z:
            raise ComponentFactoryContractError(
                "last_safe_anchor is below the calibrated clearance height"
            )
        return self._plan_position(target, self._current_gripper_command())

    def _tcp_position(self) -> Array:
        return _position(self.raw_env.agent.tcp.pose.p, "raw_env.agent.tcp.pose.p")

    def _tcp_quaternion(self) -> Array:
        return _quaternion(self.raw_env.agent.tcp.pose.q, "raw_env.agent.tcp.pose.q")

    def _robot_qpos(self) -> Array:
        qpos = _to_numpy(
            self.raw_env.agent.robot.get_qpos(),
            "raw_env.agent.robot.qpos",
        ).astype(np.float64, copy=False)
        if qpos.ndim == 2 and qpos.shape[0] == 1:
            qpos = qpos[0]
        if qpos.ndim != 1 or qpos.size < 7:
            raise ComponentFactoryContractError(
                f"robot qpos must have at least seven joints, got {qpos.shape}"
            )
        return np.array(qpos, copy=True)

    def _current_gripper_command(self) -> float:
        qpos = self._robot_qpos()
        if qpos.size == 7:
            return -1.0
        if qpos.size < 9:
            raise ComponentFactoryContractError(
                "ambiguous Panda gripper qpos layout"
            )
        return 1.0 if float(qpos[-2:].sum()) > 0.02 else -1.0

    def _hold_action(self, gripper_command: float) -> Array:
        return np.concatenate(
            [self._robot_qpos()[:7], np.asarray([gripper_command])]
        )

    def _plan_position(
        self,
        world_position: Array,
        gripper_command: float,
    ) -> tuple[Array, ...]:
        target_position = _position(world_position, "recovery target")
        world_goal = np.concatenate([target_position, self._tcp_quaternion()])
        base_goal = _to_numpy(
            self.planner.transform_goal_to_wrt_base(world_goal),
            "mplib base goal",
        ).astype(np.float64, copy=False).reshape(-1)
        if base_goal.shape != (7,):
            raise ComponentFactoryContractError(
                f"mplib base goal must have seven pose values, got {base_goal.shape}"
            )
        qpos = self._robot_qpos()
        time_step = getattr(getattr(self.solver, "base_env", None), "control_timestep", None)
        if time_step is None:
            raise ComponentFactoryContractError(
                "motion solver base_env.control_timestep is unavailable"
            )
        use_point_cloud = bool(getattr(self.solver, "use_point_cloud", False))
        common_evidence = self._planner_evidence(
            qpos=qpos,
            world_goal=world_goal,
            base_goal=base_goal,
            gripper_command=gripper_command,
            time_step=float(time_step),
            use_point_cloud=use_point_cloud,
        )
        screw_result = self._run_planner_attempt(
            plan_kind="screw",
            attempt=1,
            common_evidence=common_evidence,
            planner_call=lambda: self.planner.plan_screw(
                base_goal,
                qpos.copy(),
                time_step=float(time_step),
                use_point_cloud=use_point_cloud,
                wrt_world=False,
            ),
        )
        screw_status = self._planner_status(screw_result)
        fallback_reason = None
        if screw_status == "Success":
            screw_assessment = self._assess_path("screw", screw_result)
            rejection = screw_assessment["rejection_reason"]
            if rejection is None:
                try:
                    actions = self._actions_from_plan(
                        screw_result,
                        gripper_command,
                        plan_kind="screw",
                    )
                except ComponentFactoryContractError as error:
                    self._record_planner_result(
                        plan_kind="screw",
                        attempt=1,
                        result=screw_result,
                        assessment=screw_assessment,
                        common_evidence=common_evidence,
                        selected=False,
                        validation_error=str(error),
                    )
                    raise
                self._record_planner_result(
                    plan_kind="screw",
                    attempt=1,
                    result=screw_result,
                    assessment=screw_assessment,
                    common_evidence=common_evidence,
                    selected=True,
                )
                return actions
            self._record_planner_result(
                plan_kind="screw",
                attempt=1,
                result=screw_result,
                assessment=screw_assessment,
                common_evidence=common_evidence,
                selected=False,
            )
            fallback_reason = f"screw_quality_rejected:{rejection}"
        else:
            self._record_planner_result(
                plan_kind="screw",
                attempt=1,
                result=screw_result,
                assessment=None,
                common_evidence=common_evidence,
                selected=False,
            )
            fallback_reason = f"screw_status:{screw_status}"

        rrt_statuses = []
        for attempt in range(1, 4):
            rrt_result = self._run_planner_attempt(
                plan_kind="rrt_star",
                attempt=attempt,
                common_evidence=common_evidence,
                fallback_reason=fallback_reason,
                planner_call=lambda: self.planner.plan_qpos_to_pose(
                    base_goal,
                    qpos.copy(),
                    time_step=float(time_step),
                    use_point_cloud=use_point_cloud,
                    rrt_range=0.0,
                    planning_time=1,
                    planner_name="RRTstar",
                    wrt_world=False,
                ),
            )
            status = self._planner_status(rrt_result)
            rrt_statuses.append(status)
            if status != "Success":
                self._record_planner_result(
                    plan_kind="rrt_star",
                    attempt=attempt,
                    result=rrt_result,
                    assessment=None,
                    common_evidence=common_evidence,
                    fallback_reason=fallback_reason,
                    selected=False,
                )
                continue
            assessment = self._assess_path("rrt_star", rrt_result)
            rejection = assessment["rejection_reason"]
            if rejection is not None:
                self._record_planner_result(
                    plan_kind="rrt_star",
                    attempt=attempt,
                    result=rrt_result,
                    assessment=assessment,
                    common_evidence=common_evidence,
                    fallback_reason=fallback_reason,
                    selected=False,
                )
                raise ComponentFactoryContractError(
                    "pure recovery RRT* path failed the existing quality gate: "
                    f"{rejection}"
                )
            try:
                actions = self._actions_from_plan(
                    rrt_result,
                    gripper_command,
                    plan_kind="rrt_star",
                )
            except ComponentFactoryContractError as error:
                self._record_planner_result(
                    plan_kind="rrt_star",
                    attempt=attempt,
                    result=rrt_result,
                    assessment=assessment,
                    common_evidence=common_evidence,
                    fallback_reason=fallback_reason,
                    selected=False,
                    validation_error=str(error),
                )
                raise
            self._record_planner_result(
                plan_kind="rrt_star",
                attempt=attempt,
                result=rrt_result,
                assessment=assessment,
                common_evidence=common_evidence,
                fallback_reason=fallback_reason,
                selected=True,
            )
            return actions
        raise ComponentFactoryContractError(
            "pure recovery screw planning failed and bounded RRT* fallback "
            f"was exhausted: screw={screw_status!r}, rrt={rrt_statuses!r}"
        )

    def _actions_from_plan(
        self,
        result: Mapping[str, Any],
        gripper_command: float,
        *,
        plan_kind: str,
    ) -> tuple[Array, ...]:
        if "position" not in result:
            raise ComponentFactoryContractError(
                f"pure recovery {plan_kind} plan lacks position waypoints"
            )
        positions = _to_numpy(
            result["position"],
            f"pure recovery {plan_kind} path positions",
        )
        if positions.ndim != 2 or positions.shape[1] < 7:
            raise ComponentFactoryContractError(
                f"recovery path positions have invalid shape {positions.shape}"
            )
        if positions.shape[0] < 1:
            raise ComponentFactoryContractError(
                f"pure recovery {plan_kind} path has no waypoints"
            )
        if not np.isfinite(positions[:, :7]).all():
            raise ComponentFactoryContractError(
                f"pure recovery {plan_kind} path contains non-finite waypoints"
            )
        if positions.shape[0] > self.calibration.maximum_prefix_path_points:
            raise ComponentFactoryContractError(
                "recovery prefix path exceeds the calibrated quality bound: "
                f"{positions.shape[0]} > "
                f"{self.calibration.maximum_prefix_path_points}"
            )
        return tuple(
            np.concatenate(
                [position[:7].astype(np.float64, copy=True), [gripper_command]]
            )
            for position in positions
        )

    @staticmethod
    def _planner_status(result: Any) -> Any:
        return result.get("status") if isinstance(result, Mapping) else None

    def _run_planner_attempt(
        self,
        *,
        plan_kind: str,
        attempt: int,
        common_evidence: Mapping[str, Any],
        planner_call: Callable[[], Any],
        fallback_reason: str | None = None,
    ) -> Any:
        self._write_planner_event(
            "recovery_plan_attempt",
            phase="before_planning",
            plan_kind=plan_kind,
            attempt=attempt,
            fallback_reason=fallback_reason,
            **common_evidence,
        )
        try:
            return planner_call()
        except Exception as error:
            self._write_planner_event(
                "recovery_plan_attempt",
                phase="planner_exception",
                plan_kind=plan_kind,
                attempt=attempt,
                fallback_reason=fallback_reason,
                error_type=type(error).__name__,
                error_message=str(error),
                **common_evidence,
            )
            raise

    def _assess_path(self, plan_kind: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
        positions = np.asarray(result.get("position", []), dtype=np.float64)
        if positions.ndim != 2 or len(positions) == 0:
            return {"metrics": {"path_points": 0}, "rejection_reason": None, "warning_reason": None}
        if not np.isfinite(positions).all():
            # Do not feed NaN/Inf joint positions into Pinocchio.  Besides being
            # invalid for execution, some native FK implementations do not fail
            # safely on non-finite inputs.
            return {
                "metrics": {"path_points": int(len(positions)), "joint_path_is_finite": False},
                "rejection_reason": "nonfinite_joint_path" if self.calibration.reject_paths else None,
                "warning_reason": None,
            }

        differences = np.diff(positions, axis=0)
        step_norms = np.linalg.norm(differences, axis=1)
        joint_total_variation = float(step_norms.sum())
        joint_net = float(np.linalg.norm(positions[-1] - positions[0]))
        joint_efficiency_ratio = joint_total_variation / max(joint_net, 1e-12)

        reversal_counts = []
        for joint_index in range(positions.shape[1]):
            joint_delta = differences[:, joint_index]
            signs = np.sign(joint_delta[np.abs(joint_delta) > 1e-6])
            reversal_counts.append(int(np.sum(signs[1:] != signs[:-1])))

        metrics: dict[str, Any] = {
            "path_points": int(len(positions)),
            "joint_path_is_finite": True,
            "joint_total_variation": joint_total_variation,
            "joint_net": joint_net,
            "joint_efficiency_ratio": joint_efficiency_ratio,
            "joint_reversal_counts": reversal_counts,
            "joint_reversal_total": int(sum(reversal_counts)),
            "max_joint_step": float(np.max(np.abs(differences))) if differences.size else 0.0,
        }

        planner = getattr(self.solver, "planner", None)
        pinocchio_model = getattr(planner, "pinocchio_model", None)
        if planner is not None and pinocchio_model is not None:
            try:
                robot_qpos = self.solver.robot.get_qpos().detach().cpu().numpy()[0].astype(np.float64)
                move_group_indices = np.asarray(planner.move_group_joint_indices, dtype=np.int64)
                ee_index = planner.link_name_2_idx[planner.move_group]
                if len(positions) > 512:
                    sample_indices = np.linspace(0, len(positions) - 1, 512, dtype=np.int64)
                else:
                    sample_indices = np.arange(len(positions), dtype=np.int64)

                tcp_positions = []
                jacobian_conditions = []
                jacobian_min_singular_values = []
                for path_index in sample_indices:
                    full_qpos = robot_qpos.copy()
                    path_qpos = positions[path_index]
                    if len(path_qpos) == len(move_group_indices):
                        full_qpos[move_group_indices] = path_qpos
                    else:
                        full_qpos = path_qpos.copy()
                    pinocchio_model.compute_forward_kinematics(full_qpos)
                    tcp_pose = np.asarray(
                        pinocchio_model.get_link_pose(ee_index),
                        dtype=np.float64,
                    )
                    tcp_positions.append(tcp_pose[:3])
                    pinocchio_model.compute_full_jacobian(full_qpos)
                    jacobian = np.asarray(
                        pinocchio_model.get_link_jacobian(ee_index, local=False),
                        dtype=np.float64,
                    )
                    singular_values = np.linalg.svd(jacobian, compute_uv=False)
                    minimum = float(singular_values[-1])
                    maximum = float(singular_values[0])
                    jacobian_min_singular_values.append(minimum)
                    jacobian_conditions.append(maximum / max(minimum, 1e-12))

                tcp_positions_array = np.asarray(tcp_positions, dtype=np.float64)
                tcp_differences = np.diff(tcp_positions_array, axis=0)
                tcp_path = float(np.linalg.norm(tcp_differences, axis=1).sum())
                tcp_net = float(np.linalg.norm(tcp_positions_array[-1] - tcp_positions_array[0]))
                metrics.update(
                    tcp_path=tcp_path,
                    tcp_net=tcp_net,
                    tcp_efficiency_ratio=tcp_path / max(tcp_net, 1e-12),
                    jacobian_condition_max=float(max(jacobian_conditions)),
                    jacobian_condition_median=float(np.median(jacobian_conditions)),
                    jacobian_min_singular_value=float(min(jacobian_min_singular_values)),
                    fk_sample_count=int(len(sample_indices)),
                )
            except Exception as error:
                metrics["fk_error"] = f"{type(error).__name__}: {error}"
        rejection_reason = None
        if self.calibration.reject_paths:
            for name in (
                "joint_total_variation", "joint_net", "joint_efficiency_ratio", "max_joint_step",
                "tcp_path", "tcp_net", "tcp_efficiency_ratio", "jacobian_condition_max",
                "jacobian_condition_median", "jacobian_min_singular_value",
            ):
                value = metrics.get(name)
                if value is not None and not np.isfinite(float(value)):
                    rejection_reason = f"nonfinite_{name}"
                    break
            if rejection_reason is None and plan_kind == "screw":
                if int(metrics["path_points"]) > 160 and float(metrics["joint_efficiency_ratio"]) > 4.0:
                    rejection_reason = "long_inefficient_screw_path"
        warning_reason = None
        if (
            plan_kind == "screw"
            and int(metrics.get("path_points", 0)) > 160
            and int(metrics.get("joint_reversal_total", 0)) > 64
            and float(metrics.get("jacobian_condition_max", 0.0)) > 1e4
            and float(metrics.get("jacobian_min_singular_value", float("inf"))) < 1e-4
        ):
            warning_reason = "long_singular_oscillatory_screw_path"
        return {"metrics": metrics, "rejection_reason": rejection_reason, "warning_reason": warning_reason}

    def _record_planner_result(
        self,
        *,
        plan_kind: str,
        attempt: int,
        result: Any,
        assessment: Mapping[str, Any] | None,
        common_evidence: Mapping[str, Any],
        selected: bool,
        fallback_reason: str | None = None,
        validation_error: str | None = None,
    ) -> None:
        status = self._planner_status(result)
        positions = result.get("position") if isinstance(result, Mapping) else None
        path_shape = None
        path_digest = None
        if positions is not None:
            path_array = np.asarray(positions)
            path_shape = list(path_array.shape)
            path_digest = self._array_digest(path_array)
        metrics = None if assessment is None else assessment["metrics"]
        rejection_reason = (
            None if assessment is None else assessment["rejection_reason"]
        )
        warning_reason = (
            None if assessment is None else assessment["warning_reason"]
        )
        self._write_planner_event(
            "recovery_plan_attempt",
            phase="after_planning",
            plan_kind=plan_kind,
            attempt=attempt,
            status=status,
            path_shape=path_shape,
            path_digest=path_digest,
            metrics=metrics,
            rejection_reason=rejection_reason,
            warning_reason=warning_reason,
            fallback_reason=fallback_reason,
            validation_error=validation_error,
            selected=bool(selected),
            **common_evidence,
        )

    def _planner_evidence(
        self,
        *,
        qpos: Array,
        world_goal: Array,
        base_goal: Array,
        gripper_command: float,
        time_step: float,
        use_point_cloud: bool,
    ) -> Mapping[str, Any]:
        planner_robot = getattr(self.planner, "robot", None)
        get_base_pose = getattr(planner_robot, "get_base_pose", None)
        if not callable(get_base_pose):
            raise ComponentFactoryContractError(
                "mplib planner robot base pose is unavailable for strict telemetry"
            )
        base_pose = _to_numpy(
            get_base_pose(),
            "mplib planner base pose",
        ).astype(np.float64, copy=False).reshape(-1)
        if base_pose.shape != (7,):
            raise ComponentFactoryContractError(
                f"mplib planner base pose must have seven values, got {base_pose.shape}"
            )
        tcp_world = np.concatenate([self._tcp_position(), self._tcp_quaternion()])
        tcp_base = _to_numpy(
            self.planner.transform_goal_to_wrt_base(tcp_world),
            "mplib TCP base pose",
        ).astype(np.float64, copy=False).reshape(-1)
        lane = "main"
        if self.calibration.planner_telemetry is not None:
            parts = Path(os.path.normpath(str(self.calibration.planner_telemetry))).parts
            if "control" in parts:
                lane = "control"
            elif "enabled" in parts:
                lane = "enabled"
        return {
            "episode_step": int(getattr(self.raw_env, "elapsed_steps", -1)),
            "lane": lane,
            "qpos": qpos,
            "qpos_digest": self._array_digest(qpos),
            "tcp_world_pose": tcp_world,
            "tcp_base_pose": tcp_base,
            "robot_base_pose": base_pose,
            "world_goal": world_goal,
            "base_goal": base_goal,
            "wrt_world": False,
            "gripper_command": float(gripper_command),
            "time_step": float(time_step),
            "use_point_cloud": bool(use_point_cloud),
        }

    @staticmethod
    def _array_digest(value: Any) -> str:
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ComponentFactoryContractError(
                "planner telemetry cannot digest object arrays"
            )
        shape = json.dumps(array.shape, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(
            array.dtype.str.encode("utf-8")
            + shape
            + array.tobytes(order="C")
        ).hexdigest()

    def _write_planner_event(self, event: str, **fields: Any) -> None:
        path = self.calibration.planner_telemetry
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "time_unix": time.time(), **fields}
        with _PLANNER_EVENT_LOCK:
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(_json_value(record), sort_keys=True) + "\n")


def _actor_objects_from_scene(
    raw_env: Any,
    scene: Any,
) -> dict[str, Any]:
    """Resolve adapter names back to the exact live actor instances."""

    candidates: list[Any] = []
    for value in vars(raw_env).values():
        if isinstance(value, Mapping):
            candidates.extend(value.values())
        elif isinstance(value, (list, tuple)):
            candidates.extend(value)
        else:
            candidates.append(value)
    by_name: dict[str, list[Any]] = {}
    seen: set[int] = set()
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        name = getattr(candidate, "name", None)
        pose = getattr(candidate, "pose", None)
        if isinstance(name, str) and name and pose is not None:
            by_name.setdefault(name, []).append(candidate)

    resolved = {}
    for actor_name, actor_state in scene.actors.items():
        native_name = actor_state.attributes.get("native_name", actor_name)
        if not isinstance(native_name, str) or not native_name:
            raise ComponentFactoryContractError(
                f"actor {actor_name!r} has an invalid native_name"
            )
        matches = by_name.get(native_name, [])
        if len(matches) > 1:
            matches = [
                candidate
                for candidate in matches
                if np.array_equal(
                    _position(
                        candidate.pose.p,
                        f"live actor[{native_name!r}].pose.p",
                    ),
                    actor_state.position,
                )
            ]
        if len(matches) != 1:
            raise ComponentFactoryContractError(
                f"actor {actor_name!r} with native name {native_name!r} must "
                "resolve to exactly one live instance, "
                f"got {len(matches)}"
            )
        resolved[actor_name] = matches[0]
        live_position = _position(
            matches[0].pose.p,
            f"live actor[{actor_name!r}].pose.p",
        )
        if not np.array_equal(live_position, scene.actors[actor_name].position):
            raise ComponentFactoryContractError(
                f"adapter/live position mismatch for actor {actor_name!r}"
            )
    return resolved


def _normalized_action_predictor(
    predictor: Callable[[Array, Any], PredictedAction],
) -> Callable[[Array, Any], PredictedAction]:
    """Map normalized Panda gripper command to physical finger opening."""

    def predict(action: Array, scene: Any) -> PredictedAction:
        prediction = predictor(action, scene)
        if not isinstance(prediction, PredictedAction):
            raise ComponentFactoryContractError(
                "live action predictor must return PredictedAction"
            )
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size < 8 or not np.isfinite(action_array).all():
            raise ComponentFactoryContractError(
                "Panda joint action must contain eight finite values"
            )
        commanded_opening = 0.08 if action_array[7] > 0.0 else 0.0
        intents = set(prediction.intents)
        if commanded_opening + 1e-4 < scene.gripper_opening:
            intents.add("close_gripper")
        elif commanded_opening > scene.gripper_opening + 1e-4:
            intents.add("open_gripper")
        return dataclasses.replace(
            prediction,
            next_gripper_opening=commanded_opening,
            intents=frozenset(intents),
            metadata={
                **prediction.metadata,
                "gripper_mapping": "normalized_sign_to_0_or_0.08_m",
            },
        )

    return predict


class _EvidenceAugmentedTaskAdapter:
    """Add only launcher-provided, source-audited live evidence metrics."""

    def __init__(
        self,
        adapter: RoboMMETaskAdapter,
        raw_env: Any,
        contract: TaskInterventionContract,
        capabilities: TaskEvidenceCapabilities,
    ) -> None:
        self.adapter = adapter
        self.raw_env = raw_env
        self.contract = contract
        self.capabilities = capabilities

    @property
    def task_name(self) -> str:
        return self.adapter.task_name

    def read_scene(self) -> Any:
        scene = self.adapter.read_scene()
        metrics = dict(scene.metrics)
        for field_name in self.contract.required_live_evidence:
            metric_name = getattr(self.capabilities, field_name)
            provider_name = field_name.replace("_metric", "_provider")
            provider = getattr(self.capabilities, provider_name)
            if not isinstance(metric_name, str) or not metric_name:
                raise ComponentFactoryContractError(
                    f"{self.task_name}.{field_name} is not configured"
                )
            if not callable(provider):
                raise ComponentFactoryContractError(
                    f"{self.task_name}.{provider_name} is not installed"
                )
            try:
                value = provider(self.raw_env, scene)
            except Exception as error:
                raise ComponentFactoryContractError(
                    f"{self.task_name}.{provider_name} failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if not isinstance(value, (bool, np.bool_)):
                raise ComponentFactoryContractError(
                    f"{self.task_name}.{provider_name} must return bool"
                )
            metrics[metric_name] = bool(value)
        return dataclasses.replace(scene, metrics=metrics)

    def predict_action(
        self,
        action: Array,
        scene: Any,
    ) -> PredictedAction:
        return self.adapter.predict_action(action, scene)


def _stopcube_timing_metrics_provider(
    calibration: RuntimeCalibration,
) -> Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]:
    def provide(raw_env: Any, task_entry: Mapping[str, Any]) -> Mapping[str, Any]:
        del task_entry
        tcp = _position(raw_env.agent.tcp.pose.p, "StopCube tcp")
        button = _position(raw_env.button.pose.p, "StopCube button")
        ready_height = calibration.safe_clearance_z - 0.05
        travel = ready_height - button[2]
        if travel <= 0.0:
            raise ComponentFactoryContractError(
                "StopCube button is above the native 0.15 m ready height"
            )
        xy_distance = float(np.linalg.norm(tcp[:2] - button[:2]))
        vertical_progress = (ready_height - tcp[2]) / travel
        aligned = xy_distance <= _runtime_scale(raw_env) * 2.0
        progress = float(np.clip(vertical_progress, 0.0, 1.0)) if aligned else 0.0
        return {
            "press_progress": progress,
            "button_xy_distance": xy_distance,
            "button_xy_aligned": aligned,
            "ready_height": ready_height,
            "calibration_id": "stopcube_success_p95_20_frames_0813",
        }

    return provide


def _resolve_counterfactual_provider(
    provider: StrictCounterfactualProvider | None,
) -> tuple[
    Callable[[Array], tuple[Any, bool, str]],
    Callable[
        [InterventionDecision, Array, PolicyContinuationSnapshot],
        CounterfactualGateResult,
    ],
]:
    if provider is None:
        raise ComponentFactoryContractError(
            "strict reset-paired live counterfactual evaluator is required; "
            "single-env get_state_dict/set_state_dict and HDF5 replay are unsupported"
        )
    advance_common = getattr(provider, "advance_common_policy_action", None)
    evaluator = getattr(provider, "evaluate", None)
    if not callable(advance_common):
        raise ComponentFactoryContractError(
            "strict provider must expose advance_common_policy_action()"
        )
    if not callable(evaluator):
        raise ComponentFactoryContractError(
            "strict provider must expose evaluate()"
        )

    def policy_lockstep_step(action: Array) -> tuple[Any, bool, str]:
        result = advance_common(action)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ComponentFactoryContractError(
                "advance_common_policy_action must return the main lane "
                "(observation, stop, outcome) tuple"
            )
        observation, stop, outcome = result
        if not isinstance(stop, (bool, np.bool_)):
            raise ComponentFactoryContractError(
                "advance_common_policy_action stop value must be boolean"
            )
        if not isinstance(outcome, str):
            raise ComponentFactoryContractError(
                "advance_common_policy_action outcome must be a string"
            )
        return observation, bool(stop), outcome

    def evaluate(
        decision: InterventionDecision,
        action: Array,
        policy_continuation: PolicyContinuationSnapshot,
    ) -> CounterfactualGateResult:
        result = evaluator(decision, action, policy_continuation)
        if not isinstance(result, CounterfactualGateResult):
            raise ComponentFactoryContractError(
                "counterfactual evaluator must return CounterfactualGateResult"
            )
        return result

    return policy_lockstep_step, evaluate


def _build_recovery_context_factory(
    task_name: str,
    adapter: RoboMMETaskAdapter,
    initial_actor_positions: Mapping[str, Array],
    calibration: RuntimeCalibration,
) -> Callable[[InterventionDecision], RecoveryContext]:
    def make_context(decision: InterventionDecision) -> RecoveryContext:
        scene = adapter.read_scene()
        solver_role = ExpertSolverRole.CURRENT_AND_REMAINDER
        if scene.subtask.phase in {"place_cube", "place_in_bin", "place", "swing"}:
            solver_role = ExpertSolverRole.PLACE_AND_REMAINDER
        if scene.subtask.phase in {"route_segment", "pattern_segment"}:
            solver_role = ExpertSolverRole.ROUTE_AND_REMAINDER
        last_safe_anchor = None
        if decision.recovery_mode == RecoveryMode.RETURN_TO_SAFE_ANCHOR:
            corridor_start = scene.metrics.get("corridor_start")
            if not isinstance(corridor_start, str):
                raise ComponentFactoryContractError(
                    "route recovery lacks a semantic corridor_start"
                )
            if corridor_start not in scene.landmarks:
                raise ComponentFactoryContractError(
                    f"route recovery landmark {corridor_start!r} is absent"
                )
            last_safe_anchor = scene.landmarks[corridor_start].copy()
            last_safe_anchor[2] = max(
                last_safe_anchor[2],
                calibration.safe_clearance_z,
            )

        timing_ready = None
        maximum_hold_steps = 0
        if task_name == "StopCube":
            latest_safe = (
                calibration.stopcube_expert_completion_seconds
                + calibration.stopcube_late_margin_seconds
            )

            def timing_ready() -> bool:
                current = adapter.read_scene()
                return float(current.metrics["seconds_to_event"]) <= latest_safe

            seconds_to_event = float(scene.metrics["seconds_to_event"])
            wait_seconds = max(0.0, seconds_to_event - latest_safe)
            maximum_hold_steps = max(
                1,
                int(math.ceil(wait_seconds / calibration.step_duration_seconds)) + 2,
            )

        return RecoveryContext(
            scene=scene,
            initial_actor_positions=initial_actor_positions,
            expected_held_actor=scene.subtask.expected_actor,
            safe_clearance_z=calibration.safe_clearance_z,
            last_safe_anchor=last_safe_anchor,
            solver_role=solver_role,
            maximum_hold_steps=maximum_hold_steps,
            timing_takeover_ready=timing_ready,
        )

    return make_context


def build_robomme_v2_controller(
    *,
    env_runner: Any,
    initial_trajectory: Mapping[str, Any],
    initial_observation: Any,
    main_environment_step: Callable[[Array], tuple[Any, bool, str]],
    policy_continuation_bridge: PolicyContinuationBridge,
    reason_bridge: ReasonPersistenceBridge,
    solver_router: SolverActionRouter,
    action_predictor: Callable[[Array, Any], PredictedAction],
    strict_counterfactual_provider: StrictCounterfactualProvider,
    calibration: RuntimeCalibration | None = None,
    task_detector_calibration: TaskDetectorCalibration | None = None,
    task_evidence_capabilities: TaskEvidenceCapabilities | None = None,
) -> LiveDaggerController:
    """Build a strict live controller for one of the sixteen RoboMME tasks.

    ``initial_trajectory`` is accepted to match the smoke entrypoint, but all
    semantic identity and geometry comes from the live environment after
    reset.  Prompts and RGB observations are not used to infer actors.
    """

    del initial_trajectory
    calibration = calibration or RuntimeCalibration()
    policy_lockstep_step, counterfactual_evaluator = (
        _resolve_counterfactual_provider(strict_counterfactual_provider)
    )
    task_name = getattr(env_runner, "env_id", None)
    supported_tasks = {
        "BinFill",
        "ButtonUnmask",
        "ButtonUnmaskSwap",
        "InsertPeg",
        "MoveCube",
        "PatternLock",
        "PickHighlight",
        "PickXtimes",
        "RouteStick",
        "StopCube",
        "SwingXtimes",
        "VideoPlaceButton",
        "VideoPlaceOrder",
        "VideoRepick",
        "VideoUnmask",
        "VideoUnmaskSwap",
    }
    if task_name not in supported_tasks:
        raise ComponentFactoryContractError(
            f"live factory does not support {task_name!r}"
        )
    raw_env = getattr(getattr(env_runner, "env", None), "unwrapped", None)
    recorder = getattr(env_runner, "recorder", None)
    if raw_env is None or recorder is None:
        raise ComponentFactoryContractError(
            "env_runner must expose a reset live env and DAgger recorder"
        )
    normalized_predictor = _normalized_action_predictor(action_predictor)
    adapter_options: dict[str, Any] = {}
    if task_name == "StopCube":
        adapter_options["timing_metrics_provider"] = (
            _stopcube_timing_metrics_provider(calibration)
        )
    adapter: Any = build_robomme_task_adapter(
        task_name=task_name,
        raw_env=raw_env,
        action_predictor=normalized_predictor,
        step_duration_seconds=calibration.step_duration_seconds,
        **adapter_options,
    )
    contract = TASK_INTERVENTION_CONTRACTS[task_name]
    if contract.required_live_evidence:
        if task_evidence_capabilities is None:
            raise ComponentFactoryContractError(
                f"{task_name} requires live evidence providers before build"
            )
        adapter = _EvidenceAugmentedTaskAdapter(
            adapter,
            raw_env,
            contract,
            task_evidence_capabilities,
        )
    initial_scene = adapter.read_scene()
    actor_objects = _actor_objects_from_scene(raw_env, initial_scene)
    initial_actor_positions = {
        name: actor.position.copy()
        for name, actor in initial_scene.actors.items()
    }
    action_planners = PureMplibRecoveryActionPlanners(
        recorder=recorder,
        raw_env=raw_env,
        actor_objects=actor_objects,
        calibration=calibration,
    )
    detector = RoboMMEFactoryDetector(
        task_name=task_name,
        object_half_size=(
            _runtime_scale(raw_env)
            if task_name in {"MoveCube", "InsertPeg", "BinFill", "StopCube"}
            else None
        ),
        calibration=calibration,
        task_calibration=task_detector_calibration,
        evidence_capabilities=task_evidence_capabilities,
    )
    context_factory = _build_recovery_context_factory(
        task_name,
        adapter,
        initial_actor_positions,
        calibration,
    )
    return build_live_controller(
        env_runner=env_runner,
        initial_observation=initial_observation,
        policy_lockstep_step=policy_lockstep_step,
        main_environment_step=main_environment_step,
        reason_bridge=reason_bridge,
        solver_router=solver_router,
        task_adapter=adapter,
        task_detector=detector,
        action_planners=action_planners,
        recovery_context_factory=context_factory,
        counterfactual_evaluator=counterfactual_evaluator,
        policy_continuation_bridge=policy_continuation_bridge,
    )


def _unused_action_predictor(action: np.ndarray, scene: SceneState) -> object:
    del action, scene
    raise RuntimeError("post-action detectors do not predict simulator state")


class PostActionDetector:
    """Capture simulator truth and dispatch a real transition to one task."""

    def __init__(self, task_name: str, raw_env: Any) -> None:
        self.task_name = task_name
        self.raw_env = raw_env
        adapter_options: dict[str, Any] = {}
        if task_name == "StopCube":
            timing_calibration = RuntimeCalibration(
                detector_activate_after=1,
                stopcube_expert_completion_seconds=20.0 / 30.0,
                stopcube_early_margin_seconds=0.0,
                stopcube_late_margin_seconds=0.0,
                stopcube_minimum_ready_progress=0.25,
            )
            adapter_options["timing_metrics_provider"] = (
                _stopcube_timing_metrics_provider(timing_calibration)
            )
        self.adapter = build_robomme_task_adapter(
            task_name=task_name,
            raw_env=raw_env,
            action_predictor=_unused_action_predictor,
            step_duration_seconds=1.0 / 30.0,
            **adapter_options,
        )
        self.task_detector = build_task_detector(task_name, raw_env)

    def capture(self) -> SceneState:
        return self.adapter.read_scene()

    def observe(
        self,
        before: SceneState,
        action: np.ndarray,
        after: SceneState,
    ) -> InterventionDecision:
        return self.task_detector.observe(before, action, after)

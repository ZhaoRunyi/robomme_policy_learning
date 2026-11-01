"""dagger_review / recovery: shared behavior owned by this layer."""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
import numpy as np
from .signals import (
    InterventionDecision,
    RecoveryMode,
    SceneState,
)


Array = np.ndarray


class RecoveryContractError(ValueError):
    """Raised when semantic recovery evidence is incomplete or inconsistent."""


class RecoveryStatus(str, enum.Enum):
    COMPLETE = "complete"
    BLOCKED_BEFORE_ACTION = "blocked_before_action"
    BLOCKED_BY_LIFECYCLE = "blocked_by_lifecycle"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class RecoveryPrimitive(str, enum.Enum):
    VERTICAL_LIFT = "vertical_lift"
    RETURN_HELD_ACTOR_TO_INITIAL = "return_held_actor_to_initial"
    OPEN_GRIPPER = "open_gripper"
    VERTICAL_RETRACT = "vertical_retract"
    RETURN_TO_SAFE_ANCHOR = "return_to_safe_anchor"
    EXPERT_REMAINDER = "expert_remainder"


class ExpertSolverRole(str, enum.Enum):
    CURRENT_AND_REMAINDER = "current_and_remainder"
    PLACE_AND_REMAINDER = "place_and_remainder"
    ROUTE_AND_REMAINDER = "route_and_remainder"
    TIMED_PRESS_AND_REMAINDER = "timed_press_and_remainder"


@dataclasses.dataclass(frozen=True)
class LifecycleState:
    episode_step: int
    official_failure: bool = False
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    failure_reason: str | None = None
    task_timestep: int | None = None
    current_task_index: int | None = None
    failed_subtask_index: int | None = None
    failed_subtask_name: str | None = None
    failure_kind: str | None = None
    failure_exception_type: str | None = None
    failure_exception_message: str | None = None

    @property
    def terminal(self) -> bool:
        return bool(
            self.official_failure
            or self.terminated
            or self.truncated
            or self.success
        )


class LifecycleLatch:
    """One-way episode latch; a later false flag cannot reopen intervention."""

    def __init__(self) -> None:
        self._closed = False
        self._first_state: LifecycleState | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def first_state(self) -> LifecycleState | None:
        return self._first_state

    def observe(self, state: LifecycleState) -> None:
        if state.terminal and not self._closed:
            self._closed = True
            self._first_state = state


@dataclasses.dataclass(frozen=True)
class RecoveryPlanStep:
    primitive: RecoveryPrimitive
    actor_name: str | None = None
    position: Array | None = None
    keep_grasp: bool = False
    solver_role: ExpertSolverRole | None = None

    def __post_init__(self) -> None:
        if self.position is None:
            return
        position = np.asarray(self.position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise RecoveryContractError(
                f"recovery position must be finite shape (3,), got {position.shape}"
            )
        object.__setattr__(self, "position", position)


@dataclasses.dataclass(frozen=True)
class RecoveryContext:
    scene: SceneState
    initial_actor_positions: Mapping[str, Array] = dataclasses.field(
        default_factory=dict
    )
    expected_held_actor: str | None = None
    safe_clearance_z: float | None = None
    last_safe_anchor: Array | None = None
    solver_role: ExpertSolverRole = ExpertSolverRole.CURRENT_AND_REMAINDER
    maximum_hold_steps: int = 0
    timing_takeover_ready: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        positions = {}
        for name, value in self.initial_actor_positions.items():
            position = np.asarray(value, dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise RecoveryContractError(
                    f"initial position for {name!r} must be finite shape (3,)"
                )
            positions[name] = position
        object.__setattr__(self, "initial_actor_positions", positions)
        if self.last_safe_anchor is not None:
            anchor = np.asarray(self.last_safe_anchor, dtype=np.float64)
            if anchor.shape != (3,) or not np.isfinite(anchor).all():
                raise RecoveryContractError(
                    "last_safe_anchor must be a finite three-dimensional point"
                )
            object.__setattr__(self, "last_safe_anchor", anchor)
        if self.safe_clearance_z is not None and not math.isfinite(
            self.safe_clearance_z
        ):
            raise RecoveryContractError("safe_clearance_z must be finite")
        if self.maximum_hold_steps < 0:
            raise RecoveryContractError("maximum_hold_steps cannot be negative")


class RecoveryBackend(Protocol):
    """Robot/runtime boundary required by :class:`RecoveryDispatcher`.

    Each ``plan_*`` method must be side-effect free and must not call
    ``env.step``.  ``apply_action`` is the only operation allowed to advance
    the live rollout.
    """

    def read_lifecycle(self) -> LifecycleState:
        ...

    def plan_vertical_lift(
        self,
        clearance_z: float,
        keep_grasp: bool,
    ) -> Sequence[Any]:
        ...

    def plan_return_held_actor_to_initial(
        self,
        actor_name: str,
        initial_position: Array,
        clearance_z: float,
    ) -> Sequence[Any]:
        ...

    def plan_open_gripper(self) -> Sequence[Any]:
        ...

    def plan_vertical_retract(self, clearance_z: float) -> Sequence[Any]:
        ...

    def plan_return_to_safe_anchor(
        self,
        anchor: Array,
        clearance_z: float,
    ) -> Sequence[Any]:
        ...

    def capture_hold_action(self) -> Any:
        ...

    def plan_expert_remainder(
        self,
        role: ExpertSolverRole,
    ) -> Sequence[Any]:
        ...

    def apply_action(self, action: Any) -> LifecycleState:
        ...


@dataclasses.dataclass(frozen=True)
class RecoveryResult:
    status: RecoveryStatus
    actions_executed: int
    completed_primitives: tuple[str, ...]
    final_lifecycle: LifecycleState
    message: str

    @property
    def successful(self) -> bool:
        return self.status == RecoveryStatus.COMPLETE


class RecoveryDispatcher:
    """Build and execute a safe recovery prefix followed by native expertise."""

    def build_plan(
        self,
        decision: InterventionDecision,
        context: RecoveryContext,
    ) -> tuple[RecoveryPlanStep, ...]:
        self._validate_decision_context(decision, context)
        mode = decision.recovery_mode
        solver_role = context.solver_role

        if mode == RecoveryMode.OPEN_AND_RETRACT:
            clearance_z = self._require_clearance(context)
            if context.scene.held_actor is not None:
                raise RecoveryContractError(
                    "OPEN_AND_RETRACT cannot discard a held actor; use "
                    "RETURN_WRONG_OBJECT with its reset pose"
                )
            return (
                RecoveryPlanStep(RecoveryPrimitive.OPEN_GRIPPER),
                RecoveryPlanStep(
                    RecoveryPrimitive.VERTICAL_RETRACT,
                    position=np.asarray([0.0, 0.0, clearance_z]),
                ),
                self._solver_step(solver_role),
            )

        if mode == RecoveryMode.CANCEL_FORBIDDEN_PUSH_PICKUP:
            self._validate_forbidden_push_cancel(decision, context)
            return (
                # Cancel the predicted pickup without executing it. Holding the
                # current arm joints while opening lets the actor settle instead
                # of lifting it through MoveCube's native pickup threshold.
                RecoveryPlanStep(RecoveryPrimitive.OPEN_GRIPPER),
                self._solver_step(solver_role),
            )

        if mode == RecoveryMode.RETURN_WRONG_OBJECT:
            clearance_z = self._require_clearance(context)
            held_actor = context.scene.held_actor
            if held_actor is None or held_actor != decision.observed_actor:
                raise RecoveryContractError(
                    "RETURN_WRONG_OBJECT requires observed_actor to be the "
                    "currently held actor"
                )
            if held_actor not in context.initial_actor_positions:
                raise RecoveryContractError(
                    f"missing reset-time pose for held actor {held_actor!r}"
                )
            return (
                RecoveryPlanStep(
                    RecoveryPrimitive.VERTICAL_LIFT,
                    position=np.asarray([0.0, 0.0, clearance_z]),
                    keep_grasp=True,
                ),
                RecoveryPlanStep(
                    RecoveryPrimitive.RETURN_HELD_ACTOR_TO_INITIAL,
                    actor_name=held_actor,
                    position=context.initial_actor_positions[held_actor],
                    keep_grasp=True,
                ),
                RecoveryPlanStep(RecoveryPrimitive.OPEN_GRIPPER),
                RecoveryPlanStep(
                    RecoveryPrimitive.VERTICAL_RETRACT,
                    position=np.asarray([0.0, 0.0, clearance_z]),
                ),
                self._solver_step(solver_role),
            )

        if mode == RecoveryMode.LIFT_HOLD_THEN_PLACE:
            clearance_z = self._require_clearance(context)
            held_actor = context.scene.held_actor
            if held_actor is None:
                raise RecoveryContractError(
                    "LIFT_HOLD_THEN_PLACE requires a currently held actor"
                )
            if context.expected_held_actor is None:
                raise RecoveryContractError(
                    "transport recovery requires explicit expected_held_actor"
                )
            if held_actor != context.expected_held_actor:
                raise RecoveryContractError(
                    f"held actor {held_actor!r} is not the expected correct grasp "
                    f"{context.expected_held_actor!r}"
                )
            return (
                RecoveryPlanStep(
                    RecoveryPrimitive.VERTICAL_LIFT,
                    position=np.asarray([0.0, 0.0, clearance_z]),
                    keep_grasp=True,
                ),
                self._solver_step(ExpertSolverRole.PLACE_AND_REMAINDER),
            )

        if mode == RecoveryMode.RETURN_TO_SAFE_ANCHOR:
            self._require_clearance(context)
            if context.last_safe_anchor is None:
                raise RecoveryContractError(
                    "route recovery requires an adapter-verified last_safe_anchor"
                )
            return (
                RecoveryPlanStep(
                    RecoveryPrimitive.RETURN_TO_SAFE_ANCHOR,
                    position=context.last_safe_anchor,
                ),
                self._solver_step(ExpertSolverRole.ROUTE_AND_REMAINDER),
            )

        if mode == RecoveryMode.EXPERT_CONTINUE:
            if context.scene.task_name == "StopCube":
                self._require_timing_gate(context)
                if not context.timing_takeover_ready():
                    raise RecoveryContractError(
                        "StopCube expert takeover is not at the calibrated "
                        "latest-safe boundary"
                    )
                solver_role = ExpertSolverRole.TIMED_PRESS_AND_REMAINDER
            return (self._solver_step(solver_role),)

        if mode == RecoveryMode.HOLD_POSE:
            self._require_timing_gate(context)
            if context.maximum_hold_steps < 1:
                raise RecoveryContractError(
                    "timed hold requires a positive maximum_hold_steps bound"
                )
            return ()

        if mode == RecoveryMode.NONE:
            raise RecoveryContractError("an intervention cannot use RecoveryMode.NONE")
        raise RecoveryContractError(f"unsupported recovery mode {mode!r}")

    def execute(
        self,
        decision: InterventionDecision,
        context: RecoveryContext,
        backend: RecoveryBackend,
    ) -> RecoveryResult:
        lifecycle = backend.read_lifecycle()
        latch = LifecycleLatch()
        latch.observe(lifecycle)
        if (
            context.scene.official_failure
            or context.scene.terminal
            or latch.closed
        ):
            return RecoveryResult(
                status=RecoveryStatus.BLOCKED_BEFORE_ACTION,
                actions_executed=0,
                completed_primitives=(),
                final_lifecycle=lifecycle,
                message="official failure or terminal state already observed",
            )

        try:
            plan = self.build_plan(decision, context)
        except RecoveryContractError as error:
            return RecoveryResult(
                status=RecoveryStatus.UNSUPPORTED,
                actions_executed=0,
                completed_primitives=(),
                final_lifecycle=lifecycle,
                message=str(error),
            )

        actions_executed = 0
        completed: list[str] = []
        if decision.recovery_mode == RecoveryMode.HOLD_POSE:
            result = self._execute_timed_hold(
                context,
                backend,
                latch,
                actions_executed,
                completed,
            )
            return result

        for step in plan:
            plan_state = backend.read_lifecycle()
            latch.observe(plan_state)
            if latch.closed:
                return self._blocked_result(
                    actions_executed,
                    completed,
                    plan_state,
                    "lifecycle closed before recovery primitive",
                )
            try:
                actions = self._materialize_actions(step, context, backend)
            except RecoveryContractError as error:
                return RecoveryResult(
                    status=RecoveryStatus.FAILED,
                    actions_executed=actions_executed,
                    completed_primitives=tuple(completed),
                    final_lifecycle=backend.read_lifecycle(),
                    message=str(error),
                )
            for action in actions:
                before_action = backend.read_lifecycle()
                latch.observe(before_action)
                if latch.closed:
                    return self._blocked_result(
                        actions_executed,
                        completed,
                        before_action,
                        "lifecycle closed before applying recovery action",
                    )
                lifecycle = backend.apply_action(action)
                actions_executed += self._action_count_delta(
                    before_action,
                    lifecycle,
                )
                latch.observe(lifecycle)
                if lifecycle.success:
                    return RecoveryResult(
                        status=RecoveryStatus.COMPLETE,
                        actions_executed=actions_executed,
                        completed_primitives=tuple(
                            completed + [step.primitive.value]
                        ),
                        final_lifecycle=lifecycle,
                        message="first successful terminal frame reached",
                    )
                if latch.closed:
                    return self._blocked_result(
                        actions_executed,
                        completed,
                        lifecycle,
                        "official failure or non-success terminal during recovery",
                    )
            completed.append(step.primitive.value)

        lifecycle = backend.read_lifecycle()
        return RecoveryResult(
            status=RecoveryStatus.FAILED,
            actions_executed=actions_executed,
            completed_primitives=tuple(completed),
            final_lifecycle=lifecycle,
            message="expert remainder ended without successful terminal state",
        )

    def _execute_timed_hold(
        self,
        context: RecoveryContext,
        backend: RecoveryBackend,
        latch: LifecycleLatch,
        actions_executed: int,
        completed: list[str],
    ) -> RecoveryResult:
        readiness = self._require_timing_gate(context)
        before_capture = backend.read_lifecycle()
        hold_action = backend.capture_hold_action()
        lifecycle = backend.read_lifecycle()
        if lifecycle != before_capture:
            return RecoveryResult(
                status=RecoveryStatus.FAILED,
                actions_executed=actions_executed,
                completed_primitives=tuple(completed),
                final_lifecycle=lifecycle,
                message=(
                    "capturing the hold pose mutated the live rollout; "
                    "capture must not call env.step"
                ),
            )
        latch.observe(lifecycle)
        if latch.closed:
            return self._blocked_result(
                actions_executed,
                completed,
                lifecycle,
                "lifecycle closed while capturing the hold pose",
            )
        for _ in range(context.maximum_hold_steps):
            if readiness():
                break
            before_action = backend.read_lifecycle()
            lifecycle = backend.apply_action(hold_action)
            actions_executed += self._action_count_delta(
                before_action,
                lifecycle,
            )
            latch.observe(lifecycle)
            if lifecycle.success:
                return RecoveryResult(
                    status=RecoveryStatus.COMPLETE,
                    actions_executed=actions_executed,
                    completed_primitives=(RecoveryMode.HOLD_POSE.value,),
                    final_lifecycle=lifecycle,
                    message="timed task succeeded while holding pose",
                )
            if latch.closed:
                return self._blocked_result(
                    actions_executed,
                    completed,
                    lifecycle,
                    "official failure or terminal while holding pose",
                )
        else:
            lifecycle = backend.read_lifecycle()
            return RecoveryResult(
                status=RecoveryStatus.FAILED,
                actions_executed=actions_executed,
                completed_primitives=(RecoveryMode.HOLD_POSE.value,),
                final_lifecycle=lifecycle,
                message="timing takeover boundary was not reached within the bound",
            )

        completed.append(RecoveryMode.HOLD_POSE.value)
        solver_step = self._solver_step(
            ExpertSolverRole.TIMED_PRESS_AND_REMAINDER
        )
        actions = self._materialize_actions(solver_step, context, backend)
        for action in actions:
            before_action = backend.read_lifecycle()
            latch.observe(before_action)
            if latch.closed:
                return self._blocked_result(
                    actions_executed,
                    completed,
                    before_action,
                    "lifecycle closed before timed expert action",
                )
            lifecycle = backend.apply_action(action)
            actions_executed += self._action_count_delta(
                before_action,
                lifecycle,
            )
            latch.observe(lifecycle)
            if lifecycle.success:
                return RecoveryResult(
                    status=RecoveryStatus.COMPLETE,
                    actions_executed=actions_executed,
                    completed_primitives=tuple(
                        completed + [RecoveryPrimitive.EXPERT_REMAINDER.value]
                    ),
                    final_lifecycle=lifecycle,
                    message="timed expert reached first successful terminal frame",
                )
            if latch.closed:
                return self._blocked_result(
                    actions_executed,
                    completed,
                    lifecycle,
                    "official failure or terminal during timed expert takeover",
                )
        lifecycle = backend.read_lifecycle()
        return RecoveryResult(
            status=RecoveryStatus.FAILED,
            actions_executed=actions_executed,
            completed_primitives=tuple(completed),
            final_lifecycle=lifecycle,
            message="timed expert remainder ended without success",
        )

    @staticmethod
    def _action_count_delta(
        before: LifecycleState,
        after: LifecycleState,
    ) -> int:
        """Count real env steps, including a native expert program's batch."""

        return max(1, after.episode_step - before.episode_step)

    def _materialize_actions(
        self,
        step: RecoveryPlanStep,
        context: RecoveryContext,
        backend: RecoveryBackend,
    ) -> tuple[Any, ...]:
        before = backend.read_lifecycle()
        if step.primitive == RecoveryPrimitive.VERTICAL_LIFT:
            clearance_z = self._require_clearance(context)
            actions = backend.plan_vertical_lift(clearance_z, step.keep_grasp)
        elif step.primitive == RecoveryPrimitive.RETURN_HELD_ACTOR_TO_INITIAL:
            clearance_z = self._require_clearance(context)
            if step.actor_name is None or step.position is None:
                raise RecoveryContractError("return-held step lacks actor or pose")
            actions = backend.plan_return_held_actor_to_initial(
                step.actor_name,
                step.position,
                clearance_z,
            )
        elif step.primitive == RecoveryPrimitive.OPEN_GRIPPER:
            actions = backend.plan_open_gripper()
        elif step.primitive == RecoveryPrimitive.VERTICAL_RETRACT:
            clearance_z = self._require_clearance(context)
            actions = backend.plan_vertical_retract(clearance_z)
        elif step.primitive == RecoveryPrimitive.RETURN_TO_SAFE_ANCHOR:
            clearance_z = self._require_clearance(context)
            if step.position is None:
                raise RecoveryContractError("route recovery step lacks anchor")
            actions = backend.plan_return_to_safe_anchor(
                step.position,
                clearance_z,
            )
        elif step.primitive == RecoveryPrimitive.EXPERT_REMAINDER:
            if step.solver_role is None:
                raise RecoveryContractError("expert step lacks solver role")
            actions = backend.plan_expert_remainder(step.solver_role)
        else:
            raise RecoveryContractError(
                f"unsupported recovery primitive {step.primitive!r}"
            )
        materialized = tuple(actions)
        after = backend.read_lifecycle()
        if after != before:
            raise RecoveryContractError(
                f"planner for {step.primitive.value} mutated the live rollout; "
                "planning must not call env.step"
            )
        if not materialized:
            raise RecoveryContractError(
                f"planner for {step.primitive.value} returned no actions"
            )
        return materialized

    @staticmethod
    def _solver_step(role: ExpertSolverRole) -> RecoveryPlanStep:
        return RecoveryPlanStep(
            RecoveryPrimitive.EXPERT_REMAINDER,
            solver_role=role,
        )

    @staticmethod
    def _require_clearance(context: RecoveryContext) -> float:
        if context.safe_clearance_z is None:
            raise RecoveryContractError(
                "safe_clearance_z must be calibrated by the task adapter"
            )
        return float(context.safe_clearance_z)

    @staticmethod
    def _require_timing_gate(
        context: RecoveryContext,
    ) -> Callable[[], bool]:
        if context.timing_takeover_ready is None:
            raise RecoveryContractError(
                "timed recovery requires a calibrated live timing gate"
            )
        return context.timing_takeover_ready

    @staticmethod
    def _validate_forbidden_push_cancel(
        decision: InterventionDecision,
        context: RecoveryContext,
    ) -> None:
        scene = context.scene
        if (
            scene.task_name != "MoveCube"
            or scene.subtask.phase != "gripper_push"
            or decision.reason_code
            != "movecube.forbidden_pickup_before_push"
            or decision.metrics.get("manner") != "gripper_push"
        ):
            raise RecoveryContractError(
                "forbidden-push cancellation is scoped to MoveCube gripper_push"
            )
        if (
            scene.held_actor is None
            or scene.held_actor != decision.observed_actor
        ):
            raise RecoveryContractError(
                "forbidden-push cancellation requires the exact held actor"
            )
        try:
            current_height = float(decision.metrics["current_actor_height"])
            predicted_height = float(decision.metrics["predicted_actor_height"])
            native_height = float(decision.thresholds["native_pickup_height"])
        except (KeyError, TypeError, ValueError) as error:
            raise RecoveryContractError(
                "forbidden-push cancellation lacks finite height evidence"
            ) from error
        if not all(
            math.isfinite(value)
            for value in (current_height, predicted_height, native_height)
        ):
            raise RecoveryContractError(
                "forbidden-push cancellation height evidence is non-finite"
            )
        if not (
            decision.metrics.get("evidence_is_pre_failure") is True
            and decision.metrics.get("crosses_native_pickup_height") is True
            and current_height <= native_height < predicted_height
        ):
            raise RecoveryContractError(
                "forbidden-push cancellation is not at the pre-threshold crossing"
            )

    @staticmethod
    def _validate_decision_context(
        decision: InterventionDecision,
        context: RecoveryContext,
    ) -> None:
        scene = context.scene
        if not decision.intervene:
            raise RecoveryContractError("recovery requires intervene=True")
        if decision.task_name != scene.task_name:
            raise RecoveryContractError("decision and scene task names differ")
        if decision.step != scene.step:
            raise RecoveryContractError("decision and scene steps differ")
        if decision.subtask_index != scene.subtask.index:
            raise RecoveryContractError("decision and scene subtask indices differ")
        if scene.official_failure or scene.terminal:
            raise RecoveryContractError(
                "recovery is forbidden after official failure or terminal state"
            )

    @staticmethod
    def _blocked_result(
        actions_executed: int,
        completed: Sequence[str],
        lifecycle: LifecycleState,
        message: str,
    ) -> RecoveryResult:
        return RecoveryResult(
            status=RecoveryStatus.BLOCKED_BY_LIFECYCLE,
            actions_executed=actions_executed,
            completed_primitives=tuple(completed),
            final_lifecycle=lifecycle,
            message=message,
        )


class TaskExpertBackend:
    """Small action/planning boundary shared by task-specific experts."""

    def __init__(self, env_runner: Any, detector: Any) -> None:
        from .runtime import PureMplibRecoveryActionPlanners, RuntimeCalibration, _actor_objects_from_scene

        self.env_runner = env_runner
        self.raw_env = env_runner.env.unwrapped
        self.detector = detector
        self.calibration = RuntimeCalibration(
            detector_activate_after=1,
            stopcube_expert_completion_seconds=20.0 / 30.0,
            stopcube_early_margin_seconds=0.0,
            stopcube_late_margin_seconds=0.0,
            stopcube_minimum_ready_progress=0.25,
        )
        initial_scene = detector.capture()
        self.initial_actor_positions: Mapping[str, np.ndarray] = {
            name: actor.position.copy() for name, actor in initial_scene.actors.items()
        }
        actor_objects = _actor_objects_from_scene(self.raw_env, initial_scene)
        self.planner = PureMplibRecoveryActionPlanners(
            recorder=env_runner.recorder,
            raw_env=self.raw_env,
            actor_objects=actor_objects,
            calibration=self.calibration,
        )

    def lifecycle(self) -> LifecycleState:
        from .live import lifecycle_from_env_runner

        return lifecycle_from_env_runner(self.env_runner)

    def apply(self, actions: Sequence[np.ndarray]) -> int:
        executed = 0
        for action in actions:
            if self.lifecycle().terminal:
                break
            _, stop, outcome = self.env_runner.step(np.asarray(action))
            executed += 1
            if stop and outcome == "error":
                info = getattr(self.env_runner, "info", {})
                raise RuntimeError(f"expert recovery step failed: {info}")
        return executed

    def open_and_vertical_retract(self) -> tuple[int, tuple[str, ...]]:
        actions = [
            *self.planner.open_gripper(),
            *self.planner.vertical_retract(self.calibration.safe_clearance_z),
        ]
        return self.apply(actions), ("open_gripper", "vertical_retract")

    def open_gripper_only(self) -> tuple[int, tuple[str, ...]]:
        return self.apply(self.planner.open_gripper()), ("open_gripper",)

    def return_wrong_actor(
        self,
        actor_name: str,
    ) -> tuple[int, tuple[str, ...]]:
        if actor_name not in self.initial_actor_positions:
            raise ValueError(f"no reset pose for wrong actor {actor_name!r}")
        actions = [
            *self.planner.vertical_lift(
                self.calibration.safe_clearance_z,
                keep_grasp=True,
            ),
            *self.planner.return_held_actor_to_initial(
                actor_name,
                self.initial_actor_positions[actor_name],
                self.calibration.safe_clearance_z,
            ),
            *self.planner.open_gripper(),
            *self.planner.vertical_retract(self.calibration.safe_clearance_z),
        ]
        return self.apply(actions), (
            "vertical_lift_held_actor",
            "return_actor_to_reset",
            "open_gripper",
            "vertical_retract",
        )

    def lift_held_actor(self) -> tuple[int, tuple[str, ...]]:
        actions = self.planner.vertical_lift(
            self.calibration.safe_clearance_z,
            keep_grasp=True,
        )
        return self.apply(actions), ("vertical_lift_held_actor",)

    def route_to_native_anchor(
        self,
        anchor: np.ndarray,
    ) -> tuple[int, tuple[str, ...]]:
        target = np.asarray(anchor, dtype=np.float64).copy()
        current_tcp = self.detector.capture().tcp_position
        target[2] = current_tcp[2]
        gripper_command = self.planner._current_gripper_command()
        actions = self.planner._plan_position(target, gripper_command)
        return self.apply(actions), ("move_to_forward_route_anchor",)

    def hold_until_stopcube_press_window(
        self,
        expert_press_actions: int,
    ) -> tuple[int, tuple[str, ...]]:
        executed = 0
        maximum_hold_steps = 1300
        press_seconds = expert_press_actions / 30.0
        while executed < maximum_hold_steps:
            scene = self.detector.capture()
            seconds_to_event = float(scene.metrics["seconds_to_event"])
            if seconds_to_event <= press_seconds:
                break
            qpos = self.raw_env.agent.robot.get_qpos()
            if hasattr(qpos, "detach"):
                qpos = qpos.detach().cpu().numpy()
            qpos_array = np.asarray(qpos, dtype=np.float64).reshape(-1)
            gripper_command = 1.0 if float(qpos_array[-2:].sum()) > 0.02 else -1.0
            hold_action = np.concatenate([qpos_array[:7], [gripper_command]])
            executed += self.apply((hold_action,))
            if self.lifecycle().terminal:
                break
        return executed, ("hold_until_expert_press_window",)

    def finish_native(self, *, start_offset: int = 0, task_index: int | None = None) -> int:
        if self.lifecycle().terminal:
            return 0
        task_list = self.raw_env.task_list
        start_index = 0
        if task_index is None:
            start_index = max(0, self.detector.capture().subtask.index - start_offset)
        if task_index is not None and not 0 <= task_index < len(task_list):
            raise IndexError(f"native task index out of range: {task_index}")
        if task_index is None and start_index >= len(task_list):
            return 0
        demonstration_wrapper = self.env_runner.env.env
        demonstration_flags = [
            bool(task.get("demonstration", False)) for task in task_list
        ]
        before_step = self.lifecycle().episode_step
        self.env_runner.recorder.review_stage = "native_expert_remainder" if task_index is None else "native_expert_repair"
        for index, task in enumerate(task_list):
            task["demonstration"] = index >= start_index if task_index is None else index == task_index
        try:
            demonstration_wrapper.get_demonstration_trajectory()
        finally:
            for task, flag in zip(task_list, demonstration_flags):
                task["demonstration"] = flag
        if task_index is None:
            self.env_runner.recorder.episode_success = int(getattr(self.raw_env, "timestep", 0)) >= len(task_list)
        after_step = self.lifecycle().episode_step
        return max(0, after_step - before_step)




class TaskDaggerExpert:
    task_name = "RoboMME"

    def __init__(self, env_runner: Any, detector: Any, task_name: str | None = None) -> None:
        self.task_name = task_name or self.task_name
        self.env_runner = env_runner
        self.detector = detector
        self.backend = TaskExpertBackend(env_runner, detector)

    def recover(
        self,
        decision: InterventionDecision,
    ) -> tuple[int, tuple[str, ...]]:
        return self.pickup_or_place_recovery(decision)

    def take_over(self, decision: InterventionDecision) -> RecoveryResult:
        self.env_runner.recorder.set_expert(decision, "task_recovery")
        actions_executed = 0
        completed_primitives: tuple[str, ...] = ()
        try:
            recovered_actions, completed_primitives = self.recover(decision)
            actions_executed += recovered_actions
            actions_executed += self.finish_episode(decision)
            lifecycle = self.backend.lifecycle()
            status = (
                RecoveryStatus.COMPLETE if lifecycle.success else RecoveryStatus.FAILED
            )
            message = "task recovery and native expert completed"
        except Exception as error:
            lifecycle = self.backend.lifecycle()
            if lifecycle.success:
                status = RecoveryStatus.COMPLETE
                message = (
                    "task recovery and native expert reached success before "
                    f"{type(error).__name__}: {error}"
                )
            else:
                status = RecoveryStatus.FAILED
                message = f"{type(error).__name__}: {error}"
        return RecoveryResult(
            status=status,
            actions_executed=actions_executed,
            completed_primitives=completed_primitives,
            final_lifecycle=lifecycle,
            message=message,
        )

    def finish_episode(self, decision: InterventionDecision) -> int:
        start_offset = self.native_expert_start_offset(decision)
        return self.backend.finish_native(start_offset=start_offset)

    def native_expert_start_offset(self, decision: InterventionDecision) -> int:
        if decision.reason_code == "identity.required_actor_not_held":
            return 1
        if decision.reason_code in {
            "place.expected_target_not_aligned_descent",
            "place.wrong_target_or_release",
        }:
            return decision.subtask_index
        return 0

    def pickup_or_place_recovery(
        self,
        decision: InterventionDecision,
    ) -> tuple[int, tuple[str, ...]]:
        if decision.recovery_mode == RecoveryMode.RETURN_WRONG_OBJECT:
            if decision.observed_actor is None:
                raise ValueError("wrong-object recovery lacks observed actor")
            return self.backend.return_wrong_actor(decision.observed_actor)
        if decision.recovery_mode == RecoveryMode.LIFT_HOLD_THEN_PLACE:
            return self.backend.lift_held_actor()
        if decision.reason_code == "pickup.wrong_actor_precontact":
            return self.backend.open_gripper_only()
        if decision.recovery_mode in {
            RecoveryMode.OPEN_AND_RETRACT,
            RecoveryMode.CANCEL_FORBIDDEN_PUSH_PICKUP,
        }:
            return self.backend.open_and_vertical_retract()
        return 0, ()


class BinFillExpert(TaskDaggerExpert):
    task_name = "BinFill"


    def native_expert_start_offset(self, decision: InterventionDecision) -> int:
        if decision.reason_code == "button.prerequisite_incomplete":
            remaining_by_color = self.detector.capture().metrics.get(
                "remaining_by_color",
                {},
            )
            remaining_count = sum(int(value) for value in remaining_by_color.values())
            return min(decision.subtask_index, 2 * remaining_count)
        if decision.reason_code == "identity.wrong_actor_held":
            return 1
        return super().native_expert_start_offset(decision)

    def finish_episode(self, decision: InterventionDecision) -> int:
        from robomme.robomme_env.utils import solve_pickup

        start_offset = self.native_expert_start_offset(decision)
        preferred_actor = decision.expected_actor
        scene = self.detector.capture()
        remaining_by_color = {
            str(color): int(count)
            for color, count in scene.metrics["remaining_by_color"].items()
        }
        cube_colors = {
            str(name): str(color)
            for name, color in scene.metrics["cube_colors"].items()
        }
        if (
            scene.subtask.phase == "place_in_bin"
            and scene.held_actor in cube_colors
        ):
            held_color = cube_colors[scene.held_actor]
            remaining_by_color[held_color] = max(
                0,
                remaining_by_color[held_color] - 1,
            )

        available_names = [
            name
            for name, color in cube_colors.items()
            if remaining_by_color[color] > 0 and name in self.backend.planner.actor_objects
        ]
        planned_names: list[str] = []
        tcp_position = scene.tcp_position
        while sum(remaining_by_color.values()) > 0:
            candidates = [
                name
                for name in available_names
                if remaining_by_color[cube_colors[name]] > 0
            ]
            if not candidates:
                raise RuntimeError("BinFill lacks a live cube for a remaining color")
            if preferred_actor in candidates and not planned_names:
                selected_name = preferred_actor
            else:
                selected_name = min(
                    candidates,
                    key=lambda name: float(
                        np.linalg.norm(
                            scene.actors[name].position[:2] - tcp_position[:2]
                        )
                    ),
                )
            planned_names.append(selected_name)
            available_names.remove(selected_name)
            remaining_by_color[cube_colors[selected_name]] -= 1

        task_list = self.backend.raw_env.task_list
        start_index = max(0, scene.subtask.index - start_offset)
        pickup_indices = [
            index
            for index in range(start_index, len(task_list))
            if task_list[index].get("choice_label") == "pick up the cube"
        ]
        if len(pickup_indices) != len(planned_names):
            raise RuntimeError(
                "BinFill remaining counts do not match remaining pickup tasks: "
                f"{len(planned_names)} != {len(pickup_indices)}"
            )

        original_entries: list[tuple[int, Any, Any]] = []
        for task_index, actor_name in zip(pickup_indices, planned_names):
            task = task_list[task_index]
            original_entries.append((task_index, task["solve"], task.get("segment")))
            actor = self.backend.planner.actor_objects[actor_name]
            task["solve"] = (
                lambda env, planner, selected_actor=actor: solve_pickup(
                    env,
                    planner,
                    obj=selected_actor,
                )
            )
            task["segment"] = [actor]
        try:
            return self.backend.finish_native(start_offset=start_offset)
        finally:
            for task_index, solve, segment in original_entries:
                task_list[task_index]["solve"] = solve
                task_list[task_index]["segment"] = segment








class InsertPegExpert(TaskDaggerExpert):
    task_name = "InsertPeg"

    def recover(self, decision: InterventionDecision) -> tuple[int, tuple[str, ...]]:
        if decision.reason_code == "insert.selected_peg_not_held":
            recovery_actions, primitives = self.backend.open_and_vertical_retract()
            pickup_index = decision.subtask_index - 1
            recovery_actions += self.backend.finish_native(task_index=pickup_index)
            return recovery_actions, (*primitives, "repeat_selected_peg_pickup")
        if decision.reason_code.startswith("insert."):
            return self.backend.lift_held_actor()
        return self.pickup_or_place_recovery(decision)

    def native_expert_start_offset(self, decision: InterventionDecision) -> int:
        if decision.reason_code == "insert.selected_peg_not_held":
            return 0
        return super().native_expert_start_offset(decision)





class PatternLockExpert(TaskDaggerExpert):
    task_name = "PatternLock"

    def recover(self, decision: InterventionDecision) -> tuple[int, tuple[str, ...]]:
        del decision
        return 0, ("continue_from_current_pattern_pose",)








class RouteStickExpert(TaskDaggerExpert):
    task_name = "RouteStick"

    def recover(self, decision: InterventionDecision) -> tuple[int, tuple[str, ...]]:
        del decision
        scene = self.detector.capture()
        expected_names = tuple(scene.metrics["expected_waypoint_names"])
        if len(expected_names) != 1:
            raise ValueError("RouteStick recovery requires one expected waypoint")
        start = scene.landmarks[str(scene.metrics["corridor_start"])]
        end = scene.landmarks[expected_names[0]]
        route_vector = end[:2] - start[:2]
        route_length = float(np.linalg.norm(route_vector))
        if route_length <= 1e-12:
            raise ValueError("RouteStick recovery route has zero length")
        route_unit = route_vector / route_length
        route_perpendicular = np.asarray(
            [-route_unit[1], route_unit[0]],
            dtype=np.float64,
        )
        expected_direction = str(scene.metrics["expected_direction"])
        expected_sign = 1.0 if expected_direction == "clockwise" else -1.0
        anchor = 0.5 * (start + end)
        anchor[:2] += (
            expected_sign
            * float(scene.metrics["native_route_peak_lateral_offset"])
            * route_perpendicular
        )
        return self.backend.route_to_native_anchor(anchor)


class StopCubeExpert(TaskDaggerExpert):
    task_name = "StopCube"

    def recover(self, decision: InterventionDecision) -> tuple[int, tuple[str, ...]]:
        expert_press_actions = int(decision.thresholds["expert_press_actions"])
        return self.backend.hold_until_stopcube_press_window(expert_press_actions)


class SwingXtimesExpert(TaskDaggerExpert):
    task_name = "SwingXtimes"

    def recover(self, decision: InterventionDecision) -> tuple[int, tuple[str, ...]]:
        if decision.reason_code == "place.wrong_target_or_release":
            scene = self.detector.capture()
            if scene.held_actor is None:
                actions = self.backend.finish_native(task_index=0)
                return actions, ("repeat_target_cube_pickup",)
        if decision.reason_code == "button.prerequisite_incomplete":
            scene = self.detector.capture()
            if scene.held_actor is not None:
                return self.backend.lift_held_actor()
        return self.pickup_or_place_recovery(decision)

    def native_expert_start_offset(self, decision: InterventionDecision) -> int:
        del decision
        return 0

















TASK_EXPERTS = dict.fromkeys(
    (
        "ButtonUnmask", "ButtonUnmaskSwap", "MoveCube", "PickHighlight", "PickXtimes",
        "VideoPlaceButton", "VideoPlaceOrder", "VideoRepick", "VideoUnmask", "VideoUnmaskSwap",
    ),
    TaskDaggerExpert,
)
TASK_EXPERTS.update({expert.task_name: expert for expert in (
    BinFillExpert, InsertPegExpert, PatternLockExpert,
    RouteStickExpert, StopCubeExpert, SwingXtimesExpert,
)})


def build_dagger_expert(
    task_name: str, env_runner: Any, detector: Any
) -> TaskDaggerExpert:
    if task_name not in TASK_EXPERTS:
        raise KeyError(f"no DAgger expert for {task_name}")
    return TASK_EXPERTS[task_name](env_runner, detector, task_name=task_name)

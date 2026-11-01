"""dagger_review / live: shared behavior owned by this layer."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from .signals import InterventionDecision
import copy
import hashlib
import json
import math
import types
from collections import deque
from collections.abc import MutableMapping, Sequence
from pathlib import Path
import numpy as np
from .signals import (
    PreActionDetectorEngine,
    PredictedAction,
    SceneState,
    TaskAdapter,
    TaskDetector,
)
from .recovery import (
    ExpertSolverRole,
    LifecycleState,
    RecoveryBackend,
    RecoveryContext,
    RecoveryDispatcher,
    RecoveryResult,
    RecoveryStatus,
)


REQUIRED_CLONE_COMPONENTS = frozenset(
    {
        "physics_state",
        "task_state",
        "controller_state",
        "wrapper_state",
        "rng_state",
        "policy_state",
        "action_chunk_state",
        "detector_history",
    }
)


class SnapshotSource(str, enum.Enum):
    LIVE_EXACT_CLONE = "live_exact_clone"
    LIVE_LOCKSTEP_PAIR = "live_lockstep_pair"
    HDF5_REPLAY = "hdf5_replay"


class BranchMode(str, enum.Enum):
    CONTROL_NO_INTERVENTION = "control_no_intervention"
    ENABLED_RECOVERY = "enabled_recovery"


class BranchStatus(str, enum.Enum):
    SUCCESS = "success"
    OFFICIAL_FAILURE = "official_failure"
    HORIZON_EXHAUSTED = "horizon_exhausted"
    ERROR = "error"


class GateStatus(str, enum.Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class CloneCapability:
    supported: bool
    source: SnapshotSource
    exact: bool
    components: frozenset[str]
    detail: str = ""

    @property
    def missing_components(self) -> frozenset[str]:
        return REQUIRED_CLONE_COMPONENTS - self.components


@dataclasses.dataclass(frozen=True)
class LiveSnapshot:
    source: SnapshotSource
    episode_step: int
    prefix_digest: str
    opaque_state: Any


@dataclasses.dataclass(frozen=True)
class CounterfactualSpec:
    horizon_steps: int
    expected_failure_codes: frozenset[str]
    candidate_action_digest: str

    def __post_init__(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("counterfactual horizon must be positive")
        if not self.expected_failure_codes:
            raise ValueError("at least one expected failure code is required")
        if not self.candidate_action_digest:
            raise ValueError("candidate_action_digest must be non-empty")


@dataclasses.dataclass(frozen=True)
class BranchOutcome:
    mode: BranchMode
    status: BranchStatus
    steps: int
    initial_prefix_digest: str
    failure_code: str | None = None
    intervention_applied: bool = False
    trigger_was_pre_failure: bool = False
    terminal_count: int = 0
    post_terminal_steps: int = 0
    first_policy_action_digest: str | None = None
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class BranchExecutionError:
    """Structured evidence for a branch that raised before returning outcome."""

    mode: BranchMode
    initial_prefix_digest: str
    error_type: str
    error_message: str


class LiveBranchProvider(Protocol):
    def clone_capability(self) -> CloneCapability:
        ...

    def live_digest(self) -> str:
        ...

    def capture_live_snapshot(self) -> LiveSnapshot:
        ...

    def fork_branches(
        self,
        snapshot: LiveSnapshot,
    ) -> Mapping[BranchMode, Any]:
        ...


BranchExecutor = Callable[
    [Any, BranchMode, InterventionDecision, int],
    BranchOutcome,
]


@dataclasses.dataclass(frozen=True)
class CounterfactualGateResult:
    status: GateStatus
    message: str
    snapshot: LiveSnapshot | None = None
    control: BranchOutcome | BranchExecutionError | None = None
    enabled: BranchOutcome | BranchExecutionError | None = None
    post_evaluation_snapshot: LiveSnapshot | None = None

    @property
    def accepted(self) -> bool:
        return self.status == GateStatus.ACCEPTED


class LiveCounterfactualGate:
    """Validate that intervention prevents a named, otherwise imminent failure."""

    def evaluate(
        self,
        provider: LiveBranchProvider,
        executor: BranchExecutor,
        decision: InterventionDecision,
        spec: CounterfactualSpec,
    ) -> CounterfactualGateResult:
        if not decision.intervene:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                "counterfactual gate requires an intervention decision",
            )
        if bool(decision.metrics.get("official_failure_before_action", False)):
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                "trigger was observed after official failure",
            )

        capability = provider.clone_capability()
        unsupported = self._unsupported_capability(capability)
        if unsupported is not None:
            return CounterfactualGateResult(GateStatus.UNSUPPORTED, unsupported)

        original_digest = provider.live_digest()
        try:
            snapshot = provider.capture_live_snapshot()
        except Exception as error:
            return CounterfactualGateResult(
                GateStatus.ERROR,
                f"live snapshot capture failed: {type(error).__name__}: {error}",
            )
        if snapshot.source not in (
            SnapshotSource.LIVE_EXACT_CLONE,
            SnapshotSource.LIVE_LOCKSTEP_PAIR,
        ):
            return CounterfactualGateResult(
                GateStatus.UNSUPPORTED,
                "snapshot is not an exact live clone or verified live lockstep pair",
                snapshot=snapshot,
            )
        if snapshot.source != capability.source:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                "snapshot provenance differs from the declared clone capability",
                snapshot=snapshot,
            )
        if not snapshot.prefix_digest:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                "captured live prefix digest is empty",
                snapshot=snapshot,
            )
        if snapshot.prefix_digest != original_digest:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                "captured prefix digest differs from the live trigger state",
                snapshot=snapshot,
            )
        if snapshot.episode_step != decision.step:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                "snapshot step differs from the intervention decision step",
                snapshot=snapshot,
            )

        try:
            branches = provider.fork_branches(snapshot)
        except Exception as error:
            return CounterfactualGateResult(
                GateStatus.ERROR,
                f"live branch fork failed: {type(error).__name__}: {error}",
                snapshot=snapshot,
            )
        required_modes = {
            BranchMode.CONTROL_NO_INTERVENTION,
            BranchMode.ENABLED_RECOVERY,
        }
        if set(branches) != required_modes:
            return CounterfactualGateResult(
                GateStatus.UNSUPPORTED,
                "provider did not create exactly one control and one enabled branch",
                snapshot=snapshot,
            )

        try:
            control = executor(
                branches[BranchMode.CONTROL_NO_INTERVENTION],
                BranchMode.CONTROL_NO_INTERVENTION,
                decision,
                spec.horizon_steps,
            )
        except Exception as error:
            control_error = BranchExecutionError(
                mode=BranchMode.CONTROL_NO_INTERVENTION,
                initial_prefix_digest=snapshot.prefix_digest,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            return CounterfactualGateResult(
                GateStatus.ERROR,
                f"control branch execution failed: {type(error).__name__}: {error}",
                snapshot=snapshot,
                control=control_error,
            )

        try:
            enabled = executor(
                branches[BranchMode.ENABLED_RECOVERY],
                BranchMode.ENABLED_RECOVERY,
                decision,
                spec.horizon_steps,
            )
        except Exception as error:
            enabled_error = BranchExecutionError(
                mode=BranchMode.ENABLED_RECOVERY,
                initial_prefix_digest=snapshot.prefix_digest,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            return CounterfactualGateResult(
                GateStatus.ERROR,
                f"enabled branch execution failed: {type(error).__name__}: {error}",
                snapshot=snapshot,
                control=control,
                enabled=enabled_error,
            )

        try:
            post_evaluation_snapshot = provider.capture_live_snapshot()
        except Exception as error:
            return CounterfactualGateResult(
                GateStatus.ERROR,
                "fresh post-branch main-lane capture failed: "
                f"{type(error).__name__}: {error}",
                snapshot=snapshot,
                control=control,
                enabled=enabled,
            )
        if (
            post_evaluation_snapshot.source != snapshot.source
            or post_evaluation_snapshot.episode_step != snapshot.episode_step
            or post_evaluation_snapshot.prefix_digest != snapshot.prefix_digest
        ):
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                "fresh post-branch snapshot proves the original main lane mutated",
                snapshot=snapshot,
                control=control,
                enabled=enabled,
                post_evaluation_snapshot=post_evaluation_snapshot,
            )
        if provider.live_digest() != original_digest:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                "branch evaluation mutated the original live rollout",
                snapshot=snapshot,
                control=control,
                enabled=enabled,
                post_evaluation_snapshot=post_evaluation_snapshot,
            )
        prefix_error = self._check_prefix(snapshot, control, enabled)
        if prefix_error is not None:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                prefix_error,
                snapshot=snapshot,
                control=control,
                enabled=enabled,
                post_evaluation_snapshot=post_evaluation_snapshot,
            )
        control_error = self._check_control(control, spec)
        if control_error is not None:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                control_error,
                snapshot=snapshot,
                control=control,
                enabled=enabled,
                post_evaluation_snapshot=post_evaluation_snapshot,
            )
        enabled_error = self._check_enabled(enabled, spec)
        if enabled_error is not None:
            return CounterfactualGateResult(
                GateStatus.REJECTED,
                enabled_error,
                snapshot=snapshot,
                control=control,
                enabled=enabled,
                post_evaluation_snapshot=post_evaluation_snapshot,
            )
        return CounterfactualGateResult(
            GateStatus.ACCEPTED,
            "same-prefix live control failed as predicted and recovery succeeded",
            snapshot=snapshot,
            control=control,
            enabled=enabled,
            post_evaluation_snapshot=post_evaluation_snapshot,
        )

    @staticmethod
    def _unsupported_capability(capability: CloneCapability) -> str | None:
        if not capability.supported:
            return capability.detail or "runtime does not support live branching"
        if capability.source == SnapshotSource.HDF5_REPLAY:
            return (
                "HDF5 replay cannot prove exact long-contact counterfactual state"
            )
        if not capability.exact:
            return "runtime clone/lockstep pair is not declared exact"
        if capability.missing_components:
            missing = ", ".join(sorted(capability.missing_components))
            return f"runtime branch contract is missing state components: {missing}"
        return None

    @staticmethod
    def _check_prefix(
        snapshot: LiveSnapshot,
        control: BranchOutcome,
        enabled: BranchOutcome,
    ) -> str | None:
        if control.initial_prefix_digest != snapshot.prefix_digest:
            return "control branch does not start from the captured live prefix"
        if enabled.initial_prefix_digest != snapshot.prefix_digest:
            return "enabled branch does not start from the captured live prefix"
        if control.mode != BranchMode.CONTROL_NO_INTERVENTION:
            return "control branch outcome has the wrong mode"
        if enabled.mode != BranchMode.ENABLED_RECOVERY:
            return "enabled branch outcome has the wrong mode"
        return None

    @staticmethod
    def _check_control(
        control: BranchOutcome,
        spec: CounterfactualSpec,
    ) -> str | None:
        if control.intervention_applied:
            return "control branch unexpectedly applied an intervention"
        if control.first_policy_action_digest != spec.candidate_action_digest:
            return "control branch did not apply the captured candidate policy action"
        if control.status != BranchStatus.OFFICIAL_FAILURE:
            return "control branch did not reach an official failure within horizon"
        if control.failure_code not in spec.expected_failure_codes:
            return (
                f"control failure {control.failure_code!r} is not the named "
                "counterfactual failure"
            )
        if control.steps > spec.horizon_steps:
            return "control branch exceeded the bounded counterfactual horizon"
        if control.terminal_count != 1:
            return "control branch must latch exactly the first terminal frame"
        if control.post_terminal_steps != 0:
            return "control branch executed actions after terminal state"
        return None

    @staticmethod
    def _check_enabled(
        enabled: BranchOutcome,
        spec: CounterfactualSpec,
    ) -> str | None:
        if not enabled.intervention_applied:
            return "enabled branch did not apply the candidate intervention"
        if not enabled.trigger_was_pre_failure:
            return "enabled branch trigger was not verified pre-failure"
        if enabled.status != BranchStatus.SUCCESS:
            return "enabled recovery branch did not complete successfully"
        if enabled.steps > spec.horizon_steps:
            return "enabled branch exceeded the bounded counterfactual horizon"
        if enabled.terminal_count != 1:
            return "enabled branch must latch exactly the first terminal frame"
        if enabled.post_terminal_steps != 0:
            return "enabled branch executed actions after the terminal frame"
        return None


Array = np.ndarray
POLICY_STATE_SEAL_FIELD = "robottt_policy_state_seal"
POLICY_STATE_ATTESTATION_FIELD = "robottt_policy_state_attestation"
POLICY_STATE_SCHEMA = "robottt-policy-state-v1"


class LiveHarnessContractError(RuntimeError):
    """Raised when the real wrapper stack violates the integration contract."""


class SolverActionAfterTerminal(LiveHarnessContractError):
    """Raised when a native solver attempts an action after the terminal latch."""


class SolverActionBudgetExhausted(BaseException):
    """Stop native expert execution before it can cross a strict horizon."""


class SolverActionMultiplicityViolation(BaseException):
    """One nominal recovery action advanced more than one environment step."""


@dataclasses.dataclass(frozen=True)
class PolicyInferenceContext:
    prompt: str
    subgoal: str | None
    exec_horizon: int
    client_identity: int


@dataclasses.dataclass(frozen=True)
class PolicyContinuationSnapshot:
    candidate_action: Any
    candidate_action_digest: str
    remaining_actions: tuple[Any, ...]
    remaining_action_digests: tuple[str, ...]
    inference_context: PolicyInferenceContext
    chunk_length: int
    candidate_chunk_index: int
    policy_state_seal: str


class PolicyContinuationCursor:
    """Independent shadow cursor; it never consumes the main EpisodeState."""

    def __init__(
        self,
        snapshot: PolicyContinuationSnapshot,
        infer_next_chunk: Callable[[PolicyInferenceContext], Sequence[Any]],
    ) -> None:
        self.snapshot = snapshot
        self._pending = deque(copy.deepcopy(snapshot.remaining_actions))
        self._infer_next_chunk = infer_next_chunk
        self.inference_count = 0

    def next_action(self) -> Any:
        if not self._pending:
            next_chunk = tuple(
                copy.deepcopy(
                    tuple(self._infer_next_chunk(self.snapshot.inference_context))
                )
            )
            self.inference_count += 1
            if not next_chunk:
                raise LiveHarnessContractError(
                    "shadow policy continuation returned an empty action chunk"
                )
            self._pending.extend(next_chunk)
        return self._pending.popleft()


class PolicyContinuationBridge:
    """Capture the active evaluator deque and last inference without mutating it."""

    def __init__(
        self,
        policy_state_attestation_path: Path | None = None,
    ) -> None:
        self.active_episode_state: Any = None
        self._last_chunk: tuple[Any, ...] | None = None
        self._last_context: PolicyInferenceContext | None = None
        self._latest_policy_state_seal: str | None = None
        self._policy_state_attestation_path = policy_state_attestation_path

    def record_policy_state_response(
        self,
        response: Mapping[str, Any],
        expected_operation: str | None = None,
    ) -> None:
        seal = response.get(POLICY_STATE_SEAL_FIELD)
        attestation = response.get(POLICY_STATE_ATTESTATION_FIELD)
        if not isinstance(seal, str) or len(seal) != 64:
            raise LiveHarnessContractError(
                "policy server response has no valid sealed state digest"
            )
        if not isinstance(attestation, Mapping) or attestation.get("seal") != seal:
            raise LiveHarnessContractError(
                "policy server response attestation does not match its seal"
            )
        _validate_policy_state_attestation(attestation)
        if (
            expected_operation is not None
            and attestation.get("operation") != expected_operation
        ):
            raise LiveHarnessContractError(
                "policy state attestation operation differs from the request"
            )
        self._latest_policy_state_seal = seal

    def latest_policy_state_seal(self) -> str:
        if self._latest_policy_state_seal is None:
            raise LiveHarnessContractError(
                "policy state seal is unavailable before the first server response"
            )
        return self._latest_policy_state_seal

    def fresh_policy_state_seal(self) -> str:
        path = self._policy_state_attestation_path
        if path is None:
            raise LiveHarnessContractError(
                "policy state sidecar path was not configured"
            )
        try:
            attestation = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LiveHarnessContractError(
                f"cannot read fresh policy state sidecar: {error}"
            ) from error
        seal = attestation.get("seal")
        if not isinstance(seal, str) or len(seal) != 64:
            raise LiveHarnessContractError(
                "fresh policy state sidecar has no valid seal"
            )
        _validate_policy_state_attestation(attestation)
        if seal != self.latest_policy_state_seal():
            raise LiveHarnessContractError(
                "fresh server policy state differs from the last websocket response"
            )
        return seal

    def register_episode_state(self, episode_state: Any) -> None:
        if self.active_episode_state is not None:
            raise LiveHarnessContractError(
                "more than one active EpisodeState was registered for one smoke"
            )
        self.active_episode_state = episode_state

    def record_inference(
        self,
        *,
        client: Any,
        prompt: str,
        subgoal: str | None,
        exec_horizon: int,
        action_chunk: Sequence[Any],
    ) -> None:
        chunk = tuple(copy.deepcopy(tuple(action_chunk)))
        if not chunk:
            raise LiveHarnessContractError("policy inference returned an empty chunk")
        self._last_chunk = chunk
        self._last_context = PolicyInferenceContext(
            prompt=prompt,
            subgoal=subgoal,
            exec_horizon=int(exec_horizon),
            client_identity=id(client),
        )

    def capture(self, candidate_action: Any) -> PolicyContinuationSnapshot:
        state = self.active_episode_state
        if state is None or self._last_chunk is None or self._last_context is None:
            raise LiveHarnessContractError(
                "policy continuation is unavailable before EpisodeState and first infer"
            )
        main_remaining = tuple(state.action_plan)
        candidate_index = len(self._last_chunk) - len(main_remaining) - 1
        if candidate_index < 0 or candidate_index >= len(self._last_chunk):
            raise LiveHarnessContractError(
                "active action deque no longer aligns with the last inferred chunk"
            )
        candidate_digest = _action_digest(candidate_action)
        expected_candidate_digest = _action_digest(
            self._last_chunk[candidate_index]
        )
        if candidate_digest != expected_candidate_digest:
            raise LiveHarnessContractError(
                "current popleft action does not match the inferred chunk cursor"
            )
        expected_remaining = self._last_chunk[candidate_index + 1 :]
        if tuple(map(_action_digest, main_remaining)) != tuple(
            map(_action_digest, expected_remaining)
        ):
            raise LiveHarnessContractError(
                "main EpisodeState deque differs from the inferred chunk suffix"
            )
        return PolicyContinuationSnapshot(
            candidate_action=copy.deepcopy(candidate_action),
            candidate_action_digest=candidate_digest,
            remaining_actions=copy.deepcopy(main_remaining),
            remaining_action_digests=tuple(map(_action_digest, main_remaining)),
            inference_context=self._last_context,
            chunk_length=len(self._last_chunk),
            candidate_chunk_index=candidate_index,
            policy_state_seal=self.latest_policy_state_seal(),
        )

    @staticmethod
    def fork_cursor(
        snapshot: PolicyContinuationSnapshot,
        infer_next_chunk: Callable[[PolicyInferenceContext], Sequence[Any]],
    ) -> PolicyContinuationCursor:
        return PolicyContinuationCursor(snapshot, infer_next_chunk)


ContinuationAwareBranchExecutor = Callable[
    [
        Any,
        BranchMode,
        InterventionDecision,
        int,
        PolicyContinuationSnapshot,
    ],
    BranchOutcome,
]
CounterfactualSpecFactory = Callable[
    [InterventionDecision, Array, PolicyContinuationSnapshot],
    CounterfactualSpec,
]


class StrictCounterfactualProviderAdapter:
    """Join one three-lane provider to the strict gate and branch executor.

    Normal policy actions have exactly one owner: ``branch_provider``.  The
    main raw step is intentionally absent from this adapter, so callers cannot
    accidentally advance only the recording lane.  At a trigger, the adapter
    closes over an immutable policy-continuation snapshot while the strict
    gate runs only the control and enabled shadows.
    """

    def __init__(
        self,
        *,
        branch_provider: LiveBranchProvider,
        branch_executor: ContinuationAwareBranchExecutor,
        spec_factory: CounterfactualSpecFactory,
        gate: LiveCounterfactualGate | None = None,
    ) -> None:
        advance_common = getattr(
            branch_provider,
            "advance_common_policy_action",
            None,
        )
        if not callable(advance_common):
            raise LiveHarnessContractError(
                "strict branch provider must expose "
                "advance_common_policy_action()"
            )
        self.branch_provider = branch_provider
        self.branch_executor = branch_executor
        self.spec_factory = spec_factory
        self.gate = gate or LiveCounterfactualGate()

    def advance_common_policy_action(
        self,
        action: Array,
    ) -> tuple[Any, bool, str]:
        result = self.branch_provider.advance_common_policy_action(action)
        if not isinstance(result, tuple) or len(result) != 3:
            raise LiveHarnessContractError(
                "strict branch provider must return the main lane "
                "(observation, stop, outcome) tuple"
            )
        observation, stop, outcome = result
        if not isinstance(stop, (bool, np.bool_)) or not isinstance(outcome, str):
            raise LiveHarnessContractError(
                "strict branch provider returned an invalid main step result"
            )
        return observation, bool(stop), outcome

    def evaluate(
        self,
        decision: InterventionDecision,
        action: Array,
        policy_continuation: PolicyContinuationSnapshot,
    ) -> CounterfactualGateResult:
        candidate_digest = _action_digest(action)
        if candidate_digest != policy_continuation.candidate_action_digest:
            raise LiveHarnessContractError(
                "counterfactual candidate differs from captured policy cursor"
            )
        spec = self.spec_factory(decision, action, policy_continuation)
        if not isinstance(spec, CounterfactualSpec):
            raise LiveHarnessContractError(
                "counterfactual spec factory must return CounterfactualSpec"
            )
        if spec.candidate_action_digest != candidate_digest:
            raise LiveHarnessContractError(
                "counterfactual spec candidate digest differs from live action"
            )

        def execute_branch(
            branch: Any,
            mode: BranchMode,
            branch_decision: InterventionDecision,
            horizon_steps: int,
        ) -> BranchOutcome:
            return self.branch_executor(
                branch,
                mode,
                branch_decision,
                horizon_steps,
                policy_continuation,
            )

        return self.gate.evaluate(
            self.branch_provider,
            execute_branch,
            decision,
            spec,
        )


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _validate_policy_state_attestation(
    attestation: Mapping[str, Any],
) -> None:
    required_fields = {
        "schema",
        "sequence",
        "operation",
        "rng_digest",
        "rng_leaf_count",
        "rng_byte_count",
        "fast_state_digest",
        "fast_state_leaf_count",
        "fast_state_byte_count",
        "step_idx",
        "exec_start_idx",
        "robottt_mode",
        "seal",
        "captured_unix",
    }
    missing = required_fields - set(attestation)
    if missing:
        raise LiveHarnessContractError(
            "policy state attestation is missing fields: "
            + ", ".join(sorted(missing))
        )
    if attestation.get("schema") != POLICY_STATE_SCHEMA:
        raise LiveHarnessContractError(
            "policy state attestation uses an unknown schema"
        )
    sealed_payload = {
        key: value
        for key, value in attestation.items()
        if key not in {"seal", "captured_unix"}
    }
    encoded = json.dumps(
        sealed_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    if attestation.get("seal") != expected:
        raise LiveHarnessContractError(
            "policy state attestation payload does not reproduce its seal"
        )


def _stable_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def _action_digest(action: Any) -> str:
    try:
        array = np.asarray(action)
        payload = array.dtype.str.encode("utf-8")
        payload += str(array.shape).encode("utf-8")
        payload += np.ascontiguousarray(array).tobytes()
    except Exception:
        payload = repr(action).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class StepProvenance:
    action_source: int
    stage: str
    reason_code: str
    trigger_step: int = -1
    subtask_index: int = -1
    subtask_name: str = ""
    phase: str = ""
    expected_actor: str = ""
    observed_actor: str = ""
    recovery_mode: str = "none"
    metrics_json: str = "{}"
    thresholds_json: str = "{}"
    counterfactual_json: str = "{}"

    @classmethod
    def initial_demo(cls) -> "StepProvenance":
        return cls(
            action_source=0,
            stage="initial_or_video_demo",
            reason_code="initial_or_video_demo",
        )

    @classmethod
    def policy(cls) -> "StepProvenance":
        return cls(
            action_source=0,
            stage="policy",
            reason_code="policy",
        )

    @classmethod
    def from_decision(
        cls,
        decision: InterventionDecision,
        stage: str,
        counterfactual_result: Any,
    ) -> "StepProvenance":
        return cls(
            action_source=1,
            stage=stage,
            reason_code=decision.reason_code,
            trigger_step=decision.step,
            subtask_index=decision.subtask_index,
            subtask_name=decision.subtask_name,
            phase=decision.phase,
            expected_actor=decision.expected_actor or "",
            observed_actor=decision.observed_actor or "",
            recovery_mode=decision.recovery_mode.value,
            metrics_json=_stable_json(decision.metrics),
            thresholds_json=_stable_json(decision.thresholds),
            counterfactual_json=_stable_json(counterfactual_result),
        )

    def with_stage(self, stage: str) -> "StepProvenance":
        return dataclasses.replace(self, stage=stage)

    def as_record(self, action: Any) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["action_digest"] = _action_digest(action)
        return record


class ReasonPersistenceBridge:
    """Attach exact intervention provenance to a live DAgger recorder.

    Hooks are installed on one recorder instance after ``make_env`` and before
    reset.  The native wrapper still writes its normal schema.  After its close
    succeeds, this bridge reopens the successful episode and atomically marks
    the v2 schema complete only after every timestep and episode event is
    written.
    """

    SCHEMA_VERSION = "robottt_dagger_v2_reason_v1"

    def __init__(self, recorder: Any) -> None:
        self.recorder = recorder
        self.current = StepProvenance.initial_demo()
        self.events: list[dict[str, Any]] = []
        self._installed = False
        self._closed = False
        self._rollout_started = False
        self._original_step: Callable[..., Any] | None = None
        self._original_close: Callable[..., Any] | None = None

    def install(self) -> None:
        if self._installed:
            return
        required = ("step", "close", "buffer", "dataset_path", "episode")
        missing = [name for name in required if not hasattr(self.recorder, name)]
        if missing:
            raise LiveHarnessContractError(
                f"DAgger recorder is missing required fields: {missing}"
            )
        self._original_step = self.recorder.step
        self._original_close = self.recorder.close
        bridge = self

        def tagged_step(recorder_self: Any, action: Any) -> Any:
            buffer_start = len(recorder_self.buffer)
            recorder_self.action_source = bridge.current.action_source
            output = bridge._original_step(action)
            provenance = bridge.current.as_record(action)
            for record in recorder_self.buffer[buffer_start:]:
                record.setdefault("info", {})["_dagger_v2"] = copy.deepcopy(
                    provenance
                )
            return output

        def tagged_close(recorder_self: Any) -> Any:
            if bridge._closed:
                return None
            if not bridge._rollout_started:
                recorder_self.buffer.clear()
                for attribute_name in (
                    "video_frames",
                    "no_object_video_frames",
                ):
                    frames = getattr(recorder_self, attribute_name, None)
                    if frames is not None:
                        frames.clear()
                recorder_self.episode_success = False
                output = bridge._original_close()
                bridge._closed = True
                return output
            successful = bool(getattr(recorder_self, "episode_success", False))
            metadata = [
                copy.deepcopy(
                    record.get("info", {}).get(
                        "_dagger_v2",
                        StepProvenance.initial_demo().as_record(
                            record.get("action", {}).get("joint_action")
                        ),
                    )
                )
                for record in recorder_self.buffer
            ]
            dataset_path = Path(recorder_self.dataset_path)
            episode = int(recorder_self.episode)
            failure_evidence = _failure_evidence_from_recorder(recorder_self)
            output = bridge._original_close()
            bridge._closed = True
            if successful:
                bridge._persist(dataset_path, episode, metadata)
            else:
                bridge._persist_failed_attempt(
                    dataset_path,
                    episode,
                    metadata,
                    failure_evidence,
                )
            return output

        self.recorder.step = types.MethodType(tagged_step, self.recorder)
        self.recorder.close = types.MethodType(tagged_close, self.recorder)
        self._installed = True

    def mark_rollout_started(self) -> None:
        if not self._installed:
            raise LiveHarnessContractError(
                "reason persistence must be installed before rollout start"
            )
        if self._closed:
            raise LiveHarnessContractError(
                "cannot start rollout after recorder close"
            )
        if self._rollout_started:
            raise LiveHarnessContractError("rollout was already marked started")
        self._rollout_started = True
        self.record_event("rollout_started", {"schema_version": self.SCHEMA_VERSION})

    def set_policy(self) -> None:
        self.current = StepProvenance.policy()

    def set_intervention(
        self,
        decision: InterventionDecision,
        stage: str,
        counterfactual_result: Any,
    ) -> None:
        self.current = StepProvenance.from_decision(
            decision,
            stage,
            counterfactual_result,
        )
        self.events.append(
            {
                "event": "intervention",
                "decision": decision.as_dict(),
                "counterfactual": _json_value(counterfactual_result),
            }
        )

    def set_stage(self, stage: str) -> None:
        self.current = self.current.with_stage(stage)

    def record_event(self, event: str, payload: Any) -> None:
        self.events.append({"event": event, "payload": _json_value(payload)})

    def _persist(
        self,
        dataset_path: Path,
        episode: int,
        metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        import h5py

        with h5py.File(dataset_path, "a") as output:
            episode_name = f"episode_{episode}"
            if episode_name not in output:
                raise LiveHarnessContractError(
                    f"successful recorder output lacks {episode_name!r}"
                )
            episode_group = output[episode_name]
            expected_names = [f"timestep_{index}" for index in range(len(metadata))]
            missing = [name for name in expected_names if name not in episode_group]
            if missing:
                raise LiveHarnessContractError(
                    f"reason metadata cannot align to HDF5 timesteps: {missing[:5]}"
                )
            for timestep_name, provenance in zip(expected_names, metadata):
                info_group = episode_group[timestep_name].require_group("info")
                if "dagger_v2" in info_group:
                    del info_group["dagger_v2"]
                reason_group = info_group.create_group("dagger_v2")
                for field_name, field_value in provenance.items():
                    self._write_scalar(reason_group, field_name, field_value)

            setup_group = episode_group.require_group("setup")
            if "dagger_v2" in setup_group:
                del setup_group["dagger_v2"]
            summary_group = setup_group.create_group("dagger_v2")
            self._write_scalar(
                summary_group,
                "schema_version",
                self.SCHEMA_VERSION,
            )
            self._write_scalar(
                summary_group,
                "events_json",
                _stable_json(self.events),
            )
            summary_group.create_dataset("timestep_count", data=len(metadata))
            summary_group.create_dataset("schema_complete", data=True)
            output.flush()

    def _persist_failed_attempt(
        self,
        dataset_path: Path,
        episode: int,
        metadata: Sequence[Mapping[str, Any]],
        evidence: Mapping[str, Any],
    ) -> None:
        """Persist failure evidence outside every training ``episode_*`` group."""

        import h5py

        parent_name = f"failed_attempt_{episode}"
        staging_name = f"__{parent_name}_dagger_v2_staging"
        with h5py.File(dataset_path, "a") as output:
            if staging_name in output:
                del output[staging_name]
            staging = output.create_group(staging_name)
            self._write_scalar(staging, "schema_version", self.SCHEMA_VERSION)
            self._write_scalar(staging, "attempt_type", "failed_non_training")
            staging.create_dataset("training_eligible", data=False)
            staging.create_dataset("timestep_count", data=len(metadata))
            self._write_scalar(staging, "events_json", _stable_json(self.events))
            self._write_scalar(
                staging,
                "last_provenance_json",
                _stable_json(metadata[-1] if metadata else {}),
            )
            for field_name, field_value in evidence.items():
                self._write_scalar(staging, field_name, field_value)
            staging.create_dataset("schema_complete", data=True)
            output.flush()

            parent = output.require_group(parent_name)
            if "dagger_v2" in parent:
                del parent["dagger_v2"]
            output.move(staging_name, f"{parent_name}/dagger_v2")
            output.flush()

    @staticmethod
    def _write_scalar(group: Any, name: str, value: Any) -> None:
        import h5py

        if isinstance(value, str):
            group.create_dataset(
                name,
                data=value,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        elif isinstance(value, bool):
            group.create_dataset(name, data=value)
        elif isinstance(value, int):
            group.create_dataset(name, data=np.int64(value))
        elif isinstance(value, float):
            group.create_dataset(name, data=np.float64(value))
        else:
            group.create_dataset(
                name,
                data=_stable_json(value),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )


@dataclasses.dataclass(frozen=True)
class SolverActionTrace:
    ordinal: int
    action_digest: str
    stage: str
    terminal_after_action: bool
    status_after_action: str | None


class SolverActionRouter:
    """Expose and audit every nested action emitted by native solver closures."""

    def __init__(self, reason_bridge: ReasonPersistenceBridge) -> None:
        self.reason_bridge = reason_bridge
        self.traces: list[SolverActionTrace] = []
        self.terminal_latched = False
        self._installed = False
        self._module: Any = None
        self._original_collector: Callable[..., Any] | None = None
        self._action_budget_remaining: int | None = None
        self._action_budget_consumed = 0

    @property
    def installed(self) -> bool:
        return self._installed

    def reset_episode(self) -> None:
        self.traces.clear()
        self.terminal_latched = False

    @property
    def action_budget_active(self) -> bool:
        return self._action_budget_remaining is not None

    @property
    def action_budget_remaining(self) -> int | None:
        return self._action_budget_remaining

    @property
    def action_budget_consumed(self) -> int:
        return self._action_budget_consumed

    def begin_action_budget(self, maximum_actions: int) -> None:
        if self.action_budget_active:
            raise LiveHarnessContractError("solver action budget is already active")
        if maximum_actions < 1:
            raise LiveHarnessContractError("solver action budget must be positive")
        self._action_budget_remaining = int(maximum_actions)
        self._action_budget_consumed = 0

    def reserve_budgeted_action(self) -> None:
        if not self.action_budget_active:
            return
        if self._action_budget_remaining <= 0:
            raise SolverActionBudgetExhausted(
                "strict recovery horizon was exhausted before the next action"
            )
        self._action_budget_remaining -= 1
        self._action_budget_consumed += 1

    def validate_budgeted_step(self, before_step: int, after_step: int) -> None:
        if not self.action_budget_active:
            return
        if after_step - before_step != 1:
            raise SolverActionMultiplicityViolation(
                "one recovery action must advance exactly one environment step; "
                f"observed {before_step}->{after_step}"
            )

    def end_action_budget(self) -> tuple[int, int]:
        if not self.action_budget_active:
            raise LiveHarnessContractError("solver action budget is not active")
        consumed = self._action_budget_consumed
        remaining = int(self._action_budget_remaining)
        self._action_budget_remaining = None
        self._action_budget_consumed = 0
        return consumed, remaining

    def install(
        self,
        planner_dense_step_module: Any,
        demonstration_wrapper_type: type[Any] | None = None,
        require_lifecycle_patch: bool = True,
    ) -> None:
        if self._installed:
            return
        if require_lifecycle_patch:
            patched = bool(
                demonstration_wrapper_type is not None
                and getattr(
                    demonstration_wrapper_type,
                    "_robottt_dagger_v2_patched",
                    False,
                )
            )
            if not patched:
                raise LiveHarnessContractError(
                    "install robottt_dagger_v2_patch lifecycle guards before "
                    "routing native solver actions"
                )
        collector = getattr(planner_dense_step_module, "_collect_dense_steps", None)
        output_to_steps = getattr(planner_dense_step_module, "_output_to_steps", None)
        if not callable(collector) or not callable(output_to_steps):
            raise LiveHarnessContractError(
                "planner_denseStep lacks collector/output conversion hooks"
            )
        self._module = planner_dense_step_module
        self._original_collector = collector
        router = self

        def routed_collector(planner: Any, function: Callable[[], Any]) -> Any:
            original_step = planner.env.step

            def routed_step(action: Any) -> Any:
                if router.terminal_latched:
                    raise SolverActionAfterTerminal(
                        "native solver attempted an action after terminal latch"
                    )
                unwrapped = getattr(planner.env, "unwrapped", planner.env)
                before_step = int(getattr(unwrapped, "elapsed_steps", 0))
                router.reserve_budgeted_action()
                output = original_step(action)
                after_step = int(getattr(unwrapped, "elapsed_steps", before_step + 1))
                router.validate_budgeted_step(before_step, after_step)
                steps = output_to_steps(output)
                if not steps:
                    raise LiveHarnessContractError(
                        "native solver step returned an empty step batch"
                    )
                last_step = steps[-1]
                terminal = _scalar_bool(last_step[2]) or _scalar_bool(last_step[3])
                info = last_step[4] if isinstance(last_step[4], Mapping) else {}
                status = info.get("status")
                terminal = terminal or status in {"success", "fail", "timeout"}
                router.traces.append(
                    SolverActionTrace(
                        ordinal=len(router.traces),
                        action_digest=_action_digest(action),
                        stage=router.reason_bridge.current.stage,
                        terminal_after_action=terminal,
                        status_after_action=status,
                    )
                )
                if terminal:
                    router.terminal_latched = True
                return output

            planner.env.step = routed_step
            try:
                return collector(planner, function)
            finally:
                planner.env.step = original_step

        planner_dense_step_module._collect_dense_steps = routed_collector
        self._installed = True

    def uninstall(self) -> None:
        if self._installed and self._module is not None:
            self._module._collect_dense_steps = self._original_collector
        self._installed = False
        self._action_budget_remaining = None
        self._action_budget_consumed = 0


def _scalar_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return bool(array.reshape(-1)[0]) if array.size else False


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        return None
    try:
        scalar = array.reshape(-1)[0].item()
        integer = int(scalar)
    except (TypeError, ValueError, OverflowError):
        return None
    return integer if float(scalar) == float(integer) else None


def _subtask_context(raw_env: Any) -> dict[str, Any]:
    task_timestep = _optional_int(getattr(raw_env, "timestep", None))
    current_task_index = _optional_int(
        getattr(raw_env, "current_task_index", None)
    )
    diagnostic_index = _optional_int(
        getattr(raw_env, "_robottt_dagger_v2_failure_subtask_index", None)
    )
    failure_kind = getattr(raw_env, "_robottt_dagger_v2_failure_kind", None)
    canonical_failure_latched = bool(
        isinstance(failure_kind, str) and failure_kind.startswith("robomme.")
    )
    if canonical_failure_latched and diagnostic_index is not None:
        failed_subtask_index = diagnostic_index
    else:
        failed_subtask_index = (
            task_timestep
            if task_timestep is not None and task_timestep >= 0
            else diagnostic_index
        )
    if failed_subtask_index is None:
        failed_subtask_index = current_task_index

    diagnostic_name = getattr(
        raw_env,
        "_robottt_dagger_v2_failure_subtask_name",
        None,
    )
    if canonical_failure_latched:
        name_candidates = (
            diagnostic_name,
            getattr(raw_env, "current_task_name_online", None),
            getattr(raw_env, "current_task_name", None),
        )
    else:
        name_candidates = (
            getattr(raw_env, "current_task_name_online", None),
            diagnostic_name,
            getattr(raw_env, "current_task_name", None),
        )
    failed_subtask_name = next(
        (
            value
            for value in name_candidates
            if isinstance(value, str) and value
        ),
        None,
    )
    task_list = getattr(raw_env, "task_list", ())
    if (
        failed_subtask_name is None
        and failed_subtask_index is not None
        and isinstance(task_list, Sequence)
        and 0 <= failed_subtask_index < len(task_list)
    ):
        entry = task_list[failed_subtask_index]
        if isinstance(entry, Mapping):
            candidate = entry.get("name") or entry.get("task_name")
            if isinstance(candidate, str) and candidate:
                failed_subtask_name = candidate
    return {
        "task_timestep": task_timestep,
        "current_task_index": current_task_index,
        "failed_subtask_index": failed_subtask_index,
        "failed_subtask_name": failed_subtask_name,
    }


def _failure_context(
    raw_env: Any,
    info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    info = info or {}
    context = _subtask_context(raw_env)
    current_failure = bool(getattr(raw_env, "current_task_failure", False))
    episode_failure = _scalar_bool(getattr(raw_env, "failureflag", False))
    status = info.get("status")
    failure_kind = getattr(
        raw_env,
        "_robottt_dagger_v2_failure_kind",
        None,
    )
    if not isinstance(failure_kind, str) or not failure_kind:
        eager_values = getattr(
            raw_env,
            "_robottt_dagger_v2_eager_failure_values",
            {},
        )
        eager_value = (
            eager_values.get(context["failed_subtask_index"])
            if isinstance(eager_values, Mapping)
            else None
        )
        if (
            isinstance(eager_value, Mapping)
            and eager_value.get("coerced_true") is True
            and (
                current_failure
                or episode_failure
                or status in {"fail", "error", "timeout"}
            )
        ):
            failure_kind = "failure_eager_value_true"
        elif current_failure:
            failure_kind = "current_task_failure_uninstrumented"
        elif episode_failure:
            failure_kind = "episode_failureflag"
        elif status == "error":
            failure_kind = "runtime_exception"
        elif status == "timeout" or info.get("TimeLimit.truncated"):
            failure_kind = "timeout"
        elif status == "fail":
            failure_kind = "terminal_failure"
        else:
            failure_kind = "unspecified_failed_attempt"

    exception_type = getattr(
        raw_env,
        "_robottt_dagger_v2_failure_exception_type",
        None,
    )
    exception_message = getattr(
        raw_env,
        "_robottt_dagger_v2_failure_exception_message",
        None,
    )
    if exception_type is None:
        exception_type = info.get("exception_type")
    if exception_message is None:
        exception_message = info.get("error_message")
    context.update(
        {
            "failure_kind": failure_kind,
            "failure_reason": failure_kind,
            "failure_exception_type": exception_type,
            "failure_exception_message": exception_message,
            "terminal_status": status,
            "elapsed_steps": _optional_int(
                getattr(raw_env, "elapsed_steps", None)
            ),
            "official_current_task_failure": current_failure,
            "official_episode_failureflag": episode_failure,
        }
    )
    return context


def _failure_evidence_from_recorder(recorder: Any) -> dict[str, Any]:
    raw_env = getattr(recorder, "unwrapped", None)
    if raw_env is None:
        recorder_env = getattr(recorder, "env", None)
        raw_env = getattr(recorder_env, "unwrapped", recorder_env)
    last_info: Mapping[str, Any] = {}
    buffer = getattr(recorder, "buffer", ())
    if buffer:
        candidate = buffer[-1].get("info", {})
        if isinstance(candidate, Mapping):
            last_info = candidate
    if raw_env is None:
        return {
            "failure_kind": "missing_raw_environment",
            "failure_reason": "missing_raw_environment",
            "failed_subtask_index": None,
            "failed_subtask_name": None,
        }
    return _failure_context(raw_env, last_info)


def install_failure_diagnostics(
    raw_env: Any,
    reason_bridge: ReasonPersistenceBridge | None = None,
    coerce_failure_result: Callable[[Any], bool] | None = None,
) -> int:
    """Annotate native failure predicates without changing their result."""

    task_list = getattr(raw_env, "task_list", None)
    if not isinstance(task_list, Sequence):
        raise LiveHarnessContractError("raw environment task_list is unavailable")
    for attribute in (
        "_robottt_dagger_v2_failure_kind",
        "_robottt_dagger_v2_failure_subtask_index",
        "_robottt_dagger_v2_failure_subtask_name",
        "_robottt_dagger_v2_failure_exception_type",
        "_robottt_dagger_v2_failure_exception_message",
    ):
        if hasattr(raw_env, attribute):
            delattr(raw_env, attribute)
    raw_env._robottt_dagger_v2_eager_failure_values = {}

    def latch(
        kind: str,
        task_index: int,
        task_name: str,
        error: Exception | None = None,
    ) -> None:
        if hasattr(raw_env, "_robottt_dagger_v2_failure_kind"):
            return
        raw_env._robottt_dagger_v2_failure_kind = kind
        raw_env._robottt_dagger_v2_failure_subtask_index = task_index
        raw_env._robottt_dagger_v2_failure_subtask_name = task_name
        if error is not None:
            raw_env._robottt_dagger_v2_failure_exception_type = type(error).__name__
            raw_env._robottt_dagger_v2_failure_exception_message = str(error)
        if reason_bridge is not None:
            reason_bridge.record_event(
                kind,
                {
                    "failed_subtask_index": task_index,
                    "failed_subtask_name": task_name,
                    "task_timestep": _optional_int(
                        getattr(raw_env, "timestep", None)
                    ),
                    "elapsed_steps": _optional_int(
                        getattr(raw_env, "elapsed_steps", None)
                    ),
                    "exception_type": type(error).__name__ if error else None,
                    "exception_message": str(error) if error else None,
                },
            )

    wrapped_count = 0
    for task_index, entry in enumerate(task_list):
        if not isinstance(entry, MutableMapping):
            continue
        failure_function = entry.get("failure_func")
        if failure_function is None:
            failure_function = entry.get("failure")
        if failure_function is not None and not callable(failure_function):
            triggered = False
            coercion_error = None
            if coerce_failure_result is not None:
                try:
                    triggered = bool(coerce_failure_result(failure_function))
                except Exception as error:
                    coercion_error = f"{type(error).__name__}: {error}"
            if reason_bridge is not None:
                reason_bridge.record_event(
                    "eager_failure_value",
                    {
                        "failed_subtask_index": task_index,
                        "failed_subtask_name": (
                            entry.get("name") or entry.get("task_name") or "Unknown"
                        ),
                        "value_type": type(failure_function).__name__,
                        "coerced_true": triggered,
                        "coercion_error": coercion_error,
                    },
                )
            raw_env._robottt_dagger_v2_eager_failure_values[task_index] = {
                "value_type": type(failure_function).__name__,
                "coerced_true": triggered,
                "coercion_error": coercion_error,
            }
            continue
        if not callable(failure_function):
            continue
        if getattr(failure_function, "_robottt_dagger_v2_failure_probe", False):
            continue
        task_name = entry.get("name") or entry.get("task_name") or "Unknown"

        def wrapped_failure(
            *args: Any,
            _failure_function: Callable[..., Any] = failure_function,
            _task_index: int = task_index,
            _task_name: str = str(task_name),
            **kwargs: Any,
        ) -> Any:
            try:
                result = _failure_function(*args, **kwargs)
            except Exception as error:
                latch(
                    "failure_func_exception",
                    _task_index,
                    _task_name,
                    error,
                )
                raise
            if coerce_failure_result is not None:
                try:
                    triggered = bool(coerce_failure_result(result))
                except Exception:
                    triggered = False
                if triggered:
                    latch("failure_predicate_true", _task_index, _task_name)
            return result

        wrapped_failure._robottt_dagger_v2_failure_probe = True
        entry["failure_func"] = wrapped_failure
        wrapped_count += 1
    return wrapped_count


def lifecycle_from_env_runner(env_runner: Any) -> LifecycleState:
    raw_env = env_runner.env.unwrapped
    info = getattr(env_runner, "info", {}) or {}
    recorder = getattr(env_runner, "recorder", None)
    current_failure = bool(getattr(raw_env, "current_task_failure", False))
    episode_failure = bool(getattr(raw_env, "failureflag", False))
    status = info.get("status")
    success = bool(
        getattr(recorder, "episode_success", False)
        or status == "success"
        or getattr(raw_env, "successflag", False)
    )
    truncated = status == "timeout"
    terminated = bool(
        success
        or status in {"success", "fail", "error"}
        or getattr(raw_env, "_robottt_dagger_v2_terminal_seen", False)
    )
    failure_context = _failure_context(raw_env, info)
    failure_reason = None
    if current_failure or episode_failure or status in {"fail", "error", "timeout"}:
        failure_reason = failure_context["failure_reason"]
    return LifecycleState(
        episode_step=int(getattr(raw_env, "elapsed_steps", 0)),
        official_failure=current_failure or episode_failure,
        terminated=terminated,
        truncated=truncated,
        success=success,
        failure_reason=failure_reason,
        task_timestep=failure_context["task_timestep"],
        current_task_index=failure_context["current_task_index"],
        failed_subtask_index=failure_context["failed_subtask_index"],
        failed_subtask_name=failure_context["failed_subtask_name"],
        failure_kind=failure_context["failure_kind"],
        failure_exception_type=failure_context["failure_exception_type"],
        failure_exception_message=failure_context["failure_exception_message"],
    )


@dataclasses.dataclass(frozen=True)
class NativeExpertProgram:
    role: ExpertSolverRole


class RecoveryActionPlanners(Protocol):
    def vertical_lift(
        self,
        clearance_z: float,
        keep_grasp: bool,
    ) -> Sequence[Any]:
        ...

    def return_held_actor_to_initial(
        self,
        actor_name: str,
        initial_position: Array,
        clearance_z: float,
    ) -> Sequence[Any]:
        ...

    def open_gripper(self) -> Sequence[Any]:
        ...

    def vertical_retract(self, clearance_z: float) -> Sequence[Any]:
        ...

    def return_to_safe_anchor(
        self,
        anchor: Array,
        clearance_z: float,
    ) -> Sequence[Any]:
        ...


class EnvRunnerRecoveryBackend(RecoveryBackend):
    """Bridge safe-prefix action planners to the existing ``EnvRunner``."""

    def __init__(
        self,
        env_runner: Any,
        action_planners: RecoveryActionPlanners,
        reason_bridge: ReasonPersistenceBridge,
        solver_router: SolverActionRouter,
        environment_step: Callable[[Any], tuple[Any, bool, str]] | None = None,
    ) -> None:
        self.env_runner = env_runner
        self.action_planners = action_planners
        self.reason_bridge = reason_bridge
        self.solver_router = solver_router
        self.environment_step = environment_step or env_runner.step

    def read_lifecycle(self) -> LifecycleState:
        return lifecycle_from_env_runner(self.env_runner)

    def plan_vertical_lift(
        self,
        clearance_z: float,
        keep_grasp: bool,
    ) -> Sequence[Any]:
        return self.action_planners.vertical_lift(clearance_z, keep_grasp)

    def plan_return_held_actor_to_initial(
        self,
        actor_name: str,
        initial_position: Array,
        clearance_z: float,
    ) -> Sequence[Any]:
        return self.action_planners.return_held_actor_to_initial(
            actor_name,
            initial_position,
            clearance_z,
        )

    def plan_open_gripper(self) -> Sequence[Any]:
        return self.action_planners.open_gripper()

    def plan_vertical_retract(self, clearance_z: float) -> Sequence[Any]:
        return self.action_planners.vertical_retract(clearance_z)

    def plan_return_to_safe_anchor(
        self,
        anchor: Array,
        clearance_z: float,
    ) -> Sequence[Any]:
        return self.action_planners.return_to_safe_anchor(anchor, clearance_z)

    def capture_hold_action(self) -> Array:
        raw_env = self.env_runner.env.unwrapped
        qpos = raw_env.agent.robot.get_qpos()
        if hasattr(qpos, "detach"):
            qpos = qpos.detach().cpu().numpy()
        qpos_array = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos_array.size == 7:
            return qpos_array.copy()
        if qpos_array.size < 9:
            raise LiveHarnessContractError(
                f"cannot capture Panda hold action from qpos shape {qpos_array.shape}"
            )
        return np.concatenate([qpos_array[:7], qpos_array[7:8]])

    def plan_expert_remainder(
        self,
        role: ExpertSolverRole,
    ) -> Sequence[Any]:
        if not self.solver_router.installed:
            raise LiveHarnessContractError(
                "native expert execution requires SolverActionRouter"
            )
        return (NativeExpertProgram(role),)

    def apply_action(self, action: Any) -> LifecycleState:
        if isinstance(action, NativeExpertProgram):
            self.reason_bridge.set_stage(action.role.value)
            self.env_runner.finish_with_expert()
            return self.read_lifecycle()
        before = self.read_lifecycle()
        reserve_action = getattr(self.solver_router, "reserve_budgeted_action", None)
        if callable(reserve_action):
            reserve_action()
        _, stop, outcome = self.environment_step(action)
        if stop and outcome == "error":
            raise LiveHarnessContractError(
                f"recovery action failed: {getattr(self.env_runner, 'info', {})}"
            )
        after = self.read_lifecycle()
        validate_step = getattr(self.solver_router, "validate_budgeted_step", None)
        if callable(validate_step):
            validate_step(before.episode_step, after.episode_step)
        return after


def build_record_wrapper_action_predictor(
    recorder: Any,
    raw_env: Any,
) -> Callable[[Array, SceneState], PredictedAction]:
    """Reuse RecordWrapper FK and rigidly propagate currently attached state.

    Non-held actors remain at their current position for a one-step precontact
    prediction. The held actor and adapter-controlled points are transformed by
    the current-to-predicted TCP rigid transform. Missing FK fails closed.
    """

    fk_function = getattr(recorder, "_joint_action_to_ee_pose_dict", None)
    if not callable(fk_function):
        raise LiveHarnessContractError("record wrapper FK function is unavailable")

    def predict(action: Array, scene: SceneState) -> PredictedAction:
        saved_quat = copy.deepcopy(
            getattr(recorder, "_prev_action_ee_quat_wxyz", None)
        )
        saved_rpy = copy.deepcopy(
            getattr(recorder, "_prev_action_ee_rpy_xyz", None)
        )
        try:
            pose = fk_function(action)
        finally:
            recorder._prev_action_ee_quat_wxyz = saved_quat
            recorder._prev_action_ee_rpy_xyz = saved_rpy
        if pose is None:
            raise LiveHarnessContractError(
                "RecordWrapper FK failed for the proposed policy action"
            )
        predicted_tcp = _vector(pose["pose"], 3, "predicted TCP")
        predicted_quaternion = _vector(
            pose["quat"],
            4,
            "predicted TCP quaternion",
        )
        current_tcp_pose = raw_env.agent.tcp.pose
        current_tcp = _vector(current_tcp_pose.p, 3, "current TCP")
        current_quaternion = _vector(
            current_tcp_pose.q,
            4,
            "current TCP quaternion",
        )
        current_rotation = _quaternion_wxyz_to_matrix(current_quaternion)
        predicted_rotation = _quaternion_wxyz_to_matrix(predicted_quaternion)

        predicted_actors = {
            name: actor.position.copy() for name, actor in scene.actors.items()
        }
        if scene.held_actor is not None:
            held_position = scene.actors[scene.held_actor].position
            relative = current_rotation.T @ (held_position - current_tcp)
            predicted_actors[scene.held_actor] = (
                predicted_tcp + predicted_rotation @ relative
            )
        predicted_points = {}
        for name, position in scene.points.items():
            relative = current_rotation.T @ (position - current_tcp)
            predicted_points[name] = predicted_tcp + predicted_rotation @ relative

        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size < 7:
            raise LiveHarnessContractError("joint action must contain seven arm joints")
        if action_array.size >= 8:
            predicted_opening = 2.0 * float(action_array[7])
        else:
            predicted_opening = 0.0
        opening_delta = predicted_opening - scene.gripper_opening
        vertical_delta = float(predicted_tcp[2] - current_tcp[2])
        intents = set()
        if opening_delta < -1e-4:
            intents.add("close_gripper")
        elif opening_delta > 1e-4:
            intents.add("open_gripper")
        if vertical_delta < -1e-4:
            intents.add("descending")
        return PredictedAction(
            next_tcp_position=predicted_tcp,
            next_gripper_opening=predicted_opening,
            predicted_actor_positions=predicted_actors,
            predicted_points=predicted_points,
            intents=frozenset(intents),
            metadata={
                "predictor": "RecordWrapper FK + live rigid attachment",
                "vertical_delta": vertical_delta,
            },
        )

    return predict


def _vector(value: Any, size: int, name: str) -> Array:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise LiveHarnessContractError(
            f"{name} must be finite shape ({size},), got {vector.shape}"
        )
    return vector


def _quaternion_wxyz_to_matrix(quaternion: Array) -> Array:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise LiveHarnessContractError("TCP quaternion has zero or non-finite norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclasses.dataclass(frozen=True)
class HarnessStepResult:
    observation: Any
    stop: bool
    outcome: str
    decision: InterventionDecision
    recovery: RecoveryResult | None = None
    counterfactual: Any = None


class LiveDaggerController:
    """Per-policy-action integration point installed in place of EnvRunner.step."""

    def __init__(
        self,
        policy_lockstep_step: Callable[[Array], tuple[Any, bool, str]],
        detector_engine: PreActionDetectorEngine,
        recovery_dispatcher: RecoveryDispatcher,
        recovery_backend: RecoveryBackend,
        recovery_context_factory: Callable[
            [InterventionDecision], RecoveryContext
        ],
        counterfactual_evaluator: Callable[
            [InterventionDecision, Array, PolicyContinuationSnapshot], Any
        ],
        policy_continuation_bridge: PolicyContinuationBridge,
        reason_bridge: ReasonPersistenceBridge,
        initial_observation: Any,
    ) -> None:
        self.policy_lockstep_step = policy_lockstep_step
        self.detector_engine = detector_engine
        self.recovery_dispatcher = recovery_dispatcher
        self.recovery_backend = recovery_backend
        self.recovery_context_factory = recovery_context_factory
        self.counterfactual_evaluator = counterfactual_evaluator
        self.policy_continuation_bridge = policy_continuation_bridge
        self.reason_bridge = reason_bridge
        self.last_observation = initial_observation
        self.last_result: HarnessStepResult | None = None

    def step(self, action: Array) -> tuple[Any, bool, str]:
        decision = self.detector_engine.before_action(action)
        if not decision.intervene:
            self.reason_bridge.set_policy()
            observation, stop, outcome = self.policy_lockstep_step(action)
            if observation is not None:
                self.last_observation = observation
            self.last_result = HarnessStepResult(
                observation=observation,
                stop=stop,
                outcome=outcome,
                decision=decision,
            )
            return observation, stop, outcome

        main_lifecycle_before = self.recovery_backend.read_lifecycle()
        main_buffer_before = len(self.reason_bridge.recorder.buffer)
        main_provenance_before = self.reason_bridge.current
        main_event_count_before = len(self.reason_bridge.events)
        policy_continuation = self.policy_continuation_bridge.capture(action)
        counterfactual = self.counterfactual_evaluator(
            decision,
            action,
            policy_continuation,
        )
        main_lifecycle_after = self.recovery_backend.read_lifecycle()
        if (
            main_lifecycle_after != main_lifecycle_before
            or len(self.reason_bridge.recorder.buffer) != main_buffer_before
            or self.reason_bridge.current != main_provenance_before
            or len(self.reason_bridge.events) != main_event_count_before
        ):
            raise LiveHarnessContractError(
                "counterfactual evaluation mutated the recording main lane; "
                "control and enabled execution must use shadow runtimes"
            )
        if not isinstance(counterfactual, CounterfactualGateResult):
            raise LiveHarnessContractError(
                "counterfactual evaluator must return CounterfactualGateResult; "
                "an arbitrary accepted flag cannot prove an untouched main lane"
            )
        accepted = counterfactual.accepted
        self.reason_bridge.record_event(
            "counterfactual_gate",
            counterfactual,
        )
        if not accepted:
            message = getattr(counterfactual, "message", "counterfactual rejected")
            raise LiveHarnessContractError(
                f"strict counterfactual gate did not accept trigger: {message}"
            )
        if (
            counterfactual.snapshot is None
            or counterfactual.post_evaluation_snapshot is None
            or counterfactual.snapshot.source
            != counterfactual.post_evaluation_snapshot.source
            or counterfactual.snapshot.episode_step
            != counterfactual.post_evaluation_snapshot.episode_step
            or counterfactual.snapshot.prefix_digest
            != counterfactual.post_evaluation_snapshot.prefix_digest
        ):
            raise LiveHarnessContractError(
                "accepted counterfactual lacks a fresh unchanged-main snapshot"
            )
        self.reason_bridge.set_intervention(
            decision,
            "recovery_prefix",
            counterfactual,
        )
        context = self.recovery_context_factory(decision)
        recovery = self.recovery_dispatcher.execute(
            decision,
            context,
            self.recovery_backend,
        )
        self.reason_bridge.record_event("recovery_result", recovery)
        outcome = "success" if recovery.successful else "fail"
        self.last_result = HarnessStepResult(
            observation=self.last_observation,
            stop=True,
            outcome=outcome,
            decision=decision,
            recovery=recovery,
            counterfactual=counterfactual,
        )
        if recovery.status in {
            RecoveryStatus.UNSUPPORTED,
            RecoveryStatus.BLOCKED_BEFORE_ACTION,
            RecoveryStatus.BLOCKED_BY_LIFECYCLE,
        }:
            raise LiveHarnessContractError(
                f"recovery was not executable: {recovery.message}"
            )
        return self.last_observation, True, outcome


def build_live_controller(
    *,
    env_runner: Any,
    initial_observation: Any,
    policy_lockstep_step: Callable[[Array], tuple[Any, bool, str]],
    main_environment_step: Callable[[Array], tuple[Any, bool, str]],
    reason_bridge: ReasonPersistenceBridge,
    solver_router: SolverActionRouter,
    task_adapter: TaskAdapter,
    task_detector: TaskDetector,
    action_planners: RecoveryActionPlanners,
    recovery_context_factory: Callable[
        [InterventionDecision], RecoveryContext
    ],
    counterfactual_evaluator: Callable[
        [InterventionDecision, Array, PolicyContinuationSnapshot], Any
    ],
    policy_continuation_bridge: PolicyContinuationBridge,
) -> LiveDaggerController:
    """Wire validated task-specific components to the real one-step path."""

    detector_engine = PreActionDetectorEngine(task_adapter, task_detector)
    backend = EnvRunnerRecoveryBackend(
        env_runner=env_runner,
        action_planners=action_planners,
        reason_bridge=reason_bridge,
        solver_router=solver_router,
        environment_step=main_environment_step,
    )
    return LiveDaggerController(
        policy_lockstep_step=policy_lockstep_step,
        detector_engine=detector_engine,
        recovery_dispatcher=RecoveryDispatcher(),
        recovery_backend=backend,
        recovery_context_factory=recovery_context_factory,
        counterfactual_evaluator=counterfactual_evaluator,
        policy_continuation_bridge=policy_continuation_bridge,
        reason_bridge=reason_bridge,
        initial_observation=initial_observation,
    )

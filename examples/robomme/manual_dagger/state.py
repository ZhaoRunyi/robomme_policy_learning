"""manual_dagger / state: shared behavior owned by this layer."""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Any
import json
import os
from pathlib import Path
import threading
from dataclasses import dataclass


class Phase(str, enum.Enum):
    PREPARING = "preparing"
    READY = "ready"
    POLICY = "policy"
    PAUSED = "paused"
    INTERVENING = "intervening"
    FINISHED = "finished"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    ERROR = "error"


class PreparationCancelled(Exception):
    """The operator replaced an episode while its environment was preparing."""


class Command(str, enum.Enum):
    START = "start"
    INTERVENE = "intervene"
    ABORT = "abort"
    KEEP = "keep"
    DISCARD = "discard"
    FPS_UP = "fps_up"
    FPS_DOWN = "fps_down"
    PAUSE = "pause"
    RESET = "reset"
    NEXT_EPISODE = "next_episode"
    BLOCK_SEED = "block_seed"
    SELECT_NEXT = "select_next"
    SCHEDULE_INTERVENTION = "schedule_intervention"


KEY_COMMANDS = {
    "s": Command.START,
    "i": Command.INTERVENE,
    "p": Command.ABORT,
    "c": Command.KEEP,
    "d": Command.DISCARD,
    "+": Command.FPS_UP,
    "=": Command.FPS_UP,
    "-": Command.FPS_DOWN,
    "_": Command.FPS_DOWN,
    "z": Command.PAUSE,
    "r": Command.RESET,
    "n": Command.NEXT_EPISODE,
    "b": Command.BLOCK_SEED,
}


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    task: str
    episode: int
    take: int = 0


@dataclasses.dataclass
class SessionState:
    phase: Phase = Phase.PREPARING
    task: str = ""
    episode: int = 0
    seed: int | None = None
    difficulty: str = ""
    prompt: str = ""
    current_subtask: str = ""
    fps: int = 5
    source: str = "IDLE"
    step: int = 0
    outcome: str = ""
    eval_success: bool | None = None
    intervened: bool = False
    message: str = "Preparing episode"
    queue_index: int = 0
    queue_total: int = 0
    frame_sequence: int = 0
    command_sequence: int = 0
    last_command: str = ""
    last_command_server_ns: int | None = None
    policy_cutoff_latency_ms: float | None = None
    first_expert_action_latency_ms: float | None = None
    storage_backlog: int = 0
    policy_prefetch_ready: bool = False
    control_phase: str = "preparing"
    control_source: str = "IDLE"
    control_step: int = 0
    control_outcome: str = ""
    control_eval_success: bool | None = None
    control_message: str = "Preparing control"
    control_frame_sequence: int = 0
    control_prefetch_ready: bool = False
    control_cached: bool = False
    control_cache_ready: int = 0
    control_cache_target: int = 0
    control_video_frames: int = 0
    control_video_fps: int = 0
    episode_generation: int = 0
    enabled_generation: int = 0
    available_tasks: list[str] = dataclasses.field(default_factory=list)
    progress_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    progress_target_per_task: int = 100
    progress_total_kept: int = 0
    progress_total_target: int = 1600
    blocked_seed_count: int = 0
    next_task: str = ""
    selected_episode: int | None = None
    scheduled_intervention_step: int | None = None
    scheduled_intervention_pending: bool = False
    scheduled_intervention_fired: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["phase"] = self.phase.value
        return payload


@dataclasses.dataclass(frozen=True)
class StepResult:
    status: str
    step: int
    outcome: str = ""
    message: str = ""


@dataclasses.dataclass(frozen=True)
class CommandRequest:
    command: Command
    request_id: str
    server_receive_ns: int
    value: Any = None


def command_from_key(key: str) -> Command | None:
    return KEY_COMMANDS.get(key.lower())


def monotonic_ns() -> int:
    return time.monotonic_ns()


class CollectionProgress:
    """Track retained enabled samples and allocate stable, non-overwriting takes."""

    def __init__(
        self,
        root: Path,
        tasks: tuple[str, ...],
        episode_counts: dict[str, int],
        *,
        target_per_task: int = 100,
        episode_start: int = 0,
        training_root: Path | None = None,
    ) -> None:
        self.root = root
        self.training_root = training_root
        self.tasks = tasks
        self.episode_counts = episode_counts
        self.target_per_task = target_per_task
        self.path = root / "collection_progress.json"
        self.counts = {task: 0 for task in tasks}
        self.next_episode = {task: episode_start for task in tasks}
        self.next_take = {task: 1 for task in tasks}
        self.failure_attempts: dict[str, dict[int, int]] = {
            task: {} for task in tasks
        }
        self.blocked: dict[str, dict[int, int]] = {task: {} for task in tasks}
        self.selected: tuple[str, int | None] | None = None
        self._load()
        self._recount_artifacts()
        self._save()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        for task in self.tasks:
            self.next_episode[task] = int(
                payload.get("next_episode", {}).get(task, self.next_episode[task])
            )
            self.next_take[task] = int(
                payload.get("next_take", {}).get(task, self.next_take[task])
            )
            raw_blocked = payload.get("blocked", {}).get(task, {})
            self.blocked[task] = {
                int(episode): int(seed) for episode, seed in raw_blocked.items()
            }
            raw_attempts = payload.get("failure_attempts", {}).get(task, {})
            self.failure_attempts[task] = {
                int(episode): int(count) for episode, count in raw_attempts.items()
            }
        selected = payload.get("selected")
        if isinstance(selected, dict) and selected.get("task") in self.tasks:
            raw_episode = selected.get("episode")
            self.selected = (
                str(selected["task"]),
                None if raw_episode is None else int(raw_episode),
            )

    def _recount_artifacts(self) -> None:
        seen: set[str] = set()
        if self.training_root is not None:
            manifest_path = self.training_root / "training_manifest.json"
            try:
                entries = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "episodes",
                    [],
                )
            except (OSError, json.JSONDecodeError):
                entries = []
            for entry in entries:
                task = str(entry.get("task_name", ""))
                h5_path = Path(str(entry.get("h5_path", "")))
                if task not in self.counts or not h5_path.is_file():
                    continue
                identity = self._episode_identity(task, entry, str(h5_path))
                if identity in seen:
                    continue
                seen.add(identity)
                self.counts[task] += 1
        for task in self.tasks:
            maximum_take = 0
            for path in (self.root / task).glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("role") != "enabled" or not payload.get("kept"):
                    continue
                if not isinstance(payload.get("training_h5"), dict):
                    continue
                h5_paths = payload.get("artifacts", {}).get("h5", [])
                if len(h5_paths) != 1 or not Path(h5_paths[0]).is_file():
                    continue
                identity = self._episode_identity(
                    task,
                    payload,
                    str(payload.get("episode_id", path)),
                )
                if identity not in seen:
                    seen.add(identity)
                    self.counts[task] += 1
                episode = int(payload.get("episode", -1))
                if episode >= 0:
                    self.failure_attempts[task][episode] = max(
                        self.failure_attempts[task].get(episode, 0),
                        1,
                    )
                maximum_take = max(maximum_take, int(payload.get("take", 0)))
            self.next_take[task] = max(self.next_take[task], maximum_take + 1)

    @staticmethod
    def _episode_identity(task: str, payload: dict[str, Any], fallback: str) -> str:
        try:
            return ":".join(
                (
                    task,
                    str(int(payload["episode"])),
                    str(int(payload["seed"])),
                    str(int(payload.get("take", 0))),
                )
            )
        except (KeyError, TypeError, ValueError):
            return f"{task}:{fallback}"

    def _save(self) -> None:
        payload: dict[str, Any] = {
            "version": 1,
            "target_per_task": self.target_per_task,
            "counts": self.counts,
            "next_episode": self.next_episode,
            "next_take": self.next_take,
            "blocked": {
                task: {str(episode): seed for episode, seed in values.items()}
                for task, values in self.blocked.items()
            },
            "failure_attempts": {
                task: {str(episode): count for episode, count in values.items()}
                for task, values in self.failure_attempts.items()
            },
            "selected": (
                None
                if self.selected is None
                else {"task": self.selected[0], "episode": self.selected[1]}
            ),
            "updated_unix": time.time(),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _default_task(self, current_task: str | None) -> str:
        incomplete = [
            task for task in self.tasks if self.counts[task] < self.target_per_task
        ]
        if current_task in incomplete:
            return current_task
        if incomplete:
            return incomplete[0]
        if current_task not in self.tasks:
            return self.tasks[0]
        return self.tasks[(self.tasks.index(current_task) + 1) % len(self.tasks)]

    def preview_task(self, current_task: str | None) -> str:
        if self.selected is not None:
            return self.selected[0]
        return self._default_task(current_task)

    def select(self, task: str, episode: int | None = None) -> None:
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        if episode is not None and not 0 <= episode < self.episode_counts[task]:
            raise ValueError(f"episode {episode} is outside {task}'s legal metadata")
        self.selected = (task, episode)
        self._save()

    def allocate(self, current_task: str | None) -> EpisodeSpec:
        if self.selected is not None:
            task, selected_episode = self.selected
            self.selected = None
        else:
            task = self._default_task(current_task)
            selected_episode = None
        episode_count = self.episode_counts[task]
        if episode_count < 1:
            raise RuntimeError(f"{task} has no legal episodes")
        if selected_episode is None:
            for _ in range(episode_count):
                cursor = self.next_episode[task]
                self.next_episode[task] = cursor + 1
                episode = cursor % episode_count
                if episode not in self.blocked[task]:
                    break
            else:
                raise RuntimeError(f"every seed is blocked for {task}")
        else:
            episode = selected_episode
        take = self.next_take[task]
        self.next_take[task] = take + 1
        self._save()
        return EpisodeSpec(task, episode, take)

    def allocate_failure(
        self,
        current_task: str | None,
        candidates: dict[str, tuple[int, ...]],
    ) -> EpisodeSpec | None:
        had_selection = self.selected is not None
        if self.selected is not None:
            task, selected_episode = self.selected
            self.selected = None
        else:
            task = self._default_task(current_task)
            selected_episode = None
        available = [
            episode
            for episode in candidates.get(task, ())
            if episode not in self.blocked[task]
        ]
        if selected_episode is not None:
            if selected_episode not in available:
                raise ValueError(
                    f"episode {selected_episode} for {task} has no failed control"
                )
            episode = selected_episode
        elif available:
            episode = min(
                available,
                key=lambda item: (self.failure_attempts[task].get(item, 0), item),
            )
        else:
            if had_selection:
                self.selected = (task, selected_episode)
            self._save()
            return None
        take = self.next_take[task]
        self.next_take[task] = take + 1
        self.failure_attempts[task][episode] = (
            self.failure_attempts[task].get(episode, 0) + 1
        )
        self._save()
        return EpisodeSpec(task, episode, take)

    def record_keep(self, spec: EpisodeSpec) -> None:
        self.counts[spec.task] += 1
        self._save()

    def control_specs(self) -> tuple[EpisodeSpec, ...]:
        specs: list[EpisodeSpec] = []
        for task in self.tasks:
            episode_count = self.episode_counts[task]
            wanted = min(self.target_per_task, episode_count - len(self.blocked[task]))
            cursor = self.next_episode[task]
            task_specs: list[EpisodeSpec] = []
            while len(task_specs) < wanted:
                episode = cursor % episode_count
                cursor += 1
                if episode in self.blocked[task]:
                    continue
                spec = EpisodeSpec(task, episode, 0)
                if spec not in task_specs:
                    task_specs.append(spec)
            specs.extend(task_specs)
        return tuple(specs)

    def block(self, spec: EpisodeSpec, seed: int) -> None:
        self.blocked[spec.task][spec.episode] = seed
        self._save()

    def snapshot(self, current_task: str | None) -> dict[str, Any]:
        total_kept = sum(self.counts.values())
        total_target = self.target_per_task * len(self.tasks)
        return {
            "available_tasks": list(self.tasks),
            "progress_counts": dict(self.counts),
            "progress_target_per_task": self.target_per_task,
            "progress_total_kept": total_kept,
            "progress_total_target": total_target,
            "queue_index": total_kept + 1,
            "queue_total": total_target,
            "blocked_seed_count": sum(len(values) for values in self.blocked.values()),
            "next_task": self.preview_task(current_task),
            "selected_episode": None if self.selected is None else self.selected[1],
        }


@dataclass(frozen=True)
class ControlArtifact:
    metadata: dict[str, Any]
    video_path: Path


class ControlCache:
    """Keep failed controls available while remembering screened successes."""

    def __init__(
        self,
        root: Path,
        specs: tuple[EpisodeSpec, ...],
        mirror_root: Path | None = None,
    ) -> None:
        self.root = root
        self.specs = specs
        self.mirror_root = mirror_root
        self.state_path = root / "control_screening.json"
        self._lock = threading.Lock()
        self._successes: dict[str, dict[int, int]] = {}
        self._abandoned: dict[str, dict[int, str]] = {}
        self._failure_index: dict[str, set[int]] = {}
        self._load_state()
        self._remember_existing_successes()

    @staticmethod
    def is_failure_outcome(outcome: Any) -> bool:
        return str(outcome).strip().lower() != "success"

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._successes = {
            str(task): {int(episode): int(seed) for episode, seed in values.items()}
            for task, values in payload.get("screened_success", {}).items()
        }
        self._abandoned = {
            str(task): {int(episode): str(reason) for episode, reason in values.items()}
            for task, values in payload.get("abandoned", {}).items()
        }

    def _save_state(self) -> None:
        payload = {
            "version": 1,
            "screened_success": {
                task: {str(episode): seed for episode, seed in values.items()}
                for task, values in self._successes.items()
            },
            "abandoned": {
                task: {str(episode): reason for episode, reason in values.items()}
                for task, values in self._abandoned.items()
            },
            "updated_unix": time.time(),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(
            f".{self.state_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _artifacts(
        self,
        task: str,
        episode: int | None = None,
        seed: int | None = None,
    ) -> list[ControlArtifact]:
        task_root = self.root / task
        if not task_root.is_dir():
            return []
        pattern = (
            f"{task}_ep{episode}_seed*_control.json"
            if episode is not None
            else "*_control.json"
        )
        candidates: list[ControlArtifact] = []
        for metadata_path in task_root.glob(pattern):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata_episode = int(metadata.get("episode", -1))
                metadata_seed = int(metadata.get("seed", -1))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            video_path = metadata_path.with_suffix(".mp4")
            if (
                metadata.get("role") != "control"
                or not metadata.get("kept")
                or (episode is not None and metadata_episode != episode)
                or (seed is not None and metadata_seed != seed)
                or not video_path.is_file()
            ):
                continue
            candidates.append(ControlArtifact(metadata, video_path))
        return candidates

    @staticmethod
    def _newest(candidates: list[ControlArtifact]) -> ControlArtifact | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                bool(item.metadata.get("control_cache")),
                float(item.metadata.get("published_unix", 0.0)),
            ),
        )

    def lookup(
        self,
        task: str,
        episode: int,
        seed: int | None = None,
    ) -> ControlArtifact | None:
        """Look up any retained control, including saved-buffer compatibility."""
        return self._newest(self._artifacts(task, episode, seed))

    def lookup_failure(
        self,
        task: str,
        episode: int,
        seed: int | None = None,
    ) -> ControlArtifact | None:
        candidates = [
            artifact
            for artifact in self._artifacts(task, episode, seed)
            if artifact.metadata.get("control_cache")
            and self.is_failure_outcome(artifact.metadata.get("outcome"))
        ]
        return self._newest(candidates)

    def lookup_spec(self, spec: EpisodeSpec) -> ControlArtifact | None:
        return self.lookup(spec.task, spec.episode)

    def lookup_failure_spec(self, spec: EpisodeSpec) -> ControlArtifact | None:
        artifact = self.lookup_failure(spec.task, spec.episode)
        if artifact is not None:
            self._failure_index.setdefault(spec.task, set()).add(spec.episode)
        return artifact

    def failure_episodes(self, task: str) -> tuple[int, ...]:
        return tuple(sorted(self._failure_index.get(task, set())))

    def failure_episodes_by_task(self) -> dict[str, tuple[int, ...]]:
        return {
            task: self.failure_episodes(task)
            for task in dict.fromkeys(spec.task for spec in self.specs)
        }

    def screened(self, spec: EpisodeSpec) -> bool:
        if self.lookup_failure_spec(spec) is not None:
            return True
        with self._lock:
            if spec.episode in self._successes.get(spec.task, {}):
                return True
            if spec.episode in self._abandoned.get(spec.task, {}):
                return True
        return any(
            artifact.metadata.get("control_cache")
            for artifact in self._artifacts(spec.task, spec.episode)
        )

    def record_success(self, spec: EpisodeSpec, seed: int) -> None:
        with self._lock:
            self._successes.setdefault(spec.task, {})[spec.episode] = seed
            self._save_state()

    def abandon(self, spec: EpisodeSpec, reason: str) -> None:
        with self._lock:
            self._abandoned.setdefault(spec.task, {})[spec.episode] = reason
            self._save_state()

    def wait_for_failure(
        self,
        spec: EpisodeSpec,
        timeout: float = 30.0,
    ) -> ControlArtifact | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            artifact = self.lookup_failure_spec(spec)
            if artifact is not None:
                self._failure_index.setdefault(spec.task, set()).add(spec.episode)
                return artifact
            time.sleep(0.1)
        return self.lookup_failure_spec(spec)

    def ready_count(self) -> int:
        return sum(len(episodes) for episodes in self.failure_episodes_by_task().values())

    def _remember_existing_successes(self) -> None:
        changed = False
        for task in dict.fromkeys(spec.task for spec in self.specs):
            for artifact in self._artifacts(task):
                metadata = artifact.metadata
                if not metadata.get("control_cache"):
                    continue
                if self.is_failure_outcome(metadata.get("outcome")):
                    self._failure_index.setdefault(task, set()).add(
                        int(metadata["episode"])
                    )
                    continue
                episode = int(metadata["episode"])
                seed = int(metadata["seed"])
                if self._successes.get(task, {}).get(episode) != seed:
                    self._successes.setdefault(task, {})[episode] = seed
                    changed = True
        if changed:
            with self._lock:
                self._save_state()

    def prune_unreferenced_successes(self) -> int:
        """Remove cache-only successes without breaking retained enabled review."""
        referenced: set[tuple[str, int, int]] = set()
        for task in dict.fromkeys(spec.task for spec in self.specs):
            task_root = self.root / task
            for metadata_path in task_root.glob("*_enabled.json"):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("kept"):
                        referenced.add(
                            (task, int(metadata["episode"]), int(metadata["seed"]))
                        )
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        removed = 0
        for task in dict.fromkeys(spec.task for spec in self.specs):
            for artifact in self._artifacts(task):
                metadata = artifact.metadata
                identity = (task, int(metadata["episode"]), int(metadata["seed"]))
                if (
                    not metadata.get("control_cache")
                    or self.is_failure_outcome(metadata.get("outcome"))
                    or identity in referenced
                ):
                    continue
                for root in (self.root, self.mirror_root):
                    if root is None:
                        continue
                    task_root = root / task
                    for suffix in (".json", ".mp4"):
                        path = task_root / f"{artifact.video_path.stem}{suffix}"
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass
                removed += 1
        return removed

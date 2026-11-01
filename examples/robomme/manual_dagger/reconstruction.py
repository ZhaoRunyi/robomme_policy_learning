"""manual_dagger / reconstruction: shared behavior owned by this layer."""

from __future__ import annotations

from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
import argparse
import json
import math
import threading
import time
import traceback
from typing import Any
from manual_dagger.state import EpisodeSpec
from manual_dagger.storage import StorageProcess, write_json
from collections import Counter


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, (160, 64), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video could not be decoded: {path}")
    return frames


def even_pairs(
    old_indices: list[int],
    new_indices: list[int],
    count: int,
) -> list[tuple[int, int]]:
    size = min(count, len(old_indices), len(new_indices))
    if size <= 0:
        return []
    old_positions = np.linspace(0, len(old_indices) - 1, size, dtype=np.int32)
    new_positions = np.linspace(0, len(new_indices) - 1, size, dtype=np.int32)
    return [
        (old_indices[int(old_index)], new_indices[int(new_index)])
        for old_index, new_index in zip(old_positions, new_positions, strict=True)
    ]


def step_indices(steps: list[int], step: int) -> list[int]:
    return [index for index, value in enumerate(steps) if value == step]


def takeover_frame_pairs(
    old_steps: list[int],
    new_steps: list[int],
    takeover_step: int,
) -> dict[str, tuple[int, int]]:
    old_start = step_indices(old_steps, 1)
    new_start = step_indices(new_steps, 1)
    old_takeover = step_indices(old_steps, takeover_step)
    new_takeover = step_indices(new_steps, takeover_step)
    old_expert = step_indices(old_steps, takeover_step + 1)
    new_expert = step_indices(new_steps, takeover_step + 1)
    if not old_start or not new_start:
        raise ValueError("missing_execution_start")
    if not old_takeover or not new_takeover:
        raise ValueError("missing_policy_takeover_frame")
    if not old_expert or not new_expert:
        raise ValueError("missing_first_expert_frame")
    return {
        # Legacy videos can contain a long conditioning demonstration at step 1.
        "execution_start": (old_start[-1], new_start[-1]),
        "policy_takeover": (old_takeover[-1], new_takeover[-1]),
        "first_expert": (old_expert[0], new_expert[0]),
    }


def camera_mae(old_frame: np.ndarray, new_frame: np.ndarray) -> dict[str, float]:
    if old_frame.shape != new_frame.shape or old_frame.ndim != 2:
        raise ValueError("aligned frames must be same-shape grayscale images")
    midpoint = old_frame.shape[1] // 2
    if midpoint <= 0 or old_frame.shape[1] % 2:
        raise ValueError("camera row must contain two equally sized views")
    metrics = {}
    for name, selection in {
        "front": np.s_[:, :midpoint],
        "wrist": np.s_[:, midpoint:],
    }.items():
        metrics[name] = float(
            np.mean(np.abs(old_frame[selection].astype(np.float32) - new_frame[selection].astype(np.float32)))
        )
    return metrics


def visual_gate(
    old_video: Path,
    new_video: Path,
    old_steps: list[int],
    new_steps: list[int],
    takeover_step: int,
    *,
    maximum_mean_mae: float,
    maximum_camera_mae: float,
) -> dict[str, object]:
    old_frames = read_video(old_video)
    new_frames = read_video(new_video)
    if len(old_frames) != len(old_steps) or len(new_frames) != len(new_steps):
        return {
            "passed": False,
            "reason": "frame_steps_length_mismatch",
            "old_decoded_frames": len(old_frames),
            "old_frame_steps": len(old_steps),
            "new_decoded_frames": len(new_frames),
            "new_frame_steps": len(new_steps),
        }
    try:
        pairs = takeover_frame_pairs(old_steps, new_steps, takeover_step)
    except ValueError as error:
        return {"passed": False, "reason": str(error)}
    camera_metrics = {
        name: camera_mae(old_frames[old_index], new_frames[new_index]) for name, (old_index, new_index) in pairs.items()
    }
    transition_mae = [
        camera_metrics[phase][camera] for phase in ("policy_takeover", "first_expert") for camera in ("front", "wrist")
    ]
    transition_mean_mae = float(np.mean(transition_mae))
    transition_max_camera_mae = float(np.max(transition_mae))
    start_max_camera_mae = float(max(camera_metrics["execution_start"].values()))
    passed = (
        start_max_camera_mae <= maximum_camera_mae
        and transition_mean_mae <= maximum_mean_mae
        and transition_max_camera_mae <= maximum_camera_mae
    )
    return {
        "passed": passed,
        "reason": "matched" if passed else "visual_difference",
        "alignment": "logical_takeover_transition",
        "frame_indices": {name: {"old": old_index, "new": new_index} for name, (old_index, new_index) in pairs.items()},
        "camera_mae": camera_metrics,
        "transition_mean_mae": transition_mean_mae,
        "transition_max_camera_mae": transition_max_camera_mae,
        "start_max_camera_mae": start_max_camera_mae,
        "maximum_transition_mean_mae": maximum_mean_mae,
        "maximum_camera_mae": maximum_camera_mae,
        "old_decoded_frames": len(old_frames),
        "new_decoded_frames": len(new_frames),
    }


def aligned_review_pairs(
    old_steps: list[int],
    new_steps: list[int],
    takeover_step: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for step in range(1, takeover_step + 1):
        old_indices = step_indices(old_steps, step)
        new_indices = step_indices(new_steps, step)
        if not old_indices or not new_indices:
            raise ValueError(f"missing_aligned_policy_step_{step}")
        pairs.append((old_indices[-1], new_indices[-1]))
    old_expert = step_indices(old_steps, takeover_step + 1)
    new_expert = step_indices(new_steps, takeover_step + 1)
    if not old_expert or not new_expert:
        raise ValueError("missing_expert_suffix")
    pairs.extend(even_pairs(old_expert, new_expert, min(len(old_expert), len(new_expert))))
    return pairs


def _selected_color_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    selected = set(indices)
    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while selected:
            ok, frame = capture.read()
            if not ok:
                break
            if index in selected:
                frames[index] = frame
                selected.remove(index)
            index += 1
    finally:
        capture.release()
    if selected:
        raise RuntimeError(f"video is missing selected frames {sorted(selected)}: {path}")
    return frames


def comparison_video(
    old_video: Path,
    new_video: Path,
    old_steps: list[int],
    new_steps: list[int],
    takeover_step: int,
    destination: Path,
) -> None:
    pairs = aligned_review_pairs(old_steps, new_steps, takeover_step)
    old_frames = _selected_color_frames(old_video, [old for old, _ in pairs])
    new_frames = _selected_color_frames(new_video, [new for _, new in pairs])
    first_old = old_frames[pairs[0][0]]
    first_new = new_frames[pairs[0][1]]
    if first_old.shape != first_new.shape:
        raise ValueError(f"comparison frame shapes differ: {first_old.shape} != {first_new.shape}")
    height, width, channels = first_old.shape
    if channels != 3:
        raise ValueError(f"comparison frames must have 3 channels, got {channels}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height * 2}",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for old_index, new_index in pairs:
            stacked = np.concatenate(
                [old_frames[old_index], new_frames[new_index]],
                axis=0,
            )
            process.stdin.write(stacked.tobytes(order="C"))
    finally:
        process.stdin.close()
    return_code = process.wait(timeout=180)
    if return_code != 0:
        raise RuntimeError(f"comparison ffmpeg exited with status {return_code}")


class ReconstructionExcludedError(RuntimeError):
    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__(str(details["reason"]))
        self.details = details




def _kept_enabled(source_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for path in source_root.glob("*/*_enabled.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("role") == "enabled"
            and payload.get("kept")
            and not payload.get("training_h5")
        ):
            items.append((path, payload))
    return sorted(
        items,
        key=lambda item: (
            str(item[1]["task"]),
            int(item[1]["episode"]),
            int(item[1].get("take", 0)),
        ),
    )

def _training_manifest(entries: list[dict[str, Any]], path: Path) -> None:
    episodes: list[dict[str, Any]] = []
    cumulative = 0
    maximum_blocks = 0
    for entry in entries:
        training = entry["training_h5"]
        execution_count = int(training["num_steps"]) - int(training["exec_start_idx"])
        cumulative += execution_count
        maximum_blocks = max(
            maximum_blocks,
            math.ceil(int(training["num_steps"]) / 16),
        )
        episodes.append(
            {
                "h5_path": entry["new_h5"],
                "episode_group": training["episode_group"],
                "task_name": entry["task"],
                "task_goal": entry["prompt"],
                "num_steps": training["num_steps"],
                "exec_start_idx": training["exec_start_idx"],
                "cumulative_execution_count": cumulative,
            }
        )
    write_json(
        path,
        {
            "version": 1,
            "execution_stride": 16,
            "total_execution_count": cumulative,
            "max_phase_aligned_blocks": maximum_blocks,
            "episodes": episodes,
        },
    )


def _metadata(old: dict[str, Any], current_subtask: str, eval_success: bool) -> dict[str, Any]:
    return {
        "role": "enabled" if eval_success else "reconstruction",
        "task": old["task"],
        "episode": old["episode"],
        "take": old.get("take", 0),
        "seed": old["seed"],
        "difficulty": old.get("difficulty", ""),
        "prompt": old["prompt"],
        "current_subtask": current_subtask,
        "outcome": "success" if eval_success else "fail",
        "eval_success": eval_success,
        "intervened": True,
        "scheduled_intervention_step": old.get("scheduled_intervention_step"),
        "scheduled_intervention_fired": old.get("scheduled_intervention_fired", False),
        "intervention_server_receive_ns": None,
        "first_expert_action_ns": None,
        "policy_cutoff_latency_ms": None,
        "first_expert_action_latency_ms": None,
        "fps": old.get("fps", 30),
        "steps": int(old["steps"]),
        "reconstructed_from": old["artifact_stem"],
    }


def _advance_policy(runtime: Any, takeover_step: int, abort_event: threading.Event) -> Any | None:
    empty_event = threading.Event()
    for expected_step in range(1, takeover_step + 1):
        result = runtime.policy_step(empty_event, abort_event, empty_event)
        if result.status == "terminal" and result.step == expected_step:
            return result
        if result.status != "running" or result.step != expected_step:
            raise RuntimeError(
                f"policy replay diverged at step {expected_step}: "
                f"got {result.status}/{result.step}/{result.outcome}"
            )
    return None


def rebuild() -> None:
    from manual_dagger.backend import RoboTTTBackendFactory

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8011)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--maximum-mean-mae", type=float, default=8.0)
    parser.add_argument("--maximum-p95-frame-mae", type=float, default=12.0)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    items = _kept_enabled(arguments.source_root)
    if arguments.limit > 0:
        items = items[: arguments.limit]
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    arguments.comparison_root.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    if arguments.status_path.is_file():
        previous = json.loads(arguments.status_path.read_text(encoding="utf-8"))
    item_keys = {str(source_json) for source_json, _ in items}
    records = {
        key: value
        for key, value in previous.get("records", {}).items()
        if key in item_keys
    }
    passed = [item for item in records.values() if item.get("gate_passed")]
    completed = [item for item in records.values() if item.get("completed")]
    done_path = arguments.status_path.with_suffix(".done")
    done_path.unlink(missing_ok=True)
    write_json(
        arguments.status_path,
        {
            "state": "RUNNING",
            "total": len(items),
            "completed": len(completed),
            "gate_passed": len(passed),
            "mismatched": len(completed) - len(passed),
            "errors": sum("error" in item for item in records.values()),
            "records": records,
            "updated_unix": time.time(),
        },
    )
    _training_manifest(passed, arguments.output_root / "training_manifest.json")
    storage = StorageProcess(arguments.output_root, None, video_fps=30)
    factory = RoboTTTBackendFactory(
        storage,
        arguments.staging_root,
        policy_host=arguments.policy_host,
        policy_port=arguments.policy_port,
        max_steps=arguments.max_steps,
    )
    abort_event = threading.Event()
    try:
        for index, (source_json, old) in enumerate(items, start=1):
            key = str(source_json)
            if records.get(key, {}).get("completed"):
                continue
            started = time.time()
            record: dict[str, Any] = {
                "completed": False,
                "task": old["task"],
                "episode": int(old["episode"]),
                "seed": int(old["seed"]),
                "take": int(old.get("take", 0)),
                "prompt": old["prompt"],
                "takeover_step": int(old["steps"]),
                "source_json": key,
                "old_video": old["artifacts"]["video"],
                "started_unix": started,
            }
            runtime = None
            try:
                runtime = factory.prepare(
                    EpisodeSpec(old["task"], int(old["episode"]), int(old.get("take", 0))),
                    lambda _frame, _source: None,
                    abort_event,
                    lambda: 30,
                    "enabled",
                )
                if runtime.seed != int(old["seed"]) or runtime.prompt != old["prompt"]:
                    raise RuntimeError("replay episode identity does not match retained metadata")
                runtime.prime_policy()
                retained_takeover_step = int(old["steps"])
                effective_takeover_step = retained_takeover_step
                early_terminal = _advance_policy(
                    runtime,
                    retained_takeover_step,
                    abort_event,
                )
                if early_terminal is not None:
                    effective_takeover_step = max(0, int(early_terminal.step) - 1)
                    raise ReconstructionExcludedError(
                        {
                            "reason": "policy_terminal_before_retained_takeover",
                            "terminal_step": int(early_terminal.step),
                            "terminal_outcome": early_terminal.outcome,
                            "retained_takeover_step": retained_takeover_step,
                            "effective_takeover_step": effective_takeover_step,
                        }
                    )
                record["effective_takeover_step"] = effective_takeover_step
                result = runtime.intervene(abort_event)
                eval_success = result.status == "terminal" and result.outcome == "success"
                publication = runtime.close(
                    eval_success,
                    _metadata(old, runtime.current_subtask, eval_success),
                )
                runtime = None
                record["official_eval_success"] = eval_success
                if eval_success:
                    new_video = Path(publication["artifacts"]["video"])
                    new_metadata_path = Path(publication["artifacts"]["metadata"])
                    new_metadata = json.loads(
                        new_metadata_path.read_text(encoding="utf-8")
                    )
                    record["new_video"] = str(new_video)
                    record["new_metadata"] = str(new_metadata_path)
                    record["new_h5"] = publication["artifacts"]["h5"][0]
                    record["training_h5"] = new_metadata["training_h5"]
                    record["visual_gate"] = visual_gate(
                        Path(record["old_video"]),
                        new_video,
                        list(old.get("frame_steps", [])),
                        list(new_metadata.get("frame_steps", [])),
                        retained_takeover_step,
                        maximum_mean_mae=arguments.maximum_mean_mae,
                        maximum_camera_mae=arguments.maximum_p95_frame_mae,
                    )
                else:
                    record["excluded"] = {
                        "reason": "official_eval_failure_after_intervention",
                        "terminal_status": result.status,
                        "terminal_outcome": result.outcome,
                    }
                    record["visual_gate"] = {
                        "passed": False,
                        "reason": "official_eval_failure_after_intervention",
                    }
                record["gate_passed"] = bool(
                    eval_success and record["visual_gate"]["passed"]
                )
                if not record["gate_passed"] and eval_success:
                    comparison = arguments.comparison_root / (
                        f"{old['task']}_ep{old['episode']}_seed{old['seed']}_"
                        f"take{int(old.get('take', 0)):04d}_old-top_new-bottom.mp4"
                    )
                    try:
                        comparison_video(
                            Path(record["old_video"]),
                            new_video,
                            list(old.get("frame_steps", [])),
                            list(new_metadata.get("frame_steps", [])),
                            retained_takeover_step,
                            comparison,
                        )
                        record["comparison_video"] = str(comparison)
                    except (RuntimeError, ValueError) as error:
                        record["comparison_video_error"] = str(error)
                record["completed"] = True
            except ReconstructionExcludedError as exclusion:
                record["excluded"] = exclusion.details
                record["effective_takeover_step"] = exclusion.details[
                    "effective_takeover_step"
                ]
                record["visual_gate"] = {
                    "passed": False,
                    **exclusion.details,
                }
                record["gate_passed"] = False
                record["completed"] = True
                if runtime is not None:
                    runtime.close(
                        False,
                        _metadata(old, runtime.current_subtask, False),
                    )
                    runtime = None
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                record["traceback"] = traceback.format_exc()
                if runtime is not None:
                    try:
                        runtime.close(False, _metadata(old, runtime.current_subtask, False))
                    except Exception:
                        pass
            record["finished_unix"] = time.time()
            records[key] = record
            passed = [item for item in records.values() if item.get("gate_passed")]
            completed = [item for item in records.values() if item.get("completed")]
            status = {
                "state": "RUNNING",
                "total": len(items),
                "completed": len(completed),
                "gate_passed": len(passed),
                "mismatched": len(completed) - len(passed),
                "errors": sum("error" in item for item in records.values()),
                "current_index": index,
                "records": records,
                "updated_unix": time.time(),
            }
            write_json(arguments.status_path, status)
            _training_manifest(passed, arguments.output_root / "training_manifest.json")

        passed = [item for item in records.values() if item.get("gate_passed")]
        completed = [item for item in records.values() if item.get("completed")]
        final_state = (
            "REBUILD_COMPLETE"
            if len(completed) == len(items) and len(passed) == len(items)
            else "REBUILD_PARTIAL"
        )
        status = {
            "state": final_state,
            "total": len(items),
            "completed": len(completed),
            "gate_passed": len(passed),
            "mismatched": len(completed) - len(passed),
            "errors": sum("error" in item for item in records.values()),
            "records": records,
            "updated_unix": time.time(),
        }
        write_json(arguments.status_path, status)
        done_path.write_text(final_state + "\n", encoding="utf-8")
    finally:
        factory.close()


def _published_path(
    path: Path,
    local_root: Path,
    published_root: Path,
) -> str:
    return str(published_root / path.resolve().relative_to(local_root.resolve()))


def _review_key(record: dict[str, Any]) -> str:
    return ":".join(
        str(record[field]) for field in ("task", "episode", "seed", "take")
    )


def _load_review_decisions(
    path: Path | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions: dict[str, dict[str, Any]] = {}
    for decision in payload.get("decisions", []):
        key = _review_key(decision)
        if key in decisions:
            raise ValueError(f"duplicate review decision: {key}")
        if decision.get("decision") not in {"accept", "reject"}:
            raise ValueError(f"invalid review decision: {key}")
        decisions[key] = decision
    rules = list(payload.get("rules", []))
    for rule in rules:
        if rule.get("decision") not in {"accept", "reject"}:
            raise ValueError("invalid review rule decision")
        if not rule.get("task") or not rule.get("gate_reason"):
            raise ValueError("review rules require task and gate_reason")
    return decisions, rules


def _resolve_review_decision(
    record: dict[str, Any],
    gate: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    rules: list[dict[str, Any]],
) -> dict[str, Any] | None:
    exact = decisions.get(_review_key(record))
    if exact is not None:
        return exact
    for rule in rules:
        if (
            rule["task"] == record["task"]
            and rule["gate_reason"] == gate.get("reason")
        ):
            return {**rule, "resolved_from_rule": True}
    return None


def _apply_review_decision(
    gate: dict[str, Any],
    decision: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    if decision is None:
        return bool(gate["passed"]), None
    if decision["decision"] == "reject":
        return False, None
    if (
        gate.get("reason") == "visual_difference"
        and gate.get("alignment") == "logical_takeover_transition"
    ):
        return True, None
    return False, "manual_accept_requires_aligned_visual_difference"


def _review_entry(
    record: dict[str, Any],
    old: dict[str, Any],
    new: dict[str, Any],
    gate: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    old_video = Path(record["old_video"])
    new_video = Path(record["new_video"])
    return {
        "id": f"{new_video.stem}:takeover-reviewed",
        "source": "takeover-reviewed-reconstruction",
        "task": record["task"],
        "episode": int(record["episode"]),
        "seed": int(record["seed"]),
        "take": int(record.get("take", 0)),
        "difficulty": str(old.get("difficulty", "")),
        "prompt": str(record["prompt"]),
        "current_subtask": str(new.get("current_subtask", "")),
        "outcome": str(new.get("outcome", "")),
        "eval_success": bool(record.get("official_eval_success")),
        "intervened": True,
        "scheduled_intervention_step": int(record["takeover_step"]),
        "control_video": _published_path(
            old_video,
            arguments.source_root,
            arguments.published_source_root,
        ),
        "enabled_video": _published_path(
            new_video,
            arguments.output_root,
            arguments.published_output_root,
        ),
        "h5_path": _published_path(
            Path(record["new_h5"]),
            arguments.output_root,
            arguments.published_output_root,
        ),
        "training_h5": new["training_h5"],
        "control_video_frames": int(old.get("video_frames", len(old.get("frame_steps", [])))),
        "control_video_fps": int(old.get("video_fps", old.get("fps", 30))),
        "enabled_video_frames": int(new.get("video_frames", len(new.get("frame_steps", [])))),
        "enabled_video_fps": int(new.get("video_fps", new.get("fps", 30))),
        "takeover_gate": gate,
        "review_basis": (
            "human_takeover_frame_review"
            if (record.get("review_decision") or {}).get("decision") == "accept"
            else "automated_takeover_gate"
        ),
        "review_decision": record.get("review_decision"),
    }


def _review_training_manifest(entries: list[dict[str, Any]], path: Path) -> None:
    episodes: list[dict[str, Any]] = []
    cumulative = 0
    maximum_blocks = 0
    for entry in entries:
        training = entry["training_h5"]
        execution_count = int(training["num_steps"]) - int(training["exec_start_idx"])
        cumulative += execution_count
        maximum_blocks = max(
            maximum_blocks,
            math.ceil(int(training["num_steps"]) / 16),
        )
        episodes.append(
            {
                "h5_path": entry["h5_path"],
                "episode_group": training["episode_group"],
                "task_name": entry["task"],
                "task_goal": entry["prompt"],
                "num_steps": training["num_steps"],
                "exec_start_idx": training["exec_start_idx"],
                "cumulative_execution_count": cumulative,
            }
        )
    write_json(
        path,
        {
            "version": 1,
            "gate": "logical_takeover_transition",
            "execution_stride": 16,
            "total_execution_count": cumulative,
            "max_phase_aligned_blocks": maximum_blocks,
            "episodes": episodes,
        },
    )


def audit() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--published-source-root", type=Path, required=True)
    parser.add_argument("--published-output-root", type=Path, required=True)
    parser.add_argument("--published-comparison-root", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--maximum-mean-mae", type=float, default=8.0)
    parser.add_argument("--maximum-camera-mae", type=float, default=12.0)
    parser.add_argument("--skip-comparisons", action="store_true")
    arguments = parser.parse_args()

    status = json.loads(arguments.status_path.read_text(encoding="utf-8"))
    review_decisions, review_rules = _load_review_decisions(
        arguments.review_decisions
    )
    records: list[dict[str, Any]] = []
    review_entries: list[dict[str, Any]] = []
    for source_json, rebuild in sorted(status["records"].items()):
        old = json.loads(Path(source_json).read_text(encoding="utf-8"))
        new_metadata_path = Path(rebuild["new_metadata"])
        new = json.loads(new_metadata_path.read_text(encoding="utf-8"))
        retained_takeover = int(rebuild["takeover_step"])
        effective_takeover = int(rebuild.get("effective_takeover_step", retained_takeover))
        if not rebuild.get("official_eval_success"):
            gate = {"passed": False, "reason": "official_eval_failure"}
        elif effective_takeover != retained_takeover:
            gate = {
                "passed": False,
                "reason": "policy_terminal_before_retained_takeover",
                "retained_takeover_step": retained_takeover,
                "effective_takeover_step": effective_takeover,
            }
        else:
            gate = visual_gate(
                Path(rebuild["old_video"]),
                Path(rebuild["new_video"]),
                list(old.get("frame_steps", [])),
                list(new.get("frame_steps", [])),
                retained_takeover,
                maximum_mean_mae=arguments.maximum_mean_mae,
                maximum_camera_mae=arguments.maximum_camera_mae,
            )
        record = {
            "task": rebuild["task"],
            "episode": int(rebuild["episode"]),
            "seed": int(rebuild["seed"]),
            "take": int(rebuild.get("take", 0)),
            "takeover_step": retained_takeover,
            "prompt": str(rebuild["prompt"]),
            "source_json": source_json,
            "new_metadata": str(new_metadata_path),
            "old_video": rebuild["old_video"],
            "new_video": rebuild["new_video"],
            "new_h5": rebuild.get("new_h5"),
            "official_eval_success": bool(rebuild.get("official_eval_success")),
            "previous_visual_gate": rebuild.get("visual_gate"),
            "takeover_gate": gate,
            "automated_gate_passed": bool(gate["passed"]),
        }
        decision = _resolve_review_decision(
            record,
            gate,
            review_decisions,
            review_rules,
        )
        record["review_decision"] = decision
        record["gate_passed"], decision_error = _apply_review_decision(
            gate,
            decision,
        )
        if decision_error is not None:
            record["review_decision_error"] = decision_error
        if record["gate_passed"]:
            review_entries.append(_review_entry(record, old, new, gate, arguments))
        elif not arguments.skip_comparisons:
            comparison = arguments.comparison_root / (
                f"{record['task']}_ep{record['episode']}_seed{record['seed']}_"
                f"take{record['take']:04d}_old-top_new-bottom.mp4"
            )
            try:
                comparison_video(
                    Path(record["old_video"]),
                    Path(record["new_video"]),
                    list(old.get("frame_steps", [])),
                    list(new.get("frame_steps", [])),
                    retained_takeover,
                    comparison,
                )
                record["comparison_video"] = _published_path(
                    comparison,
                    arguments.comparison_root,
                    arguments.published_comparison_root,
                )
            except (RuntimeError, ValueError) as error:
                record["comparison_video_error"] = str(error)
        records.append(record)
        write_json(
            arguments.audit_path,
            {
                "version": 2,
                "state": "RUNNING",
                "total": len(status["records"]),
                "completed": len(records),
                "gate_passed": sum(item["gate_passed"] for item in records),
                "automated_gate_passed": sum(
                    item["automated_gate_passed"] for item in records
                ),
                "records": records,
                "updated_unix": time.time(),
            },
        )

    by_task = Counter(item["task"] for item in records if item["gate_passed"])
    automated_by_task = Counter(
        item["task"] for item in records if item["automated_gate_passed"]
    )
    audit = {
        "version": 3,
        "state": "TAKEOVER_AUDIT_COMPLETE",
        "standard": {
            "alignment": "logical execution start, final policy frame, first expert frame",
            "cameras": ["front", "wrist"],
            "maximum_transition_mean_mae": arguments.maximum_mean_mae,
            "maximum_camera_mae": arguments.maximum_camera_mae,
            "requires_retained_takeover_step": True,
            "requires_official_eval_success": True,
        },
        "total": len(records),
        "completed": len(records),
        "gate_passed": len(review_entries),
        "automated_gate_passed": sum(
            item["automated_gate_passed"] for item in records
        ),
        "manual_accepts": sum(
            (item.get("review_decision") or {}).get("decision") == "accept"
            for item in records
        ),
        "manual_rejects": sum(
            (item.get("review_decision") or {}).get("decision") == "reject"
            for item in records
        ),
        "mismatched": len(records) - len(review_entries),
        "gate_passed_by_task": dict(sorted(by_task.items())),
        "automated_gate_passed_by_task": dict(sorted(automated_by_task.items())),
        "records": records,
        "updated_unix": time.time(),
    }
    write_json(arguments.audit_path, audit)
    write_json(
        arguments.review_manifest,
        {
            "version": 1,
            "source_audit": str(arguments.audit_path),
            "entries": review_entries,
            "updated_unix": time.time(),
        },
    )
    _review_training_manifest(review_entries, arguments.training_manifest)

if __name__ == "__main__":
    import sys
    action = sys.argv.pop(1)
    if action == "audit":
        audit()
    elif action == "rebuild":
        rebuild()
    else:
        raise SystemExit("Choose audit or rebuild")

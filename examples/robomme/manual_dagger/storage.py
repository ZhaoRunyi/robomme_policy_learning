"""manual_dagger / storage: shared behavior owned by this layer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
import multiprocessing as mp
import queue
import subprocess
import threading
import time
import h5py
import imageio_ffmpeg
import numpy as np
import argparse


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_h5(source: Path, root: Path, task: str) -> Path:
    destination = root / task / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return destination
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination


def manifest_entry(
    destination: Path,
    task: str,
    prompt: str,
    training_h5: dict[str, Any],
    **provenance: Any,
) -> dict[str, Any]:
    return {
        "h5_path": str(destination),
        "episode_group": training_h5["episode_group"],
        "task_name": task,
        "task_goal": prompt,
        "num_steps": int(training_h5["num_steps"]),
        "exec_start_idx": int(training_h5["exec_start_idx"]),
        **provenance,
    }


def write_training_manifest(path: Path, entries: list[dict[str, Any]]) -> None:
    cumulative = 0
    maximum_blocks = 0
    episodes = []
    for raw_entry in sorted(entries, key=lambda item: (item["task_name"], item["h5_path"])):
        entry = dict(raw_entry)
        cumulative += int(entry["num_steps"]) - int(entry["exec_start_idx"])
        maximum_blocks = max(
            maximum_blocks,
            math.ceil(int(entry["num_steps"]) / 16),
        )
        entry["cumulative_execution_count"] = cumulative
        episodes.append(entry)
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


def append_training_episode(path: Path, entry: dict[str, Any]) -> None:
    entries: list[dict[str, Any]] = []
    if path.is_file():
        entries = json.loads(path.read_text(encoding="utf-8")).get("episodes", [])
    by_path = {item["h5_path"]: item for item in entries}
    by_path[entry["h5_path"]] = entry
    write_training_manifest(path, list(by_path.values()))


_TRAINING_STEP_FIELDS = (
    "obs/front_rgb",
    "obs/wrist_rgb",
    "obs/joint_state",
    "obs/gripper_state",
    "action/joint_action",
    "info/is_video_demo",
    "info/action_source",
)


def validate_enabled_training_h5(path: Path) -> dict[str, Any]:
    """Validate the raw fields and masks consumed by RoboTTTSequenceDataset."""
    with h5py.File(path, "r") as source:
        episode_names = sorted(name for name in source if name.startswith("episode_"))
        if len(episode_names) != 1:
            raise ValueError(f"expected exactly one episode group, got {episode_names}")
        episode = source[episode_names[0]]
        if "setup/task_goal" not in episode:
            raise ValueError("missing setup/task_goal")
        task_goal = episode["setup/task_goal"][()]
        if np.asarray(task_goal).size == 0:
            raise ValueError("setup/task_goal is empty")
        for field in ("official_eval_success", "human_intervened"):
            if field not in episode["setup"] or not bool(episode[f"setup/{field}"][()]):
                raise ValueError(f"setup/{field} must be true")

        timestep_names = sorted(
            (name for name in episode if name.startswith("timestep_")),
            key=lambda name: int(name.removeprefix("timestep_")),
        )
        expected_names = [f"timestep_{index}" for index in range(len(timestep_names))]
        if timestep_names != expected_names:
            raise ValueError("timestep groups must be contiguous from timestep_0")
        if not timestep_names:
            raise ValueError("HDF5 has no timestep records")

        video_flags: list[bool] = []
        action_sources: list[bool] = []
        for name in timestep_names:
            timestep = episode[name]
            missing = [field for field in _TRAINING_STEP_FIELDS if field not in timestep]
            if missing:
                raise ValueError(f"{name} is missing training fields: {missing}")
            front = timestep["obs/front_rgb"]
            wrist = timestep["obs/wrist_rgb"]
            if front.shape != (256, 256, 3) or front.dtype != np.dtype(np.uint8):
                raise ValueError(f"{name}/obs/front_rgb has {front.shape} {front.dtype}")
            if wrist.shape != (256, 256, 3) or wrist.dtype != np.dtype(np.uint8):
                raise ValueError(f"{name}/obs/wrist_rgb has {wrist.shape} {wrist.dtype}")
            joint_state = np.asarray(timestep["obs/joint_state"][()])
            gripper_state = np.asarray(timestep["obs/gripper_state"][()])
            joint_action = np.asarray(timestep["action/joint_action"][()])
            if joint_state.shape != (7,) or not np.isfinite(joint_state).all():
                raise ValueError(f"{name}/obs/joint_state must be finite shape (7,)")
            if gripper_state.ndim != 1 or gripper_state.size < 1 or not np.isfinite(gripper_state).all():
                raise ValueError(f"{name}/obs/gripper_state must be a finite vector")
            if joint_action.shape != (8,) or not np.isfinite(joint_action).all():
                raise ValueError(f"{name}/action/joint_action must be finite shape (8,)")
            video_flags.append(bool(timestep["info/is_video_demo"][()]))
            action_sources.append(bool(timestep["info/action_source"][()]))

        execution_start = next(
            (index for index, is_video in enumerate(video_flags) if not is_video),
            None,
        )
        if execution_start is None:
            raise ValueError("HDF5 has no execution timestep")
        if any(video_flags[execution_start:]):
            raise ValueError("info/is_video_demo must be one true prefix followed by execution")
        if any(action_sources[:execution_start]):
            raise ValueError("conditioning-video timesteps cannot be expert targets")
        expert_steps = sum(action_sources[execution_start:])
        if expert_steps == 0:
            raise ValueError("enabled DAgger HDF5 has no expert action_source timestep")
        return {
            "episode_group": episode_names[0],
            "num_steps": len(timestep_names),
            "exec_start_idx": execution_start,
            "expert_steps": expert_steps,
        }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


class _EpisodeWriter:
    def __init__(self, staging_root: Path, video_fps: float, video_path: Path | None = None) -> None:
        self.staging_root = staging_root
        self.video_fps = video_fps
        self.video_path = video_path or staging_root / "camera.mp4"
        self.process: subprocess.Popen[bytes] | None = None
        self.frame_shape: tuple[int, int, int] | None = None
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"video frame must be HxWx3, got {array.shape}")
        if self.process is None:
            self._open(array.shape)
        if array.shape != self.frame_shape:
            raise ValueError(f"video frame shape changed: {array.shape} != {self.frame_shape}")
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(array.tobytes(order="C"))
        self.frames += 1

    def _open(self, shape: tuple[int, int, int]) -> None:
        self.staging_root.mkdir(parents=True, exist_ok=True)
        height, width, _ = shape
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(self.video_fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.video_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.frame_shape = shape

    def close(self) -> None:
        if self.process is None:
            return
        assert self.process.stdin is not None
        self.process.stdin.close()
        return_code = self.process.wait(timeout=180)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def encode_video(frames, path: Path, fps: float) -> dict[str, Any]:
    """Reuse the manual recorder's encoder for benchmark-specific camera exports."""
    import cv2

    writer = _EpisodeWriter(path.parent, fps, path)
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.close()
    if writer.frame_shape is None:
        raise ValueError("Cannot encode an empty video")
    capture = cv2.VideoCapture(str(path))
    decoded_frames = 0
    while capture.read()[0]:
        decoded_frames += 1
    capture.release()
    if decoded_frames != writer.frames:
        raise RuntimeError(f"Video frame count mismatch: {decoded_frames} != {writer.frames}")
    height, width, _ = writer.frame_shape
    return {"height": height, "width": width, "decoded_frames": decoded_frames}


def _publish(
    writer: _EpisodeWriter,
    keep: bool,
    h5_paths: list[str],
    metadata: dict[str, Any],
    primary_root: Path,
    mirror_root: Path | None,
    training_root: Path | None,
) -> dict[str, Any]:
    writer.close()
    if not keep:
        shutil.rmtree(writer.staging_root, ignore_errors=True)
        return {"status": "discarded", "episode_id": metadata["episode_id"]}
    if writer.frames <= 0 or not writer.video_path.is_file():
        raise RuntimeError("kept episode has no encoded video")
    role = metadata.get("role")
    if role not in {"enabled", "control", "reconstruction"}:
        raise RuntimeError(f"unsupported storage role {role!r}")
    if not h5_paths and role == "enabled":
        raise RuntimeError("kept episode has no DAgger HDF5 artifact")
    if role == "enabled":
        if metadata.get("eval_success") is not True:
            raise RuntimeError("kept enabled episode is not an official eval success")
        if len(h5_paths) != 1:
            raise RuntimeError("kept enabled episode must contain exactly one HDF5 artifact")
        for raw_path in h5_paths:
            metadata["training_h5"] = validate_enabled_training_h5(Path(raw_path))
    task = str(metadata["task"])
    stem = str(metadata["artifact_stem"])
    destinations: dict[str, str] = {}
    task_root = primary_root / task
    video_destination = task_root / f"{stem}.mp4"
    _atomic_copy(writer.video_path, video_destination)
    destinations["video"] = str(video_destination)
    copied_h5 = []
    for index, raw_path in enumerate(h5_paths):
        source = Path(raw_path)
        suffix = "" if len(h5_paths) == 1 else f"_{index:02d}"
        destination = task_root / f"{stem}{suffix}.h5"
        _atomic_copy(source, destination)
        copied_h5.append(str(destination))
    destinations["h5"] = copied_h5
    payload = {
        **metadata,
        "video_frames": writer.frames,
        "video_fps": writer.video_fps,
        "artifacts": destinations,
        "published_unix": time.time(),
    }
    if role == "enabled" and training_root is not None:
        training_destination = materialize_h5(
            Path(copied_h5[0]),
            training_root,
            task,
        )
        append_training_episode(
            training_root / "training_manifest.json",
            manifest_entry(
                training_destination,
                task,
                str(metadata["prompt"]),
                metadata["training_h5"],
                source_kind="live_collection",
                source_h5_path=copied_h5[0],
                episode=int(metadata["episode"]),
                seed=int(metadata["seed"]),
                take=int(metadata.get("take", 0)),
            ),
        )
        payload["training_dataset_h5"] = str(training_destination)
    metadata_destination = task_root / f"{stem}.json"
    _atomic_json(metadata_destination, payload)
    destinations["metadata"] = str(metadata_destination)
    primary_root.mkdir(parents=True, exist_ok=True)
    with (primary_root / "manifest.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    if mirror_root is not None:
        mirror_task_root = mirror_root / task
        for artifact in [video_destination, metadata_destination, *map(Path, copied_h5)]:
            _atomic_copy(artifact, mirror_task_root / artifact.name)
    shutil.rmtree(writer.staging_root, ignore_errors=True)
    return {
        "status": "published",
        "episode_id": metadata["episode_id"],
        "artifacts": destinations,
    }


def _storage_main(
    messages: mp.Queue[dict[str, Any]],
    results: mp.Queue[dict[str, Any]],
) -> None:
    writers: dict[str, _EpisodeWriter] = {}
    while True:
        message = messages.get()
        kind = message["kind"]
        if kind == "stop":
            for writer in writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            return
        episode_id = str(message["episode_id"])
        writer: _EpisodeWriter | None = None
        try:
            if kind == "begin":
                writers[episode_id] = _EpisodeWriter(
                    Path(message["staging_root"]),
                    int(message["video_fps"]),
                )
            elif kind == "frame":
                writers[episode_id].write(message["frame"])
            elif kind == "finish":
                writer = writers.pop(episode_id)
                result = _publish(
                    writer,
                    bool(message["keep"]),
                    list(message["h5_paths"]),
                    dict(message["metadata"]),
                    Path(message["primary_root"]),
                    (None if not message.get("mirror_root") else Path(message["mirror_root"])),
                    (None if not message.get("training_root") else Path(message["training_root"])),
                )
                results.put(result)
        except Exception as error:
            if kind == "finish" and writer is not None:
                shutil.rmtree(writer.staging_root, ignore_errors=True)
            results.put(
                {
                    "status": "error",
                    "episode_id": episode_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )


class StorageProcess:
    """Own exactly one process for video encoding and persistent copies."""

    def __init__(
        self,
        primary_root: Path,
        mirror_root: Path | None,
        *,
        training_root: Path | None = None,
        video_fps: int = 30,
        queue_size: int = 64,
    ) -> None:
        context = mp.get_context("spawn")
        self.primary_root = primary_root
        self.mirror_root = mirror_root
        self.training_root = training_root
        self.video_fps = video_fps
        self.messages: mp.Queue[dict[str, Any]] = context.Queue(queue_size)
        self.results: mp.Queue[dict[str, Any]] = context.Queue()
        self.process = context.Process(
            target=_storage_main,
            args=(self.messages, self.results),
            name="manual-dagger-storage",
        )
        self.process.start()
        self._finish_lock = threading.Lock()

    def begin(self, episode_id: str, staging_root: Path) -> None:
        self.check_health()
        self.messages.put(
            {
                "kind": "begin",
                "episode_id": episode_id,
                "staging_root": str(staging_root),
                "video_fps": self.video_fps,
            },
            timeout=10,
        )

    def frame(self, episode_id: str, frame: np.ndarray) -> None:
        try:
            self.messages.put(
                {"kind": "frame", "episode_id": episode_id, "frame": frame},
                timeout=2,
            )
        except queue.Full as error:
            raise RuntimeError("storage process did not drain video frames") from error

    def finish(
        self,
        episode_id: str,
        keep: bool,
        h5_paths: list[Path],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        with self._finish_lock:
            self.messages.put(
                {
                    "kind": "finish",
                    "episode_id": episode_id,
                    "keep": keep,
                    "h5_paths": [str(path) for path in h5_paths],
                    "metadata": metadata,
                    "primary_root": str(self.primary_root),
                    "mirror_root": ("" if self.mirror_root is None else str(self.mirror_root)),
                    "training_root": (
                        "" if self.training_root is None else str(self.training_root)
                    ),
                },
                timeout=10,
            )
            result = self.results.get(timeout=300)
            if result.get("episode_id") != episode_id:
                raise RuntimeError(
                    f"storage result identity mismatch: {result.get('episode_id')} != {episode_id}"
                )
            if result.get("status") == "error":
                raise RuntimeError(result["error"])
            return result

    def get_result(self, timeout: float | None = None) -> dict[str, Any]:
        return self.results.get(timeout=timeout)

    def check_health(self) -> None:
        if not self.process.is_alive():
            raise RuntimeError(f"storage process exited unexpectedly with {self.process.exitcode}")
        self._raise_reported_errors()

    def _raise_reported_errors(self) -> None:
        errors = []
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            if result.get("status") == "error":
                errors.append(f"{result['episode_id']}: {result['error']}")
        if errors:
            raise RuntimeError("storage publication failed: " + "; ".join(errors))

    def close(self) -> None:
        if not self.process.is_alive():
            self._raise_reported_errors()
            return
        self.messages.put({"kind": "stop"}, timeout=10)
        self.process.join(timeout=180)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=30)
            raise RuntimeError("storage process did not shut down cleanly")
        self._raise_reported_errors()


def _reviewed_entries(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    payload = json.loads(arguments.reviewed_manifest.read_text(encoding="utf-8"))
    entries = []
    for reviewed in payload["entries"]:
        relative = Path(reviewed["h5_path"]).relative_to(arguments.published_reconstructed_root)
        source = arguments.reconstructed_root / relative
        training_h5 = validate_enabled_training_h5(source)
        destination = materialize_h5(source, arguments.output_root, reviewed["task"])
        entries.append(
            manifest_entry(
                destination,
                reviewed["task"],
                reviewed["prompt"],
                training_h5,
                source_kind="reviewed_reconstruction",
                source_h5_path=str(arguments.published_reconstructed_root / relative),
                episode=int(reviewed["episode"]),
                seed=int(reviewed["seed"]),
                take=int(reviewed["take"]),
            )
        )
    return entries


def _live_entries(
    arguments: argparse.Namespace,
    existing_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    existing_sources = existing_sources or set()
    entries = []
    for metadata_path in sorted(arguments.live_root.glob("*/*_enabled.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        h5_paths = metadata.get("artifacts", {}).get("h5", [])
        if not (
            metadata.get("role") == "enabled"
            and metadata.get("kept")
            and metadata.get("eval_success") is True
            and isinstance(metadata.get("training_h5"), dict)
            and len(h5_paths) == 1
        ):
            continue
        source = Path(h5_paths[0])
        published_source = arguments.published_live_root / metadata["task"] / source.name
        if str(published_source) in existing_sources:
            continue
        training_h5 = validate_enabled_training_h5(source)
        destination = materialize_h5(source, arguments.output_root, metadata["task"])
        entries.append(
            manifest_entry(
                destination,
                metadata["task"],
                metadata["prompt"],
                training_h5,
                source_kind="live_collection",
                source_h5_path=str(published_source),
                episode=int(metadata["episode"]),
                seed=int(metadata["seed"]),
                take=int(metadata.get("take", 0)),
            )
        )
    return entries


def materialize() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--reconstructed-root", type=Path, required=True)
    parser.add_argument("--published-reconstructed-root", type=Path, required=True)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--published-live-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--incremental-live-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.published_live_root is None:
        arguments.published_live_root = arguments.live_root

    if arguments.incremental_live_only:
        manifest_path = arguments.output_root / "training_manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))["episodes"]
        existing_sources = {entry["source_h5_path"] for entry in existing}
        new_entries = _live_entries(arguments, existing_sources)
        entries = existing + new_entries
    else:
        new_entries = []
        entries = _reviewed_entries(arguments) + _live_entries(arguments)
    write_training_manifest(arguments.output_root / "training_manifest.json", entries)
    source_entries = [
        {
            key: entry[key]
            for key in (
                "h5_path",
                "source_kind",
                "source_h5_path",
                "task_name",
                "episode",
                "seed",
                "take",
            )
        }
        for entry in entries
    ]
    write_json(
        arguments.output_root / "source_manifest.json",
        {"version": 1, "entries": source_entries},
    )
    print(
        json.dumps(
            {
                "output_root": str(arguments.output_root),
                "episodes": len(entries),
                "added": len(new_entries),
                "reviewed": sum(entry["source_kind"] == "reviewed_reconstruction" for entry in entries),
                "live": sum(entry["source_kind"] == "live_collection" for entry in entries),
            },
            sort_keys=True,
        )
    )

if __name__ == "__main__":
    materialize()

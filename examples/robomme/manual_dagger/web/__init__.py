"""manual_dagger / web: shared behavior owned by this layer."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
from pathlib import Path
import time
from typing import Any
from aiohttp import WSMsgType
from aiohttp import web
import cv2
import numpy as np
from manual_dagger.state import ControlCache
from manual_dagger.controller import ManualDaggerController
from manual_dagger.state import Command
from manual_dagger.state import CommandRequest
from manual_dagger.state import command_from_key


class BrowserHub:
    def __init__(self, token: str) -> None:
        self.token = token
        self._clients: set[web.WebSocketResponse] = set()
        self._frame_lock = asyncio.Lock()
        self._latest_frames: dict[str, bytes] = {}
        self._control_frame_history: list[bytes] = []
        self._frame_sequences = {"control": 0, "enabled": 0}
        self._episode_generation: int | None = None

    def add(self, websocket: web.WebSocketResponse) -> None:
        self._clients.add(websocket)

    def remove(self, websocket: web.WebSocketResponse) -> None:
        self._clients.discard(websocket)

    async def publish_state(self, state: dict[str, Any]) -> None:
        generation = int(state.get("episode_generation", 0))
        async with self._frame_lock:
            if generation != self._episode_generation:
                self._episode_generation = generation
                self._latest_frames.clear()
                self._control_frame_history.clear()
        await self._broadcast_text({"type": "state", "state": state})

    async def publish_frame(self, stream: str, frame: np.ndarray, step: int) -> None:
        marker = {"control": b"C", "enabled": b"E"}.get(stream)
        if marker is None:
            raise ValueError(f"unknown video stream {stream!r}")
        async with self._frame_lock:
            success, encoded = await asyncio.to_thread(
                cv2.imencode,
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )
            if not success:
                raise RuntimeError("JPEG encoding failed")
            self._frame_sequences[stream] += 1
            sequence = self._frame_sequences[stream]
            payload = (
                marker
                + sequence.to_bytes(4, "big")
                + step.to_bytes(4, "big")
                + encoded.tobytes()
            )
            self._latest_frames[stream] = payload
            if stream == "control":
                self._control_frame_history.append(payload)
            stale = []
            for websocket in tuple(self._clients):
                try:
                    await asyncio.wait_for(websocket.send_bytes(payload), timeout=0.25)
                except (ConnectionError, RuntimeError, TimeoutError):
                    stale.append(websocket)
            for websocket in stale:
                self.remove(websocket)

    async def send_latest_frame(self, websocket: web.WebSocketResponse) -> None:
        async with self._frame_lock:
            control_payloads = list(self._control_frame_history)
            if not control_payloads and "control" in self._latest_frames:
                control_payloads.append(self._latest_frames["control"])
            payloads = control_payloads
            if "enabled" in self._latest_frames:
                payloads.append(self._latest_frames["enabled"])
        for payload in payloads:
            await websocket.send_bytes(payload)

    async def _broadcast_text(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        stale = []
        for websocket in tuple(self._clients):
            try:
                await asyncio.wait_for(websocket.send_str(text), timeout=0.25)
            except (ConnectionError, RuntimeError, TimeoutError):
                stale.append(websocket)
        for websocket in stale:
            self.remove(websocket)


def create_app(
    controller: ManualDaggerController,
    hub: BrowserHub,
    saved_root: Path | None = None,
    control_cache: ControlCache | None = None,
    review_manifest: Path | None = None,
    review_media_roots: tuple[Path, ...] = (),
) -> web.Application:
    static_root = Path(__file__).resolve().parent / "static"
    application = web.Application()
    command_results: OrderedDict[str, dict[str, Any]] = OrderedDict()

    async def startup(app: web.Application) -> None:
        del app
        await controller.start()

    async def shutdown(app: web.Application) -> None:
        del app
        await controller.stop()

    async def index(request: web.Request) -> web.StreamResponse:
        if request.query.get("token", "") != hub.token:
            return web.json_response({"error": "invalid access token"}, status=403)
        response = web.FileResponse(static_root / "index.html")
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    async def healthz(request: web.Request) -> web.StreamResponse:
        del request
        return web.json_response({"ok": True, "state": await controller.snapshot()})

    def authenticated(request: web.Request) -> bool:
        return request.query.get("token", "") == hub.token

    def saved_task_root(task: str) -> Path | None:
        if saved_root is None or not task or Path(task).name != task:
            return None
        task_root = (saved_root / task).resolve()
        if task_root.parent != saved_root.resolve():
            return None
        return task_root

    def reviewed_entries(task: str) -> list[dict[str, Any]]:
        if review_manifest is None or not review_manifest.is_file():
            return []
        try:
            payload = json.loads(review_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [
            entry
            for entry in payload.get("entries", [])
            if entry.get("task") == task and entry.get("source") == "takeover-reviewed-reconstruction"
        ]

    def reviewed_media(task: str, entry_id: str, kind: str) -> Path | None:
        if kind not in {"control", "enabled"}:
            return None
        entry = next(
            (item for item in reviewed_entries(task) if item.get("id") == entry_id),
            None,
        )
        if entry is None:
            return None
        path = Path(str(entry.get(f"{kind}_video", ""))).resolve()
        for root in review_media_roots:
            resolved_root = root.resolve()
            try:
                path.relative_to(resolved_root)
            except ValueError:
                continue
            if path.is_file():
                return path
        return None

    async def saved_episodes(request: web.Request) -> web.StreamResponse:
        if not authenticated(request):
            return web.json_response({"error": "invalid access token"}, status=403)
        task = request.query.get("task", "")
        task_root = saved_task_root(task)
        if task_root is None:
            return web.json_response({"error": "invalid task"}, status=400)
        entries = []
        for metadata_path in sorted(task_root.glob("*_enabled.json"), reverse=True):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("role") != "enabled" or not metadata.get("kept"):
                continue
            enabled_video = metadata_path.with_suffix(".mp4")
            control_stem = metadata_path.stem.removesuffix("_enabled") + "_control"
            control_video = task_root / f"{control_stem}.mp4"
            control_metadata: dict[str, Any] = {}
            control_metadata_path = control_video.with_suffix(".json")
            if control_metadata_path.is_file():
                try:
                    control_metadata = json.loads(
                        control_metadata_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    control_metadata = {}
            if not control_video.is_file() and control_cache is not None:
                artifact = control_cache.lookup(
                    task,
                    int(metadata.get("episode", 0)),
                    int(metadata.get("seed", -1)),
                )
                if artifact is not None:
                    control_video = artifact.video_path
                    control_metadata = artifact.metadata
            if not enabled_video.is_file() or not control_video.is_file():
                continue
            entries.append(
                {
                    "id": metadata_path.stem,
                    "archive_source": "saved",
                    "task": task,
                    "episode": int(metadata.get("episode", 0)),
                    "seed": metadata.get("seed"),
                    "take": int(metadata.get("take", 0)),
                    "difficulty": str(metadata.get("difficulty", "")),
                    "prompt": str(metadata.get("prompt", "")),
                    "current_subtask": str(metadata.get("current_subtask", "")),
                    "outcome": str(metadata.get("outcome", "")),
                    "eval_success": metadata.get(
                        "eval_success",
                        metadata.get("outcome") == "success",
                    ),
                    "intervened": bool(metadata.get("intervened", False)),
                    "scheduled_intervention_step": metadata.get(
                        "scheduled_intervention_step"
                    ),
                    "enabled_video": enabled_video.name,
                    "control_video": control_video.name,
                    "enabled_video_frames": int(metadata.get("video_frames", 0)),
                    "enabled_video_fps": int(metadata.get("video_fps", 30)),
                    "control_video_frames": int(
                        control_metadata.get("video_frames", 0)
                    ),
                    "control_video_fps": int(control_metadata.get("video_fps", 30)),
                }
            )
        for reviewed in reviewed_entries(task):
            entries.append(
                {
                    **reviewed,
                    "archive_source": "takeover-reviewed",
                    "control_video": Path(reviewed["control_video"]).name,
                    "enabled_video": Path(reviewed["enabled_video"]).name,
                }
            )
        response = web.json_response({"task": task, "entries": entries})
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    async def cached_control(request: web.Request) -> web.StreamResponse:
        if not authenticated(request):
            return web.json_response({"error": "invalid access token"}, status=403)
        if control_cache is None:
            return web.json_response({"ready": False}, status=404)
        task = request.query.get("task", "")
        try:
            episode = int(request.query["episode"])
            seed = int(request.query["seed"])
        except (KeyError, TypeError, ValueError):
            return web.json_response({"error": "invalid control identity"}, status=400)
        artifact = control_cache.lookup_failure(task, episode, seed)
        if artifact is None:
            response = web.json_response({"ready": False}, status=202)
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        metadata = artifact.metadata
        frames = int(metadata.get("video_frames", 0))
        raw_steps = metadata.get("frame_steps", [])
        frame_steps = (
            [int(step) for step in raw_steps]
            if isinstance(raw_steps, list) and len(raw_steps) == frames
            else list(range(1, frames + 1))
        )
        response = web.json_response(
            {
                "ready": True,
                "task": task,
                "episode": episode,
                "seed": seed,
                "outcome": str(metadata.get("outcome", "")),
                "video_frames": frames,
                "video_fps": int(metadata.get("video_fps", 30)),
                "frame_steps": frame_steps,
            }
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    async def cached_control_media(request: web.Request) -> web.StreamResponse:
        if not authenticated(request):
            raise web.HTTPForbidden(text="invalid access token")
        if control_cache is None:
            raise web.HTTPNotFound(text="control cache is unavailable")
        task = request.query.get("task", "")
        try:
            episode = int(request.query["episode"])
            seed = int(request.query["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text="invalid control identity") from error
        artifact = control_cache.lookup_failure(task, episode, seed)
        if artifact is None:
            raise web.HTTPNotFound(text="cached control not found")
        response = web.FileResponse(artifact.video_path)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    async def saved_media(request: web.Request) -> web.StreamResponse:
        if not authenticated(request):
            raise web.HTTPForbidden(text="invalid access token")
        if request.query.get("source") == "takeover-reviewed":
            media_path = reviewed_media(
                request.query.get("task", ""),
                request.query.get("id", ""),
                request.query.get("kind", ""),
            )
            if media_path is None:
                raise web.HTTPNotFound(text="reviewed media not found")
            response = web.FileResponse(media_path)
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        task_root = saved_task_root(request.query.get("task", ""))
        filename = request.query.get("file", "")
        if (
            task_root is None
            or not filename
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".mp4"
        ):
            raise web.HTTPBadRequest(text="invalid saved media path")
        media_path = (task_root / filename).resolve()
        if media_path.parent != task_root or not media_path.is_file():
            raise web.HTTPNotFound(text="saved media not found")
        response = web.FileResponse(media_path)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    async def websocket_endpoint(request: web.Request) -> web.StreamResponse:
        if request.query.get("token", "") != hub.token:
            raise web.HTTPForbidden(text="invalid access token")
        websocket = web.WebSocketResponse(heartbeat=20.0)
        await websocket.prepare(request)
        hub.add(websocket)
        await websocket.send_json({"type": "state", "state": await controller.snapshot()})
        history_task = asyncio.create_task(
            hub.send_latest_frame(websocket),
            name="manual-dagger-frame-history",
        )
        try:
            async for message in websocket:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError:
                    payload = {}
                request_id = str(payload.get("request_id", ""))
                cached_result = command_results.get(request_id)
                if request_id and cached_result is not None:
                    command_results.move_to_end(request_id)
                    await websocket.send_json({**cached_result, "replayed": True})
                    continue
                key = str(payload.get("key", ""))
                action = str(payload.get("action", ""))
                action_commands = {
                    Command.SELECT_NEXT.value: Command.SELECT_NEXT,
                    Command.SCHEDULE_INTERVENTION.value: Command.SCHEDULE_INTERVENTION,
                }
                command = action_commands.get(action) or command_from_key(key)
                receive_ns = time.monotonic_ns()
                if command is None:
                    accepted, detail = False, f"unsupported key {key!r}"
                else:
                    accepted, detail = await controller.submit(
                        CommandRequest(
                            command,
                            request_id,
                            receive_ns,
                            payload.get("value"),
                        )
                    )
                result = {
                    "type": "command_ack",
                    "request_id": request_id,
                    "accepted": accepted,
                    "detail": detail,
                    "server_receive_ns": receive_ns,
                    "server_ack_ns": time.monotonic_ns(),
                }
                if request_id:
                    command_results[request_id] = result
                    command_results.move_to_end(request_id)
                    while len(command_results) > 2048:
                        command_results.popitem(last=False)
                await websocket.send_json(result)
        finally:
            hub.remove(websocket)
            history_task.cancel()
            await asyncio.gather(history_task, return_exceptions=True)
        return websocket

    application.on_startup.append(startup)
    application.on_shutdown.append(shutdown)
    application.router.add_get("/", index)
    application.router.add_get("/healthz", healthz)
    application.router.add_get("/saved", saved_episodes)
    application.router.add_get("/saved/media", saved_media)
    application.router.add_get("/control-cache", cached_control)
    application.router.add_get("/control-cache/media", cached_control_media)
    application.router.add_get("/ws", websocket_endpoint)
    return application


TARGET_MARKER_RGB = (139, 92, 246)


def overlay_world_target_marker(
    frame: np.ndarray,
    camera: dict[str, Any],
    world_position: Any,
) -> np.ndarray:
    """Project one world-space target into an RGB frame and mark it in purple."""
    output = np.asarray(frame, dtype=np.uint8).copy()
    position = np.asarray(world_position, dtype=np.float64).reshape(-1)
    intrinsic = np.asarray(camera.get("intrinsic_cv"), dtype=np.float64)
    extrinsic = np.asarray(camera.get("extrinsic_cv"), dtype=np.float64)
    if (
        position.size < 3
        or intrinsic.shape != (3, 3)
        or extrinsic.shape not in {(3, 4), (4, 4)}
    ):
        return output
    camera_position = extrinsic @ np.r_[position[:3], 1.0]
    if not np.isfinite(camera_position).all() or camera_position[2] <= 1e-6:
        return output
    pixel = intrinsic @ camera_position[:3]
    center_x, center_y = np.rint(pixel[:2] / pixel[2]).astype(int)
    height, width = output.shape[:2]
    if not 0 <= center_x < width or not 0 <= center_y < height:
        return output
    radius = max(2, int(round(min(height, width) * 0.012)))
    cv2.circle(
        output,
        (int(center_x), int(center_y)),
        radius,
        TARGET_MARKER_RGB,
        -1,
        lineType=cv2.LINE_AA,
    )
    return output


def _flatten_segments(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_segments(item))
        return flattened
    return [] if value is None else [value]


def _current_task_entry(environment: Any) -> dict[str, Any]:
    task_list = getattr(environment, "task_list", ())
    index = int(getattr(environment, "current_task_index", -1))
    if index < 0:
        index = int(getattr(environment, "timestep", 0))
    if 0 <= index < len(task_list):
        return task_list[index]
    return {}


def _binfill_pickup_segments(environment: Any) -> list[Any] | None:
    task_entry = _current_task_entry(environment)
    task_name = str(task_entry.get("name", "")).lower()
    if "pick up" not in task_name:
        return None

    segments: list[Any] = []
    for color in ("red", "blue", "green"):
        target = int(getattr(environment, f"{color}_cubes_target_number", 0))
        completed = int(getattr(environment, f"{color}_cubes_in_bin", 0))
        if completed < target:
            segments.extend(getattr(environment, f"{color}_cubes", ()))
    return segments


def current_target_segments(recorder: Any, task: str) -> list[Any]:
    """Return official current-subtask targets, with BinFill's valid set expanded."""
    environment = recorder.unwrapped
    if task == "BinFill":
        binfill_segments = _binfill_pickup_segments(environment)
        if binfill_segments is not None:
            return binfill_segments
    current_segment = getattr(environment, "current_segment", None)
    if current_segment is None:
        current_segment = getattr(recorder, "current_segment", None)
    if current_segment is None:
        current_segment = _current_task_entry(environment).get("segment")
    return _flatten_segments(current_segment)


def overlay_target_markers(
    base_frame: np.ndarray,
    segmentation: np.ndarray | None,
    recorder: Any,
    task: str,
) -> np.ndarray:
    """Draw purple dots on third-person RGB only; simulator state stays untouched."""
    output = np.asarray(base_frame, dtype=np.uint8).copy()
    if segmentation is None:
        return output
    segmentation_2d = np.asarray(segmentation).squeeze()
    if segmentation_2d.ndim != 2:
        return output

    targets = current_target_segments(recorder, task)
    segmentation_id_map = getattr(recorder, "segmentation_id_map", None)
    if not targets or not isinstance(segmentation_id_map, dict):
        return output

    radius = max(2, int(round(min(output.shape[:2]) * 0.012)))
    for target in targets:
        target_ids = [
            object_id
            for object_id, actor in segmentation_id_map.items()
            if actor is target
        ]
        if not target_ids:
            continue
        coordinates = np.argwhere(np.isin(segmentation_2d, target_ids))
        if not coordinates.size:
            continue
        center_y, center_x = np.rint(coordinates.mean(axis=0)).astype(int)
        cv2.circle(
            output,
            (int(center_x), int(center_y)),
            radius,
            TARGET_MARKER_RGB,
            -1,
            lineType=cv2.LINE_AA,
        )
    return output



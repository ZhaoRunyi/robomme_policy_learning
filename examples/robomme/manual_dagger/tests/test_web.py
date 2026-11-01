"""tests / test_web: shared behavior owned by this layer."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import patch
from aiohttp.test_utils import TestClient
from aiohttp.test_utils import TestServer
from manual_dagger.state import ControlCache
from manual_dagger.state import EpisodeSpec
from manual_dagger.web import BrowserHub
from manual_dagger.web import create_app
import numpy as np


def test_new_browser_receives_cached_latest_frame() -> None:
    async def scenario() -> None:
        hub = BrowserHub("token")
        hub._latest_frames = {
            "control": b"C\x00\x00\x00\x07\x00\x00\x00\x03\xff\xd8control-jpeg",
            "enabled": b"E\x00\x00\x00\x08\x00\x00\x00\x04\xff\xd8enabled-jpeg",
        }
        hub._control_frame_history = [
            b"C\x00\x00\x00\x06\x00\x00\x00\x02\xff\xd8older-control-jpeg",
            b"C\x00\x00\x00\x07\x00\x00\x00\x03\xff\xd8control-jpeg",
        ]

        class WebSocket:
            payloads: list[bytes] = []

            async def send_bytes(self, payload: bytes) -> None:
                self.payloads.append(payload)

        websocket = WebSocket()
        await hub.send_latest_frame(websocket)  # type: ignore[arg-type]
        assert websocket.payloads == [
            b"C\x00\x00\x00\x06\x00\x00\x00\x02\xff\xd8older-control-jpeg",
            b"C\x00\x00\x00\x07\x00\x00\x00\x03\xff\xd8control-jpeg",
            b"E\x00\x00\x00\x08\x00\x00\x00\x04\xff\xd8enabled-jpeg",
        ]

    asyncio.run(scenario())


def test_new_episode_clears_cached_control_history() -> None:
    async def scenario() -> None:
        hub = BrowserHub("token")
        first = {"episode_generation": 1}
        second = {"episode_generation": 2}
        await hub.publish_state(first)
        hub._latest_frames = {"control": b"old-control", "enabled": b"old-enabled"}
        hub._control_frame_history = [b"old-control"]

        await hub.publish_state(first)
        assert hub._control_frame_history == [b"old-control"]
        await hub.publish_state(second)
        assert hub._control_frame_history == []
        assert hub._latest_frames == {}

    asyncio.run(scenario())


def test_published_frames_include_stream_and_monotonic_sequence() -> None:
    async def scenario() -> None:
        hub = BrowserHub("token")

        class WebSocket:
            payloads: list[bytes] = []

            async def send_bytes(self, payload: bytes) -> None:
                self.payloads.append(payload)

        websocket = WebSocket()
        hub.add(websocket)  # type: ignore[arg-type]
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        encoded = np.asarray([0xFF, 0xD8, 0xFF, 0xD9], dtype=np.uint8)
        with patch(
            "manual_dagger.web.asyncio.to_thread",
            new=AsyncMock(return_value=(True, encoded)),
        ):
            await hub.publish_frame("control", frame, 7)
            await hub.publish_frame("control", frame, 8)
        assert hub._control_frame_history == websocket.payloads
        assert [payload[0:1] for payload in websocket.payloads] == [b"C", b"C"]
        assert [int.from_bytes(payload[1:5], "big") for payload in websocket.payloads] == [1, 2]
        assert [int.from_bytes(payload[5:9], "big") for payload in websocket.payloads] == [7, 8]
        assert all(payload[9:11] == b"\xff\xd8" for payload in websocket.payloads)

    asyncio.run(scenario())


def test_index_is_not_cached_and_retried_command_is_idempotent() -> None:
    async def scenario() -> None:
        controller = AsyncMock()
        controller.snapshot.return_value = {"episode_generation": 1}
        controller.submit.return_value = (True, "scheduled")
        client = TestClient(TestServer(create_app(controller, BrowserHub("token"))))
        await client.start_server()
        try:
            response = await client.get("/?token=token")
            assert response.headers["Cache-Control"] == "no-store, max-age=0"

            websocket = await client.ws_connect("/ws?token=token")
            await websocket.receive_json()
            command = {
                "action": "schedule_intervention",
                "request_id": "same-request",
                "value": {"step": 12, "frame_sequence": 34},
            }
            await websocket.send_json(command)
            first_ack = await websocket.receive_json()
            await websocket.send_json(command)
            replayed_ack = await websocket.receive_json()

            assert first_ack["accepted"] is True
            assert replayed_ack["accepted"] is True
            assert replayed_ack["replayed"] is True
            assert controller.submit.await_count == 1
            await websocket.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_saved_buffer_lists_complete_pairs_and_serves_only_task_mp4(tmp_path) -> None:
    async def scenario() -> None:
        task_root = tmp_path / "VideoPlaceOrder"
        task_root.mkdir()
        enabled_stem = "VideoPlaceOrder_ep10_seed12002_take0011_enabled"
        control_stem = "VideoPlaceOrder_ep10_seed12002_take0011_control"
        (task_root / f"{enabled_stem}.json").write_text(
            json.dumps(
                {
                    "role": "enabled",
                    "kept": True,
                    "task": "VideoPlaceOrder",
                    "episode": 10,
                    "seed": 12002,
                    "take": 11,
                    "prompt": "place the blue cube",
                    "outcome": "fail",
                    "intervened": True,
                    "video_frames": 77,
                    "video_fps": 30,
                }
            ),
            encoding="utf-8",
        )
        (task_root / f"{enabled_stem}.mp4").write_bytes(b"enabled-video")
        (task_root / f"{control_stem}.mp4").write_bytes(b"control-video")
        (task_root / f"{control_stem}.json").write_text(
            json.dumps({"video_frames": 91, "video_fps": 24}),
            encoding="utf-8",
        )
        controller = AsyncMock()
        controller.snapshot.return_value = {"episode_generation": 1}
        client = TestClient(
            TestServer(create_app(controller, BrowserHub("token"), tmp_path))
        )
        await client.start_server()
        try:
            response = await client.get(
                "/saved?token=token&task=VideoPlaceOrder"
            )
            assert response.status == 200
            entries = (await response.json())["entries"]
            assert len(entries) == 1
            assert entries[0]["episode"] == 10
            assert entries[0]["enabled_video"] == f"{enabled_stem}.mp4"
            assert entries[0]["enabled_video_frames"] == 77
            assert entries[0]["enabled_video_fps"] == 30
            assert entries[0]["control_video_frames"] == 91
            assert entries[0]["control_video_fps"] == 24
            media = await client.get(
                "/saved/media",
                params={
                    "token": "token",
                    "task": "VideoPlaceOrder",
                    "file": f"{control_stem}.mp4",
                },
            )
            assert media.status == 200
            assert await media.read() == b"control-video"
            traversal = await client.get(
                "/saved/media",
                params={
                    "token": "token",
                    "task": "VideoPlaceOrder",
                    "file": "../secret.mp4",
                },
            )
            assert traversal.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())


def test_saved_buffer_includes_takeover_reviewed_reconstruction(tmp_path) -> None:
    async def scenario() -> None:
        saved_root = tmp_path / "saved"
        saved_root.mkdir()
        media_root = tmp_path / "media"
        media_root.mkdir()
        old_video = media_root / "old.mp4"
        new_video = media_root / "new.mp4"
        old_video.write_bytes(b"old-video")
        new_video.write_bytes(b"new-video")
        manifest = tmp_path / "reviewed.json"
        manifest.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "id": "reviewed-one",
                            "source": "takeover-reviewed-reconstruction",
                            "task": "PatternLock",
                            "episode": 7,
                            "seed": 15700,
                            "take": 5,
                            "prompt": "follow the pattern",
                            "eval_success": True,
                            "intervened": True,
                            "control_video": str(old_video),
                            "enabled_video": str(new_video),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        controller = AsyncMock()
        controller.snapshot.return_value = {"episode_generation": 1}
        client = TestClient(
            TestServer(
                create_app(
                    controller,
                    BrowserHub("token"),
                    saved_root,
                    review_manifest=manifest,
                    review_media_roots=(media_root,),
                )
            )
        )
        await client.start_server()
        try:
            response = await client.get("/saved?token=token&task=PatternLock")
            entry = (await response.json())["entries"][0]
            assert entry["archive_source"] == "takeover-reviewed"
            assert entry["control_video"] == "old.mp4"
            media = await client.get(
                "/saved/media",
                params={
                    "token": "token",
                    "task": "PatternLock",
                    "source": "takeover-reviewed",
                    "id": "reviewed-one",
                    "kind": "enabled",
                },
            )
            assert media.status == 200
            assert await media.read() == b"new-video"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_control_cache_metadata_and_media_are_served_by_episode_identity(tmp_path) -> None:
    async def scenario() -> None:
        task = "BinFill"
        task_root = tmp_path / task
        task_root.mkdir()
        stem = "BinFill_ep3_seed4301_take0000_control"
        (task_root / f"{stem}.json").write_text(
            json.dumps(
                {
                    "role": "control",
                    "kept": True,
                    "control_cache": True,
                    "task": task,
                    "episode": 3,
                    "seed": 4301,
                    "steps": 3,
                    "outcome": "fail",
                    "video_frames": 3,
                    "video_fps": 30,
                    "frame_steps": [1, 2, 3],
                }
            ),
            encoding="utf-8",
        )
        (task_root / f"{stem}.mp4").write_bytes(b"cached-control")
        cache = ControlCache(tmp_path, (EpisodeSpec(task, 3, 0),))
        controller = AsyncMock()
        controller.snapshot.return_value = {"episode_generation": 1}
        client = TestClient(
            TestServer(create_app(controller, BrowserHub("token"), tmp_path, cache))
        )
        await client.start_server()
        try:
            response = await client.get(
                "/control-cache",
                params={"token": "token", "task": task, "episode": 3, "seed": 4301},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["ready"] is True
            assert payload["frame_steps"] == [1, 2, 3]
            media = await client.get(
                "/control-cache/media",
                params={"token": "token", "task": task, "episode": 3, "seed": 4301},
            )
            assert await media.read() == b"cached-control"
        finally:
            await client.close()

    asyncio.run(scenario())

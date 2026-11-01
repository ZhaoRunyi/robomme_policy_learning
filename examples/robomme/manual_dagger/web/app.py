"""manual_dagger / web / app: shared behavior owned by this layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import time
from aiohttp import web
from manual_dagger.state import ControlCache
from manual_dagger.controller import ManualDaggerController
from manual_dagger.controller import FakeBackendFactory
from manual_dagger.state import CollectionProgress
from manual_dagger.state import EpisodeSpec
from manual_dagger.web import BrowserHub
from manual_dagger.web import create_app


TASKS = (
    "VideoPlaceOrder",
    "PatternLock",
    "VideoPlaceButton",
    "VideoUnmask",
    "VideoRepick",
    "StopCube",
    "BinFill",
    "ButtonUnmask",
    "PickXtimes",
    "RouteStick",
    "InsertPeg",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "MoveCube",
    "SwingXtimes",
    "PickHighlight",
)


def _episode_queue(
    tasks_value: str,
    episode_start: int,
    episodes_per_task: int,
) -> tuple[EpisodeSpec, ...]:
    tasks = _selected_tasks(tasks_value)
    if episodes_per_task < 1:
        raise ValueError("episodes-per-task must be positive")
    return tuple(
        EpisodeSpec(task, episode)
        for task in tasks
        for episode in range(episode_start, episode_start + episodes_per_task)
    )


def _selected_tasks(tasks_value: str) -> tuple[str, ...]:
    tasks = (
        TASKS
        if tasks_value == "all"
        else tuple(item.strip() for item in tasks_value.split(",") if item.strip())
    )
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8011)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--target-per-task", type=int, default=100)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--session", default="")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--mirror-root", type=Path, default=None)
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--review-manifest", type=Path, default=None)
    parser.add_argument(
        "--review-media-root",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--access-token", default="")
    parser.add_argument("--runtime-config", type=Path, default=Path("/workspace/rollout_materials/robomme_capture_config.json"))
    parser.add_argument("--simulate", action="store_true")
    arguments = parser.parse_args()

    runtime_config_path = arguments.runtime_config
    runtime_config = (
        json.loads(runtime_config_path.read_text(encoding="utf-8"))
        if runtime_config_path.is_file()
        else {}
    )
    session = str(
        runtime_config.get("session")
        or arguments.session
        or time.strftime("manual-dagger-%Y%m%d-%H%M%S")
    )
    output_root = Path(
        runtime_config.get("output_root")
        or arguments.output_root
        or Path("/workspace/ttt_ws/runs/robottt/manual_dagger") / session
    )
    staging_root = Path(
        runtime_config.get("staging_root")
        or arguments.staging_root
        or Path("/tmp") / session
    )
    mirror_root = (
        Path(runtime_config["mirror_root"])
        if runtime_config.get("mirror_root")
        else arguments.mirror_root
    )
    target_per_task = int(
        runtime_config.get("target_per_task", arguments.target_per_task)
    )
    token = arguments.access_token or secrets.token_urlsafe(24)
    episodes = _episode_queue(
        arguments.tasks,
        arguments.episode_start,
        arguments.episodes_per_task,
    )
    hub = BrowserHub(token)
    if arguments.simulate:
        factory = FakeBackendFactory()
        collection = None
        control_cache = None
    else:
        from manual_dagger.backend import RoboTTTBackendFactory
        from manual_dagger.storage import StorageProcess

        storage = StorageProcess(
            output_root,
            mirror_root,
            training_root=arguments.training_root,
            video_fps=arguments.video_fps,
        )
        factory = RoboTTTBackendFactory(
            storage,
            staging_root,
            policy_host=arguments.policy_host,
            policy_port=arguments.policy_port,
            max_steps=arguments.max_steps,
        )
        tasks = _selected_tasks(arguments.tasks)
        collection = CollectionProgress(
            output_root,
            tasks,
            {task: factory.episode_count(task) for task in tasks},
            target_per_task=target_per_task,
            episode_start=arguments.episode_start,
            training_root=arguments.training_root,
        )
        control_cache = ControlCache(
            output_root,
            collection.control_specs(),
            mirror_root,
        )
    controller = ManualDaggerController(
        factory,
        episodes,
        hub.publish_state,
        hub.publish_frame,
        default_fps=arguments.fps,
        collection=collection,
        control_cache=control_cache,
    )
    application = create_app(
        controller,
        hub,
        output_root,
        control_cache,
        arguments.review_manifest,
        tuple(arguments.review_media_root),
    )
    print(f"MANUAL_DAGGER_SESSION={session}", flush=True)
    print(f"MANUAL_DAGGER_OUTPUT={output_root}", flush=True)
    print(f"MANUAL_DAGGER_TOKEN={token}", flush=True)
    web.run_app(application, host=arguments.host, port=arguments.port)

if __name__ == "__main__":
    main()

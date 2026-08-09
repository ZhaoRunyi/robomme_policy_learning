import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def wilson(successes, count):
    probability = successes / count
    center = (probability + 1.96**2 / (2 * count)) / (1 + 1.96**2 / count)
    margin = 1.96 * np.sqrt(
        probability * (1 - probability) / count + 1.96**2 / (4 * count**2)
    ) / (1 + 1.96**2 / count)
    return [center - margin, center + margin]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shards_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    args = parser.parse_args()

    progress = {}
    details = {}
    videos = []
    progress_paths = sorted(args.shards_dir.glob("*/**/progress.json"))
    if not progress_paths:
        raise RuntimeError(f"No evaluation shards found under {args.shards_dir}")

    failure_reasons = {}
    for progress_path in progress_paths:
        shard_dir = progress_path.parent
        shard_progress = json.loads(progress_path.read_text())
        shard_details_path = shard_dir / "details.json"
        shard_details = json.loads(shard_details_path.read_text())
        failure_reasons.update(shard_progress.pop("_failure_reasons", {}))
        for task_name, task_results in shard_progress.items():
            output_results = progress.setdefault(task_name, {})
            duplicate = output_results.keys() & task_results.keys()
            if duplicate:
                raise RuntimeError(f"Duplicate episodes for {task_name}: {sorted(duplicate)}")
            output_results.update(task_results)
        duplicate = details.keys() & shard_details.keys()
        if duplicate:
            raise RuntimeError(f"Duplicate episode details: {sorted(duplicate)}")
        details.update(shard_details)
        videos.extend((shard_dir / "videos").glob("*.mp4"))

    episode_count = sum(len(task_results) for task_results in progress.values())
    if episode_count != args.expected_episodes or len(details) != episode_count:
        raise RuntimeError(
            f"Incomplete shards: episodes={episode_count}, details={len(details)}"
        )

    video_dir = args.output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        output_video = video_dir / video.name
        if output_video.exists():
            raise RuntimeError(f"Duplicate video: {video.name}")
        shutil.copy2(video, output_video)
    if len(videos) != episode_count:
        raise RuntimeError(f"Incomplete videos: videos={len(videos)}, episodes={episode_count}")

    success_rate = {
        task_name: sum(task_results.values()) / len(task_results)
        for task_name, task_results in progress.items()
    }
    all_results = [result for task_results in progress.values() for result in task_results.values()]
    final_results = {
        "success_rate": success_rate,
        "total_success_rate": sum(success_rate.values()) / len(success_rate),
        "episode_details": details,
        "wilson_95": {
            task_name: wilson(sum(task_results.values()), len(task_results))
            for task_name, task_results in progress.items()
        },
        "total_wilson_95": wilson(sum(all_results), len(all_results)),
    }
    for name, value in (
        ("progress.json", {**progress, "_failure_reasons": failure_reasons}),
        ("details.json", details),
        ("log.json", final_results),
    ):
        (args.output_dir / name).write_text(json.dumps(value, indent=2))
    shutil.rmtree(args.shards_dir)


if __name__ == "__main__":
    main()

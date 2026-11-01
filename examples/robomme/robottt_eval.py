"""RoboTTT episode metrics and parallel shard assembly on the standard evaluator."""

import json
import shutil
from pathlib import Path

from eval import EpisodeEvaluator


class RoboTTTEpisodeEvaluator(EpisodeEvaluator):
    def __init__(self, args, save_dir):
        super().__init__(args, save_dir)
        self.reset_options = {"robottt_mode": args.robottt_mode}

    def init_episode(self, env_runner, state, video_save_dir):
        result = super().init_episode(env_runner, state, video_save_dir)
        self.episode_metrics["video_blocks_seen"] = len(range(state.exec_start_idx % 16, state.exec_start_idx, 16))
        return result

    def finish_metrics(self, state):
        metrics = self.episode_metrics
        mode = self.args.robottt_mode or "normal"
        updated_video = 0 if mode == "no_video" else metrics["video_blocks_seen"]
        video_inner_tokens = 36
        decisions = len(metrics["infer_time_ms"])
        total_blocks = metrics["video_blocks_seen"] + decisions
        metrics.update(
            raw_env_steps=state.count,
            policy_decisions=decisions,
            video_blocks_updated=updated_video,
            execution_blocks=decisions,
            inner_valid_tokens=video_inner_tokens * updated_video + 36 * decisions,
            first_extrapolated_block=96 if total_blocks > 96 else None,
            extrapolated_blocks=max(total_blocks - 96, 0),
        )



    @classmethod
    def merge_shards(cls, shards_dir: Path, output_dir: Path, expected_episodes: int):
        progress = {}
        details = {}
        videos = []
        progress_paths = sorted(shards_dir.glob("*/**/progress.json"))
        if not progress_paths:
            raise RuntimeError(f"No evaluation shards found under {shards_dir}")

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
        if episode_count != expected_episodes or len(details) != episode_count:
            raise RuntimeError(
                f"Incomplete shards: episodes={episode_count}, details={len(details)}"
            )

        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        for video in videos:
            output_video = video_dir / video.name
            if output_video.exists():
                raise RuntimeError(f"Duplicate video: {video.name}")
            shutil.copy2(video, output_video)
        if len(videos) != episode_count:
            raise RuntimeError(f"Incomplete videos: videos={len(videos)}, episodes={episode_count}")

        final_results = cls.summarize(progress, details)
        for name, value in (
            ("progress.json", {**progress, "_failure_reasons": failure_reasons}),
            ("details.json", details),
            ("log.json", final_results),
        ):
            (output_dir / name).write_text(json.dumps(value, indent=2))
        shutil.rmtree(shards_dir)

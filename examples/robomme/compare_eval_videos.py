import argparse
import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg


VIDEO_RE = re.compile(r"^(?P<task>.+)_ep(?P<episode>\d+)_(?P<flag>success|fail|timeout|error|unknown)_.+\.mp4$")


def index_videos(video_dir: Path):
    videos = {}
    flags = {}
    for video in video_dir.glob("*.mp4"):
        match = VIDEO_RE.match(video.name)
        if match is None:
            continue
        key = f"{match.group('task')}/{match.group('episode')}"
        videos[key] = video
        flags[key] = match.group("flag")
    return videos, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("ttt_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=800)
    args = parser.parse_args()

    baseline_videos, baseline_flags = index_videos(args.baseline_dir / "videos")
    ttt_videos, ttt_flags = index_videos(args.ttt_dir / "videos")
    keys = sorted(set(baseline_videos) & set(ttt_videos))
    if len(keys) != args.expected_episodes:
        raise RuntimeError(f"matched videos={len(keys)}, expected={args.expected_episodes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    summary = {}
    for key in keys:
        task, episode = key.split("/")
        baseline_flag = baseline_flags[key]
        ttt_flag = ttt_flags[key]
        output = args.output_dir / f"{task}_ep{episode}_baseline-{baseline_flag}_ttt-{ttt_flag}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i", str(baseline_videos[key]),
                "-i", str(ttt_videos[key]),
                "-filter_complex", "vstack=inputs=2",
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "veryfast",
                "-an",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        summary[key] = {"baseline": baseline_flag, "ttt": ttt_flag, "video": output.name}
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Any, Tuple, TYPE_CHECKING

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy
from utils import (
    pack_buffer,
    check_args,
    TASK_NAME_LIST,
    TASK_WITH_VIDEO_DEMO,
    SUBGOAL_TYPES,
    EpisodeState,
)
from utils import RolloutRecorder
if TYPE_CHECKING:
    from env_runner import EnvRunner

# qwen3-vl environment variables
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'



@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8011

    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation"
    overwrite: bool = False

    # Policy identity and memory behavior.
    policy_name: str = "dummy_test"
    model_seed: int = 42
    model_ckpt_id: int = 80000
    use_history: bool = True
    robottt_mode: Optional[str] = None

    # Episode and task selection.
    episodes_per_task: Optional[int] = None
    episode_start: int = 0
    episode_stride: int = 1
    re_eval_tasks: str = "" # tasks split by comma
    only_tasks: str = "" # tasks split by comma
    exclude_tasks: str = "" # tasks split by comma

    # Optional DAgger recording.
    dagger_dir: Optional[str] = None
    dagger_dataset: str = "train"

    # VLM subgoal predictor
    use_oracle: bool = False
    use_qwenvl: bool = False
    use_memer: bool = False
    use_gemini: bool = False
    subgoal_type: Optional[str] = None  # [simple_subgoal, grounded_subgoal]
    gemini_model_name: str = "gemini-2.5-pro"
    qwenvl_simpleSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/simple_subgoal/checkpoint-1400"
    qwenvl_groundSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200"
    memer_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300"
    subgoal_keep_period: int = 1 # ever subgoal should be kept for this many steps
    # this can accelerate the evaluation process for symbolic memory
    # In our experiments, we just set this to 1



class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir
        self.reset_options = {}

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        subgoal_predictor: Any | None,
        video_save_dir: Path,
    ) -> str:
        self.client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(
            self.args.host, self.args.port
        )
        resp = self.client.reset(**self.reset_options)
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        self.episode_metrics = {
            "reset_time_ms": resp.get("reset_time_ms", 0),
            "prefill_time_ms": 0, "infer_time_ms": [],
        }
        epstate = EpisodeState()
        task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)
        if subgoal_predictor is not None:
            subgoal_predictor.start_episode(epstate, env_runner)

        img, wrist_img, robot_state = epstate.get_current_obs()
        prompt = task_goal
        success_flag = "unknown"
        subgoal = None
        last_subgoal = None
        while True:
            if subgoal_predictor is not None:
                subgoal_predictor.step(epstate)

            if not epstate.action_plan:
                if subgoal_predictor is None:
                    subgoal, has_api_error = None, False
                elif epstate.count % self.args.subgoal_keep_period == 0 or last_subgoal is None:
                    subgoal, has_api_error = subgoal_predictor.get_subgoal(
                        epstate.count,
                        subgoal,
                        last_subgoal,
                    )
                else:
                    subgoal = last_subgoal
                    has_api_error = False

                if has_api_error:
                    break

                action_chunk = self.get_action_chunk(
                    self.client, epstate, img, wrist_img, robot_state, prompt, subgoal,
                    exec_horizon=self.args.obs_horizon
                )

                epstate.action_plan.extend(action_chunk)
                epstate.clear_buffers()

                last_subgoal = subgoal

            action = epstate.action_plan.popleft()
            obs, stop_flag, success_flag = env_runner.step(action)
            if stop_flag and success_flag == "error":
                raise RuntimeError(f"{env_runner.info.get('exception_type')}: {env_runner.info.get('error_message')}")
            epstate.count += 1
            if epstate.count > self.args.max_steps:
                success_flag = "timeout"
                break

            img, wrist_img, robot_state = obs

            epstate.add_observation(img, wrist_img, robot_state, action)
            recorder.record(
                image=img.copy(),
                wrist_image=wrist_img.copy(),
                state=robot_state.copy(),
                action=action.copy(),
                subgoal=subgoal,
            )

            if stop_flag:
                break

        if success_flag == "unknown":
            self.finish_metrics(epstate)
            self.client.close()
            return "unknown"

        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        if subgoal_predictor is not None:
            subgoal_predictor.end_episode(epstate, success_flag)
        self.finish_metrics(epstate)
        self.client.close()
        return success_flag

    def finish_metrics(self, state):
        self.episode_metrics.update(raw_env_steps=state.count, policy_decisions=len(self.episode_metrics["infer_time_ms"]))

    def init_episode(
        self,
        env_runner: EnvRunner,
        epstate: EpisodeState,
        video_save_dir: Path,
    ) -> Tuple[str, RolloutRecorder]:
        pre_traj = env_runner.get_init_obs()
        task_goal = pre_traj["task_goal"]

        recorder = RolloutRecorder(video_save_dir, task_goal, fps=30)

        print(f"task_goal: {task_goal}")

        epstate.image_buffer.extend(pre_traj["images"])
        epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
        epstate.state_buffer.extend(pre_traj["states"])
        epstate.action_buffer.extend(pre_traj["actions"])

        for i in range(len(pre_traj["images"])):
            recorder.record(
                image=pre_traj["images"][i].copy(),
                wrist_image=pre_traj["wrist_images"][i].copy(),
                state=pre_traj["states"][i].copy(),
                is_video_demo=env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1,
                subgoal=None if self.args.subgoal_type is None else "[initializing...]",
            )

        epstate.exec_start_idx = len(epstate.image_buffer) - 1
        print(f"exec_start_idx: {epstate.exec_start_idx}")
        return task_goal, recorder

    def get_action_chunk(
        self,
        client,
        state: EpisodeState,
        img: np.ndarray,
        wrist_img: np.ndarray,
        robot_state: np.ndarray,
        prompt: str,
        subgoal: Optional[str],
        exec_horizon: int,
    ) -> list:
        if self.args.use_history:
            buffer = pack_buffer(
                state.image_buffer,
                state.state_buffer,
                state.action_buffer,
                state.exec_start_idx,
            )
            buffer.update(wrist_images=np.stack(state.wrist_image_buffer).astype(np.uint8)[:, None], prompt=prompt)
            resp = client.add_buffer(buffer)
            while not resp.get("add_buffer_finished", False):
                time.sleep(0.1)
            if not self.episode_metrics["infer_time_ms"]:
                self.episode_metrics["prefill_time_ms"] = resp.get("add_buffer_time_ms", 0)

        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": robot_state,
            "prompt": prompt,
        }

        if subgoal is not None:
            element['simple_subgoal'] = subgoal
            element['grounded_subgoal'] = subgoal

        response = client.infer(element)
        self.episode_metrics["infer_time_ms"].append(response.get("infer_time_ms", 0))
        action_chunk = response["actions"]
        return action_chunk[:exec_horizon]


    @staticmethod
    def summarize(progress, details=None):
        success_rate = {
            task: sum(results.values()) / len(results)
            for task, results in progress.items() if not task.startswith("_")
        }
        output = {
            "success_rate": success_rate,
            "total_success_rate": sum(success_rate.values()) / len(success_rate),
        }
        if details is not None:
            output["episode_details"] = details
        return output




def setup_save_directory(args: Args) -> Path:
    """Set up and validate save directories."""
    save_dir = (
        Path(args.save_dir)
        / args.policy_name
        / f"ckpt{args.model_ckpt_id}"
        / f"seed{args.model_seed}"
    )

    if args.subgoal_type in SUBGOAL_TYPES:
        if args.use_gemini:
            save_dir = save_dir / "gemini"
        elif args.use_qwenvl:
            save_dir = save_dir / "qwenvl"
        elif args.use_memer:
            save_dir = save_dir / "memer"
        else:
            save_dir = save_dir / "oracle"
    if args.robottt_mode is not None:
        save_dir = save_dir / args.robottt_mode

    if save_dir.exists():
        if args.overwrite:
            shutil.rmtree(save_dir)
            print(f"we will overwrite the evaluation at {save_dir}")
        else:
            print("we will resume the evaluation")

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def setup_log_dict(save_dir: Path, args: Args) -> dict:
    if os.path.exists(save_dir / "progress.json"):
        with open(save_dir / "progress.json", "r") as f:
            log_dict = json.load(f)

    elif os.path.exists(save_dir / "log.json"):
        with open(save_dir / "log.json", "r") as f:
            log_dict = json.load(f)
        for key in ("success_rate", "total_success_rate", "episode_details", "wilson_95", "total_wilson_95"):
            log_dict.pop(key, None)
    else:
        log_dict = {}

    log_dict.setdefault("_failure_reasons", {})
    for task_name in [key for key in log_dict if not key.startswith("_")]:
        error_list = []
        for k, v in log_dict[task_name].items():
            if v == "error" and args.robottt_mode is None:
                error_list.append(k)
        for k in error_list:
            log_dict[task_name].pop(k)

    if args.re_eval_tasks:
        for task_name in args.re_eval_tasks.split(","):
            if task_name in log_dict:
                del log_dict[task_name]
                os.system(f"rm -f {save_dir / 'videos' / f'{task_name}_ep*.mp4'}")

    with open(save_dir / "progress.json", "w") as f:
        json.dump(log_dict, f, indent=2)

    return log_dict


def evaluate(args: Args):
    """Main evaluation function."""
    from env_runner import EnvRunner

    check_args(args)
    if args.robottt_mode not in (None, "normal", "no_video", "no_carry"):
        raise ValueError(f"Unsupported RoboTTT mode: {args.robottt_mode}")
    if args.dagger_dataset not in ("train", "test"):
        raise ValueError(f"Unsupported DAgger dataset: {args.dagger_dataset}")

    save_dir = setup_save_directory(args)
    video_save_dir = save_dir / "videos"
    log_dict = setup_log_dict(save_dir, args)
    failure_reasons = log_dict.setdefault("_failure_reasons", {})
    details_path = save_dir / "details.json"
    details = json.load(open(details_path)) if details_path.exists() else {}
    if args.robottt_mode is not None and (save_dir / "log.json").exists():
        (save_dir / "log.json").unlink()

    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = TASK_NAME_LIST

    if args.exclude_tasks:
        task_names = [task_name for task_name in task_names if task_name not in args.exclude_tasks.split(",")]
        for task in args.exclude_tasks.split(","):
            log_dict[task] = {str(i): False for i in range(50)}

    subgoal_predictor = None
    if args.use_oracle or args.use_qwenvl or args.use_memer or args.use_gemini:
        from subgoal_predictor import build_subgoal_predictor
        subgoal_predictor = build_subgoal_predictor(args, save_dir)
    evaluator_class = EpisodeEvaluator
    if args.robottt_mode is not None:
        from robottt_eval import RoboTTTEpisodeEvaluator
        evaluator_class = RoboTTTEpisodeEvaluator
    evaluator = evaluator_class(args, save_dir)
    for task_name in task_names:
        if task_name not in log_dict:
            log_dict[task_name] = {}

        runner_class = EnvRunner
        runner_options = {}
        if args.dagger_dir:
            from dagger_review.env_runner import DAggerEnvRunner
            runner_class = DAggerEnvRunner
            runner_options = {"dagger_dir": args.dagger_dir, "dataset": args.dagger_dataset}
        env_runner = runner_class(
            task_name,
            video_save_dir,
            max_steps=args.max_steps,
            **runner_options,
        )
        num_episodes = min(env_runner.num_episodes, args.episodes_per_task or env_runner.num_episodes)
        success_flag = "unknown"

        for episode_id in range(args.episode_start, num_episodes, args.episode_stride):
            if str(episode_id) in log_dict.get(task_name, {}):
                print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                continue

            env_runner.make_env(episode_id)
            print(f"\n[robomme] env for task {task_name} episode {episode_id} setup finished")

            try:
                success_flag = evaluator.eval_each_episode(env_runner, subgoal_predictor, video_save_dir)
                episode_key = f"{task_name}/{episode_id}"
                if success_flag == "unknown":
                    log_dict[task_name][episode_id] = False if args.robottt_mode is not None else "error"
                    evaluator.episode_metrics["error"] = "unknown policy response"
                else:
                    log_dict[task_name][episode_id] = success_flag == "success"
                if success_flag != "success":
                    failure_reasons[episode_key] = env_runner.failure_reason(success_flag)
                if args.robottt_mode is not None:
                    details[episode_key] = evaluator.episode_metrics
            except Exception as e:
                print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                log_dict[task_name][episode_id] = False if args.robottt_mode is not None else "error"
                episode_key = f"{task_name}/{episode_id}"
                failure_reasons[episode_key] = env_runner.failure_reason("error", e)
                details[episode_key] = {"error": str(e)}
                if hasattr(evaluator, "client"):
                    evaluator.client.close()

            env_runner.close_env()
            with open(save_dir / "progress.json", "w") as f:
                json.dump(log_dict, f, indent=2)
            if args.robottt_mode is not None:
                with open(details_path, "w") as f:
                    json.dump(details, f, indent=2)

            if success_flag == "unknown" and args.robottt_mode is None:
                print("API calling error, aborting...")
                return

        del env_runner
        time.sleep(1)

    try:
        final_results = evaluator.summarize(log_dict, details if args.robottt_mode is not None else None)
        (save_dir / "log.json").write_text(json.dumps(final_results, indent=2))
    except Exception as error:
        print(f"Error saving final results: {error}")


if __name__ == "__main__":
    import sys
    import tyro

    if sys.argv[1:2] == ["merge"]:
        from robottt_eval import RoboTTTEpisodeEvaluator
        tyro.cli(RoboTTTEpisodeEvaluator.merge_shards, args=sys.argv[2:])
    else:
        tyro.cli(evaluate)

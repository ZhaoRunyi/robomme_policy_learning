import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Any, Tuple

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
from env_runner import EnvRunner
from subgoal_predictor import build_subgoal_predictor, SubgoalPredictorBase

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

    use_history: bool = True
    policy_name: str = "dummy_test"
    model_seed: int = 42
    model_ckpt_id: int = 80000
    robottt_mode: Optional[str] = None
    episodes_per_task: Optional[int] = None
    episode_start: int = 0
    episode_stride: int = 1
    dagger_dir: Optional[str] = None
    dagger_dataset: str = "train"
    dagger_stagnation_steps: int = 128

    # task control
    re_eval_tasks: str = "" # tasks split by comma
    only_tasks: str = "" # tasks split by comma
    exclude_tasks: str = "" # tasks split by comma

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

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        subgoal_predictor: SubgoalPredictorBase,
        video_save_dir: Path,
    ) -> str:
        self.client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(
            self.args.host, self.args.port
        )
        client = self.client
        resp = client.reset(self.args.robottt_mode)
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        epstate = EpisodeState()
        task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)
        video_blocks = len(range(epstate.exec_start_idx % 16, epstate.exec_start_idx, 16))
        self.episode_metrics = {
            "reset_time_ms": resp.get("reset_time_ms", 0), "video_blocks_seen": video_blocks,
            "prefill_time_ms": 0, "infer_time_ms": [],
        }
        subgoal_predictor.start_episode(epstate, env_runner)        

        img, wrist_img, robot_state = epstate.get_current_obs()
        prompt = task_goal
        success_flag = "unknown"
        subgoal = None
        last_subgoal = None
        previous_progress = env_runner.subtask_progress
        stagnant_steps = 0

        while True:
            subgoal_predictor.step(epstate)

            if not epstate.action_plan:
                if epstate.count % self.args.subgoal_keep_period == 0 or last_subgoal is None:
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
                    client, epstate, img, wrist_img, robot_state, prompt, subgoal,
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
            progress = env_runner.subtask_progress
            stagnant_steps = stagnant_steps + 1 if progress == previous_progress else 0
            previous_progress = progress
            if self.args.dagger_dir and (
                env_runner.subtask_failed
                or stagnant_steps >= self.args.dagger_stagnation_steps
            ):
                intervention_progress = env_runner.subtask_progress
                expert_success = env_runner.finish_with_expert()
                print(
                    f"DAgger intervention: progress {intervention_progress}"
                    f" -> {env_runner.subtask_progress}, success={expert_success}"
                )
                success_flag = "success" if expert_success else "fail"
                break

            if epstate.count > self.args.max_steps:
                success_flag = "timeout"
                break

            img, wrist_img, robot_state = obs

            epstate.add_observation(img, wrist_img, robot_state)
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
            client.close()
            return "unknown"

        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        subgoal_predictor.end_episode(epstate, success_flag)
        self.finish_metrics(epstate)
        client.close()
        return success_flag

    def finish_metrics(self, state):
        metrics = self.episode_metrics
        mode = self.args.robottt_mode or "normal"
        updated_video = 0 if mode == "no_video" else metrics["video_blocks_seen"]
        decisions = len(metrics["infer_time_ms"])
        total_blocks = metrics["video_blocks_seen"] + decisions
        metrics.update(
            raw_env_steps=state.count,
            policy_decisions=decisions,
            video_blocks_updated=updated_video,
            execution_blocks=decisions,
            inner_valid_tokens=16 * updated_video + 36 * decisions,
            first_extrapolated_block=96 if total_blocks > 96 else None,
            extrapolated_blocks=max(total_blocks - 96, 0),
        )


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

    for task_name in log_dict:
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
    check_args(args)
    if args.robottt_mode not in (None, "normal", "no_video", "no_carry"):
        raise ValueError(f"Unsupported RoboTTT mode: {args.robottt_mode}")
    if args.dagger_dataset not in ("train", "test"):
        raise ValueError(f"Unsupported DAgger dataset: {args.dagger_dataset}")

    save_dir = setup_save_directory(args)
    video_save_dir = save_dir / "videos"
    log_dict = setup_log_dict(save_dir, args)
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

    subgoal_predictor = build_subgoal_predictor(args, save_dir)
    evaluator = EpisodeEvaluator(args, save_dir)
    for task_name in task_names:
        if task_name not in log_dict:
            log_dict[task_name] = {}

        env_runner = EnvRunner(
            task_name,
            video_save_dir,
            max_steps=args.max_steps,
            dagger_dir=args.dagger_dir,
            dataset=args.dagger_dataset if args.dagger_dir else "test",
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
                if success_flag == "unknown":
                    log_dict[task_name][episode_id] = False if args.robottt_mode is not None else "error"
                    evaluator.episode_metrics["error"] = "unknown policy response"
                else:
                    log_dict[task_name][episode_id] = success_flag == "success"
                if args.robottt_mode is not None:
                    details[f"{task_name}/{episode_id}"] = evaluator.episode_metrics
            except Exception as e:
                print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                log_dict[task_name][episode_id] = False if args.robottt_mode is not None else "error"
                details[f"{task_name}/{episode_id}"] = {"error": str(e)}
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
        def wilson(successes, count):
            probability = successes / count
            center = (probability + 1.96**2 / (2 * count)) / (1 + 1.96**2 / count)
            margin = 1.96 * np.sqrt(probability * (1 - probability) / count + 1.96**2 / (4 * count**2)) / (1 + 1.96**2 / count)
            return [center - margin, center + margin]

        final_results = {}
        final_results["success_rate"] = {
            task_name: sum(log_dict[task_name].values()) / len(log_dict[task_name].values())
            for task_name in log_dict.keys()
        }
        final_results["total_success_rate"] = (
            sum(final_results["success_rate"].values()) / len(final_results["success_rate"].values())
        )
        if args.robottt_mode is not None:
            final_results["episode_details"] = details
            final_results["wilson_95"] = {
                task_name: wilson(sum(values.values()), len(values)) for task_name, values in log_dict.items()}
            all_results = [value for values in log_dict.values() for value in values.values()]
            final_results["total_wilson_95"] = wilson(sum(all_results), len(all_results))
        with open(save_dir / "log.json", "w") as f:
            json.dump(final_results, f, indent=2)
    except Exception as e:
        print(f"Error saving final results: {e}")


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)

"""DAgger recording and native one-subtask takeover, outside the evaluation runner."""

from pathlib import Path

import h5py
import numpy as np
from robomme.env_record_wrapper import RobommeRecordWrapper
from env_runner import EnvRunner


class DAggerRecordWrapper(RobommeRecordWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, save_video=True, **kwargs)
        self.action_source = 0

    def step(self, action):
        buffer_start = len(self.buffer)
        result = super().step(action)
        progress = max(0, int(getattr(self.unwrapped, "current_task_index", 0)))
        for record in self.buffer[buffer_start:]:
            record["info"]["action_source"] = np.uint8(self.action_source)
            record["info"]["subtask_progress"] = np.int32(progress)
            if self.action_source:
                record["info"]["is_video_demo"] = False
        return result

    def close(self):
        metadata = [record["info"] for record in self.buffer]
        successful = self.episode_success
        path = self.dataset_path
        super().close()
        if successful:
            with h5py.File(path, "a") as output:
                episode = output[f"episode_{self.episode}"]
                for index, info in enumerate(metadata):
                    group = episode[f"timestep_{index}"]["info"]
                    group.create_dataset("action_source", data=info["action_source"])
                    group.create_dataset("subtask_progress", data=info["subtask_progress"])



class DAggerEnvRunner(EnvRunner):
    record_wrapper_class = DAggerRecordWrapper

    def __init__(self, *args, dagger_dir, dataset="train", **kwargs):
        super().__init__(*args, dataset=dataset, **kwargs)
        self.dagger_dir = dagger_dir
        self.recorder = None

    def make_env(self, episode_id):
        super().make_env(episode_id)
        demonstration_wrapper = self.env.env
        seed, _ = self.env_builder.resolve_episode(episode_id)
        self.recorder = self.record_wrapper_class(
            demonstration_wrapper.env,
            dataset=Path(self.dagger_dir),
            env_id=self.env_id,
            episode=episode_id,
            seed=seed,
        )
        demonstration_wrapper.env = self.recorder
        self.env.unwrapped.use_demonstrationwrapper = True

    def finish_with_expert(self, *, max_subtasks: int | None = None) -> bool:
        demonstration_wrapper = self.env.env
        task_list = self.env.unwrapped.task_list
        self.recorder.action_source = 1
        current = int(np.asarray(self.env.unwrapped.timestep).reshape(-1)[0])
        demonstration_flags = [bool(task.get("demonstration", False)) for task in task_list]
        end = len(task_list) if max_subtasks is None else current + max_subtasks
        for index, task in enumerate(task_list):
            task["demonstration"] = current <= index < end
        try:
            demonstration_wrapper.get_demonstration_trajectory()
        finally:
            for task, flag in zip(task_list, demonstration_flags):
                task["demonstration"] = flag
        official_success = getattr(self.env.unwrapped, "successflag", False)
        if hasattr(official_success, "item"):
            official_success = official_success.item()
        success = bool(official_success)
        self.recorder.episode_success = success
        return success


import os
import bisect
import json
import logging
import h5py
import jax
import numpy as np
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import re

from openpi.training import config as _config
from openpi.training.data_loader import Dataset
from openpi import transforms as _transforms
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent
import pickle

random.seed(0)
logger = logging.getLogger(__name__)


def load_vector_file(vector_path: str, step_idx: int) -> tuple[dict, int]:
    with open(vector_path, "rb") as f:
        return np.load(f, allow_pickle=True).item(), step_idx
    
    

class SampleDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.stats = json.load(open(os.path.join(self.dataset_path, "meta", "stats.json")))
    
    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        else:
            return self.stats["total_samples"]
    
    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path,  "data", f"{idx}.pkl"), "rb") as f:
            return pickle.load(f)


class RoboMMEDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_config: _config.DataConfig,
        history_config: DictConfig | None,
        action_horizon: int,
        compute_norm_stats: bool = False,
    ):
        self.history_config = history_config
        
        
        self.action_horizon = action_horizon
        self.dataset = SampleDataset(dataset_path)
        self.feature_dir = Path(self.dataset.dataset_path) / "features"

        if self.history_config is not None:
            self.img_emb_dim = self.history_config.memory_feature.img.input_dim
            self.pos_emb_dim = self.history_config.memory_feature.pos.input_dim
            self.state_emb_dim = self.history_config.memory_feature.state.input_dim
            self.num_views = self.history_config.num_views
            self.streaming_obs_horizon = self.history_config.streaming_obs_horizon
        
            if self.history_config.representation_type == "perceptual":
                self.mem_buffer = MemoryBuffer(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    token_drop_stride=self.streaming_obs_horizon // 2,
                )
            elif self.history_config.representation_type == "recurrent":
                self.mem_buffer = MemoryBufferRecurrent(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    input_obs_horizon=self.streaming_obs_horizon,
                    max_recur_steps=self.history_config.recurrent_memory.max_recur_steps,
                    max_video_steps=self.history_config.recurrent_memory.max_pretraj_steps,
                )
            else:
                # symbolic memory does not need mem_buffer
                self.mem_buffer = None
        else:
            logger.info("=== Do not use history ===")
        
        self.compute_norm_stats = compute_norm_stats
        
        if not compute_norm_stats:
            self.state_norm_stats = data_config.norm_stats['state']
            self.use_quantiles = data_config.use_quantile_norm
        
    def _gather_history_feat(self, indices_to_load: list[int], epis_idx: int):
        history_feats = {}
        history_paths = []
        for idx in indices_to_load:
            history_feats[idx] = {}
            history_paths.append(
                os.path.join(self.feature_dir, f"episode_{epis_idx}", f"token_emb_{idx}.npy")
            )
        
        if self.history_config.representation_type == "recurrent":
            # load_vector_fn = load_vector_file_lru4096 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(48, max(4, len(history_paths)))
        else:
            # load_vector_fn = load_vector_file_lru1024 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(36, max(4, len(history_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(load_vector_fn, path, idx): path
                for path, idx in zip(history_paths, indices_to_load)
            }
            for future in as_completed(future_to_path):
                np_dict, idx = future.result()
                history_feats[idx] = np_dict
        return history_feats
    
    
    def prepare_token_drop(self, epis_idx, step_idx):
        token_budget = self.history_config.budget
        kept_indices = json.load(open(os.path.join(self.feature_dir, f"episode_{epis_idx}", "kept_indices.json")))
        
        return self.mem_buffer.prepare_token_dropping(
            step_idx, token_budget, self._gather_history_feat, 
            kept_indices=kept_indices, epis_idx=epis_idx)
        

    def prepare_frame_sampling(self,  epis_idx,  step_idx):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget

        return self.mem_buffer.prepare_frame_sampling(
            step_idx, token_budget, token_per_image, self._gather_history_feat, 
            epis_idx=epis_idx)


    
    def prepare_token_recurrent(self, epis_idx, step_idx, exec_start_idx):      
        self.mem_buffer.exec_start_idx = exec_start_idx
        
        return self.mem_buffer.prepare_token_recurrent(
            step_idx, exec_start_idx, self._gather_history_feat,
            epis_idx=epis_idx)
        
    
    def __len__(self):
        return len(self.dataset)
    
    
    def _normalize_state(self, state):
        if self.compute_norm_stats:
            return state
        else:
            if self.use_quantiles:
                return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
            else:
                return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)
    
    def _truncated_gaussian_noise(self, max_val: int, std: float = None) -> int:
        """Generate truncated Gaussian noise centered at 0, clamped to [-max_val, max_val]."""
        if std is None:
            std = max_val / 2.5  # ~95% of samples within range before clamping
        noise = random.gauss(0, std)
        noise = max(-max_val, min(max_val, noise))  # clamp to [-max_val, max_val]
        return int(round(noise))

    def add_grounding_augmentation(self, subgoal: str, noise_range: int = 8) -> str:
        if not subgoal:
            return subgoal
        
        matches = re.findall(r'at <(\d+), (\d+)>', subgoal)
        
        if len(matches) == 0 or len(matches) > 1:
            return subgoal
        
        x, y = matches[0]
        x = int(x)
        y = int(y)
        noise_x = self._truncated_gaussian_noise(noise_range)
        noise_y = self._truncated_gaussian_noise(noise_range)

        new_subgoal = subgoal.replace(f'at <{x}, {y}>', f'at <{x + noise_x}, {y + noise_y}>')
        return new_subgoal
        

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        data["actions"] = data["actions"][:self.action_horizon]
        
        # During online evaluation, the ground-truth subgoal may change earlier than when it was recorded.
        # To make the model robust to this temporal shift and avoid train/test distribution mismatch,
        # we randomly sample from either subgoal or subgoal_online (early change).
        if self.history_config is not None \
            and self.history_config.representation_type == "symbolic" \
            and self.history_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"] \
            and random.random() < 0.5: 
            data["simple_subgoal"] = data["simple_subgoal_online"]
            data["grounded_subgoal"] = data["grounded_subgoal_online"]
        data.pop("simple_subgoal_online")
        data.pop("grounded_subgoal_online")
        
        if self.history_config is not None and self.history_config.representation_type == "symbolic":
            data["grounded_subgoal"] = self.add_grounding_augmentation(data["grounded_subgoal"], noise_range=8)
            data["simple_subgoal"] = self.add_grounding_augmentation(data["simple_subgoal"], noise_range=8)
 
        
        if self.history_config is not None:
            # use history
            epis_idx = data["epis_idx"].item()
            step_idx = data["step_idx"].item()
            exec_start_idx = data["exec_start_idx"].item()

            if self.history_config.representation_type == "perceptual":
                if self.history_config.perceptual_memory.type == "token_dropping":
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # >=64
                    ) = self.prepare_token_drop(epis_idx, step_idx)
                else:
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # slow >=64, mid >=16,fast >=4
                    ) = self.prepare_frame_sampling(epis_idx, step_idx)
                
                data["static_image_emb"] = static_img_emb
                data["static_pos_emb"] = static_pos_emb
                data["static_state_emb"] = self._normalize_state(static_state_emb)
                data["static_mask"] = static_mask

            elif self.history_config.representation_type == "recurrent":             
                recur_img_emb, recur_pos_emb, recur_state_emb, recur_mask = (
                    self.prepare_token_recurrent(
                        epis_idx, step_idx, exec_start_idx)
                )

                data["recur_image_emb"] = recur_img_emb
                data["recur_pos_emb"] = recur_pos_emb
                data["recur_state_emb"] = self._normalize_state(recur_state_emb)
                data["recur_mask"] = recur_mask  # 1 ~ max_exec_steps

            elif self.history_config.representation_type == "symbolic":
                pass
            else:
                raise ValueError(
                    f"Not supported representation type: {self.history_config.representation_type}"
                )
        
        
        for key in [
            "static_image_emb",
            "static_pos_emb",
            "static_state_emb",
            "static_mask",
            
            "recur_image_emb",
            "recur_pos_emb",
            "recur_state_emb",
            "recur_mask",
            
            "simple_subgoal",
            "grounded_subgoal",
            "prompt"
        ]:
            if key not in data:
                data[key] = None        

        return data


class RoboTTTSequenceDataset(Dataset):
    def __init__(self, manifest_path, data_config, temporal_blocks, segment_length):
        manifest = json.load(open(manifest_path))
        if manifest["max_phase_aligned_blocks"] > 96:
            raise ValueError("RoboTTT manifest exceeds the approved T96 horizon")
        self.episodes = manifest["episodes"]
        self.ends = [episode["cumulative_execution_count"] for episode in self.episodes]
        self.temporal_blocks = temporal_blocks
        self.segment_length = 1 if temporal_blocks == 1 else segment_length
        self.files = {}
        model_transforms = data_config.model_transforms.inputs
        self.transform = _transforms.compose(
            [*data_config.data_transforms.inputs,
             _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
             *model_transforms[:2], *model_transforms[3:]]
        )
        self.tokenize = model_transforms[2]

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        episode_index = bisect.bisect_right(self.ends, index)
        episode = self.episodes[episode_index]
        previous_end = self.ends[episode_index - 1] if episode_index else 0
        anchor = episode["exec_start_idx"] + index - previous_end
        start = max(anchor - 16 * (self.temporal_blocks - 1), anchor % 16)
        block_indices = np.arange(start, anchor + 1, 16, dtype=np.int32)
        is_video = block_indices < episode["exec_start_idx"]

        path = episode["h5_path"]
        if path not in self.files:
            self.files[path] = h5py.File(path, "r")
        episode_group = self.files[path][episode["episode_group"]]
        tokenized = self.tokenize({
            "prompt": episode["task_goal"], "state": np.zeros(8),
            "simple_subgoal": None, "grounded_subgoal": None,
        })

        blocks = []
        for block_index, video in zip(block_indices, is_video):
            timestep = episode_group[f"timestep_{block_index}"]
            action_indices = np.minimum(block_index + np.arange(20), episode["num_steps"] - 1)
            sample = {
                "observation/image": timestep["obs/front_rgb"][()],
                "observation/wrist_image": timestep["obs/wrist_rgb"][()],
                "observation/state": np.concatenate(
                    [timestep["obs/joint_state"][()], timestep["obs/gripper_state"][()][:1]]
                ).astype(np.float32),
                "actions": np.stack([
                    episode_group[f"timestep_{step}"]["action/joint_action"][()]
                    for step in action_indices
                ]).astype(np.float32),
                "prompt": episode["task_goal"],
            }
            block = self.transform(sample)
            block = {key: value for key, value in block.items() if value is not None}
            block.pop("prompt", None)
            block.update({key: tokenized[key] for key in ("tokenized_prompt", "tokenized_prompt_mask")})
            if video:
                block["actions"][:] = 0
            blocks.append(block)
        sequence = jax.tree.map(lambda *values: np.stack(values), *blocks)

        segment_length = self.segment_length
        first_execution = int(np.sum(is_video))
        left_padding = (segment_length - 1 - first_execution) % segment_length
        target_segments = 1 + (self.temporal_blocks - 1 + segment_length - 1) // segment_length
        right_padding = target_segments * segment_length - left_padding - len(block_indices)
        padding = (left_padding, right_padding)
        packed = jax.tree.map(
            lambda value: np.pad(value, [padding] + [(0, 0)] * (value.ndim - 1)),
            sequence,
        )
        valid = np.pad(np.ones(len(block_indices), dtype=np.bool_), padding)
        video_mask = np.pad(is_video, padding)
        inner_mask = np.repeat(valid[..., None], 36, axis=-1)
        inner_mask[video_mask, :20] = False
        packed["robottt"] = {
            "valid": valid,
            "inner": inner_mask,
            "outer": valid & ~video_mask,
        }
        return packed

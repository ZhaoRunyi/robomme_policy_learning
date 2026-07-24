"""
We implemented our own data loader,
which can be 5-10x faster than LeRobot dataloader and can avoid memory explosion issue
"""

import jax
import logging
import torch
from omegaconf import DictConfig
from openpi.models import model as _model
from openpi.training.data_loader import DataLoader, TorchDataLoader,transform_dataset
import openpi.training.config as _config

from mme_vla_suite.training.dataset import RoboMMEDataset, RoboTTTSequenceDataset
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.config.utils import get_history_config



class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
        self._total_samples = len(data_loader._data_loader.dataset)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            if "robottt" not in batch:
                batch = HistAugObservation.from_dict(batch), batch["actions"]
            yield batch


class StepSampler(torch.utils.data.Sampler):
    def __init__(self, size, batch_size, seed, step):
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch, self.batch_offset = divmod(step, size // batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator)[: self.size // self.batch_size * self.batch_size]
        indices = indices[self.batch_offset * self.batch_size:]
        self.epoch += 1
        self.batch_offset = 0
        return iter(indices.tolist())

def create_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    history_config: str | DictConfig | None,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    temporal_blocks: int | None = None,
    start_step: int = 0,
) -> DataLoader[tuple[HistAugObservation, _model.Actions]]:
    
    history_config = get_history_config(history_config)

    if temporal_blocks is not None:
        dataset = RoboTTTSequenceDataset(dataset_path, data_config, temporal_blocks)
    else:
        dataset = RoboMMEDataset(dataset_path, data_config, history_config, action_horizon)
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")
    
    sampler = StepSampler(len(dataset), local_batch_size, seed, start_step) if temporal_blocks is not None else None
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if temporal_blocks is not None else sharding,
        shuffle=shuffle,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="pytorch" if temporal_blocks is not None else "jax",
    )

    return DataLoaderImpl(data_config, data_loader)

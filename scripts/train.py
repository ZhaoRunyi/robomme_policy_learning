import dataclasses
import concurrent.futures
import functools
import gc
import logging
import os
import platform
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import time

from typing import Any
import tqdm_loggable.auto as tqdm
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


import mme_vla_suite.models.integration.history_pi0 as _model
from mme_vla_suite.models.integration.history_observation import (
    HistAugObservation,
)
import mme_vla_suite.training.config as _config
import mme_vla_suite.training.dataloader as _data_loader
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.representation.robottt import create_fast_state


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="allow", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

def init_history_config(config: _config.TrainConfig):
    # this is for evaluation config checking
    if config.model.history_config is not None:
        with open(config.checkpoint_dir / "history_config.txt", "w") as f:
            f.write(config.model.history_config)

def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True
    )

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


def params_split(params, trainable_filter):
    memory_filter = params.filter(trainable_filter).filter(
        nnx.All(nnx.Param, nnx_utils.PathRegex(".*mem.*"))
    )
    non_memory_filter = params.filter(trainable_filter).filter(
        nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*mem.*")))
    )
    return memory_filter, non_memory_filter


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    use_robottt = "robottt" in str(config.model.history_config)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    if use_robottt:
        path_text = lambda path: jax.tree_util.keystr(path).lower()
        robottt_mask = lambda params: jax.tree_util.tree_map_with_path(
            lambda path, _: "robottt" in path_text(path), params)
        decay_names = ("kernel", "w", "gating_einsum", "linear", "initial_w1", "initial_w2")
        decay_mask = lambda params: jax.tree_util.tree_map_with_path(
            lambda path, _: any(f"['{name}']" in path_text(path) for name in decay_names), params)
        schedule = optax.join_schedules(
            (
                optax.linear_schedule(1e-5 / 10_001, 1e-5, 10_000),
                optax.constant_schedule(1e-5),
                optax.cosine_decay_schedule(1e-5, 20_000, alpha=0.1),
            ),
            (10_000, 80_000),
        )
        tx = optax.chain(
            config.optimizer.create(schedule, decay_mask),
            optax.masked(optax.scale(5.0), robottt_mask),
        )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )
        logging.info(
            f"Total Model Size: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1024 / 1024} MB"
        )
        logging.info(
            f"Trainable Model Size: {sum(x.size for x in jax.tree_util.tree_leaves(params.filter(config.trainable_filter))) / 1024 / 1024} MB"
        )

        memory_params, non_memory_params = params_split(params, config.trainable_filter)
        logging.info(
            f"Memory-related  Size: {sum(x.size for x in jax.tree_util.tree_leaves(memory_params))/1024/1024} MB"
        )
        logging.info(
            f"Non-Memory Size: {sum(x.size for x in jax.tree_util.tree_leaves(non_memory_params))/1024/1024} MB"
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume and not use_robottt:
        # replace pi05_base with the checkpoint id
        ckpt_epath = config.checkpoint_dir / str(config.resum_ckpt_id) / "params"
        weight_loader = _weight_loaders.CheckpointWeightLoader(str(ckpt_epath))
        partial_params = _load_weights_and_validate(weight_loader, train_state_shape.params.to_pure_dict())
    elif resume:
        partial_params = _load_weights_and_validate(
            _weight_loaders.NoOpWeightLoader(), train_state_shape.params.to_pure_dict())
    else:
        partial_params = _load_weights_and_validate(
            config.weight_loader, train_state_shape.params.to_pure_dict()
        )
        
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[HistAugObservation, _model.Actions],
) -> tuple[
    training_utils.TrainState, dict[str, at.Array], Any
]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.HistoryPi0,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        actions: _model.Actions,
    ):
        chunked_loss, stats = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), stats

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, stats), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True
    )(model, train_rng, observation, actions)

    return apply_gradients(config, state, model, grads, loss, stats)


def apply_gradients(config, state, model, grads, loss, stats):
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )

    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "llm_grad_norm": optax.global_norm(grads.PaliGemma.llm),
    }
    if "robottt" in str(config.model.history_config):
        info["train/ttt/slow_grad_norm"] = optax.global_norm(grads.filter(nnx.PathContains("robottt")))
    if config.model.use_history and hasattr(grads, "mem_encoder"):
        info["mem_enc_norm"] = optax.global_norm(grads.mem_encoder)

    return new_state, info, stats


def robottt_segment_grad(config, rng, state, batch, fast_state, position, segment_index):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model):
        numerator, count, next_state, next_position = model.compute_robottt_segment_loss(
            rng, segment_index, *batch, fast_state, position)
        return numerator, (count, next_state, next_position)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (numerator, (count, fast_state, position)), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True)(model)
    return grads, numerator, count, fast_state, position


def robottt_bundle_grad(config, rng, state, batch, fast_state, position, segment_indices):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model):
        def run_segment(carry, inputs):
            numerator, count, fast_state, position = carry
            observation, actions, masks, segment_index = inputs
            next_numerator, next_count, fast_state, position = model.compute_robottt_segment_loss(
                rng, segment_index, observation, actions, masks, fast_state, position)
            return (
                numerator + next_numerator,
                count + next_count,
                fast_state,
                position,
            ), None

        scan_batch = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), batch)
        initial = (jnp.float32(0), jnp.int32(0), fast_state, position)
        result, _ = jax.lax.scan(run_segment, initial, (*scan_batch, segment_indices))
        numerator, count, next_fast_state, next_position = result
        return numerator, (count, next_fast_state, next_position)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (numerator, (count, fast_state, position)), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True)(model)
    return grads, numerator, count, fast_state, position


def robottt_stream_bundle_grad(
    config, rng, state, batch, fast_state, position, segment_indices):
    first_batch = jax.tree.map(lambda value: value[:, 0], batch)
    accumulated = robottt_segment_grad(
        config, rng, state, first_batch, fast_state, position, segment_indices[0])

    def run_segment(carry, inputs):
        gradient_sum, numerator_sum, count_sum, fast_state, position = carry
        observation, actions, masks, segment_index = inputs
        grads, numerator, count, fast_state, position = robottt_segment_grad(
            config,
            rng,
            state,
            (observation, actions, masks),
            fast_state,
            position,
            segment_index,
        )
        return (
            jax.tree.map(jnp.add, gradient_sum, grads),
            numerator_sum + numerator,
            count_sum + count,
            fast_state,
            position,
        ), None

    scan_batch = jax.tree.map(lambda value: jnp.swapaxes(value[:, 1:], 0, 1), batch)
    accumulated, _ = jax.lax.scan(
        run_segment,
        accumulated,
        (*scan_batch, segment_indices[1:]),
    )
    return accumulated


def robottt_segment_forward(config, rng, state, batch, fast_state, position, segment_index):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    _, _, fast_state, position = model.compute_robottt_segment_loss(
        rng, segment_index, *batch, fast_state, position)
    return fast_state, position


def robottt_bundle_forward(config, rng, state, batch, fast_state, position, segment_indices):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def run_segment(carry, inputs):
        fast_state, position = carry
        observation, actions, masks, segment_index = inputs
        _, _, fast_state, position = model.compute_robottt_segment_loss(
            rng, segment_index, observation, actions, masks, fast_state, position)
        return (fast_state, position), None

    scan_batch = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), batch)
    (fast_state, position), _ = jax.lax.scan(
        run_segment, (fast_state, position), (*scan_batch, segment_indices))
    return fast_state, position


def robottt_segment_accumulate(
    config, rng, state, batch, fast_state, position, segment_index, accumulated):
    grads, numerator, count, fast_state, position = robottt_segment_grad(
        config, rng, state, batch, fast_state, position, segment_index)
    gradient_sum, numerator_sum, count_sum = accumulated
    accumulated = (
        jax.tree.map(jnp.add, gradient_sum, grads),
        numerator_sum + numerator,
        count_sum + count,
    )
    return accumulated, fast_state, position


def robottt_bundle_accumulate(
    config, rng, state, batch, fast_state, position, segment_indices, accumulated):
    grads, numerator, count, fast_state, position = robottt_bundle_grad(
        config, rng, state, batch, fast_state, position, segment_indices)
    gradient_sum, numerator_sum, count_sum = accumulated
    accumulated = (
        jax.tree.map(jnp.add, gradient_sum, grads),
        numerator_sum + numerator,
        count_sum + count,
    )
    return accumulated, fast_state, position


def robottt_stream_bundle_accumulate(
    config, rng, state, batch, fast_state, position, segment_indices, accumulated):
    grads, numerator, count, fast_state, position = robottt_stream_bundle_grad(
        config, rng, state, batch, fast_state, position, segment_indices)
    gradient_sum, numerator_sum, count_sum = accumulated
    accumulated = (
        jax.tree.map(jnp.add, gradient_sum, grads),
        numerator_sum + numerator,
        count_sum + count,
    )
    return accumulated, fast_state, position


def apply_robottt_gradients(config, state, grads, numerator, count):
    model = nnx.merge(state.model_def, state.params)
    grads = jax.tree.map(lambda value: value / jnp.maximum(count, 1), grads)
    return apply_gradients(
        config, state, model, grads, numerator / jnp.maximum(count, 1), None)[:2]


@functools.partial(jax.jit, donate_argnums=(0,))
def accumulate_robottt_gradients(accumulated, current):
    gradient_sum, numerator_sum, count_sum = accumulated
    grads, numerator, count = current
    return jax.tree.map(jnp.add, gradient_sum, grads), numerator_sum + numerator, count_sum + count


def get_stats(stats_dict) -> dict[str, at.Array]:
    mask = stats_dict["mask"]
    if mask.ndim == 2:
        b, l = mask.shape  # for rmt
    else:
        b, _, l = mask.shape  # for ttt
        stats_dict = {k: v.mean(axis=1) for k, v in stats_dict.items()}

    dic = {}
    for k in stats_dict.keys():
        dic[k] = [[] for _ in range(l)]

    mask = stats_dict["mask"]
    for batch_idx in range(b):
        for step_idx in range(l):
            if mask[batch_idx, step_idx]:
                for k, v in stats_dict.items():
                    dic[k][step_idx].append(v[batch_idx, step_idx])

    stats = {}
    for k, v in dic.items():
        stats[k] = np.zeros((l,))
        for step in range(l):
            if len(v[step]) > 0:
                stats[k][step] = np.array(v[step]).mean()
    return stats


def main(config: _config.TrainConfig, tentative_run: bool = False):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"TrainConfig: {config}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path(f"~/.cache/jax_{config.exp_name}").expanduser()),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())

    use_robottt = "robottt" in str(config.model.history_config)
    if use_robottt and config.resum_ckpt_id is not None:
        raise ValueError("RoboTTT weights-only warm starts use --weight-loader")
    if (config.resume_from_dir is None) != (config.resume_step is None):
        raise ValueError("--resume-from-dir and --resume-step must be set together")
    if not use_robottt and config.resume_from_dir is not None:
        raise ValueError("--resume-from-dir is only supported by RoboTTT")
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        full_state=use_robottt,
    )
    restore_manager = checkpoint_manager
    restore_step = checkpoint_manager.latest_step() if use_robottt and resuming else None
    if use_robottt and config.resume_from_dir is not None:
        if config.resume:
            raise ValueError("--resume and --resume-from-dir are mutually exclusive")
        restore_manager, _ = _checkpoints.initialize_checkpoint_dir(
            config.resume_from_dir, keep_period=None, overwrite=False, resume=True,
            full_state=True, read_only=True)
        restore_step = config.resume_step
    if restore_step is not None:
        if restore_step not in restore_manager.all_steps():
            raise FileNotFoundError(f"Checkpoint step {restore_step} is missing")
        if restore_step >= config.num_train_steps:
            logging.info(f"Checkpoint step {restore_step} already reached target {config.num_train_steps}")
            restore_manager.close()
            if restore_manager is not checkpoint_manager:
                checkpoint_manager.close()
            return
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    init_history_config(config)
    history_config = get_history_config(config.model.history_config)
    
    if history_config:
        if history_config.streaming_obs_horizon == 16:
            assert config.model.action_horizon == 20, "action_horizon must be 20 when streaming_obs_horizon is 16"
        else:
            raise ValueError(f"Unsupported streaming_obs_horizon: {history_config.streaming_obs_horizon}")

    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    curriculum = ((0, 1), (50_000, 4), (60_000, 8), (70_000, 32), (80_000, 64), (90_000, 96))
    stage_at = lambda step: next(stage for stage in reversed(curriculum) if step >= stage[0])

    make_loader = functools.partial(
        _data_loader.create_data_loader,
        config.dataset_path,
        data_config,
        history_config=config.model.history_config,
        sharding=data_sharding,
        shuffle=True,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    selected_step = restore_step or 0
    stage_start, temporal_blocks = stage_at(selected_step) if use_robottt else (0, None)
    data_loader = make_loader(
        seed=config.seed + (temporal_blocks or 0), temporal_blocks=temporal_blocks,
        start_step=selected_step - stage_start)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(
        f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
    )

    # Log images from first batch to sanity check.
    images_to_log = [] if use_robottt else [
        wandb.Image(
            np.concatenate(
                [np.array(img[i]) for img in batch[0].images.values()], axis=1
            )
        )
        for i in range(min(1, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=restore_step is not None if use_robottt else resuming)
    if use_robottt and restore_step is not None:
        train_state = _checkpoints.restore_state(
            restore_manager, train_state, data_loader, step=restore_step, full_state=True)
        if int(train_state.step) != restore_step:
            raise ValueError(f"Restored state.step {int(train_state.step)} != checkpoint {restore_step}")
        if restore_manager is not checkpoint_manager:
            restore_manager.close()
    jax.block_until_ready(train_state)
    logging.info(
        f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params.filter(nnx.All(nnx.Param)))}"
    )
    if use_robottt:
        robottt_parameter_count = sum(
            value.size for value in jax.tree.leaves(train_state.params.filter(nnx.PathContains("robottt"))))
        register_parameter_count = train_state.params["robottt_register_tokens"].value.size
        layer_parameter_count = robottt_parameter_count
        fast_state_shape = jax.eval_shape(lambda: create_fast_state(1))
        fast_state_bytes = sum(
            value.size * value.dtype.itemsize for value in jax.tree.leaves(fast_state_shape))
        logging.info(
            "RoboTTT parameters: %d per layer, %d total; fast state: %.2f MiB per sample",
            layer_parameter_count // 18,
            robottt_parameter_count + register_parameter_count,
            fast_state_bytes / 2**20,
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    if use_robottt:
        fast_state_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(None, sharding.DATA_AXIS))
        logging.info(f"RoboTTT fast-state sharding: {fast_state_sharding.spec}")
        psegment_grad = jax.jit(functools.partial(robottt_segment_grad, config))
        pbundle_grad = jax.jit(functools.partial(robottt_bundle_grad, config))
        pstream_bundle_grad = jax.jit(
            functools.partial(robottt_stream_bundle_grad, config))
        psegment_forward = jax.jit(
            functools.partial(robottt_segment_forward, config), donate_argnums=(3, 4))
        pbundle_forward = jax.jit(
            functools.partial(robottt_bundle_forward, config), donate_argnums=(3, 4))
        psegment_accumulate = jax.jit(
            functools.partial(robottt_segment_accumulate, config),
            donate_argnums=(3, 4, 6),
        )
        pbundle_accumulate = jax.jit(
            functools.partial(robottt_bundle_accumulate, config),
            donate_argnums=(3, 4, 6),
        )
        pstream_bundle_accumulate = jax.jit(
            functools.partial(robottt_stream_bundle_accumulate, config),
            donate_argnums=(3, 4, 6),
        )
        papply_gradients = jax.jit(
            functools.partial(apply_robottt_gradients, config),
            donate_argnums=(0, 1),
        )
        pcreate_fast_state = jax.jit(create_fast_state, static_argnums=0, out_shardings=fast_state_sharding)
        def run_robottt_batch(state, host_batch, microbatch_size, segment_length):
            gradient_sum = None
            numerator_sum, count_sum = jnp.float32(0), jnp.int32(0)
            segment_calls = 0
            padding_segments_skipped = 0
            context_forward_calls = 0
            bundle_size = int(os.environ.get("ROBOTTT_BUNDLE", "1"))
            accumulation = os.environ.get("ROBOTTT_GRAD_ACCUMULATION", "tree_add")
            context_forward = os.environ.get("ROBOTTT_CONTEXT_FORWARD") == "1"
            stream_bundle = os.environ.get("ROBOTTT_STREAM_BUNDLE") == "1"
            for micro_start in range(0, config.batch_size, microbatch_size):
                micro_slice = slice(micro_start, micro_start + microbatch_size)
                fast_state = pcreate_fast_state(microbatch_size)
                position = jax.device_put(jnp.zeros(microbatch_size, jnp.int32), data_sharding)
                micro_rng = jax.random.fold_in(jax.random.fold_in(train_rng, state.step), micro_start // microbatch_size)
                segments = []
                segment_indices = []
                for segment_start in range(0, host_batch["actions"].shape[1], segment_length):
                    segment = jax.tree.map(
                        lambda value: np.asarray(
                            value[micro_slice, segment_start : segment_start + segment_length]
                        ),
                        host_batch,
                    )
                    if (
                        os.environ.get("ROBOTTT_PADDING_SKIP") == "1"
                        and not np.any(segment["robottt"]["valid"])
                    ):
                        padding_segments_skipped += 1
                        continue
                    segments.append(segment)
                    segment_indices.append(segment_start // segment_length)
                work_bundles = []
                for segment, segment_index in zip(segments, segment_indices, strict=True):
                    is_context = (
                        context_forward
                        and np.any(segment["robottt"]["valid"])
                        and not np.any(segment["robottt"]["outer"])
                    )
                    if (
                        not work_bundles
                        or len(work_bundles[-1]) == bundle_size
                        or work_bundles[-1][0][2] != is_context
                    ):
                        work_bundles.append([])
                    work_bundles[-1].append((segment, segment_index, is_context))
                for work_bundle in work_bundles:
                    bundle = [item[0] for item in work_bundle]
                    indices = np.asarray([item[1] for item in work_bundle])
                    is_context = work_bundle[0][2]
                    if len(bundle) == 1:
                        segment = bundle[0]
                        masks = segment.pop("robottt")
                        actions = segment.pop("actions")
                        device_batch = jax.device_put(
                            (HistAugObservation.from_dict(segment), actions, masks), data_sharding)
                        grad_function = psegment_grad
                        forward_function = psegment_forward
                        accumulate_function = psegment_accumulate
                        grad_args = (device_batch, fast_state, position, int(indices[0]))
                    else:
                        segment = jax.tree.map(lambda *values: np.stack(values, axis=1), *bundle)
                        masks = segment.pop("robottt")
                        actions = segment.pop("actions")
                        device_batch = jax.device_put(
                            (HistAugObservation.from_dict(segment), actions, masks), data_sharding)
                        grad_function = pbundle_grad
                        forward_function = pbundle_forward
                        accumulate_function = pbundle_accumulate
                        if stream_bundle:
                            grad_function = pstream_bundle_grad
                            accumulate_function = pstream_bundle_accumulate
                        grad_args = (device_batch, fast_state, position, jax.device_put(indices))
                    with sharding.set_mesh(mesh):
                        if is_context:
                            fast_state, position = forward_function(micro_rng, state, *grad_args)
                        elif accumulation == "fused_accum" and gradient_sum is not None:
                            accumulated, fast_state, position = accumulate_function(
                                micro_rng,
                                state,
                                *grad_args,
                                (gradient_sum, numerator_sum, count_sum),
                            )
                            gradient_sum, numerator_sum, count_sum = accumulated
                        else:
                            grads, numerator, count, fast_state, position = grad_function(
                                micro_rng, state, *grad_args)
                    segment_calls += 1
                    if is_context:
                        context_forward_calls += 1
                    elif accumulation == "fused_accum" and gradient_sum is not None:
                        pass
                    elif gradient_sum is None:
                        gradient_sum, numerator_sum, count_sum = grads, numerator, count
                    elif accumulation == "jitted_tree_add":
                        gradient_sum, numerator_sum, count_sum = accumulate_robottt_gradients(
                            (gradient_sum, numerator_sum, count_sum), (grads, numerator, count))
                    else:
                        gradient_sum = jax.tree.map(jnp.add, gradient_sum, grads)
                        numerator_sum += numerator
                        count_sum += count
            with sharding.set_mesh(mesh):
                return (
                    *papply_gradients(state, gradient_sum, numerator_sum, count_sum),
                    {
                        "segment_calls": segment_calls,
                        "padding_segments_skipped": padding_segments_skipped,
                        "context_forward_calls": context_forward_calls,
                    },
                )

    prefetch_executor = None
    next_batch = None
    if os.environ.get("ROBOTTT_HOST_PREFETCH") == "1":
        prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        next_batch = prefetch_executor.submit(next, data_iter)

    start_step = int(train_state.step)
    tentative_run_step = 10 # on our cluster, we need to run the tentative run for a few steps to warm up the A40 machine. 
    # Otherwise it would be very slow. I guess it is because of JAX compilation cache.
    
    if config.resum_ckpt_id is not None and not use_robottt:
        start_step += config.resum_ckpt_id
        tentative_run_step += config.resum_ckpt_id
    microbatch_by_stage = {
        1: 64,
        4: 16,
        8: 16,
        32: 8,
        64: 8,
        96: 8,
    }
    if microbatch_values := os.environ.get("ROBOTTT_MICROBATCH_BY_STAGE"):
        microbatch_by_stage.update(
            dict(zip(microbatch_by_stage, map(int, microbatch_values.split(",")), strict=True)))
    
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        next_stage_start, next_temporal_blocks = stage_at(step) if use_robottt else (0, None)
        if use_robottt and next_temporal_blocks != temporal_blocks:
            checkpoint_manager.wait_until_finished()
            jax.block_until_ready(train_state)
            jax.clear_caches()
            gc.collect()
            stage_start, temporal_blocks = next_stage_start, next_temporal_blocks
            data_loader = make_loader(
                seed=config.seed + temporal_blocks, temporal_blocks=temporal_blocks,
                start_step=step - stage_start)
            data_iter = iter(data_loader)
            batch = next(data_iter)
            if prefetch_executor is not None:
                next_batch.cancel()
                next_batch = prefetch_executor.submit(next, data_iter)
        if use_robottt:
            segment_length = 1 if temporal_blocks == 1 else history_config.recurrent_memory.tbptt_segment_length
            if step == start_step or step == stage_start:
                logging.info(f"RoboTTT T{temporal_blocks}: microbatch {microbatch_by_stage[temporal_blocks]}")
            update_start = time.perf_counter()
            train_state, info, stats = run_robottt_batch(
                train_state, batch, microbatch_by_stage[temporal_blocks], segment_length)
            jax.block_until_ready(train_state)
            update_seconds = time.perf_counter() - update_start
            info.update({
                "train/perf/update_seconds": update_seconds,
                "train/perf/valid_blocks_per_second": (
                    float(batch["robottt"]["valid"].sum()) / update_seconds
                ),
                "train/perf/segment_calls": stats["segment_calls"],
                "train/perf/padding_segments_skipped": stats["padding_segments_skipped"],
                "train/perf/context_forward_calls": stats["context_forward_calls"],
            })
        else:
            with sharding.set_mesh(mesh):
                train_state, info, stats = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if use_robottt:
                robottt_params = train_state.params["PaliGemma"]["llm"]["layers"]["robottt"]
                inner_learning_rate = 0.1 * jax.nn.sigmoid(robottt_params["inner_learning_rate_raw"].value)
                gate = jnp.abs(jnp.tanh(robottt_params["residual_gate"].value))
                reduced_info.update({
                    "train/ttt/inner_lr_mean": float(jnp.mean(inner_learning_rate)),
                    "train/ttt/inner_lr_min": float(jnp.min(inner_learning_rate)),
                    "train/ttt/inner_lr_max": float(jnp.max(inner_learning_rate)),
                    "train/ttt/gate_abs_mean": float(jnp.mean(gate)),
                    "train/ttt/gate_abs_max": float(jnp.max(gate)),
                })

            if (
                not use_robottt and config.model.use_history and history_config.representation_type == "recurrent"
                and history_config.recurrent_memory.output_stats
            ):
                stats = jax.device_get(stats)
                if stats:
                    stats_dict = get_stats(stats)
                    pbar.write(f"Recurrent Memory Stats: {stats_dict}")

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if prefetch_executor is None:
            batch = next(data_iter)
        else:
            batch = next_batch.result()
            next_batch = prefetch_executor.submit(next, data_iter)

        if tentative_run and step > tentative_run_step:
            print("\n\n\n==========Tentative run completed==========\n\n\n")
            break

        checkpoint_step = int(train_state.step) if use_robottt else step
        if (
            checkpoint_step % config.save_interval == 0 and checkpoint_step > start_step
        ) or step == config.num_train_steps - 1:
            _checkpoints.save_state(
                checkpoint_manager, train_state, data_loader, checkpoint_step, full_state=use_robottt
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    train_config = _config.cli()
    use_robottt = "robottt" in str(train_config.model.history_config)
    main(train_config, tentative_run=not use_robottt)
    if not use_robottt:
        time.sleep(20)
        main(train_config)

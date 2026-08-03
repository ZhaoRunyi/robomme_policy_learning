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
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


from mme_vla_suite.models.integration.history_observation import (
    HistAugObservation,
)
import mme_vla_suite.training.config as _config
import mme_vla_suite.training.dataloader as _data_loader
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.robottt_fast import (
    PACKED_SEGMENTS,
    alias_frozen_ema,
    collect_lane,
    make_boundary_transport_jit,
    merge_packed,
    pack_context,
    packed_grad,
    padding_plan,
    prepare_lane,
    regroup_lane,
)


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


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    if not config.train_robottt_only:
        raise ValueError("The packed RoboTTT path requires --train-robottt-only")
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

    loader = _weight_loaders.NoOpWeightLoader() if resume else config.weight_loader
    partial_params = _load_weights_and_validate(
        loader, train_state_shape.params.to_pure_dict())

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def apply_gradients(config, state, model, grads, loss):
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
                new_params.filter(config.trainable_filter),
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
    info["train/ttt/slow_grad_norm"] = optax.global_norm(
        grads.filter(nnx.PathContains("robottt")))
    return new_state, info


def apply_robottt_gradients(config, state, grads, numerator, count):
    model = nnx.merge(state.model_def, state.params)
    grads = jax.tree.map(lambda value: value / jnp.maximum(count, 1), grads)
    return apply_gradients(
        config, state, model, grads, numerator / jnp.maximum(count, 1))


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"TrainConfig: {config}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    cache_home = epath.Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
    jax.config.update(
        "jax_compilation_cache_dir",
        str(cache_home / f"jax_{config.exp_name}"),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    if "robottt" not in str(config.model.history_config):
        raise ValueError("The packed training path only supports RoboTTT")
    if config.resum_ckpt_id is not None:
        raise ValueError("RoboTTT weights-only warm starts use --weight-loader")
    if (config.resume_from_dir is None) != (config.resume_step is None):
        raise ValueError("--resume-from-dir and --resume-step must be set together")
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        full_state=True,
    )
    restore_manager = checkpoint_manager
    restore_step = checkpoint_manager.latest_step() if resuming else None
    if config.resume_from_dir is not None:
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
    if history_config.streaming_obs_horizon != 16:
        raise ValueError(f"Unsupported streaming_obs_horizon: {history_config.streaming_obs_horizon}")
    assert config.model.action_horizon == 20, "action_horizon must be 20 when streaming_obs_horizon is 16"

    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    curriculum = ((0, 1), (50_000, 4), (60_000, 8), (70_000, 32), (80_000, 64), (90_000, 96))
    if config.robottt_stage_steps is not None:
        curriculum = tuple((index * config.robottt_stage_steps, blocks)
                           for index, blocks in enumerate((1, 4, 8, 32, 64, 96)))
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
    stage_start, temporal_blocks = stage_at(selected_step)
    data_loader = make_loader(
        seed=config.seed + temporal_blocks, temporal_blocks=temporal_blocks,
        start_step=selected_step - stage_start)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(
        f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
    )

    train_state, _ = init_train_state(
        config, init_rng, mesh, resume=restore_step is not None)
    if restore_step is not None:
        train_state = _checkpoints.restore_state(
            restore_manager, train_state, data_loader, step=restore_step, full_state=True)
        if int(train_state.step) != restore_step:
            raise ValueError(f"Restored state.step {int(train_state.step)} != checkpoint {restore_step}")
        if restore_manager is not checkpoint_manager:
            restore_manager.close()
    jax.block_until_ready(train_state)
    train_state = alias_frozen_ema(config, train_state)
    logging.info(
        f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params.filter(nnx.All(nnx.Param)))}"
    )
    pprepare_lane = jax.jit(functools.partial(prepare_lane, config))
    pregroup_lane = jax.jit(regroup_lane, static_argnums=(3, 4))
    pcollect_lane = jax.jit(
        functools.partial(collect_lane, config), static_argnums=3)
    ppack_context = make_boundary_transport_jit(pack_context, mesh)
    pmerge_packed = make_boundary_transport_jit(
        merge_packed, mesh, donate_argnums=(0, 1))
    ppacked_grad = jax.jit(
        functools.partial(packed_grad, config),
        donate_argnums=(1, 2),
    )
    papply_gradients_jit = jax.jit(
        functools.partial(apply_robottt_gradients, config),
        donate_argnums=(0, 1),
    )

    if "robottt" in str(config.model.history_config):
        def papply_gradients(state, gradients, numerator, count):
            trainable_ema, _ = state.ema_params.split(
                config.trainable_filter, ...)
            stripped_state = dataclasses.replace(
                state, ema_params=trainable_ema)
            state, info = papply_gradients_jit(
                stripped_state, gradients, numerator, count)
            return alias_frozen_ema(config, state), info

        def run_robottt_batch(state, host_batch, microbatch_size, segment_length):
            gradient_sum = None
            numerator_sum, count_sum = jnp.float32(0), jnp.int32(0)
            packed_calls = 0
            temporal_blocks = host_batch["actions"].shape[1]
            segment_count = temporal_blocks // segment_length
            if temporal_blocks <= 32:
                lane_group_size = 8
            elif temporal_blocks == 64:
                lane_group_size = 2
            else:
                lane_group_size = 1
            plans = padding_plan(
                host_batch["robottt"]["valid"], microbatch_size,
                lane_group_size, segment_length,
            )
            pending_packed = None
            pending_count = 0

            def accumulate_packed(packed):
                nonlocal gradient_sum, numerator_sum, count_sum, packed_calls
                with sharding.set_mesh(mesh):
                    gradient_sum, numerator_sum, count_sum = ppacked_grad(
                        state,
                        packed,
                        gradient_sum,
                        numerator_sum,
                        count_sum,
                    )
                packed_calls += 1

            for group_start in sorted({plan[0] for plan in plans}):
                group_plans = [plan for plan in plans if plan[0] == group_start]
                group_stop = group_plans[0][1]
                prepared_lanes = []
                mask_lanes = []
                for lane_start in range(
                    group_start, group_stop, microbatch_size):
                    lane_slice = slice(
                        lane_start, lane_start + microbatch_size)
                    lane_batch = jax.tree.map(
                        lambda value, selected=lane_slice: np.asarray(
                            value[selected]),
                        host_batch,
                    )
                    masks = lane_batch.pop("robottt")
                    actions = lane_batch.pop("actions")
                    device_batch = jax.device_put(
                        (
                            HistAugObservation.from_dict(
                                lane_batch, normalize_images=False),
                            actions, masks,
                        ),
                        data_sharding,
                    )
                    lane_rng = jax.random.fold_in(
                        jax.random.fold_in(train_rng, state.step),
                        lane_start // microbatch_size,
                    )
                    with sharding.set_mesh(mesh):
                        prepared, masks = pprepare_lane(
                            lane_rng, state, device_batch,
                            jax.device_put(np.arange(segment_count)),
                        )
                    prepared_lanes.append(prepared)
                    mask_lanes.append(masks)
                prepared_lanes = jax.tree.map(
                    lambda *values: jnp.stack(values), *prepared_lanes)
                mask_lanes = jax.tree.map(
                    lambda *values: jnp.stack(values), *mask_lanes)

                for _, _, sample_indices, lane_blocks in group_plans:
                    with sharding.set_mesh(mesh):
                        prepared, masks = pregroup_lane(
                            prepared_lanes, mask_lanes,
                            jax.device_put(sample_indices),
                            lane_blocks, microbatch_size)
                        context = pcollect_lane(
                            state, prepared, masks, segment_length)
                    observation = context.prepared.observation.replace(
                        images={}, image_masks={},
                        state=context.prepared.time[..., None],
                        tokenized_prompt=None, tokenized_prompt_mask=None,
                    )
                    context = context._replace(
                        prepared=context.prepared._replace(
                            observation=observation))
                    global_indices = group_start + sample_indices
                    outer = np.asarray(
                        host_batch["robottt"]["outer"][
                            global_indices, :lane_blocks
                        ]
                    ).reshape(
                        (
                            microbatch_size,
                            lane_blocks // segment_length,
                            segment_length,
                        )
                    )
                    pair_segments, pair_samples = np.nonzero(
                        np.any(outer, axis=-1).T
                    )

                    def pack_pairs(pair_start, pair_count):
                        pair_stop = pair_start + pair_count
                        pad = PACKED_SEGMENTS - pair_count
                        segment_indices = np.pad(
                            pair_segments[pair_start:pair_stop], (0, pad))
                        packed_sample_indices = np.pad(
                            pair_samples[pair_start:pair_stop], (0, pad))
                        packed_valid = np.arange(PACKED_SEGMENTS) < pair_count
                        with sharding.set_mesh(mesh):
                            return ppack_context(
                                context,
                                jax.device_put(segment_indices),
                                jax.device_put(packed_sample_indices),
                                jax.device_put(packed_valid),
                            )

                    pair_start = 0
                    if pending_packed is not None and len(pair_segments):
                        pair_count = min(
                            PACKED_SEGMENTS - pending_count,
                            len(pair_segments),
                        )
                        right_packed = pack_pairs(pair_start, pair_count)
                        with sharding.set_mesh(mesh):
                            pending_packed = pmerge_packed(
                                pending_packed,
                                right_packed,
                                jnp.int32(pending_count),
                            )
                        pending_count += pair_count
                        pair_start += pair_count
                        if pending_count == PACKED_SEGMENTS:
                            accumulate_packed(pending_packed)
                            pending_packed = None
                            pending_count = 0
                    while len(pair_segments) - pair_start >= PACKED_SEGMENTS:
                        accumulate_packed(
                            pack_pairs(pair_start, PACKED_SEGMENTS))
                        pair_start += PACKED_SEGMENTS
                    if pair_start < len(pair_segments):
                        pending_count = len(pair_segments) - pair_start
                        pending_packed = pack_pairs(
                            pair_start, pending_count)
            if pending_packed is not None:
                accumulate_packed(pending_packed)
            with sharding.set_mesh(mesh):
                return (
                    *papply_gradients(state, gradient_sum, numerator_sum, count_sum),
                    {
                        "segment_calls": packed_calls,
                        "padding_segments_skipped": 0,
                        "context_forward_calls": len(plans),
                    },
                )

    prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    next_batch = prefetch_executor.submit(next, data_iter)

    start_step = int(train_state.step)
    microbatch_by_stage = {
        1: 8,
        4: 16,
        8: 16,
        32: 8,
        64: 8,
        96: 8,
    }
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        next_stage_start, next_temporal_blocks = stage_at(step)
        if next_temporal_blocks != temporal_blocks:
            checkpoint_manager.wait_until_finished()
            jax.block_until_ready(train_state)
            jax.clear_caches()
            gc.collect()
            infos = []
            stage_start, temporal_blocks = next_stage_start, next_temporal_blocks
            data_loader = make_loader(
                seed=config.seed + temporal_blocks, temporal_blocks=temporal_blocks,
                start_step=step - stage_start)
            data_iter = iter(data_loader)
            batch = next(data_iter)
            next_batch.cancel()
            next_batch = prefetch_executor.submit(next, data_iter)
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
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
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

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next_batch.result()
        next_batch = prefetch_executor.submit(next, data_iter)

        checkpoint_step = int(train_state.step)
        if (
            checkpoint_step % config.save_interval == 0 and checkpoint_step > start_step
        ) or step == config.num_train_steps - 1:
            _checkpoints.save_state(
                checkpoint_manager, train_state, data_loader, checkpoint_step, full_state=True
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    prefetch_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main(_config.cli())

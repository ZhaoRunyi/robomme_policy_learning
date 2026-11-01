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
    LaneContext,
    alias_frozen_ema,
    collect_lane,
    make_boundary_transport_jit,
    merge_packed_layer_rows,
    merge_packed_rows,
    pack_context,
    packed_grad,
    padding_plan,
    prepare_lane,
    regroup_lane,
)








@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
    load_weights,
) -> tuple[training_utils.TrainState, Any]:
    settings = config.model.robottt
    base_learning_rate = settings.slow_learning_rate / settings.slow_learning_rate_scale
    path_text = lambda path: jax.tree_util.keystr(path).lower()
    robottt_mask = lambda params: jax.tree_util.tree_map_with_path(
        lambda path, _: "robottt" in path_text(path), params)
    decay_names = ("kernel", "w", "gating_einsum", "linear", "initial_w1", "initial_w2", "initial_weight")
    decay_mask = lambda params: jax.tree_util.tree_map_with_path(
        lambda path, _: any(f"['{name}']" in path_text(path) for name in decay_names), params)
    schedule = optax.join_schedules(
        (
            optax.linear_schedule(base_learning_rate / (settings.slow_warmup_steps + 1), base_learning_rate, settings.slow_warmup_steps),
            optax.constant_schedule(base_learning_rate),
            optax.cosine_decay_schedule(base_learning_rate, settings.slow_decay_steps, alpha=settings.slow_decay_ratio),
        ),
        (settings.slow_warmup_steps, settings.slow_decay_start),
    )
    tx = optax.chain(
        config.optimizer.create(schedule, decay_mask),
        optax.masked(optax.scale(settings.slow_learning_rate_scale), robottt_mask),
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
            config.frozen_cast_filter,
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
    partial_params = load_weights(
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
    info["ttt/slow_grad_norm"] = optax.global_norm(
        grads.filter(nnx.PathContains("robottt")))
    return new_state, info


def apply_robottt_gradients(config, state, grads, numerator, count):
    model = nnx.merge(state.model_def, state.params)
    grads = jax.tree.map(lambda value: value / jnp.maximum(count, 1), grads)
    return apply_gradients(
        config, state, model, grads, numerator / jnp.maximum(count, 1))


def main(config: _config.TrainConfig, *, init_logging, init_wandb, init_history_config, load_weights):
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
    if restore_step is None:
        logging.info("Weight provenance: initialization loader %r", config.weight_loader)
    else:
        restore_root = config.resume_from_dir or config.checkpoint_dir
        logging.info(
            "Weight provenance: selected full-state restore source %s/%d; "
            "config loader %r is unused",
            restore_root,
            restore_step,
            config.weight_loader,
        )
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

    settings = config.model.robottt
    curriculum = settings.schedule(config.num_train_steps)
    packed_segments = settings.packed_segments
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
        register_tokens=settings.register_tokens,
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
        config, init_rng, mesh, resume=restore_step is not None, load_weights=load_weights)
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
    replicated = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())
    boundary_rows = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, sharding.BATCH_AXIS))
    pmerge_rows = jax.jit(
        merge_packed_rows, donate_argnums=(0,), out_shardings=replicated)
    pmerge_layer_rows = jax.jit(
        merge_packed_layer_rows,
        donate_argnums=(0,),
        out_shardings=replicated,
    )
    pmerge_boundary_rows = jax.jit(
        merge_packed_layer_rows,
        donate_argnums=(0,),
        out_shardings=boundary_rows,
    )

    def merge_tree(
        left_tree,
        right_tree,
        left_count,
        merge_leaf,
        *,
        serialize_large=False,
    ):
        left_leaves, tree_def = jax.tree_util.tree_flatten(left_tree)
        right_leaves = jax.tree_util.tree_leaves(right_tree)
        merged_leaves = []
        for left_value, right_value in zip(left_leaves, right_leaves, strict=True):
            merged_value = merge_leaf(left_value, right_value, left_count)
            if (
                serialize_large
                and merged_value.size * merged_value.dtype.itemsize >= 1 << 30
            ):
                merged_value.block_until_ready()
            merged_leaves.append(merged_value)
        return jax.tree_util.tree_unflatten(tree_def, merged_leaves)

    def pmerge_packed(left, right, left_count):
        return LaneContext(
            merge_tree(left.prepared, right.prepared, left_count, pmerge_rows),
            merge_tree(left.masks, right.masks, left_count, pmerge_rows),
            merge_tree(
                left.prefix_kv, right.prefix_kv, left_count, pmerge_layer_rows
            ),
            pmerge_rows(
                left.suffix_positions, right.suffix_positions, left_count),
            pmerge_rows(
                left.suffix_attention_mask,
                right.suffix_attention_mask,
                left_count,
            ),
            merge_tree(
                left.boundary_states,
                right.boundary_states,
                left_count,
                pmerge_boundary_rows,
                serialize_large=True,
            ),
            pmerge_rows(
                left.boundary_positions, right.boundary_positions, left_count),
        )
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
            trainable_ema = None if state.ema_params is None else state.ema_params.filter(config.trainable_filter)
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
                    if config.robottt_training_scope == "ttt_only":
                        observation = context.prepared.observation.replace(
                            images={}, image_masks={},
                            tokenized_prompt=None, tokenized_prompt_mask=None,
                        )
                        context = context._replace(prepared=context.prepared._replace(observation=observation))
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
                        pad = packed_segments - pair_count
                        segment_indices = np.pad(
                            pair_segments[pair_start:pair_stop], (0, pad))
                        packed_sample_indices = np.pad(
                            pair_samples[pair_start:pair_stop], (0, pad))
                        packed_valid = np.arange(packed_segments) < pair_count
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
                            packed_segments - pending_count,
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
                        if pending_count == packed_segments:
                            accumulate_packed(pending_packed)
                            pending_packed = None
                            pending_count = 0
                    while len(pair_segments) - pair_start >= packed_segments:
                        accumulate_packed(
                            pack_pairs(pair_start, packed_segments))
                        pair_start += packed_segments
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

        if settings.training_implementation == "reference":
            def run_robottt_batch(state, host_batch, microbatch_size, segment_length):
                gradient_sum = None
                numerator_sum, count_sum = jnp.float32(0), jnp.int32(0)
                temporal_blocks = host_batch["actions"].shape[1]
                segment_count = temporal_blocks // segment_length
                calls = 0
                for lane_start in range(0, config.batch_size, microbatch_size):
                    lane_batch = jax.tree.map(
                        lambda value: np.asarray(value[lane_start:lane_start + microbatch_size]), host_batch,
                    )
                    masks = lane_batch.pop("robottt")
                    actions = lane_batch.pop("actions")
                    device_batch = jax.device_put(
                        (HistAugObservation.from_dict(lane_batch, normalize_images=False), actions, masks), data_sharding,
                    )
                    lane_rng = jax.random.fold_in(jax.random.fold_in(train_rng, state.step), lane_start // microbatch_size)
                    with sharding.set_mesh(mesh):
                        prepared, device_masks = pprepare_lane(lane_rng, state, device_batch, jnp.arange(segment_count))
                        context = pcollect_lane(state, prepared, device_masks, segment_length)
                    for segment_index in range(segment_count):
                        if not np.any(masks["outer"][:, segment_index * segment_length:(segment_index + 1) * segment_length]):
                            continue
                        with sharding.set_mesh(mesh):
                            packed = ppack_context(
                                context, jnp.full(microbatch_size, segment_index),
                                jnp.arange(microbatch_size), jnp.ones(microbatch_size, bool),
                            )
                            gradient_sum, numerator_sum, count_sum = ppacked_grad(
                                state, packed, gradient_sum, numerator_sum, count_sum,
                            )
                        calls += 1
                with sharding.set_mesh(mesh):
                    return (
                        *papply_gradients(state, gradient_sum, numerator_sum, count_sum),
                        {"segment_calls": calls, "padding_segments_skipped": 0,
                         "context_forward_calls": config.batch_size // microbatch_size},
                    )

    prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    next_batch = prefetch_executor.submit(next, data_iter)

    start_step = int(train_state.step)
    microbatch_by_stage = dict(settings.microbatch_sizes)
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
        segment_length = 1 if temporal_blocks == 1 else settings.tbptt_segment_length
        if step == start_step or step == stage_start:
            logging.info(f"RoboTTT T{temporal_blocks}: microbatch {microbatch_by_stage[temporal_blocks]}")
        update_start = time.perf_counter()
        train_state, info, stats = run_robottt_batch(
            train_state, batch, microbatch_by_stage[temporal_blocks], segment_length)
        jax.block_until_ready(train_state)
        update_seconds = time.perf_counter() - update_start
        info.update({
            "perf/update_seconds": update_seconds,
            "perf/valid_blocks_per_second": (
                float(batch["robottt"]["valid"].sum()) / update_seconds
            ),
            "perf/segment_calls": stats["segment_calls"],
            "perf/padding_segments_skipped": stats["padding_segments_skipped"],
            "perf/context_forward_calls": stats["context_forward_calls"],
        })
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            robottt_params = train_state.params["PaliGemma"]["llm"]["layers"]["robottt"]
            inner_learning_rate = settings.fast_learning_rate * jax.nn.sigmoid(robottt_params["inner_learning_rate_raw"].value)
            gate = jnp.abs(jnp.tanh(robottt_params["residual_gate"].value))
            reduced_info.update({
                "ttt/inner_lr_mean": float(jnp.mean(inner_learning_rate)),
                "ttt/inner_lr_min": float(jnp.min(inner_learning_rate)),
                "ttt/inner_lr_max": float(jnp.max(inner_learning_rate)),
                "ttt/gate_abs_mean": float(jnp.mean(gate)),
                "ttt/gate_abs_max": float(jnp.max(gate)),
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



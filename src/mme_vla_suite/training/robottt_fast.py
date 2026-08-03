"""Frozen π₀.₅ RoboTTT training with padding-aware packed S4 VJPs."""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.models.integration.history_observation import preprocess_observation
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
from mme_vla_suite.models.representation.robottt import create_fast_state

PACKED_SEGMENTS = 32


class PreparedSequence(NamedTuple):
    observation: Any
    noisy_actions: jax.Array
    time: jax.Array
    target_velocity: jax.Array


class LaneContext(NamedTuple):
    prepared: PreparedSequence
    masks: Any
    prefix_kv: Any
    suffix_positions: jax.Array
    suffix_attention_mask: jax.Array
    boundary_states: Any
    boundary_positions: jax.Array


def _flatten_sequence(tree, batch_size, temporal_blocks):
    return jax.tree.map(
        lambda value: value.reshape(
            (batch_size * temporal_blocks, *value.shape[2:])
        ),
        tree,
    )


def prepare_lane(config, rng, train_state, batch, segment_indices):
    """Draw stochastic inputs before padding-aware sample regrouping."""
    del config
    model = nnx.merge(train_state.model_def, train_state.params)
    observation, actions, masks = batch
    batch_size, temporal_blocks = actions.shape[:2]
    segment_length = temporal_blocks // segment_indices.shape[0]
    observation = _flatten_sequence(
        observation, batch_size, temporal_blocks
    )
    observation = preprocess_observation(
        jax.random.fold_in(rng, 0),
        observation,
        train=True,
        augmentation_group_size=temporal_blocks,
    )
    observation = jax.tree.map(
        lambda value: value.reshape(
            (batch_size, temporal_blocks, *value.shape[1:])
        ),
        observation,
    )

    block_keys = [
        jax.random.fold_in(
            jax.random.fold_in(
                rng,
                segment_indices[block_index // segment_length] + 1,
            ),
            block_index % segment_length,
        )
        for block_index in range(temporal_blocks)
    ]
    noise_keys, time_keys = jax.vmap(jax.random.split)(
        jnp.stack(block_keys)
    ).swapaxes(0, 1)
    noise_shape = actions.shape[:1] + actions.shape[2:]
    noise = jax.vmap(
        lambda key: jax.random.normal(key, noise_shape)
    )(noise_keys).swapaxes(0, 1)
    time = jax.vmap(
        lambda key: jax.random.beta(key, 1.5, 1, (batch_size,))
    )(time_keys).swapaxes(0, 1)
    time = time * 0.999 + 0.001
    execution = jnp.any(
        masks["inner"][..., : model.action_horizon], axis=-1
    )
    time = jnp.where(execution, time, 0.1)
    noisy_actions = (
        time[..., None, None] * noise
        + (1 - time[..., None, None]) * actions
    )
    noisy_actions = jnp.where(
        execution[..., None, None], noisy_actions, 0
    )
    return PreparedSequence(
        observation, noisy_actions, time, noise - actions
    ), masks


def regroup_lane(
    prepared_lanes,
    mask_lanes,
    sample_indices,
    temporal_blocks,
    microbatch_size,
):
    """Regroup prepared samples and omit complete trailing padding."""
    source_lanes = sample_indices // microbatch_size
    source_samples = sample_indices % microbatch_size

    def gather(value):
        return value[source_lanes, source_samples, :temporal_blocks]

    return jax.tree.map(gather, prepared_lanes), jax.tree.map(
        gather, mask_lanes
    )


def collect_lane(config, train_state, prepared, masks, segment_length):
    """Compute frozen prefix KV and chronological S-boundary fast states."""
    del config
    model = nnx.merge(train_state.model_def, train_state.params)
    model.train()
    batch_size, temporal_blocks = prepared.time.shape
    flat_observation = _flatten_sequence(
        prepared.observation, batch_size, temporal_blocks
    )
    prefix_tokens, prefix_mask, prefix_ar, prefix_na, _ = (
        model.embed_prefix(flat_observation)
    )
    flat_actions = prepared.noisy_actions.reshape(
        (
            batch_size * temporal_blocks,
            *prepared.noisy_actions.shape[2:],
        )
    )
    suffix = model.embed_suffix(
        flat_observation,
        flat_actions,
        prepared.time.reshape((batch_size * temporal_blocks,)),
    )
    suffix_tokens, suffix_mask, suffix_ar, suffix_na, adarms_cond = suffix
    prefix_result = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=make_attn_mask(prefix_mask, prefix_ar, prefix_na),
        positions=jnp.cumsum(prefix_mask, axis=1) - 1,
        deterministic=model.deterministic,
    )
    prefix_kv = jax.tree.map(
        lambda value: jax.lax.stop_gradient(
            value.reshape(
                (
                    value.shape[0],
                    batch_size,
                    temporal_blocks,
                    *value.shape[2:],
                )
            )
        ),
        prefix_result[1],
    )

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attention_mask = make_attn_mask(
        input_mask,
        jnp.concatenate([prefix_ar, suffix_ar]),
        jnp.concatenate([prefix_na, suffix_na]),
    )
    positions = jnp.cumsum(input_mask, axis=1) - 1
    prefix_length = prefix_tokens.shape[1]
    suffix_tokens = suffix_tokens.reshape(
        (batch_size, temporal_blocks, *suffix_tokens.shape[1:])
    )
    suffix_positions = positions[:, prefix_length:].reshape(
        (batch_size, temporal_blocks, -1)
    )
    suffix_attention_mask = attention_mask[:, prefix_length:].reshape(
        (
            batch_size,
            temporal_blocks,
            suffix_tokens.shape[2],
            attention_mask.shape[-1],
        )
    )
    adarms_cond = adarms_cond.reshape(
        (batch_size, temporal_blocks, -1)
    )
    valid = masks["valid"].astype(jnp.int32)
    block_positions = jnp.cumsum(valid, axis=1) - valid
    result = model.PaliGemma.llm(
        [None, suffix_tokens],
        kv_cache=prefix_kv,
        positions=suffix_positions,
        mask=suffix_attention_mask,
        adarms_cond=[None, adarms_cond],
        robottt=(
            create_fast_state(batch_size),
            masks["inner"],
            block_positions,
            segment_length,
        ),
        deterministic=model.deterministic,
    )
    boundary_positions = block_positions[:, ::segment_length].swapaxes(
        0, 1
    )
    return LaneContext(
        prepared,
        masks,
        prefix_kv,
        suffix_positions,
        suffix_attention_mask,
        result[3],
        boundary_positions,
    )


def pack_context(
    context, segment_indices, sample_indices, packed_valid
):
    """Gather one fixed-shape suffix-only segment batch."""
    batch_size = context.prepared.time.shape[0]
    segment_count = context.boundary_positions.shape[0]
    segment_length = context.prepared.time.shape[1] // segment_count

    def gather(value):
        value = value.reshape(
            (
                batch_size,
                segment_count,
                segment_length,
                *value.shape[2:],
            )
        )
        return value[sample_indices, segment_indices]

    def gather_layer(value):
        value = value.reshape(
            (
                value.shape[0],
                batch_size,
                segment_count,
                segment_length,
                *value.shape[3:],
            )
        )
        return value[:, sample_indices, segment_indices]

    masks = jax.tree.map(gather, context.masks)
    masks = jax.tree.map(
        lambda value: value
        & packed_valid.reshape(
            (packed_valid.shape[0], *([1] * (value.ndim - 1)))
        ),
        masks,
    )
    return LaneContext(
        jax.tree.map(gather, context.prepared),
        masks,
        jax.tree.map(gather_layer, context.prefix_kv),
        gather(context.suffix_positions),
        gather(context.suffix_attention_mask),
        jax.tree.map(
            lambda value: value[:, segment_indices, sample_indices],
            context.boundary_states,
        ),
        context.boundary_positions[segment_indices, sample_indices],
    )


def merge_packed(left, right, left_count):
    """Fill one P32 batch with valid rows from two lane-local packs."""
    output_rows = jnp.arange(PACKED_SEGMENTS)
    use_right = output_rows >= left_count
    source = use_right.astype(jnp.int32)
    rows = jnp.where(use_right, output_rows - left_count, output_rows)

    def gather(left_value, right_value):
        values = jnp.stack((left_value, right_value))
        return values[source, rows]

    def gather_layer(left_value, right_value):
        values = jnp.stack((left_value, right_value))
        return jnp.swapaxes(values[source, :, rows], 0, 1)

    return LaneContext(
        jax.tree.map(gather, left.prepared, right.prepared),
        jax.tree.map(gather, left.masks, right.masks),
        jax.tree.map(gather_layer, left.prefix_kv, right.prefix_kv),
        gather(left.suffix_positions, right.suffix_positions),
        gather(
            left.suffix_attention_mask,
            right.suffix_attention_mask,
        ),
        jax.tree.map(
            gather_layer, left.boundary_states, right.boundary_states
        ),
        gather(left.boundary_positions, right.boundary_positions),
    )


def packed_grad(config, train_state, packed):
    """Differentiate one fixed-shape suffix-only segment batch."""
    model = nnx.merge(train_state.model_def, train_state.params)
    model.train()

    def loss_fn(model):
        prepared = packed.prepared
        batch_size, segment_length = prepared.time.shape
        observation = _flatten_sequence(
            prepared.observation, batch_size, segment_length
        )
        noisy_actions = prepared.noisy_actions.reshape(
            (
                batch_size * segment_length,
                *prepared.noisy_actions.shape[2:],
            )
        )
        suffix = model.embed_suffix(
            observation,
            noisy_actions,
            prepared.time.reshape((batch_size * segment_length,)),
        )
        suffix_tokens, _, _, _, adarms_cond = suffix
        suffix_tokens = suffix_tokens.reshape(
            (batch_size, segment_length, *suffix_tokens.shape[1:])
        )
        adarms_cond = adarms_cond.reshape(
            (batch_size, segment_length, -1)
        )
        valid = packed.masks["valid"].astype(jnp.int32)
        block_positions = (
            packed.boundary_positions[:, None]
            + jnp.cumsum(valid, axis=1)
            - valid
        )
        result = model.PaliGemma.llm(
            [None, suffix_tokens],
            kv_cache=packed.prefix_kv,
            positions=packed.suffix_positions,
            mask=packed.suffix_attention_mask,
            adarms_cond=[None, adarms_cond],
            robottt=(
                packed.boundary_states,
                packed.masks["inner"],
                block_positions,
                segment_length,
            ),
            deterministic=model.deterministic,
        )
        action_out = result[0][-1][
            :, :, : model.action_horizon
        ]
        velocity = model.action_out_proj(action_out)
        loss = jnp.mean(
            jnp.square(velocity - prepared.target_velocity), axis=-1
        )
        numerator = jnp.sum(
            loss * packed.masks["outer"][..., None]
        )
        count = (
            jnp.sum(packed.masks["outer"]).astype(jnp.int32)
            * model.action_horizon
        )
        return numerator, count

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (numerator, count), gradients = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True
    )(model)
    return gradients, numerator, count


def padding_plan(valid, microbatch_size, lane_group_size, segment_length):
    """Return stable group-local sample order and retained S multiples."""
    valid = np.asarray(valid, dtype=bool)
    plans = []
    group_size = microbatch_size * lane_group_size
    for group_start in range(0, valid.shape[0], group_size):
        group_stop = min(group_start + group_size, valid.shape[0])
        lengths = valid[group_start:group_stop].sum(axis=1)
        order = np.argsort(lengths, kind="stable")
        for lane_start in range(0, len(order), microbatch_size):
            sample_indices = order[
                lane_start : lane_start + microbatch_size
            ]
            max_blocks = int(lengths[sample_indices].max(initial=0))
            temporal_blocks = segment_length
            while temporal_blocks < max_blocks:
                temporal_blocks = min(
                    temporal_blocks * 2, valid.shape[1]
                )
            plans.append(
                (group_start, group_stop, sample_indices, temporal_blocks)
            )
    return plans


def alias_frozen_ema(config, train_state):
    """Alias frozen EMA leaves to frozen parameter buffers."""
    trainable_ema, _ = train_state.ema_params.split(
        config.trainable_filter, ...
    )
    _, frozen_params = train_state.params.split(
        config.trainable_filter, ...
    )
    return dataclasses.replace(
        train_state,
        ema_params=nnx.State.merge(frozen_params, trainable_ema),
    )


def strip_frozen_ema(config, train_state):
    """Keep only trainable EMA leaves inside the optimizer JIT."""
    trainable_ema, _ = train_state.ema_params.split(
        config.trainable_filter, ...
    )
    return dataclasses.replace(train_state, ema_params=trainable_ema)

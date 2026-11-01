"""Shared RoboTTT fast learner and explicit model/training design settings."""

import dataclasses
import json
import math
import pathlib
from typing import Literal
import warnings

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models.gemma import _apply_rope
import openpi.training.sharding as sharding

TrainingScope = Literal["ttt_only", "non_ttt_only", "all"]


@dataclasses.dataclass(frozen=True)
class RoboTTTConfig:
    # Training scope compatibility (the upper-level scope remains authoritative).
    frozen: bool | None = None
    freeze_ttt: bool | None = None
    strict: bool = True

    # Fast learner architecture.
    fast_model: Literal["mlp", "linear"] = "mlp"
    fast_width: int = 2048
    fast_update: Literal["all", "last_layer"] = "all"
    fast_bias: bool = True
    fast_output_bias: bool | None = None
    register_tokens: int = 16

    # Inner update and read/write semantics.
    operation_order: Literal["write_then_read", "read_then_write"] = "write_then_read"
    fast_loss: Literal["mse", "negative_dot"] = "mse"
    fast_learning_rate: float = 0.1
    initial_gate: float = 0.001

    # Feature, target and weight normalization.
    qk_normalization: Literal["l2", "silu_l2", "none"] = "l2"
    target_normalization: Literal["none", "v_minus_k_layernorm"] = "none"
    weight_normalization: Literal["none", "unit_input_norm"] = "none"
    normalization_epsilon: float = 1e-5
    target_normalization_epsilon: float = 1e-8

    # Outer optimizer schedule.
    slow_learning_rate: float = 5e-5
    slow_learning_rate_scale: float = 5.0
    slow_warmup_steps: int = 10_000
    slow_decay_start: int = 80_000
    slow_decay_steps: int = 20_000
    slow_decay_ratio: float = 0.1

    # Temporal curriculum and batching.
    curriculum: dict[int, int] = dataclasses.field(
        default_factory=lambda: {1: 50_000, 4: 10_000, 8: 10_000, 32: 10_000, 64: 10_000, 96: 10_000},
        hash=False,
    )
    tbptt_segment_length: int = 4
    microbatch_sizes: tuple[tuple[int, int], ...] = ((1, 8), (4, 16), (8, 16), (16, 8), (32, 8), (64, 8), (96, 8))

    # Execution strategy (does not select a different mathematical objective).
    training_implementation: Literal["packed", "reference"] = "packed"
    packed_segments: int = 24
    packed_loss_remat: bool = False
    packed_gradient_barrier: bool = True
    prefix_batching: bool = False

    # Episode state lifecycle.
    reset_rng_on_episode: bool = True

    def __post_init__(self):
        choices = {
            "fast_model": ("mlp", "linear"),
            "fast_update": ("all", "last_layer"),
            "operation_order": ("write_then_read", "read_then_write"),
            "fast_loss": ("mse", "negative_dot"),
            "qk_normalization": ("l2", "silu_l2", "none"),
            "target_normalization": ("none", "v_minus_k_layernorm"),
            "weight_normalization": ("none", "unit_input_norm"),
            "training_implementation": ("packed", "reference"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        for name in ("fast_width", "register_tokens", "tbptt_segment_length", "slow_warmup_steps", "slow_decay_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "fast_learning_rate",
            "slow_learning_rate",
            "slow_learning_rate_scale",
            "normalization_epsilon",
            "target_normalization_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.initial_gate) or not 0 <= self.slow_decay_ratio <= 1:
            raise ValueError("initial_gate must be finite and slow_decay_ratio must lie in [0, 1]")
        if not self.strict:
            raise ValueError("Approximate/stop-gradient outer training is not a standard RoboTTT path")
        if self.packed_segments not in (8, 16, 24, 32):
            raise ValueError("packed_segments must be one of 8, 16, 24, or 32")
        if not isinstance(self.curriculum, dict) or not self.curriculum:
            raise ValueError("curriculum must be a nonempty {temporal_blocks: training_steps} dictionary")
        for blocks, steps in self.curriculum.items():
            if not isinstance(blocks, int) or not isinstance(steps, int) or blocks <= 0 or steps <= 0:
                raise ValueError("Curriculum blocks and durations must be positive integers")
            if blocks != 1 and blocks % self.tbptt_segment_length:
                raise ValueError(f"T{blocks} must be divisible by tbptt_segment_length")
            if blocks not in dict(self.microbatch_sizes):
                raise ValueError(f"T{blocks} has no configured microbatch size")

    @property
    def output_bias(self):
        return self.fast_bias if self.fast_output_bias is None else self.fast_output_bias

    def schedule(self, total_steps: int | None = None):
        if total_steps is not None and total_steps <= 0:
            raise ValueError("num_train_steps must be positive")
        if total_steps is not None and sum(self.curriculum.values()) != total_steps:
            warnings.warn(
                f"Curriculum durations sum to {sum(self.curriculum.values())}, not num_train_steps={total_steps}; "
                "the final T continues to the training endpoint; overflowing earlier stages are truncated "
                "to leave at least one update for the final T.",
                stacklevel=2,
            )
        start = 0
        stages = []
        entries = tuple(self.curriculum.items())
        for blocks, duration in entries[:-1]:
            if total_steps is not None and start >= total_steps - 1:
                break
            stages.append((start, blocks))
            start += duration
            if total_steps is not None:
                start = min(start, total_steps - 1)
        stages.append((start, entries[-1][0]))
        return tuple(stages)

    def stage_at(self, step: int, total_steps: int | None = None):
        return next(stage for stage in reversed(self.schedule(total_steps)) if step >= stage[0])

    def blocks_at(self, step: int, total_steps: int | None = None):
        return self.stage_at(step, total_steps)[1]

    def microbatch_at(self, temporal_blocks: int):
        return dict(self.microbatch_sizes)[temporal_blocks]

    def create_state(self, depth: int, batch_size: int, width: int):
        return create_fast_state(self, depth, batch_size, width)


def resolve_training_scope(scope: TrainingScope | None, config: RoboTTTConfig, legacy_ttt_only: bool | None = None):
    if scope is None:
        if legacy_ttt_only is not None:
            scope = "ttt_only" if legacy_ttt_only else "all"
        elif config.freeze_ttt:
            scope = "non_ttt_only"
        elif config.frozen is False:
            scope = "all"
        else:
            scope = "ttt_only"
    if scope not in ("ttt_only", "non_ttt_only", "all"):
        raise ValueError(f"Invalid RoboTTT training scope: {scope}")
    expected = {"frozen": scope == "ttt_only", "freeze_ttt": scope == "non_ttt_only"}
    for name, value in expected.items():
        explicit = getattr(config, name)
        if explicit is not None and explicit != value:
            raise ValueError(f"robottt_training_scope={scope} conflicts with robottt.{name}={explicit}")
    if legacy_ttt_only is not None and legacy_ttt_only != (scope == "ttt_only"):
        raise ValueError(f"robottt_training_scope={scope} conflicts with train_robottt_only={legacy_ttt_only}")
    return scope


def config_from_candidate(config: RoboTTTConfig, candidate: dict):
    """Restore declared design settings, including the existing experiment manifests."""
    if candidate.get("video_action_source", "ground_truth") != "ground_truth":
        raise ValueError("This standard implementation supports only ground_truth video actions")
    settings = dict(candidate.get("robottt_config", {}))
    legacy_modes = {
        "full_two_layer_mse": {},
        "final_layer_w2_b2": {"fast_update": "last_layer"},
        "final_layer_w2": {"fast_update": "last_layer", "fast_output_bias": False},
        "single_linear_w_b": {"fast_model": "linear"},
        "single_linear_w": {"fast_model": "linear", "fast_bias": False},
        "full_two_layer_mse_read_then_write": {"operation_order": "read_then_write"},
        "full_two_layer_frobenius_dot": {"fast_loss": "negative_dot"},
        "full_two_layer_frobenius_dot_lactnorm": {
            "fast_loss": "negative_dot",
            "qk_normalization": "silu_l2",
            "weight_normalization": "unit_input_norm",
        },
        "full_two_layer_mse_vminus_k_layernorm": {"target_normalization": "v_minus_k_layernorm"},
    }
    if not settings:
        mode = candidate.get("fast_update_mode", "full_two_layer_mse")
        if mode not in legacy_modes:
            raise ValueError(f"Unsupported candidate fast_update_mode: {mode}")
        settings.update(legacy_modes[mode])
        for old_name, new_name in (
            ("inner_lr_scale", "fast_learning_rate"),
            ("initial_gate", "initial_gate"),
            ("register_tokens", "register_tokens"),
        ):
            if old_name in candidate:
                settings[new_name] = candidate[old_name]
    if "curriculum" in settings:
        settings["curriculum"] = {int(blocks): int(steps) for blocks, steps in settings["curriculum"].items()}
    if "microbatch_sizes" in settings:
        settings["microbatch_sizes"] = tuple(tuple(pair) for pair in settings["microbatch_sizes"])
    return dataclasses.replace(config, **settings)


def read_candidate(checkpoint_dir: pathlib.Path):
    if not any((root / "history_config.txt").is_file() for root in (checkpoint_dir.parent, checkpoint_dir)):
        raise FileNotFoundError("RoboTTT checkpoint requires history_config.txt beside its numbered checkpoints")
    for root in (checkpoint_dir.parent, checkpoint_dir):
        path = root / "candidate.json"
        if path.is_file():
            candidate = json.loads(path.read_text())
            if not isinstance(candidate, dict):
                raise ValueError("candidate.json must contain a JSON object")
            return candidate
    raise FileNotFoundError("RoboTTT checkpoint requires candidate.json beside its numbered checkpoints")


def write_candidate(
    run_root: pathlib.Path, settings: RoboTTTConfig, scope: TrainingScope, history_config: str | None = None
):
    """Keep one self-describing configuration beside all full-state checkpoints."""
    path = run_root / "candidate.json"
    existing = json.loads(path.read_text()) if path.is_file() else {}
    if existing:
        restored = config_from_candidate(settings, existing)
        if restored != settings or existing.get("training_scope", scope) != scope:
            raise ValueError("Existing candidate.json conflicts with this training configuration")
    history_path = run_root / "history_config.txt"
    if history_config is not None and history_path.is_file() and history_path.read_text().strip() != history_config:
        raise ValueError("Existing history_config.txt conflicts with this training configuration")
    candidate = {
        **existing,
        "robottt_config": dataclasses.asdict(settings),
        "training_scope": scope,
        "video_action_source": "ground_truth",
        "inner_lr_scale": settings.fast_learning_rate,
        "register_tokens": settings.register_tokens,
        "initial_gate": settings.initial_gate,
    }
    path.write_text(json.dumps(candidate, indent=2) + "\n")
    if history_config is not None:
        history_path.write_text(history_config + "\n")


@struct.dataclass
class FastState:
    w1: jax.Array | None
    b1: jax.Array | None
    w2: jax.Array
    b2: jax.Array | None
    initialized: jax.Array


@struct.dataclass
class LinearFastState:
    weight: jax.Array
    bias: jax.Array | None
    initialized: jax.Array


def create_fast_state(config: RoboTTTConfig, depth: int, batch_size: int, width: int):
    layer_batch = (depth, batch_size)
    initialized = jnp.zeros(layer_batch, dtype=jnp.bool_)
    if config.fast_model == "linear":
        return LinearFastState(
            jnp.zeros((*layer_batch, width, width)),
            jnp.zeros((*layer_batch, width)) if config.output_bias else None,
            initialized,
        )
    return FastState(
        jnp.zeros((*layer_batch, width, config.fast_width)) if config.fast_update == "all" else None,
        jnp.zeros((*layer_batch, config.fast_width)) if config.fast_update == "all" and config.fast_bias else None,
        jnp.zeros((*layer_batch, config.fast_width, width)),
        jnp.zeros((*layer_batch, width)) if config.output_bias else None,
        initialized,
    )


def _linear(inputs, weight, bias):
    output = jnp.einsum("bnd,bdh->bnh", inputs, weight)
    return output if bias is None else output + bias[:, None]


def _select(valid, candidate, previous):
    if previous is None:
        return None
    return jnp.where(valid.reshape((*valid.shape, *([1] * (previous.ndim - valid.ndim)))), candidate, previous)


class Layer(nn.Module):
    config: RoboTTTConfig
    width: int
    heads: int = 4

    def apply_context(self, hidden, context):
        """Apply the same fast learner to one block or a segmented sequence."""
        state = context.fast_state
        if context.inner_mask.ndim != 3:
            hidden, state = self(hidden, state, context.inner_mask, context.block_position)
            return hidden, state, None

        batch_size, temporal_blocks = hidden.shape[:2]
        segment_length = context.segment_length
        segment_count = temporal_blocks // segment_length
        scan_inputs = jax.tree.map(
            lambda value: jnp.swapaxes(
                value.reshape((batch_size, segment_count, segment_length, *value.shape[2:])), 0, 1
            ),
            (hidden, context.inner_mask, context.block_position),
        )

        def update_segment(layer, current_state, inputs):
            hidden_slots, mask_slots, position_slots = inputs
            current_state = jax.tree.map(jax.lax.stop_gradient, current_state)
            boundary = current_state
            outputs = []
            for slot_index in range(segment_length):
                hidden, current_state = layer(
                    hidden_slots[:, slot_index], current_state,
                    mask_slots[:, slot_index], position_slots[:, slot_index],
                )
                outputs.append(hidden)
            current_state = jax.tree.map(jax.lax.stop_gradient, current_state)
            return current_state, (jnp.stack(outputs, axis=1), boundary)

        state, (hidden, boundaries) = nn.scan(
            update_segment,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=segment_count,
        )(self, state, scan_inputs)
        hidden = jnp.swapaxes(hidden, 0, 1).reshape((batch_size, temporal_blocks, *hidden.shape[3:]))
        return hidden, state, boundaries

    @nn.compact
    def __call__(self, hidden, state, inner_mask, block_position):
        normal = nn.initializers.normal(0.02)
        zeros = nn.initializers.zeros_init()
        initialized = state.initialized

        def parameter(name, shape, previous, initializer=normal):
            initial = self.param(name, initializer, shape, jnp.float32)
            if previous is None:
                return jnp.broadcast_to(initial[None], (hidden.shape[0], *shape))
            return _select(initialized, previous, jnp.broadcast_to(initial[None], previous.shape))

        if self.config.fast_model == "linear":
            weight1 = parameter("initial_weight", (self.width, self.width), state.weight)
            bias1 = parameter("initial_bias", (self.width,), state.bias, zeros) if self.config.output_bias else None
            weight2 = bias2 = None
        else:
            weight1 = parameter("initial_w1", (self.width, self.config.fast_width), state.w1)
            bias1 = (
                parameter("initial_b1", (self.config.fast_width,), state.b1, zeros) if self.config.fast_bias else None
            )
            weight2 = parameter("initial_w2", (self.config.fast_width, self.width), state.w2)
            bias2 = parameter("initial_b2", (self.width,), state.b2, zeros) if self.config.output_bias else None

        dense = dict(
            features=self.width, dtype=jnp.float32, param_dtype=jnp.float32, kernel_init=normal, bias_init=zeros
        )
        hidden_fp32 = hidden.astype(jnp.float32)
        query = nn.Dense(name="query", **dense)(hidden_fp32)
        key = nn.Dense(name="key", **dense)(hidden_fp32)
        value = nn.Dense(name="value", **dense)(hidden_fp32)
        query, key, value = sharding.activation_sharding_constraint((query, key, value))
        positions = block_position[:, None] * hidden.shape[1] + jnp.arange(hidden.shape[1])[None]
        rope_shape = (*hidden.shape[:2], self.heads, self.width // self.heads)
        query, key = query.reshape(rope_shape), key.reshape(rope_shape)
        if self.config.qk_normalization == "silu_l2":
            query, key = jax.nn.silu(query), jax.nn.silu(key)
        if self.config.qk_normalization != "none":
            query = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + self.config.normalization_epsilon)
            key = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + self.config.normalization_epsilon)
        query = _apply_rope(query, positions=positions, max_wavelength=10_000.0).reshape(hidden.shape)
        key = _apply_rope(key, positions=positions, max_wavelength=10_000.0).reshape(hidden.shape)
        if self.config.target_normalization == "v_minus_k_layernorm":
            scale = self.param("target_norm_scale", nn.initializers.ones_init(), rope_shape[-2:], jnp.float32)
            bias = self.param("target_norm_bias", zeros, rope_shape[-2:], jnp.float32)
            delta = (value - key).reshape(rope_shape)
            normalized = (delta - jnp.mean(delta, axis=-1, keepdims=True)) / (
                jnp.std(delta, axis=-1, correction=1, keepdims=True) + self.config.target_normalization_epsilon
            )
            value = key + (scale[None, None] * normalized + bias[None, None]).reshape(hidden.shape)

        preactivation = _linear(key, weight1, bias1)
        if self.config.fast_model == "linear":
            prediction = preactivation
        else:
            gelu_scale = jnp.sqrt(jnp.asarray(2.0 / jnp.pi, preactivation.dtype))
            gelu_tanh = jnp.tanh(gelu_scale * (preactivation + 0.044715 * preactivation**3))
            activation = preactivation * 0.5 * (1 + gelu_tanh)
            prediction = _linear(activation, weight2, bias2)
        mask = inner_mask.astype(jnp.float32)
        prediction_gradient = 2 * (prediction - value) if self.config.fast_loss == "mse" else -value
        prediction_gradient *= mask[:, :, None]
        prediction_gradient /= jnp.maximum(jnp.sum(mask, axis=-1) * self.width, 1)[:, None, None]

        learning_rate_raw = self.param("inner_learning_rate_raw", zeros, (), jnp.float32)
        learning_rate = self.config.fast_learning_rate * jax.nn.sigmoid(learning_rate_raw)

        def update(weight, gradient, *, matrix=False):
            if weight is None:
                return None
            candidate = weight - learning_rate * gradient
            if matrix and self.config.weight_normalization == "unit_input_norm":
                candidate /= jnp.linalg.norm(candidate, axis=-2, keepdims=True) + self.config.normalization_epsilon
            return candidate

        valid = jnp.any(inner_mask, axis=-1)
        if self.config.fast_model == "linear":
            candidate_weight1 = update(weight1, jnp.einsum("bnd,bnh->bdh", key, prediction_gradient), matrix=True)
            candidate_bias1 = update(bias1, jnp.sum(prediction_gradient, axis=1))
            next_state = LinearFastState(
                _select(valid, candidate_weight1, state.weight),
                _select(valid, candidate_bias1, state.bias),
                jnp.logical_or(initialized, valid),
            )
            read_weight1, read_bias1 = next_state.weight, next_state.bias
            read_weight2 = read_bias2 = None
        else:
            candidate_weight2 = update(
                weight2, jnp.einsum("bnh,bnd->bhd", activation, prediction_gradient), matrix=True
            )
            candidate_bias2 = update(bias2, jnp.sum(prediction_gradient, axis=1))
            candidate_weight1, candidate_bias1 = weight1, bias1
            if self.config.fast_update == "all":
                activation_gradient = jnp.einsum("bnd,bhd->bnh", prediction_gradient, weight2)
                gelu_gradient = 0.5 * (1 + gelu_tanh) + (
                    0.5 * preactivation * (1 - gelu_tanh**2) * gelu_scale * (1 + 3 * 0.044715 * preactivation**2)
                )
                preactivation_gradient = activation_gradient * gelu_gradient
                candidate_weight1 = update(
                    weight1, jnp.einsum("bnd,bnh->bdh", key, preactivation_gradient), matrix=True
                )
                candidate_bias1 = update(bias1, jnp.sum(preactivation_gradient, axis=1))
            next_state = FastState(
                _select(valid, candidate_weight1, state.w1),
                _select(valid, candidate_bias1, state.b1),
                _select(valid, candidate_weight2, state.w2),
                _select(valid, candidate_bias2, state.b2),
                jnp.logical_or(initialized, valid),
            )
            read_weight1 = candidate_weight1 if state.w1 is None else next_state.w1
            read_bias1 = candidate_bias1 if state.b1 is None else next_state.b1
            read_weight2, read_bias2 = next_state.w2, next_state.b2

        if self.config.operation_order == "read_then_write":
            read_weight1, read_bias1, read_weight2, read_bias2 = weight1, bias1, weight2, bias2
        output = _linear(query, read_weight1, read_bias1)
        if self.config.fast_model == "mlp":
            output = _linear(jax.nn.gelu(output), read_weight2, read_bias2)
        output = nn.Dense(name="output", **dense)(output)
        gate = self.param(
            "residual_gate", nn.initializers.constant(self.config.initial_gate), (self.width,), jnp.float32
        )
        output = jnp.where(valid[:, None, None], jnp.tanh(gate) * output, 0)
        return hidden + output.astype(hidden.dtype), next_state

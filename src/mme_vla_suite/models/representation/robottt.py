import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import os

from openpi.models.gemma import _apply_rope
import openpi.training.sharding as sharding
DEPTH, WIDTH, FAST_WIDTH, HEADS = 18, 1024, 2048, 4
@struct.dataclass
class RoboTTTFastState:
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    initialized: jax.Array
def create_fast_state(batch_size):
    layer_batch = (DEPTH, batch_size)
    return RoboTTTFastState(
        jnp.zeros((*layer_batch, WIDTH, FAST_WIDTH)),
        jnp.zeros((*layer_batch, FAST_WIDTH)),
        jnp.zeros((*layer_batch, FAST_WIDTH, WIDTH)),
        jnp.zeros((*layer_batch, WIDTH)),
        jnp.zeros(layer_batch, dtype=jnp.bool_),
    )
def fast_mlp(inputs, weight1, bias1, weight2, bias2):
    hidden = jnp.einsum("bnd,bdh->bnh", inputs, weight1) + bias1[:, None]
    return jnp.einsum("bnh,bhd->bnd", jax.nn.gelu(hidden), weight2) + bias2[:, None]
class RoboTTTLayer(nn.Module):
    @nn.compact
    def __call__(self, hidden, state, inner_mask, block_position):
        normal = nn.initializers.normal(0.02)
        zeros = nn.initializers.zeros_init()
        initial_w1 = self.param("initial_w1", normal, (WIDTH, FAST_WIDTH), jnp.float32)
        initial_b1 = self.param("initial_b1", zeros, (FAST_WIDTH,), jnp.float32)
        initial_w2 = self.param("initial_w2", normal, (FAST_WIDTH, WIDTH), jnp.float32)
        initial_b2 = self.param("initial_b2", zeros, (WIDTH,), jnp.float32)
        initialized = state.initialized
        weight1 = jnp.where(initialized[:, None, None], state.w1, initial_w1[None])
        bias1 = jnp.where(initialized[:, None], state.b1, initial_b1[None])
        weight2 = jnp.where(initialized[:, None, None], state.w2, initial_w2[None])
        bias2 = jnp.where(initialized[:, None], state.b2, initial_b2[None])
        dense = dict(features=WIDTH, dtype=jnp.float32, param_dtype=jnp.float32, kernel_init=normal, bias_init=zeros)
        hidden_fp32 = hidden.astype(jnp.float32)
        query = nn.Dense(name="query", **dense)(hidden_fp32)
        key = nn.Dense(name="key", **dense)(hidden_fp32)
        value = nn.Dense(name="value", **dense)(hidden_fp32)
        if os.environ.get("ROBOTTT_QKV_CONSTRAINT") == "1":
            query, key, value = sharding.activation_sharding_constraint((query, key, value))
        positions = block_position[:, None] * hidden.shape[1] + jnp.arange(hidden.shape[1])[None]
        rope_shape = (*hidden.shape[:2], HEADS, WIDTH // HEADS)
        query = query.reshape(rope_shape)
        key = key.reshape(rope_shape)
        query = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-5)
        key = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + 1e-5)
        query = _apply_rope(
            query, positions=positions, max_wavelength=10_000.0).reshape(hidden.shape)
        key = _apply_rope(
            key, positions=positions, max_wavelength=10_000.0).reshape(hidden.shape)
        def inner_loss(w1, b1, w2, b2, sample_key, sample_value, mask):
            prediction = fast_mlp(
                sample_key[None], w1[None], b1[None], w2[None], b2[None])[0]
            error = jnp.square(prediction - sample_value) * mask[:, None]
            return jnp.sum(error) / jnp.maximum(jnp.sum(mask) * WIDTH, 1)
        gradients = jax.vmap(jax.grad(inner_loss, argnums=(0, 1, 2, 3)))(
            weight1, bias1, weight2, bias2, key, value, inner_mask.astype(jnp.float32)
        )
        inner_learning_rate_raw = self.param(
            "inner_learning_rate_raw", zeros, (), jnp.float32
        )
        learning_rate = 0.1 * jax.nn.sigmoid(inner_learning_rate_raw)
        candidates = tuple(
            weight - learning_rate * gradient
            for weight, gradient in zip(
                (weight1, bias1, weight2, bias2), gradients, strict=True))
        valid = jnp.any(inner_mask, axis=-1)
        state = RoboTTTFastState(
            jnp.where(valid[:, None, None], candidates[0], state.w1),
            jnp.where(valid[:, None], candidates[1], state.b1),
            jnp.where(valid[:, None, None], candidates[2], state.w2),
            jnp.where(valid[:, None], candidates[3], state.b2),
            jnp.logical_or(initialized, valid),
        )
        output = fast_mlp(query, state.w1, state.b1, state.w2, state.b2)
        output = nn.Dense(name="output", **dense)(output)
        gate = self.param("residual_gate", nn.initializers.constant(0.001), (WIDTH,), jnp.float32)
        output = jnp.where(valid[:, None, None], jnp.tanh(gate) * output, 0)
        return hidden + output.astype(hidden.dtype), state

"""Regression for prefix gradients in non-TTT and full-model packed training."""

import types
import dataclasses
import importlib.util
import pathlib

from flax import nnx, struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax

from openpi.models.robottt import RoboTTTConfig


@struct.dataclass
class Observation:
    images: dict
    state: jax.Array


class LanguageModel(nnx.Module):
    def __init__(self):
        self.robottt = nnx.Param(jnp.float32(0.25))

    def __call__(self, streams, *, kv_cache=None, **kwargs):
        if streams[0] is not None:
            return None, (streams[0][None],)
        output = streams[1] + kv_cache[0][0] + self.robottt.value
        return (None, output), None


class PaliGemma(nnx.Module):
    def __init__(self):
        self.llm = LanguageModel()


class Model(nnx.Module):
    def __init__(self):
        self.vision = nnx.Param(jnp.float32(2))
        self.PaliGemma = PaliGemma()
        self.deterministic = True
        self.action_horizon = 1
        self.robottt_config = RoboTTTConfig()

    def embed_prefix(self, observation):
        tokens = observation.images["camera"] * self.vision.value
        mask = jnp.ones(tokens.shape[:2], bool)
        ar_mask = jnp.zeros(tokens.shape[1], bool)
        return tokens, mask, ar_mask, ar_mask, None

    def embed_suffix(self, observation, actions, time):
        del observation, time
        mask = jnp.ones(actions.shape[:2], bool)
        ar_mask = jnp.zeros(actions.shape[1], bool)
        condition = jnp.zeros((actions.shape[0], 1))
        return actions, mask, ar_mask, ar_mask, condition

    def action_out_proj(self, hidden):
        return hidden


@pytest.mark.parametrize("scope", ["ttt_only", "non_ttt_only", "all"])
def test_prefix_gradient_is_present_exactly_when_prefix_is_trainable(scope):
    from mme_vla_suite.training import robottt_fast as training

    model = Model()
    graph, params = nnx.split(model)
    train_state = types.SimpleNamespace(model_def=graph, params=params)
    filter_ttt = nnx.PathContains("robottt")
    trainable_filter = {"ttt_only": filter_ttt, "non_ttt_only": nnx.Not(filter_ttt), "all": nnx.Param}[scope]
    settings = RoboTTTConfig()
    config = types.SimpleNamespace(
        robottt_training_scope=scope, model=types.SimpleNamespace(robottt=settings), trainable_filter=trainable_filter
    )
    prepared = training.PreparedSequence(
        Observation({"camera": jnp.ones((1, 1, 1, 1))}, jnp.zeros((1, 1, 1))),
        jnp.ones((1, 1, 1, 1)),
        jnp.ones((1, 1)),
        jnp.zeros((1, 1, 1, 1)),
    )
    packed = training.LaneContext(
        prepared,
        {"valid": jnp.ones((1, 1), bool), "inner": jnp.ones((1, 1, 1), bool), "outer": jnp.ones((1, 1), bool)},
        (jnp.full((1, 1, 1, 1, 1), 2.0),),
        jnp.ones((1, 1, 1), jnp.int32),
        jnp.ones((1, 1, 1, 2), bool),
        None,
        jnp.zeros((1,), jnp.int32),
    )
    gradients, numerator, count = training.packed_grad(config, train_state, packed, None, jnp.float32(0), jnp.int32(0))
    assert count == 1
    np.testing.assert_allclose(numerator, 3.25**2)
    if scope == "ttt_only":
        assert "vision" not in gradients
    else:
        np.testing.assert_allclose(gradients.vision.value, 6.5)
    if scope == "non_ttt_only":
        assert "PaliGemma" not in gradients
    else:
        np.testing.assert_allclose(gradients.PaliGemma.llm.robottt.value, 6.5)


@pytest.mark.parametrize("scope", ["ttt_only", "non_ttt_only", "all"])
@pytest.mark.parametrize("ema_decay", [None, 0.99])
def test_optimizer_update_preserves_frozen_parameters_and_ema(scope, ema_decay):
    from mme_vla_suite.training import robottt_fast as training
    from openpi.training.utils import TrainState

    script = pathlib.Path(__file__).resolve().parents[3] / "src/mme_vla_suite/training/robottt.py"
    spec = importlib.util.spec_from_file_location("robottt_train_update_test", script)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    model = Model()
    model.PaliGemma.llm.slow = nnx.Param(jnp.float32(1))
    graph, params = nnx.split(model)
    ttt_filter = nnx.PathContains("robottt")
    trainable_filter = {
        "ttt_only": ttt_filter,
        "non_ttt_only": nnx.Not(ttt_filter),
        "all": nnx.Param,
    }[scope]
    config = types.SimpleNamespace(trainable_filter=trainable_filter)
    selected = params.filter(trainable_filter)
    gradients = jax.tree.map(lambda value: jnp.full_like(value, 6.5), selected)
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adamw(0.01, weight_decay=0.001))
    state = TrainState(
        step=jnp.int32(0),
        params=params,
        model_def=graph,
        tx=optimizer,
        opt_state=optimizer.init(selected),
        ema_decay=ema_decay,
        ema_params=None if ema_decay is None else params,
    )
    normalized = jax.tree.map(lambda value: value / 2, gradients)
    updates, expected_optimizer = optimizer.update(normalized, state.opt_state, selected)
    expected_selected = optax.apply_updates(selected, updates)
    _, frozen = params.split(trainable_filter, ...)
    expected_params = nnx.State.merge(frozen, expected_selected)
    stripped = dataclasses.replace(
        state,
        ema_params=None if state.ema_params is None else state.ema_params.filter(trainable_filter),
    )
    result, info = train.apply_robottt_gradients(config, stripped, gradients, jnp.float32(20), jnp.int32(2))
    result = training.alias_frozen_ema(config, result)
    assert int(result.step) == 1
    assert float(info["loss"]) == 10
    for expected, actual in zip(jax.tree.leaves(expected_params), jax.tree.leaves(result.params), strict=True):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(jax.tree.leaves(expected_optimizer), jax.tree.leaves(result.opt_state), strict=True):
        np.testing.assert_array_equal(expected, actual)
    if ema_decay is None:
        assert result.ema_params is None
    else:
        expected_ema = jax.tree.map(
            lambda old, new: ema_decay * old + (1 - ema_decay) * new,
            selected,
            expected_selected,
        )
        expected_ema = nnx.State.merge(frozen, expected_ema)
        for expected, actual in zip(jax.tree.leaves(expected_ema), jax.tree.leaves(result.ema_params), strict=True):
            np.testing.assert_array_equal(expected, actual)

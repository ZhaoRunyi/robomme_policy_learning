"""Contract and small-width numerical regression tests for design choices."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import robottt


@pytest.mark.parametrize(
    "scope,frozen,freeze_ttt", [("ttt_only", True, False), ("non_ttt_only", False, True), ("all", False, False)]
)
def test_scope_conflicts(scope, frozen, freeze_ttt):
    settings = robottt.RoboTTTConfig(frozen=frozen, freeze_ttt=freeze_ttt)
    assert robottt.resolve_training_scope(scope, settings) == scope
    with pytest.raises(ValueError, match="conflicts"):
        robottt.resolve_training_scope(scope, dataclasses.replace(settings, frozen=not frozen))
    with pytest.raises(ValueError, match="conflicts"):
        robottt.resolve_training_scope(scope, dataclasses.replace(settings, freeze_ttt=not freeze_ttt))


def test_curriculum_durations_and_endpoint_warning():
    settings = robottt.RoboTTTConfig(curriculum={4: 10, 16: 20, 96: 30})
    assert settings.stage_at(9) == (0, 4)
    assert settings.stage_at(10) == (10, 16)
    assert settings.stage_at(1_000) == (30, 96)
    with pytest.warns(UserWarning, match="final T continues"):
        assert settings.schedule(100) == ((0, 4), (10, 16), (30, 96))
    with pytest.warns(UserWarning, match="final T continues"):
        assert settings.schedule(15) == ((0, 4), (10, 16), (14, 96))
    with pytest.warns(UserWarning, match="final T continues"):
        assert settings.schedule(1) == ((0, 96),)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"fast_update": "last_layer"},
        {"fast_update": "last_layer", "fast_output_bias": False},
        {"fast_model": "linear"},
        {"fast_model": "linear", "fast_bias": False},
        {"fast_bias": False},
        {"operation_order": "read_then_write"},
        {"qk_normalization": "silu_l2", "weight_normalization": "unit_input_norm", "fast_loss": "negative_dot"},
        {"target_normalization": "v_minus_k_layernorm"},
        {"qk_normalization": "none"},
    ],
)
def test_finite_masked_update_and_outer_gradients(changes):
    settings = robottt.RoboTTTConfig(fast_width=12, **changes)
    layer = robottt.Layer(settings, width=8, heads=2)
    hidden = jax.random.normal(jax.random.key(1), (2, 3, 8))
    state = jax.tree.map(lambda value: value[0], settings.create_state(1, 2, 8))
    mask = jnp.array([[True, True, False], [False, False, False]])
    position = jnp.zeros(2, dtype=jnp.int32)
    variables = layer.init(jax.random.key(2), hidden, state, mask, position)
    output, next_state = layer.apply(variables, hidden, state, mask, position)
    np.testing.assert_array_equal(output[1], hidden[1])
    for previous, current in zip(jax.tree.leaves(state), jax.tree.leaves(next_state), strict=True):
        np.testing.assert_array_equal(current[1], previous[1])
        assert np.isfinite(current).all()
    gradients = jax.grad(lambda params: jnp.sum(layer.apply({"params": params}, hidden, state, mask, position)[0]))(
        variables["params"]
    )
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(gradients))
    assert any(np.any(value != 0) for value in jax.tree.leaves(gradients))


def test_order_only_changes_read_not_write():
    settings = robottt.RoboTTTConfig(fast_width=12)
    layer = robottt.Layer(settings, 8, 2)
    hidden = jnp.ones((1, 2, 8))
    state = jax.tree.map(lambda value: value[0], settings.create_state(1, 1, 8))
    args = (hidden, state, jnp.ones((1, 2), bool), jnp.zeros(1, jnp.int32))
    variables = layer.init(jax.random.key(0), *args)
    output_wtr, state_wtr = layer.apply(variables, *args)
    output_rtw, state_rtw = robottt.Layer(dataclasses.replace(settings, operation_order="read_then_write"), 8, 2).apply(
        variables, *args
    )
    for left, right in zip(jax.tree.leaves(state_wtr), jax.tree.leaves(state_rtw), strict=True):
        np.testing.assert_array_equal(left, right)
    assert np.any(output_wtr != output_rtw)


def test_candidate_json_roundtrip_and_removed_design():
    settings = robottt.RoboTTTConfig(fast_model="linear", fast_bias=False, curriculum={96: 60_000})
    candidate = {"robottt_config": dataclasses.asdict(settings), "video_action_source": "ground_truth"}
    candidate["robottt_config"]["curriculum"] = {"96": 60_000}
    assert robottt.config_from_candidate(robottt.RoboTTTConfig(), candidate) == settings
    with pytest.raises(ValueError, match="ground_truth"):
        robottt.config_from_candidate(settings, {"video_action_source": "none"})


def test_candidate_history_conflict_does_not_change_manifest(tmp_path):
    settings = robottt.RoboTTTConfig()
    robottt.write_candidate(tmp_path, settings, "ttt_only", "robottt")
    before = (tmp_path / "candidate.json").read_text()
    with pytest.raises(ValueError, match="history_config"):
        robottt.write_candidate(tmp_path, settings, "ttt_only", "different_history")
    assert (tmp_path / "candidate.json").read_text() == before
    assert robottt.read_candidate(tmp_path / "60000")["training_scope"] == "ttt_only"
    (tmp_path / "history_config.txt").unlink()
    with pytest.raises(FileNotFoundError, match="history_config"):
        robottt.read_candidate(tmp_path / "60000")

import json
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import dataclasses

import openpi.models.model as _model
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms

import mme_vla_suite.policies.policy as _policy
from openpi.models.robottt import config_from_candidate
import mme_vla_suite.training.config as _config


def _read_checkpoint_file(checkpoint_dir: pathlib.Path, name: str) -> str | None:
    for path in (checkpoint_dir.parent / name, checkpoint_dir / name):
        if path.exists():
            return path.read_text().strip()
    return None


def _load_candidate(checkpoint_dir: pathlib.Path) -> dict[str, Any] | None:
    raw_candidate = _read_checkpoint_file(checkpoint_dir, "candidate.json")
    if raw_candidate is None:
        return None
    candidate = json.loads(raw_candidate)
    if not isinstance(candidate, dict):
        raise ValueError("candidate.json must contain a JSON object")
    return candidate


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.MME_VLA_Policy:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    repack_transforms = repack_transforms or transforms.Group()

    logging.info(f"Checking history config")
    history_config = _read_checkpoint_file(checkpoint_dir, "history_config.txt")
    if history_config is None and train_config.model.history_config is not None:
        raise FileNotFoundError(
            "Checkpoint history_config.txt is missing; refusing to silently "
            "disable the configured history/RoboTTT model"
        )

    if train_config.model.history_config != history_config:
        print(f" == You are using {train_config.model.history_config}, changing to {history_config} ==")
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, history_config=history_config, use_history=history_config is not None)
        )

    candidate = _load_candidate(checkpoint_dir)
    if "robottt" in str(history_config):
        if candidate is None:
            raise FileNotFoundError("RoboTTT checkpoint requires candidate.json beside its numbered checkpoints")
        settings = config_from_candidate(train_config.model.robottt, candidate)
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, robottt=settings),
            robottt_training_scope=candidate.get("training_scope", train_config.robottt_training_scope),
            train_robottt_only=None,
        )

    logging.info("Loading model...")
    restore_dtype = None if "robottt" in str(train_config.model.history_config) else jnp.bfloat16
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=restore_dtype))
    expected_robottt = "robottt" in str(history_config)
    if bool(model.use_robottt) != expected_robottt:
        raise RuntimeError(
            f"RoboTTT restore mismatch: expected={expected_robottt}, "
            f"actual={model.use_robottt}"
        )
    register_tokens = None
    if model.use_robottt:
        register_tokens = int(model.robottt_register_tokens.value.shape[0])
    expected_register_tokens = None if candidate is None else candidate.get("register_tokens")
    if expected_register_tokens is not None and register_tokens != int(expected_register_tokens):
        raise RuntimeError(
            f"RoboTTT register-token mismatch: expected={expected_register_tokens}, "
            f"actual={register_tokens}"
        )
    effective_inner_lr_scale = train_config.model.robottt.fast_learning_rate
    print(
        "EVAL_CONTRACT "
        f"checkpoint={checkpoint_dir} "
        f"history_config={history_config} "
        f"uses_robottt={model.use_robottt} "
        f"inner_lr_scale={effective_inner_lr_scale} "
        f"register_tokens={register_tokens} "
        f"video_action_source=ground_truth "
        f"fast_model={train_config.model.robottt.fast_model} "
        f"fast_update={train_config.model.robottt.fast_update} "
        f"operation_order={train_config.model.robottt.operation_order}"
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    return _policy.MME_VLA_Policy(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm
    )

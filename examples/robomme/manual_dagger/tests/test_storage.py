"""tests / test_storage: shared behavior owned by this layer."""

from __future__ import annotations

import json
from pathlib import Path
import h5py
from manual_dagger.storage import StorageProcess
from manual_dagger.storage import validate_enabled_training_h5
import numpy as np
import pytest
from unittest.mock import patch
from manual_dagger.reconstruction import _apply_review_decision
from manual_dagger.reconstruction import _resolve_review_decision
from manual_dagger.reconstruction import takeover_frame_pairs
from manual_dagger.reconstruction import visual_gate


def _write_trainable_h5(path: Path, *, expert: bool = True) -> None:
    with h5py.File(path, "w") as output:
        episode = output.create_group("episode_0")
        setup = episode.create_group("setup")
        setup.create_dataset(
            "task_goal",
            data=np.asarray(["put the cube in the bin"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        setup.create_dataset("official_eval_success", data=True)
        setup.create_dataset("human_intervened", data=True)
        for index in range(3):
            timestep = episode.create_group(f"timestep_{index}")
            obs = timestep.create_group("obs")
            obs.create_dataset("front_rgb", data=np.zeros((256, 256, 3), dtype=np.uint8))
            obs.create_dataset("wrist_rgb", data=np.zeros((256, 256, 3), dtype=np.uint8))
            obs.create_dataset("joint_state", data=np.zeros(7, dtype=np.float32))
            obs.create_dataset("gripper_state", data=np.zeros(2, dtype=np.float32))
            action = timestep.create_group("action")
            action.create_dataset("joint_action", data=np.zeros(8, dtype=np.float32))
            info = timestep.create_group("info")
            info.create_dataset("is_video_demo", data=index == 0)
            info.create_dataset("action_source", data=expert and index == 2)


def test_storage_process_encodes_and_publishes(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    primary = tmp_path / "primary"
    mirror = tmp_path / "mirror"
    staging.mkdir()
    h5_path = staging / "episode.h5"
    _write_trainable_h5(h5_path)
    storage = StorageProcess(primary, mirror, video_fps=10, queue_size=8)
    try:
        storage.begin("episode-id", staging)
        for value in range(4):
            storage.frame(
                "episode-id",
                np.full((64, 128, 3), value * 50, dtype=np.uint8),
            )
        result = storage.finish(
            "episode-id",
            True,
            [h5_path],
            {
                "episode_id": "episode-id",
                "artifact_stem": "BinFill_ep0_seed1",
                "task": "BinFill",
                "role": "enabled",
                "eval_success": True,
            },
        )
        assert result["status"] == "published"
        assert result["artifacts"]["h5"]
        assert (primary / "BinFill/BinFill_ep0_seed1.mp4").is_file()
        assert (primary / "BinFill/BinFill_ep0_seed1.h5").is_file()
        assert (mirror / "BinFill/BinFill_ep0_seed1.mp4").is_file()
    finally:
        storage.close()


def test_storage_process_materializes_trainable_dataset(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    primary = tmp_path / "primary"
    training = tmp_path / "training"
    staging.mkdir()
    h5_path = staging / "episode.h5"
    _write_trainable_h5(h5_path)
    storage = StorageProcess(
        primary,
        None,
        training_root=training,
        video_fps=10,
        queue_size=8,
    )
    try:
        storage.begin("episode-id", staging)
        storage.frame(
            "episode-id",
            np.zeros((64, 128, 3), dtype=np.uint8),
        )
        result = storage.finish(
            "episode-id",
            True,
            [h5_path],
            {
                "episode_id": "episode-id",
                "artifact_stem": "BinFill_ep0_seed1_enabled",
                "task": "BinFill",
                "episode": 0,
                "seed": 1,
                "take": 1,
                "prompt": "put the cube in the bin",
                "role": "enabled",
                "eval_success": True,
                "intervened": True,
            },
        )
        dataset_h5 = training / "BinFill/BinFill_ep0_seed1_enabled.h5"
        assert dataset_h5.is_file()
        assert result["artifacts"]["h5"]
        manifest = json.loads(
            (training / "training_manifest.json").read_text(encoding="utf-8")
        )
        assert len(manifest["episodes"]) == 1
        assert manifest["episodes"][0]["h5_path"] == str(dataset_h5)
        assert manifest["total_execution_count"] == 2
    finally:
        storage.close()


def test_storage_process_rejects_empty_enabled_h5(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    primary = tmp_path / "primary"
    staging.mkdir()
    h5_path = staging / "episode.h5"
    with h5py.File(h5_path, "w") as output:
        output.create_group("episode_0")
    storage = StorageProcess(primary, None, video_fps=10, queue_size=8)
    try:
        storage.begin("episode-id", staging)
        storage.frame(
            "episode-id",
            np.zeros((64, 128, 3), dtype=np.uint8),
        )
        with pytest.raises(RuntimeError, match="missing setup/task_goal"):
            storage.finish(
                "episode-id",
                True,
                [h5_path],
                {
                    "episode_id": "episode-id",
                    "artifact_stem": "BinFill_ep0_seed1_enabled",
                    "task": "BinFill",
                    "role": "enabled",
                    "eval_success": True,
                },
            )
        assert not (primary / "BinFill/BinFill_ep0_seed1_enabled.h5").exists()
    finally:
        storage.close()


def test_storage_process_allows_control_video_without_h5(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    primary = tmp_path / "primary"
    staging.mkdir()
    storage = StorageProcess(primary, None, video_fps=10, queue_size=8)
    try:
        storage.begin("control-id", staging)
        storage.frame(
            "control-id",
            np.zeros((64, 128, 3), dtype=np.uint8),
        )
        result = storage.finish(
            "control-id",
            True,
            [],
            {
                "episode_id": "control-id",
                "artifact_stem": "BinFill_ep0_seed1_control",
                "task": "BinFill",
                "role": "control",
            },
        )
        assert result["status"] == "published"
        assert (primary / "BinFill/BinFill_ep0_seed1_control.mp4").is_file()
        assert result["artifacts"]["h5"] == []
    finally:
        storage.close()


def test_training_validator_rejects_enabled_h5_without_expert_targets(tmp_path: Path) -> None:
    path = tmp_path / "episode.h5"
    _write_trainable_h5(path, expert=False)
    try:
        validate_enabled_training_h5(path)
    except ValueError as error:
        assert "no expert action_source" in str(error)
    else:
        raise AssertionError("validator accepted an HDF5 with no supervised DAgger targets")


def test_trainable_h5_is_consumed_by_robottt_sequence_dataset(tmp_path: Path) -> None:
    from mme_vla_suite.training.dataset import RoboTTTSequenceDataset

    path = tmp_path / "episode.h5"
    _write_trainable_h5(path)
    dataset = RoboTTTSequenceDataset.__new__(RoboTTTSequenceDataset)
    dataset.episodes = [
        {
            "h5_path": str(path),
            "episode_group": "episode_0",
            "task_goal": "put the cube in the bin",
            "num_steps": 3,
            "exec_start_idx": 1,
            "cumulative_execution_count": 2,
        }
    ]
    dataset.ends = [2]
    dataset.temporal_blocks = 1
    dataset.inner_width = 36
    dataset.files = {}
    dataset.tokenize = lambda sample: {
        "tokenized_prompt": np.zeros(4, dtype=np.int32),
        "tokenized_prompt_mask": np.ones(4, dtype=np.bool_),
    }
    dataset.transform = lambda sample: {
        "image": sample["observation/image"],
        "wrist_image": sample["observation/wrist_image"],
        "state": sample["observation/state"],
        "actions": sample["actions"],
        "prompt": sample["prompt"],
    }

    sample = dataset[1]

    assert sample["actions"].shape == (1, 20, 8)
    assert sample["state"].shape == (1, 8)
    assert sample["robottt"]["outer"].tolist() == [True]


def test_takeover_pairs_skip_legacy_demonstration_frames() -> None:
    pairs = takeover_frame_pairs(
        [1, 1, 1, 2, 3, 3],
        [1, 2, 3, 3],
        2,
    )
    assert pairs == {
        "execution_start": (2, 0),
        "policy_takeover": (3, 1),
        "first_expert": (4, 2),
    }


def test_visual_gate_checks_each_camera_at_takeover() -> None:
    old = [np.zeros((4, 8), dtype=np.uint8) for _ in range(3)]
    new = [np.zeros((4, 8), dtype=np.uint8) for _ in range(3)]
    new[1][:, 4:] = 13
    with patch(
        "manual_dagger.reconstruction.read_video",
        side_effect=[old, new],
    ):
        result = visual_gate(
            Path("old.mp4"),
            Path("new.mp4"),
            [1, 2, 3],
            [1, 2, 3],
            2,
            maximum_mean_mae=8.0,
            maximum_camera_mae=12.0,
        )
    assert result["passed"] is False
    assert result["camera_mae"]["policy_takeover"]["front"] == 0.0
    assert result["camera_mae"]["policy_takeover"]["wrist"] == 13.0
    assert result["transition_max_camera_mae"] == 13.0


def test_human_review_can_accept_only_aligned_visual_difference() -> None:
    decision = {"decision": "accept"}
    assert _apply_review_decision(
        {
            "passed": False,
            "reason": "visual_difference",
            "alignment": "logical_takeover_transition",
        },
        decision,
    ) == (True, None)
    assert _apply_review_decision(
        {
            "passed": False,
            "reason": "policy_terminal_before_retained_takeover",
        },
        decision,
    ) == (False, "manual_accept_requires_aligned_visual_difference")


def test_human_reject_overrides_automatic_gate() -> None:
    assert _apply_review_decision(
        {"passed": True, "reason": "matched"},
        {"decision": "reject"},
    ) == (False, None)


def test_exact_reject_overrides_task_review_rule() -> None:
    record = {
        "task": "VideoPlaceOrder",
        "episode": 3,
        "seed": 11305,
        "take": 43,
    }
    decisions = {
        "VideoPlaceOrder:3:11305:43": {
            **record,
            "decision": "reject",
        }
    }
    rule = {
        "task": "VideoPlaceOrder",
        "gate_reason": "visual_difference",
        "decision": "accept",
    }
    decision = _resolve_review_decision(
        record,
        {"reason": "visual_difference"},
        decisions,
        [rule],
    )
    assert decision is not None
    assert decision["decision"] == "reject"

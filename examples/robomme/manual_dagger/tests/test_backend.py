"""tests / test_backend: shared behavior owned by this layer."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mme_vla_suite.policies.policy import MME_VLA_Policy
from manual_dagger.web import overlay_target_markers
from manual_dagger.web import overlay_world_target_marker


POLICY_ROOT = Path(__file__).resolve().parents[4]


def _load_planner_state_module():
    module_path = (
        POLICY_ROOT
        / "third_party/robomme_benchmark/src/robomme/env_record_wrapper/planner_state.py"
    )
    spec = importlib.util.spec_from_file_location("planner_state", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_finish_with_expert_method():
    source_path = POLICY_ROOT / "examples/robomme/dagger_review/env_runner.py"
    tree = ast.parse(source_path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "finish_with_expert"
    )
    class_node = ast.ClassDef(
        name="ExtractedEnvRunner",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[class_node], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    runner_class = namespace["ExtractedEnvRunner"]
    runner_class.subtask_progress = property(
        lambda self: self.env.unwrapped.current_task_index
    )
    return runner_class.finish_with_expert


def test_fresh_expert_planner_restores_closed_live_gripper() -> None:
    planner_state = _load_planner_state_module()
    planner = SimpleNamespace(OPEN=1.0, CLOSED=-1.0, gripper_state=1.0)
    robot = SimpleNamespace(get_qpos=lambda: np.array([0.0] * 9))

    result = planner_state.sync_gripper_state_from_robot(planner, robot)

    assert result == -1.0
    assert planner.gripper_state == -1.0


def test_fresh_expert_planner_preserves_open_live_gripper() -> None:
    planner_state = _load_planner_state_module()
    planner = SimpleNamespace(OPEN=1.0, CLOSED=-1.0, gripper_state=-1.0)
    robot = SimpleNamespace(
        get_qpos=lambda: np.array([0.0] * 7 + [0.02, 0.02])
    )

    result = planner_state.sync_gripper_state_from_robot(planner, robot)

    assert result == 1.0
    assert planner.gripper_state == 1.0


def test_fresh_expert_planner_keeps_grasp_command_with_object_between_fingers() -> None:
    planner_state = _load_planner_state_module()
    planner = SimpleNamespace(OPEN=1.0, CLOSED=-1.0, gripper_state=1.0)
    robot = SimpleNamespace(
        get_qpos=lambda: np.array([0.0] * 7 + [0.025, 0.025])
    )
    controller = SimpleNamespace(
        _target_qpos=np.array([[0.0] * 7 + [-0.01, -0.01]]),
        control_joint_indices=np.array([7]),
    )

    result = planner_state.sync_gripper_state_from_robot(
        planner,
        robot,
        controller,
    )

    assert result == -1.0
    assert planner.gripper_state == -1.0


def test_expert_resumes_authoritative_task_instead_of_lagging_progress() -> None:
    finish_with_expert = _load_finish_with_expert_method()
    task_list = [
        {"name": "pickup", "demonstration": False},
        {"name": "place", "demonstration": False},
    ]
    raw_env = SimpleNamespace(
        task_list=task_list,
        timestep=1,
        current_task_index=0,
        successflag=False,
    )
    observed_flags = []

    class DemonstrationWrapper:
        def get_demonstration_trajectory(self) -> None:
            observed_flags.append(
                [task["demonstration"] for task in task_list]
            )
            raw_env.current_task_index = len(task_list)
            raw_env.successflag = True

    runner_class = type(
        "Runner",
        (),
        {
            "subtask_progress": property(
                lambda self: self.env.unwrapped.current_task_index
            )
        },
    )
    runner = runner_class()
    runner.env = SimpleNamespace(
        env=DemonstrationWrapper(),
        unwrapped=raw_env,
    )
    runner.recorder = SimpleNamespace(action_source=0, episode_success=False)

    success = finish_with_expert(runner)

    assert success
    assert observed_flags == [[False, True]]
    assert [task["demonstration"] for task in task_list] == [False, False]


def test_single_subtask_takeover_leaves_later_subtasks_unselected() -> None:
    finish_with_expert = _load_finish_with_expert_method()
    tasks = [{"name": str(index), "demonstration": False} for index in range(4)]
    environment = SimpleNamespace(task_list=tasks, timestep=1, successflag=False)
    seen = []

    def run_selected():
        seen.append([task["demonstration"] for task in tasks])
        environment.timestep += 1

    runner = SimpleNamespace(
        env=SimpleNamespace(
            unwrapped=environment,
            env=SimpleNamespace(get_demonstration_trajectory=run_selected),
        ),
        recorder=SimpleNamespace(action_source=0, episode_success=False),
    )
    assert not finish_with_expert(runner, max_subtasks=1)
    assert seen == [[False, True, False, False]]
    assert environment.timestep == 2
    assert not any(task["demonstration"] for task in tasks)


def test_robottt_policy_reset_point_restores_prefilled_state() -> None:
    policy = MME_VLA_Policy.__new__(MME_VLA_Policy)
    policy._uses_robottt = True
    policy._rng = np.asarray([1, 2], dtype=np.uint32)
    policy.step_idx = 17
    policy.exec_start_idx = 256
    policy._robottt_mode = "normal"
    initial_fast_state = {"weights": np.asarray([3.0, 4.0])}
    policy._episode_fast_state = initial_fast_state

    policy.save_reset_point()
    policy._rng = np.asarray([9, 9], dtype=np.uint32)
    policy.step_idx = 99
    policy.exec_start_idx = 0
    policy._robottt_mode = "no_video"
    policy._episode_fast_state = {"weights": np.asarray([8.0, 8.0])}

    policy.restore_reset_point()

    np.testing.assert_array_equal(policy._rng, [1, 2])
    assert policy.step_idx == 17
    assert policy.exec_start_idx == 256
    assert policy._robottt_mode == "normal"
    assert policy._episode_fast_state is initial_fast_state


class Actor:
    pass


def _recorder(environment, mapping):
    return SimpleNamespace(unwrapped=environment, segmentation_id_map=mapping)


def test_generic_target_draws_only_a_small_center_dot() -> None:
    target = Actor()
    distractor = Actor()
    environment = SimpleNamespace(current_segment=target)
    segmentation = np.zeros((24, 24), dtype=np.int32)
    segmentation[4:9, 6:11] = 1
    segmentation[14:20, 15:21] = 2
    frame = np.zeros((24, 24, 3), dtype=np.uint8)

    marked = overlay_target_markers(
        frame,
        segmentation,
        _recorder(environment, {1: target, 2: distractor}),
        "MoveCube",
    )

    assert np.array_equal(marked[6, 8], [139, 92, 246])
    assert np.array_equal(marked[16, 17], [0, 0, 0])
    assert np.count_nonzero(np.all(marked == [139, 92, 246], axis=-1)) < 40
    assert not np.any(frame)


def test_binfill_marks_all_visible_remaining_valid_colors() -> None:
    red = Actor()
    blue = Actor()
    green = Actor()
    environment = SimpleNamespace(
        current_task_index=0,
        task_list=[{"name": "pick up the first blue cube"}],
        current_segment=blue,
        red_cubes=[red],
        blue_cubes=[blue],
        green_cubes=[green],
        red_cubes_target_number=0,
        blue_cubes_target_number=3,
        green_cubes_target_number=1,
        red_cubes_in_bin=0,
        blue_cubes_in_bin=0,
        green_cubes_in_bin=0,
    )
    segmentation = np.zeros((30, 30), dtype=np.int32)
    segmentation[3:6, 3:6] = 1
    segmentation[12:15, 12:15] = 2
    segmentation[21:24, 21:24] = 3
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    recorder = _recorder(environment, {1: red, 2: blue, 3: green})

    marked = overlay_target_markers(frame, segmentation, recorder, "BinFill")
    assert np.array_equal(marked[13, 13], [139, 92, 246])
    assert np.array_equal(marked[22, 22], [139, 92, 246])
    assert np.array_equal(marked[4, 4], [0, 0, 0])

    environment.green_cubes_in_bin = 1
    marked_after_green = overlay_target_markers(
        frame,
        segmentation,
        recorder,
        "BinFill",
    )
    assert np.array_equal(marked_after_green[13, 13], [139, 92, 246])
    assert np.array_equal(marked_after_green[22, 22], [0, 0, 0])


def test_world_target_projects_to_a_purple_dot() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    camera = {
        "intrinsic_cv": np.array([[20.0, 0.0, 30.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]),
        "extrinsic_cv": np.eye(4),
    }

    marked = overlay_world_target_marker(frame, camera, [0.0, 0.0, 1.0])

    assert np.array_equal(marked[20, 30], [139, 92, 246])
    assert not np.any(frame)


def test_world_target_accepts_sapien_three_by_four_extrinsic() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    camera = {
        "intrinsic_cv": np.array([[20.0, 0.0, 30.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]),
        "extrinsic_cv": np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
    }

    marked = overlay_world_target_marker(frame, camera, [0.0, 0.0, 1.0])

    assert np.array_equal(marked[20, 30], [139, 92, 246])

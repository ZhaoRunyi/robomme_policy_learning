"""
Participants need to modify this file

This is a sample script about how to adapt a model into a remote evaluation policy for CVPR challenge.

Basically, You need to implement the `step` and `reset` methods.
"""


from typing import Any
import numpy as np
from mme_vla_suite.policies.policy import MME_VLA_Policy as _InnerMMEVLAPolicy
from typing_extensions import override


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    # pack into 8-dim state, same as the joint action space
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


class Policy:
    def infer(self, inputs: dict):
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class MyPolicy_for_CVPR_Challenge(Policy):
    """
    Adapter around the existing `MME_VLA_Policy` that exposes only the
    `infer` and `reset` interface required by the CVPR remote evaluation
    server, while internally managing the history buffer.

    This class does NOT change the original `MME_VLA_Policy` interface;
    it simply wraps an instance of it and forwards calls.

    """

    def __init__(self, model: _InnerMMEVLAPolicy, **_: Any):
        """
        Wrap an already-constructed `MME_VLA_Policy`.

        `create_trained_policy` in `policy_config.py` already returns an
        `MME_VLA_Policy` with the correct model, transforms, metadata,
        and normalization statistics, so we simply keep a reference to
        that instance here.
        """
        self._inner_policy = model
        self.chunk_size = 16

    @override
    def infer(self, inputs: dict) -> dict:
        """
        Public `infer` interface expected by the CVPR server.

        If `inputs` carries history information compatible with the
        original `add_buffer` API, this method will update the memory
        buffer automatically before calling the wrapped policy's
        `infer`.
        """
        # Auto-update buffer if history is provided in this call.
        # We keep this logic intentionally lightweight and permissive:
        # if the keys are present, we assume they are meant for the
        # history buffer in the same format used by `add_buffer`.
                
        robot_state_list = [pack_state(joint_state, gripper_state) for joint_state, gripper_state in zip(inputs["joint_state_list"], inputs["gripper_state_list"])]

        add_buffer_payload = {
            "images": np.stack(inputs["front_rgb_list"], axis=0).astype(np.uint8)[:, None],
            "state": np.stack(robot_state_list, axis=0).astype(np.float32), # this is actually not used in the MME-VLA-Suite model
        }
        if inputs["is_first_step"]:
            exec_start_idx = len(inputs["front_rgb_list"]) - 1
        else:
            exec_start_idx = -1 # placeholder
        
        add_buffer_payload["exec_start_idx"] = exec_start_idx

        self._inner_policy.add_buffer(add_buffer_payload)

        # Forward the (possibly richer) inputs directly to the original
        # MME_VLA_Policy.infer implementation.
        element = {
            "observation/image": inputs["front_rgb_list"][-1],
            "observation/wrist_image": inputs["wrist_rgb_list"][-1],
            "observation/state": robot_state_list[-1],
            "prompt": inputs["task_goal"][0].lower(),
        }
                
        outputs = self._inner_policy.infer(element)
        return {"actions": outputs["actions"][:self.chunk_size, :]}

    @override
    def reset(self) -> None:
        """Public `reset` interface simply forwards to the wrapped policy."""
        self._inner_policy.reset()

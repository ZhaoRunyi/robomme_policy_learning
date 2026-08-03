import copy
import time

import jax
import jax.numpy as jnp
import numpy as np


class BatchedRoboTTTPolicy:
    def __init__(self, policy, lane_count: int, inference_batch_size: int):
        if not policy._uses_robottt:
            raise ValueError("Parallel clients are only supported for RoboTTT policies")
        self.policy = policy
        self.inference_batch_size = inference_batch_size
        self.lanes = [copy.copy(policy) for _ in range(lane_count)]
        self.connected = np.zeros(lane_count, dtype=bool)
        for lane in self.lanes:
            lane.reset()

    def allocate(self) -> int:
        available = np.flatnonzero(~self.connected)
        if not len(available):
            raise RuntimeError("All parallel policy lanes are occupied")
        slot = int(available[0])
        self.connected[slot] = True
        return slot

    def release(self, slot: int) -> None:
        self.connected[slot] = False

    def reset(self, slot: int, mode: str) -> None:
        self.lanes[slot].reset(mode)

    def add_buffer(self, slot: int, observation: dict) -> None:
        self.lanes[slot].add_buffer(observation)

    def infer(self, requests):
        prepared = []
        noises = []
        states = []
        positions = []
        for slot, observation, _ in requests:
            lane = self.lanes[slot]
            prepared.append(lane._prepare_observation(observation))
            lane._rng, sample_rng = jax.random.split(lane._rng)
            noises.append(jax.random.normal(
                sample_rng,
                (1, lane._model.action_horizon, lane._model.action_dim),
            ))
            states.append(lane._episode_fast_state)
            positions.append(lane.step_idx)

        while len(prepared) < self.inference_batch_size:
            prepared.append(prepared[0])
            noises.append(jnp.zeros_like(noises[0]))
            states.append(states[0])
            positions.append(positions[0])

        observation = jax.tree.map(lambda *values: jnp.concatenate(values), *prepared)
        fast_state = jax.tree.map(lambda *values: jnp.concatenate(values, axis=1), *states)
        started = time.monotonic()
        actions, candidate_state = self.policy._sample_actions(
            jax.random.key(0),
            observation,
            noise=jnp.concatenate(noises),
            fast_state=fast_state,
            block_position=jnp.asarray(positions),
        )
        actions = np.asarray(actions)
        candidate_state = jax.tree.map(np.asarray, candidate_state)
        infer_time_ms = (time.monotonic() - started) * 1000

        outputs = []
        for batch_index, (slot, _, future) in enumerate(requests):
            lane = self.lanes[slot]
            if lane._robottt_mode != "no_carry":
                lane._episode_fast_state = jax.tree.map(
                    lambda value: value[:, batch_index:batch_index + 1], candidate_state)
            lane.step_idx += 1
            result = lane._output_transform({
                "state": np.asarray(observation.state[batch_index]),
                "actions": actions[batch_index],
            })
            result["infer_time_ms"] = infer_time_ms
            outputs.append((future, result))
        return outputs

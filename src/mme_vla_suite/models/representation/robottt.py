"""RoboMME adapter for the shared OpenPI RoboTTT fast learner."""

from openpi.models.robottt import FastState as RoboTTTFastState
from openpi.models.robottt import Layer
from openpi.models.robottt import RoboTTTConfig
from openpi.models.robottt import create_fast_state as _create_fast_state


DEPTH, WIDTH, FAST_WIDTH, HEADS = 18, 1024, 2048, 4
__all__ = ["RoboTTTFastState", "RoboTTTLayer", "create_fast_state"]


def create_fast_state(batch_size, config=None, *, depth=DEPTH, width=WIDTH):
    return _create_fast_state(config or RoboTTTConfig(), depth, batch_size, width)


class RoboTTTLayer(Layer):
    config: RoboTTTConfig = RoboTTTConfig()
    width: int = WIDTH
    heads: int = HEADS

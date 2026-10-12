"""Validated high-level actions and their MineDojo translations."""

from __future__ import annotations

from enum import Enum
import time
from typing import Callable, Dict, Tuple

import numpy as np


class AgentAction(str, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    STRAFE_LEFT = "STRAFE_LEFT"
    STRAFE_RIGHT = "STRAFE_RIGHT"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    LOOK_UP = "LOOK_UP"
    LOOK_DOWN = "LOOK_DOWN"
    JUMP = "JUMP"
    ATTACK = "ATTACK"
    BREAK_BLOCK = "BREAK_BLOCK"
    USE = "USE"
    WAIT = "WAIT"
    STOP = "STOP"


def _set(index: int, value: int) -> Callable[[np.ndarray], None]:
    def configure(action: np.ndarray) -> None:
        action[index] = value

    return configure


def _offset(index: int, delta: int) -> Callable[[np.ndarray], None]:
    """Offset a value from the environment-specific no-op action."""
    def configure(action: np.ndarray) -> None:
        action[index] += delta

    return configure


ACTION_CONFIGURERS: Dict[AgentAction, Callable[[np.ndarray], None]] = {
    AgentAction.MOVE_FORWARD: _set(0, 1),
    AgentAction.MOVE_BACKWARD: _set(0, 2),
    AgentAction.STRAFE_LEFT: _set(1, 1),
    AgentAction.STRAFE_RIGHT: _set(1, 2),
    AgentAction.TURN_LEFT: _offset(4, -1),
    AgentAction.TURN_RIGHT: _offset(4, 1),
    AgentAction.LOOK_UP: _offset(3, -1),
    AgentAction.LOOK_DOWN: _offset(3, 1),
    AgentAction.JUMP: _set(2, 1),
    AgentAction.ATTACK: _set(5, 3),
    AgentAction.BREAK_BLOCK: _set(5, 3),
    AgentAction.USE: _set(5, 1),
    AgentAction.WAIT: lambda action: None,
}


ACTION_DURATION_LIMITS: Dict[AgentAction, Tuple[int, int]] = {
    AgentAction.MOVE_FORWARD: (1, 8),
    AgentAction.MOVE_BACKWARD: (1, 8),
    AgentAction.STRAFE_LEFT: (1, 8),
    AgentAction.STRAFE_RIGHT: (1, 8),
    AgentAction.TURN_LEFT: (1, 18),
    AgentAction.TURN_RIGHT: (1, 18),
    AgentAction.LOOK_UP: (1, 12),
    AgentAction.LOOK_DOWN: (1, 12),
    AgentAction.JUMP: (1, 1),
    AgentAction.ATTACK: (1, 1),
    # A wooden chest takes roughly 75 ticks to break by hand. Keep enough range
    # for the smoke-test agent to spill and collect chest contents.
    AgentAction.BREAK_BLOCK: (2, 100),
    AgentAction.USE: (1, 1),
    AgentAction.WAIT: (1, 6),
}


def execute_action(
    env,
    action_name: AgentAction,
    duration: int,
    step_delay: float = 0.05,
):
    """Execute a bounded high-level action and return the last transition."""
    if action_name == AgentAction.STOP:
        raise ValueError("STOP is a control signal and cannot be stepped")
    minimum, maximum = ACTION_DURATION_LIMITS[action_name]
    if not minimum <= duration <= maximum:
        raise ValueError(
            f"{action_name.value} duration must be between {minimum} and {maximum}"
        )

    configure = ACTION_CONFIGURERS[action_name]
    transition = None
    for _ in range(duration):
        action = env.action_space.no_op()
        configure(action)
        transition = env.step(action)
        if transition[2]:
            return transition
        if step_delay:
            time.sleep(step_delay)

    # Malmo key commands are stateful: 1 means key down and 0 means key up.
    # Release every macro before waiting for the next remote-model response.
    transition = env.step(env.action_space.no_op())
    if step_delay:
        time.sleep(step_delay)
    return transition

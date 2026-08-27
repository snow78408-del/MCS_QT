from __future__ import annotations

from enum import Enum


class SystemState(str, Enum):
    IDLE = "IDLE"
    CONFIGURED = "CONFIGURED"
    VIDEO_READY = "VIDEO_READY"
    INITIALIZING = "INITIALIZING"
    INITIALIZED = "INITIALIZED"
    OPTIMIZING = "OPTIMIZING"
    STABILIZING = "STABILIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


ALLOWED_TRANSITIONS: dict[SystemState, frozenset[SystemState]] = {
    SystemState.IDLE: frozenset({SystemState.CONFIGURED, SystemState.ERROR}),
    SystemState.CONFIGURED: frozenset({SystemState.VIDEO_READY, SystemState.INITIALIZING, SystemState.ERROR}),
    SystemState.VIDEO_READY: frozenset({SystemState.INITIALIZING, SystemState.CONFIGURED, SystemState.ERROR}),
    SystemState.INITIALIZING: frozenset({SystemState.INITIALIZED, SystemState.ERROR, SystemState.STOPPING}),
    SystemState.INITIALIZED: frozenset({SystemState.OPTIMIZING, SystemState.RUNNING, SystemState.CONFIGURED, SystemState.STOPPING, SystemState.ERROR}),
    SystemState.OPTIMIZING: frozenset({SystemState.STABILIZING, SystemState.PAUSED, SystemState.STOPPING, SystemState.ERROR}),
    SystemState.STABILIZING: frozenset({SystemState.RUNNING, SystemState.PAUSED, SystemState.STOPPING, SystemState.ERROR}),
    SystemState.RUNNING: frozenset({SystemState.PAUSED, SystemState.STOPPING, SystemState.ERROR}),
    SystemState.PAUSED: frozenset({SystemState.INITIALIZED, SystemState.STOPPING, SystemState.ERROR}),
    SystemState.STOPPING: frozenset({SystemState.STOPPED, SystemState.ERROR}),
    SystemState.STOPPED: frozenset({SystemState.CONFIGURED, SystemState.INITIALIZING, SystemState.OPTIMIZING, SystemState.RUNNING, SystemState.ERROR}),
    SystemState.ERROR: frozenset({SystemState.CONFIGURED, SystemState.INITIALIZING, SystemState.STOPPING, SystemState.STOPPED}),
}


def transition_allowed(current: SystemState, target: SystemState) -> bool:
    return current == target or target in ALLOWED_TRANSITIONS[current]


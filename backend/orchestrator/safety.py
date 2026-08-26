from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class SafetyState(str, Enum):
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    PAUSED_LOCKED = "PAUSED_LOCKED"
    STOPPING = "STOPPING"
    ESTOP_LATCHED = "ESTOP_LATCHED"
    FAULT_LATCHED = "FAULT_LATCHED"


@dataclass(frozen=True, slots=True)
class RunToken:
    session_id: str
    generation: int


@dataclass(frozen=True, slots=True)
class SafetySnapshot:
    state: SafetyState
    session_id: str
    generation: int
    reason: str
    stop_verified: bool


class SafetySupervisor:
    """Fail-closed pump guard independent from the UI/control-loop thread.

    A trip invalidates the current token before any pump I/O.  The supervisor
    then retries the supplied stop callback on its own thread until a verified
    stop is reported.  A latched trip can only be cleared while the pump is
    verified stopped.
    """

    def __init__(
        self,
        stop_pump: Callable[[], bool],
        *,
        logger: Callable[[str], None] | None = None,
        retry_interval_s: float = 0.25,
    ) -> None:
        self._stop_pump = stop_pump
        self._log = logger or (lambda _message: None)
        self._retry_interval_s = max(0.05, float(retry_interval_s))
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._state = SafetyState.DISARMED
        self._session_id = ""
        self._generation = 0
        self._reason = ""
        self._stop_verified = True
        self._heartbeat_deadline: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="pump-safety-supervisor",
            daemon=True,
        )
        self._thread.start()

    def begin_session(self) -> RunToken:
        with self._lock:
            if self._state in {SafetyState.ESTOP_LATCHED, SafetyState.FAULT_LATCHED}:
                raise RuntimeError(f"safety fault is latched: {self._reason}")
            if not self._stop_verified:
                raise RuntimeError("pump stop has not been verified")
            self._generation += 1
            self._session_id = uuid.uuid4().hex
            self._state = SafetyState.DISARMED
            self._reason = ""
            return self._token_locked()

    def arm(self, token: RunToken, *, heartbeat_timeout_s: float) -> None:
        with self._lock:
            self._require_token_locked(token)
            if not self._stop_verified:
                raise RuntimeError("cannot arm before a verified pump stop")
            self._state = SafetyState.ARMED
            self._stop_verified = False
            self._heartbeat_deadline = time.monotonic() + max(1.0, float(heartbeat_timeout_s))
            self._wake.set()

    def heartbeat(self, token: RunToken, *, timeout_s: float) -> bool:
        with self._lock:
            if not self._permits_locked(token):
                return False
            self._heartbeat_deadline = time.monotonic() + max(1.0, float(timeout_s))
            return True

    def permits(self, token: RunToken | None) -> bool:
        with self._lock:
            return bool(token is not None and self._permits_locked(token))

    def pause(self, reason: str = "operator pause") -> RunToken:
        with self._lock:
            self._invalidate_locked(SafetyState.PAUSED_LOCKED, reason)
            token = self._token_locked()
            self._wake.set()
            return token

    def request_stop(self, reason: str = "operator stop") -> RunToken:
        with self._lock:
            self._invalidate_locked(SafetyState.STOPPING, reason)
            token = self._token_locked()
            self._wake.set()
            return token

    def trip(self, reason: str, *, emergency: bool = False) -> RunToken:
        state = SafetyState.ESTOP_LATCHED if emergency else SafetyState.FAULT_LATCHED
        with self._lock:
            self._invalidate_locked(state, reason)
            token = self._token_locked()
            self._wake.set()
        self._log(f"[SAFETY][TRIP] state={state.value} reason={reason}")
        return token

    def confirm_stopped(self) -> None:
        with self._lock:
            self._stop_verified = True
            self._heartbeat_deadline = None
            if self._state == SafetyState.STOPPING:
                self._state = SafetyState.DISARMED

    def reset_latch(self) -> None:
        with self._lock:
            if self._state not in {SafetyState.ESTOP_LATCHED, SafetyState.FAULT_LATCHED}:
                return
            if not self._stop_verified:
                raise RuntimeError("cannot reset safety latch before pump stop is verified")
            self._generation += 1
            self._session_id = ""
            self._state = SafetyState.DISARMED
            self._reason = ""

    def snapshot(self) -> SafetySnapshot:
        with self._lock:
            return SafetySnapshot(
                state=self._state,
                session_id=self._session_id,
                generation=self._generation,
                reason=self._reason,
                stop_verified=self._stop_verified,
            )

    def shutdown(self) -> None:
        self._shutdown.set()
        self._wake.set()

    def _run(self) -> None:
        while not self._shutdown.is_set():
            self._wake.wait(timeout=self._retry_interval_s)
            self._wake.clear()
            with self._lock:
                deadline = self._heartbeat_deadline
                state = self._state
                stop_verified = self._stop_verified
            if state == SafetyState.ARMED and deadline is not None and time.monotonic() > deadline:
                self.trip("control-loop heartbeat expired")
                state = SafetyState.FAULT_LATCHED
            if state not in {
                SafetyState.PAUSED_LOCKED,
                SafetyState.STOPPING,
                SafetyState.ESTOP_LATCHED,
                SafetyState.FAULT_LATCHED,
            }:
                continue
            # PAUSED_LOCKED remains stop-required until an explicit new run.
            # Once verified, do not flood the serial line with stop writes.
            if stop_verified:
                continue
            try:
                stopped = bool(self._stop_pump())
            except Exception as exc:  # pragma: no cover - hardware boundary
                stopped = False
                self._log(f"[SAFETY][STOP][ERROR] {exc}")
            if stopped:
                self.confirm_stopped()
                self._log("[SAFETY][STOP][VERIFIED]")
            else:
                self._log("[SAFETY][STOP][RETRY] pump stop not verified")

    def _invalidate_locked(self, state: SafetyState, reason: str) -> None:
        self._generation += 1
        self._state = state
        self._reason = str(reason or state.value)
        self._stop_verified = False
        self._heartbeat_deadline = None

    def _token_locked(self) -> RunToken:
        return RunToken(self._session_id, self._generation)

    def _require_token_locked(self, token: RunToken) -> None:
        if token != self._token_locked():
            raise RuntimeError("stale safety run token")

    def _permits_locked(self, token: RunToken) -> bool:
        return self._state == SafetyState.ARMED and token == self._token_locked()

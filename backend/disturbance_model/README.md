# Disturbance Model

This package records synchronized control-cycle samples, stores them in SQLite
without blocking the PID loop, trains a lightweight disturbance model in a
background thread, and exposes a thread-safe prediction for PID feedforward.

The frontend must not call this package directly. The orchestrator owns sample
collection and prediction handoff.

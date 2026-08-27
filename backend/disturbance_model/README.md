# Disturbance Model

This package records synchronized control-cycle samples, stores them in SQLite
without blocking the PID loop, trains a lightweight disturbance model in a
background thread, and exposes a thread-safe prediction for PID feedforward.

Training is fail-closed. It requires explicit disturbance events, non-empty
experiment/chip identifiers, and independent experiment/chip groups for
train/validation/test splitting. Validation scores the predicted diameter
change directly and must beat a persistence (zero-change) baseline. The sample
pairing horizon follows the observed control cycle when that is longer than the
configured minimum horizon.

Model confidence alone never authorizes actuation. Low-weight feedforward also
requires a completed shadow window with acceptable diameter-change error and
direction accuracy; both low- and full-weight actuation require explicit
authorization and are disabled by default. Feedforward remains disabled
until `PIDConfig.feedforward_calibrated` is explicitly enabled after identifying
`feedforward_gain` in controller-output units per micrometre. Legacy unscaled
model files are rejected. Online retraining is opt-in and disabled by default.

The frontend must not call this package directly. The orchestrator owns sample
collection and prediction handoff.

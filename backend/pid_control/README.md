# backend/pid_control

本目录用于 PID 反馈部分。

## 当前控制链路

- 当前仅保留“液滴平均直径”这一条反馈链路。

## 输入

- 目标平均直径
- 当前平均直径
- 当前泵状态

## 输出

- 两个泵的目标流速：`Q1`、`Q2`

## 控制约束

- 单胞率仅识别和显示，不参与控制。
- 当样本不足时，应冻结控制输出。
- 当任一关键流速小于等于 0 时，应触发停机逻辑。
- 当前实验包络固定为 Q1 `20–200 uL/min`、Q2 `5–25 uL/min`，PID 与 BO 使用同一边界。
- 1200 波特率泵的实时闭环周期下限为 `7500 ms`。
# PID Control

`pid_control` owns all feedback-control math. The orchestrator passes a
`PIDInput` and receives a `PIDCommand`; it does not calculate PID gains or
feedforward compensation itself.

Supported modes:

- `CLASSIC_PID`: fixed base `kp/ki/kd`.
- `ADAPTIVE_PID`: bounded, interval-based `kp/ki/kd` adaptation.
- `ADAPTIVE_PID_WITH_FEEDFORWARD`: legacy combined selection, retained for old
  callers. Explicit `disturbance_feedforward_enabled` overrides that selection
  without changing the feedback mode; `target_feedforward_enabled` is separate.

Production defaults both feedforward switches to false. A validated generation
calibration still selects fixed PI (Kd=0); enabling target feedforward does not
enable the old heuristic adaptation. The target compensator inverts the local
log-diameter model along the existing Q1/Q2 allocation direction. It computes an
absolute correction relative to the saved operating-point bias, so identical
targets do not accumulate repeated flow increments. The requested target must
be reachable inside the calibrated/local pump ranges, phase gap and total-flow
limit; otherwise target feedforward is zero and bounded PI remains available.
PI and both feedforward terms share the original saturation, step limits,
anti-windup and speculative transaction commit path. Component outputs are
requested corrections; the final pump command may be limited.

The legacy disturbance gain has incompatible units with calibrated log-diameter
control and remains blocked in that path, even when the independent switch is
selected. A plant/target calibration does not authorize disturbance feedforward.
Existing calibration, causal lead, freshness and confidence gates remain in
place for the legacy compatible-unit path.

Safety behavior is internal to this package:

- invalid vision or pump communication freezes feedback,
- repeated `frame_id` is rejected,
- integral/output/feedforward are bounded,
- output rate changes are limited,
- optional `PIDInput.integration_dt` limits integration without changing the
  actual derivative time base; orchestration commits speculative PID state only
  after a successful pump transaction,
- feedforward falls back to zero when the model is stale, invalid, or low
  confidence.
- BO hands its confirmed Q1/Q2 point to `set_operating_point()`, which resets
  controller history before using that point as the PID bias.
- PID and feedforward are summed inside this package and pass through one
  actuator allocator. Saturation is reported and integral windup is prevented.
- A validated model-supplied local inverse may provide bounded low-weight
  feedforward. Static-gain feedforward still requires a measured physical pump
  response delay and a causal signal whose lead exceeds that delay plus margin.

Generation-zone plant calibration now writes schema-v3 records. Visible
channel width and out-of-plane depth are independent. Full combined-step
curves are robustly fitted to a shared FOPDT model, followed by separate
validation steps. Old schema-v1/v2 records remain loadable for audit but do not
authorize the real-time PI loop.

Calibration observation horizons are persisted with experiment settings:
`minimum_response_wait_s=30`, `low_response_wait_s=60`, and
`stability_duration_s=3`. These are operator-adjustable starting values, not
identified pump constants. A confirmed onset permits response completion before
the 30-second horizon, using a full stable window after both onset and transaction
completion. Baseline waiting has an independent optional `baseline_wait_s`
(finite, nonnegative); omitted values retain `minimum_response_wait_s` for
backward compatibility. Zero removes only the fixed wait, preserving stable
sample gates. The low-response horizon and
stable window run concurrently. Orchestration requires valid droplet counts
and elapsed capture time before classifying a stable response; full transient
curves remain available for fitting. The default is now 8 trials (6 modeling
steps plus 2 held-out validation steps), with one modeling repetition. Saved
settings remain respected; the dialog offers an explicit single-pass preset.
Validation thresholds are unchanged. Repeatability studies require additional
repetitions; fewer trials do not establish equivalent identification precision.

Identification now compares changes relative to each trial's own baseline in
linear and log space. Between-trial baseline offsets are not actuator gain;
unobserved drift during a trial is not corrected by this method. FOPDT fitting
uses a mean of per-trial median normalized errors, giving each curve equal weight.
Held-out validation does not set the baseline operating point or flow envelope.

Only explicit legacy linear mode ends identification before validation when
all combined trials lack a detected response. Default nonlinear mode completes
the planned trials first. Failed/cancelled experiments archive
completed measurements and the partial trial after safety cleanup in a non-loadable
`incomplete_*.measurements.json`. Successful audits include `diagnostics` with
response counts, repeated-direction changes, baseline spans and per-validation
predicted/observed changes, MAE and NRMSE. No trials are discarded and authorization
thresholds remain unchanged. A separate non-authorizing MPC dataset is also saved.

## Validated delivery

Default identification is `quadratic_response`: fit five scaled flow terms
(Q1, Q2, Q1 squared, Q1*Q2, Q2 squared) to per-trial diameter changes, using
feature differences between verified baseline and step flows. All training
trials participate, including observations below the per-droplet onset threshold.
Held-out validation never changes coefficients, dynamics, operating point or bounds.
Dynamic curves fit a shared first-order lag and delay; this is a local empirical
model, not a claim of a universal physical law. Local derivatives yield PI
allocation and tuning; curvature restricts the deployment range.
Rank deficiency, uncertain/zero local slopes or poor held-out prediction prevent
PI authorization but preserve a separate MPC building dataset and fitted surface
when available. Raw curves remain suitable for evaluating other black-box models.
The explicit `legacy_local_linear` mode retains the previous API behavior.

`maximum_attempts` defaults to 1; legacy values (1–5) remain readable, but the
orchestrator always executes a single round. `require_mpc_validation` defaults
to false. Desktop calibration requires PI qualification; explicit API callers
may require MPC qualification too, without automatic retries.
`accepted`/`require_accepted()` distinguish an internal
candidate from success; export rejects unaccepted candidates before writing files.

Accepted export writes the loadable JSON and `.measurements.json`, plus
`.mpc-training.json` regardless of current MPC qualification. The package records
`validated_for_mpc` without authorizing deployment. Training and held-out validation trials
remain separate with flows, times, observations, conditions and units. Readback
flows are device parameters, not independent physical measurements. This dataset
supports subsequent MPC training, not deployment of an already trained controller.
Formal JSON is written last. Failures retain non-loadable diagnostic measurements;
they do not automatically restart or become a deployable PID calibration.

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
# PID Control

`pid_control` owns all feedback-control math. The orchestrator passes a
`PIDInput` and receives a `PIDCommand`; it does not calculate PID gains or
feedforward compensation itself.

Supported modes:

- `CLASSIC_PID`: fixed base `kp/ki/kd`, no feedforward.
- `ADAPTIVE_PID`: bounded, interval-based `kp/ki/kd` adaptation.
- `ADAPTIVE_PID_WITH_FEEDFORWARD`: adaptive PID plus disturbance-model
  feedforward.

Safety behavior is internal to this package:

- invalid vision or pump communication freezes feedback,
- repeated `frame_id` is rejected,
- integral/output/feedforward are bounded,
- output rate changes are limited,
- feedforward falls back to zero when the model is stale, invalid, or low
  confidence.
- BO hands its confirmed Q1/Q2 point to `set_operating_point()`, which resets
  controller history before using that point as the PID bias.
- PID and feedforward are summed inside this package and pass through one
  actuator allocator. Saturation is reported and integral windup is prevented.
- Feedforward also requires a measured physical pump response delay and a
  causal signal whose lead time exceeds that delay plus the configured margin.

# BO + 扰动预测 + PID + 前馈控制链

## 控制权设计

泵只有一个写入口，控制权由 orchestrator 的状态机独占分配：

1. `OPTIMIZING`：只有 BO 可以下发 Q1/Q2；PID 和前馈不执行。
2. `STABILIZING`：保持 BO 找到的最优 Q1/Q2，不产生新调节量。
3. `RUNNING`：BO 不再写泵；PID 控制器在内部计算 `u_pid + u_ff`，经过同一套限幅、变化率限制、Q1/Q2 相位约束和总流量约束后，只生成一条泵命令。

因此 BO、PID、前馈不会作为三个并行写入者竞争同一执行量。快照中的 `control_owner` 可取 `BO`、`HOLD`、`PID` 或 `PID+FEEDFORWARD`；`requested_output`、`realized_output` 和 `actuator_saturated` 用于审计分配与饱和情况。

## 启动 BO

完成配置、视频准备、系统初始化和实验前检查后，在运行监控页点击“BO寻优”。必须提供：

- Q1/Q2 的实验安全边界；
- 通过实际阶跃实验测得的“泵命令到液滴响应”延迟；
- 延迟测量的不确定度或保守安全余量；
- 不短于“延迟 + 不确定度”的候选点稳定时间；
- 可追溯的延迟测量来源。

串口应答或参数回读耗时不是物理响应延迟，配置会拒绝 `serial_reply`、`device_readback`、`unmeasured` 等来源。代码调用方式：

```python
from backend.orchestrator import BayesianOptimizationConfig

orchestrator.start_optimization(
    BayesianOptimizationConfig(
        target_diameter_um=60.0,
        q1_min=40.0,
        q1_max=60.0,
        q2_min=10.0,
        q2_max=25.0,
        measured_response_delay_ms=1200.0,
        response_delay_uncertainty_ms=200.0,
        settling_time_ms=2000.0,
        response_delay_source="step experiment 2026-08-27",
    )
)
```

BO 首先测量当前已经过泵回读验证的 Q1/Q2 工作点，再使用安全可行域内的 Latin hypercube 初始点和 Matérn 5/2 高斯过程期望改进。目标直径采用容差带损失，并可同时加入频率目标、CV、无效液滴比例和移动代价。无效视觉窗口会重试；连续超过限制、候选超时、泵回读不一致或达到最大观测数仍未确认目标都会失败并触发现有安全停机路径。成功点必须在独立稳定窗口中重复确认。

## BO 到 PID 的切换

BO 成功后先进入 `STABILIZING`，继续保持最优点并等待一个稳定时间和有效视觉窗口。随后调用 `set_operating_point(q1, q2)`：清空 PID 积分、微分历史和前馈历史，并把最优 Q1/Q2 设为新的输出偏置，实现无突跳接管。

PID 的最终命令由一个分配器产生。若执行器边界、Q1/Q2 最小相差、总流量或单周期变化量导致命令不能完全实现，积分会冻结或按实际输出回算，防止 windup。

## 前馈启用条件

扰动模型自动采集并在线训练；当前装置仅授权低权重前馈。真正作用于泵还必须同时满足：

- 模型 ready、valid、置信度和时效通过；
- 至少完成 20 次影子验证，变化方向与误差门槛通过；
- 模型能从局部 Q1/Q2 灵敏度反演出有限的补偿命令；若模型不能反演，则仍要求独立标定的静态前馈增益；
- 使用静态增益路径时，泵物理响应延迟、真实提前信号和预测时域必须全部满足因果性检查。

通过 `set_disturbance_context()` 提供当前事件的提前信号：

```python
orchestrator.set_disturbance_context(
    disturbance_name="scheduled pressure pulse",
    leading_signal_available=True,
    signal_lead_time_ms=1800.0,
    leading_signal_name="upstream pressure trigger",
)
```

提前量从信号登记时刻开始递减，耗尽后信号自动失效。通过影子验证的模型局部反演命令可以作为模型预测补偿，不依赖静态增益；其他情况仍要求真实提前信号。前馈最大权限还受低权重阶段、变化率和 `feedforward_max_output_fraction` 共同限制。

## 实时控制周期

实时控制周期由用户填写，允许范围为 7500–30000 ms；7500 ms 只是当前泵在 1200 波特率下完成双通道写入和回读所需约 6 秒后的安全下限，不是固定周期。系统保留用户填写的准确数值，低于下限时明确拒绝配置，不会静默改写成 7500 ms。视觉统计窗口、PID 调度和界面配置使用该周期；硬件事务超时会跳过过期槽位，不会连续补跑。

## 现场使用建议

先完成响应延迟和控制方向辨识，再做 BO。低权重前馈仍需模型训练和影子验证自动通过；完整前馈保持未授权。暂停或停止会立即使运行令牌失效；任何正在途中的泵命令都不能跨生命周期继续生效。

# Q2正式起点与切换前预测检查点

新增检查点只用于带 `stage_gates` 的Q2固定两段计划。旧计划保留原计时行为；
预检会输出 `stage_gates_enabled`、`readiness_mode` 和 `prediction_required`，不能把旧计划误当已启用。
当天芯片、相机、补液、当前参考及设备生命周期条件仍须按原流程满足；默认不连接硬件。

## 1. 现场确认第一段起点

启泵后标记 `awaiting_operator_ready`，相机继续存原图、累计用液。
程序生成一次性 `readiness_request.json`，只接受本次请求、当前会话的明确人工确认。
**不运行“每帧识别通过才开始”的门槛**。操作者确认双壁及完整滴两端可见后，
才记录第一段观测起点，完整采集300秒。第二段从事务后立即观测，不再次等液滴恢复。

默认30秒提醒而非失败；等待中不断重新检查够不够完整两段、60秒切换预留和10μL余量。
最多等待90秒，余量不足可更早结束；超时沿既有最终停泵路径退出，保留原图和计量，
没有自动重启、无限等待或清零。暂停本次有界实验意味着安全结束，不从半截自动续跑。

只有收到现场人员明确确认后才执行以下文件确认命令；不可根据算法结果自动代填：

```powershell
.venv/Scripts/python.exe tools/q2_gate_control.py --session <当前采集目录> --action ready --operator <人员标识> --walls-visible --complete-droplets
.venv/Scripts/python.exe tools/q2_gate_control.py --session <当前采集目录> --action stop
.venv/Scripts/python.exe tools/q2_gate_control.py --session <当前采集目录> --action pause
```

辅助命令只写控制文件，不连接设备；`stop`/`pause`走现有STOP检查，须等待会话确认停泵。
过期、旧会话或缺少明确确认的响应不能启动正式计时。

## 2. 第二段切换前保存预测

验证计划必须启用预测保存，训练及对照可不启用。模型包在设备连接前核验并固定字节副本，
核对芯片、会话组、相机参数、当前参考、视觉源码及基线计算源码哈希。验证块不能列为训练块。
目录写入和模型双向数值计算也先做预检，避免带流时才发现明显错误。

第一段结束后，固定当时已经落盘的原始清单快照。在独立进程计算最后60秒：
每0.2秒格首张，10秒箱至少40张原图/10格完整候选，6箱至少5箱合格。
使用这些箱的中位数作初值；事前趋势只使用这6个前段箱。没有有效候选不填零。
这条基线方法名为 `tail60_fresh_pipeline_5hz_v1`：每次基线创建新管线，
不沿用其他块的时序识别状态。**训练模型的初值也必须按同一方法计算；旧模型不能直接冒用。**

保存一阶/FIR模型预测、保持值、事前趋势的后300秒曲线及SHA256，读取复核后才允许停启改流量。
预测保存后仍再次检查停止请求、预算和文件哈希。观测起点随后绑定到独立日志，预测数值不变。
不判断预测“准不准”才允许切换；程序从不读取第二段观测来决定是否启动它。

默认计算上限20秒（可显式设置，不得超过30秒），计算时继续采集和生命周期检查，
第一段结束至第二段开始的计算加事务总预留60秒。计算超时、无足够基线、写入失败、
身份/源码不匹配或停止请求都不发送目标命令，并由原 finally 清理路径核验停泵。
子进程终止后回收，不遗留后台计算；已采第一段不会删除。

支持固定Q1=70、Q2=20/25，输出条带像素。暂不支持其他流量或任意外部计算命令。
登记几何和原有静态参考保护仍启用；该检查点不授予物理测量或MPC控制资格。

## 配置与冻结文件

将下列对象放入完整采集计划；训练/对照的 role 改为 `training`/`sham` 并将 prediction.required 设为false。
验证计划仍需顶层 `study_id`，以及原有 `session_id`、现场条件和设备配置。

```json
{
  "stage_gates": {
    "schema_version": 1,
    "role": "validation",
    "readiness": {"mode": "operator", "remind_after_s": 30, "maximum_wait_s": 90},
    "prediction": {
      "required": true,
      "timeout_s": 20,
      "bundle_path": "<已冻结模型文件>",
      "bundle_sha256": "<完整SHA256>"
    }
  }
}
```

拟合完成后的候选JSON需包含 `baseline_method`（上述同口径名称）、`training_block_ids`、
`primary_model` 和 `models`。一阶模型为 `{kind: fopdt, gain, tau_s, delay_s}`；
FIR为 `{kind: fir, knots_s, coefficients}`，结点由0起严格递增，系数个数比结点少1。
参数均为对数相对滴长对Q2对数变化的响应。冻结工具只打包已拟合参数，不进行拟合或自动验收：

```powershell
.venv/Scripts/python.exe tools/freeze_q2_prediction_bundle.py --plan <验证计划.json> --candidate <同口径拟合候选.json> --output <新冻结包.json>
```

将输出路径及打印的哈希填入两份验证计划；两个验证使用同一个冻结包，不能看完第一块后重拟合。
工具拒绝覆盖旧冻结文件。若更新模型、参考、相机设置或测量源码，须重新冻结并声明验证状态变化。

## 审计与验证

保留 readiness_request/confirmation/accepted、frozen_prediction_bundle、原始清单快照、
prediction_request/candidate、prospective_prediction、prediction_seal 和 prediction_observation_binding。
曲线封存发生在改流量事务之前；实际命令起点和回读时间沿用原日志，缺失仍记缺失。

无设备测试覆盖晚确认、提醒、超时、余量不足、旧确认、保存先于命令、子进程失败/取消/超时、
停止/暂停、键盘中断、磁盘错误、篡改预测、启动前模型失配、完整finally停泵路径，以及不读后段数据。
2026-10-25项目虚拟环境验证：相关测试50项通过；完整回归1387项、6个子测试通过（102.40秒），未连接设备。
使用旧F1末60秒300张原图实跑独立预测进程约4.64秒，6箱有效；该结果只验证运行和耗时，
使用的是测试参数与未验收旧几何，不是新模型预测验收，也不保证现场耗时。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_q2_capture_gates.py tests/test_q2_lab_capture.py tests/test_next_day_capture_profiles.py -q
.venv/Scripts/python.exe -m pytest -q
```

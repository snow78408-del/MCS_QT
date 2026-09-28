# 追溯收尾复审

## 已通过

本轮 Codex 实际复跑 tests/test_provenance.py、tests/test_capture_result_layers.py、tests/test_transient_capture_lifecycle.py：87 passed in 9.26s。第一次执行因隔离父目录未建立而出现测试环境错误，建立目录后复跑通过；不计为产品缺陷。986 passed + 6 subtests 为实施方完整测试记录，本轮未重复全套。

独立检查结果：
- 修改真实记录的预期哈希后，核验 CLI 返回退出码 1、ok=False、matched=False，已正确传播失败。
- 新交付回放第 40 行可从原始录像解码索引 6701，硬件帧号 6835，内容哈希匹配，退出码 0。
- generation_edge_mad_multiplier 从 3 改为 13 后，live_effective_config 的配置指纹改变；完整检测参数序列化修复成立。
- 现场 CLI 已移除 --no-provenance，缺规格的前置拒绝及无设备测试接口有定向回归通过。

这些修复接受，不要求重新开展大范围修复或重复全部录像回放。

## 剩余一项 [P1]：固定墙线绑定辅助功能没有接到诊断入口

位置：tools/run_channel_validation_live.py:245–256；tools/plant_flow_transient_capture.py:live_effective_config。

诊断入口从 output/current-wall-fullspan-20260923/proposal.json 读取 walls，交给 MeasurementSink；但随后调用 live_effective_config 时只传 wall_source='visually_selected_session_proposal'，未传 wall_binding_block。

live_effective_config 默认分支仅在 wall_source=='reused_proposal' 且 apparatus.wall_proposal 存在时读取提议文件。当前入口不满足这两个条件，实际得到的 wall_binding 只有 source_label，没有墙线值、wall_lines_sha256 或 proposal_content_sha256。

因此固定墙线内容改变时，本入口的配置指纹仍不反映该变化。与逐帧自动定位回放只记来源标签的情形不同：这里确实有复用的固定墙线，必须绑定。

最小整改：
1. 在实际诊断入口，把同一份供 MeasurementSink 使用的墙线值传入 wall_binding，生成内容指纹，并传给 live_effective_config。若附提议文件指纹，需保证与已加载墙线来自同一次内容读取，避免读两次期间发生变化。
2. 通过入口配置组装路径测试，而不只测试 wall_binding 辅助函数：两套不同墙线必须得到不同配置指纹，记录中的墙线必须与 sink 使用值一致；缺绑定的固定提议模式应在设备动作前拒绝。
3. 不重写历史追溯包。用新建的无硬件示例验证即可，不需要再跑全部 2797 帧。

此项修复前，结论为“检测参数与核验/阻断已通过；固定墙线现场入口追溯尚未完整”。

## 两处非阻塞文档/输出整理

- replay_results.md 的逐段表存在旧数字：例如 b1_s5 写 78/288/32，handoff 对应 66/323/9。直接汇总本轮七个 *_frames.ndjson 得到 2797 帧、ambiguous 1378 / pending_motion 1231 / rejected 188，与 handoff 汇总一致。请从同一份行记录重新生成表格，避免混用旧派生产物。
- 错误哈希时 CLI 的 levels_achieved 仍包含 content_match，虽然该级 ok=False 且总体已正确失败。宜改名 levels_checked，或只把通过级别放入 achieved；不影响本轮已确认的退出码修复。

## 实验边界

未连接设备、未改产品代码或原始记录。真实录像定位成功数仍为 0，物理标尺未验证；本轮软件验收不授权现场启动，不产生物理稳态时间或 MPC 参数。完成上面唯一接线项后，可结束这轮追溯收尾，把后续工作集中在真实画面的目标通道选择与候选竞争诊断。

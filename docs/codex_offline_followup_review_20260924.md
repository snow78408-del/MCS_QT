# 第二次修复复审：主要修复通过，追溯剩三项收尾

## 已核验与边界

Codex 本轮实际执行五组定向测试（measurement_entry_consistency、localization_mode_retention、capture_result_layers、provenance、current_frame_detection_gate）：71 passed in 7.75s，使用项目虚拟环境与独立 MCS_DATA_DIR。没有重跑全套；967 passed + 6 subtests 是实施方的执行记录。

核对实时管线已给 detector 传本帧横向跨度及显式轴向；缺验收证据不再默认通过。软硬件帧号分属不同计数器，取消二者相等要求是合理修正，应继续保留同帧引用及采集身份验证。

实际使用核验 CLI 从新回放的第 40 行找回原录像帧，resolved=True、matched=True。目视查看 b1_s1_70_20_f0006661_annotated.jpg，失败帧已画出候选线，不再只有状态文字。这只能支持候选证据已可查看，不能直接确定竞争线的物理来源。

上述进展可以接受，不要求重复返工。真实录像定位仍未通过，不能发布物理稳态时间或 MPC 参数，也不放行正式阶跃实验。

本轮独立反例：output/codex-followup-review-20260924/probe.py；结果 probe_results.json。只读取原录像，将修改过的反例行写到新的审核目录。未改原记录、未改产品代码、未连接设备。

## 剩余三项（限定整改范围）

### 1. [P1] 内容校验失败未传播到 CLI 总结果

位置：tools/verify_provenance.py，解析 --row 后只检查 outcome.resolved，未检查 matched。

复现：同一条真实记录，原哈希时 exit_code=0 / ok=True / resolved=True / matched=True；只把 content_sha256 改成 64 个 0 后，仍 exit_code=0 / ok=True / resolved=True，但 matched=False。

因此底层已经发现内容不匹配，顶层却继续报成功，自动验收会误收。

修复：明确检查请求的级别。--decode 按目前声明包含解码及内容比对，则匹配失败或缺少预期哈希必须使总体 ok=False、退出非零；各级结果仍分别保留，避免把“可解码”写成“内容匹配”。补 CLI 子进程回归：正确哈希、错误哈希、缺哈希、无法解码四种情况。不要只测试底层 resolve_source_frame。

### 2. [P1] 生效配置只保存部分参数，改检测行为可以不改指纹

位置：tools/plant_flow_transient_capture.py:live_effective_config。检测参数仍是手写字段白名单；定位参数也只保存部分字段。

复现：DetectorConfig.generation_edge_mad_multiplier 从 3.0 改为 13.0，经 live_effective_config 得到的配置指纹完全相同。这是检测器实际使用的边缘门槛参数，不能遗漏。

修复：序列化实际生效配置对象的完整行为参数，避免手写少数字段。定位器同理；没有该组件的入口应标“未使用”，不要以默认实例冒充实际生效对象。复用管壁的入口还需绑定实际墙线值/几何版本或提议文件内容指纹，仅写来源名称不能复现扶正。记录运行期覆盖值与配置版本变化，或明确禁止改变。

补回归：改变上述边缘参数、任一实际使用的轮廓参数、定位阈值或墙线，都应改变对应生效配置/几何指纹；配置与测量行引用一致。不要重写旧记录来补齐历史配置。

### 3. [P1] 现场入口可以显式绕开强制追溯

位置：tools/run_r1_front4_live.py 的 --no-provenance；它把 provenance_spec 设为 None 后照常创建并运行现场会话。LiveCaptureSession._prepare_provenance 对 None 直接返回，且称其为“离线模式”，但此处确实是现场入口。

这是代码路径核对，未启动该命令或任何设备。它与“没有可核验的运行记录，就不允许动设备”的契约矛盾。

修复：现场入口禁止跳过追溯。若纯离线单元测试需要关闭，应通过明确的无设备测试接口，不能暴露为真机 CLI 绕过参数。缺追溯规格与追溯写入失败均在设备动作前拒绝。补 CLI/会话连接测试，证明真实入口参数不能绕过，使用设备桩断言无连接、启动或流量指令。

## 本轮验收结论与下一步

- 测量入口、模式保持、缺测量证据的验收处理等修复接受本轮定向证据。
- 追溯暂不整体验收，仅要求完成以上三处具体收尾。
- 现有真实录像的定位仍不通过；候选图已可检查，接下来才能做目标通道选择与候选竞争的离线诊断，不能把当前失败归因为已确定的另一条真实管道。
- 不要求重跑全部 2797 帧来证明这三处改动；复用同一原帧做核验正反例即可。涉及入口阻断的改动完成后按项目要求运行定向与一次完整测试。

交付建议使用新目录 output/deepseek-provenance-closeout-20260924/，含三处修复前后结果、实际测试日志及清晰的限制说明。未完成真实定位前，不启动新现场轮次，不产出稳态模型。

# 旧数据复用实跑结果与固定样本包

日期：2026-09-20。本轮只进行离线分析，未连接泵或相机，未更改业务代码或设备配置。

## 实际执行结果

| 项目 | 结果 | 可用范围 |
| --- | --- | --- |
| 参考录像 | 63 帧；壁宽 26.931 px；周期 166.451 px；3 个速度窗全部 ALIAS_UNRESOLVED | 像素几何、检测与混叠回归 |
| 长录像 | 1500 帧；前 300 帧拟合壁宽 27.032 px；全记录周期 171.945 px；8 窗全部 ALIAS_UNRESOLVED | 逐窗运动候选、非稳态算法检查 |
| 相位平均 | 参考/长录保留 63/63、1500/1500 帧；强度峰峰值 25.291/20.741 | 轮廓检查；亮度占空比不作相别/体积真值 |
| 历史重拟合 | 37 条建模 + 4 条验证；常数模型 CV RMSE 4.190 µm，最优线性流量模型约 4.197 µm | 质量与可辨识性诊断；仍不能作控制增益 |
| 相关回归测试 | 59 passed in 13.54s | 软件回归行为；不是现场实验验证 |

两段输入 SHA-256 和空间周期与 9 月 19 日修复后报告逐项一致。没有用重复运行制造新的独立物理证据。

## 固定回归候选样本

位置：`output/data-reuse-20260920/regression-samples/`。

每段录像分前、中、后三段；每段在均匀抽样的帧中计算条带轴向灰度梯度，选第 25 百分位及最高分各一帧，共 12 帧。这样同时保留相对困难和较清晰候选，不只展示高分图。该指标是选样代理，不是清晰度真值，更不是检测准确率。

| 录像 | 前段 | 中段 | 后段 |
| --- | --- | --- | --- |
| 参考 | 13、7 | 37、39 | 57、52 |
| 长录 | 264、300 | 632、740 | 1360、1204 |

帧号为从 0 开始的原序列索引，每格依次为较低梯度和最高梯度样本。

- `manifest.json`：源文件哈希、帧号、选择规则、采样变换、检测器配置和代码哈希、算法候选及人工标注状态。
- `*_frame_raw.png`：原始灰度整帧。
- `*_strip_raw.png`：沿拟合管道取样的原始灰度条带，变换已记录。
- `*_label_view.png`：增强显示及条带列坐标，用于边界标注；不包含算法边界叠加。
- `contact_sheet.png`：12 帧一览，供定位和目视筛查。

本次检测器给出 22 个候选，未将它们充当真值；`manual_label_status=pending`，人工边界数组为空。已目视检查一览图，边缘仍有低对比度和模糊，不能依据图像外观宣称像素级精度。

检测器为复现历史行为仍传入旧标尺参数 1.8511，完整配置已存入 manifest；这不是独立标尺，本样本包只报告像素输出。候选等效直径也依赖内部截面模型，不能视为直接测得的球径。

后续比较算法时固定这些源文件与帧号。先独立标注完整液滴的成对弯月面及不确定区间，再与同帧候选匹配；同时统计漏检、误检和长度误差。不可标注的帧单独报告。没有人工标签前只比较软件行为，不报告准确率。

## 实际命令

以下输出目录已存在，复跑请换新目录，避免覆盖本轮证据：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/review_periodic_flow.py --video output/crops/ref_video.npy --rate 101.613 --time-source reference_report_rate_not_independently_verified --output output/data-reuse-20260920/reference --window 30 --step 30
.\.venv\Scripts\python.exe -X utf8 tools/review_periodic_flow.py --video output/longcap-20260918-210955/stack.npy --rate 317.924590068 --time-source saved_host_receive_median_dt_no_frame_timestamps --output output/data-reuse-20260920/long --window 200 --step 200
.\.venv\Scripts\python.exe -X utf8 tools/fit_plant_history_model.py --history D:/MCS_QT_Data/calibrations --output output/data-reuse-20260920/history --no-figure --quiet
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_offline_analysis.py tests/test_flow_locking.py tests/test_plug_geometry.py tests/test_plant_geometry_solver.py -q -p no:cacheprovider
```

测试日志：`output/data-reuse-20260920/tests.log`。样本包通过虚拟环境内 NumPy/OpenCV 离线生成，12 帧原图可解码、帧号唯一且与源图一致；核查结果见结果根目录 `verification.json`。

本轮没有代码变更，未重跑完整 pytest。此前 525 项全量通过不冒充为本轮结果。

## 交给 DeepSeek 的下一步

使用 [实体实验执行任务书](deepseek_bench_execution_plan_20260920.md) 和 [会话待填模板](bench_session_template_20260920.json)。先完成采集器缺口修复与离线失败路径测试，并整理具体待填运行表；现场信息齐备后再做实体采集。旧数据已经可用于回归，真实流量与控制参数仍需新实验。

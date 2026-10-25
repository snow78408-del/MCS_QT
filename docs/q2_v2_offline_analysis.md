# Q2 v2 离线主分析准备

本入口用于 [2026-10-23 六块方案](experiment_20261023_v2.md) 的原始 PNG 记录。
它只读已完成并核验停泵的采集目录，不导入采集入口，不连接相机或泵。
它不拟合模型、不接触盲复核对应表、不授予留出通过或 MPC 资格。

## 输入与运行

需要当块原始 `frame_manifest.ndjson`、`session_summary.json`、
`registered_geometry.json`，以及测量源码 SHA256 契约。不得给它旧视频分析的
自动选墙结果，也不得拿旧图替代当前几何。

以下为完成 SA 采集后的示例；当前没有真实采集数据时不要执行此示例：

```powershell
& E:/MCS_QT/.venv/Scripts/python.exe E:/MCS_QT/tools/analyze_q2_v2.py `
  --session E:/MCS_QT_Data/experiments/next_day_design_v2_20261022/captures/SA_20261023_run01 `
  --contract E:/MCS_QT_Data/experiments/next_day_design_v2_20261022/measurement_contract.json `
  --output E:/MCS_QT_Data/experiments/next_day_design_v2_20261022/analysis/SA_run01
```

输出目录必须是新目录，已有结果不会被覆盖。源码与契约不一致时终止。
`geometry_sha256` 非空时还必须与该块注册文件一致。不同块独立注册的几何应各自
保存契约副本，不覆盖前一块契约。工具不会把空白相机设置或现场字段自动补齐。
相机设置哈希只登记契约声明，尚不替代对实际相机设置记录的核对。

可选 `--review <人工标签.json>` 接受独立物理状态：

```json
{
  "frames": {
    "frames/SA_pre_sw00000001_hw00000001.png": {
      "state": "filament",
      "reviewer": "operator",
      "note": "两端未完整断开"
    }
  }
}
```

状态允许 `complete_droplet`、`gap`、`filament`、`bubble`、`uncertain`。
未标注的原图记 `unreviewed`；没有候选记 `no_complete_candidate`，不擅自判断
物理状态。除完整滴外的人工状态保留候选诊断，但不进入该帧的长度统计。
标签允许只覆盖部分帧，因此所有输出仍明确标记测量未验收。
这份逐帧状态文件与24份重复样本的盲复核答案是两个不同输入。

## 固定统计口径

- 根据每段观测起点的主机单调时间，划分1500个半开区间 `[0, 0.2)`、
  `[0.2, 0.4)` 等。每格只取原始清单中的首张图，不依赖采集器的 `grid_slot`
  或阶段名称，不插值。首张图损坏也不换成同格后续图。
- 一帧多个完整候选先取长度中位数。10秒内50格至少40格原图可读、至少10格有
  可测完整候选，才输出这些逐帧中位数的箱中位数。未达到条件记 `null`，不填零。
- 使用当前规则的 `.02` 候选支撑及源码中 `.03` 有界收紧，学习候选关闭。
  主几何固定为当块注册管壁；另将两条壁的传感器纵坐标同时平移±1像素。
  三种几何共用同一批首帧，不能按平滑程度选择主结果。
- 输出主指标只有条带像素长度，不产生未经标定的 μm 或独立液滴计数。
  当前入口未实现次要1 Hz / 20秒均值分析。
- Δ使用前段最后60秒与后段60–180秒箱中位数之差。三种几何只在共同合格箱上
  计算该差和 `U_geo`；同时列出所有共同箱号。若6个前箱或12个后箱不完整，
  差值仅作描述，工程分辨检查资格为假。
- 后段24/30箱、末120秒10/12箱只是覆盖率条件；满足条件不会自动判留出通过。

## 输出与追溯

`grid_main.ndjson` 和两个平移诊断文件逐格保留缺失、原始帧号、主机时间、
原始 ticks、图像引用及哈希、人工状态、完整候选长度和帧中位数。
设备 ticks 不换算成秒。`analysis.json` 保存三种几何的箱值、效应统计、
覆盖率、源码与输入哈希，以及当前能取得的事务记录。

日志没有提供的停泵确认、参数核验完成、重启确认、实际参数写入和停机时长保持
缺失。参数回读时间不能用来补造这些时间。采集入口也已取消缺失回读时间的
当前时钟回填。

拟合、模型冻结、事前预测文件及两个基线、共同箱评分仍需按方案另行完成；
本工具的分箱结果只能作为它们的输入，不能替代这些证据。

## 无设备回归

```powershell
& E:/MCS_QT/.venv/Scripts/python.exe -m pytest `
  E:/MCS_QT/tests/test_q2_v2_analysis.py `
  E:/MCS_QT/tests/test_next_day_capture_profiles.py `
  E:/MCS_QT/tests/test_q2_lab_capture.py -q
```

回归使用模拟时钟、人工构造的原图清单与合成 Mono8 图像，不要求相机、串口或显示器。

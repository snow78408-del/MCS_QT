# 周期液滴离线复算：相位、速度候选与验收边界

本轮落实 [结果审计](deepseek_results_audit_20260919.md) 后的第一步：修复离线分析并重新处理现有录像。未连接相机或泵，未修改在线检测、控制增益或设备配置。

## 改动

- `backend/vision/offline_analysis.py`：先把每帧实际采样按空间周期折叠，再独立进行循环相位对齐。不再累积无限位移，不做越界端点填充，不从相位差自动推断真实速度。静态背景先移除；输出逐帧相关性、保留/拒绝帧、采样支持和亮度轮廓。
- 同模块的 `velocity_windows()`：逐窗估计周期与位移，至少要求两组足够强且一致的帧跨度；保留低质量窗口及拒绝原因。比较前后半窗以识别剧烈变化，稳定候选只是局部指标，不是完整系统达到稳态的证明。
- `screen_velocity()`：方向先验本身不能确定混叠阶数。无独立位移上界时输出 `ALIAS_UNRESOLVED`，实际速度/通过频率字段为空；提供带来源的上界后，只有唯一分支才输出 `CONDITIONAL`，仍不能授权控制。
- 旧 `tools/plant_geometry_solver.py` 保留为条件情景计算器，JSON 新增 `validity=UNVALIDATED_SCENARIO` 及控制禁止标记，文本输出开头明确提示相别、真实速度未确认；其旧相别候选字段不是实测判定。
- `tools/review_periodic_flow.py`：正式可复现入口，读取 uint8 灰度 `.npy`，输出来源哈希、完整 JSON、简表及三张抽样帧的原始裁剪/带刻度管壁叠加图。只写新目录，避免覆盖旧结果。
- 旧本地入口 `output/phase_average_probe.py` 已委托新实现；旧文件保存在 `output/deepseek-audit-20260919/phase_average_probe_before_fix.py`。审计脚本固定读取旧快照，避免修复后审计证据变样。

不把异常符号简单翻转或平滑：周期信号的 `u` 与 `u+n·pitch` 都可能解释画面。默认展示代表分支及有限候选示例，并明确示例不是完整候选集合；只有独立上界内的枚举才是穷尽的。

## 本次数据结果

| 项目 | 参考录像 | 长录像 |
| --- | --- | --- |
| 帧数 | 63 | 1500 |
| 拟合管壁间距 | 26.931 px | 27.032 px（前 300 帧拟合） |
| 全记录空间周期 | 166.451 px | 171.945 px |
| 相位平均保留帧 | 63/63 | 1500/1500 |
| 修复后轮廓强度峰峰值 | 25.291 | 20.741 |
| 亮度占空比 | 0.518 | 0.581 |
| 速度窗口 | 3 | 8 |
| 默认状态 | 全部 ALIAS_UNRESOLVED | 全部 ALIAS_UNRESOLVED |

参考窗口代表位移约 -47.34、-49.38、-49.53 px/帧；长录代表分支约 -55.74→-84.30 px/帧。保留尾部窗口时会与上一窗重叠，不能把这些窗口当成独立重复实验。局部周期随窗变化，因此没有用单一固定周期硬解末段符号。

旧相位脚本的长录峰峰值为 0.13、参考为 4.20，但它平均的是大量越界填充点；新数值来自周期折叠的真实样本，二者不是同一个统计量，不能把变化比例当作精度提升倍数。亮度占空比仍不是分散相体积比例，不能据它判定相别。

输出：

- `output/offline-flow-review-20260919/reference/report.md`、`report.json`
- `output/offline-flow-review-20260919/long/report.md`、`report.json`
- 两目录内的 `frame_*_raw.png` 为未增强裁剪；`frame_*_walls.png` 为两倍放大、增强显示和管壁叠加，顶部为原图列坐标。

抽样图已目视检查：管壁覆盖位置基本吻合，但弯月面低对比度、边缘模糊，不能据本次目视检查生成高精度长度真值。`annotations` 中保留空标注和 `pending` 状态，未冒充人工标注通过。后续应逐帧成对标注弯月面坐标及不确定区间，与同一帧检测结果比较，不再把前段长度中位数与末段速度混用。

## 使用

所有命令使用项目虚拟环境。`--output` 必须是尚不存在的目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/review_periodic_flow.py --video output/crops/ref_video.npy --rate 101.613 --time-source reference_report_rate_not_independently_verified --output output/new-reference-review --window 30 --step 30
.\.venv\Scripts\python.exe -X utf8 tools/review_periodic_flow.py --video output/longcap-20260918-210955/stack.npy --rate 317.924590068 --time-source saved_host_receive_median_dt_no_frame_timestamps --output output/new-long-review --window 200 --step 200
```

默认 `--direction unknown`。`--offset 13`、`--min-pitch 120 --max-pitch 230` 是这批观测窗的分析设置，不是通用芯片常数；其他录像必须检查采样线位置和周期搜索范围。相关峰强度是按重叠长度归一化的自相关数值，可能略大于 1，不是概率。

如提供 `--max-displacement`，单位是 **px/帧**，必须同时提供 `--bound-source` 说明独立证据；不能把当前候选位移或希望得到的流量作为上界来源。工具不会从泵指令猜测相别、标尺、输送比、截面或速度上界。

## 检查与未解决项

- 新增合成数据回归覆盖长时间运动、非整数周期、相差整周期的速度、静态背景、坏帧、弱相关、高残差、方向先验、上界冲突、多解、尾窗、突变与非有限输入。
- 最终全量测试：`525 passed, 6 subtests passed in 73.38s`。本轮新增 18 项用例（离线分析 17 项、旧求解器边界 1 项）。
- 最终全量命令：`.venv/Scripts/python.exe -X utf8 -m pytest -q -p no:cacheprovider --basetemp=output/offline-flow-review-20260919/pytest-final-temp`；完整日志在该结果根目录 `pytest-final.log`。
- 相关测试：`.venv/Scripts/python.exe -X utf8 -m pytest tests/test_offline_analysis.py tests/test_flow_locking.py tests/test_plug_geometry.py tests/test_plant_geometry_solver.py -q -p no:cacheprovider`，`59 passed in 12.85s`。当前虚拟环境未安装 Ruff，未运行其检查；未为此安装依赖。
- 实际录像已通过新入口处理，旧入口也已重跑。测试只证明软件行为，不能替代光学标尺、设备时间戳、注射器配置核对或独立流量测量。
- 真正的相别、像素尺度、三维截面和真实流量依然待现场验证；本轮不输出新的 PID/MPC 增益。

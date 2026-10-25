# 跨日期响应与停启对照离线审计

两个入口只读取保存的原图和元数据，不导入采集器，不连接硬件，不输出控制器可加载的标定。

## 可借鉴的方法与本项目实现

- 预测学的事前基线与衰减趋势：只用切换前末60秒拟合对数长度的直线。
  对比保持值、线性外推和固定60秒衰减的趋势。衰减形式为
  `intercept + slope * 60 * (1-exp(-t/60))`，借鉴阻尼趋势思想，
  不是直接实现或自动调参的 Holt ETS。
- 跨域验证：按日期隔离训练与测试；较晚日期的所有后段不进入较早模型训练。
  同一补液块的相邻转换绑定为一组，训练与汇总均按块等权，避免帧数制造虚假样本量。
- 实验对照：10月23日分别用SA/SB建立A/B状态的停启曲线，
  在F1/R1上拟合共享方向增益或分方向增益，再预测R2/F2整个后300秒。
  R2与F2之间不重拟合。曲线可迁移的假设仍待验证，只有一次对照不能估计对照间方差。
- 计量的敏感性分析：所有日期重新使用同一测量源码及完整候选标志，
  主几何和上下1传感器像素平移分别计算；不得选择最平滑的结果。
- 信号处理的采样相位检查：从已有5 Hz逐格记录取0/0.2/0.4/0.6/0.8秒
  五个偏移的1 Hz子序列，对比末60秒趋势；这是同一批图的敏感性检查，不是五次实验。
  不擅自对缺失或不同滴身份的长度序列做插值及低通后宣称物理真值。

方法来源：

- [Forecasting: Principles and Practice：评估、基线与时间序列交叉验证](https://otexts.com/fpp3/toolbox.html)
- [Forecasting: Principles and Practice：阻尼趋势](https://otexts.com/fpp3/exponential-smoothing.html)
- [scikit-learn：分组与时间序列交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)
- [NIST：测量过程时间变化的分析](https://www.itl.nist.gov/div898/handbook/mpc/section1/mpc123.htm)
- [SciPy：降采样前抗混叠的约束](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html)

这里的停启曲线不是NIST意义上的稳定核查标准；过程输出不是仪器误差真值。

## 输入与复算

`replay_measurement_audit.py` 的清单含 `stages`，每段要求：
`uid/date/block/q1/q2/walls/exclusion/frames`。`walls` 为两条归一化直线，
`frames` 按段内时间排列，字段为 `t/image/geometry_ok`。图片路径指向已有原始
Mono8 PNG或NPY。每秒取首张，不插值、不用后续帧替代首帧；坏图报错。
同组内清单顺序必须为实际采集顺序，不猜测跨组连续性。

```powershell
& .venv/Scripts/python.exe tools/replay_measurement_audit.py --manifest <source_manifest.json> --contract <measurement_contract.json> --output <new_replay_folder>
& .venv/Scripts/python.exe tools/evaluate_response_transfer.py --dataset <new_replay_folder>/dataset.json --output <new_evaluation_folder>
& .venv/Scripts/python.exe tools/audit_sampling_phase.py --review-dir <saved_5hz_review_folder> --output <new_phase_audit.json>
```

清单准备时保留原始索引及哈希；几何来源必须区分现场注册和事后估计。
源码契约不符即停止。输出目录必须不存在，结果不覆盖。

统一1 Hz是跨日期敏感性审计口径，不替代10月23日原定5 Hz主分析。
10秒箱至少8/10格有原图且3格有完整候选；无候选不填零、不等同无滴。
这比主分析最低10/50候选的门槛不同，因此不得将两份结果混作同一主评分。
原图只用于重测，供展示的亮度调整不进入计算。

## 预测模型和评分

预测量为 `log(L_after/L_pre_tail)`，基线为前段末60秒至少4个有效箱的中位数。
两路激励为 `log(Q1_after/Q1_before)` 和 `log(Q2_after/Q2_before)`。
响应核为 `1-exp(-max(t-delay,0)/tau)`；网格tau=10/20/30/60/90/120/240秒，
delay=0/5/15/30/60秒，仅按训练块等权损失选择。固定1e-6岭惩罚稳定数值，
激励矩阵秩单独输出，不能把岭解当作未受激励方向的物理识别。
跨日期幅值归一化只消除常数倍率；不会消除不同几何、照明、物相或芯片差异。

每个日期仅用更早日期的改流量块训练，不借用该日停启对照。
局部模型仅用F1/R1拟合响应参数，停启项仅从SA/SB取得；分方向模型每方向只有一个训练块。
超参数使用训练误差选取，未声称内层交叉验证。全部比较为事后开发分析。

三个几何版本使用共同有效箱，所有模型也在共同可评分箱上评分。
后段至少24/30箱、末120秒至少10/12箱；报告像素RMSE、相对前段基线的百分比RMSE、
末段中心误差。跨日期汇总先按补液块汇总，报告各日期结果，不混合绝对像素尺度。
像素标尺未标定，不输出微米；回读/命令流量不冒充实测流量。

`forecast()`接口不接受后段真实值；预测文件哈希仅证明事后可复算，
不证明采集前生成。已看过R2/F2，任何本次比较都不能恢复其前瞻留出资格。

## 无设备测试

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_response_transfer_audit.py tests/test_q2_v2_analysis.py -q
```

覆盖合成增益/延迟恢复、带已知停启项的恢复、后段真实值变动不影响预测、
同块泄漏拒绝、事件段不跨接、缺失不补零及共同评分覆盖率。

## 已保存预测的分量审计

`tools/audit_response_components.py --source <results.json> --output <new_folder>`
从已保存模型重建保持值、仅停启项、仅流量项及完整预测，检查与原预测逐点一致。
仅流量项沿用完整模型参数，不重新拟合。分别报告0–60、60–180、180–300秒
误差与偏差；中心化误差只作诊断，不能用测试段平均偏差回填预测。

另保存训练损失距最优不超过5%的参数网格在测试块上的误差，
不根据测试结果挑选参数。这是描述性容差集合，不是参数置信区间。
回归测试见 `tests/test_response_components.py`，不连接硬件。

## 多帧几何候选审计

`tools/audit_temporal_geometry.py --source <source_manifest.json> --output <new_folder>`
仅处理20261023两段式块，每块用前段180–298秒每2秒首帧的时间中位图，
在原墙线±8px内估计亮脊直线，并记录两组交错参考图拟合差异。
原图哈希、旧/新坐标、拟合拒绝原因与叠加图一起保存。
至少半数脊点不能支持拟合时拒绝该块，不放宽条件，不混用新旧几何作为通过证据。
时间中位亮脊是独立候选而非人工真值；两半参考图高度相关，不能称完整Gauge R&R。
输出清单只包含候选块，可用原图重放工具在相同采样口径下比较。
测试见 `tests/test_temporal_geometry.py`。

## 现场反馈后的资格隔离与原图审计

`tools/prepare_mpc_evidence.py --root <cross_domain_root> --review <5hz_review> --inventory <six_block_inventory.json> --output <new_folder>`
针对20261023六块的已知现场反馈，将R1/R2双壁测量限制写入新数据集的
`exclusion`，保留原记录及其他排除原因；不修改历史数据集。重新评估时所需
六块不齐则不拟合局部双向模型，报告明确写出原因。剩余块仍未验收。

工具逐张解码5Hz主分析所选原图并核对原有SHA256，输出亮度、对比度、
拉普拉斯方差和固定上部纹理区域相对本块首帧的位移诊断。当前仅支持
720×540 Mono8。相关响应0.3、位移2px是诊断分流阈值，不是已验证的物理
管壁/完整滴质量门控。块内画面稳定不能证明初始视野正确；本工具不判断物相。
命令证据直接读取各块`commands.ndjson`，不以汇总文件缺项推断事务不存在。
命令至读回间隔不等于停机时长；未保存的细分事件保持缺失。

## 人工盲复核页面

`tools/build_blind_boundary_review.py --sources <source_list.json> --output <new_folder>`
源列表记录`image/sha256`，每张复制两次，乱序且重复至少相隔10项；
公共页面不包含块名和原路径，对应关系仅存在`key_keep_closed.json`。
页面内嵌原始PNG以支持本地显示增亮，不改原图像素；标注为原传感器坐标，
不能直接代替条带像素门槛。答案需主动导出，无自动持久保存。
所有管壁、对象和复核结论初始为空；无法从样本选择推断人工类别。
不在人工复核完成前打开对应表评分，也不把同人重复称作跨人员再现性。

无设备回归：

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_mpc_evidence.py tests/test_blind_boundary_review.py tests/test_response_transfer_audit.py -q
```

两项工具均为离线准备，不连接设备、不输出可加载标定。正式起点与验证前预测
提交仍需在合格测量接口确定后接入采集生命周期；计划JSON中的额外字段不代表
现有采集器已实现这些门控。

## 盲复核答案校验与重复性诊断

```powershell
& .venv/Scripts/python.exe tools/score_blind_boundary_review.py --answers <exported_answers.json> --package <review_folder> --output <new_result.json>
```

检查包标识、样本编号、复核状态、原图坐标范围和壁可见性。空答案返回
`human_review_pending`，不会读取对应表；全部标注和复核人信息齐全后才读取
对应表，并核对其哈希。可见管壁必须有两点，不可见管壁不能填坐标；完成但
对象为空的图必须有说明，空对象列表不推断为无滴。

对象仅在轴向区间互相唯一重叠时配对；合并、拆分、漏标或歧义均另计。
类别一致率只描述唯一匹配子集，并列报告未匹配数量，不充当工程分类验收率。
完整滴端点误差为原图传感器坐标的欧氏距离P90，不与2条带px门槛混用。
工具不生成测量通过、模型通过或MPC授权，即使重复答案完全一致也不例外。
回归测试使用合成标注，不替代人工复核：

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_score_blind_boundary_review.py tests/test_blind_boundary_review.py tests/test_mpc_evidence.py tests/test_response_transfer_audit.py -q
```

## 换泵后全范围与单输入离线控制

`tools/replacement_pump_model_audit.py --dataset <curated_dataset.json> --output <new_folder>`
纳入清单内全部换泵后日期，保留逐段排除；执行整补液块、整日期和同日期整块
三种回顾性留出。整日期留出允许使用其他较晚日期，是迁移诊断而非事前预测。
逐日期保留局部模型；日期不等于芯片身份，不能自动据此建立物理芯片参数。
输出Q1/Q2是否同时变化，避免把混合激励称为单独Q2反向证据。

独立的离线log长度MPC原型仅优化Q2，整条计划固定Q1。`OfflineChipSettings`
保留逐芯片流量范围、参考流量、变化率上限与可选物理尺寸；未知尺寸为None。
尺寸字段不参与当前log像素模型，不冒充物理直径换算。默认70/20–25只是本轮
仿真配置，非所有芯片统一参数。与生产μm控制器隔离，不绕过其标定授权。
仿真覆盖模型增益变化、迟缓、反号、测量中断、未标记偏差及事务失败。
仿真结果不证明录制数据中未执行动作的反事实收益；事务失败后的停止仅是
模拟事件，不能称实机停止验证。相关回归见`tests/test_replacement_pump_model_audit.py`。

## 原图候选到真实记录的离线交付链

`tools/offline_q2_delivery.py --root <cross_domain_root> --review <5hz_review> --output <new_folder>`
运行六块固定每5秒原图的七几何一致性候选：原几何、上下平移、壁距变化及倾角变化。
匹配必须互相唯一，端点跨度超过2条带px则拒绝。此阈值是算法筛选规则，不能冒充
人工重复性门槛；保留未筛选候选和每个拒绝原因。R1/R2已知几何未确认，直接拒测。

对已有单独Q2变化做整块重采样参数稳定性分析，输出80次重采样分位数及近优网格。
仅三个上调块，不解释为可靠置信区间，也不授权反向外推。影子模型另外用此前日期
单输入块按同一新测量方法重测训练，避免新测量套旧模型；覆盖不足时模型为空。

真实记录回放读取原帧号、单调时间与原始命令日志，六块后段各1500个时间格。
重算候选仅在5秒抽样点，其余格标记未重测；稀疏候选基线至少8/12样本。
每10秒可计算一次Q1固定、Q2受限建议，但不下发，也不改变后续记录输入。
正常及视野移动、壁缺失、无测量、过期帧、事务失败情景各自保存逐格日志。
事务失败锁定停止请求，但不声称真实停止。所有资格标志保持false。
这一路为独立离线工具，尚未接入生产orchestrator；测试见`tests/test_offline_q2_delivery.py`。

# Q2方向现场采集

现场先试运行B状态时，可使用`tools/run_q2_sb_pilot.py --plan <file> --live`：
仅允许固定70/25、单段60秒，结束沿用原核验停泵路径；不执行正式SB的同值停启对照。
`capture_mode`须为`pilot_60s_not_formal_sb`，profile为`q2_sb_pilot_70_25_60s_20261023`，
阶段标签`SB_pilot_60s`。前30秒请求全速落盘，余下20Hz；启动前仍做原10秒存储试采和30秒指令前记录。
不带`--live`仅预检。试运行消耗必须记入当前补液记录，不能把余量重置为1000μL后直接接正式SB，
也不能把试运行当成正式300秒对照或模型辨识块。

2026-10-23新增三种固定配置：`sham_70_20_20261023`与`sham_70_25_20261023`
在同一流量下仍执行核验停泵/重启；`q2_reverse_70_25_to70_20_20261023`
用于Q2反向响应。第二段记录`same_setting_transaction`，不把同值事务省略为无操作。
每段300秒、每针累计900μL上限及原始图采集约束不变。
六块实验、1000μL容量预算与模型留出设计见[10月23日方案v2](experiment_20261023_v2.md)。
原始PNG的固定5Hz取样、10秒分箱和同批±1px几何诊断见
[v2离线主分析入口](q2_v2_offline_analysis.md)。

新芯片可在计划的`geometry_guard.reference_patch_px`声明当前视野中的静态参照矩形
`[x,y,width,height]`，坐标为传感器像素。自定义矩形必须完整位于图内且不与通道相交，
在启泵前以当前原图确认；不得用液滴、拟合误差或留出表现选择参照。
未声明时沿用原矩形。配准0.85相关门槛、运行中±2px漂移限制和异常停止路径保持。
允许准备时用高曝光确认管壁；正式运行前恢复指定低曝光、保存该设置下的静态参照，
确认视野未移动并冻结相机设置，正式原图不混入准备时的高曝光样本。

`tools/run_q2_lab_capture.py --plan <file>`默认无设备预检；`--live`明确启动固定70/20→70/25，各300秒，不再增加统计稳态等待。用户指定各段末120秒留作后续拟合，今日仅检查采集质量，不在线拟合或启用MPC。下一块必须停泵后重新核算补液与行程。

复用ResponseSession的当前几何注册、每2秒静态特征检查、有界内存快照、异常安全停止，以及原始相机/写入线程、STOP文件和耗液累计。参数更新需要停止确认、写入读回、重启确认；Q2上升期间按25.5μL/min上界累计，切换前后均检查剩余目标预算。每针累计上限900μL，新满1mL确认另记，不是实际余量传感器读数。全流程含30秒切换及10μL余量预算，油750.25μL、水252.75μL，加实际启动累计。

采集原始Mono8 PNG、真实帧号和时间戳：常规20Hz，各段0–30、180–210、270–300秒100fps，切换100fps。曝光/增益/ROI不因拟合目标而改变。几何仅处理参考特征平移，仍需原图复核，不能宣称已验证物理直径或流态。默认不连接设备的测试覆盖单步时序、上升水相预算、错误读回、几何失败和事务失败；控制生命周期改动需完整pytest。

同入口另有两种冻结采集块，不允许任意自定义流量：`q1_60_20_to50_20_20261016`为60/20→50/20，`holdout_60_20_to65_22_20261016`为60/20→65/22。均每段300秒、固定时间诊断，不在线拟合。每块必须有独立计划和新补液状态，不能一个满针确认用于连续三块。切换期间按两个平台各通道的较大上界累计；读回通过后才切换至新平台上界。中间点预先标记为后续模型检验数据，不因明天拟合结果更换点位。

反向块`q1_reverse_50_20_to60_20_20261016`为50/20→60/20，各300秒。沿用相同事务和图像监测，不增加任意流量入口。两平台、30秒切换及10μL保留量预算为油595.25μL、水225.25μL，另加实际启动累计。与另一补液块的60/20→50/20比较时保留块效应，不能直接称为同一次行程的迟滞往返。

减水块`q2_70_20_to70_18_20261016`为70/20→70/18，各300秒。与加水块对照，切换全过程仍按20.5μL/min水相上界累计，读回核验后再用18.5μL/min。两平台、30秒切换及10μL保留量预算为油750.25μL、水215.25μL，另加实际启动累计；每针累计上限仍为900μL。
# Explicit raw-only acquisition

`capture_plan.raw_only=true` with `detector_executed=false` records the fixed
two-stage profile without registering walls or stopping on geometry correlation.
Use only when explicitly requested for raw data collection. No geometry or
measurement acceptance is claimed. The storage pilot, frame-reader checks,
delivery budget, STOP file, verified stop/restart transaction and final verified
stop remain active. A prior aborted run still counts against the same refill;
reset the refill budget only after a new physical refill confirmation.

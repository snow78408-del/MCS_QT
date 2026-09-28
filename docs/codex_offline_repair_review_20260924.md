# 离线修复交付复审：部分通过，尚不能整体验收

## 本轮执行与结论

Codex 实际复跑四个新增测试文件：51 passed in 5.01s。第一次运行受 Windows 临时目录权限影响而中断，换用获准的隔离执行后通过；不能将第一次环境错误计为产品失败。未重跑全套，949 passed + 6 subtests 仍为实施方完整测试记录。

独立反例：output/codex-repair-review-20260924/probe.py；输出：同目录 probe_results.json。只用合成帧、模拟会话与新建文件，无真实设备操作。未修改产品代码或已有证据。

逐行汇总 7 个回放文件，确认 2797 帧：pending_motion 1231、ambiguous 1378、rejected 188。该统计成立，不代表已经确定物理失败原因。

严格模式接口分离、stop=None 不再崩溃、共享测量辅助入口的改动可以保留。但“全部生产入口已统一”和“追溯已通过”的结论不成立，下列问题须修正。

## 1. [P1] 实时 VisionPipeline 未接入本帧像素几何

位置：backend/vision/pipeline.py:147，仍为 self.detector.detect(gray)。适配器实际调用该管线；新增辅助测量路径不能替代这条调用链。

独立探针用真实 VisionPipeline、真实 detector，输入已提供当前墙线的合成帧。扶正结果 shape=(40,180)，检测实际收到 kwargs={}，trace 给出 reference_width_px=50.0、reference_width_source=config_nominal_geometry。

要求：将实际实时管线接到统一入口，在适配器 → 管线 → 检测器层测试同帧一致性。必须覆盖辅助测量开关关闭的默认路径，不能仅比较两个辅助函数。

附带 [P2]：rectified_axes() 仍用长边猜流向、min(height,width) 作横向宽度，仅把旧推断包进函数；80×40 的输出被认作沿 y 流动。rectify_channel_frame 本身定义了目标 x 为轴向，应从变换契约或显式元数据确定轴，若不支持短而宽的 ROI 就拒绝，不得静默转轴。另外 N 个像素的索引跨度是 N−1，当前 span=N 却称 pixel_index，文档自相矛盾，需区分像素个数、索引跨度与原图几何距离。

## 2. [P1] 缺测量证据时仍默认验收通过

位置：tools/plant_flow_transient_capture.py:1133–1134，frame_summary.get('measurement_accepted', True)。

模拟会话仅提供 sampling_premise_ok=True，没有任何测量验收字段；运行完成且停泵已确认后，report() 给 measurement_accepted=True、task_goal=campaign、task_goal_met=True，exit_code=0。这把采样完整再次当成测量验收。

要求：缺失验收证据必须为未知/未验收，并附原因；campaign 不得通过。诊断采集目标可以单独通过，但测量仍未验收。补真实 report() 路径回归，不只手工拼 result_layers 测退出码。

## 3. [P1] 追溯尚未接入现场入口，配置也未完整绑定

三个现场入口均无 snapshot_sources 调用，也未接入等价追溯机制。目前只在 replay_wall_localization 中创建快照。任务书要求的“快照/配置失败则在硬件动作前拒绝启动”未落地。

生效配置仅手写几个字段；build_detector 可加载其他调优参数，但 effective_config 未保存完整合并配置，也无定位器完整参数、配置内容指纹与测量行版本绑定。SOURCE_MANIFEST 遗漏实时 pipeline、vision_adapter/service 等影响实际链的模块，不能宣称覆盖整个生产依赖链。

要求：接入入口执行链，纯模拟验证追溯写入失败时设备调用数为零；完整保存实际对象生效参数及哈希，测量引用其版本，明确源码清单边界。只保存离线示例不能判新现场记录已具备追溯性。

## 4. [P1] 核验器能接受不存在的文件、负索引及任意配置

位置：backend/provenance.py:292、309；tools/verify_provenance.py:46。

独立反例：container_path='does-not-exist.mkv'，software_frame_index=decoded_frame_index=-4，verify_frame_references 返回 ok=True，resolve_source_frame 返回 resolved=True。函数只整理引用字段，没有打开媒体或确认帧存在。

另仅写入 {'detector': {'generation_min_length_ratio': 999}}，verify_config 仍 ok=True：没有比对配置哈希、要求完整结构或绑定可信记录。--still 只计算当前哈希，也没有与记录值比较；不能称为静帧篡改核验。

要求：区分“引用结构合法”“媒体存在”“帧可解码”“内容匹配”；加入明确会话根目录，校验非负整数索引、实际文件、索引范围与内容哈希。解码找回原帧才可称 resolved。静帧和原始块需有对应解析分支。配置校验必须核对保存时的内容指纹和版本绑定。软件帧序号与解码帧索引可以合法不同，不能将强制二者相等当作真实性检验。

## 5. [P2] 快照一致性并未检查复制后的源文件

位置：backend/provenance.py:136。before 与副本比较，未在快照末尾重新读取源文件。

独立反例：copy2 完成后把源文件由 original 改为 changed after copy，仍 consistent_snapshot=True、changed_during_snapshot=[]。

要求：所有文件初始哈希 → 复制 → 所有源文件末次哈希，比较源/副本/初始值。明确这只能检出相应变更，不等于证明进程实际加载了磁盘同一版本；运行版本固定策略和限制应写清。补“复制后修改先前文件”的回归。

## 6. [P2] 回放传入的硬件帧身份不真实，失败图缺少候选证据

位置：tools/replay_wall_localization.py:247，FrameEvidence(frame_id=index, hardware_frame_id=index)，未使用事实记录中的硬件帧号。输出行却保存了真实硬件帧号，例如软件 6661 对应硬件 6794。定位全部失败使该分支尚未运行，不等于正路径正确。

要求：按真实身份契约同步定位器、测量证据与输出，不通过把软件序号填入硬件字段来绕过一致性检查。需用真实定位/检测的合成可通过序列覆盖此分支，并设置软件/硬件号明确不同。

目视抽查 b1_s1 的 6661 和 6961 标注图：只有顶部状态字符串，看不到竞争线对。annotate() 仅在 usable=True 时绘制墙线，所以本轮全部失败帧均不能帮助审核“哪些线造成歧义”。行记录仅有截断理由，没有完整 motion/coverage/竞争对几何。不能据此确定“另一个真实右侧管道”就是原因，也不能断言软件改进无法解决。

要求：失败时同样保存候选线对及坐标、分数、共见区间、运动/覆盖证据与拒绝原因，绑定帧号并可视化。文字改为“定位器报告候选竞争，物理来源尚未确认”。给出证据诊断，不要求放宽阈值强行通过。

回放还在循环结束后才一次写出 rows（约第 309 行），不满足异常保留已提交行。应随帧增量写盘；新增中途检测/写盘失败的恢复检查。

## 下一轮范围与验收

只修上述具体问题，不另跑大规模参数搜索，不启动硬件，不重写历史原始记录。

优先修 1、2、3、4，再修 5、6。保留本轮有用的接口与测试。先让本次独立反例变成失败被正确识别或路径行为一致，再补针对性回归。

验收材料：真实适配器/管线正路径对照；无测量证据的 campaign 拒绝；现场入口追溯失败前置阻断的模拟证据；从测量行实际读取原帧的核验；源码/配置/媒体变更的拒绝证据；包含竞争线对的真实失败片段。完成后跑受影响测试与一次全量测试。

“同片段修复前后差异为零”应区分实测对照与逻辑推断。若未保存并运行修复前的可执行版本，只能说新路径在这些片段没有进入测量，不能写成已完成前后实测。

交付到新的 output/deepseek-offline-repair-followup-20260924/，不要覆盖本轮目录。最终仍分别汇报软件、真实录像、追溯与现场未验证项。

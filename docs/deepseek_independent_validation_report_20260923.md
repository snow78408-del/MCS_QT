# 独立验收报告：2026-09-23 管壁定位、扶正、短液柱识别与实体诊断

- 审计对象：`docs/deepseek_independent_validation_20260923.md`（下称「任务书」）所列 E01–E05、F01–F04、P01–P02、O01–O03、N01
- 工作目录：`E:\MCS_QT`；Python：`.venv\Scripts\python.exe`（Python 3.13.7，cv2 5.0.0）
- 权限范围：只读审计 + 无硬件测试。未连接泵或相机、未下发流量、未启动实验、未删除或改写任何原始记录、未上传录像。
- 新增产物仅限独立审计目录 `output/audit-independent-20260923/`（脚本、日志、统计、叠加图）。**未修改任何产品代码、配置或既有记录。**
- 审计脚本与原始输出：`output/audit-independent-20260923/`（`*.py`、`pytest_targeted.log`、`pytest_full.log`、`video_verify.log`、`overlays/`、`column_stats.json`）

---

## 1. 一句话结论

**不能开始正式多组阶跃实验。** 最直接的阻塞项是：**被抽样的 27 个复核点全部定位失败（`status='not_measured'`，27/27）**——这证明这些抽样点在测量链上未取得有效定位，**尚不等于**已逐帧统计整段录像的定位成功率；而两次 90 秒短液柱诊断又完全绕过了该定位器、直接复用一张**未经验证的单帧目视提议**。因此「每次检测先定位当前管壁再扶正」这一首要要求**目前没有任何一次经证实成功的实例**。

短液柱方面，`0 → 349` 的机制已确定为**检测阈值放宽**（`generation_min_length_ratio` 2.50→0.50，叠加轮廓判据改写），而非精度提升；但 349 不能作为液滴计数或精度证据——其中存在异常的位置重复候选，其物理来源**尚无同步真值可判定**（详见 §3 P3、§12）。

> **本报告已经过一轮独立复审（2026-09-24）**，复审在 §5.1 的坐标映射、§3 P3 的因果推断、§3 P4 的措辞、§2 全幅扶正行及一处**真实 bug**（`make_overlays.py` 缺少逆透视变换）上提出了更正。§12 记录逐条处理结果；**上文与 §2–§11 已按复审更正**，其中被推翻的表述见 §12.1。

---

## 2. 验收矩阵

| 验收项 | 结论 | 依据 |
|---|---|---|
| 当前帧定位（auto per-frame） | **失败（按抽样点）** | `r1-front4-20260923/ten_point_review/localized_report.json` 12/12、`v1-b1-20260923/fifteen_point_review/localized_report.json` 15/15 全部 `not_measured`。理由含「当前帧没有可判定的位移证据，无法证明这两条是内壁而非停泵液柱或固定纹理」「带内存在**静止**的同向长边，无法区分壁面与液柱边缘」「有 1 组同样可信的线对，无法唯一确定目标管道」。**注意**：27 个点是抽样复核点，本项只证明「被抽样的点在测量链上未取得有效定位」；整段录像的逐帧定位成功率**尚未统计**，不能写作「逐帧成功率 0」 |
| 全幅扶正 | **证据不足（延伸段未经逐位置验证）** | `current-wall-fullspan-20260923/proposal.json` 是 `single_frame_proposal_not_validated`；墙线 x 由 0.1556→0.9722（原图 112→700 px）为**同斜率直线延长**（斜率 0.105 与短版完全一致），延伸段占扶正条带宽 587 px 中的约 187 px（32%）。**更正**：不能因原提议止于 x=514 就断言右侧无图像边缘证据——复审目视在原图右侧仍见条带与液柱轮廓；这些轮廓是否为内壁、是否足以支撑沿程定位，**需逐位置对照，本轮未判定**。E04/E05 用的就是这套墙线（`run_channel_validation_live.py:142`）。修正版叠加图见 `overlays/*_v2_annotated.jpg` |
| 短液柱识别 | **失败（计数不可采信）** | 见 §3 P3。`frames_with_complete_plugs=349` 中 ≥93 行落在扶正坐标 x≈[263,360]、长度恒为 93–98 px 的同一位置簇（首末相对首行 0.403 s / 57.275 s），另有 17 行发生在 `start-infusion` **之前**（泵未运行）。**位置重复本身不足以判定其物理来源**（可能是固定纹理、滞留对象、重复生成或采样混叠），但 349 因此不能作为独立液滴计数或召回/精度证据 |
| 物理标尺 | **失败** | 无独立标尺。`measurement_chain_status.physical_scale_validated=false`（但为硬编码，见 P12）；`configure_expected_diameter(0, 1.0)` 使 `_pixel_to_micron=1.0`，即把名义 50 µm 通道当作 50 px（`detector.py:50-55, 151-157`）；plan 内 `scale_um_per_px=1.725` 自带 `"historical only"` 标注 |
| 时间线 | **通过（有附注）** | 见 §4。三段长录像可逐帧锚定墙钟，观察窗严格 300.0 s；附注：E02 含 457.7 ms 连续空洞（P8），且缺口非均匀分布 |
| 限时运行 | **通过** | E01 300 s 限时；E02 四段 ×300 s；E03 五段 ×300 s（观察窗 300.005–300.015 s）；E04/E05 90 s。均按 `max_session_s`/`max_segment_s` 正常结束 |
| 安全停止 | **通过（附一条退出码缺陷）** | 5 次实体运行 + F04 全部 `stop.verdict=STOPPED, verified=true`；停泵 4 次重试、`STOP_UNVERIFIED` 优先于 `FAILED`、`requires_onsite_confirmation` 与 `onsite_instruction` 齐备（`plant_flow_transient_capture.py:1026-1054, 1074-1110`）。缺陷见 P5（退出码只看停泵、`stop=None` 抛异常） |
| 工艺稳态 | **不成立（未评估）** | 无真实阶跃数据可评；`stability_criteria.spread_limit_with_units` 自述 `"not frozen because stage-A physical measurement validation failed"`；90 s 资料不满足 120 s 有效窗口 |

---

## 3. 按风险排序的问题

### P1（阻塞）被抽样的 27 个复核点定位全部失败（抽样结果，非逐帧成功率）

- **文件**：`backend/vision/parallel_walls.py:1222-1241`（判定分支）；证据 `output/r1-front4-20260923/ten_point_review/localized_report.json`、`output/v1-b1-20260923/fifteen_point_review/localized_report.json`
- **最小复现**：
  ```bash
  .venv\Scripts\python.exe output\audit-independent-20260923\localized_status.py
  ```
  输出 27 条全部 `status='not_measured'`，`status counts={'not_measured': 27}`。
- **实际影响**：这是用户首要要求（「每次检测先定位当前管壁再扶正」）的直接实测结果——**被抽样的点无一成功**。严格模式下 `vision_adapter.py:1566` 传 `current_wall_lines=[]`，`pipeline.py:104-116` 依 `len!=2` 拒测，于是**严格模式在被抽样的这些帧上会全部拒测**。这也解释了两次 90 秒诊断为何绕过定位器：走定位器将一无所获。
- **本项的范围（复审更正）**：`localized_report.json` 只覆盖**被抽样的 27 个复核点**（12 + 15），不构成对整段录像的逐帧定位统计。因此本项结论是「抽样点全部未取得有效定位」，**不是**「整段录像定位成功率为零」。要给出成功率，需对录像做连续片段的逐帧定位统计（见 §11 第 3 条）。
- **性质（事实 / 推测分开）**：**事实**为状态与理由字符串（`localized_report` 原文）。**推测**为「失败原因主要是运动证据不足」——理由中 20/27 提到位移/稳定性不足，6/12 与 1/15 提到「带内静止同向长边」；但未逐帧复核原图，故不能断言其余失败与场景无关。
- **最小修复建议**：不要放宽 `pending_motion` 门槛来「让它通过」。应先满足运动判据的可观测前提（曝光/增益使管内沿轴结构 `motion_min_axial_structure_gray` 可达），再重测；并对每次定位持久化 `motion`/`coverage` 子项，便于定位失败到底卡在哪一条。
- **附**：`reason='no_current_frame_evidence'` 是未经翻译的原始码，混在中文理由里（`localized_report` B1 第 5 点）。

### P2（阻塞）产生两次 90 秒诊断的代码在本次可访问范围内找不到

- **文件**：`tools/run_channel_validation_live.py`（mtime `09-23 20:49:59`），`git ls-files` 报 `did not match any file(s) known to git`；`output/*/{a,b}/session_summary.json` 的 `events`
- **最小复现**：
  ```bash
  git status --porcelain tools/
  ```
  输出全部为 `??`（未跟踪）。再对全仓搜索：
  ```bash
  grep -rn "measurement_validation_90s" E:\MCS_QT
  ```
  **全仓无任何命中**，而 `channel-validation-live-20260923a/b` 的 `session_summary.json` 的 `events` 里就有 `measurement_validation_90s_started` / `measurement_validation_90s_finished`。
- **旁证**：当前源码记录的事件名是 `baseline_started`/`baseline_finished`（`run_channel_validation_live.py:88, 108`）；两次运行的 `segments` 为空数组，而当前代码会写入一条 `C_baseline` 段记录（同文件 `:89-91`）；两次运行的 `commands.ndjson` 与摘要 `commands` 中**都没有** `baseline-run-state` 记录，即任务书 §2.1 所称「周期泵状态回读」在实体运行时**未发生**。
- **结论的边界（复审更正）**：以上只能证明「**已找到的当前源码与产生这两次运行的代码不一致**」。mtime 与未跟踪状态本身**不能**证明工作区之外不存在任何历史副本（另一台机器、压缩包、编辑器本地历史等）。因此本条主张的范围是：**在本次可访问范围内（工作区 + git 跟踪 + 全仓文本搜索）找不到产生该运行的源码**，而不是断言「该代码已不存在」。
- **实际影响**：两次 90 秒诊断（短液柱的唯一实体证据）在本仓库内**不可复现、不可代码级审计**。计划内 `software_readiness.code_hash: null`，因此连代码状态都没有锚点。
- **最小修复建议**：承认本期短液柱结论为「证据不足」；在入口加 `code_hash`（对相关源码做哈希写入 plan/摘要），并把入口脚本纳入版本控制后再重跑一次 90 秒诊断。

### P3（高）0 → 349 的机制是检测阈值放宽；重复候选异常，但物理来源未经同步真值判定

- **文件**：`backend/vision/config.py:88`（`generation_min_length_ratio` 2.50→0.50）、同文件 `:96-108` 新增 `generation_outline_row_window_ratio=0.70` / `generation_outline_gap_min_ratio=0.55` / `generation_outline_gap_max_ratio=1.45`；证据 `output/channel-validation-live-20260923{a,b}/measurements.ndjson`
- **最小复现**：
  ```bash
  .venv\Scripts\python.exe output\audit-independent-20260923\analyse_measurements.py channel-validation-live-20260923a channel-validation-live-20260923b
  .venv\Scripts\python.exe output\audit-independent-20260923\band_timing.py
  ```
- **事实**：
  1. 两次运行的**整帧统计量**几乎相同：E04 十张 `mean≈86.8, std≈6.75`，E05 十张 `mean≈86.7, std≈6.85`；两批都是 gain 6.0。**更正**：整帧 `mean`/`std` 相近**不证明**物理场景相同——复审目视 `a_check00`、`a_check05`、`b_check02`、`b_check05`、`b_check09` 的 `plain` 图，报告在管内不同位置可见带圆弧端部的液柱状结构（`b_check09` 尤为明显）。低对比度对象的出现/消失对整帧均值影响很小，因此本项**只说明整体照度与对比度级别未变**，不能说明内容未变。
  2. E04（旧配置）`selected_pixels` 全空（0/476），但其 `outline_checks` 原始候选中，x∈[262,265]×[357,360]、长度 93–98 px 的区间出现在 473 个有候选的行中的约 45 行。该长度 > 旧门槛 35×2.5 = 87.5 px，**即长度门槛不是它被拒的原因**；拒它的是旧的轮廓/capsule 判据（`config.py` 注释自述旧算法「follows the fixed walls and the background texture ... every real low-contrast capsule was rejected on the outline gate」）。
  3. E05（新配置）同一位置簇被**接受**：93/349 行，首次与末次相对首行分别为 **0.403 s / 57.275 s**，长度恒为 93–98 px。
  4. 被接受区间的起点直方图在扶正坐标 x∈[250,400) 堆积 **344/489 = 70%**。
  5. E05 有 **17 个被接受行发生在 `start-infusion` 之前**（泵未运行），另 2 行在停泵之后。
- **实际影响**：`0 → 349` 的成因是**检测判据放宽**（长度阈值 + 轮廓判据），这一点由第 2 条（旧配置下同一区间已作为原始候选出现、且长度已过旧门槛，却被轮廓判据拒掉）与第 1 条的照度不可比性共同支持。但 **349 不能作为液滴计数或吞吐/精度证据**，因为其中存在显著的位置重复候选与泵未运行期间的候选。
- **性质（事实 / 推测分开）**：**事实**为上述计数与时间分布（可复现）。**推测及其替代解释**：位置重复**不等于**同一个静止对象——重复生成、采样混叠（5 Hz 采样对 ~100 fps 场景）、位置相关的检测偏好、固定纹理、以及滞留/被困对象，都能产生位置聚集；现有记录中**没有与测量行绑定的同步图像，也没有同一对象的连续轨迹**，因此**无法定性**该区间是固定光照/背景、滞留液滴、还是反复生成的真实液柱。同理，泵启动前的候选**可以是真实存在的残留对象**（残留液滴、气泡），它们不能计作本轮新生成/输送的液滴，但**不是检测器出错的充分证据**。
- **对 `band_is_fixed.py` 的自我更正**：该脚本分析的是 **B1/R1 的录像**，而 E05 没有录像（`video_saved=false`）；且 B1/R1 与 E05 的成像条件不同（见 P9）。因此该脚本的列方差**不能**用于判定 E05 候选区间的对象身份——报告中不引用其结论。E05 的位置重复现象**目前没有可用的时间维检验**。
- **最小修复建议**：在报告层把 `frames_with_complete_plugs` 改名为「至少一个接受候选的帧数」，并**分别**输出：候选帧数、经位置去重后的位置簇数、以及（在有连续录像时）能形成连续轨迹的对象数；在 `start-infusion` 之前与停泵之后单独标注、不计入新生成/输送计数；对重复出现的区间标注为「疑似固定或滞留，需人工判定」，而**不**直接判为假阳性。修复方向是补齐**同步真值**（带帧号的对照图或短段连续录像），不是继续调阈值。

### P4（高）生产路径与 bench 工具的 plug 长度门槛不同

- **文件**：`backend/vision/detector.py:98`（`detect()` 调 `self._detect_generation_plugs(gray)`，**不传** `channel_width_px`）对比 `tools/run_channel_validation_live.py:52`、`tools/capture_detector_review.py:41`、`tools/review_current_wall_pair.py:64`、`tools/preflight_current_channel.py:87`（四处均传 `channel_width_px=float(min(gray.shape))`）
- **最小复现**：
  ```bash
  grep -rn "channel_width_px" backend tools
  ```
- **事实与量化**：`detector.py:150-157` 中 `reference_width_px` 默认取 `min(generation_channel_height_um, generation_channel_width_um)/scale`；生产路径 `configure_expected_diameter(0,1.0)` 使 `scale=1.0` → **参考宽 50 px → `min_length = 50×0.5 = 25 px`**。bench 工具传 `min(gray.shape)`，对扶正图 `(35, 587)` 为 35 → **`min_length = 35×0.5 = 17.5 px`**。
- **附带事实**：`channel_width_px` **不是 `DetectorConfig` 字段**（`backend/vision/config.py:76-235` 无此项），因此无法被 `vision_tuning_parameters.json` 保存/加载；线上调用无任何路径传入它。
- **实际影响**：离线/bench 复核里「短液柱可识别」的结论**不迁移到线上**——线上门槛严 1.43×。这正是用户「真实液柱可以很短」需求的直接受害者。
- **关于任务书假设的更正（重要）**：任务书 §3 担心「以裁剪高度代替真实内径」。**审计未证实该指控**：`min(gray.shape)=35` 当扶正条带高<宽时，恰好等于 `wall_line_quad` 的 `output_height`（两壁端到端平均像素距离，`rectified_roi.py:76`），因此它**不是任意的裁剪高度**。**但（复审更正）**：35 px 是**该目视提议几何的输出**，与提议自洽**不等于**墙线已通过真实性验证——不得改称「真实管壁间距已证实」。真正的问题是它从**未验证的单帧目视提议**推导而来，几何误差会直接传入长度门槛。
- **附带的换算说明（复审提出）**：`output_height` 由输入墙线端点距离经取整与下限约束得到（`rectified_roi.py:76`），而透视目标把端点映到 `output_height-1`（同文件 `:93-96`）。因此「输出像素索引跨度」与「几何长度」相差一个像素量级的归一化因子；用 35 作为长度门槛的像素参考时，该差异虽小但应显式记录，避免后续把索引差直接当几何长度。
- **另附**：`min()` 依赖「条带高<宽」，方向一变即会取错轴，属脆弱写法。
- **最小修复建议**：把像素参考提升为 `DetectorConfig` 字段（可在 tuning profile 中保存），线上与离线统一取值来源；用显式的 `rectified_height_px` 替代 `min(gray.shape)`。

### P5（高）入口退出码只看停泵；`stop=None` 时抛异常

- **文件**：`tools/run_channel_validation_live.py:152`、`tools/run_r1_front4_live.py:317`，两处均为
  `return 0 if result.get("stop", {}).get("verified") else 3`
- **最小复现（两个独立缺陷各一条）**：
  ```bash
  .venv\Scripts\python.exe -c "result={'verdict':'FAILED','stop':None}; print(result.get('stop', {}).get('verified'))"
  ```
  第二条：
  ```bash
  .venv\Scripts\python.exe -c "result={'verdict':'FAILED','stop':None}; print(0 if result.get('stop', {}).get('verified') else 3)"
  ```
  第二条输出 `AttributeError: 'NoneType' object has no attribute 'get'`。
- **事实**：
  1. 全部 5 次实体运行的 `stop.verified` 均为 `true`，故**全部返回退出码 0**，尽管 `verdict=PREMISE_REJECTED`、`completed=false`。任务书 §4E 的担心成立。
  2. `report()` 在 `pump_may_be_running` 从未置位时输出 `"stop": None`（`plant_flow_transient_capture.py:1097`）。此时 `dict.get("stop", {})` 返回 **None**（键存在），非默认 `{}`，于是 `.get()` 抛异常。**F01、F02、F03 三次尝试的 `stop` 正是 `None`**（`list_summaries.py` 输出）。
- **实际影响**：任何按退出码判断「本次运行是否可采信」的包装脚本/CI 都会把 `PREMISE_REJECTED` 读成成功；而启动前失败的那类尝试会以 `AttributeError` 崩掉，掩盖真实失败原因。
- **最小修复建议**：退出码按 `completed` 与 `verdict` 分级（例如 `0` 仅当 `completed is True`；`PREMISE_REJECTED`→非 0；`STOP_UNVERIFIED`→单独的更严重码），并把 `stop=None` 显式处理为「未进入过停泵路径」而非异常；`completed`、测量验收、停泵三项分别报告。

### P6（高）严格定位模式会被前端几何设置静默清除

- **文件**：`backend/orchestrator/service.py:1845-1857` 对比 `:1783-1789`；`backend/orchestrator/vision_adapter.py:639`
- **最小复现（代码级）**：`service.py:1845` 取 `roi = dict(self._cfg.recognition_roi or {})`，`roi.update({...四个 generation 字段...})`，随后 `setter(roi)`。而 `strict_detection_localization=True` 只写在 `:1783` 的**局部副本** `roi_config` 上，**从未回写 `self._cfg.recognition_roi`**。`vision_adapter.py:639` 以 `values.get("strict_detection_localization", False)` 重读 → 得 `False`。
- **可达性**：该路径仅在 `SystemState` 不属于 `RUNNING/CALIBRATING` 时才放行（`:1837-1844`），即停止态/初始化态即可触发。
- **实际影响**：前端改一次生成区几何参数，**当前帧严格定位即被关闭**且无任何提示；此后 `vision_adapter.py:1569` 走非严格分支、回退到旧的 ROI 裁剪路径。
- **性质**：**事实**为键未回写与默认值路径（可直接读代码确认）；**推测**为该键在前端确有调用入口（`qt_app.py` 的生成区参数设置路径），本次审计未执行前端以端到端确认。
- **最小修复建议**：`set_recognition_roi` 不应由「未携带该键」推断为关闭；改为只在该键**显式存在**时才改变严格模式（`if "strict_detection_localization" in values`），或在 `service.py` 侧把该键回写进 `self._cfg.recognition_roi`。

### P7（中）逐帧 `exposure_time_us` 恒为 0，且不自我标注

- **文件**：`tools/plant_flow_transient_capture.py:458-460`；`backend/vision/cameras/adapters/hikrobot_direct.py:341`
- **最小复现**：
  ```bash
  .venv\Scripts\python.exe output\audit-independent-20260923\timeline.py v1-b1-20260923
  ```
  输出 `per-frame exposure: zero=152618 nonzero=0`。
- **事实**：`_measured(..., unset_when=lambda v: v is None, ...)` 只把 `None` 视为未填充，**放行 `0.0`**，产出 `{"value": 0.0, "reason": null}`——即看起来「已测得 0 µs」。同文件 `:418-422` 的 docstring 恰恰声明「不把 dataclass 的默认 0 当成实测值」；同一 dict 里 `lost_packet_count` 用 `v < 0`、`hardware_frame_id` 用 `v <= 0`，**唯独曝光缺此保护**。
- **交叉验证**：F02 的 `original_error` 原文为 `CameraSetupError('相机 exposure 回读与设定不符：设定 80.0，回读 0.0，容差 4.0')`——直接证明「从帧取曝光」的回退路径确会返回 0.0。
- **实际影响**：`frames.ndjson` 的 exposure 列**不可用**，但表现为「有值」。真正的曝光核验来自 SDK getter（`camera_exposure_verified` 事件），与逐帧字段是两条不同路径。任务书 §4A 要求区分二者——**现可确认：不能声称每帧曝光已验证**。
- **修复建议**：`unset_when` 改为 `lambda v: v is None or v <= 0.0`，并把 `reason` 写明「设备流回调未提供有效曝光时间」；或改用与设定值 80 µs 的一致性校验。

### P8（中）E02 存在 457.7 ms 连续空洞，且不在摘要中披露

- **文件**：`output/r1-front4-20260923/session_summary.json` 的 `frame_facts.sampling_premise_reason`；证据 `frames.ndjson`
- **最小复现**：
  ```bash
  .venv\Scripts\python.exe output\audit-independent-20260923\gap_shape.py r1-front4-20260923
  ```
- **事实**：E02 共 292 个缺口事件，其中 284 个单帧、7 个双帧、**1 个 44 帧连续缺口**（`host_monotonic` 3378.167→3378.624，**dt=0.4577 s**）。E03（B1）最大空洞仅 43.9 ms（3 帧）。对照 B1：232 事件 / 223 单帧 / 最大 3 帧。
  - 该 44 帧空洞位于 **seg3（label `L`, q1=60）** 内，距该段结束约 36 s，**落在末 60 s 与 120 s 候选稳态窗口之内**。
  - 三段录像 `lost_packet_count` 全帧求和均为 **0**，即相机未把这些缺失报为丢包。
- **影响分级**：缺口对**尺寸**影响可忽略（单帧 10 ms）；对**起点**影响有限；对**频率/速度**影响最大——单个 458 ms 空洞与 342 个 10 ms 散点不可等同，而摘要只报「帧号缺口 342 帧」，未披露此结构。
- **最小修复建议**：`reason` 中同时输出最大连续空洞（帧数与毫秒）与其所在段/时间；对落在稳态窗口内的连续空洞单独告警。

### P9（中）长录像与 90 秒诊断光度不可比（增益不同）

- **文件**：`tools/run_channel_validation_live.py:110-112`（`_apply_feature_verified("gain", 6.0)`）对比 `tools/plant_flow_transient_capture.py:799-820` 的基础配置路径
- **最小复现**：
  ```bash
  .venv\Scripts\python.exe output\audit-independent-20260923\contrast_compare.py
  ```
- **事实**：E01–E03 的 `events` 中**没有** `camera_gain_verified`；E04/E05 有。像素统计：
  - B1 原始帧 `mean=44.7, std=2.52, 值域 33–61`；R1 `mean=39.5, std=3.55, 值域 29–59`；`r1-precheck/raw_00*.png` `mean≈40, std≈3.55`；`current-channel-preflight-20260923.png` `mean=43.9, std=3.48`
  - E04/E05 静帧与 `current-channel-gain6-20260923.png`、`short-plug-current-20260923.png`：`mean≈86.7, std≈6.8, 值域 66–129`
  - 即 gain 6.0 组均值约 **1.94×**、std 约 **2.7×** 于未设增益组
- **以及**：B1 整段 25 分钟（152,618 帧）内，第 0 帧与第 152,000 帧相比，**仅 0.40% 像素变化 >10 灰度级，0.000% 变化 >30 级**。
- **实际影响**：`docs/current_frame_review_20260923.md` 一类由长录像得出的「未见清晰液柱」观察，**与用户看到清晰短液柱的增益-6 影像不是同一成像条件**，不能用作「无液柱」的事实依据。任务书 §2.1 的这处提醒**得到定量证实**。
- **修复建议**：把相机增益/曝光作为一等采集事实写入 `session_summary` 与每段记录；跨运行比较前先校验成像参数一致，不一致即拒绝合并解释。

### P10（中）F04 摘要的 `raw_archive` 指向父目录，与自身录像不符

- **文件**：`output/r1-front4-20260923/attempt1_start_unverified/session_summary.json` → `frame_facts.raw_archive`
- **最小复现**：
  ```bash
  .venv\Scripts\python.exe output\audit-independent-20260923\verify_videos.py attempt1_start_unverified
  ```
- **事实**：F04 自身 `frames.ndjson` 有 973 条记录（index 0–972），其同目录 `raw_frames.mkv` 容器帧数 **973**，可解码、720×540、FFV1——**证明子目录录像确为 F04 自己的**。但摘要 `raw_archive` 写的是 `E:\MCS_QT\output\r1-front4-20260923\raw_frames.mkv`（父目录），该路径现为 E02 主运行的 11.7 GB / 121,994 帧录像。
- **实际影响**：按摘要字段取材的审计者会把 E02 的录像误归于 F04；反过来 F04 自己的 973 帧录像无法从摘要中发现。这是**唯一**一处 `raw_archive` 失效：E01/E02/E03 的该字段与文件一致（§4 已验证）。
- **推测**：F04 输出随后被移入子目录而摘要字段未同步（文件 mtime 一致，符合移动而非复制）。
- **修复建议**：`raw_archive` 落盘时用 `Path.resolve()` 的绝对真实路径；或在移动/归档后提供校验命令。

### P11（中）R1 复核索引用假定 100 fps 计算时间

- **文件**：`output/r1-front4-20260923/ten_point_review/sample_index.json`（每条为 `{point, video_time_s, frame_index, path}`）
- **最小复现**：
  ```bash
  .venv\Scripts\python.exe output\audit-independent-20260923\review_reports.py
  ```
- **事实**：`video_time_s` 恒等于 `frame_index/100`（首点 `frame_index=3000 → video_time_s=30.0`；末点 `118994 → 1189.94`）。而 E02 实测均帧率为 **99.719 fps**（`timeline.py`），故末点真实相对时间约 1193.3 s，**偏差约 3.4 s**。对照 B1 的同名产物 `fifteen_point_review/sample_index.json`，其字段为 `software_frame_index` + `host_monotonic` + `segment` + `segment_offset_s`（**正确**）。
- **实际影响**：R1 复核点的时间归属用了任务书明确禁止的「以固定播放帧率代替真实时长」，且与 B1 的同类产物约定不一致；偏差量级小，但约定错误会使后续推导不可比。
- **附带事实**：目录名为 `ten_point_review`（任务书亦称「十点」），实际 `sample_index.json` 有 **12** 个点（`point` 1–12）；B1 的 `fifteen_point_review` 为 15 个点，名实相符。
- **修复建议**：R1 索引改用 `host_monotonic`（`frames.ndjson` 已逐帧具备）与其墙钟锚点，字段与 B1 对齐。

### P12（低）`measurement_chain_status` 是硬编码字面量

- **文件**：`tools/run_r1_front4_live.py:61-65`
- **事实**：`physical_scale_validated: False`、`ten_point_automatic_detection_passed: False` 为常量，不随任何实际校验改变。
- **影响**：当前取值**恰好正确**，故不影响本期结论；但它不会在将来校验通过时变 `True`，属会误导后续读者的「看起来像结论」的常量。
- **修复建议**：改为从实际校验结果计算，或明确改名为 `not_evaluated`。

### P13（低）E04/E05 的 `sampling_premise_ok=false` 是硬编码，与 E01–E03 的同名布尔值语义不同

- **文件**：`tools/run_channel_validation_live.py:70-75`（`MeasurementSink.summary()` 内 `"sampling_premise_ok": False` 为字面量）对比 `tools/plant_flow_transient_capture.py:603-615`（由帧号缺口/重复/单调性计算）
- **实际影响**：E04/E05 的 `PREMISE_REJECTED` 是**构造出来的**，不是实测得出；它与 E01–E03 由「帧号缺口 4/342/243 帧」算出的同名 `false` 含义不同，混用会误读。
- **修复建议**：分别命名（例如 `measurement_premise_declared` vs `sampling_premise_ok`）。

### P14（低）采样窗 97.5 s ≠ 泵运行窗 90.0 s

- **文件**：`tools/run_channel_validation_live.py:36`（`self.start = time.monotonic()` 在构造时即计时）
- **最小复现**：
  ```bash
  .venv\Scripts\python.exe output\audit-independent-20260923\window_compare.py
  ```
- **事实**：a/b 采样窗均 97.5 s，泵运行窗 90.01 s；**36 个样本（7.6%）在 `start-infusion` 之前**，2 个在停泵之后。E05 的 17 个「被接受」行落在泵启动前。
- **影响**：`sampled_frames=476` 覆盖的不是 90 s 泵运行窗口；任何按「90 s」标称的速率/频率解释都会偏差。
- **修复建议**：摘要中分别给出采样窗与泵运行窗，并在逐行标注该行是否处于泵运行中。

### P15（低）E04/E05 的 plan 顶层 `steps` 仍是 B1 的五段 300 s

- **文件**：`tools/run_channel_validation_live.py:123-131`
- **事实**：`:127` 只清空了 `flow_plan.steps`；顶层 `"steps": ["B1 C baseline 300s", ...]` 原样保留，而实际只跑了一个 90 s 基线段。a/b 两份 plan 除 `output_directory` 外逐字节相同（已用 SHA256 与逐行 diff 确认）。
- **影响**：plan 与执行内容不一致；且 a/b 的 plan **完全没有记录两次运行的检测配置差异**（差异实际来自 20:39:5x 对 `config.py`/`detector.py` 的工作区改动与 `code_hash: null`）。
- **修复建议**：一并清空顶层 `steps`；把生效的检测参数摘要写入 plan。

---

## 4. 证据清单与时间线（A 部分）

### 4.1 录像与逐帧事实的一一对应：**通过**

`output/audit-independent-20260923/video_verify.log`（全部由本次审计独立生成）：

| 录像 | `frames.ndjson` 记录 | 容器帧数 | 分辨率/编码 | 抽查解码 | 结论 |
|---|---|---|---|---|---|
| `live-c-baseline-20260923/raw_frames.mkv` | 29996（index 0–29995） | 29996 | 720×540 FFV1 | 0/14998/29995 均可解 | MATCH |
| `r1-front4-20260923/raw_frames.mkv` | 121994（0–121993） | 121994 | 720×540 FFV1 | 可解 | MATCH |
| `v1-b1-20260923/raw_frames.mkv` | 152618（0–152617） | 152618 | 720×540 FFV1 | 可解 | MATCH |
| `r1-front4-20260923/attempt1_start_unverified/raw_frames.mkv` | 973（0–972） | 973 | 720×540 FFV1 | 可解 | MATCH |
| `lossless-recorder-selftest/raw_frames.mkv` | 1 | 1 | 72×54 FFV1 | 可解 | MATCH（自测，非实体实验） |

### 4.2 墙钟锚定与真实时间线

逐帧 `adapter_timestamp` 为 epoch 秒，可作墙钟锚点；`host_monotonic_timestamp` 为单调时钟。两者同帧并存，无需假定换算。

| 运行 | 首帧墙钟 | 末帧墙钟 | 采集跨度 | 实测均帧率 | 帧号缺口 | `lost_packet_count` 求和 |
|---|---|---|---|---|---|---|
| E01 `live-c-baseline` | 16:03:35 | 16:08:35 | **299.994 s（5.00 min）** | 99.989 fps | 4（3 事件） | 0 |
| E02 `r1-front4` | 17:06:44 | 17:27:08 | **1223.372 s（20.39 min）** | 99.719 fps | 342（292 事件） | 0 |
| E03 `v1-b1` | 18:01:51 | 18:27:20 | **1528.633 s（25.48 min）** | 99.840 fps | 243（232 事件） | 0 |

**E03 五段（观察窗逐段精确 300 s）**：

| 段 | label | q1/q2 (µL/min) | 观察窗 | 段内帧 | 段内缺硬件帧 | 墙钟 |
|---|---|---|---|---|---|---|
| 1 | C_baseline | 70/20 | 300.011 s | 29937 | **63** | 18:01:59→18:06:59 |
| 2 | Q1_50 | 50/20 | 300.011 s | 29984 | 16 | 18:07:04→18:12:04 |
| 3 | C_return_from_50 | 70/20 | 300.007 s | 29988 | 12 | 18:12:09→18:17:09 |
| 4 | Q1_80 | 80/20 | 300.009 s | 29987 | 13 | 18:17:15→18:22:15 |
| 5 | C_return_from_80 | 70/20 | 300.009 s | **29868** | **132** | 18:22:20→18:27:20 |

- 段间 ~5.3 s 泵写入/回读间隙**不计入**观察窗（`command_readback_monotonic` 后才起算 `observation_started_monotonic`）——这是正确设计。
- **缺口非均匀**：seg1+seg5 合计 195/243（**80%**）落在基线段与末段恢复段。
- 容器 `fps=100.0` 只是请求值；真实时长必须用单调时钟（本节即如此）。

**E02 四段**（70→70 同值重写→60→70，label 为 `C_baseline`/`C_same_value_write`/`L`/`C_return`，`L` 的 q1=60.0）：段内缺硬件帧分别为 124 / 59 / 60 / 83，段内帧 29876 / 29942 / 29941 / 29917，观察窗均 300.0 s。17:06:52→17:27:07。

**E01** 无 `segments`（`_run_bounded` 单线程有界运行），`frames.ndjson` 全部 29996 帧落在观察窗外——与摘要 `segments=[]` 一致。

### 4.3 采样前提判定

- E01/E02/E03 的 `sampling_premise_ok=false` 由 `FrameFactsRecorder.sampling_premise_ok()`（`plant_flow_transient_capture.py:603-615`）**实测计算**：任何帧号缺口即拒绝（零容忍）。缺口比例分别为 0.013%、0.28%、0.16%。
- E04/E05 的同名字段是**硬编码 false**（见 P13），语义不同。
- 三次长录像的 `frame_facts.timeline_source='host_monotonic'`、`timeline_assumed=false`、`duplicate_frames=0`、`non_monotonic_timestamps=0` ——时间轴非假定值。

### 4.4 待纠正的旧文档

`docs/current_frame_review_20260923.md` 仍含「895 项」「仅第一次诊断」「后期未见清晰液柱」等旧表述。其中**「后期未见清晰液柱」不可作为无液柱的事实依据**，理由见 P9（成像条件不同）。本次独立复跑的测试数为 **897 passed + 6 subtests passed**（§6）。

---

## 5. 短液柱人工对照与小样本统计（C 部分）

### 5.1 方法说明（重要局限）

任务书要求「先人工盲标原图，再查看算法候选」。**本次审计无法直接查看图像**：Read 工具对 `check_*.png` 与重编码后的 JPEG 均返回 `Unsupported Image`。因此改用**可复现的定量评估**替代目视标注：

1. 先只读原图的像素统计与**帧间/列间变化**（不看算法候选）：`contrast_compare.py`、`sanity_check_frames.py`；
2. 再看 `measurements.ndjson` 的候选与接受结果：`analyse_measurements.py`、`band_timing.py`；
3. 审计目录 `overlays/` 生成原图 JPEG 与叠加图。**初版叠加图存在坐标错误**（见 §12.1），已修正为 `overlays/*_v2_annotated.jpg`；旧图保留作历史。

因此下表的「真值」一栏只能给出**可从数据判定的属性**（是否在泵运行时出现、跨帧是否位置重复），**不含**「这是不是一个真实液柱」的目视真值。这是本报告最主要的未完成项。

**复审补充（2026-09-24）**：复审方能查看图像，目视 `a_check00`、`a_check05`、`b_check02`、`b_check05`、`b_check09` 后在管内不同位置**看到带圆弧端部的液柱状结构**（`b_check09` 尤为明显）。这**推翻**了本报告初版「两批静帧物理场景未变」的推断（见 §12.1 第 3 条）。静帧不能给出速度、对象身份、相组成或逐行真值，但足以说明仅凭整帧 `mean`/`std` 断言场景不变是不成立的。复审亦声明其自身属**知情复核**（已先读过本报告与叠图），不能称为独立盲标；因此**视觉真值一项在本轮仍未以盲标方式取得**。

### 5.2 E04（a）与 E05（b）对照表

| 项 | E04 (`...20260923a`) | E05 (`...20260923b`) |
|---|---|---|
| 采样行数 / 采样窗 | 476 / 97.51 s | 476 / 97.54 s |
| 泵运行窗 | 90.01 s | 90.01 s |
| 泵启动前取样 | 36 行 | 36 行（其中 **17 行被接受**） |
| 停泵后取样 | 2 行 | 2 行（其中 2 行被接受） |
| `selected_pixels` 非空行 | **0 / 476** | **349 / 476** |
| 被接受区间总数 | 0 | 489 |
| 含 ≥2 个接受区间的行 | 0 | 133 |
| `outline_checks` 原始候选行数 / 候选总数 | 473 / 483 | 475 / 1187 |
| 被接受区间长度 | — | min 18.0 / p25 29.0 / **med 51.0** / p75 84.0 / max 142.0 px |
| 长度直方图（10 px 分箱） | — | 10s:22, 20s:101, 30s:59, 40s:59, 50s:52, 60s:30, 70s:31, 80s:26, **90s:89**, 100s:7, 110s:3, 120s:4, 130s:4, 140s:2 |
| `geometry_source` | `visually_selected_session_proposal`（全部 476） | 同左 |

### 5.3 重复候选的位置与时间统计（P3 的量化）

> 下表只报告**候选在位置与时间上的重复**这一可计算事实。重复**不等于**同一对象，**不等于**假阳性；其物理来源见 P3「性质」一栏与 §12.1 第 2 条。

| 检验 | E04 | E05 |
|---|---|---|
| x≈[263,360] 出现在原始候选的行数 | ~45（含 x∈[262,265]×[357,360]，长度 93–98 px 共 4 类各 34–45 行） | 原始候选中 x=[263,360] 14 行等 |
| x≈[263,360] 出现在**被接受**的行数 | 0（未接受） | **93 / 349（27%）** |
| 该位置簇的出现时刻分布 | — | 0.403 s … 57.275 s（97.5 s 采样窗内） |
| 该位置簇长度 | 93–98 px | 93–98 px |
| 另一重复位置 x≈[407,433]（逆变换后为原图 x≈519–545，处延伸段） | — | 原始候选 12 行；被接受 10 行；1.5 s…54.0 s |
| 被接受区间起点直方图 | — | 0s:31, 50s:7, 100s:27, 150s:53, 200s:1, **250s:218**, 300s:62, 350s:64, 400s:21, 450s:5 → x∈[250,400) 占 **344/489=70%** |
| 十张静帧像素变化 | vs still0：0.37–0.70% 像素 >10 级，max 22–35 | vs still0：1.01–1.60% 像素 >10 级，max 40–46 |

**结论**：`frames_with_complete_plugs = 349` 中有 ≥93 行落在同一位置簇、≥19 行出现在泵未运行时。**该计数不能作为短液柱召回或精度的度量**；同时**不能**据位置重复断定这些候选是假阳性——重复生成、采样混叠、检测偏好、固定纹理、滞留对象均可产生位置聚集，判定需要同步真值（带帧号的对照图或连续短段录像）。

### 5.4 未完成的对照

任务书要求的以下条目**本次未能完成**，原因均为无法在该数据上取得独立真值：

- 人工标注完整对象两端、可见性、不确定范围；
- TP/FP/FN、precision/recall、端点与长度像素误差（无真值即无法定义分子分母）；
- 旧/新配置的单变量消融（只改长度阈值 / 只改像素参考 / 两者同改）。**替代**：本次从 `wall_check`（20:30:31，旧配置）与 `short_plug_check`（20:40:07，新配置）的派生产物确认二者源图**同为 `check_02.png`、管壁完全相同**，差异仅来自检测配置——`wall_check` 得 `raw_checks` 2 条、`selected_intervals=[]`；`short_plug_check` 得 `raw_checks` 4 条、`selected_intervals=[[261,319,58.0]]`。这给出了配置敏感的定性证据，但**不构成**任务书要求的阈值消融（两次运行时刻不同、且产生它们的代码已不存在，见 P2）。

---

## 6. 实际执行的测试与未测项目（E 部分）

### 6.1 本次真实执行的命令与结果

隔离设置（未设 `MCS_DATA_DIR` 会回落到 `D:/MCS_QT_Data` 真实配置）：
```bash
$env:MCS_DATA_DIR = "E:\MCS_QT\output\audit-independent-20260923\audit_data"
```

定向集（任务书 §4E 指定的 9 个文件）：
```bash
.venv\Scripts\python.exe -m pytest tests/test_current_frame_detection_gate.py tests/test_parallel_walls.py tests/test_short_rectified_plugs.py tests/test_generation_zone_detector.py tests/test_capture_detector_review.py tests/test_rectified_measurement.py tests/test_transient_capture_lifecycle.py tests/test_transient_capture_entrypoints.py tests/test_pump_flow_readback.py -q
```
**结果：205 passed in 29.75s**

全量：
```bash
.venv\Scripts\python.exe -m pytest -q
```
**结果：897 passed, 6 subtests passed in 129.97s (0:02:09)**

日志：`output/audit-independent-20260923/pytest_targeted.log`、`pytest_full.log`。

**隔离已确认**：`audit_data` 仅收到 5 个条目（`calibrations/incomplete_*.measurements.json|.mpc-*`、`data/disturbance_model.sqlite3`），未触碰真实数据目录。

**与任务书自报数字的关系**：任务书称「此前报告过全套 897 项及 6 项子测试通过」。本次为**独立复跑**，数字一致——但**这 897 项未捕获本报告 P1–P15 中的任何一条**。例如 P5 的退出码表达式、P6 的严格模式清除、P7 的曝光字段，均无对应断言。

### 6.2 无硬件安全回归逐项核查（对照任务书 §4E）

| 检查项 | 结论 | 依据 |
|---|---|---|
| 默认不连接硬件 | 通过 | `--execute` 未给出时打印干运行摘要并 `return 0`，且**先于** `mkdir`（`run_channel_validation_live.py:136-140`） |
| 时长越界拒绝 | 通过 | `:121-122` argparse 校验 90..300 + `ValidationSession.__init__` 二次校验 `:81-82` |
| 已有输出不覆盖 | 通过 | `:140` `args.output.mkdir(parents=True, exist_ok=False)` |
| 正常限时结束 | 通过 | E01–E05 实体运行 |
| 状态回读失败/超时 | 通过 | `pytest-lock-borrow-20260923` 内 `test_no_valid_frame_times_out_0`、`test_start_command_raising_sti0`、`test_pump_may_be_running_is_se0` |
| 相机断流 | 通过 | `NoValidFrameError` 路径 + `max_no_valid_frame_s` 上限 |
| 检测异常 | **部分** | sink 抛异常会被 reader 线程捕获并经 `_raise_reader_error()` 传播（`run_r1_front4_live.py:126-134`）；但 `_run_segments` 在发起 `update_flow_while_running` 前**不先检查** reader 错误，须等下一次 `_wait_segment` 才中止 |
| 写盘异常 | 通过 | `_finalize_outputs` 降级为 `OUTPUT_WRITE_FAILED`（`plant_flow_transient_capture.py:722-750`）；`test_downgraded_failure_record0`、`test_storage_backlog_failure_s0` |
| 用户中断 | 通过 | `except BaseException`（含 KeyboardInterrupt），`test_interrupt_enters_cleanup_0/1` |
| 锁冲突 | 通过 | `_acquire_locks` 固定顺序 + 回滚（`:692-711`）；`test_busy_device_lock_prevents0` |
| 停止失败不伪装成功 | 通过 | `_stop_pump_with_retries` 返回 `STOP_UNVERIFIED` 且 `verdict` 优先级最高（`:1074-1076`）；`test_stop_failure_gives_stop_u0` |
| 周期状态回读耗时是否计入段期限 | **是，计入** | `run_channel_validation_live.py:92` 的 `while self._clock() < deadline` 包住了回读。单次回读实测 ~0.3 s（`stop-attempt-1` 的 14779.314→14779.635），90 s 内 ≤3 次，影响小但未扣除 |
| 是否有仅写在计划中未执行的约束 | **有** | `frames_with_complete_plugs` 语义（P3）；`code_hash: null`（P2）；`steps` 未清空（P15）；`max_cumulative_delivery_each_ul=400` 与 90 s 实耗（0.105/0.030 mL）无矛盾 |
| `result.get("stop", {}).get(...)` 对 `stop=None` | **失败** | P5，已给出 `AttributeError` 复现 |
| 采集线程停止/释放 | 通过（附注） | `_reader_stop.set()` → `join(timeout=3.0)` → 超时记 `cleanup_errors`（`run_r1_front4_live.py:223-227`）；随后才 `_close_sink()`，顺序正确。附注：若 join 超时，reader 可能仍在向已关闭 sink 提交，该异常只进 `cleanup_errors`，不掩盖停泵结论 |
| 运行结束/测量验收/停泵分别报告 | **失败** | P5：入口只按停泵成败给退出码，三项未分别报告 |

### 6.3 未测项目与原因

- **未做任何真机操作**：无连接泵/相机、无流量下发（超出本次授权）。
- **未运行 GUI/前端**：P6 的端到端可达性因此为代码级推断，未实测。
- **未完成目视人工盲标**：Read 工具在本会话无法显示图像（见 §5.1），仅完成定量盲标；`overlays/` 已备好对照图。
- **未复跑 a/b 诊断**：产生它们的代码已不存在（P2），无法复跑。
- **未做单变量阈值消融**：缺真值，且两次运行的条件差异未被记录（P15）。

---

## 7. 总账：E01–E05、F01–F04、P01–P02、O01–O03、N01 逐项结论

| 编号 | 对象 | 实体执行 | 采集质量 | 测量精度 | 科学结论 |
|---|---|---|---|---|---|
| **E01** | 70:20 基线 5 min | 已执行（29996 帧，300 s） | 缺口 4 帧（0.013%），`sampling_premise_ok=false` | 未评估（无标尺） | 不可采信；`PREMISE_REJECTED` |
| **E02** | R1 四段 20 min | 已执行（121994 帧，4×300 s） | 缺口 342 帧（0.28%），**含 1 个 457.7 ms 连续空洞**（P8） | 未评估 | 不可采信；`PREMISE_REJECTED` |
| **E03** | B1 五段 25 min | 已执行（152618 帧，5×300 s，跨度 1528.633 s） | 缺口 243 帧（0.16%），80% 集中于 seg1/seg5 | 未评估 | 不可采信；`PREMISE_REJECTED`。**不得写成「B1 未执行」** |
| **E04** | 首次短时诊断 90 s | 已执行（9746 帧） | `sampling_premise_ok` 为硬编码 false（P13）；无录像，10 静帧 | 0 接受候选 | 不可采信；`PREMISE_REJECTED` |
| **E05** | 修改后短时诊断 90 s | 已执行（9745 帧） | 同上；10 静帧 | 349 行有接受候选，但存在显著位置重复（≥93 行同一位置簇）且 ≥19 行在泵未运行时（P3）；物理来源未经同步真值判定 | 不可采信；`PREMISE_REJECTED`，**不代表精度提升** |
| **F01** | 基线锁冲突 | 尝试失败，0 帧 | `stop=None` | — | 不能据 `touched_hardware=true` 推断已输液 |
| **F02** | 基线曝光回读失败 | 尝试失败，0 帧 | `stop=None`；错误原文 `CameraSetupError('相机 exposure 回读与设定不符：设定 80.0，回读 0.0，容差 4.0')` | — | 失败；此错误是 P7 的交叉证据 |
| **F03** | 基线首帧号校验失败 | 尝试失败，0 帧 | `stop=None`；`CameraSetupError('direct 相机帧未提供硬件帧号')` | — | 失败 |
| **F04** | R1 启动未确认 | 973 帧 + 录像 | 3 帧缺口；`PumpStartUnverifiedError('...sys_runstate expect=0x07, got=0x00, target_run_mask=0x06; safe stop verified')` | — | 不算完整一段；**不能断言未出液**。摘要 `raw_archive` 指向错误（P10） |
| **P01** | R1 预检抽样 | 索引 12 点多记录为 `video_time_s`（假定 100 fps，P11）；原图 `mean≈40, std≈3.55` | 供追溯 | 无 | 不是额外实体运行 |
| **P02** | 相机/曝光/增益/视野预检 | 静帧存在；`current-channel-preflight` 为未设增益（`mean=43.9`），`current-channel-gain6` 为 gain 6（`mean=86.6`） | 二者光度不可比（P9） | 无 | 不从图片推断泵状态 |
| **O01** | 管壁提议及全幅延伸 | 单帧诊断 | `status='single_frame_proposal_not_validated'`，`physical_scale_validated=false` | 无 | 全幅版为**同斜率直线外推**，x>514 无边缘证据（32% 带宽）；三份 proposal 的墙线：`proposal` 与 `gain6` **相同**（至 x=514），`fullspan` 延长至 x=700 |
| **O02** | R1 十二点 / B1 十五点复核（抽样） | 离线复核 | **抽样点 27/27 `not_measured`** | 无 | **定位失败是测量链失败，不等于录像中无液柱**（P1）。抽样点失败**不等于**已逐帧统计整段录像的定位成功率 |
| **O03** | 短时诊断离线复核 | `wall_check`（20:30:31，旧配置）与 `short_plug_check`（20:40:07，新配置）为**同一批资料的派生分析**，源图同为 `check_02.png`、管壁相同 | 不重复计入实体实验 | 配置敏感但非消融（§5.4） | 不足以支撑短液柱结论 |
| **N01** | 修复后 300 s 独立入口 | **未执行** | 入口 `run_channel_validation_live.py` 当前源码**从未以该形态运行过**（P2：事件名不符、无 `baseline-run-state` 记录、无 `C_baseline` 段记录）；仅做过 dry-run | — | 不否定 E01–E03 已有的 300 s 段 |

**观察长度合计**：E01+E02+E03+E04+E05 ≈ 5 + 20 + 25 + 1.5 + 1.5 = **约 53 分钟**，不含切换、启动、失败尝试与相机预检，**不是**连续输液时长。

---

## 8. 全天时间线与配置变化

| 墙钟 | 事件 | 证据 |
|---|---|---|
| 15:52–16:02 | E01 三次失败尝试 F01/F02/F03（锁冲突 → 曝光回读 0.0 → 无硬件帧号） | `live-c-baseline-20260923/attempt*.session_summary.json` |
| 16:03:35–16:08:35 | **E01** 70:20 基线 5 min，29996 帧 | 同上 + `timeline.py` |
| 16:39 | R1 预检抽样 P01（10 点 / 12 条索引） | `r1-precheck-20260923/sample_index.json` |
| 17:03:39–17:03:49 | **F04** R1 启动未确认（973 帧） | `r1-front4-20260923/attempt1_start_unverified/` |
| 17:06:44–17:27:08 | **E02** R1 四段 20 min，121994 帧 | 同上 |
| 17:29–20:07 | O02 十点复核（`localized_report` 12/12 失败） | `r1-front4-20260923/ten_point_review/` |
| 18:01:51–18:27:20 | **E03** B1 五段 25 min，152618 帧 | `v1-b1-20260923/` |
| 18:28–20:07 | O02 十五点复核（15/15 失败） | `v1-b1-20260923/fifteen_point_review/` |
| 20:01:00 | P02 预检 `current-channel-preflight-20260923.png`（**未设增益**） | 文件 mtime + 像素统计 |
| 20:17:35–20:17:35 | O01 短跨距提议（至 x=514） | `current-wall-proposal-20260923/proposal.json` |
| 20:18:29 | `current-channel-gain6-20260923.png`（临时增益 6） | 同 P02 |
| 20:22:47–20:23:41 | O01 gain6 提议（墙线与前者相同）→ **全幅延伸提议**（至 x=700） | 两份 `proposal.json` |
| 20:29:39–20:31:28 | **E04** 90 s 诊断（0 接受） | `channel-validation-live-20260923a/` |
| 20:30:31 | O03 `wall_check`（**旧配置**派生） | `...23a/wall_check/` mtime |
| **20:39:50–20:40:00** | **配置改动**：`config.py`（`generation_min_length_ratio` 2.50→**0.50** + 三个新 outline 参数）、`detector.py`、`capture_detector_review.py`、`review_current_wall_pair.py`、`preflight_current_channel.py` | 文件 mtime + `git diff` |
| 20:40:07 | O03 `short_plug_check`（**新配置**派生，1 个接受区间） | `...23a/short_plug_check/` mtime |
| 20:42:47 | `short-plug-current-20260923.png`（gain 6） | mtime |
| 20:44:42–20:46:31 | **E05** 90 s 诊断（349 行有接受候选） | `channel-validation-live-20260923b/` |
| **20:49:59** | `tools/run_channel_validation_live.py` **再次被改**（晚于两次运行） | mtime |

**补液记录**：两次 90 秒诊断的 plan 均写 `"User reports both syringes freshly refilled to approximately 1 mL"`（用户报告，**无液位读数**）。脚本内 `remaining_volume_each_ul=700` 自称 `"Conservative working estimate ... not a physical readback"`。**补液后未重新建立基线**，跨补液不可拼成连续恢复实验。

**配置变化是否被记录**：**没有**。plan 的 `software_readiness.code_hash: null`；a/b 两份 plan 除 `output_directory` 外逐字节相同（SHA256 + 逐行 diff 确认）；`measurements.ndjson` 的行字段（`hardware_frame_id`/`capture_monotonic`/`selected_pixels`/`outline_checks`/`geometry_source`/`physical_scale_validated`）不含任何配置指纹。

### 逐批次可复用数据清单

| 批次 | 可复用内容 | 前置条件 |
|---|---|---|
| E01 | `raw_frames.mkv` + `frames.ndjson`（帧↔录像已核对一致） | 只可作像素/时间线证据；无 plug 检测输出；成像条件见 P9 |
| E02 | 同上（121994 帧） | 同上；seg3 含 457.7 ms 空洞（P8） |
| E03 | 同上（152618 帧，时间线最完整） | 同上；seg1/seg5 缺口集中 |
| E04/E05 | 10 张静帧 + `measurements.ndjson` | 静帧可作目视素材；`measurements` 的接受集合不可作召回依据（P3） |
| P01/O02 | 原图 + 索引 + `localized_report` | 索引时间约定需先修正（P11）；`localized_report` 27/27 失败本身即结论 |
| O01/P02 | 三份 `proposal.json` + 预检/增益/短柱静帧 | 均为**目视提议**，非自动定位结果；全幅版含外推段 |

---

## 9. 覆盖缺口（不以缺资料当作通过）

1. **产生 E04/E05 的源码不可得**（P2）——最重要的覆盖缺口。
2. **无 `code_hash`**：任何运行都无法绑定到确定的代码状态。
3. **图像无法在本审计会话目视**：短液柱的目视真值、重复候选的物理来源、全幅延伸段与原图边缘的逐位置比对，均**未完成**（§5.1、§5.4）。复审方虽可看图，但其复核属**知情复核而非盲标**。特别地：**不得因原提议止于 x=514 就断言右侧无图像边缘证据**——这需要逐位置对照，本轮未做。
4. **整段录像的逐帧定位成功率未统计**：`localized_report.json` 只覆盖 27 个抽样复核点（§3 P1）。
5. **无现场液位读数**：剩余量、补液量均为用户报告与估算。
6. **无 stdout 日志**：a/b 与 E01–E03 均未保留入口进程的 stdout，`self.log(...)` 输出的 `baseline_elapsed_s` 等运行期信息不可得。
7. **无逐帧图像块索引**：`frames.ndjson` 的 `image_ref` 显式为 `null` + `"未配置数据块索引"`。逐帧↔录像的对应**只能靠容器帧序与记录序一致**来间接支持（§4.1 已验证这一点）；静帧 `check_NN.png` 与具体测量行的对应**未被持久化**（文件名与行内均无相互引用），故不能按文件序号猜时间或宣称逐帧精度通过。
8. **未运行前端**：P6 的可达性为代码级推断。
9. **未做阈值消融**（§5.4）。
10. **未评估 50/80 优劣、响应时间、基准恢复**：无真实阶跃数据，且像素数据不能套用 µm/s 阈值。

---

## 10. 已完成与未完成清单

**已完成（实体执行层面）**
- E01（5 min）、E02（4×300 s）、E03（5×300 s，25.48 min 跨度，152618 帧）三组长录像**已跑完并留原录像**；三段录像的容器帧数与本帧事实记录**逐段一致**（本次独立核对）。
- E04、E05 两次 90 秒诊断已跑完。
- 全部 5 次运行 + F04 的**停泵均回读确认**（`STOPPED/verified=true`）。
- 三次失败尝试 F01/F02/F03 与一次启动未确认 F04 均已留档。

**未完成（测量有效 / 实验通过层面）**
- **B1 五段实体执行完成，但科学验收未通过**；E01/E02 同。
- **工况变化确实已经执行过**：B1 完成了 70→50→70→80→70 五次流量写入（每次回读确认），R1 完成了 70→70（同值重写）→60→70。这**不是**「从未发生」。
- **但没有可信的尺寸响应数据**：五段录像的 `measurement_chain_status` 均为 `physical_scale_validated=false`、`ten_point_automatic_detection_passed=false`，且三次长录像**完全没有** plug 检测输出（只有 `frames.ndjson` + 录像）。因此「已执行的流量阶跃」与「可用于尺寸-流量建模的响应数据」必须分开表述：前者成立，后者不存在。
- 两次 90 秒诊断同样未通过阶段 A。
- 修复后的新一轮独立 5 分钟基线**未执行**，且其入口**从未以当前形态运行过**。
- 当前帧定位 0/27（抽样）；短液柱无精度结论；物理标尺未建立；R1 十点精度验收未通过。

**必须区分的表述**：「已跑完」≠「测量有效」≠「实验通过」。

---

## 11. 放行建议

1. **不放行**正式多组阶跃实验。阻塞项按优先级：
   - P1：被抽样的 27 个复核点**全部**未取得有效定位（**不等于**已逐帧统计整段录像的成功率）；
   - P2：产生两次 90 秒诊断的代码在本次可访问范围内找不到，短液柱证据不可复现；
   - P3：`frames_with_complete_plugs = 349` 这个**候选帧统计本身成立**，但它不能代表独立液滴数，也不能代表检测精度。
2. **建议先做**（按顺序）：
   - 把入口脚本纳入版本控制并加 `code_hash` 落盘（解 P2、P15、P12）；
   - 修正 P5 的退出码与 `stop=None` 处理，重新做一次 300 s dry-run + mock 安全回归。**注意**：无需耗液即可修的缺陷**不止**退出码一处——像素几何入口统一（P4）、严格模式保持（P6）、运行追溯与帧绑定（本报告 §9 第 7 条）同样可以完全离线修复，不应把退出码当成唯一的离线段；
   - 用**修正版** `overlays/*_v2_annotated.jpg` 完成目视比对：确认全幅延伸段（原图 x>514）是否截断管腔或引入管外背景、确认扶正坐标 x≈[263,360] 重复候选簇的物理内容。注意静帧未绑定测量行帧号，这些图只能展示**区域位置**，不能冒充某一行的检测叠图；
   - 从长录像取**连续短片段**（而非孤立帧）核查管壁证据与内部运动；若可见结构不足，明确记为「不能验证」，不要靠放宽阈值取得非零计数；
   - 统一线上与离线的像素参考（解 P4），使「短液柱可识别」在线上可复现；
   - 统一 gain/曝光并写入摘要（解 P9），之后才可跨运行比较。
3. **物理标尺未通过前，只可建议额外的像素级诊断，不放行任何 µm 尺寸优化。** 本审计的完成**不构成**任何启动硬件的授权。
4. 发现阻塞缺陷后，**不要用更多实体实验消耗液体来掩盖测量链问题**；上述第 2 条前两项均无需上机。

---

## 12. 复审更正记录（2026-09-24）

复审文档：`docs/codex_independent_validation_review_20260924.md`。复审维持「不放行」结论，但对本报告的因果判断提出 5 处更正与 1 处**产物 bug**。**逐条处理如下；凡标注「已更正」的，§1–§11 正文已同步修改。**

### 12.1 被推翻或收窄的表述

| # | 复审意见 | 我的判定 | 正文处置 |
|---|---|---|---|
| 1 | `make_overlays.py` 把**扶正图**横坐标直接当**原图**横坐标画竖线，缺逆透视变换；红线 x=263/360 未指向所统计区间 | **接受，确为我的 bug** | 已修正（§12.2）。原图已在 §5.1 标注；§2 全幅行改引 v2 |
| 2 | 重复区间 ≠ 同一个静止对象；`band_is_fixed.py` 分析的是 B1/R1 而非 E05 同步录像，且成像条件不同，不能替代对象身份验证；泵启动前的候选可以是真实残留对象，不是假阳性的充分证据 | **接受** | P3 标题与正文重写；「系统性固定假阳性」「至少 27% 假阳性」全部删除；改为「重复候选异常，物理来源未经同步真值判定」；明确声明 `band_is_fixed.py` 结论**不予引用** |
| 3 | 整帧 `mean`/`std` 相近**不证明**物理场景不变；复审目视 a_check00/a_check05/b_check02/b_check05/b_check09 可见带圆弧端部的液柱状结构 | **接受**（复审能看图，我不能） | §1、P3 第 1 条、§5.1 均已更正；删除「已证明该变化来自检测配置而非场景」这一表述 |
| 4 | 35 px 是提议几何的**输出**，与提议自洽不等于墙线已通过真实性验证；不得改称「真实管壁间距已证实」；像素索引跨度与几何长度的换算应显式说明 | **接受（措辞收紧）** | P4 相应段落重写并补换算说明 |
| 5 | 静态提议延伸到 x=700 的「认证证据不足」成立，但**不能断言原图右侧没有边缘证据**——目视图右侧仍见条带与液柱轮廓，需逐位置对照 | **接受** | §2 全幅行与 §9 第 3 条已更正为「延伸段未经逐位置验证，本轮未判定」 |
| 6 | 27 个抽样点失败 ≠ 已逐帧验证整段录像定位成功率为零 | **接受** | P1 标题与正文已加范围限定；§9 增列该项为独立缺口 |
| 7 | 当前源码搜索为负只能说明「已找到的当前源码与旧运行不一致」；mtime 与未跟踪状态本身不能证明所有历史副本均不存在 | **接受（收窄）** | P2 的「无任何源码记录」改为「在本次可访问范围内（工作区 + git 跟踪 + 全仓文本搜索）找不到」 |

### 12.2 产物 bug 修正（已在本次执行）

复审的坐标指控经**独立复算确认**：`wall_line_quad(720,540)` 得 `output 587×35`；扶正 (263,17)→原图 (375.213, 273.442)、(360,17)→(472.127, 281.087)，**与复审数字小数点后三位一致**。原图线的真实扶正坐标仅约 151/248，即**错位约 110 px**。

- 新增 `output/audit-independent-20260923/verify_coord_claim.py`（复算脚本）与 `make_overlays_v2.py`（修正版）。
- 修正版：扶正坐标区间四角经逆变换映回原图**四边形**；在图上标注坐标系名称；并在图内说明「仅区域位置，未绑定测量行」。
- 输出 `overlays/*_v2_annotated.jpg`（8 组）。**v1 图保留作历史，未删除**（v1 文件名不含 `_v2`）。
- 扶正↔原图对照（供引用）：`rect 0→orig 113.36`、`263→375.21`、`360→472.13`、`407→519.15`、`433→545.18`、`586→698.63`。
- 该 bug **不影响** §5.2/§5.3 的统计（那些数字一律以**扶正坐标**记录并已如此标注），只影响叠加图的指向。

### 12.3 我维持原判、复审亦未提出异议的部分

- **P5 入口退出码**（只按 `stop.verified` 判定；`stop=None` 触发 `AttributeError`，已给可执行复现）——复审同意「实验验收失败与成功停泵必须分别表达」。
- **P6 严格定位模式会被前端几何设置静默清除**——复审同意「风险在代码上成立」，并同意未运行 GUI 故未端到端确认。
- **P4 中 `detect()` 不传 `channel_width_px` 而诊断工具传入**——复审同意「同一套算法存在像素参考来源不一致」。
- **P2 的实质**（当前源码与产生 E04/E05 的代码不一致，事件名与段记录均对不上）——复审同意「只能说明已找到的当前源码与旧运行不一致」，即该实质成立，仅收窄了绝对化措辞。
- **P7 逐帧 `exposure_time_us` 恒 0 且不自我标注**、**P8 E02 的 457.7 ms 连续空洞**、**P9 增益导致长录像与诊断不可比**、**P10 F04 `raw_archive` 指向错误**、**P11 R1 复核索引假定 100 fps**、**P12/P13 硬编码字面量**、**P14 采样窗 ≠ 泵运行窗**、**P15 plan 顶层 `steps` 未清空**：复审**未逐条独立验证**这些条目。
  - **必须区分的表述**：复审「未提出异议」**不等于**复审已独立确认它们成立。上列各条的结论仍由**本报告**的单方证据支持（各自的 `文件:行号` 与最小复现命令已给出），属于**待复核**状态；不得写成「Codex 已验证 P7–P15」。
- **§4 的录像↔逐帧事实一致性核对**（5 段 MATCH）与 **§6 的测试执行记录**：复审未提出异议；复审亦明确声明其**未复跑**测试，故 §6 的数字是**本轮审计自行执行**的结果。

### 12.4 对复审「下一步最小工作包」的回应

复审提出的 5 项中：
- 第 1 项（修正叠加坐标、区分候选帧数/对象数/流动对象数、删去超出证据的定性）——**本次已完成**（§12.1、§12.2；P3 修复建议已改为按帧数/位置簇数/轨迹对象数分别报告）。
- 第 2 项（统一像素几何入口、修复严格模式保持与退出码、补无硬件回归）——属**产品代码改动**，本轮授权为「只读审计 + 无硬件测试，发现问题先报告，不顺手修改产品代码」，故**未执行**，待明确授权。
- 第 3–5 项（连续短片段核查、运行快照与 SHA-256、离线回放验证后再谈现场补采）——属后续工作，本报告在 §11 中已列为建议次序。

---

## 附录：审计目录内容索引

`output/audit-independent-20260923/`
- 脚本（初版）：`summarize_sessions.py`、`dump_evidence.py`、`verify_videos.py`、`timeline.py`、`gap_shape.py`、`compare_proposals.py`、`analyse_measurements.py`、`band_timing.py`、`band_is_fixed.py`、`sanity_check_frames.py`、`contrast_compare.py`、`window_compare.py`、`list_summaries.py`、`review_reports.py`、`localized_status.py`、`make_overlays.py`
- 脚本（复审后新增）：`verify_coord_claim.py`（复算复审的坐标指控）、`make_overlays_v2.py`（修正版叠加图）
- 日志：`pytest_targeted.log`、`pytest_full.log`、`video_verify.log`
- 统计：`column_stats.json`
- 对照图：`overlays/*_plain.jpg`、`overlays/*_annotated.jpg`（初版，**坐标有误，保留作历史**）、`overlays/*_v2_annotated.jpg`（修正版，8 组）、`b1_frame0_rectified_norm.png`
- 隔离数据：`audit_data/`（测试用，未触碰真实配置目录）

**状态声明**：本轮审计未修改任何产品代码、配置或原始实验记录，未连接泵或相机。报告已修订至含复审更正；复审文档本身（`docs/codex_independent_validation_review_20260924.md`）为他人新增，未改动。

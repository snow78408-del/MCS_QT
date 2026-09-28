# 采集安全与测量可信度：复查整改交付（2026-09-20）

对应复查任务书 §1–§6 与随后的设计决定。**未提交 Git，未开展现场实验**；全部验证使用 mock，
未连接真实设备。

## 〇、状态口径

本文件只用四种状态，**不使用「全部完成」这类说法**：

| 状态 | 含义 |
| --- | --- |
| **已实现** | 代码已落地，但未必有测试、也未必接进调用路径 |
| **mock 已验证** | 有测试覆盖，测试中未连接真实设备 |
| **尚未接入** | 实现存在，但没有接到实际调用路径上 |
| **实机未验证** | 没有任何真实设备上的证据 |

## 一、逐项状态

| 条目 | 状态 | 依据 |
| --- | --- | --- |
| §1 实机入口封印（模式互斥、无参数不出手、计划完整校验、门槛清单） | **mock 已验证** | `tests/test_transient_capture_entrypoints.py`：子进程哨兵证明零设备调用，另有哨兵自检 |
| §2 设备生命周期重排、`pump_may_be_running` 前置、清理路径、`STOP_UNVERIFIED` | **mock 已验证** | `tests/test_transient_capture_lifecycle.py` |
| §2 会话接进 `--live` | **尚未接入** | `run_live` 仍只做校验并返回 `LIVE_BLOCKED` |
| §3 逐帧采集事实／逐命令记录／时间戳来源分离 | **mock 已验证** | `tests/test_transient_capture_facts.py` |
| §3 时间轴资格检查与**实测间隔**换算速度 | **mock 已验证** | `tests/test_transient_capture_timeline.py`（含「真实间隔加倍 ⇒ 速度减半」） |
| §3 实机运行中真正落盘逐帧事实 | **尚未接入** | 会话尚未注入 `FrameFactsRecorder` |
| §4 速度／稳定性输出分离、`ALIAS_UNRESOLVED`、标定门控、稳定窗口完整驻留 | **mock 已验证** | `tests/test_plant_flow_transient_capture.py` |
| §5 严格回读解析、缺失回读也拒绝 | **mock 已验证** | `tests/test_pump_flow_readback.py`（23 例） |
| 设备锁模块（OS 建议锁、锁目录解耦、进程内唯一、句柄登记、多设备顺序与回滚） | **mock 已验证** | `tests/test_device_lock.py` |
| 泵连接入口接线（`PumpClient.connect/disconnect`，锁键＝规范化串口且不含地址） | **mock 已验证** | `tests/test_device_lock_wiring.py` |
| 相机连接入口接线（`open_selected` / `test_device` / `_reconnect` / `close_selected`） | **mock 已验证** | `tests/test_camera_device_lock.py` |
| 主程序与脚本的**跨进程**竞争（走真实连接入口） | **mock 已验证** | 上述两个文件中的子进程竞争用例 |
| 采集脚本侧取同一批设备锁 | **尚未接入** | `LiveCaptureSession` 已支持多锁与固定顺序，但 `--live` 未接线 |
| 任何环节的真实设备行为 | **实机未验证** | 本轮无现场实验 |

## 二、修改清单（本轮）

### 时间轴（复查意见第 3 条）

| 文件 | 改动 |
| --- | --- |
| `tools/plant_flow_transient_capture.py` | 新增 `FrameTiming`：主机接收时刻／墙钟／设备 ticks／硬件帧号**分开记录，互不顶替**。 |
| 同上 | 新增 `qualify_window()`：逐窗口检查时间戳有限、**严格递增**、采样间隔均匀（不一致即视为丢帧、**不用平均间隔掩盖**）、设备帧号连续；合格时返回**实测间隔**。 |
| 同上 | `TransientTracker.push(frame, *, timing=...)` 取代原先的 `timestamp=`——**删掉了「随便传一个时间戳就当作采集时刻」的入口**，避免把墙钟或设备 ticks 误标成采集时钟。 |
| 同上 | `measure()` 用窗口实测间隔调用 `estimate_velocity` 并换算速度；不合格窗口 `px_per_second` 为 `None`，样本带 `timing_ok`/`timing_reason`/`dt_s`/`dt_source`/`dt_assumed`。 |
| 同上 | `_physical(screened, dt)` 改为**用传入的实测间隔**换算（原先用 `self.dt`，这正是「标签换了、速度仍按设定帧率算」的原因）。 |
| 同上 | 只要存在不合格窗口，`analyse` 即判定时间轴不可用：不给任何稳定平台，结论为 `PREMISE_REJECTED`。 |

### 两处明确缺口（复查意见第 4 条）

| 文件 | 改动 |
| --- | --- |
| 同上 `_apply_feature_verified()` | 在容差比较**之前**拒绝非有限的设定值与回读值。`abs(nan - target) > tolerance` 恒为 False，只比差值会让 NaN 静默通过。 |
| 同上 `_finalize_outputs()` | 落盘失败不再只追加错误：结论降级为 `OUTPUT_WRITE_FAILED`、`completed=False`，再尝试保存降级后的失败记录；连失败记录都存不下时在返回值里显式标出 `failure_record_saved=False` 与原因。 |

### 设备锁（设计决定第 1 条）

| 文件 | 改动 |
| --- | --- |
| `backend/runtime_paths.py` | 新增 `device_lock_dir()`：**刻意不经过 `MCS_DATA_DIR`**——否则不同数据目录的进程会各拿一把锁，互斥失效。 |
| `backend/device_lock.py` | 新增 `normalize_port_key()`（`COM3`/`com3`/`\\.\COM3` 归一，**地址不入键**）、`camera_lock_key()`（稳定设备标识）。 |
| 同上 | 进程内登记表：同一进程内同一把锁只能有一个持有者（Windows 字节区间锁在进程内可重复取得，不能让「同进程」当共享）。 |
| 同上 | 登记表**同时持有文件句柄**：`DeviceLock(key).acquire()` 即使调用方不保留对象，也不会被引用计数回收句柄而静默丢锁（这条是实测踩到的：holder 脚本写成 `DeviceLock(...).acquire()`，子进程打印了「已持有」却什么都没锁住）。 |
| 同上 | `acquire_device_locks()`/`release_device_locks()`：**固定顺序**（内部排序，不信任调用方顺序）获取多把锁，任一失败即回滚本次已取得的。 |
| `backend/pump_hardware/client.py` | `connect()` 在打开串口**之前**取锁，连上才保留；任何失败路径（含奇偶校验全部失败、异常、中断）都释放。`disconnect()` 关闭串口后释放。锁键＝`normalize_port_key(port)`。 |
| `backend/pump_hardware/service.py` | `connect_and_probe()` 单独捕获 `DeviceLockError`：**换奇偶校验重试没有意义**，立刻返回并记 `failed["device_lock"]`，不再被误报成「串口打开失败」。 |
| `backend/vision/cameras/manager.py` | 三处 open 全部覆盖：`open_selected()`（主运行路径）、`test_device()`（自有 open，已持有同设备时**借用**而不是再取一把）、`_reconnect()`（自动重开，锁随连接保留）。`close_selected()` 关闭后释放；打开失败与重连彻底失败时都释放。 |
| 同上 | 顺带修正两处旧问题：打开失败时不再留下半开的适配器（`_fail_open` 关闭并释放）；换设备时**先关旧连接再换锁**，消除「旧设备锁已放开、句柄还开着」的窗口。 |

### 采集链标尺边界（设计决定第 2 条）

| 文件 | 改动 |
| --- | --- |
| `tools/plant_flow_transient_capture.py` | `LiveCaptureSession(..., locks=[...])`：按固定顺序获取多把设备锁，失败回滚，释放逆序进行。 |
| 同上 | 新增两条结构性回归测试：采集链**不得**有 `pixel_to_micron` 默认值，且**不得**引入 `flow_locking` 的物理输出入口（`measure_flow`、`ChannelLockFlow`，其默认标尺是 1.725）。 |

## 三、设备锁设计说明

* **生命周期**：锁跟随**实际持有的设备连接**，不绑定 `initialize_system()/stop()`。相机预览与设备测试会
  提前打开设备，`stop()` 也不保证释放串口，因此锁在**后端连接入口**取、在**连接关闭后**放。
* **暂停与「已停止但仍连着」继续持锁**：只要连接还在，锁不放。
* **锁键**：泵＝规范化串口标识（同一串口不同地址仍是同一台设备）；相机＝稳定设备标识。
* **多设备**：固定顺序获取，后续失败即回滚本次已取得的。
* **锁目录**：机器／用户级运行时位置，与 `MCS_DATA_DIR` 解耦；GUI 实例锁（`application.lock`）保持原用途、不合并。
* **不覆盖**：跨用户争用（Windows `LOCALAPPDATA`、POSIX 按 uid 分目录）。

## 四、生命周期说明（§2）

```
校验计划（不通过则完全不碰设备）
→ 按固定顺序获取全部设备锁（失败即回滚）
→ 连接并核验初始状态
→ 配置相机并回读（未知特征名是静默 no-op，必须比对回读值）
→ 开始连续采集，保存指令前基线
→ 写泵参数并验证
→ 【置 pump_may_be_running】发出启动指令并验证
→ 有界运行（会话时长 / 无有效帧 / 存储积压三重上限）
→ 停泵并验证（有限重试，全部在锁内）
→ 结束采集、关闭设备
→ 逆序释放设备锁
→ 落盘会话摘要与逐命令记录（失败则结论降级）
```

不变量：`pump_may_be_running` 只要可能发出过启动指令就为真，此后任何退出路径都必须执行停泵验证；
停泵未确认 ⇒ `STOP_UNVERIFIED` + 现场确认提示，不自动重启；只有「采集完整」且「停机确认成功」才报告完成；
原始异常与清理异常分开保存。

## 五、实际测试结果

- 命令：`.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`
- **全量：`693 passed, 6 subtests passed in 88.94s`，0 失败 / 0 跳过 / 0 错误**（最终冻结修订上的实测值）
- 本轮新增测试文件：`tests/test_device_lock.py`、`tests/test_device_lock_wiring.py`、
  `tests/test_camera_device_lock.py`、`tests/test_transient_capture_timeline.py`
  （另扩写了 `test_transient_capture_lifecycle.py`、`test_transient_capture_facts.py`、
  `test_transient_capture_entrypoints.py`、`test_plant_flow_transient_capture.py`）
- **复审提示**：`tools/plant_flow_transient_capture.py`、`tests/test_plant_flow_transient_capture.py`
  与 `backend/device_lock.py` 在复审基线里**未被 git 跟踪**，改动落在其中，需直接读文件或先 `git add`。
- 未删除任何失败测试、未弱化校验、未改动验证门槛。§4 与时间轴改变了输出契约，因此相关断言随契约更新
  （新断言检查的是**更严格**的规则，例如 `ALIAS_UNRESOLVED`、窗口不得跨过失效样本、速度必须用实测间隔）。

## 六、剩余限制

1. **实机路径仍禁用，且按能力验收、不按适配器名称放行。**`--live` 会列出全部未满足项后退出。
   `direct` 只是**候选**实现，不能因为「强制 direct」就启用。放开前必须逐项核验：
   实际设备身份、时间信息、帧号、参数回读、记录能力。
2. **默认（legacy）相机路径的能力缺口**：只填 7 个字段，硬件帧号／硬件时间戳／主机单调时钟／曝光／丢包
   全部不填，且 `frame_id` 是纯软件计数器、**永不缺口**——因此实机运行既检不出丢帧、也拿不到可用时间轴。
   软件帧号不能证明硬件没丢帧；墙钟递增也不等于可靠采集时钟。
3. **`direct` 路径的设备标识不稳定**：`hikrobot_direct.py` 写死 `unique_id=f"HIKROBOT:DIRECT:{index}"`，
   基于枚举序号，重枚举可能变化。
4. **计划缺少设备身份字段**：没有相机 `unique_id` 与泵串口/地址，无法核验「实测设备与计划一致」。
5. **采集脚本侧锁尚未接入**：会话已支持多锁与固定顺序，但 `--live` 未接线。
6. **跨用户设备争用不在范围内**（锁目录按用户隔离）。

## 七、登记为后续整改（**不得称为已清理**）

1. **独立库 CLI 仍以默认标尺输出物理量**：`backend/vision/flow_locking.py` 的 `main()` 在 `:838` 调用
   `measure_flow(..., pixel_to_micron=args.scale, ...)`，而 `--scale` 默认 `1.725`、`--rate` 默认 `320`。
   按设计决定本轮只处理采集链及其实际调用路径，未改动该库 CLI。
2. **界面占位比例**：`frontend/qt_app.py` 的 `get("pixel_to_micron", 1.0)`（`:760`、`:815`、`:875`、`:876`）
   用于通道几何标定与预览，与流量结论分开；`:876` 已带 `fallback_to_configured_scale` 标记，
   `:760`/`:815` 没有，需要与有效标定状态显式区分。

**未提交 Git；未开展现场实验。**

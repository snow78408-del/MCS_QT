# 优化方向清单（2026-09-20）

本文是一次只读审查的结论汇总，目的是把「哪些地方值得改、为什么、证据在哪」固定下来。
本轮**未修改任何业务代码**，只做了调查、测试和仓库卫生检查。

## 审查方法与可信度

| 结论类型 | 来源 | 可信度 |
| --- | --- | --- |
| 测试基线与运行结果 | 实际执行 `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` | 实测 |
| git / 工作区状态 | 实际执行 `git status`、`git check-ignore`、`git ls-files` | 实测 |
| 文件行数、目录体量 | 实际统计 | 实测 |
| 关键文件内容（`AGENTS.md`、`README.md`、`TODO.md`、`pyproject.toml`、`.gitignore`、`tests/test_flow_locking.py`） | 逐行读过 | 实测 |
| 静默吞异常清单、重复实现、god object 行号 | 静态审查（未逐条运行验证） | 需复核 |

## 当前基线

- 测试：`544 passed, 6 subtests passed in 74.87s`，0 失败 / 0 跳过 / 0 错误
  （审查时为 `525 passed`；本轮两个修复各带回归测试，P0-2 新增 10 个、P0-1 新增 9 个）。
  **`TODO.md` 里写的 `117 passed, 5 subtests passed` 已过期 4.5 倍。**
- 工作区：17 个已跟踪文件被修改，52 个未跟踪条目，尚未提交。
- 已跟踪文件 218 个，其中**没有任何生成物**（无 `.pyc`、`.sqlite`、`.log`、`.avi`）。
- 全仓 `sk-` 密钥模式 **0 命中**，无硬编码凭证。

总体上这是一个维护得不错的项目：`.gitignore` 覆盖全面、测试隔离机制到位、控制循环的线程模型正确。
问题集中在三处：**标定逻辑双份实现、硬件异常被静默吞掉、以及大量文档与死代码的熵**。

---

## P0 · 影响正确性与硬件安全

### P0-1 标定记录装配逻辑重复，且两条分支已经开始漂移

**先更正一个误判**：这两处**不是竞争实现**。`build_plant_calibration_result`（`calibration_experiment.py:756`）
本身就是 dispatcher，开头即按辨识模型分派：

```python
if config.identification_model == "quadratic_response":
    from .nonlinear_calibration import build_nonlinear_calibration
    return build_nonlinear_calibration(...)
```

两个函数服务于两种**不同的**辨识模型（`linear` 的 FOPDT 增益 vs `quadratic_response` 的响应曲面），
由 `config.identification_model` 选择。两条分支都是各自模型下的权威实现，**不能合并、不能删任何一条**。

**已修复（2026-09-20）**：抽出公共装配函数 `build_calibration_record`
（`backend/pid_control/calibration.py`），两条分支各自只剩一次调用。四个真正漂移的字段改为
**必填参数、不给默认值**，调用方物理上无法再静默漏建。同时统一了 `calibration_id` 的生成、
`config` 派生值的归一化，以及两条分支原本各写一遍的固定字段
（`schema_version=3`、`measurement_region="generation"`、`controller_kd=0.0`、
`flow_measurement_kind="device_parameter_readback"`）。回归测试见
`tests/test_calibration_record_builder.py`。

| 字段 | `linear` 分支 | `quadratic_response` 分支 | 是否真漂移 |
| --- | --- | --- | --- |
| `controller_kd` | `0.0`（`:943`） | 不传 → 默认 `0.0` | **否**，取值相同 |
| `flow_measurement_kind` | `"device_parameter_readback"`（`:979`） | 不传 → 同默认值 | **否**，取值相同 |
| `q1_response_delay_ms` / `q2_response_delay_ms` | 实测中位数（`:950-955`） | 不传 → `0.0` | **是** |
| `baseline_generation_frequency_hz` / `baseline_diameter_cv` | 实测中位数（`:969-978`） | 不传 → `0.0` | **是** |
| `calibration_id` | `plant-cal-{stamp}-{session前8位}`（`:904`），可回溯 | `plant-cal-{uuid}`（`nonlinear_calibration.py:141`），**不可回溯** | **是** |

> 本文初稿把上表前两行也列为漂移，**那是错的**：`linear` 显式传的值恰好等于 dataclass 默认值，
> 属于冗余书写而非漂移。核对默认值（`calibration.py:37-75`）后才分辨出来。

**「拿不到」还是「漏了」——答案是漏了**：两个函数接收的是同一个
`PlantCalibrationMeasurement` 类型（`build_plant_calibration_result` 原样透传 `measurements`），
`channel`、`response_detected`、`response_delay_ms`、`baseline_observations` 两条分支都拿得到，
`linear` 就是用这些算的。所以不存在「数据拿不到」的情况。

**仍待确认的语义问题**：per-channel 延迟与基线生成频率/CV 是否对 `quadratic_response` 模型有意义。
目前二次分支显式传 `0.0`（与修复前的实际落值一致）并在调用点注明；若要改为按同一份 measurements
实测补建，需先确认响应曲面模型是否需要这些量。

**本次另发现的第 5 处漂移**：`linear` 传归一化后的值（`str(...).strip()` / `float(...)`），
`quadratic_response` 传 `config` 原始值。若 `plant_id` 带尾随空白，二次分支的记录会原样保留。
现已统一由 builder 归一化。

**为什么重要**：标定是闭环控制的授权前提。漂移的字段里包含延迟参数，
而这类缺失不会报错——只会安静地落默认值。

### P0-2 硬件故障被静默吞成正常读数

**已修复（2026-09-20）**：解析拆成 `_parse_channel_flow`（核心）/ `flow_from_channel_params`
（宽松，行为与改动前逐位一致，供日志与展示）/ `flow_from_channel_params_strict`（严格，专供回读校验）。
新增 `ChannelFlowParseError` 以区分「无回读」与「回读损坏」。`orchestrator/service.py` 新增
`_verified_flow_actual()`，两个专门做回读校验的 `_apply_calibration_flow` 与 `_apply_optimizer_flow`
改用它——损坏回读现在抛错，不再被替换成指令值。回归测试见 `tests/test_pump_flow_readback.py`。

> **本文初稿把这条写成「异常被吞掉」的卫生问题，实际影响严重得多。** 完整链路是：
> 泵回读损坏 → 解析抛异常被吞 → 返回 `None` → 调用方 `or float(q)` **把回读替换成指令值**
> → `_flow_matches("Q1", q, q_actual)` 变成拿指令值和它自己比 → 恒真 → 报告「更新成功」。
> 即 `_flow_matches` 这个校验**在它本该生效的那个场景里失效了**。
> `backend/pump_hardware/service.py:24-27` 的注释写明「只有在 RSP 回读换算回请求值后才算写入成功」——
> 恒真式恰好破坏了这个成文契约。

`AGENTS.md:62` 明确禁止「为了通过测试……吞掉异常」。

最严重的一处：

- `backend/pump_hardware/service.py:94` — `flow_from_channel_params` 内 `except Exception: return None`。
  参数损坏会被下游解释为「无流量」，把协议/硬件故障伪装成合法读数。

**本次明确未处理的两项**：

- 控制循环内的同类兜底（`orchestrator/service.py` 约 `:3949-3953`、`:3974-3975` 及对应位置）**保持原样**。
  那几处不承担回读校验职责，回退到指令值属于既有容错语义（「拿不到实测值就假定达到指令值」），
  且 `:3949-3951` 是显式写出的 `if verified_q1 is None` 分支。收紧它们会改变控制循环的失败语义，超出本次范围。
- `_volume_unit_to_ul` / `_time_unit_to_min` 用 `dict.get(code, 1.0)`：**未知 unit code 静默当 1.0**，
  可能让流量差几个数量级。同类「静默默认值」问题，本次未覆盖。

其余：

| 位置 | 处数 |
| --- | --- |
| `backend/pump_hardware/client.py:117,130`（close/reset 静默） | 2 |
| `backend/orchestrator/vision_adapter.py:1180,1195,1217,1231`（相机关闭、编码失败） | 4 |
| `backend/vision/cameras/manager.py:176-181,402,411`（**含重连失败**） | 3+ |
| `backend/vision/cameras/adapters/hikrobot_camera.py` | 7 |
| `backend/vision/cameras/adapters/flir_camera.py` | 6 |
| `backend/vision/cameras/adapters/gentl_camera.py` | 4 |

**反面的好消息**：`backend/vision/detector.py` 与 `backend/vision/flow_locking.py` 的核心算法**没有**静默吞噬。

**待定**：每处的正确行为不同（抛错 / 记录日志 / 返回显式错误态），会影响调用方与真实硬件行为，需逐处判断。

### P0-3 前端 `PumpPage` 破坏分层

`frontend/qt_app.py` 中 `PumpPage`（类定义在 `:1292`）绕过 orchestrator 直连底层：

- `:1327-1328` — `from serial.tools import list_ports`，前端自己枚举串口
- `:1354` — 取 `self.app.orchestrator.pump_service` 并直接改写其 `serial_config`（port/address/baudrate/parity）
- `:1355-1357` — 直接调 `connect_and_probe()`、`read_rsp(ch)`、`read_rss()`、`read_rse()`、`client.connected_parity`

违反 `AGENTS.md:50`「不要从页面直接调用底层泵、相机或控制器」。

**范围有限，不必惊慌**：写路径 `:1366` 已经走 orchestrator 的 `run_pump_interaction_test`；
相机路径（`VideoPage` 的 `:842/870/872`）完全合规。**只需改这一个页面**。

---

## P1 · 测试与文档的可信度

### P1-1 TODO 基线过期

`TODO.md` 的「当前基线」写 `117 passed, 5 subtests passed`，实测是 `525 passed, 6 subtests passed in 69.34s`。
基线失真会让「以上为重构修复后的剩余工作」这个判断失去依据。（「当前修改尚未提交」这条是对的。）

### P1-2 两个真实的测试缺口

- `initialize_system` 和 `prepare_video` 在 `tests/` 中出现 **0 次**。
- 全部 8 处 `get_snapshot()` 调用都是**读快照做断言**，没有一处在生命周期序列里。

即「configure → prepare_video → initialize_system → start → pause → resume → stop」这条链路没有回归保护。
`TODO.md` 的相应条目属实，但范围比它写的小：`start/pause/resume/stop` 在别处被碰到过，真正空白的是那两个方法。

### P1-3 参考视频没有归宿

`tests/test_flow_locking.py:28-39`：

```python
REFERENCE_CANDIDATES = (
    REPO_ROOT / "output" / "crops" / "ref_video.npy",   # 实际命中的是这个
    Path(r"E:\MCS_QT\output\crops\ref_video.npy"),      # 与上一行同文件，纯冗余
    Path(r"C:\missing-test-data\ref_video.npy"),     # 该文件不存在，是死候选
)
```

- 第 31 行是**失效的死候选**，该文件在本机不存在，应删除。
- 第 30 行与第 29 行指向同一文件，冗余。
- 真正命中的 `output/crops/ref_video.npy`（23.4 MB）位于 **已被 `.gitignore` 忽略、完全未跟踪**的 `output/` 下。
  因此任何干净克隆上，依赖它的测试都会**静默 skip**（`skipif(_reference_path() is None)`）——不是红，是绿。
  该文件 21 个 test 中只有 2 个受此影响，其余 19 个用 `train_stack()` 生成合成数据，符合 `AGENTS.md:54` 的仿真数据要求。

**待定**：这 2 个测试要么配一个小体积 fixture 提交入库，要么在文档里写清重新生成的方法。

### P1-4 `docs/` 无索引

35 个文件，命名全靠 `*_202609xx.md` 日期后缀，没有 README 或索引。
其中 `plant_model_*` 是 6 篇同主题的演化序列（`_audit` / `_fit` / `_literature_notes` / `_measured` / `_revision` / `_wall_geometry`），
`deepseek_bench_*`、`data_reuse_run_*` 属于一次性运行记录。缺索引时，无法判断哪篇是现行结论。

---

## P2 · 可维护性

### P2-1 巨型类与巨型方法

| 文件 | 行数 |
| --- | --- |
| `backend/orchestrator/service.py` | 4166 |
| `frontend/qt_app.py` | 2403 |
| `backend/orchestrator/vision_adapter.py` | 2045 |
| `backend/pump_hardware/service.py` | 1160 |
| `frontend/vision_tuning.py` | 1061 |

`OrchestratorService`（`service.py:64`）单类 **100+ 方法**，同时承担：泵安全/急停（`:194-244`）、
约 700 行标定实验（`:912-1630`）、控制循环（`:3240-3340`）、贝叶斯优化（`:3985-4340`）、
快照构造（`:2988-3128`），甚至 mojibake 清理（`:4346-4358`）。

`backend/vision/detector.py:99` 的 `_detect_generation_plugs` 单方法 264 行，把预处理、峰值检测、几何换算、校验揉在一起。

### P2-2 重复实现

- `pitch_from_autocorrelation`（`backend/vision/flow_locking.py:513`）与
  `spatial_pitch`（`backend/vision/offline_analysis.py:21`）是同一个空间自相关峰算法的两份实现。
- 抛物线亚像素公式在 `backend/vision/offline_analysis.py:39` 和 `:106` 写了两遍。
- `backend/orchestrator/vision_adapter.py` 的 `_encode_png_base64:1183`、`_encode_jpeg:1198`、`_encode_pgm:1220`
  三个函数结构相同（resize + dtype 归一 + encode），各自带一份独立的 `except Exception: return None`。
- JSON 落盘一致性不统一：只有 `calibration_experiment.py:993` 的 `_write_json_atomic` 是原子写，
  `backend/disturbance_model/` 仍在用非原子 `json.dumps`。

### P2-3 lint gate 实际跑不起来

`pyproject.toml` 的 `dev` extras 声明了 `ruff` 和 `mypy`，但 `.venv` 里**两个都没装**（实测 `No module named ruff`、`No module named mypy`）。
另有 22 个文件缺 `from __future__ import annotations`，其中真实代码为
`backend/vision/config.py`、`frontend/app.py`、`frontend/paths.py`。

---

## P3 · 死代码与仓库卫生

### P3-1 可直接删除（已验证零引用）

- `backend/orchestrator/flow.py` — 102 行、17 处 TODO、1 个纯 `pass` 方法体（`:110-125`）。
  全仓 grep `SystemFlow` / `orchestrator.flow` 只命中它自己，**连测试都没有**。文件已被 git 跟踪且无改动，删除可恢复。
- `frontend/pages/`（8 文件 1546 行）+ `frontend/components/`（6 文件 437 行）— Tkinter 遗留。
  从 `run.py:58` → `frontend.app` → `frontend.qt_app` 这条链**不可达**：两个包的 `__init__.py` 只有 docstring 和空的 `__all__`，不 re-export 子模块。
  唯一引用者是 `tests/test_frontend_imports.py:23`，它 import 的是这两个*包*本身。
  `frontend/video_process.py:35` 也只在函数内延迟 import tkinter，只被死掉的 `monitor_page.py` 和 `test_video_process.py` 引用。
  `pyproject.toml:44` 已排除它们打包。

### P3-2 不能直接删

- `backend/vision/camera_adapters/`（4 文件 1022 行）**尚未死透**：live 的
  `backend/vision/cameras/adapters/hikrobot_camera.py:7-8` 反过来 import 旧模块的
  `LegacyHikrobotAdapter` 和 `HikrobotSdkLoader`，并在 `:29,159,176,207-232` 委托调用。
  **必须先把 Hikrobot 迁移到 `cameras/` 体系，再删 `camera_adapters/`**，顺序反了会打断 Hikrobot 相机。
- `backend/vision/legacy/` — 风险其实已消除：无任何非 legacy 模块 import 它，
  `pyproject.toml:44` 已排除打包，`pyproject.toml:29` 的 `testpaths = ["tests"]` 加上文件名不符合 `test_*.py`，pytest 不会收集。
  只剩物理归档和清理目录内的历史 `__pycache__/*.pyc`。

### P3-3 `.gitignore` 缺口

以下**未跟踪且未被 ignore**，常驻 `git status`：

| 路径 | 体量 |
| --- | --- |
| `pytest-cache-files-*` | 29 个空目录（`_trash_20260919/` 里另存 29 个） |
| `.test-batch-control-*` 等 7 个 | 1.37 MB |
| `test-artifacts/` | 2.54 MB |
| `_trash_20260919/` | — |
| `.test_tmp/`、`.tmp_pytest_vision_apply_*` | ~0 |
| `.plant-calibration-test-*.mpc-training.json` ×2 | — |

合计约 3.9 MB，**磁盘上不值得管**。问题是它们让 `git status` 长期脏着，
而 `AGENTS.md` 提交前检查第 1 条正是「检查 `git diff`，确认没有……生成文件」——噪声一多这条检查就形同虚设。

**附带观察**：`.test-*` 目录内的 `test_algorithm_parameter_edito0/parameters.json` 这种
「测试名截断到 30 字符」的目录名，是 **pytest `tmp_path` 的标准命名**。
说明测试的临时目录落在了仓库根而非系统 temp。若是手写 `--basetemp` 则无碍；
若不是，则说明 pytest 无法使用默认 temp 路径——那 29 个空的 `pytest-cache-files-*`
正是 pytest 在默认 basetemp 不可用时才会生成的形态，值得查一下根因。

### P3-4 `TODO.md:20` 的方向反了

该条要求「统一 README、开发指南和脚本中的 Python 命令，明确使用 `.venv/bin/python`」。
但这是 Windows 机器：实测 `.venv\Scripts\python.exe` 存在，`.venv\bin` **不存在**。
照这条做会把 `README.md:64-70` 那段唯一正确的 Windows 命令改错。
正确方向是反过来——统一到 `Scripts\python.exe`，或明确标注两条路径各自的适用平台。

> 注：`AGENTS.md` 全文使用 `.venv/bin/python`。若该项目同时有 Linux 开发环境，这份指南本身对 Windows 是不准确的；
> 若只有 Windows，则应整体改写。**修订前需确认，不宜单方面改动项目开发指南。**

---

## 附录：`TODO.md` 逐条核实结果

| TODO 条目 | 核实结论 |
| --- | --- |
| P1 补 Qt 前端完整 GUI 生命周期测试 | **属实**，缺口确认 |
| P1 修复 `qt_app.py` 泵页面直连硬件 | **属实**，见 P0-3 |
| P1 为真实相机 SDK 和串口泵加模拟器/集成测试 | 未验证（超出本轮范围） |
| P2 清理 `frontend/pages/` 和 `components/` 的 Tkinter | **属实且已可执行**，见 P3-1 |
| P2 统一 `camera_adapters/` 与 `cameras/` | **部分属实**，见 P3-2（Hikrobot 仍依赖旧实现） |
| P2 隔离 `backend/vision/legacy/` | **风险已不存在**，只剩物理归档 |
| P2 重构 `orchestrator/flow.py` | **比描述更彻底**——整文件零引用，建议直接删而非重构 |
| P2 补 orchestrator 生命周期 API 测试 | **属实**，见 P1-2 |
| P2 补相机适配器契约测试 | 未验证 |
| 文档·统一 Python 命令为 `.venv/bin/python` | **方向反了**，见 P3-4 |
| 文档·明确 `harvesters` 等可选依赖 | 未验证 |
| 文档·记录跨平台支持范围与测试矩阵 | 属实，目前无测试矩阵文档 |
| 硬件验证 3 条 | 未验证（需真实设备） |
| 当前基线 `117 passed` | **过期**，实测 525，见 P1-1 |

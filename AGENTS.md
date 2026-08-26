# MCS_QT 开发指南

## 项目概览

这是一个基于 Python、PySide6/Qt 6 的微流控液滴控制系统。顶层 `run.py` 是统一启动入口；代码按职责分为前端、视觉、控制、硬件和流程编排模块。

## 目录职责

- `frontend/`：Qt 页面、组件、用户交互和状态展示；不直接执行识别、PID 或硬件通信。
- `backend/vision/`：相机适配、图像预处理、液滴检测、跟踪与统计。
- `backend/pid_control/`：PID、前馈、自适应和安全控制逻辑。
- `backend/pump_hardware/`：串口泵通信、协议、参数读写和停机控制。
- `backend/orchestrator/`：系统流程、状态管理以及前后端协调。
- `backend/disturbance_model/`：扰动数据采集、训练、预测和存储。
- `tests/`：自动化测试。
- `tools/`：模拟、诊断和开发辅助脚本。
- `docs/`：系统设计和流程文档。
- `drops_videos/`：本地视频样本，不应在提交中新增大型媒体文件，除非需求明确要求。

## 环境与运行

项目要求 Python `>=3.10`。推荐使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

常用命令：

```bash
python run.py                    # 启动前端（默认）
python run.py frontend           # 显式启动前端
python run.py vision --video input.mp4
python run.py vision --camera 0
pytest                           # 运行全部测试
python -m pytest tests/test_pid_controller.py
python -m pip install -e .       # 安装为可编辑命令行工具
microfluidic-control
```

也可以使用 `uv sync` 和 `uv run ...` 管理环境。不要把 `.venv`、缓存、运行日志或本地硬件配置提交到 Git。

## 开发约定

- 保持 Python 类型标注和现有 `from __future__ import annotations` 风格；优先编写清晰、短小、可测试的函数。
- 修改前先阅读相关模块的 README、数据模型和现有测试；尽量沿用已有接口、命名和线程模型。
- 前端通过 orchestrator 的公开接口交互：`configure()`、`prepare_video()`、`initialize_system()`、`start()`、`pause()`、`resume()`、`stop()`、`get_snapshot()`。不要从页面直接调用底层泵、相机或控制器。
- 后端线程、队列、定时器和 Qt 信号的修改必须考虑生命周期、停止、异常传播和线程安全；避免在 GUI 线程执行阻塞 I/O 或计算。
- 控制周期使用已有的时钟/事件机制，避免用无界队列或不受控的 `sleep` 改变实时行为。
- 新增硬件适配器时复用抽象基类、注册表和现有协议解析；真实设备操作必须有超时、错误处理和安全停机路径。
- 不要在测试或开发脚本中默认连接真实相机、串口泵或修改设备参数；优先使用 mock、仿真数据和样本视频。
- 配置、日志和用户设置应使用项目已有的配置/存储/日志设施，不要写入源码目录中的临时文件。
- 注释解释原因和约束，不重复代码本身；公共接口变更时同步更新 README、相关子目录文档和测试。

## 测试要求

- 每次代码修改至少运行受影响的测试；涉及公共流程、控制或线程时运行完整 `pytest`。
- 新增行为应补充回归测试，测试应可在无相机、无串口和无显示器的环境中运行。
- 不要为了通过测试弱化安全校验、吞掉异常或改变真实硬件行为。
- 若测试依赖可选 SDK、设备驱动或 GUI 环境，应明确隔离并在变更说明中记录未执行原因。

## 提交前检查

1. 检查 `git diff`，确认没有调试输出、密钥、个人路径、设备信息或生成文件。
2. 运行相关测试，并记录实际执行的命令和结果。
3. 检查导入、启动路径和文档示例是否仍然有效。
4. 涉及硬件控制的变更，确认异常、暂停、停止和退出时均能进入安全状态。

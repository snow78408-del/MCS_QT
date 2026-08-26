# microfluidic_control_system

微流控液滴控制系统。桌面界面使用 PySide6（Qt 6），项目按职责拆成 `frontend/` 与 `backend/`，顶层 `run.py` 是统一启动入口。

## 目录结构

```text
microfluidic_control_system/
├── run.py
├── pyproject.toml
├── requirements.txt
├── frontend/
└── backend/
    ├── vision/
    ├── pid_control/
    ├── pump_hardware/
    └── orchestrator/
```

## 启动方式

默认启动前端界面：

```bash
python run.py
```

界面采用 Qt 原生组件、信号槽、线程池和定时器；基础参数、视频源、初始化、运行监控与系统状态均通过左侧导航访问。

“相机识别与读写”页面会扫描海康、GenTL、Basler、大恒、FLIR、Allied Vision 和 OpenCV 相机后端，并对曝光、增益、帧率与分辨率执行写入回读。“泵机识别与读写”页面会枚举串口，通过 RSS/RSE/RSP 协议确认泵机身份、读取各通道状态，并安全写入和校验 Q1/Q2。

相机参数测试会同步采集并显示测试帧，可直接在画面上拖拽框选 ROI。相机发现、打开、参数下发、回读和取帧过程写入用户数据目录的 `logs/runtime_*.log`，也可从相机页面导出日志。可用 `MCS_DATA_DIR` 显式指定数据目录。

实时运行采用解耦流水线：相机采集、预览、采样和分析使用独立的有界队列；过期预览帧允许丢弃。系统分别报告硬件帧号缺口、采集 FPS、处理 FPS、有效控制周期和处理替换数，不预设或宣称未经台架验证的帧率。PID 的周期、超时、新鲜度与跟踪预测使用单调时钟。

也可以显式启动：

```bash
python run.py frontend
```

独立运行视觉流水线：

```bash
python run.py vision --video input.mp4
python run.py vision --camera 0
```

`vision` 后面的参数会继续传给 `backend/vision/run_vision.py`。

## 环境配置

任选一种 Python 环境管理方式即可，项目不依赖固定目录中的虚拟环境。

普通 Python / venv：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python run.py
```

Windows 中不激活环境也可以直接运行虚拟环境里的 Python：

```bat
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe run.py
```

conda：

```bash
conda create -n microfluidic-control python=3.11
conda activate microfluidic-control
conda install -c conda-forge numpy opencv pyserial
python run.py
```

uv：

```bash
uv sync
uv run python run.py
```

`requirements.lock` 用于可复现安装；`requirements.txt` 保留为宽松的直接依赖说明。工业相机厂商 SDK、GenTL Producer 和驱动属于可选的设备依赖，必须按对应厂商版本安装，纯文件视觉模式不要求安装它们。

## 安全和测量前提

闭环运行必须在“基础参数”页选择带版本、日期、倍率、视野、标定图像哈希和不确定度的像素标定 JSON；未标定比例只允许预览，实验前检查和 PID 启动会拒绝闭环。泵回读被明确标记为“设备参数换算值”，不是物理流量。称重/流量传感器校准、泵物理响应延迟和目标帧率必须在具体硬件上测量后才能形成实验结论。

完整状态矩阵、标定格式、延迟采集点和故障注入验收方法见 [`docs/safety_and_measurement.md`](docs/safety_and_measurement.md)。

安装为命令行工具后也可运行：

```bash
python -m pip install -e .
microfluidic-control
```

只需核心功能时可安装 `python -m pip install -e .`；使用 GenTL/Harvester 时安装 `python -m pip install -e ".[industrial-camera]"`。版本标签会通过 `package` 工作流生成 wheel 和源码包；厂商驱动及 SDK 不会打包进项目制品。

## 子目录职责概览

- `frontend/`：前端用户交互页面与状态展示层。
- `backend/vision/`：图像识别与统计输出。
- `backend/pid_control/`：PID 反馈控制逻辑（当前仅保留平均直径反馈链路）。
- `backend/pump_hardware/`：泵硬件连接、下发与停机控制。
- `backend/orchestrator/`：后端主流程耦合与状态调度入口。

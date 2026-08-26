# vision 模块

## 1. 模块职责
`vision` 负责图像识别链路：液滴检测、轨迹跟踪、液滴内磁珠统计、统计指标计算和标准化输出。
本模块可独立运行测试，不耦合前端界面、不包含 PID 控制和泵通信。

## 2. 当前目录结构
```text
vision/
├── camera_profiles.py
├── cameras/
│   ├── base.py
│   ├── registry.py
│   ├── manager.py
│   └── adapters/
├── detector.py
├── tracker.py
├── nearest_tracker.py
├── kalman_tracker.py
├── bead_counter.py
├── metrics.py
├── pipeline.py
├── config.py
├── run_vision.py
├── preprocess/
│   ├── rotate_video.py
│   └── README.md
├── README.md
├── requirements.txt
├── legacy/
│   ├── README.md
│   ├── droplet_tracking_and_counting(2)(1).py
│   ├── droplet_tracking_and_counting(2).py
│   ├── droplet_tracking_connected_40mum(1).py
│   └── rotate_video_90ccw(1).py
```

## 3. 每个文件功能
- `config.py`：统一管理参数（半径范围、圆度阈值、ROI、匹配距离、最大未匹配帧数、磁珠面积、debug 开关、tracker 类型、Kalman 参数等）。
- `camera_profiles.py`：按相机厂商/型号提供可编辑的推荐起始参数，并负责下发前的类型和范围预校验。
- `cameras/`：工业相机统一接口、设备注册表和生命周期管理；当前优先使用海康机器人适配器，其他厂商通过同一适配器接口扩展。
- `detector.py`：Hough-only 液滴检测模块，经过几何、目标尺寸、边缘归属和最终评分过滤后输出圆心、半径及磁珠模块所需的辅助掩膜。
- `tracker.py`：统一跟踪接口和数据结构（`DropletTrack`、`TrackingResult`、`BaseTracker`）。
- `nearest_tracker.py`：最近邻跟踪实现，保留基础行为用于快速基线验证。
- `kalman_tracker.py`：Kalman 跟踪实现（`x,y,vx,vy`），支持预测-匹配-更新和短时丢检维持。
- `bead_counter.py`：液滴内磁珠/小黑点识别与统计，统一 intensity 与 connected 两种模式。
- `metrics.py`：统计指标计算，区分控制输出与分析输出。
- `pipeline.py`：主流程编排，串联 detector/tracker/bead_counter/metrics，输出统一 `VisionResult`。
- `run_vision.py`：vision 独立运行入口（本地视频、摄像头、可视化、统计输出）。
- `preprocess/rotate_video.py`：视频预处理工具，提供旋转能力并可命令行使用。
- `legacy/`：旧脚本归档目录，仅作历史参考。

## 4. detector / tracker / bead_counter / metrics / pipeline 关系
- `detector`：从单帧中得到候选液滴（圆心/半径）。
- `tracker`：将跨帧检测结果关联成稳定轨迹 ID。
- `bead_counter`：基于活动液滴轨迹统计每个液滴的磁珠数量。
- `metrics`：基于轨迹与磁珠统计生成控制和分析指标。
- `pipeline`：统一调度上述模块并产出标准化 `VisionResult`。

## 5. nearest 与 kalman 跟踪器区别
- `nearest`：实现简单、调参直观；在短时漏检/运动扰动下更容易出现 ID 跳变。
- `kalman`：引入运动状态预测（`x,y,vx,vy`），短时丢检时可依赖预测维持轨迹，稳定性更高。

## 6. 默认版本与切换方式
- 默认：`kalman`。
- 切换方式：运行时通过 `--tracker nearest|kalman` 切换。

## 7. preprocess 子模块职责
`preprocess/` 专门放置识别前预处理工具（旋转、裁剪、分辨率、方向标准化等），与检测/跟踪/统计主链路解耦。

## 8. rotate_video.py 作用
- 将输入视频按指定模式旋转后输出（`ccw90` / `cw90` / `180` / `auto` 占位）。
- 可作为可复用函数被 `run_vision.py` 调用，也可独立命令行运行。

## 9. 独立运行 vision 模块
示例：
- 本地视频：`python run.py vision --video input.mp4`
- 摄像头：`python run.py vision --camera 0`
- 启用 Kalman：`python run.py vision --video input.mp4 --tracker kalman`
- 运行前预处理旋转：`python run.py vision --video input.mp4 --preprocess-rotate ccw90`

## 10. 独立液滴识别调参工作台

现在可以从主软件左侧导航进入“7 液滴算法调参”，完整调参工作台会直接嵌入主页面，不进入控制流程。也可以在不启动主软件时直接运行：

```bash
python run.py tune --video path/to/sample.mp4
```

工作台只读取本地视频并调用 `DropletDetector`，支持逐帧预览和 JSON 参数保存。视频路径、加载按钮和帧滑块集中在顶部紧凑工具栏中。液滴候选现在只来源于灰度图上的 Hough 梯度圆变换，不再融合或回退到连通域、二值化轮廓及亮度峰值。

当前帧用多宫格依次展示原图、灰度转换、对比度归一化、输入高斯平滑、CLAHE 局部增强、Hough 中值滤波、Canny 边缘、Hough 原始圆、Hough 几何过滤、目标尺寸过滤、圆周边缘归属和最终评分抑制。Hough 的 `dp`、圆心距离、`param1`、`param2`、半径范围和边缘支撑阈值均可在所属卡片直接编辑。所有常用数值参数同时提供算法安全范围内的滑条和精确输入框；拖动滑条时不重复运行 Hough，松开后自动刷新，文本输入仍采用 500ms 防抖刷新。

密集液滴场景会先根据可靠的期望半径和容差拒绝尺寸异常圆，再把圆周附近的真实边缘像素归属给径向残差最小的候选圆。主要借用相邻液滴边缘拼成的候选因独占边缘比例不足而被拒绝；诊断卡片以红色显示进入过滤的圆、绿色显示保留圆。归一化、输入平滑、CLAHE、中值滤波、目标尺寸过滤和边缘归属过滤可通过卡片内复选框跳过；Hough 圆变换本身不能关闭。每一步都显示用途、实际参数和统计结果，并可放大查看。工作台不会连接相机、泵机，不运行跟踪、磁珠识别或 PID。

## 11. 环境准备
- 推荐使用项目顶层 `requirements.txt` 或 `pyproject.toml` 安装依赖。
- 从项目根目录运行：`python run.py vision --video input.mp4`。

## 12. 当前版本优点
- 从大脚本拆分为模块化架构，职责边界清晰。
- 提供最近邻与 Kalman 双跟踪器，可按配置切换。
- 输出数据结构标准化（`VisionResult`、`DropletTrack`、`TrackingResult`、`BeadResult`）。
- 预处理与识别主流程解耦，便于后续扩展。
- 预览帧与识别结果分别维护帧号和时间戳，PID 只使用同一次识别产生的数据。
- 实时统计按轨迹 ID 去重，避免同一液滴停留多帧时被重复计入控制样本。
- 检测尺寸范围随用户目标直径与标定比例缩放，不绑定固定的 50 μm。
- 相机测试和正式运行使用同一套参数下发路径；支持的参数写入失败时会终止测试，不静默继续。

## 13. 相机适配边界
- `cameras/base.py` 定义厂商无关接口，`registry.py` 负责后端注册和优先级。
- 海康机器人是当前启用和优先后端，保留现有 MVS/直接 SDK 兼容路径。
- 新相机只需新增适配器并注册，不应把厂商 SDK 调用写入 detector、pipeline、orchestrator 或前端。
- 海康直接 DLL 路径可确认写入调用是否成功；曝光、增益和帧率的精确回读能力取决于当前 SDK 路径，分辨率会通过实际测试帧再次校验。

## 14. 待改进方向
- 引入 Hungarian/IoU 等更稳健匹配策略。
- ROI 自动估计与自适应阈值策略。
- 多通道融合和光照鲁棒性增强。
- 引入更细粒度的质量评估与异常帧剔除机制。

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
├── channel_region.py
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
- `channel_region.py`：液滴识别前的启动检定；先将每帧空间梯度转换为局部高频能量，以统一量纲融合 12 帧并过滤未达到最低持续帧比例的瞬时信号，由“管内高频、管外低频”的明显区域界线产生边界候选，再把两侧界线拟合、验证为近似平行的直线管壁，最后透视摆正有效区域。原始图像中的强直线不再直接参与候选生成。采样期间不产生液滴/PID 有效结果；该步骤可关闭，可信度不足时自动回退整帧识别。
- `detector.py`：液滴检测模块；对有效管道区域执行灰度转换、大尺度高斯背景校正、CLAHE 和高斯平滑，再以 `cv2.HoughCircles` 输出液滴圆，按从上到下、从左到右稳定排序，并对最终半径应用统一的百分比尺寸调节。
- `tracker.py`：统一跟踪接口和数据结构（`DropletTrack`、`TrackingResult`、`BaseTracker`）。
- `nearest_tracker.py`：最近邻跟踪实现，保留基础行为用于快速基线验证。
- `kalman_tracker.py`：Kalman 跟踪实现（`x,y,vx,vy`），支持预测-匹配-更新和短时丢检维持。
- `bead_counter.py`：液滴内磁珠/小黑点识别与统计，统一 intensity 与 connected 两种模式。
- `metrics.py`：统计指标计算，区分控制输出与分析输出。
- `pipeline.py`：主流程编排，串联 detector/tracker/bead_counter/metrics，输出统一 `VisionResult`。
- `run_vision.py`：vision 独立运行入口（本地视频、摄像头、可视化、统计输出）。
- `preprocess/rotate_video.py`：视频预处理工具，提供旋转能力并可命令行使用。
- `legacy/`：旧脚本归档目录，仅作历史参考。

## 4. channel_region / detector / tracker / bead_counter / metrics / pipeline 关系
- `channel_region`：启动时从原始大图确定两条直线管壁围成的有效区域；手动 ROI/管壁优先，自动失败则回退整帧。
- `detector`：从有效管道区域的单帧中得到候选液滴（圆心/半径）。
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
- 跳过自动管道检定：`python run.py vision --video input.mp4 --skip-channel-region`
- 启用 Kalman：`python run.py vision --video input.mp4 --tracker kalman`
- 运行前预处理旋转：`python run.py vision --video input.mp4 --preprocess-rotate ccw90`

## 10. 独立液滴识别调参工作台

现在可以从主软件左侧导航进入“7 液滴算法调参”，完整调参工作台会直接嵌入主页面，不进入控制流程。也可以在不启动主软件时直接运行：

```bash
python run.py tune --video path/to/sample.mp4
```

工作台只读取本地视频并调用 `DropletDetector`，支持逐帧预览和 JSON 参数保存。视频路径、加载按钮和帧滑块集中在顶部紧凑工具栏中。当前算法只保留整帧 Hough 圆检测，不再执行轮廓、背景差分、连通域、Watershed、候选融合、边缘支撑过滤、尺寸门控或亮度评分。默认半径范围为 18–32 px、最小圆心距离为 32 px、敏感度为 0.96；敏感度按 `param2 = 45 - 25 × sensitivity` 映射到 Hough 累加阈值。最终结果提供 -20%～+20% 的“液滴整体尺寸调节”，默认 0%，调节后的半径会进入正式运行的跟踪、直径统计和 PID 数据链路。

当前帧先用四个大卡片展示“管道区域检定”：原始大图、局部高频区域、高低频界线拟合和最终有效区域/可信度；之后再依次展示灰度转换、光照背景估计、光照校正、CLAHE、Hough 前高斯平滑、Hough 原始圆和最终结果。管道检定和液滴检测参数均可直接调整，检定开关关闭时会明确显示“已跳过”。保存文件采用兼容分组 JSON，包含 `channel_region` 与 `detector` 两组参数。

跟踪器仍默认采用 3 帧窗口至少 2 次命中才确认轨迹，确认后短时漏检由 Kalman 预测维持。工作台不会连接相机、泵机，不运行跟踪、磁珠识别或 PID。

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
- 检测尺寸由图像域半径范围或显式图像标定约束，不受 PID 目标直径影响。
- 相机测试和正式运行使用同一套参数下发路径；支持的参数写入失败时会终止测试，不静默继续。

## 13. 相机适配边界
- `cameras/base.py` 定义厂商无关接口，`registry.py` 负责后端注册和优先级。
- 海康机器人是当前启用和优先后端，保留现有 MVS/直接 SDK 兼容路径。
- 新相机只需新增适配器并注册，不应把厂商 SDK 调用写入 detector、pipeline、orchestrator 或前端。
- 海康直接 DLL 路径可确认写入调用是否成功；曝光、增益和帧率的精确回读能力取决于当前 SDK 路径，分辨率会通过实际测试帧再次校验。

## 14. 待改进方向
- 引入 Hungarian/IoU 等更稳健匹配策略。
- 管道区域检定阈值的更多真实相机样本自适应。
- 多通道融合和光照鲁棒性增强。
- 引入更细粒度的质量评估与异常帧剔除机制。

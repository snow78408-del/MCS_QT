# legacy

本目录用于归档旧版实验脚本，避免继续作为主实现入口。该目录已从安装包中排除；历史脚本不得用于论文数据生成或实时泵控制。

## 归档文件

- `droplet_tracking_connected_40mum(1).py`
- `rotate_video_90ccw(1).py`

## 归档原因

- 旧脚本存在多职责耦合（检测、跟踪、磁珠统计、视频循环、输出打印耦合在单文件）。
- 不利于后续扩展 Kalman 跟踪与统一输出接口。

## 新结构替代关系

- 液滴检测统一由 `detector.py` 中的 EdgeDrawing 实现提供
- 旧版最近邻跟踪逻辑 -> `nearest_tracker.py`
- 新增 Kalman 跟踪逻辑 -> `kalman_tracker.py`
- 旧版磁珠/小黑点统计逻辑 -> `bead_counter.py`
- 旧版统计汇总逻辑 -> `metrics.py`
- 旧版流程主循环 -> `pipeline.py` + `run_vision.py`
- 旧版视频旋转脚本 -> `preprocess/rotate_video.py`

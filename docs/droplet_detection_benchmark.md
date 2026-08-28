# 液滴检测人工标注与评测

评测文件只记录帧号、圆心和半径，不提交原始大型视频。JSON 格式如下：

```json
{
  "frames": [
    {
      "index": 0,
      "droplets": [
        {"x": 125.0, "y": 82.0, "radius": 24.5}
      ]
    }
  ]
}
```

建议从不同曝光、流速、密度和粘连程度的视频中均匀抽取至少 200 帧；空帧也必须保留，并将 `droplets` 写为空列表。训练/调参视频和最终验收视频应分开，避免用同一批帧反复调参后再报告准确率。

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe tools\benchmark_droplet_detector.py `
  --video drops_videos\sample.mp4 `
  --annotations drops_videos\sample.labels.json
```

可通过 `--config` 传入调参工作台导出的检测器 JSON。输出包括 Precision、Recall、F1、平均半径绝对误差、平均帧耗时、P95 帧耗时，以及是否满足默认验收目标：Precision ≥ 98%、Recall ≥ 95%、平均半径误差 ≤ 3 px。

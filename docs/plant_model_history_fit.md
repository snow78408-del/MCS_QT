# 从历史标定记录拟合模型

`tools/fit_plant_history_model.py` 汇总设备保存的历史标定记录，拟合响应模型，
用于事后复核、模型形式对比和下一轮标定设计。工具只读历史目录，输出写到单独目录，
不会生成可被控制器加载的标定文件。

## 输入数据

- 优先读取 `*.measurements.json`；没有主文件时回退到同名 `*.measurements.mpc-training.json`。
- 使用字段：基线与实际回读流量、基线与稳定尺寸、`diameter_change_um`、`response_observations`、
  `response_detected`、会话 id 与实验配置。validation 通道试验不参与拟合，只用于门槛复核。
- 建模口径以回读流量变化为准，不信任 `channel` 标签：个别会话的 combined 试验实际只改变了一路流量，
  工具按真实流量差构建设计矩阵，并在诊断里标出这些会话。

## 用法

```bash
.venv/bin/python tools/fit_plant_history_model.py \
    --history D:/MCS_QT_Data/calibrations \
    --output output/history-model
```

Windows 下解释器为 `.venv\Scripts\python.exe`。省略 `--history` 时使用 `backend.runtime_paths.user_data_dir()/calibrations`。

| 参数 | 作用 |
| --- | --- |
| `--only-detected` | 只用分类为 `detected_stable` 的试验 |
| `--min-response-observations N` | 丢弃响应曲线点数不足的试验（默认 3） |
| `--include-session` / `--exclude-session` | 按会话 id 白名单/黑名单（可重复） |
| `--allow-zero-flow-trials` | 保留回读流量没有变化的试验 |
| `--no-session-offsets` | 关闭含会话截距的候选模型 |
| `--no-figure` | 不生成 PNG 诊断图 |
| `--quiet` | 只输出结果目录 |

## 候选模型与选择

| 家族 | 形式 | 参数 |
| --- | --- | --- |
| `constant` | 常数（噪声下限参照） | 1 |
| `linear_delta` | a1·ΔQ1 + a2·ΔQ2 | 2 |
| `quadratic_delta` | 应用同形式：中心化缩放后的二次响应面 | 5 |
| `log_linear` | a1·ln(Q1/Q1b) + a2·ln(Q2/Q2b) | 2 |
| `log_quadratic` | 对数比值的二次式 | 5 |
| `power_law` | a1·((Q1/Q1b)^p1−1) + a2·((Q2/Q2b)^p2−1) | 2 + 网格搜索 p1、p2 |

每个家族都可用最小二乘或 Huber 稳健回归拟合，并可加入每会话截距以吸收会话间慢漂移。
选择依据是留一会话交叉验证（LOSO）RMSE 最小，同时报告去掉会话偏置后的 RMSE 与 AICc；
幂指数在每一折内部网格搜索，避免用留出会话调参。

## 统计口径

- NRMSE 与应用一致：以 `max(|预测变化|, 0.25)` 归一化。模型增益趋近 0 时该值会被急剧放大，
  因此只用于门槛复核，不用于比较模型优劣。
- 系数同时给出经典标准误与会话聚类标准误；同一会话内的重复不独立，置信区间按会话数取 t 分位。
- `flow_response_identifiable` 仅在两路增益的 95% 置信区间都排除 0 时为真，避免把噪声当作增益。
- 应用门槛沿用 `calibration_experiment` 默认：MAE ≤ 2 µm、NRMSE ≤ 0.25。历史 validation 试验
  来自同一批会话，只作复核，不构成应用要求的独立验证。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `model.json` | 会话与筛选记录、诊断、候选排序、选定模型系数与置信区间、交叉验证指标、动态拟合、门槛复核、`validity` 与建议 |
| `report.md` | 同内容的中文报告，含数据质量、可辨识性、模型比较、选定模型与结论 |
| `dataset.csv` | 汇总后的试验表（会话、通道、流量差、Δd、质量字段），便于在表格软件中复核 |
| `response_map.png` | 观测-预测对照与激励分布图（需要 OpenCV，可用 `--no-figure` 关闭） |

`model.json` 中若二次响应面是可选形式，会附带应用同结构的 `app_nonlinear_model` 或
`app_nonlinear_model_candidate`，供人工比对；这不代表可以直接加载。

## 结论边界

- 工具不会写入历史目录，也不生成主标定文件；`validity.control_authorized` 恒为 `false`。
- 历史拟合只能产生候选模型。控制器要求一次新的独立验证，见 `calibration_files.md`
  与 `nonlinear_calibration.md` 的适用边界说明。
- 2026-09-18 对 `D:\MCS_QT_Data\calibrations` 的实际运行结果：37 个建模试验、4 个会话，
  留一会话交叉验证最优为常数模型（RMSE 4.19 µm），两路流量增益的 95% 置信区间都覆盖 0，
  历史 validation 复核未达门槛。说明这批记录的响应幅度低于会话间与重复间的漂移量级。

## 测试

```bash
.venv/bin/python -m pytest tests/test_plant_history_model_fit.py
```

测试使用合成记录，不连接相机、泵或真实数据目录。

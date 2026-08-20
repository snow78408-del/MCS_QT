# frontend 模块

## 前端职责
前端是用户交互入口，负责：
- 参数输入
- 视频来源选择
- 初始化参数输入
- 运行状态展示
- 操作按钮触发（初始化、开始、暂停、继续、停止）
- 周期性读取 `orchestrator.get_snapshot()` 并刷新页面

## 前端不负责
前端不直接执行：
- 图像识别计算
- PID 反馈计算
- 泵硬件通信
- 串口命令下发与协议解析
- 控制循环调度

以上全部由后端 orchestrator 统一调度并对外提供接口。

## 目录结构
- `app.py`：前端入口，负责主窗口、页面切换、后端任务调度
- `config.py`：前端显示与刷新配置
- `pages/`
  - `parameter_page.py`：参数设定页面
  - `video_source_page.py`：视频来源选择页面
  - `init_page.py`：初始化参数页面
  - `monitor_page.py`：运行监控页面
  - `status_page.py`：状态详情页面
- `components/`
  - `status_panel.py`：系统状态面板
  - `pump_panel.py`：泵状态面板
  - `recognition_panel.py`：识别结果面板
  - `control_buttons.py`：控制按钮组件

## 页面功能
1. 参数设定页面
- 输入目标直径、像素转微米系数、控制周期
- 基础合法性校验
- 进入视频来源页面

2. 视频来源页面
- 选择实时摄像头或本地视频
- 摄像头编号/视频路径校验
- 进入初始化页面

3. 初始化页面
- 输入初始 `Q1/Q2`
- 调用 `configure() -> prepare_video() -> initialize_system()`
- 显示初始化状态

4. 监控页面
- 显示识别、泵状态、PID 控制结果、系统状态
- 提供开始/暂停/继续/停止按钮
- 周期刷新快照

5. 状态页面
- 展示系统整体状态及原始快照视图

## 组件功能
1. `StatusPanel`
- 显示 `system_state/message/error`

2. `PumpPanel`
- 显示泵连接、通信、就绪状态、Q1/Q2、运行态、最近错误

3. `RecognitionPanel`
- 显示累计液滴总数，以及用户设定控制周期内的液滴数、平均直径、单胞率和有效性

4. `ControlButtons`
- 封装初始化/开始/暂停/继续/停止按钮
- 根据系统状态自动控制可点击性

## 与 orchestrator 交互方式
前端仅调用以下接口：
- `configure(system_config)`
- `prepare_video()`
- `initialize_system()`
- `start()`
- `pause()`
- `resume()`
- `stop()`
- `get_snapshot()`

## 刷新机制
- 监控页面和状态页面通过 `after()` 定时器轮询
- 默认刷新间隔 `300ms`（可在 `config.py` 调整）
- 刷新过程只读快照，不阻塞主线程
- 后端耗时操作通过后台线程执行

## 用户操作流程
1. 参数设定
2. 视频来源选择
3. 初始化参数输入并初始化系统
4. 进入监控页面开始运行
5. 运行中可暂停/继续
6. 可随时停止并查看状态页面

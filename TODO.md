# TODO

## P1：需要优先处理

- [ ] 补充 Qt 前端完整 GUI 生命周期测试：初始化、启动、暂停、恢复、停止、关闭窗口及后台任务退出顺序。
- [ ] 评估并修复 `frontend/qt_app.py` 中泵测试页面对硬件服务的直接访问，统一通过 orchestrator 公开接口。
- [ ] 为真实相机 SDK 和串口泵增加模拟器/集成测试，覆盖启动失败、停机失败、参数回读失败和多设备并发。

## P2：架构清理

- [ ] 清理 `frontend/pages/` 和 `frontend/components/` 中遗留的 Tkinter 页面与组件；确认无业务依赖后移出或删除。
- [ ] 统一相机适配器体系，明确 `backend/vision/camera_adapters/` 与 `backend/vision/cameras/` 的唯一规范实现。
- [ ] 隔离或归档 `backend/vision/legacy/` 下的旧脚本，避免旧模块被打包、导入或参与测试。
- [ ] 修复并重构 `backend/orchestrator/flow.py` 中的旧流程骨架、TODO 和空实现，统一使用当前 `OrchestratorService`。
- [ ] 增加 orchestrator 公共生命周期 API 的完整测试：`configure()`、`prepare_video()`、`initialize_system()`、`start()`、`pause()`、`resume()`、`stop()`、`get_snapshot()`。
- [ ] 增加相机适配器契约测试，确保各厂商后端不会虚假声明可读/可写能力。

## 文档与环境

- [ ] 统一 README、开发指南和脚本中的 Python 命令，明确使用 `.venv/bin/python`。
- [ ] 明确 `harvesters` 等可选/必需依赖及其对应后端，完善启动时的分命令依赖检查。
- [ ] 记录 Linux、Windows、无显示器环境下的支持范围和测试矩阵。

## 硬件验证

- [ ] 在真实 HIKROBOT、FLIR、Allied Vision 等相机上验证参数配置时序、读写回读和采集模式。
- [ ] 在真实串口泵或硬件模拟器上验证地址校验、停止失败、部分流量更新和恢复流程。
- [ ] 验证多泵地址冲突、串口超时、设备断连和异常停机时的安全状态。

## 当前基线

- 最近一次测试：`117 passed, 5 subtests passed`
- 当前修改尚未提交 Git commit。
- 以上事项为重构修复后的剩余工作，不代表已在本轮全部实现。

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk


class PumpTestPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.port_var = tk.StringVar(value="")
        self.address_var = tk.StringVar(value="1")
        self.baud_var = tk.StringVar(value="1200")
        self.parity_var = tk.StringVar(value="N")
        self.q1_var = tk.StringVar(value="50")
        self.q2_var = tk.StringVar(value="20")
        self.status_var = tk.StringVar(value="等待测试")
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=24, pady=(20, 8))
        ttk.Button(top, text="返回参数页", command=lambda: self.app.show_page("parameter")).pack(side="left")
        ttk.Label(top, text="泵机交互测试", font=("Microsoft YaHei UI", 16, "bold")).pack(side="left", padx=18)

        card = ttk.LabelFrame(self, text="通信与灌注参数")
        card.pack(fill="x", padx=24, pady=8)
        fields = (
            ("泵串口号", self.port_var),
            ("泵地址", self.address_var),
            ("波特率", self.baud_var),
            ("Q1 (uL/min)", self.q1_var),
            ("Q2 (uL/min)", self.q2_var),
        )
        for row, (label, variable) in enumerate(fields):
            ttk.Label(card, text=label).grid(row=row, column=0, padx=8, pady=6, sticky="w")
            ttk.Entry(card, textvariable=variable, width=26).grid(row=row, column=1, padx=8, pady=6, sticky="w")
        ttk.Label(card, text="校验位").grid(row=5, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(card, textvariable=self.parity_var, values=("N", "E"), state="readonly", width=23).grid(
            row=5, column=1, padx=8, pady=6, sticky="w"
        )

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=24, pady=8)
        self.test_button = ttk.Button(actions, text="开始完整测试", command=self._run_test)
        self.test_button.pack(side="left")
        ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=14)

        result_card = ttk.LabelFrame(self, text="测试步骤与结果")
        result_card.pack(fill="both", expand=True, padx=24, pady=(8, 24))
        self.result_text = tk.Text(result_card, height=16, wrap="word", state="disabled")
        self.result_text.pack(fill="both", expand=True, padx=8, pady=8)

    def on_show(self) -> None:
        cfg = self.app.frontend_config
        configured_port = str(cfg.get("pump_port", self.port_var.get()) or self.port_var.get()).strip().upper()
        if not configured_port:
            try:
                from serial.tools import list_ports

                ports = [str(item.device).upper() for item in list_ports.comports()]
                configured_port = ports[0] if ports else ""
            except Exception:
                configured_port = ""
        self.port_var.set(configured_port)
        self.address_var.set(str(cfg.get("pump_address", self.address_var.get()) or "1"))
        self.baud_var.set(str(cfg.get("pump_baudrate", self.baud_var.get()) or "1200"))
        self.parity_var.set(str(cfg.get("pump_parity", self.parity_var.get()) or "N").upper())
        self.q1_var.set(str(cfg.get("initial_q1", self.q1_var.get()) or "50"))
        self.q2_var.set(str(cfg.get("initial_q2", self.q2_var.get()) or "20"))

    def _show_result(self, lines: list[str]) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "\n".join(lines))
        self.result_text.configure(state="disabled")

    def _run_test(self) -> None:
        try:
            values = {
                "port": self.port_var.get().strip().upper(),
                "address": int(self.address_var.get().strip()),
                "baudrate": int(self.baud_var.get().strip()),
                "parity": self.parity_var.get().strip().upper(),
                "q1": float(self.q1_var.get().strip()),
                "q2": float(self.q2_var.get().strip()),
            }
            if not values["port"]:
                raise ValueError("泵串口号不能为空，例如 COM3")
        except Exception as exc:
            messagebox.showerror("输入错误", str(exc))
            return

        self.app.update_frontend_config(
            pump_port=values["port"], pump_address=values["address"],
            pump_baudrate=values["baudrate"], pump_parity=values["parity"],
            initial_q1=values["q1"], initial_q2=values["q2"],
        )
        self.test_button.configure(state="disabled")
        self.status_var.set("测试中，请观察泵机…")
        self._show_result(["正在执行：连接 → 参数下发 → 启动灌注 → 关闭灌注"])
        result: dict[str, object] = {}

        def task() -> None:
            result.update(self.app.orchestrator.run_pump_interaction_test(**values))

        def done() -> None:
            self.test_button.configure(state="normal")
            ok = bool(result.get("ok"))
            self.status_var.set("测试通过" if ok else "测试失败")
            lines = []
            for index, step in enumerate(result.get("steps", []), 1):
                mark = "通过" if step.get("ok") else "失败"
                lines.append(f"{index}. [{mark}] {step.get('name')}: {step.get('detail')}")
            self._show_result(lines or ["没有收到测试结果"])

        def failed(exc: Exception) -> None:
            self.test_button.configure(state="normal")
            self.status_var.set("测试异常")
            self._show_result([f"测试异常：{exc}"])
            messagebox.showerror("泵机测试异常", str(exc))

        self.app.run_backend_task(task, on_success=done, on_error=failed)

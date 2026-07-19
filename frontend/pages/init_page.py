from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox, ttk


class InitPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.q1_var = tk.StringVar(value="50")
        self.q2_var = tk.StringVar(value="20")
        self.port_var = tk.StringVar(value="")
        self.addr_var = tk.StringVar(value="1")
        self.baud_var = tk.StringVar(value="1200")
        self.parity_var = tk.StringVar(value="N")
        self.status_var = tk.StringVar(value="未初始化")
        self._build()

    def on_show(self) -> None:
        self.q1_var.set(str(self.app.frontend_config.get("initial_q1", self.q1_var.get()) or "50"))
        self.q2_var.set(str(self.app.frontend_config.get("initial_q2", self.q2_var.get()) or "20"))
        self.addr_var.set(str(self.app.frontend_config.get("pump_address", self.addr_var.get()) or "1"))
        self.baud_var.set(str(self.app.frontend_config.get("pump_baudrate", self.baud_var.get()) or "1200"))
        self.parity_var.set(str(self.app.frontend_config.get("pump_parity", self.parity_var.get()) or "N").upper())

        configured_port = str(self.app.frontend_config.get("pump_port", "") or "").strip().upper()
        detected_ports = self._detect_ports()
        if configured_port and (not detected_ports or configured_port in detected_ports):
            self.port_var.set(configured_port)
            return
        if detected_ports:
            self.port_var.set(detected_ports[0])

    def _build(self) -> None:
        card = ttk.LabelFrame(self, text="初始化参数设置")
        card.pack(fill="x", padx=24, pady=24)

        ttk.Label(card, text="初始 Q1 流速").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.q1_var, width=24).grid(row=0, column=1, padx=8, pady=8, sticky="w")
        ttk.Label(card, text="uL/min").grid(row=0, column=2, padx=8, pady=8, sticky="w")

        ttk.Label(card, text="初始 Q2 流速").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.q2_var, width=24).grid(row=1, column=1, padx=8, pady=8, sticky="w")
        ttk.Label(card, text="uL/min").grid(row=1, column=2, padx=8, pady=8, sticky="w")

        ttk.Label(card, text="泵串口号").grid(row=2, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.port_var, width=24).grid(row=2, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(card, text="泵地址").grid(row=3, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.addr_var, width=24).grid(row=3, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(card, text="波特率").grid(row=4, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.baud_var, width=24).grid(row=4, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(card, text="校验位").grid(row=5, column=0, padx=8, pady=8, sticky="w")
        ttk.Combobox(card, textvariable=self.parity_var, values=("N", "E"), width=21, state="readonly").grid(
            row=5, column=1, padx=8, pady=8, sticky="w"
        )

        ttk.Label(card, text="初始化状态").grid(row=6, column=0, padx=8, pady=8, sticky="w")
        ttk.Label(card, textvariable=self.status_var).grid(row=6, column=1, padx=8, pady=8, sticky="w")

        btns = ttk.Frame(card)
        btns.grid(row=7, column=0, columnspan=3, sticky="ew", padx=8, pady=16)
        ttk.Button(btns, text="上一步", command=lambda: self.app.show_page("video_source")).pack(side="left")
        ttk.Button(btns, text="初始化系统", command=self._initialize).pack(side="right")
        ttk.Button(btns, text="进入监控页面", command=lambda: self.app.show_page("monitor")).pack(side="right", padx=6)

    def _initialize(self) -> None:
        try:
            try:
                import serial  # noqa: F401
            except Exception as e:
                raise ValueError(
                    "当前 Python 环境缺少 pyserial。\n"
                    "请使用 D:\\学习\\microfluidic_control_system\\backend\\venv\\Scripts\\python.exe 启动。\n"
                    f"当前解释器: {sys.executable}"
                ) from e

            q1 = float(self.q1_var.get().strip())
            q2 = float(self.q2_var.get().strip())
            if q1 <= 0:
                raise ValueError("初始 Q1 流速必须大于 0，单位为 uL/min")
            if q2 <= 0:
                raise ValueError("初始 Q2 流速必须大于 0，单位为 uL/min")

            port = self.port_var.get().strip().upper()
            if not port:
                raise ValueError("泵串口号不能为空，请填写设备管理器中的实际 COM 口，例如 COM3")

            addr = int(self.addr_var.get().strip())
            if not (0 <= addr <= 255):
                raise ValueError("泵地址必须在 0~255 之间")

            baud = int(self.baud_var.get().strip())
            if baud <= 0:
                raise ValueError("波特率必须大于 0")

            parity = self.parity_var.get().strip().upper()
            if parity not in {"E", "N"}:
                raise ValueError("校验位仅支持 E 或 N")
        except Exception as e:
            messagebox.showerror("输入错误", str(e))
            return

        self.app.frontend_config["initial_q1"] = q1
        self.app.frontend_config["initial_q2"] = q2
        self.app.frontend_config["pump_port"] = port
        self.app.frontend_config["pump_address"] = addr
        self.app.frontend_config["pump_baudrate"] = baud
        self.app.frontend_config["pump_parity"] = parity
        self.app.runtime_logger(
            "[UI][INIT_SUBMIT] "
            f"q1={q1:.6f}uL/min q2={q2:.6f}uL/min "
            f"port={port} addr={addr} baud={baud} parity={parity}"
        )
        self.status_var.set("初始化中...")

        def task():
            self.app.configure_prepare_initialize()

        def ok():
            self.status_var.set("初始化完成")
            messagebox.showinfo("初始化成功", "系统已初始化，正在进入监控页面")
            self.app.show_page("monitor")

        def fail(err: Exception):
            self.status_var.set("初始化失败")
            detail = str(err)
            diagnose = (
                f"当前参数: port={port}, addr={addr}, baud={baud}, parity={parity}, "
                f"q1={q1} uL/min, q2={q2} uL/min\n\n"
                "建议检查:\n"
                "1. 串口号是否正确且未被其他软件占用\n"
                "2. 泵地址/波特率/校验位是否与设备一致\n"
                "3. 设备上电并通信线连接正常\n"
                "4. 若首次下发失败，重试一次初始化"
            )
            messagebox.showerror("初始化失败", f"{detail}\n\n{diagnose}")

        self.app.run_backend_task(task, on_success=ok, on_error=fail)

    @staticmethod
    def _detect_ports() -> list[str]:
        try:
            from serial.tools import list_ports

            ports = [p.device for p in list_ports.comports() if getattr(p, "device", "")]
        except Exception:
            ports = []
        if not ports:
            return []
        ports.sort(key=lambda item: (len(str(item)), str(item)))
        return [str(port).upper() for port in ports]


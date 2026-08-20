from __future__ import annotations

import threading
import multiprocessing as mp
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

try:
    from backend.orchestrator import OrchestratorService
    from backend.orchestrator.models import SystemConfig
except Exception:  # pragma: no cover
    from ..backend.orchestrator import OrchestratorService
    from ..backend.orchestrator.models import SystemConfig

from .config import APP_TITLE, APP_WINDOW_SIZE, DEFAULT_REFRESH_INTERVAL_MS
from .pages.init_page import InitPage
from .pages.camera_test_page import CameraTestPage
from .pages.monitor_page import MonitorPage
from .pages.parameter_page import ParameterPage
from .pages.pump_test_page import PumpTestPage
from .pages.status_page import StatusPage
from .pages.video_source_page import VideoSourcePage
from .runtime_logging import create_runtime_logger
from .settings_store import FrontendSettingsStore


class FrontendApp(tk.Tk):
    def __init__(
        self,
        orchestrator: OrchestratorService | None = None,
        settings_store: FrontendSettingsStore | None = None,
    ):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(APP_WINDOW_SIZE)
        self.minsize(1080, 720)

        self.runtime_logger = create_runtime_logger()
        self.orchestrator = orchestrator or OrchestratorService(logger=self.runtime_logger)
        self.settings_store = settings_store or FrontendSettingsStore()
        self.frontend_config: dict[str, object] = self.settings_store.load()
        if self.frontend_config:
            self.runtime_logger(f"[APP][SETTINGS] loaded={self.settings_store.path}")
        self.refresh_interval_ms = DEFAULT_REFRESH_INTERVAL_MS
        self._current_page = None

        self._build_layout()
        self._build_pages()
        self.show_page("parameter")

    def update_frontend_config(self, **values: object) -> None:
        """Update validated user input and persist it for the next launch."""
        self.frontend_config.update(values)
        try:
            self.settings_store.save(self.frontend_config)
            self.runtime_logger(f"[APP][SETTINGS] saved={self.settings_store.path}")
        except (OSError, TypeError, ValueError) as exc:
            # A settings write must never prevent the current session from running.
            self.runtime_logger(f"[APP][SETTINGS][ERROR] {exc}")

    def _build_layout(self) -> None:
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

    def _build_pages(self) -> None:
        self.pages = {
            "parameter": ParameterPage(self.container, self),
            "camera_test": CameraTestPage(self.container, self),
            "pump_test": PumpTestPage(self.container, self),
            "video_source": VideoSourcePage(self.container, self),
            "init": InitPage(self.container, self),
            "monitor": MonitorPage(self.container, self),
            "status": StatusPage(self.container, self),
        }
        for page in self.pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

    def show_page(self, key: str) -> None:
        if key not in self.pages:
            raise KeyError(f"unknown page: {key}")
        self.runtime_logger(f"[UI][PAGE] show={key}")
        if self._current_page is not None and hasattr(self._current_page, "on_hide"):
            self._current_page.on_hide()
        page = self.pages[key]
        page.tkraise()
        self._current_page = page
        if hasattr(page, "on_show"):
            page.on_show()

    def build_system_config(self) -> SystemConfig:
        required = (
            "target_diameter",
            "pixel_to_micron",
            "video_source_type",
            "video_source",
            "initial_q1",
            "initial_q2",
            "control_interval_ms",
        )
        missing = [k for k in required if k not in self.frontend_config]
        if missing:
            raise ValueError(f"missing config fields: {missing}")

        return SystemConfig(
            target_diameter=float(self.frontend_config["target_diameter"]),
            pixel_to_micron=float(self.frontend_config["pixel_to_micron"]),
            video_source_type=str(self.frontend_config["video_source_type"]),
            video_source=str(self.frontend_config["video_source"]),
            initial_q1=float(self.frontend_config["initial_q1"]),
            initial_q2=float(self.frontend_config["initial_q2"]),
            control_interval_ms=int(self.frontend_config["control_interval_ms"]),
            pump_port=str(self.frontend_config.get("pump_port", "")).strip(),
            pump_address=int(self.frontend_config.get("pump_address", 1)),
            pump_baudrate=int(self.frontend_config.get("pump_baudrate", 1200)),
            pump_parity=str(self.frontend_config.get("pump_parity", "N")).strip().upper() or "N",
            mvs_sdk_path=str(self.frontend_config.get("mvs_sdk_path", "")).strip(),
            camera_backend=str(self.frontend_config.get("camera_backend", "")).strip(),
            camera_parameters=dict(self.frontend_config.get("camera_parameters", {}) or {}),
            recognition_roi=dict(self.frontend_config.get("recognition_roi", {}) or {}),
        )

    def configure_prepare_initialize(self) -> None:
        cfg = self.build_system_config()
        self.runtime_logger(
            "[APP][CONFIG] "
            f"video_source_type={cfg.video_source_type} video_source={cfg.video_source} "
            f"camera_backend={cfg.camera_backend} control_interval_ms={cfg.control_interval_ms} "
            f"magnification={float(self.frontend_config.get('magnification', 0.0) or 0.0):.6f} "
            f"camera_pixel_size_um={float(self.frontend_config.get('camera_pixel_size_um', 0.0) or 0.0):.6f} "
            f"pixel_to_micron={cfg.pixel_to_micron:.6f} "
            f"pump_port={cfg.pump_port} pump_addr={cfg.pump_address} "
            f"initial_q1={cfg.initial_q1:.6f}uL/min initial_q2={cfg.initial_q2:.6f}uL/min"
        )
        self.orchestrator.configure(cfg)
        self.orchestrator.prepare_video()
        self.orchestrator.initialize_system()

    def run_backend_task(
        self,
        task: Callable[[], object],
        on_success: Callable[[], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        def worker():
            try:
                task()
            except Exception as e:
                error = e
                self.runtime_logger(f"[APP][TASK][ERROR] {error}")
                if on_error is not None:
                    self.after(0, lambda error=error: on_error(error))
                else:
                    self.after(0, lambda error=error: messagebox.showerror("操作失败", str(error)))
                return
            if on_success is not None:
                self.after(0, on_success)

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    mp.freeze_support()
    app = FrontendApp()
    app.mainloop()


if __name__ == "__main__":
    main()

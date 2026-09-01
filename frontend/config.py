from __future__ import annotations

APP_TITLE = "微流控液滴直径反馈控制系统"
APP_WINDOW_SIZE = "1280x840"

DEFAULT_REFRESH_INTERVAL_MS = 300
MIN_REFRESH_INTERVAL_MS = 200
MAX_REFRESH_INTERVAL_MS = 500

# Initial UI suggestion only. Once saved, the user's exact value is retained.
DEFAULT_CONTROL_INTERVAL_MS = 7500
MIN_CONTROL_INTERVAL_MS = 7500
MAX_CONTROL_INTERVAL_MS = 30000

# Fixed commissioning envelope for the current microfluidic setup. The pump
# input page and BO dialog share these values; backend safety checks enforce the
# same envelope before any command can reach the pumps.
DEFAULT_BO_Q1_RANGE = (20.0, 200.0)
DEFAULT_BO_Q2_RANGE = (5.0, 25.0)


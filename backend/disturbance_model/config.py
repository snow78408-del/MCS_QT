from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DisturbanceModelConfig:
    database_path: str = str(Path("data") / "disturbance_model.sqlite3")
    model_path: str = str(Path("data") / "disturbance_model.json")
    sample_interval_ms: int = 200
    minimum_training_samples: int = 50
    minimum_disturbance_events: int = 1
    training_window_size: int = 1000
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    minimum_r2: float = 0.1
    maximum_rmse: float = 30.0
    prediction_timeout_ms: int = 2000
    online_update_enabled: bool = True
    model_update_interval: int = 100
    model_version_limit: int = 5
    storage_queue_size: int = 2000
    storage_batch_size: int = 50
    storage_flush_interval_s: float = 0.5

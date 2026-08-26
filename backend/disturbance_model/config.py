from __future__ import annotations

from dataclasses import dataclass
from .models import DisturbanceControlStage
from ..runtime_paths import user_data_dir


@dataclass(slots=True)
class DisturbanceModelConfig:
    database_path: str = str(user_data_dir() / "data" / "disturbance_model.sqlite3")
    model_path: str = str(user_data_dir() / "models" / "disturbance_model.json")
    sample_interval_ms: int = 200
    prediction_horizon_ms: int = 200
    prediction_horizon_tolerance_ms: int = 100
    minimum_valid_droplets: int = 1
    deployment_stage: str = DisturbanceControlStage.COLLECT_ONLY.value
    low_weight_feedforward_weight: float = 0.2
    full_feedforward_weight: float = 1.0
    shadow_validation_window: int = 40
    shadow_min_comparisons: int = 20
    shadow_max_mae_um: float = 8.0
    max_consecutive_prediction_errors: int = 3
    nonlinear_l2_regularization: float = 0.1
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

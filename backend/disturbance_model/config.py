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
    align_horizon_to_control_cycle: bool = True
    horizon_tolerance_fraction: float = 0.3
    minimum_valid_droplets: int = 1
    deployment_stage: str = DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD.value
    low_weight_feedforward_weight: float = 0.2
    full_feedforward_weight: float = 1.0
    allow_low_weight_feedforward: bool = True
    allow_full_feedforward: bool = False
    shadow_validation_window: int = 40
    shadow_min_comparisons: int = 20
    shadow_max_mae_um: float = 8.0
    shadow_max_change_mae_um: float = 8.0
    shadow_min_direction_accuracy: float = 0.6
    max_consecutive_prediction_errors: int = 3
    nonlinear_l2_regularization: float = 0.1
    minimum_training_samples: int = 50
    minimum_disturbance_events: int = 1
    require_group_metadata: bool = True
    minimum_evaluation_groups: int = 3
    training_window_size: int = 1000
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    minimum_r2: float = 0.1
    maximum_rmse: float = 30.0
    minimum_direction_accuracy: float = 0.6
    minimum_persistence_improvement: float = 0.05
    prediction_timeout_ms: int = 2000
    online_update_enabled: bool = True
    model_update_interval: int = 100
    model_version_limit: int = 5
    storage_queue_size: int = 2000
    storage_batch_size: int = 50
    storage_flush_interval_s: float = 0.5
    inverse_probe_output: float = 1.0
    inverse_min_sensitivity_um_per_output: float = 0.02
    inverse_max_output: float = 50.0
    inverse_q1_control_sign: float = -1.0
    inverse_q2_control_sign: float = 1.0
    inverse_q1_output_gain: float = 2.0
    inverse_q2_output_gain: float = 1.0

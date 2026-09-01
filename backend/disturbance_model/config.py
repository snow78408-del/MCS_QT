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
    deployment_stage: str = DisturbanceControlStage.COLLECT_ONLY.value
    low_weight_feedforward_weight: float = 0.2
    full_feedforward_weight: float = 1.0
    allow_low_weight_feedforward: bool = False
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
    # Online work may train candidates, but candidates never replace the
    # active model or gain pump authority without explicit promotion.
    online_update_enabled: bool = True
    model_update_interval: int = 100
    model_version_limit: int = 5
    candidate_min_relative_improvement: float = 0.05
    storage_queue_size: int = 2000
    storage_batch_size: int = 50
    storage_flush_interval_s: float = 0.5

    def __post_init__(self) -> None:
        if self.minimum_training_samples < 1:
            raise ValueError("minimum_training_samples must be positive")
        if self.model_update_interval < 1:
            raise ValueError("model_update_interval must be positive")
        if self.shadow_min_comparisons < 1 or self.shadow_validation_window < 1:
            raise ValueError("shadow comparison counts must be positive")
        if self.shadow_min_comparisons > self.shadow_validation_window:
            raise ValueError("shadow_min_comparisons cannot exceed shadow_validation_window")
        if not 0.0 <= self.candidate_min_relative_improvement <= 1.0:
            raise ValueError("candidate_min_relative_improvement must be in [0, 1]")

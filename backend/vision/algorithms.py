from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Protocol

import numpy as np

from .config import DebugConfig, DetectorConfig
from .detector import DetectionResult, DropletDetector


@dataclass(frozen=True)
class AlgorithmParameter:
    stage_index: int
    key: str
    label: str
    kind: str = "number"
    text: str = ""


class DetectorAlgorithm(Protocol):
    def detect(self, frame: np.ndarray) -> DetectionResult: ...

    def configure_expected_diameter(self, diameter_um: float, pixel_to_micron: float) -> None: ...

    def reset_adaptive_size(self) -> None: ...

    def runtime_radius_range(self) -> tuple[float, float, float]: ...


@dataclass(frozen=True)
class AlgorithmPlugin:
    plugin_id: str
    display_name: str
    description: str
    config_type: type
    detector_factory: Callable[[Any, DebugConfig], DetectorAlgorithm]
    inspector: Callable[[np.ndarray, Any], tuple[DetectionResult, list[Any]]]
    parameters: tuple[AlgorithmParameter, ...] = ()

    def default_parameters(self) -> dict[str, Any]:
        return asdict(self.config_type())

    def build_config(self, values: dict[str, Any] | None = None) -> Any:
        defaults = self.default_parameters()
        known = {item.name for item in fields(self.config_type)}
        defaults.update({key: value for key, value in dict(values or {}).items() if key in known})
        return self.config_type(**defaults)

    def serialize_config(self, config: Any) -> dict[str, Any]:
        return asdict(config)


_REGISTRY: dict[str, AlgorithmPlugin] = {}


def register_algorithm(plugin: AlgorithmPlugin) -> None:
    plugin_id = str(plugin.plugin_id).strip()
    if not plugin_id:
        raise ValueError("算法插件 ID 不能为空")
    if plugin_id in _REGISTRY:
        raise ValueError(f"算法插件 ID 重复：{plugin_id}")
    _REGISTRY[plugin_id] = plugin


def get_algorithm(plugin_id: str) -> AlgorithmPlugin:
    try:
        return _REGISTRY[str(plugin_id)]
    except KeyError as exc:
        raise ValueError(f"算法插件不存在：{plugin_id}") from exc


def list_algorithms() -> tuple[AlgorithmPlugin, ...]:
    return tuple(_REGISTRY.values())


def _inspect_hybrid(frame: np.ndarray, config: DetectorConfig):
    # Lazy import avoids making the real-time pipeline depend on tuning UI helpers.
    from .tuning import inspect_hybrid_frame

    return inspect_hybrid_frame(frame, config)


_HYBRID_PARAMETERS = (
    AlgorithmParameter(2, "enable_intensity_normalization", "启用归一化", "check", "执行此步骤"),
    AlgorithmParameter(3, "enable_gaussian_blur", "启用输入平滑", "check", "执行此步骤"),
    AlgorithmParameter(3, "gaussian_blur_size", "高斯核大小"),
    AlgorithmParameter(4, "enable_contour_candidates", "启用快速轮廓主检测", "check", "执行此步骤"),
    AlgorithmParameter(4, "contour_work_scale", "候选检测缩放"),
    AlgorithmParameter(5, "adaptive_threshold_block_size", "自适应阈值块大小"),
    AlgorithmParameter(5, "adaptive_threshold_c", "自适应阈值 C"),
    AlgorithmParameter(5, "enable_morphology", "启用形态学清理", "check", "执行此步骤"),
    AlgorithmParameter(5, "morphology_open_kernel", "开运算核"),
    AlgorithmParameter(5, "morphology_close_kernel", "闭运算核"),
    AlgorithmParameter(6, "enable_background_subtraction", "启用背景差分", "check", "执行此步骤"),
    AlgorithmParameter(6, "background_difference_threshold", "背景差分阈值"),
    AlgorithmParameter(6, "background_learning_rate", "背景学习率"),
    AlgorithmParameter(7, "contour_canny_low", "Canny 低阈值"),
    AlgorithmParameter(7, "contour_canny_high", "Canny 高阈值"),
    AlgorithmParameter(7, "contour_close_kernel", "边缘闭运算核"),
    AlgorithmParameter(8, "min_radius", "绝对最小半径"),
    AlgorithmParameter(8, "max_radius", "绝对最大半径"),
    AlgorithmParameter(8, "contour_min_circularity", "最小圆度"),
    AlgorithmParameter(8, "contour_min_axis_ratio", "最小长短轴比"),
    AlgorithmParameter(8, "contour_min_edge_support", "最小轮廓边缘支撑"),
    AlgorithmParameter(8, "contour_min_area_fill_ratio", "最小椭圆填充率"),
    AlgorithmParameter(9, "enable_watershed_split", "启用局部 Watershed", "check", "执行此步骤"),
    AlgorithmParameter(9, "watershed_peak_ratio", "距离峰比例"),
    AlgorithmParameter(9, "watershed_min_peak_radius_ratio", "最小峰半径比例"),
    AlgorithmParameter(9, "watershed_max_markers", "局部最大标记数"),
    AlgorithmParameter(10, "enable_hough_candidates", "启用疑难区域 Hough", "check", "执行此步骤"),
    AlgorithmParameter(10, "local_hough_padding_ratio", "局部 Hough 边距比例"),
    AlgorithmParameter(10, "local_hough_max_regions", "每帧最大疑难区域"),
    AlgorithmParameter(10, "hough_refresh_interval", "全帧 Hough 周期（0=关闭）"),
    AlgorithmParameter(10, "hough_dp", "累加器分辨率 dp"),
    AlgorithmParameter(10, "hough_min_distance", "最小圆心距离（0=自动）"),
    AlgorithmParameter(10, "hough_param1", "Hough 梯度阈值"),
    AlgorithmParameter(10, "hough_param2", "Hough 累加阈值"),
    AlgorithmParameter(10, "hough_min_radius", "Hough 最小半径"),
    AlgorithmParameter(10, "hough_max_radius", "Hough 最大半径"),
    AlgorithmParameter(10, "hough_edge_support_threshold", "Hough 边缘支撑"),
    AlgorithmParameter(11, "enable_expected_size_filter", "启用目标尺寸过滤", "check", "执行此步骤"),
    AlgorithmParameter(11, "expected_radius", "期望液滴半径（px）"),
    AlgorithmParameter(11, "expected_radius_tolerance_ratio", "半径容差比例"),
    AlgorithmParameter(11, "candidate_min_edge_support", "最终最小边缘支撑"),
    AlgorithmParameter(11, "candidate_min_visible_circle_ratio", "跟踪最小可见圆周"),
    AlgorithmParameter(11, "candidate_full_circle_ratio", "直径有效完整度"),
)

register_algorithm(
    AlgorithmPlugin(
        plugin_id="hybrid_v1",
        display_name="混合轮廓 + Hough（现有算法）",
        description="现有的轮廓、局部 Watershed、Hough 融合液滴检测流程。",
        config_type=DetectorConfig,
        detector_factory=DropletDetector,
        inspector=_inspect_hybrid,
        parameters=_HYBRID_PARAMETERS,
    )
)

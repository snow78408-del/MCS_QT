"""Plug-geometry conversions shared by the detector, the tuner and the plant model.

The generation-zone detector reports one number per plug: an *equivalent-sphere*
diameter.  It is not the diameter of any sphere that exists inside the device.
It is the diameter of the sphere that would have the same volume as the plug
that fills the duct.  Everything downstream (PID target, MPC steady state,
calibration records, the on-screen number) is written in that unit, so the
conversion must exist in exactly one place:

    A_eff = h*w - (4 - pi) / (2/h + 2/w)**2      rounded-rectangle cross-section
    V     = correction * A_eff * (L - w/3)       meniscus end correction
    d_eq  = cbrt(6*V/pi)                         volume-equivalent sphere

For the project's square duct (``h == w``) the area term collapses to
``0.94635 * w**2``; ``generation_volume_correction`` stays a free calibration
factor so the bench can absorb the corner films the formula cannot see.

Worked example, 2026-09-18 observation window (gap 27.011 px = 50 um, so
``scale = 1.8511 um/px``): a plug 107 px (198 um, 3.96 channel widths) long is
reported as ``d_eq = 50.56 px = 93.6 um``.  The number is a *volume* statement,
not a droplet size: the physical plug is a 4:1 slug in a 50 um channel, and the
same volume would form a 94 um sphere.  Reading ``d_eq`` as the droplet diameter
is the most common misreading of these records.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_END_CORRECTION_WIDTHS = 1.0 / 3.0
"""Axial end correction, in channel widths, applied inside the volume formula.

The detector has always subtracted ``w/3`` (van Steijn et al., Sci. Rep. 2017).
An independent fit of the 2026-09-18 observation window prefers ``0.48*w``; the
difference is 4% of length and 1.4% of volume, so the detector's value is kept
until a bench measurement justifies changing it.
"""

RECTIFIED_FLOW_AXIS = "x"
"""扶正图里流向所在的轴。由扶正变换的目标四边形定死，不是按图像长边推断的。"""


@dataclass(frozen=True)
class RectifiedAxes:
    """扶正图的坐标轴与跨度定义。**这是轴向/横向/跨度定义的唯一来源。**

    轴向由**扶正变换契约定死**，不是推断出来的：``rectify_channel_frame`` 的目标四边形是
    ``[[0,0],[W-1,0],[W-1,H-1],[0,H-1]]``，管壁方向映到 **x**、跨管方向映到 **y**。
    因此扶正图的 ``flow_axis`` 恒为 ``"x"``，与图像哪一边更长**无关**。
    早先按「流向沿较长边」推断的写法会把 80×40 的扶正图判成沿 y 流动，是错的。

    三个量必须分开命名，不可互相替代：

    * ``transverse_pixel_count`` / ``axial_pixel_count``：该轴上的**像素个数**
      （就是 ``shape`` 的分量）。
    * ``transverse_index_span`` / ``axial_index_span``：该轴上的**索引跨度** =
      像素个数 − 1。测量得到的柱塞长度是索引差，所以**门槛与检测参考宽必须用索引跨度**。
    * **几何长度**（源图中的实际距离）由
      :func:`backend.vision.rectified_roi.wall_separation_px` 给出，即透视输出高度的
      像素距离；它与索引跨度相差端点归一化（目标把端点映到 ``W-1``/``H-1``），
      两者不可混用。

    ``transverse_index_span`` 在数值上等于 ``min(shape) - 1``。这**只**因为上面的变换
    契约才成立，所以调用方必须用 :func:`rectified_axes` 取值，不得再用 ``min(shape)``
    猜——那个写法既没表达减一，也没表达轴向由契约决定。
    """

    output_width: int
    output_height: int
    axial_pixel_count: int
    transverse_pixel_count: int
    axial_index_span: int
    transverse_index_span: int
    flow_axis: str = RECTIFIED_FLOW_AXIS

    def to_dict(self) -> dict[str, int | str]:
        return {
            "output_width": int(self.output_width),
            "output_height": int(self.output_height),
            "flow_axis": self.flow_axis,
            "axial_pixel_count": int(self.axial_pixel_count),
            "transverse_pixel_count": int(self.transverse_pixel_count),
            "axial_index_span": int(self.axial_index_span),
            "transverse_index_span": int(self.transverse_index_span),
            "span_units": "index_span",
            "pixel_count_units": "pixels",
            "axial_axis_defined_by": "rectification_contract",
            "wider_than_tall": bool(self.output_width >= self.output_height),
        }


def rectified_axes(shape) -> RectifiedAxes:
    """由扶正图的 ``shape`` 给出坐标轴与跨度。非法尺寸直接拒绝，不回退。

    轴向**不推断**：扶正变换把管壁方向映到 x，所以 ``flow_axis`` 恒为 ``"x"``。
    只接受**恰好两个**维度的尺寸（调用方传 ``shape[:2]``）。传整幅三维 shape 会被
    拒绝，避免把通道数当成高度这类静默错误。
    """
    try:
        values = tuple(int(v) for v in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法由 {shape!r} 取得扶正图尺寸") from exc
    if len(values) != 2:
        raise ValueError(f"扶正图尺寸必须是 (height, width)，得到 {shape!r}")
    height, width = values
    if height <= 0 or width <= 0:
        raise ValueError(f"扶正图尺寸必须为正，得到 {width}x{height}")
    return RectifiedAxes(
        output_width=width,
        output_height=height,
        axial_pixel_count=width,
        transverse_pixel_count=height,
        axial_index_span=width - 1,
        transverse_index_span=height - 1,
    )


def effective_area_px2(height_px: float, width_px: float) -> float:
    """Cross-section area of a duct with rounded corners, in px**2."""
    height = float(height_px)
    width = float(width_px)
    if height <= 0.0 or width <= 0.0:
        raise ValueError("通道截面尺寸必须为正")
    inverse_sum = (2.0 / height) + (2.0 / width)
    return height * width - (4.0 - np.pi) * inverse_sum ** -2


def plug_volume_px3(
    length_px: float,
    height_px: float,
    width_px: float,
    correction: float = 1.0,
    end_correction_widths: float = DEFAULT_END_CORRECTION_WIDTHS,
) -> float:
    """Volume of a plug seen between two menisci, in px**3.

    ``length_px`` is the meniscus-to-meniscus axial extent.  Both menisci are
    rounded caps, so a width-scaled term is removed before multiplying by the
    cross-section.
    """
    length = float(length_px)
    width = float(width_px)
    factor = float(correction)
    if width <= 0.0:
        raise ValueError("通道宽度必须为正")
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("体积修正系数必须为正的有限值")
    filled = length - float(end_correction_widths) * width
    if not np.isfinite(filled) or filled <= 0.0:
        raise ValueError("柱塞长度不足以形成有效体积")
    return factor * effective_area_px2(height_px, width_px) * filled


def equivalent_sphere_diameter_px(
    length_px: float,
    height_px: float,
    width_px: float,
    correction: float = 1.0,
    end_correction_widths: float = DEFAULT_END_CORRECTION_WIDTHS,
) -> float:
    """Volume-equivalent sphere diameter of a plug, in px.

    This is the number the generation-zone detector publishes.
    """
    volume = plug_volume_px3(
        length_px, height_px, width_px, correction, end_correction_widths
    )
    return float(np.cbrt(6.0 * volume / np.pi))


def plug_length_px_from_equivalent_diameter(
    diameter_px: float,
    height_px: float,
    width_px: float,
    correction: float = 1.0,
    end_correction_widths: float = DEFAULT_END_CORRECTION_WIDTHS,
) -> float:
    """Inverse of :func:`equivalent_sphere_diameter_px`.

    Used to translate a PID/MPC setpoint back into the plug the optics can
    actually produce, which is what a bench operator can verify with a ruler.
    """
    diameter = float(diameter_px)
    width = float(width_px)
    factor = float(correction)
    if diameter <= 0.0:
        raise ValueError("等效直径必须为正")
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("体积修正系数必须为正的有限值")
    volume = np.pi * diameter ** 3 / 6.0
    return float(
        volume / (factor * effective_area_px2(height_px, width_px))
        + float(end_correction_widths) * width
    )


@dataclass(frozen=True)
class PlugGeometry:
    """One plug in both the pixel and the micron domain."""

    length_px: float
    duct_width_px: float
    scale_um_per_px: float
    height_px: float | None = None
    correction: float = 1.0

    @property
    def duct_height_px(self) -> float:
        return self.duct_width_px if self.height_px is None else float(self.height_px)

    @property
    def length_um(self) -> float:
        return float(self.length_px) * float(self.scale_um_per_px)

    @property
    def duct_width_um(self) -> float:
        return float(self.duct_width_px) * float(self.scale_um_per_px)

    @property
    def aspect_ratio(self) -> float:
        return float(self.length_px) / float(self.duct_width_px)

    @property
    def equivalent_diameter_px(self) -> float:
        return equivalent_sphere_diameter_px(
            self.length_px, self.duct_height_px, self.duct_width_px, self.correction
        )

    @property
    def equivalent_diameter_um(self) -> float:
        return self.equivalent_diameter_px * float(self.scale_um_per_px)

    @property
    def volume_um3(self) -> float:
        return plug_volume_px3(
            self.length_px, self.duct_height_px, self.duct_width_px, self.correction
        ) * float(self.scale_um_per_px) ** 3

    @property
    def volume_nl(self) -> float:
        return self.volume_um3 / 1.0e6

    def to_dict(self) -> dict[str, float]:
        return {
            "length_px": float(self.length_px),
            "length_um": self.length_um,
            "duct_width_px": float(self.duct_width_px),
            "duct_width_um": self.duct_width_um,
            "aspect_ratio": self.aspect_ratio,
            "equivalent_diameter_px": self.equivalent_diameter_px,
            "equivalent_diameter_um": self.equivalent_diameter_um,
            "volume_um3": self.volume_um3,
            "volume_nl": self.volume_nl,
            "correction": float(self.correction),
        }


def sphere_diameter_um(volume_um3: float) -> float:
    """Diameter of the sphere holding ``volume_um3`` micrometres cubed."""
    volume = float(volume_um3)
    if volume <= 0.0:
        raise ValueError("体积必须为正")
    return float(np.cbrt(6.0 * volume / np.pi))

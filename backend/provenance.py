"""运行追溯：源码快照、生效配置、帧引用与离线核验。

**清单范围**（:data:`SOURCE_MANIFEST`）覆盖实际入口及其**依赖链**：定位、扶正、检测、
测量几何、配置、实时管线、编排适配与服务、记录/时钟、设备互斥。只对入口取哈希不足以
复现一次运行。**边界**：不覆盖与本次测量链无关的模块（PID 控制器内部、前端、扰动模型、
优化器）——它们影响的是控制策略，不是这张图上的像素与几何。

为什么保存**文件内容副本**而不只存 Git 提交号：本仓库的台架入口脚本大量处于未跟踪或
未提交状态，提交号代表不了它们。Git 提交号与 ``git status`` 只作辅助并显式标注角色。

核验分四级，**逐级独立**，不可互相代替：

1. **引用结构合法**：字段类型、非负整数索引、路径齐备（:func:`validate_reference_structure`）。
2. **媒体存在**：容器/静帧文件在当前会话根目录下确实存在（:func:`verify_frame_references`）。
3. **帧可解码**：解码帧索引落在容器帧数范围内（:func:`resolve_source_frame`）。
4. **内容匹配**：静帧 SHA-256 与记录值一致；原始块按 offset/nbytes 取出的内容哈希一致。

只有第 4 级通过才算 ``matched``；``resolved`` 只在第 3 级通过时给出。**软件帧序号与解码
帧索引可以合法不同**，所以不把二者相等当作真实性检验。

本模块**不导入即连接设备**：只读写文件。
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_MANIFEST_VERSION = 2
FRAME_REFERENCE_VERSION = 2
PROVENANCE_VERSION = 2

SOURCE_MANIFEST: tuple[str, ...] = (
    # 入口
    "tools/plant_flow_transient_capture.py",
    "tools/run_r1_front4_live.py",
    "tools/run_channel_validation_live.py",
    "tools/replay_wall_localization.py",
    "tools/verify_provenance.py",
    # 追溯自身
    "backend/provenance.py",
    # 定位
    "backend/vision/parallel_walls.py",
    # 扶正
    "backend/vision/rectified_roi.py",
    # 检测
    "backend/vision/detector.py",
    "backend/vision/capsule_profile.py",
    # 测量与几何
    "backend/vision/rectified_measurement.py",
    "backend/vision/plug_geometry.py",
    # 配置
    "backend/vision/config.py",
    # 实时管线与编排链（复审指出此前遗漏：它们决定实际喂给检测器的几何）
    "backend/vision/pipeline.py",
    "backend/orchestrator/vision_adapter.py",
    "backend/orchestrator/service.py",
    "backend/orchestrator/state.py",
    "backend/orchestrator/models.py",
    # 记录与时钟
    "backend/vision/cameras/models.py",
    # 设备互斥
    "backend/device_lock.py",
    "backend/runtime_paths.py",
)
"""参与快照的源文件。边界见模块 docstring；新增依赖必须显式加进来。"""

REQUIRED_CONFIG_SECTIONS: tuple[str, ...] = (
    "detector", "strict_detection_localization", "wall_source", "wall_binding",
    "image_shape", "scale_declaration", "depth_declaration", "imaging", "sampling",
    "localization", "config_change_policy",
)
"""生效配置必须齐备的分区。缺分区即视为结构不完整。"""

KEY_DEPENDENCIES: tuple[str, ...] = ("numpy", "cv2", "PySide6")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                                text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_versions() -> dict:
    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for name in KEY_DEPENDENCIES:
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - 缺依赖只是记录为不可用
            versions[name] = None
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def source_fingerprint(entries: list[dict]) -> str:
    """把逐文件 SHA-256 汇总成一个总指纹（与文件顺序无关）。"""
    digest = hashlib.sha256()
    for item in sorted(entries, key=lambda entry: entry["path"]):
        digest.update(str(item["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def config_fingerprint(effective: dict) -> str:
    """生效配置的内容指纹。测量行引用它来绑定「当时的配置版本」。"""
    return sha256_bytes(canonical_json(effective))


class ProvenanceError(RuntimeError):
    """追溯写入失败。现场入口必须据此在**硬件动作之前**拒绝启动。"""


def snapshot_sources(destination: Path, *, root: Path | None = None) -> dict:
    """保存源文件副本 + 逐文件 SHA-256 + 总指纹。

    一致性检查读的是**源文件**：先记录复制前的哈希，复制，再在末尾**重新读取源文件**
    与副本，三者比对。只比「副本 vs 复制前值」查不出「复制之后源文件又变了」。

    **限制**：这只能检出快照期间的源文件变化，**不等于**证明运行进程实际加载的就是磁盘上
    这一版（进程可能早已导入旧版）。运行版本需要用源码快照 + 一次性写入策略来固定。
    """
    root = Path(root or Path(__file__).resolve().parents[1])
    destination = Path(destination)
    copies = destination / "sources"
    copies.mkdir(parents=True, exist_ok=True)
    before: dict[str, str] = {}
    missing: list[str] = []
    for relative in SOURCE_MANIFEST:
        source = root / relative
        if not source.exists():
            missing.append(relative)
            continue
        before[relative] = sha256_file(source)
        target = copies / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    entries: list[dict] = []
    changed_during: list[str] = []
    source_changed_after_copy: list[str] = []
    for relative in SOURCE_MANIFEST:
        if relative in missing:
            continue
        copy = copies / relative
        copy_hash = sha256_file(copy)
        source_hash_now = sha256_file(root / relative)
        if copy_hash != before[relative]:
            changed_during.append(relative)
        # 复制之后源文件又被改动：副本既不是当时的源，也不等于现在的源。
        if source_hash_now != before[relative]:
            source_changed_after_copy.append(relative)
        entries.append({
            "path": relative,
            "sha256": copy_hash,
            "bytes": copy.stat().st_size,
            "source_mtime_ns": (root / relative).stat().st_mtime_ns,
            "copied_as": f"sources/{relative}",
        })
    payload = {
        "manifest_version": SOURCE_MANIFEST_VERSION,
        "scope_note": ("覆盖入口 + 定位/扶正/检测/测量几何/配置/实时管线/编排/记录时钟/"
                       "设备互斥依赖链；不含 PID 内部、前端、扰动模型与优化器"),
        "scope_boundary": {"included": "影响图像几何与测量链的模块",
                           "excluded": ["backend/pid_control/**", "frontend/**",
                                        "backend/disturbance_model/**",
                                        "backend/optimization/**"]},
        "files": entries,
        "missing_manifest_files": missing,
        "source_fingerprint": source_fingerprint(entries),
        "changed_during_snapshot": changed_during,
        "source_changed_after_copy": source_changed_after_copy,
        "consistent_snapshot": (not changed_during and not source_changed_after_copy
                                and not missing),
        "consistency_limits": ("只检出快照期间/复制后的源文件变化；不证明运行进程加载的是"
                               "磁盘同一版本"),
        "runtime_versions": runtime_versions(),
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "describe": _git(root, "describe", "--always", "--dirty"),
            "status_porcelain": (_git(root, "status", "--porcelain") or "").splitlines(),
            "role": "辅助信息：未跟踪/未提交的代码由文件快照代表，不由提交号代表",
        },
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_effective_config(destination: Path, effective: dict, *, config_version: int = 1) -> dict:
    """写出生效配置并把它的内容指纹登记进快照清单。"""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(effective)
    (destination / "effective_config.json").write_text(
        json.dumps(effective, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = destination / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"manifest_version": SOURCE_MANIFEST_VERSION}
    manifest["config_fingerprint"] = fingerprint
    manifest["config_version"] = int(config_version)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return {"config_fingerprint": fingerprint, "config_version": int(config_version)}


def build_bundle(destination: Path, *, effective: dict, root: Path | None = None,
                 config_version: int = 1) -> dict:
    """一次完整的追溯写入。失败即抛 :class:`ProvenanceError`，调用方必须在硬件动作前拒绝启动。"""
    try:
        snapshot = snapshot_sources(destination, root=root)
        config = write_effective_config(destination, effective, config_version=config_version)
    except Exception as exc:  # noqa: BLE001 - 转成显式追溯失败
        raise ProvenanceError(f"追溯写入失败，禁止进入硬件动作：{exc!r}") from exc
    if not snapshot.get("consistent_snapshot"):
        raise ProvenanceError(
            "源码快照不一致（快照期间或复制后源文件发生变化），不能作为确定的运行版本："
            f"changed_during_snapshot={snapshot.get('changed_during_snapshot')} "
            f"source_changed_after_copy={snapshot.get('source_changed_after_copy')} "
            f"missing={snapshot.get('missing_manifest_files')}")
    return {"source": snapshot, "config": config,
            "source_fingerprint": snapshot["source_fingerprint"],
            "config_fingerprint": config["config_fingerprint"]}


def verify_source_snapshot(directory: Path) -> dict:
    """离线核验：逐文件哈希、按**实际内容**重算的总指纹、缺失与篡改。"""
    directory = Path(directory)
    manifest_path = directory / "source_manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "reason": "source_manifest_missing",
                "manifest_path": str(manifest_path)}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(payload.get("files") or [])
    missing: list[str] = []
    mismatched: list[str] = []
    actual_entries: list[dict] = []
    for item in entries:
        copy = directory / str(item.get("copied_as") or "")
        if not copy.exists():
            missing.append(str(item.get("path")))
            continue
        digest = sha256_file(copy)
        if digest != item.get("sha256"):
            mismatched.append(str(item.get("path")))
        actual_entries.append({"path": str(item.get("path")), "sha256": digest})
    fingerprint = source_fingerprint(actual_entries)
    return {
        "ok": bool(not missing and not mismatched
                   and fingerprint == payload.get("source_fingerprint")
                   and payload.get("consistent_snapshot")),
        "checked": len(entries),
        "missing_snapshot_files": missing,
        "mismatched": mismatched,
        "fingerprint_ok": fingerprint == payload.get("source_fingerprint"),
        "recomputed_fingerprint": fingerprint,
        "recorded_fingerprint": payload.get("source_fingerprint"),
        "consistent_snapshot": payload.get("consistent_snapshot"),
        "source_changed_after_copy": payload.get("source_changed_after_copy"),
    }


def verify_effective_config(directory: Path) -> dict:
    """离线核验生效配置：结构完整性 + **内容指纹**与快照清单登记值一致。"""
    directory = Path(directory)
    config_path = directory / "effective_config.json"
    manifest_path = directory / "source_manifest.json"
    if not config_path.exists():
        return {"ok": False, "reason": "effective_config_missing"}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    leaves: list[tuple[str, dict]] = []
    _walk_leaves("", payload, leaves)
    missing_sections = [name for name in REQUIRED_CONFIG_SECTIONS if name not in payload]
    withheld = []
    bad_readback = []
    for name, item in leaves:
        if item.get("value") is None:
            withheld.append(name)
        if name.split(".")[-1].endswith("_readback") and not item.get("readback_source"):
            bad_readback.append(name)
        if "readback_source" in item and not item.get("readback_source"):
            bad_readback.append(name)
    recorded = None
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("config_fingerprint")
    actual = config_fingerprint(payload)
    return {
        "ok": bool(not missing_sections and not bad_readback and recorded is not None
                   and recorded == actual),
        "version": payload.get("provenance_version"),
        "missing_sections": missing_sections,
        "withheld_unknown": sorted(withheld),
        "readback_without_source": sorted(set(bad_readback)),
        "config_fingerprint_ok": recorded == actual,
        "recorded_config_fingerprint": recorded,
        "recomputed_config_fingerprint": actual,
    }


def _walk_leaves(prefix: str, node, out: list[tuple[str, dict]]) -> None:
    if isinstance(node, dict):
        if "value" in node and "reason" in node:
            out.append((prefix, node))
            return
        for key, child in node.items():
            _walk_leaves(f"{prefix}.{key}" if prefix else str(key), child, out)


# --------------------------------------------------------------- 生效配置

@dataclass
class EffectiveConfig:
    """合并默认值、配置文件与命令行覆盖后的**生效**配置。"""

    detector: dict = field(default_factory=dict)
    strict_detection_localization: dict = field(default_factory=dict)
    wall_source: dict = field(default_factory=dict)
    wall_binding: dict = field(default_factory=dict)
    image_shape: dict = field(default_factory=dict)
    scale_declaration: dict = field(default_factory=dict)
    depth_declaration: dict = field(default_factory=dict)
    imaging: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    localization: dict = field(default_factory=dict)
    config_change_policy: dict = field(default_factory=dict)
    provenance_version: int = PROVENANCE_VERSION

    def to_dict(self) -> dict:
        return {
            "provenance_version": int(self.provenance_version),
            "detector": dict(self.detector),
            "strict_detection_localization": dict(self.strict_detection_localization),
            "wall_source": dict(self.wall_source),
            "wall_binding": dict(self.wall_binding),
            "image_shape": dict(self.image_shape),
            "scale_declaration": dict(self.scale_declaration),
            "depth_declaration": dict(self.depth_declaration),
            "imaging": dict(self.imaging),
            "sampling": dict(self.sampling),
            "localization": dict(self.localization),
            "config_change_policy": dict(self.config_change_policy),
            "unknown_policy": ("未知写 null + reason；设定值不等于实测值，"
                               "读回值必须附 readback_source"),
        }


def unknown(reason: str) -> dict:
    return {"value": None, "reason": str(reason)}


def declared(value, *, reason: str) -> dict:
    return {"value": value, "reason": str(reason)}


def readback(value, *, source: str) -> dict:
    if not source:
        raise ValueError("读回值必须给出来源（readback_source）")
    return {"value": value, "reason": str(source), "readback_source": str(source)}


# --------------------------------------------------------------- 帧引用

def validate_reference_structure(reference: dict) -> list[str]:
    """第 1 级：结构合法性。返回问题列表（空即通过）。

    校验按**媒体类型**分别要求：容器引用必须有解码帧索引；静帧与原始块没有解码帧索引。
    ``hardware_frame_id`` 可以缺失——实机上确实存在取不到硬件帧号的情况（例如 direct
    路径未填充），缺失不是结构错误；但给出来就必须是非负整数。
    """
    problems: list[str] = []
    if not isinstance(reference, dict):
        return ["reference_not_a_dict"]
    if reference.get("available") is not True:
        return [] if reference.get("missing_reason") else ["missing_reason_absent"]

    software = reference.get("software_frame_index")
    if isinstance(software, bool) or not isinstance(software, int):
        problems.append("software_frame_index_not_int")
    elif software < 0:
        problems.append("software_frame_index_negative")

    hardware = reference.get("hardware_frame_id")
    if hardware is not None:
        if isinstance(hardware, bool) or not isinstance(hardware, int):
            problems.append("hardware_frame_id_not_int")
        elif hardware < 0:
            problems.append("hardware_frame_id_negative")

    monotonic = reference.get("capture_monotonic")
    if not isinstance(monotonic, (int, float)) or isinstance(monotonic, bool):
        problems.append("capture_monotonic_not_number")
    elif float(monotonic) <= 0.0:
        problems.append("capture_monotonic_not_positive")

    if not str(reference.get("session_root") or "").strip():
        problems.append("session_root_absent")

    kinds = [name for name in ("container_path", "still_path", "raw_block")
             if reference.get(name) is not None]
    if not kinds:
        problems.append("no_media_reference")
    elif len(kinds) > 1:
        problems.append("multiple_media_references")

    if reference.get("container_path") is not None:
        index = reference.get("decoded_frame_index")
        if isinstance(index, bool) or not isinstance(index, int):
            problems.append("decoded_frame_index_not_int")
        elif index < 0:
            problems.append("decoded_frame_index_negative")

    block = reference.get("raw_block")
    if isinstance(block, dict):
        if not str(block.get("path") or "").strip():
            problems.append("raw_block_path_absent")
        offset = block.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            problems.append("raw_block_offset_invalid")
        nbytes = block.get("nbytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
            problems.append("raw_block_nbytes_invalid")
    return problems


def frame_reference(*, session_id: str, capture_id: str, session_root: str,
                    software_frame_index: int | None, hardware_frame_id: int | None,
                    capture_monotonic: float | None, container_path: str | None,
                    decoded_frame_index: int | None,
                    still_path: str | None = None, still_sha256: str | None = None,
                    raw_block: dict | None = None,
                    content_sha256: str | None = None,
                    geometry_version: str | None = None,
                    coordinate_space: str | None = None,
                    transform: list | None = None,
                    config_fingerprint: str | None = None,
                    missing_reason: str | None = None) -> dict:
    """一帧的完整引用。

    软件序号、硬件帧号、采集时间**分别**保存，互不冒名；硬件帧号可能重启或回绕，
    因此另有 ``session_id``/``capture_id`` 作为会话内主键。``session_root`` 是相对
    路径的基准目录，搬迁后仍可核验。
    """
    if missing_reason:
        return {"version": FRAME_REFERENCE_VERSION, "available": False,
                "missing_reason": str(missing_reason),
                "session_id": session_id, "capture_id": capture_id,
                "session_root": session_root}
    if container_path is None and still_path is None and raw_block is None:
        raise ValueError("必须给出容器路径、静帧路径或原始块描述之一")
    root = str(session_root or "").strip()
    if not root:
        raise ValueError("必须给出非空的 session_root：相对路径需要一个基准目录")
    return {
        "version": FRAME_REFERENCE_VERSION,
        "available": True,
        "session_id": str(session_id),
        "capture_id": str(capture_id),
        "session_root": root,
        "software_frame_index": None if software_frame_index is None else int(software_frame_index),
        "hardware_frame_id": None if hardware_frame_id is None else int(hardware_frame_id),
        "capture_monotonic": None if capture_monotonic is None else float(capture_monotonic),
        "container_path": container_path,
        "decoded_frame_index": (None if decoded_frame_index is None
                                else int(decoded_frame_index)),
        "still_path": still_path,
        "still_sha256": still_sha256,
        "raw_block": raw_block,
        "content_sha256": content_sha256,
        "geometry_version": geometry_version,
        "coordinate_space": coordinate_space,
        "transform": transform,
        "config_fingerprint": config_fingerprint,
        "path_note": "路径相对 session_root，搬迁后仍可核验",
    }


def _media_result(level: str, ok: bool, detail: str, **extra) -> dict:
    return {"level": level, "ok": bool(ok), "detail": detail, **extra}


def resolve_source_frame(reference: dict, *, session_root: Path | str | None = None,
                         decode: bool = False) -> dict:
    """从一条测量行找回确切源帧，逐级校验：结构 → 媒体存在 → 帧可解码 → 内容匹配。

    ``resolved`` 只在**帧可解码**（或容器帧数确认索引在范围内）时为真；``matched`` 只在
    内容哈希比对通过时为真。缺引用一律显式标缺失，绝不用邻近静帧或序号猜配对。
    """
    levels: list[dict] = []
    if not isinstance(reference, dict) or reference.get("available") is not True:
        return {"resolved": False, "matched": False,
                "reason": (reference or {}).get("missing_reason", "reference_missing")
                if isinstance(reference, dict) else "reference_missing",
                "levels": levels}

    problems = validate_reference_structure(reference)
    levels.append(_media_result("structure", not problems,
                               "结构合法" if not problems else "; ".join(problems)))
    if problems:
        return {"resolved": False, "matched": False, "reason": "reference_structure_invalid",
                "problems": problems, "levels": levels}

    root = Path(session_root) if session_root is not None else Path(reference["session_root"])
    container = reference.get("container_path")
    if container is not None:
        media = (root / container).resolve()
        if not media.exists():
            levels.append(_media_result("media_exists", False, f"容器不存在：{media}"))
            return {"resolved": False, "matched": False, "reason": "container_missing",
                    "container_path": str(media), "levels": levels}
        levels.append(_media_result("media_exists", True, f"容器存在：{media}",
                                    bytes=media.stat().st_size))
        import cv2  # 局部导入：模块导入阶段不引入图像依赖

        capture = cv2.VideoCapture(str(media))
        try:
            if not capture.isOpened():
                levels.append(_media_result("decodable", False, "容器无法打开"))
                return {"resolved": False, "matched": False, "reason": "container_unopenable",
                        "levels": levels}
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            index = int(reference["decoded_frame_index"])
            if total > 0 and not 0 <= index < total:
                levels.append(_media_result("frame_in_range", False,
                                            f"索引 {index} 超出 [0,{total})"))
                return {"resolved": False, "matched": False, "reason": "frame_index_out_of_range",
                        "container_frames": total, "levels": levels}
            levels.append(_media_result("frame_in_range", True,
                                        f"索引 {index} 在 [0,{total}) 内",
                                        container_frames=total))
            if decode:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    levels.append(_media_result("decodable", False, f"索引 {index} 解码失败"))
                    return {"resolved": False, "matched": False, "reason": "frame_undecodable",
                            "levels": levels}
                digest = sha256_bytes(frame.tobytes())
                levels.append(_media_result("decodable", True, f"索引 {index} 解码成功",
                                            decoded_shape=list(frame.shape)))
                expected = reference.get("content_sha256")
                matched = expected is not None and expected == digest
                levels.append(_media_result("content_match", matched,
                                            "内容哈希一致" if matched
                                            else ("未记录内容哈希，无法比对" if expected is None
                                                  else "内容哈希不一致"),
                                            decoded_sha256=digest,
                                            expected_sha256=expected))
                return {"resolved": True, "matched": matched,
                        "container_path": str(media), "decoded_frame_index": index,
                        "software_frame_index": reference.get("software_frame_index"),
                        "hardware_frame_id": reference.get("hardware_frame_id"),
                        "capture_monotonic": reference.get("capture_monotonic"),
                        "geometry_version": reference.get("geometry_version"),
                        "config_fingerprint": reference.get("config_fingerprint"),
                        "decoded_sha256": digest, "levels": levels}
            return {"resolved": True, "matched": None, "container_path": str(media),
                    "decoded_frame_index": index,
                    "software_frame_index": reference.get("software_frame_index"),
                    "hardware_frame_id": reference.get("hardware_frame_id"),
                    "capture_monotonic": reference.get("capture_monotonic"),
                    "geometry_version": reference.get("geometry_version"),
                    "config_fingerprint": reference.get("config_fingerprint"),
                    "levels": levels}
        finally:
            capture.release()

    still = reference.get("still_path")
    if still is not None:
        path = (root / still).resolve()
        if not path.exists():
            levels.append(_media_result("media_exists", False, f"静帧不存在：{path}"))
            return {"resolved": False, "matched": False, "reason": "still_missing",
                    "still_path": str(path), "levels": levels}
        digest = sha256_file(path)
        expected = reference.get("still_sha256")
        matched = expected is not None and expected == digest
        levels.append(_media_result("media_exists", True, f"静帧存在：{path}", bytes=path.stat().st_size))
        levels.append(_media_result("content_match", matched,
                                    "静帧哈希一致" if matched
                                    else ("未记录静帧哈希，无法比对" if expected is None
                                          else "静帧哈希不一致"),
                                    still_sha256=digest, expected_sha256=expected))
        return {"resolved": matched, "matched": matched, "still_path": str(path),
                "still_sha256": digest, "levels": levels}

    block = reference.get("raw_block")
    if block is not None:
        path = (root / str(block.get("path"))).resolve()
        if not path.exists():
            levels.append(_media_result("media_exists", False, f"原始块文件不存在：{path}"))
            return {"resolved": False, "matched": False, "reason": "raw_block_file_missing",
                    "raw_block_path": str(path), "levels": levels}
        offset = int(block.get("offset", 0))
        nbytes = int(block.get("nbytes", 0))
        if offset < 0 or nbytes <= 0 or offset + nbytes > path.stat().st_size:
            levels.append(_media_result("frame_in_range", False,
                                        f"offset/nbytes 越界：offset={offset} nbytes={nbytes} "
                                        f"size={path.stat().st_size}"))
            return {"resolved": False, "matched": False, "reason": "raw_block_out_of_range",
                    "levels": levels}
        with path.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(nbytes)
        digest = sha256_bytes(payload)
        expected = block.get("sha256")
        matched = expected is not None and expected == digest
        levels.append(_media_result("frame_in_range", True, "offset/nbytes 在文件范围内"))
        levels.append(_media_result("content_match", matched,
                                    "原始块哈希一致" if matched
                                    else ("未记录原始块哈希，无法比对" if expected is None
                                          else "原始块哈希不一致"),
                                    raw_block_sha256=digest, expected_sha256=expected))
        return {"resolved": True, "matched": matched, "raw_block_path": str(path),
                "raw_block_sha256": digest, "levels": levels}

    return {"resolved": False, "matched": False, "reason": "no_media_reference",
            "levels": levels}


def verify_frame_references(rows: list[dict], *, session_root: Path | str | None = None,
                            check_media: bool = False) -> dict:
    """逐行核验帧引用。

    区分三种情况，因为含义完全不同：

    * ``available``：有可用引用；
    * **显式缺失**：``available=false`` 且带 ``missing_reason``——诚实声明，不算失败；
    * **未声明缺口**：整行没有 ``source_frame_ref`` 字段——漏写，**判定失败**。

    结构错误一律判失败。**不把「软件序号 == 解码帧索引」当作真实性检验**：两者可以合法
    不同（抽帧、重编码、硬件帧号回绕）。``check_media=True`` 时额外核验媒体存在与索引
    范围（较慢，逐文件打开容器）。
    """
    total = 0
    available = 0
    declared_missing = 0
    undeclared_missing = 0
    structure_errors: list[dict] = []
    media_missing: list[dict] = []
    index_out_of_range: list[dict] = []
    for position, row in enumerate(rows):
        total += 1
        reference = row.get("source_frame_ref") if isinstance(row, dict) else None
        if not isinstance(reference, dict):
            undeclared_missing += 1
            continue
        if reference.get("available") is not True:
            if reference.get("missing_reason"):
                declared_missing += 1
            else:
                undeclared_missing += 1
            continue
        available += 1
        problems = validate_reference_structure(reference)
        if problems:
            structure_errors.append({"row": position, "problems": problems})
            continue
        if check_media:
            outcome = resolve_source_frame(reference, session_root=session_root)
            if outcome.get("reason") in {"container_missing", "still_missing",
                                        "raw_block_file_missing"}:
                media_missing.append({"row": position, "reason": outcome["reason"],
                                      "detail": outcome.get("levels")})
            elif outcome.get("reason") in {"frame_index_out_of_range",
                                           "raw_block_out_of_range"}:
                index_out_of_range.append({"row": position, "reason": outcome["reason"]})
    ok = (not structure_errors and undeclared_missing == 0
          and not media_missing and not index_out_of_range)
    return {"total_rows": total, "available_references": available,
            "declared_missing_references": declared_missing,
            "undeclared_missing_references": undeclared_missing,
            "structure_errors": structure_errors,
            "media_checked": bool(check_media),
            "media_missing": media_missing,
            "index_out_of_range": index_out_of_range,
            "ok": ok}


def verify_still(path: Path, expected_sha256: str | None = None) -> dict:
    """静帧核验：存在 + 内容哈希与记录值比对。缺记录值即无法判定匹配。"""
    path = Path(path)
    if not path.exists():
        return {"ok": False, "exists": False, "path": str(path), "reason": "still_missing"}
    digest = sha256_file(path)
    if expected_sha256 is None:
        return {"ok": False, "exists": True, "path": str(path), "sha256": digest,
                "matches": None, "reason": "expected_sha256_absent"}
    return {"ok": digest == expected_sha256, "exists": True, "path": str(path),
            "sha256": digest, "expected_sha256": expected_sha256,
            "matches": digest == expected_sha256}


def write_index(path: Path, rows: list[dict]) -> None:
    """一次性写盘（调用方负责决定何时 flush）。增量写见 :func:`open_index_writer`。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def _scalar(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _scalar(item) for key, item in value.items()}
    return repr(value)


def serialize_component(obj, *, reason: str) -> dict:
    """把一个生效配置对象的**全部**字段序列化成 ``{field: {value, reason}}``。

    只序列化手写白名单会让「改了实际使用的参数、配置指纹却不变」成为可能
    （复审反例：``DetectorConfig.generation_edge_mad_multiplier`` 3.0→13.0，
    配置指纹完全相同）。因此这里遍历 dataclass 的**所有**字段，嵌套 dataclass 递归展开。

    递归不会丢失叶子：``component_leaf_count`` 对该分区计数，回归里断言它接近
    实际字段数，而不是「只存了八个」。
    """
    import dataclasses

    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"{type(obj).__name__} 不是 dataclass 实例，无法完整序列化")
    out: dict = {}
    for item in dataclasses.fields(obj):
        value = getattr(obj, item.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            out[item.name] = serialize_component(value, reason=reason)
        else:
            out[item.name] = {"value": _scalar(value), "reason": reason}
    return out


def component_leaf_count(section: dict) -> int:
    """序列化分区里的叶子字段数（``{value, reason}`` 视为叶子）。"""
    total = 0
    for value in section.values():
        if isinstance(value, dict) and "value" not in value:
            total += component_leaf_count(value)
        else:
            total += 1
    return total


def unused_component(reason: str) -> dict:
    """该入口**未使用**这一组件。不以默认实例冒充实际生效对象。"""
    return {"used": declared(False, reason=reason)}


def wall_binding(*, source_label: str, wall_lines=None,
                 proposal_path: Path | None = None,
                 geometry_version: str | None = None) -> dict:
    """复用管壁的绑定：必须能复现扶正，只写来源名称不够。

    给出 ``wall_lines``（归一化坐标）即视为逐帧**当前帧**定位；复用提议时给出
    ``proposal_path``，其**内容指纹**一并记录——文件被改过就看得出来。

    **注意**：同时需要墙线与提议指纹时请用 :func:`wall_binding_from_file`，
    它把两者绑在**同一次读取**上，避免两次读取之间文件变化造成的自相矛盾。
    """
    payload: dict = {"source_label": declared(source_label, reason="管壁来源标签")}
    if wall_lines is not None:
        payload["wall_lines"] = declared(_scalar(list(wall_lines)),
                                        reason="实际使用的两条归一化管壁线")
        payload["wall_lines_sha256"] = declared(sha256_bytes(canonical_json(_scalar(list(wall_lines)))),
                                                reason="管壁线内容指纹")
        payload["binding_scope"] = declared("fixed_wall_lines",
                                            reason="固定墙线：可复现扶正")
    if proposal_path is not None:
        path = Path(proposal_path)
        exists = path.exists()
        payload["proposal_path"] = declared(str(path), reason="提议文件路径")
        payload["proposal_exists"] = declared(exists, reason="采集前核对")
        payload["proposal_content_sha256"] = (
            declared(sha256_file(path), reason="提议文件内容指纹")
            if exists else unknown("提议文件不存在，无法绑定其内容"))
    if geometry_version is not None:
        payload["geometry_version"] = declared(geometry_version, reason="几何版本")
    return payload


def wall_binding_from_file(proposal_path: Path, *, source_label: str,
                           geometry_version: str | None = None) -> tuple[dict, list]:
    """**一次**读取提议文件：同一份内容既解析墙线、又算内容指纹。

    为什么必须一次读取：若先解析墙线、再单独读一次算指纹，两次读取之间文件被改，
    就会得到「指纹属于 A、墙线属于 B」的自相矛盾记录。这里把两者绑在同一次读取上。

    返回 ``(绑定块, 墙线)``；墙线就是调用方应该交给 sink 的那一份。
    """
    path = Path(proposal_path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    walls = [dict(line) for line in (payload.get("walls") or [])]
    block = {
        "source_label": declared(source_label, reason="管壁来源标签"),
        "proposal_path": declared(str(path), reason="提议文件路径"),
        "proposal_content_sha256": declared(
            sha256_bytes(raw), reason="同一次读取的内容指纹（与解析出的墙线同源）"),
        "wall_lines": declared(_scalar(walls),
                               reason="从同一份内容解析出的两条归一化管壁线"),
        "wall_lines_sha256": declared(
            sha256_bytes(canonical_json(_scalar(walls))), reason="墙线内容指纹"),
        "binding_scope": declared("fixed_reused_proposal",
                                  reason="固定复用提议：墙线值可复现扶正"),
    }
    if geometry_version is not None:
        block["geometry_version"] = declared(geometry_version, reason="几何版本")
    return block, walls


class IncrementalIndexWriter:
    """逐行增量写盘：**已提交行**（已 flush 且计入 ``committed``）与待写数据分开。

    写盘中断后，已经提交的行可直接用于核对；进程内还没写出的行不会被伪造成「可读取」。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self.committed = 0

    def append(self, row: dict) -> None:
        self._stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._stream.flush()
        self.committed += 1

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "IncrementalIndexWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_committed_rows(path: Path) -> list[dict]:
    """读取已提交行；末行若是半截 JSON 则丢弃并报告，不把它当成有效记录。"""
    rows: list[dict] = []
    truncated = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            truncated = True
            break
    return rows if not truncated else rows + [{"_truncated_tail": True}]

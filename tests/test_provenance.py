"""运行追溯：源码快照、生效配置指纹、帧引用四级校验、离线核验。

覆盖任务书 §4/§6 要求与复审更正：未跟踪源码纳入清单、改源码可被发现、
帧号重复但采集身份不同、缺失图像/错误索引/篡改静帧、移动会话目录后仍能核验、
旧记录不被覆盖、导入与离线核验不连接硬件；以及新增的
「复制后修改源文件」「不存在媒体」「负索引」「配置指纹不匹配」。
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend import provenance as prov
import tools.plant_flow_transient_capture as tool_module

ROOT = Path(__file__).resolve().parents[1]
SESSION_ROOT = "sessions"


# ------------------------------------------------------------------ 源码快照

def test_untracked_tool_sources_and_the_live_chain_are_in_the_manifest() -> None:
    """未跟踪的台架入口、实时管线与编排链都必须在清单里——只对入口取哈希不够。"""
    for relative in ("tools/run_channel_validation_live.py",
                     "tools/replay_wall_localization.py",
                     "backend/vision/pipeline.py",
                     "backend/orchestrator/vision_adapter.py",
                     "backend/orchestrator/service.py",
                     "backend/vision/parallel_walls.py"):
        assert relative in prov.SOURCE_MANIFEST, relative
    assert "backend/provenance.py" in prov.SOURCE_MANIFEST


def test_snapshot_covers_vision_and_orchestration_but_states_its_boundary(tmp_path: Path) -> None:
    payload = prov.snapshot_sources(tmp_path, root=ROOT)
    paths = {item["path"] for item in payload["files"]}
    for expected in ("backend/vision/pipeline.py", "backend/orchestrator/vision_adapter.py",
                     "backend/orchestrator/service.py", "backend/orchestrator/state.py",
                     "backend/orchestrator/models.py"):
        assert expected in paths
    boundary = payload["scope_boundary"]
    assert "backend/pid_control/**" in boundary["excluded"]
    assert payload["consistent_snapshot"] is True
    assert payload["source_changed_after_copy"] == []


def test_run_time_versions_and_git_auxiliary_are_recorded(tmp_path: Path) -> None:
    payload = prov.snapshot_sources(tmp_path, root=ROOT)
    assert payload["runtime_versions"]["python"]
    assert "辅助" in payload["git"]["role"]


def test_modified_snapshot_copy_is_detected(tmp_path: Path) -> None:
    prov.snapshot_sources(tmp_path, root=ROOT)
    assert prov.verify_source_snapshot(tmp_path)["ok"] is True
    target = tmp_path / "sources" / "backend" / "vision" / "detector.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    result = prov.verify_source_snapshot(tmp_path)
    assert result["ok"] is False
    assert result["mismatched"] == ["backend/vision/detector.py"]
    assert result["fingerprint_ok"] is False


def test_missing_snapshot_copy_is_detected(tmp_path: Path) -> None:
    prov.snapshot_sources(tmp_path, root=ROOT)
    (tmp_path / "sources" / "backend" / "device_lock.py").unlink()
    result = prov.verify_source_snapshot(tmp_path)
    assert result["ok"] is False
    assert "backend/device_lock.py" in result["missing_snapshot_files"]


def test_source_changed_after_copy_is_detected(tmp_path: Path, monkeypatch) -> None:
    """复审反例：复制之后再改**源文件**。用打补丁的 copy2 精确构造这个时间窗。"""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "mod.py"
    source.write_text("original\n", encoding="utf-8")
    original_manifest = prov.SOURCE_MANIFEST
    real_copy2 = prov.shutil.copy2

    def copy_then_mutate(src, dst, *args, **kwargs):
        real_copy2(src, dst, *args, **kwargs)
        if Path(src).name == "mod.py":
            Path(src).write_text("changed right after copy\n", encoding="utf-8")

    try:
        prov.SOURCE_MANIFEST = ("mod.py",)
        monkeypatch.setattr(prov.shutil, "copy2", copy_then_mutate)
        payload = prov.snapshot_sources(workspace / "prov", root=workspace)
    finally:
        prov.SOURCE_MANIFEST = original_manifest
        monkeypatch.setattr(prov.shutil, "copy2", real_copy2)
    assert payload["source_changed_after_copy"] == ["mod.py"]
    assert payload["consistent_snapshot"] is False
    assert "不证明" in payload["consistency_limits"]


def test_build_bundle_refuses_an_inconsistent_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "mod.py").write_text("x\n", encoding="utf-8")
    original = prov.SOURCE_MANIFEST
    try:
        prov.SOURCE_MANIFEST = ("mod.py", "absent.py")
        with pytest.raises(prov.ProvenanceError):
            prov.build_bundle(workspace / "prov", effective={}, root=workspace)
    finally:
        prov.SOURCE_MANIFEST = original


def test_verification_writes_nothing(tmp_path: Path) -> None:
    prov.snapshot_sources(tmp_path, root=ROOT)
    manifest = tmp_path / "source_manifest.json"
    before_bytes, before_mtime = manifest.read_bytes(), manifest.stat().st_mtime_ns
    prov.verify_source_snapshot(tmp_path)
    assert manifest.read_bytes() == before_bytes
    assert manifest.stat().st_mtime_ns == before_mtime


def test_verification_survives_moving_the_directory(tmp_path: Path) -> None:
    original = tmp_path / "prov"
    prov.snapshot_sources(original, root=ROOT)
    moved = tmp_path / "moved" / "prov"
    moved.parent.mkdir()
    shutil.move(str(original), str(moved))
    assert prov.verify_source_snapshot(moved)["ok"] is True


# ------------------------------------------------------------------ 生效配置

def _effective(**overrides) -> dict:
    config = prov.EffectiveConfig(
        detector={"measurement_mode": prov.declared("generation_plug", reason="命令行")},
        strict_detection_localization=prov.declared(True, reason="实时源"),
        wall_source=prov.declared("current_frame_localized", reason="逐帧定位"),
        wall_binding=prov.wall_binding(source_label="current_frame_localized",
                                       wall_lines=[{"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.2}]),
        image_shape=prov.declared([720, 540], reason="实测"),
        scale_declaration=prov.unknown("未声明独立标尺"),
        depth_declaration=prov.unknown("芯片深度未声明"),
        imaging={"gain": prov.unknown("录像未记录增益")},
        sampling=prov.declared({"row_stride": 1}, reason="逐帧"),
        localization={"max_buffer_frames": prov.declared(24, reason="有界历史")},
        config_change_policy=prov.declared("forbidden_after_provenance_write",
                                           reason="运行中不改配置"),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config.to_dict()


def test_config_fingerprint_binds_the_measurement_rows(tmp_path: Path) -> None:
    effective = _effective()
    prov.snapshot_sources(tmp_path, root=ROOT)
    registered = prov.write_effective_config(tmp_path, effective)
    result = prov.verify_effective_config(tmp_path)
    assert result["ok"] is True
    assert result["config_fingerprint_ok"] is True
    assert result["recorded_config_fingerprint"] == registered["config_fingerprint"]
    assert result["recorded_config_fingerprint"] == prov.config_fingerprint(effective)


def test_config_tampering_is_detected(tmp_path: Path) -> None:
    effective = _effective()
    prov.snapshot_sources(tmp_path, root=ROOT)
    prov.write_effective_config(tmp_path, effective)
    tampered = _effective()
    tampered["detector"]["generation_min_length_ratio"] = {"value": 999, "reason": "tampered"}
    (tmp_path / "effective_config.json").write_text(
        json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    result = prov.verify_effective_config(tmp_path)
    assert result["ok"] is False
    assert result["config_fingerprint_ok"] is False


def test_config_missing_sections_is_rejected(tmp_path: Path) -> None:
    prov.write_effective_config(tmp_path, {"detector": {}})
    result = prov.verify_effective_config(tmp_path)
    assert result["ok"] is False
    assert "sampling" in result["missing_sections"]


def test_unknown_values_are_withheld_and_readback_requires_a_source() -> None:
    config = _effective()
    assert config["depth_declaration"]["value"] is None
    assert config["depth_declaration"]["reason"]
    assert "不等于实测值" in config["unknown_policy"]
    assert prov.readback(80.0, source="sdk_feature_getter")["readback_source"] == "sdk_feature_getter"
    with pytest.raises(ValueError):
        prov.readback(80.0, source="")


# ------------------------------------------------------------------ 帧引用

def _reference(**overrides):
    values = dict(session_id="s1", capture_id="c1", session_root=SESSION_ROOT,
                  software_frame_index=100, hardware_frame_id=5, capture_monotonic=1234.5,
                  container_path="raw_frames.mkv", decoded_frame_index=100)
    values.update(overrides)
    return prov.frame_reference(**values)


def test_structure_rejects_negative_indices_and_missing_root() -> None:
    problems = prov.validate_reference_structure(_reference(decoded_frame_index=-4))
    assert "decoded_frame_index_negative" in problems
    problems = prov.validate_reference_structure(_reference(hardware_frame_id=-1))
    assert "hardware_frame_id_negative" in problems
    # 手工构造缺 session_root 的引用：必须被结构校验指出。
    hand_built = _reference()
    hand_built["session_root"] = ""
    assert "session_root_absent" in prov.validate_reference_structure(hand_built)
    # 构造时缺 session_root 直接拒绝，不写成字符串 "None"。
    with pytest.raises(ValueError):
        _reference(session_root=None)


def test_hardware_frame_id_may_be_absent() -> None:
    """实机上确实取不到硬件帧号；缺失不是结构错误。"""
    assert prov.validate_reference_structure(_reference(hardware_frame_id=None)) == []


def test_nonexistent_media_is_not_resolved(tmp_path: Path) -> None:
    """复审反例：媒体不存在时不得返回 resolved=True。"""
    reference = _reference(container_path="does-not-exist.mkv")
    outcome = prov.resolve_source_frame(reference, session_root=tmp_path)
    assert outcome["resolved"] is False
    assert outcome["reason"] == "container_missing"
    assert [level["level"] for level in outcome["levels"]] == ["structure", "media_exists"]


def test_out_of_range_index_is_not_resolved(tmp_path: Path) -> None:
    container = tmp_path / "raw_frames.mkv"
    writer = cv2.VideoWriter(str(container), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (16, 8))
    for _ in range(5):
        writer.write(np.zeros((8, 16, 3), np.uint8))
    writer.release()
    outcome = prov.resolve_source_frame(_reference(decoded_frame_index=99), session_root=tmp_path)
    assert outcome["resolved"] is False
    assert outcome["reason"] == "frame_index_out_of_range"


def test_decoding_finds_the_original_frame_and_checks_content(tmp_path: Path) -> None:
    container = tmp_path / "raw_frames.mkv"
    writer = cv2.VideoWriter(str(container), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (16, 8))
    for value in range(5):
        writer.write(np.full((8, 16, 3), value * 10, np.uint8))
    writer.release()
    reference = _reference(decoded_frame_index=3)
    decoded = prov.resolve_source_frame(reference, session_root=tmp_path, decode=True)
    assert decoded["resolved"] is True
    assert decoded["matched"] is False, "未记录内容哈希时不能声称内容匹配"
    reference_with_hash = _reference(decoded_frame_index=3,
                                     content_sha256=decoded["decoded_sha256"])
    matched = prov.resolve_source_frame(reference_with_hash, session_root=tmp_path, decode=True)
    assert matched["resolved"] is True and matched["matched"] is True


def test_still_hash_is_compared_with_the_recorded_value(tmp_path: Path) -> None:
    still = tmp_path / "check.png"
    cv2.imwrite(str(still), np.zeros((8, 8, 3), np.uint8))
    digest = prov.sha256_file(still)
    assert prov.verify_still(still, digest)["matches"] is True
    assert prov.verify_still(still, "0" * 64)["matches"] is False
    # 缺记录值即无法判定匹配，不得算通过。
    assert prov.verify_still(still, None)["ok"] is False


def test_raw_block_reference_is_checked_against_the_file(tmp_path: Path) -> None:
    blob = tmp_path / "raw.bin"
    payload = b"\x01\x02\x03\x04" * 8
    blob.write_bytes(b"HEAD" + payload + b"TAIL")
    block = {"path": "raw.bin", "offset": 4, "nbytes": len(payload),
             "shape": [2, 16], "dtype": "uint8", "sha256": prov.sha256_bytes(payload)}
    reference = prov.frame_reference(session_id="s", capture_id="c", session_root=SESSION_ROOT,
                                     software_frame_index=1, hardware_frame_id=1,
                                     capture_monotonic=1.0, container_path=None,
                                     decoded_frame_index=None, raw_block=block)
    outcome = prov.resolve_source_frame(reference, session_root=tmp_path)
    assert outcome["resolved"] is True and outcome["matched"] is True
    block["sha256"] = "0" * 64
    reference = prov.frame_reference(session_id="s", capture_id="c", session_root=SESSION_ROOT,
                                     software_frame_index=1, hardware_frame_id=1,
                                     capture_monotonic=1.0, container_path=None,
                                     decoded_frame_index=None, raw_block=block)
    assert prov.resolve_source_frame(reference, session_root=tmp_path)["matched"] is False


def test_hardware_frame_id_alone_is_not_a_global_key() -> None:
    first = _reference(session_id="run-a", capture_id="a#1", hardware_frame_id=7,
                       container_path="output/run-a/raw_frames.mkv")
    second = _reference(session_id="run-b", capture_id="b#1", hardware_frame_id=7,
                        container_path="output/run-b/raw_frames.mkv")
    assert first["hardware_frame_id"] == second["hardware_frame_id"]
    assert first["container_path"] != second["container_path"]


def test_reference_verification_separates_declared_from_undeclared_gaps() -> None:
    rows = [
        {"source_frame_ref": _reference()},
        {"source_frame_ref": prov.frame_reference(
            session_id="s", capture_id="c", session_root=SESSION_ROOT,
            software_frame_index=None, hardware_frame_id=None, capture_monotonic=None,
            container_path=None, decoded_frame_index=None, missing_reason="未落盘引用")},
        {},
    ]
    result = prov.verify_frame_references(rows)
    assert result["available_references"] == 1
    assert result["declared_missing_references"] == 1
    assert result["undeclared_missing_references"] == 1
    assert result["ok"] is False


def test_declared_missing_alone_does_not_fail_verification() -> None:
    rows = [{"source_frame_ref": prov.frame_reference(
        session_id="s", capture_id="c", session_root=SESSION_ROOT,
        software_frame_index=None, hardware_frame_id=None, capture_monotonic=None,
        container_path=None, decoded_frame_index=None, missing_reason="未配置数据块索引")}]
    assert prov.verify_frame_references(rows)["ok"] is True


def test_software_and_hardware_identity_may_differ_legitimately() -> None:
    """软件序号与硬件帧号可以合法不同；不把二者相等当作真实性检验。"""
    reference = _reference(software_frame_index=6661, hardware_frame_id=6794,
                           decoded_frame_index=6661)
    result = prov.verify_frame_references([{"source_frame_ref": reference}])
    assert result["ok"] is True
    assert result["structure_errors"] == []


# ------------------------------------------------------------------ 增量写盘

def test_incremental_writer_commits_rows_and_reports_a_truncated_tail(tmp_path: Path) -> None:
    path = tmp_path / "rows.ndjson"
    with prov.IncrementalIndexWriter(path) as writer:
        for index in range(3):
            writer.append({"index": index})
        assert writer.committed == 3
    assert len(prov.read_committed_rows(path)) == 3

    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"index": 3, "ha')       # 模拟写盘中断
    rows = prov.read_committed_rows(path)
    assert rows[-1] == {"_truncated_tail": True}
    assert len(rows) == 4


# ------------------------------------------------------------------ 入口前置阻断

def _device_command_calls(pump_calls: list[str], camera_calls: list[str]) -> list[str]:
    """筛出**会驱动设备的命令**（清理时的 disconnect/close 不算）。"""
    work = {"connect_and_probe", "get_current_q_state", "write_wsp_and_verify",
            "start_infusion_and_verify", "update_flow_while_running",
            "stop_system_and_verify"}
    return ([f"pump.{name}" for name in pump_calls if name in work]
            + [f"camera.{name}" for name in camera_calls
               if name in {"discover_devices", "open", "start_stream"}]
            + [f"camera.{name}" for name in camera_calls if name.startswith("set_feature")])


def test_provenance_failure_aborts_before_any_device_command(tmp_path: Path) -> None:
    """追溯写入失败时必须在**任何设备命令之前**拒绝启动。"""
    from test_transient_capture_lifecycle import build, make_plan

    session, pump, camera, _lock, _sink, _order = build(make_plan(), tmp_path)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("这是一个文件，不是目录", encoding="utf-8")
    session.provenance_spec = {"directory": blocker, "effective": {}}

    report = session.run()

    assert _device_command_calls(pump.calls, camera.calls) == [], \
        f"追溯失败后仍发出设备命令：pump={pump.calls} camera={camera.calls}"
    assert pump.calls == ["disconnect"], f"只应有清理时的断开：{pump.calls}"
    assert report["touched_hardware"] is False
    assert report["provenance"]["written"] is False
    assert "追溯" in report["provenance"]["why"] or "ProvenanceError" in report["provenance"]["why"]
    assert report["result_layers"]["run_completed"] is False
    assert report["result_layers"]["stop"] == {"state": "NOT_ATTEMPTED", "verified": False}
    assert tool_module.exit_code(report) != tool_module.EXIT_OK


def test_successful_session_writes_provenance_and_marks_hardware_touched(tmp_path: Path) -> None:
    from test_transient_capture_lifecycle import build, make_plan

    session, _pump, _camera, _lock, _sink, _order = build(make_plan(), tmp_path)
    session.provenance_spec = {
        "directory": tmp_path / "prov",
        "effective": _effective(),
    }
    report = session.run()
    assert report["provenance"]["written"] is True
    assert report["provenance"]["config_fingerprint"]
    assert report["provenance"]["source_fingerprint"]
    assert report["touched_hardware"] is True
    assert prov.verify_source_snapshot(tmp_path / "prov")["ok"] is True
    assert prov.verify_effective_config(tmp_path / "prov")["ok"] is True


def test_session_without_a_provenance_spec_says_so(tmp_path: Path) -> None:
    from test_transient_capture_lifecycle import build, make_plan

    session, _pump, _camera, _lock, _sink, _order = build(make_plan(), tmp_path)
    report = session.run()
    assert report["provenance"]["written"] is False
    # 无设备测试接口必须**如实**标明：这不是现场入口可以走的路径。
    assert "无设备测试" in report["provenance"]["why"]

def test_provenance_module_never_imports_device_modules() -> None:
    tree = ast.parse((ROOT / "backend" / "provenance.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for name in imported:
        assert not any(name.startswith(item) for item in
                       ("backend.vision.cameras", "backend.pump_hardware", "serial",
                        "MvCameraControl")), name


def test_verification_cli_runs_offline(tmp_path: Path) -> None:
    prov.snapshot_sources(tmp_path, root=ROOT)
    prov.write_effective_config(tmp_path, _effective())
    rows = [{"source_frame_ref": _reference()}]
    prov.write_index(tmp_path / "rows.ndjson", rows)
    from tools.verify_provenance import verify_effective_config, verify_source_snapshot
    assert verify_source_snapshot(tmp_path)["ok"] is True
    assert verify_effective_config(tmp_path)["ok"] is True
    assert verify_effective_config(tmp_path)["withheld_unknown"] == [
        "depth_declaration", "imaging.gain", "scale_declaration"]


# ------------------------------------------------------------------ CLI 级别传播

def _synthetic_container(directory: Path, frames: int = 5) -> Path:
    container = directory / "raw_frames.mkv"
    writer = cv2.VideoWriter(str(container), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (16, 8))
    for value in range(frames):
        writer.write(np.full((8, 16, 3), value * 10, np.uint8))
    writer.release()
    return container


def _decoded_sha256(container: Path, index: int) -> str:
    capture = cv2.VideoCapture(str(container))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        assert ok
        return prov.sha256_bytes(frame.tobytes())
    finally:
        capture.release()


def _run_cli(rows: Path, *extra: str) -> dict:
    """按**子进程**调用核验 CLI——模拟自动验收只看退出码与总 ok。"""
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "verify_provenance.py"),
         "--rows", str(rows), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=600)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {}
    return {"returncode": completed.returncode, "ok": payload.get("ok"),
            "requested_level": payload.get("requested_level"),
            "levels_checked": payload.get("levels_checked"),
            "levels_passed": payload.get("levels_passed"),
            "resolved": (payload.get("resolved_source_frame") or {}).get("resolved"),
            "matched": (payload.get("resolved_source_frame") or {}).get("matched")}


@pytest.mark.parametrize("case,expected",
                         [("correct", True), ("wrong", False), ("absent", False)])
def test_cli_decode_level_requires_content_match(tmp_path: Path, case: str,
                                                expected: bool) -> None:
    """--decode 声明的级别包含内容比对：不匹配或缺预期哈希都必须让总结果失败。

    只测底层 ``resolve_source_frame`` 不够——曾经底层 matched=False、顶层仍报 ok=True。
    """
    container = _synthetic_container(tmp_path)
    digest = _decoded_sha256(container, 3)
    kwargs = {"software_frame_index": 3, "hardware_frame_id": 30,
              "capture_monotonic": 3.0, "container_path": "raw_frames.mkv",
              "decoded_frame_index": 3}
    if case == "correct":
        kwargs["content_sha256"] = digest
    elif case == "wrong":
        kwargs["content_sha256"] = "0" * 64
    reference = prov.frame_reference(session_id="s", capture_id="c",
                                     session_root=str(tmp_path), **kwargs)
    rows_path = tmp_path / "rows.ndjson"
    prov.write_index(rows_path, [{"source_frame_ref": reference}])

    outcome = _run_cli(rows_path, "--row", "0", "--decode")
    assert outcome["requested_level"] == "decode_and_content"
    assert outcome["resolved"] is True
    assert outcome["matched"] is expected
    assert (outcome["returncode"] == 0) is expected
    assert outcome["ok"] is expected
    # 字段名与语义要一致：失败的那一级出现在 checked 里、**不**出现在 passed 里。
    assert "content_match" in outcome["levels_checked"]
    if expected:
        assert outcome["levels_passed"] == outcome["levels_checked"]
    else:
        assert "content_match" not in outcome["levels_passed"]


def test_cli_decode_level_fails_when_the_frame_cannot_be_decoded(tmp_path: Path) -> None:
    container = _synthetic_container(tmp_path)
    reference = prov.frame_reference(session_id="s", capture_id="c",
                                     session_root=str(tmp_path),
                                     software_frame_index=99, hardware_frame_id=99,
                                     capture_monotonic=9.0,
                                     container_path="raw_frames.mkv",
                                     decoded_frame_index=99,
                                     content_sha256=_decoded_sha256(container, 0))
    rows_path = tmp_path / "rows.ndjson"
    prov.write_index(rows_path, [{"source_frame_ref": reference}])
    outcome = _run_cli(rows_path, "--row", "0", "--decode")
    assert outcome["resolved"] is False
    assert outcome["returncode"] != 0
    assert outcome["ok"] is False


def test_cli_without_decode_only_claims_the_structure_level(tmp_path: Path) -> None:
    """默认级别只声明结构校验；不得因此在报告里暗示内容已比对。"""
    reference = prov.frame_reference(session_id="s", capture_id="c",
                                     session_root=str(tmp_path),
                                     software_frame_index=1, hardware_frame_id=1,
                                     capture_monotonic=1.0,
                                     container_path="raw_frames.mkv",
                                     decoded_frame_index=1)
    rows_path = tmp_path / "rows.ndjson"
    prov.write_index(rows_path, [{"source_frame_ref": reference}])
    outcome = _run_cli(rows_path)
    assert outcome["requested_level"] == "structure"
    assert outcome["ok"] is True


# ------------------------------------------------- 生效配置指纹的敏感性

def _config_for(detector, *, thresholds=None) -> dict:
    from backend.vision.config import DetectorConfig
    from backend.vision.parallel_walls import LocalizationThresholds
    import tools.plant_flow_transient_capture as tool_module

    return tool_module.live_effective_config(
        plan={"capture_plan": {}, "apparatus": {}},
        detector_config=detector or DetectorConfig(),
        localization_thresholds=thresholds or LocalizationThresholds())


@pytest.mark.parametrize("field,value", [
    ("generation_edge_mad_multiplier", 13.0),
    ("generation_min_raw_outline_contrast", 3.0),
    ("generation_outline_gap_min_ratio", 0.9),
    ("generation_min_profile_contrast_sigma", 0.9),
    ("generation_max_length_ratio", 30.0),
    ("min_radius", 21.0),
])
def test_changing_any_detector_parameter_changes_the_config_fingerprint(field: str,
                                                                       value: float) -> None:
    """检测器**实际使用**的参数改了，配置指纹必须跟着变。

    复审反例：``generation_edge_mad_multiplier`` 3.0→13.0 指纹完全相同——
    因为当时只序列化手写白名单里的 8 个字段。
    """
    from backend.vision.config import DetectorConfig

    base = DetectorConfig()
    changed = DetectorConfig()
    assert getattr(base, field) != value
    setattr(changed, field, value)
    base_cfg = _config_for(base)
    changed_cfg = _config_for(changed)
    assert prov.config_fingerprint(base_cfg) != prov.config_fingerprint(changed_cfg), field
    assert changed_cfg["detector"][field]["value"] == value


def test_changing_a_localization_threshold_changes_the_config_fingerprint() -> None:
    import dataclasses

    from backend.vision.parallel_walls import LocalizationThresholds

    base = LocalizationThresholds()
    changed = dataclasses.replace(base, motion_min_axial_shift_px=7.5)
    assert (prov.config_fingerprint(_config_for(None, thresholds=base))
            != prov.config_fingerprint(_config_for(None, thresholds=changed)))


def test_component_serialization_covers_all_fields_not_a_whitelist() -> None:
    import dataclasses

    from backend.vision.config import DetectorConfig

    config = _config_for(DetectorConfig())
    leaves = prov.component_leaf_count(config["detector"])
    fields = len(dataclasses.fields(DetectorConfig()))
    assert leaves >= fields, f"序列化叶子 {leaves} 少于字段数 {fields}"
    assert leaves > 50, "只序列化了少数几个字段，等于白名单"


def test_unused_component_is_marked_not_faked_with_a_default_instance() -> None:
    import tools.plant_flow_transient_capture as tool_module

    config = tool_module.live_effective_config(plan={"capture_plan": {}, "apparatus": {}},
                                               detector_config=None,
                                               localization_thresholds=None)
    assert config["detector"]["value"] is None          # 不接受用默认值冒充
    assert config["localization"]["used"]["value"] is False
    assert "不以默认阈值实例冒充" in config["localization"]["used"]["reason"]


def test_wall_binding_binds_content_not_only_a_label(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.json"
    proposal.write_text('{"walls": []}', encoding="utf-8")
    block = prov.wall_binding(source_label="reused_proposal", proposal_path=proposal)
    assert block["proposal_exists"]["value"] is True
    first = block["proposal_content_sha256"]["value"]
    proposal.write_text('{"walls": [1]}', encoding="utf-8")
    second = prov.wall_binding(source_label="reused_proposal",
                               proposal_path=proposal)["proposal_content_sha256"]["value"]
    assert first != second, "提议文件被改过，绑定必须看得出来"

    lines = [{"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.25}]
    with_lines = prov.wall_binding(source_label="current_frame_localized", wall_lines=lines)
    assert with_lines["wall_lines"]["value"] == lines
    assert with_lines["wall_lines_sha256"]["value"]


# ------------------------------------------------- 现场入口不得绕过追溯

def test_every_effective_config_builder_uses_full_serialization() -> None:
    """任何组装生效配置的地方都必须用 ``serialize_component``，不得手写字段白名单。

    这条守卫来自一次真实回归：``live_effective_config`` 已改为完整序列化，
    但回放工具自己的生效配置仍留着 6 个手写检测器字段——同一个毛病换个地方复现。
    """
    tools = ROOT / "tools"
    for name in ("plant_flow_transient_capture.py", "replay_wall_localization.py"):
        source = (tools / name).read_text(encoding="utf-8")
        assert "serialize_component(" in source, f"{name} 未做完整序列化"
        for field in ("measurement_mode", "generation_min_length_ratio",
                      "generation_max_length_ratio", "generation_edge_mad_multiplier",
                      "generation_outline_gap_min_ratio"):
            assert f'"{field}": declared(' not in source, \
                f"{name} 仍手写检测器字段 {field}"


# ------------------------------------- 固定复用墙线的绑定（走入口配置组装路径）

def _proposal_file(directory: Path, name: str, walls: list[dict]) -> Path:
    path = directory / name
    path.write_text(json.dumps({"walls": walls}, ensure_ascii=False), encoding="utf-8")
    return path


def _diagnostic_config(proposal_path: Path):
    """走**诊断入口的配置组装路径**：该入口的 load_walls_and_binding + live_effective_config。"""
    from backend.vision.config import DetectorConfig
    from tools.run_channel_validation_live import load_walls_and_binding
    import tools.plant_flow_transient_capture as tool_module

    walls, block = load_walls_and_binding(proposal_path, source_label="reused_proposal")
    config = tool_module.live_effective_config(
        plan={"capture_plan": {}, "apparatus": {}},
        detector_config=DetectorConfig(),
        strict_localization=prov.declared(False, reason="诊断入口不启用严格定位"),
        wall_source="reused_proposal",
        wall_binding_block=block)
    return walls, block, config


def test_fixed_proposal_requires_a_wall_binding() -> None:
    """复用固定提议却不给绑定 → 直接拒绝，不得组装出只有来源标签的生效配置。"""
    import tools.plant_flow_transient_capture as tool_module

    with pytest.raises(ValueError, match="复用固定提议必须绑定"):
        tool_module.live_effective_config(
            plan={"capture_plan": {}, "apparatus": {}},
            detector_config=None, wall_source="reused_proposal",
            wall_binding_block={"source_label": prov.declared("x", reason="只有标签")})


def test_entry_assembly_records_the_walls_the_sink_uses(tmp_path: Path) -> None:
    """记录里的墙线必须**就是**交给 sink 的那一份，且内容指纹来自同一次读取。"""
    walls = [{"x1": 0.15, "y1": 0.44, "x2": 0.97, "y2": 0.52},
             {"x1": 0.15, "y1": 0.50, "x2": 0.97, "y2": 0.59}]
    proposal = _proposal_file(tmp_path, "p1.json", walls)

    sink_walls, block, config = _diagnostic_config(proposal)

    assert sink_walls == walls
    assert block["wall_lines"]["value"] == walls
    assert block["wall_lines_sha256"]["value"] == prov.sha256_bytes(
        prov.canonical_json(walls))
    assert block["proposal_content_sha256"]["value"] == prov.sha256_file(proposal)
    assert block["binding_scope"]["value"] == "fixed_reused_proposal"
    assert config["wall_binding"]["wall_lines"]["value"] == sink_walls
    assert config["wall_source"]["value"] == "reused_proposal"


def test_proposal_is_read_exactly_once(tmp_path: Path, monkeypatch) -> None:
    """墙线与内容指纹必须来自**同一次**读取：读两次期间文件变化会造成自相矛盾。"""
    walls = [{"x1": 0.1, "y1": 0.4, "x2": 0.9, "y2": 0.5},
             {"x1": 0.1, "y1": 0.6, "x2": 0.9, "y2": 0.7}]
    proposal = _proposal_file(tmp_path, "p1.json", walls)
    reads = {"count": 0}
    original = Path.read_bytes

    def counting_read_bytes(self):
        if self == proposal:
            reads["count"] += 1
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    _diagnostic_config(proposal)
    assert reads["count"] == 1, f"提议文件被读了 {reads['count']} 次"


def test_two_different_wall_sets_give_different_config_fingerprints(tmp_path: Path) -> None:
    """换了固定墙线，配置指纹必须变——这正是复审指出的缺口。"""
    first = [{"x1": 0.15, "y1": 0.44, "x2": 0.97, "y2": 0.52},
             {"x1": 0.15, "y1": 0.50, "x2": 0.97, "y2": 0.59}]
    second = [{"x1": 0.15, "y1": 0.42, "x2": 0.97, "y2": 0.50},
              {"x1": 0.15, "y1": 0.51, "x2": 0.97, "y2": 0.60}]
    _, _, config_a = _diagnostic_config(_proposal_file(tmp_path, "a.json", first))
    _, _, config_b = _diagnostic_config(_proposal_file(tmp_path, "b.json", second))
    assert prov.config_fingerprint(config_a) != prov.config_fingerprint(config_b)


def test_proposal_with_a_wrong_number_of_walls_is_refused(tmp_path: Path) -> None:
    from tools.run_channel_validation_live import load_walls_and_binding

    proposal = _proposal_file(tmp_path, "bad.json",
                              [{"x1": 0.1, "y1": 0.4, "x2": 0.9, "y2": 0.5}])
    with pytest.raises(ValueError, match="恰好两条墙线"):
        load_walls_and_binding(proposal, source_label="reused_proposal")


def test_live_entrypoints_expose_no_provenance_bypass() -> None:
    """真机入口不得提供关闭追溯的开关，也不得调用无设备测试接口。

    该接口只允许在编排模块里**被定义**（供 tests/ 使用）。用 AST 检查**调用节点**，
    不受注释与文档字符串影响。
    """
    tools = ROOT / "tools"
    for name in ("run_r1_front4_live.py", "run_channel_validation_live.py"):
        source = (tools / name).read_text(encoding="utf-8")
        assert "--no-provenance" not in source, name
        assert "allow_unprovenanced" not in source, name
        assert not _calls_named(source, "for_device_free_tests"), name

    orchestrator = (tools / "plant_flow_transient_capture.py").read_text(encoding="utf-8")
    assert "--no-provenance" not in orchestrator
    # 只允许 `def for_device_free_tests(...)` 这一处定义，不得有任何调用点。
    assert not _calls_named(orchestrator, "for_device_free_tests")


def _calls_named(source: str, name: str) -> list[int]:
    """返回调用了 ``name(...)`` 的行号；``def name(...)`` 不算调用。"""
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        called = (target.attr if isinstance(target, ast.Attribute)
                  else target.id if isinstance(target, ast.Name) else None)
        if called == name:
            lines.append(node.lineno)
    return lines


def test_live_session_without_a_provenance_spec_refuses_before_device_actions(
        tmp_path: Path) -> None:
    """真机构造路径（无测试接口）：缺追溯规格必须在任何设备命令之前拒绝。"""
    from test_transient_capture_lifecycle import build, make_plan

    session, pump, camera, _lock, _sink, _order = build(make_plan(), tmp_path,
                                                       device_free=False)
    assert session.allow_unprovenanced is False
    report = session.run()

    # 与既有约定一致：清理时的断开不算「设备命令」；不得出现连接/启动/流量指令。
    work = {"connect_and_probe", "get_current_q_state", "write_wsp_and_verify",
            "start_infusion_and_verify", "update_flow_while_running"}
    assert not (set(pump.calls) & work), f"缺追溯规格却发出了泵命令：{pump.calls}"
    assert pump.calls == ["disconnect"], f"只应有清理时的断开：{pump.calls}"
    camera_work = {"discover_devices", "open", "start_stream"}
    assert not (set(camera.calls) & camera_work), f"缺追溯规格却开了相机：{camera.calls}"
    assert not [name for name in camera.calls if name.startswith("set_feature")], camera.calls
    assert camera.calls == ["stop_stream", "close"], f"只应有清理时的关闭：{camera.calls}"
    assert report["touched_hardware"] is False
    assert report["provenance"]["written"] is False
    assert "缺追溯规格" in report["provenance"]["why"]
    assert report["result_layers"]["task_goal_met"] is False
    assert tool_module.exit_code(report) != tool_module.EXIT_OK


def test_device_free_interface_is_explicitly_marked_in_the_report(tmp_path: Path) -> None:
    from test_transient_capture_lifecycle import build, make_plan

    session, _pump, _camera, _lock, _sink, _order = build(make_plan(), tmp_path)
    report = session.run()
    assert report["provenance"]["written"] is False
    assert "无设备测试" in report["provenance"]["why"]

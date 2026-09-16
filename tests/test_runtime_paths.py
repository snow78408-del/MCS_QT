from __future__ import annotations

from pathlib import Path

from backend import runtime_paths


def test_windows_runtime_data_defaults_to_d_drive(monkeypatch):
    monkeypatch.delenv("MCS_DATA_DIR", raising=False)
    monkeypatch.setattr(runtime_paths.sys, "platform", "win32")
    assert runtime_paths.user_data_dir() == Path("D:/MCS_QT_Data")


def test_explicit_runtime_override_and_subdirectory(monkeypatch, tmp_path):
    monkeypatch.setenv("MCS_DATA_DIR", str(tmp_path))
    assert runtime_paths.user_data_dir() == tmp_path.resolve()
    assert runtime_paths.ensure_user_subdir("calibrations") == tmp_path / "calibrations"
    assert (tmp_path / "calibrations").is_dir()

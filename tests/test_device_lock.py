"""设备独占锁的回归测试。

对应「按实际持有设备连接的生命周期管理锁」这一设计决定。锁的全部意义在于**跨进程**互斥，
所以关键性质都用真实子进程验证，而不是在同一个进程里自证；进程内唯一性、锁键规范化与
锁目录统一则各自单独钉住。

另外验证「持有者死亡后由内核自动释放」——这是选用 OS 建议锁而非 PID 文件判活的理由，
必须有测试兜住，否则一旦退化成 PID 文件就会留下永久陈旧锁。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from backend.device_lock import (
    DeviceLock,
    DeviceLockError,
    acquire_device_locks,
    camera_lock_key,
    normalize_port_key,
    release_device_locks,
)
from backend.runtime_paths import device_lock_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

_HOLDER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, sys.argv[1])
    from backend.device_lock import DeviceLock
    lock = DeviceLock(sys.argv[4], path=sys.argv[2])
    lock.acquire()
    print("HELD", flush=True)
    time.sleep(float(sys.argv[3]))
    """
)


def _start_holder(lock_path: Path, hold_s: float = 30.0) -> subprocess.Popen:
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(REPO_ROOT), str(lock_path), str(hold_s),
         f"holder-{uuid.uuid4().hex[:8]}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert holder.stdout is not None
    line = holder.stdout.readline().strip()
    assert line == "HELD", f"holder 未就绪：{line!r} {holder.stderr.read() if holder.stderr else ''}"
    return holder


@pytest.fixture()
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / f"hardware-{uuid.uuid4().hex[:8]}.lock"


def _lock(lock_path: Path, key: str = "test-device") -> DeviceLock:
    return DeviceLock(key, path=lock_path)


# ---------------------------------------------------------------- 跨进程

def test_lock_is_exclusive_across_processes(lock_path: Path) -> None:
    """另一个进程持有时必须拒绝，不能用「脚本重复启动」之类的弱检查代替。"""
    holder = _start_holder(lock_path)
    try:
        with pytest.raises(DeviceLockError):
            _lock(lock_path).acquire()
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_lock_can_be_taken_again_after_the_holder_is_killed(lock_path: Path) -> None:
    """持有者被强杀后内核必须释放锁，不能留下需要人工清理的陈旧锁。"""
    holder = _start_holder(lock_path)
    holder.terminate()
    holder.wait(timeout=30)

    lock = _lock(lock_path)
    lock.acquire()
    try:
        assert lock.held is True
        assert lock_path.exists()
    finally:
        lock.release()
    assert lock.held is False


# ---------------------------------------------------------------- 进程内唯一

def test_same_process_cannot_hold_the_same_device_twice(lock_path: Path) -> None:
    """「同进程」不等于可以随意共享设备。"""
    first = _lock(lock_path, key="same-key")
    first.acquire()
    try:
        with pytest.raises(DeviceLockError) as excinfo:
            _lock(lock_path, key="same-key").acquire()
        assert "本进程已持有" in str(excinfo.value)
    finally:
        first.release()


def test_different_keys_on_the_same_path_still_exclude(lock_path: Path) -> None:
    """互斥的对象是资源（同一把锁文件），不是键名。"""
    first = _lock(lock_path, key="pump-com3")
    first.acquire()
    try:
        with pytest.raises(DeviceLockError):
            _lock(lock_path, key="pump-com4-with-a-different-name").acquire()
    finally:
        first.release()


def test_lock_is_reusable_within_one_process(lock_path: Path) -> None:
    lock = _lock(lock_path)
    with lock:
        assert lock.held is True
    assert lock.held is False
    with lock:
        assert lock.held is True


def test_double_acquire_on_the_same_object_is_refused(lock_path: Path) -> None:
    lock = _lock(lock_path)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError):
            lock.acquire()
    finally:
        lock.release()


def test_release_without_acquire_is_a_no_op(lock_path: Path) -> None:
    _lock(lock_path).release()


def test_lock_survives_a_dropped_reference(lock_path: Path) -> None:
    """``DeviceLock(key).acquire()`` 不保留对象也必须继续持锁。

    否则 CPython 的引用计数会立刻回收对象、关闭文件句柄，操作系统随之释放锁——调用方以为
    自己持有设备，实际互斥早已丢失。这条曾经真的踩到：holder 脚本写成
    ``DeviceLock(...).acquire()``，子进程打印了「已持有」却什么都没锁住。

    注：本测试故意让这把锁保持到进程结束（路径是各自独立的临时文件，不影响其他测试）。
    """
    DeviceLock("dropped-ref", path=lock_path).acquire()

    with pytest.raises(DeviceLockError):
        DeviceLock("dropped-ref", path=lock_path).acquire()


# ---------------------------------------------------------------- 锁键

@pytest.mark.parametrize("raw", ["COM3", "com3", " COM3 ", "\\\\.\\COM3", r"\\.\com3"])
def test_port_key_normalizes_every_spelling_of_the_same_port(raw: str) -> None:
    assert normalize_port_key(raw) == "pump-com3"


@pytest.mark.parametrize("raw,expected", [("COM10", "pump-com10"), ("COM4", "pump-com4")])
def test_port_key_keeps_distinct_ports_distinct(raw: str, expected: str) -> None:
    assert normalize_port_key(raw) == expected
    assert normalize_port_key("COM4") != normalize_port_key("COM10")


def test_port_key_ignores_the_bus_address() -> None:
    """锁键只由串口标识决定——同一串口上的不同泵地址仍是同一台物理设备。"""
    import inspect

    assert list(inspect.signature(normalize_port_key).parameters) == ["port"], \
        "锁键函数不得接受地址参数，否则同一串口的两个地址会得到两把锁"
    assert normalize_port_key("COM3") == "pump-com3"


def test_empty_or_unknown_port_still_gets_a_stable_key() -> None:
    assert normalize_port_key(None) == "pump-unknown"
    assert normalize_port_key("") == "pump-unknown"
    assert normalize_port_key(" /dev/ttyUSB0 ") == normalize_port_key("/dev/ttyUSB0")


def test_camera_key_is_stable_and_case_insensitive() -> None:
    assert camera_lock_key("Hikrobot:Direct:0") == camera_lock_key("hikrobot:direct:0")
    assert camera_lock_key("SN-12345") == "camera-sn-12345"
    assert camera_lock_key(None) == "camera-unknown"


# ---------------------------------------------------------------- 锁目录统一

def test_lock_directory_ignores_the_data_dir_override(monkeypatch) -> None:
    """不同 MCS_DATA_DIR 必须映射到同一把锁，否则互斥形同虚设。"""
    monkeypatch.setenv("MCS_DATA_DIR", "D:/one/data")
    first = device_lock_dir()
    monkeypatch.setenv("MCS_DATA_DIR", "E:/another/place")
    second = device_lock_dir()
    assert first == second


def test_default_lock_path_uses_the_shared_lock_directory() -> None:
    lock = DeviceLock("pump-com7")
    assert lock.path.parent == device_lock_dir()
    assert lock.path.name == "pump-com7.lock"


# ---------------------------------------------------------------- 多设备顺序与回滚

def test_multi_device_acquisition_is_ordered_and_rolls_back(tmp_path: Path,
                                                           monkeypatch) -> None:
    """按固定顺序获取；后续失败时释放本次已取得的，不留半持有状态。

    这里挡住排序在后的 ``pump-com2``：``pump-com1`` 会先成功，随后失败必须把 com1 释放掉。
    """
    monkeypatch.setattr("backend.device_lock.device_lock_dir", lambda: tmp_path)
    blocked = _lock(tmp_path / "pump-com2.lock", key="pump-com2")
    blocked.acquire()
    try:
        with pytest.raises(DeviceLockError):
            acquire_device_locks(["pump-com1", "pump-com2"])
    finally:
        blocked.release()

    # 回滚必须已释放 pump-com1：若没释放，这里会在进程内登记表或 OS 锁上失败
    probe = _lock(tmp_path / "pump-com1.lock", key="pump-com1")
    probe.acquire()
    try:
        assert probe.held is True
    finally:
        probe.release()


def test_multi_device_acquisition_order_is_internal(tmp_path: Path, monkeypatch) -> None:
    """顺序由实现内部排序决定，不信任调用方给的顺序。"""
    monkeypatch.setattr("backend.device_lock.device_lock_dir", lambda: tmp_path)
    forward = acquire_device_locks(["pump-com3", "pump-com1", "pump-com2"])
    try:
        assert [lock.key for lock in forward] == ["pump-com1", "pump-com2", "pump-com3"]
    finally:
        release_device_locks(forward)


def test_release_device_locks_reports_release_failures() -> None:
    class Exploding:
        key = "boom"

        def release(self) -> None:
            raise OSError("release failed")

    errors = release_device_locks([Exploding()])
    assert len(errors) == 1
    assert isinstance(errors[0], OSError)

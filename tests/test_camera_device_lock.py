"""相机设备锁接线测试：覆盖 ``CameraManager`` 的三处 open。

复查意见要求「覆盖主程序、设备测试、采集脚本和重连路径」——相机侧有三处会真正打开设备：
``open_selected()``（主运行路径）、``test_device()``（设备测试，自有 open）、
``_reconnect()``（自动重开，绕过 ``open_selected``）。这里逐条钉住，并用跨进程竞争验证
拿不到锁时**适配器的 open 根本没被调用**。

全部使用假适配器，不连接真实相机。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.device_lock import DeviceLock, DeviceLockError, camera_lock_key
from backend.vision.cameras.manager import CameraManager
from backend.vision.cameras.models import CameraCapabilities, CameraDeviceInfo

REPO_ROOT = Path(__file__).resolve().parents[1]

_SCRIPT_HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path

    sys.path.insert(0, sys.argv[1])
    import backend.device_lock as device_lock
    device_lock.device_lock_dir = lambda: Path(sys.argv[2])
    from backend.device_lock import DeviceLock

    lock = DeviceLock(sys.argv[3])
    lock.acquire()
    print("HELD", flush=True)
    time.sleep(float(sys.argv[4]))
    """
)


class FakeAdapter:
    def __init__(self, *, open_error: BaseException | None = None) -> None:
        self.calls: list[str] = []
        self.open_error = open_error

    def open(self, device) -> None:
        self.calls.append("open")
        if self.open_error is not None:
            raise self.open_error

    def close(self) -> None:
        self.calls.append("close")

    def stop_stream(self) -> None:
        self.calls.append("stop_stream")

    def start_stream(self) -> None:
        self.calls.append("start_stream")

    def get_capabilities(self) -> CameraCapabilities:
        return CameraCapabilities()

    def read_frame(self, timeout_ms: int):
        raise AssertionError("refusal tests must not reach read_frame")


@pytest.fixture()
def lock_dir(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "locks"
    directory.mkdir()
    monkeypatch.setattr("backend.device_lock.device_lock_dir", lambda: directory)
    return directory


def make_device(unique_id: str = "HIKROBOT:SN-0001") -> CameraDeviceInfo:
    return CameraDeviceInfo(device_id=f"hikrobot:{unique_id}", unique_id=unique_id,
                            backend_name="hikrobot", manufacturer="HIKROBOT",
                            model="MV-CU", serial_number="SN-0001",
                            selected_backend="hikrobot")


def make_manager(device: CameraDeviceInfo, adapter: FakeAdapter) -> CameraManager:
    manager = CameraManager()
    manager._selected_device = device
    manager._build_adapter = lambda _device: adapter      # 注入假适配器
    return manager


def _start_script_holder(lock_dir: Path, key: str) -> subprocess.Popen:
    holder = subprocess.Popen(
        [sys.executable, "-c", _SCRIPT_HOLDER, str(REPO_ROOT), str(lock_dir), key, "30"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert holder.stdout is not None
    line = holder.stdout.readline().strip()
    assert line == "HELD", f"holder 未就绪：{line!r} {holder.stderr.read() if holder.stderr else ''}"
    return holder


# ---------------------------------------------------------------- 主运行路径

def test_open_selected_takes_the_lock_and_close_releases_it(lock_dir) -> None:
    device = make_device()
    adapter = FakeAdapter()
    manager = make_manager(device, adapter)

    manager.open_selected()
    try:
        assert adapter.calls == ["open"]
        assert manager._device_lock is not None
        assert manager._device_lock_key == camera_lock_key(device.unique_id)
        assert manager._device_lock.path.parent == lock_dir
    finally:
        manager.close_selected()

    assert manager._device_lock is None
    assert adapter.calls[-1] == "close"
    with DeviceLock(camera_lock_key(device.unique_id)):   # 关闭后互斥释放
        pass


def test_open_failure_closes_the_adapter_and_releases_the_lock(lock_dir) -> None:
    device = make_device()
    adapter = FakeAdapter(open_error=RuntimeError("设备被厂商软件占用"))
    manager = make_manager(device, adapter)

    with pytest.raises(RuntimeError):
        manager.open_selected()

    assert "close" in adapter.calls, "打开失败必须关闭半开的适配器"
    assert manager._device_lock is None, "打开失败必须释放设备锁"
    with DeviceLock(camera_lock_key(device.unique_id)):
        pass


def test_switching_devices_swaps_the_lock(lock_dir) -> None:
    first, second = make_device("HIKROBOT:SN-0001"), make_device("HIKROBOT:SN-0002")
    manager = make_manager(first, FakeAdapter())
    manager.open_selected()
    try:
        first_key = camera_lock_key(first.unique_id)
        manager._selected_device = second
        manager._build_adapter = lambda _device: FakeAdapter()
        manager.open_selected()
        try:
            assert manager._device_lock_key == camera_lock_key(second.unique_id)
            # 旧设备的锁已释放，新设备的锁被持有
            with DeviceLock(first_key):
                pass
            with pytest.raises(DeviceLockError):
                DeviceLock(camera_lock_key(second.unique_id)).acquire()
        finally:
            manager.close_selected()
    finally:
        manager.close_selected()


def test_reacquiring_the_same_device_is_idempotent(lock_dir) -> None:
    """同一设备重复获取是借用，不新建也不释放——设备测试依赖这一点。"""
    device = make_device()
    manager = make_manager(device, FakeAdapter())
    manager._acquire_device_lock(device)
    held = manager._device_lock
    try:
        manager._acquire_device_lock(device)
        assert manager._device_lock is held
    finally:
        manager._release_device_lock()


# ---------------------------------------------------------------- 跨进程竞争

def test_open_selected_is_refused_while_the_script_holds_the_camera(lock_dir) -> None:
    device = make_device()
    holder = _start_script_holder(lock_dir, camera_lock_key(device.unique_id))
    try:
        adapter = FakeAdapter()
        manager = make_manager(device, adapter)

        with pytest.raises(DeviceLockError):
            manager.open_selected()

        assert adapter.calls == [], "拿不到设备锁时绝不能打开相机"
        assert manager._device_lock is None
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_test_device_is_refused_while_the_script_holds_the_camera(lock_dir) -> None:
    """设备测试路径也必须被挡住，并且不会留下半持有的锁。"""
    device = make_device()
    holder = _start_script_holder(lock_dir, camera_lock_key(device.unique_id))
    try:
        adapter = FakeAdapter()
        manager = make_manager(device, adapter)

        result = manager.test_device()

        assert result.ok is False
        assert "设备锁" in str(result.error) or "占用" in str(result.error)
        assert "open" not in adapter.calls, "拿不到设备锁时绝不能打开相机"
        assert manager._device_lock is None
    finally:
        holder.terminate()
        holder.wait(timeout=30)


# ---------------------------------------------------------------- 重连路径

def test_reconnect_giving_up_releases_the_lock(lock_dir) -> None:
    """重连彻底失败后不再持有可用连接，锁必须释放，否则设备被无限期占住。"""
    device = make_device()
    manager = make_manager(device, FakeAdapter())
    manager._acquire_device_lock(device)
    assert manager._device_lock is not None

    manager._stop_event.set()          # 让重连循环立即放弃
    assert manager._reconnect(FakeAdapter()) is None

    assert manager._device_lock is None
    with DeviceLock(camera_lock_key(device.unique_id)):
        pass


def test_reconnect_success_keeps_the_lock(lock_dir) -> None:
    """重连成功时设备仍是我们的：锁必须保持，不能放开让别的进程插进来。"""
    device = make_device()
    manager = make_manager(device, FakeAdapter())
    manager._acquire_device_lock(device)
    held = manager._device_lock
    manager.config = SimpleNamespace(reconnect_max_attempts=1, reconnect_initial_delay_s=0.01,
                                     reconnect_max_delay_s=0.01)

    failed = FakeAdapter()
    replacement = FakeAdapter()
    manager._build_adapter = lambda _device: replacement
    result = manager._reconnect(failed)

    assert result is replacement
    assert manager._device_lock is held, "重连成功后不得换锁或放开锁"
    assert manager._device_lock_key == camera_lock_key(device.unique_id)
    assert "close" in failed.calls, "旧适配器应被关闭"
    with pytest.raises(DeviceLockError):        # 锁仍被持有
        DeviceLock(camera_lock_key(device.unique_id)).acquire()
    manager._release_device_lock()

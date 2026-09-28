"""设备锁接线的竞争测试：走**真实的主程序连接入口**，不是两个 DeviceLock 自证。

复查意见明确指出：「仅测试两个 ``DeviceLock`` 对象互斥，不足以证明接入完成」。所以这里
让一端通过 ``PumpClient.connect()``（主程序与设备测试共用同一入口）持锁，另一端用主程序
路径去连，断言：① 拿不到锁时**串口根本没被打开**；② 冲突原因被单独报告，不会被记成
「串口打开失败」。

全部使用伪造的 ``serial`` 模块，不连接真实设备。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.device_lock import DeviceLock, DeviceLockError
from backend.pump_hardware import client as client_module
from backend.pump_hardware.client import PumpClient, PumpClientError
from backend.pump_hardware.config import SerialConfig
from backend.pump_hardware.service import PumpHardwareService

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeSerialPort:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def reset_input_buffer(self) -> None:
        pass


class FakeSerialModule:
    """记录打开尝试的假 pyserial；``fail_times`` 让前 N 次构造失败。"""

    PARITY_EVEN = "E"
    PARITY_NONE = "N"
    EIGHTBITS = 8
    STOPBITS_ONE = 1

    def __init__(self, fail_times: int = 0) -> None:
        self.attempts: list = []
        self.fail_times = fail_times

    def Serial(self, **kwargs):  # noqa: N802 - 模仿 pyserial 的命名
        self.attempts.append(kwargs)
        if len(self.attempts) <= self.fail_times:
            raise OSError("access is denied")
        return FakeSerialPort(**kwargs)


@pytest.fixture()
def lock_dir(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "locks"
    directory.mkdir()
    monkeypatch.setattr("backend.device_lock.device_lock_dir", lambda: directory)
    return directory


@pytest.fixture()
def fake_serial(monkeypatch):
    fake = FakeSerialModule()
    monkeypatch.setattr(client_module, "serial", fake)
    return fake


# ---------------------------------------------------------------- 生命周期

def test_pump_connect_takes_the_lock_and_disconnect_releases_it(lock_dir, fake_serial) -> None:
    pump = PumpClient(SerialConfig(port="COM7"))
    pump.connect()
    try:
        assert fake_serial.attempts, "应当尝试打开串口"
        assert pump._device_lock is not None
        assert pump._device_lock.key == "pump-com7"
        # 同一设备已被占用：本进程的另一持有者也不得再取
        with pytest.raises(DeviceLockError):
            DeviceLock("pump-com7").acquire()
    finally:
        pump.disconnect()

    assert pump._device_lock is None
    with DeviceLock("pump-com7"):        # 断开后互斥释放
        pass


def test_lock_is_released_when_the_port_cannot_be_opened(lock_dir, monkeypatch) -> None:
    """打开失败必须释放锁，否则设备被永久占住。"""
    monkeypatch.setattr(client_module, "serial", FakeSerialModule(fail_times=99))
    pump = PumpClient(SerialConfig(port="COM7"))

    with pytest.raises(PumpClientError):
        pump.connect()

    assert pump._device_lock is None
    with DeviceLock("pump-com7"):
        pass


def test_same_port_with_different_address_shares_one_lock(lock_dir, fake_serial) -> None:
    """同一串口上的不同泵地址仍是同一台设备，不能由两个持有者分别控制。"""
    first = PumpClient(SerialConfig(port="COM7", address=1))
    second = PumpClient(SerialConfig(port="COM7", address=2))
    assert first.device_lock_key == second.device_lock_key == "pump-com7"

    first.connect()
    try:
        with pytest.raises(DeviceLockError):
            second.connect()
    finally:
        first.disconnect()


def test_port_spelling_variants_collide_on_one_lock(lock_dir, fake_serial) -> None:
    holder = PumpClient(SerialConfig(port="com7"))
    holder.connect()
    try:
        other = PumpClient(SerialConfig(port="\\\\.\\COM7"))
        assert other.device_lock_key == "pump-com7"
        with pytest.raises(DeviceLockError):
            other.connect()
    finally:
        holder.disconnect()


def test_connect_and_probe_reports_a_lock_conflict_distinctly(lock_dir, monkeypatch) -> None:
    """冲突必须单独报告：不能被记成「串口打开失败」，也不该白试三种奇偶校验。"""
    fake = FakeSerialModule()
    monkeypatch.setattr(client_module, "serial", fake)
    holder = DeviceLock("pump-com7")
    holder.acquire()
    try:
        service = PumpHardwareService(serial_config=SerialConfig(port="COM7"))
        state = service.connect_and_probe()
    finally:
        holder.release()

    assert state.comm_established is False
    assert "device_lock" in state.failed
    assert "serial" not in state.failed, "锁冲突不得被记成串口打开失败"
    assert fake.attempts == [], "拿不到设备锁时不得打开串口"


# ---------------------------------------------------------------- 跨进程竞争

_MAIN_PROGRAM_HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, sys.argv[1])

    import backend.device_lock as device_lock
    device_lock.device_lock_dir = lambda: Path(sys.argv[2])

    from backend.pump_hardware import client as client_module
    from backend.pump_hardware.client import PumpClient
    from backend.pump_hardware.config import SerialConfig

    class Port:
        def __init__(self, **kwargs):
            self.is_open = True
        def close(self):
            self.is_open = False
        def reset_input_buffer(self):
            pass

    client_module.serial = SimpleNamespace(
        PARITY_EVEN="E", PARITY_NONE="N", EIGHTBITS=8, STOPBITS_ONE=1,
        Serial=lambda **kwargs: Port(**kwargs),
    )

    pump = PumpClient(SerialConfig(port=sys.argv[3]))
    pump.connect()
    print("HELD", flush=True)
    time.sleep(float(sys.argv[4]))
    """
)

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


def _start_holder(script: str, lock_dir: Path, key: str) -> subprocess.Popen:
    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(REPO_ROOT), str(lock_dir), key, "30"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert holder.stdout is not None
    line = holder.stdout.readline().strip()
    assert line == "HELD", f"holder 未就绪：{line!r} {holder.stderr.read() if holder.stderr else ''}"
    return holder


def test_main_program_cannot_open_a_port_already_held_by_another_process(
    lock_dir: Path, monkeypatch,
) -> None:
    """另一端走主程序连接入口持锁，本端用主程序入口去连：必须被拒且不碰串口。"""
    holder = _start_holder(_MAIN_PROGRAM_HOLDER, lock_dir, "COM7")
    try:
        fake = FakeSerialModule()
        monkeypatch.setattr(client_module, "serial", fake)
        pump = PumpClient(SerialConfig(port="COM7"))

        with pytest.raises(DeviceLockError):
            pump.connect()

        assert fake.attempts == [], "拿不到设备锁时绝不能打开串口"
        assert pump.is_connected() is False
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_main_program_is_blocked_while_the_capture_script_holds_the_device(
    lock_dir: Path, monkeypatch,
) -> None:
    """脚本侧先持锁时，主程序连接必须失败。"""
    holder = _start_holder(_SCRIPT_HOLDER, lock_dir, "pump-com7")
    try:
        fake = FakeSerialModule()
        monkeypatch.setattr(client_module, "serial", fake)
        pump = PumpClient(SerialConfig(port="COM7"))

        with pytest.raises(DeviceLockError):
            pump.connect()
        assert fake.attempts == []
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_lock_is_free_again_after_the_other_process_exits(lock_dir: Path,
                                                         monkeypatch) -> None:
    """另一端（走主程序入口）退出后，本端对**同一串口**应能立刻连上。"""
    holder = subprocess.Popen(
        [sys.executable, "-c", _MAIN_PROGRAM_HOLDER, str(REPO_ROOT), str(lock_dir), "COM7", "0.4"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "HELD"
    holder.wait(timeout=30)

    fake = FakeSerialModule()
    monkeypatch.setattr(client_module, "serial", fake)
    pump = PumpClient(SerialConfig(port="COM7"))
    pump.connect()
    try:
        assert pump.is_connected() is True
        assert pump.device_lock_key == "pump-com7"
    finally:
        pump.disconnect()

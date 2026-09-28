"""设备独占锁：同一台物理设备在任一时刻只能由一个进程持有。

为什么不用 PID 文件判活：``output/hardware_lock.py`` 用 PID 文件 + ``os.kill(pid, 0)``
判断锁持有者是否还活着。在 Windows 上 CPython 对非控制信号的 ``os.kill`` 实现是
``TerminateProcess``，``sig=0`` 会**终止**目标进程而不是只探测存活——拿它当判活手段
是危险的。这里改用操作系统级建议锁（Windows ``msvcrt.locking`` / POSIX ``fcntl.flock``）：
进程退出时内核自动释放，既不需要判活启发式，也不会把陈旧锁留成永久占位。

两条容易漏掉的互斥要求，都在这里实现：

1. **锁目录不随数据目录变化**（见 ``runtime_paths.device_lock_dir``）。否则不同
   ``MCS_DATA_DIR`` 的进程会各拿一把锁。
2. **进程内也必须唯一**。Windows 的字节区间锁在同一个进程内是可以重复取得的，所以
   「同一个进程」不能当作可以随意共享设备；这里用一个进程级登记表兜住。登记表同时持有
   文件句柄，因此 ``DeviceLock(key).acquire()`` 即使不保留对象也不会被引用计数回收掉句柄
   而静默丢锁。
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import IO

from .runtime_paths import device_lock_dir

_LOCK_BYTES = 1
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

_HANDLES: dict[str, IO[bytes]] = {}
_HELD_GUARD = threading.Lock()


class DeviceLockError(RuntimeError):
    """设备已被占用（其他进程或本进程的其他持有者），拒绝并发访问。"""


def normalize_port_key(port: object) -> str:
    """把串口标识规范成锁键。

    同一串口的不同写法必须归一到同一个键：``COM3`` / ``com3`` / ``\\\\.\\COM3``
    / `` COM3 `` 都是 ``pump-com3``。

    **地址不参与锁键**：同一个串口上的不同泵地址仍然是同一台物理设备，不得由两个进程
    分别控制。
    """
    text = str(port or "").strip().replace("\\\\.\\", "").replace("/", "")
    lowered = text.lower()
    if lowered.startswith("com") and lowered[3:].isdigit():
        return f"pump-com{int(lowered[3:])}"
    return f"pump-{_SAFE.sub('-', lowered).strip('-') or 'unknown'}"


def camera_lock_key(unique_id: object) -> str:
    """相机锁键：设备稳定标识（如 unique_id），不随枚举顺序变化。"""
    text = str(unique_id or "").strip().lower()
    return f"camera-{_SAFE.sub('-', text).strip('-') or 'unknown'}"


class DeviceLock:
    """基于 OS 建议锁的设备独占锁，支持 ``with`` 与手动 acquire/release。

    ``key`` 是设备身份（泵用 ``normalize_port_key(串口)``，相机用 ``camera_lock_key(unique_id)``）；
    锁文件名由它导出，目录取自 :func:`backend.runtime_paths.device_lock_dir`。
    """

    def __init__(self, key: str, *, path: Path | None = None) -> None:
        self.key = str(key)
        self.path = (Path(path) if path is not None
                     else device_lock_dir() / f"{_SAFE.sub('-', self.key).strip('-')}.lock")
        self._registry_key = str(self.path.resolve())
        self._handle: IO[bytes] | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError(f"锁已被本对象持有：{self.path}")
        with _HELD_GUARD:
            if self._registry_key in _HANDLES:
                raise DeviceLockError(
                    f"本进程已持有 {self.key} 的设备锁（{self.path}）；"
                    "同进程也不得共享同一设备")
            handle = open(self.path, "a+b")
            try:
                self._lock(handle)
            except OSError as exc:
                handle.close()
                raise DeviceLockError(
                    f"设备锁 {self.path} 已被其他进程占用，拒绝并发访问同一设备；"
                    "请确认没有第二个程序（含前端界面或采集脚本）正在使用该设备"
                ) from exc
            # 登记句柄：即使调用方不保留本对象，句柄也不会被回收、锁也不会静默丢失。
            _HANDLES[self._registry_key] = handle
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        with _HELD_GUARD:
            _HANDLES.pop(self._registry_key, None)
        try:
            self._unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> "DeviceLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    @staticmethod
    def _lock(handle: IO[bytes]) -> None:
        # msvcrt 只能锁已存在的字节区间，先确保文件至少有一个字节。
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_BYTES)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: IO[bytes]) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_BYTES)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def acquire_device_locks(keys) -> list[DeviceLock]:
    """按**固定顺序**获取多把设备锁；任一失败即释放本次已取得的，不留半持有状态。

    顺序由本函数内部排序决定，**不信任调用方给的顺序**——两个进程以不同顺序请求同一组
    设备才会互相等待，固定顺序是防死锁的前提。
    """
    acquired: list[DeviceLock] = []
    try:
        for key in sorted({str(item) for item in keys}):
            lock = DeviceLock(key)
            lock.acquire()
            acquired.append(lock)
    except BaseException:
        release_device_locks(acquired)
        raise
    return acquired


def release_device_locks(locks) -> list[BaseException]:
    """逆序释放；返回释放过程中的异常（不抛出，交由调用方处置）。"""
    errors: list[BaseException] = []
    for lock in reversed(list(locks)):
        try:
            lock.release()
        except BaseException as exc:  # 释放失败要报出来，不能吞
            errors.append(exc)
    return errors

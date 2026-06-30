from __future__ import annotations

import argparse
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    WinDLL,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_ubyte,
    c_uint,
    c_ushort,
    c_void_p,
    cast,
    memset,
    sizeof,
)
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.vision.camera_adapters.hikrobot_camera import HikrobotSdkLoader  # noqa: E402


DEFAULT_MVS_DLL_PATHS = (
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll",
    r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll",
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86\MvCameraControl.dll",
)

MV_MAX_DEVICE_NUM = 256
MV_GIGE_DEVICE = 0x00000001
MV_USB_DEVICE = 0x00000004
MV_ACCESS_EXCLUSIVE = 1

PIXEL_MONO8 = 0x01080001
PIXEL_BAYER_GR8 = 0x01080008
PIXEL_BAYER_RG8 = 0x01080009
PIXEL_BAYER_GB8 = 0x0108000A
PIXEL_BAYER_BG8 = 0x0108000B
PIXEL_RGB8 = 0x02180014
PIXEL_BGR8 = 0x02180015


class DirectDeviceInfoList(Structure):
    _fields_ = [
        ("nDeviceNum", c_uint),
        ("pDeviceInfo", c_void_p * MV_MAX_DEVICE_NUM),
    ]


class DirectFrameInfo(Structure):
    _fields_ = [
        ("nWidth", c_ushort),
        ("nHeight", c_ushort),
        ("enPixelType", c_uint),
        ("nFrameNum", c_uint),
        ("nDevTimeStampHigh", c_uint),
        ("nDevTimeStampLow", c_uint),
        ("nReserved0", c_uint),
        ("nHostTimeStamp", c_uint * 2),
        ("nFrameLen", c_uint),
        ("nSecondCount", c_uint),
        ("nCycleCount", c_uint),
        ("nCycleOffset", c_uint),
        ("fGain", c_float),
        ("fExposureTime", c_float),
        ("nAverageBrightness", c_uint),
        ("nRed", c_uint),
        ("nGreen", c_uint),
        ("nBlue", c_uint),
        ("nFrameCounter", c_uint),
        ("nTriggerIndex", c_uint),
        ("nInput", c_uint),
        ("nOutput", c_uint),
        ("nOffsetX", c_ushort),
        ("nOffsetY", c_ushort),
        ("nChunkWidth", c_ushort),
        ("nChunkHeight", c_ushort),
        ("nLostPacket", c_uint),
        ("nUnparsedChunkNum", c_uint),
        ("nReserved", c_uint * 36),
    ]


class DirectPixelConvertParam(Structure):
    _fields_ = [
        ("nWidth", c_ushort),
        ("nHeight", c_ushort),
        ("enSrcPixelType", c_uint),
        ("pSrcData", POINTER(c_ubyte)),
        ("nSrcDataLen", c_uint),
        ("enDstPixelType", c_uint),
        ("pDstBuffer", POINTER(c_ubyte)),
        ("nDstBufferSize", c_uint),
        ("nDstLen", c_uint),
        ("nReserved", c_uint * 4),
    ]


def _ret_hex(ret: int) -> str:
    return f"0x{int(ret) & 0xFFFFFFFF:08X}"


def _decode_mvs_text(value: Any) -> str:
    try:
        raw = bytes(value)
    except Exception:
        raw = str(value).encode("latin1", errors="ignore")
    raw = raw.split(b"\x00", 1)[0]
    for encoding in ("utf-8", "gbk", "latin1"):
        try:
            return raw.decode(encoding).strip()
        except Exception:
            continue
    return raw.decode("latin1", errors="ignore").strip()


def _load_mvs(sdk_path: str):
    loader = HikrobotSdkLoader(configured_path=sdk_path, logger=print)
    mvs = loader.load()
    print(f"[SDK] python_interface={loader.python_interface_path}")
    print(f"[SDK] dll={loader.dll_path}")
    return mvs


def _find_mvs_dll(sdk_path: str) -> Path:
    candidates: list[Path] = []
    if sdk_path:
        root = Path(sdk_path).expanduser()
        if root.is_file() and root.name.lower() == "mvcameracontrol.dll":
            candidates.append(root)
        elif root.exists():
            candidates.append(root / "MvCameraControl.dll")
            try:
                candidates.extend(root.rglob("MvCameraControl.dll"))
            except Exception:
                pass
    candidates.extend(Path(path) for path in DEFAULT_MVS_DLL_PATHS)
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError(
        "MvCameraControl.dll was not found. Pass --sdk-path to MVS Runtime or DLL path."
    )


def _load_direct_dll(sdk_path: str):
    dll_path = _find_mvs_dll(sdk_path)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_path.parent))
    dll = WinDLL(str(dll_path))
    print(f"[SDK][DIRECT] dll={dll_path}")

    dll.MV_CC_EnumDevices.argtypes = [c_uint, POINTER(DirectDeviceInfoList)]
    dll.MV_CC_EnumDevices.restype = c_int
    dll.MV_CC_CreateHandle.argtypes = [POINTER(c_void_p), c_void_p]
    dll.MV_CC_CreateHandle.restype = c_int
    dll.MV_CC_OpenDevice.argtypes = [c_void_p, c_uint, c_ushort]
    dll.MV_CC_OpenDevice.restype = c_int
    dll.MV_CC_CloseDevice.argtypes = [c_void_p]
    dll.MV_CC_CloseDevice.restype = c_int
    dll.MV_CC_DestroyHandle.argtypes = [c_void_p]
    dll.MV_CC_DestroyHandle.restype = c_int
    dll.MV_CC_StartGrabbing.argtypes = [c_void_p]
    dll.MV_CC_StartGrabbing.restype = c_int
    dll.MV_CC_StopGrabbing.argtypes = [c_void_p]
    dll.MV_CC_StopGrabbing.restype = c_int
    dll.MV_CC_GetOneFrameTimeout.argtypes = [
        c_void_p,
        POINTER(c_ubyte),
        c_uint,
        POINTER(DirectFrameInfo),
        c_uint,
    ]
    dll.MV_CC_GetOneFrameTimeout.restype = c_int
    dll.MV_CC_SetEnumValue.argtypes = [c_void_p, c_char_p, c_uint]
    dll.MV_CC_SetEnumValue.restype = c_int
    dll.MV_CC_SetIntValue.argtypes = [c_void_p, c_char_p, c_uint]
    dll.MV_CC_SetIntValue.restype = c_int
    dll.MV_CC_GetOptimalPacketSize.argtypes = [c_void_p]
    dll.MV_CC_GetOptimalPacketSize.restype = c_int
    if hasattr(dll, "MV_CC_ConvertPixelType"):
        dll.MV_CC_ConvertPixelType.argtypes = [c_void_p, POINTER(DirectPixelConvertParam)]
        dll.MV_CC_ConvertPixelType.restype = c_int
    return dll


def _enum_devices_direct(dll):
    device_list = DirectDeviceInfoList()
    ret = dll.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, byref(device_list))
    if ret != 0:
        raise RuntimeError(f"MV_CC_EnumDevices failed: {_ret_hex(ret)}")
    count = int(device_list.nDeviceNum)
    print(f"[ENUM][DIRECT] device_count={count}")
    if count <= 0:
        raise RuntimeError("No Hikrobot cameras found. Close MVS preview windows and check cable/network.")
    devices = []
    for idx in range(count):
        ptr = device_list.pDeviceInfo[idx]
        print(f"[ENUM][DIRECT] [{idx}] device_info_ptr=0x{int(ptr or 0):X}")
        devices.append(ptr)
    return devices


def _open_camera_direct(dll, device_ptr):
    handle = c_void_p()
    ret = dll.MV_CC_CreateHandle(byref(handle), c_void_p(device_ptr))
    if ret != 0:
        raise RuntimeError(f"MV_CC_CreateHandle failed: {_ret_hex(ret)}")
    opened = False
    try:
        ret = dll.MV_CC_OpenDevice(handle, MV_ACCESS_EXCLUSIVE, 0)
        if ret != 0:
            raise RuntimeError(f"MV_CC_OpenDevice failed: {_ret_hex(ret)}")
        opened = True
        dll.MV_CC_SetEnumValue(handle, b"TriggerMode", 0)
        packet_size = int(dll.MV_CC_GetOptimalPacketSize(handle))
        if packet_size > 0:
            dll.MV_CC_SetIntValue(handle, b"GevSCPSPacketSize", packet_size)
        return handle
    except Exception:
        if opened:
            try:
                dll.MV_CC_CloseDevice(handle)
            except Exception:
                pass
        try:
            dll.MV_CC_DestroyHandle(handle)
        except Exception:
            pass
        raise


def _capture_one_direct(dll, handle, timeout_ms: int, buffer_mb: int):
    buffer_size = int(buffer_mb) * 1024 * 1024
    buffer = (c_ubyte * buffer_size)()
    info = DirectFrameInfo()
    ret = dll.MV_CC_GetOneFrameTimeout(handle, buffer, buffer_size, byref(info), int(timeout_ms))
    if ret != 0:
        raise RuntimeError(f"MV_CC_GetOneFrameTimeout failed: {_ret_hex(ret)}")

    width = int(info.nWidth)
    height = int(info.nHeight)
    frame_len = int(info.nFrameLen)
    pixel_type = int(info.enPixelType)
    actual_len = _expected_frame_len(width, height, pixel_type, frame_len)
    if width <= 0 or height <= 0 or actual_len <= 0:
        raise RuntimeError(
            f"Invalid frame metadata: width={width}, height={height}, len={frame_len}, "
            f"pixel={_ret_hex(pixel_type)}"
        )
    if frame_len != actual_len:
        print(f"[FRAME][DIRECT][WARN] nFrameLen={frame_len}, using computed_len={actual_len}")
    raw = np.ctypeslib.as_array(buffer)[:actual_len].copy()
    image, pixel_name = _convert_frame_direct(dll, handle, raw, width, height, pixel_type, actual_len)
    print(
        f"[FRAME][DIRECT] width={width} height={height} len={frame_len} "
        f"pixel={pixel_name} frame_num={int(info.nFrameNum)}"
    )
    return image


def _expected_frame_len(width: int, height: int, pixel_type: int, reported_len: int) -> int:
    pixels = int(width) * int(height)
    if pixels <= 0:
        return 0
    if pixel_type in {PIXEL_MONO8, PIXEL_BAYER_GR8, PIXEL_BAYER_RG8, PIXEL_BAYER_GB8, PIXEL_BAYER_BG8}:
        return pixels
    if pixel_type in {PIXEL_RGB8, PIXEL_BGR8}:
        return pixels * 3
    if 0 < int(reported_len) < 512 * 1024 * 1024:
        return int(reported_len)
    return pixels * 3


def _enum_devices(mvs):
    device_list = mvs.MV_CC_DEVICE_INFO_LIST()
    layer_type = int(getattr(mvs, "MV_GIGE_DEVICE", 0x00000001)) | int(
        getattr(mvs, "MV_USB_DEVICE", 0x00000004)
    )
    ret = mvs.MvCamera.MV_CC_EnumDevices(layer_type, device_list)
    if ret != 0:
        raise RuntimeError(f"MV_CC_EnumDevices failed: {_ret_hex(ret)}")

    count = int(device_list.nDeviceNum)
    print(f"[ENUM] device_count={count}")
    if count <= 0:
        raise RuntimeError("No Hikrobot cameras found. Close MVS preview windows and check cable/network.")

    info_type = getattr(mvs, "MV_CC_DEVICE_INFO")
    devices = []
    for idx in range(count):
        info = cast(device_list.pDeviceInfo[idx], POINTER(info_type)).contents
        label = _device_label(mvs, info, idx)
        print(f"[ENUM] [{idx}] {label}")
        devices.append(info)
    return devices


def _device_label(mvs, info, index: int) -> str:
    layer_type = int(getattr(info, "nTLayerType", 0))
    gige_type = int(getattr(mvs, "MV_GIGE_DEVICE", 0x00000001))
    usb_type = int(getattr(mvs, "MV_USB_DEVICE", 0x00000004))
    if layer_type == gige_type:
        transport = "GigE"
        detail = info.SpecialInfo.stGigEInfo
    elif layer_type == usb_type:
        transport = "USB3"
        detail = info.SpecialInfo.stUsb3VInfo
    else:
        transport = f"type={layer_type}"
        detail = None

    if detail is None:
        return f"{transport} device index={index}"
    model = _decode_mvs_text(getattr(detail, "chModelName", b""))
    serial = _decode_mvs_text(getattr(detail, "chSerialNumber", b""))
    user_name = _decode_mvs_text(getattr(detail, "chUserDefinedName", b""))
    return f"{transport} model={model or '--'} serial={serial or '--'} name={user_name or '--'}"


def _open_camera(mvs, device_info):
    cam = mvs.MvCamera()
    ret = cam.MV_CC_CreateHandle(device_info)
    if ret != 0:
        raise RuntimeError(f"MV_CC_CreateHandle failed: {_ret_hex(ret)}")

    opened = False
    try:
        access = int(getattr(mvs, "MV_ACCESS_Exclusive", 1))
        ret = cam.MV_CC_OpenDevice(access, 0)
        if ret != 0:
            raise RuntimeError(f"MV_CC_OpenDevice failed: {_ret_hex(ret)}")
        opened = True

        try:
            cam.MV_CC_SetEnumValue("TriggerMode", 0)
        except Exception:
            pass

        try:
            packet_size = int(cam.MV_CC_GetOptimalPacketSize())
            if packet_size > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
        except Exception:
            pass

        return cam
    except Exception:
        if opened:
            try:
                cam.MV_CC_CloseDevice()
            except Exception:
                pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        raise


def _capture_one(mvs, cam, timeout_ms: int):
    frame_out = mvs.MV_FRAME_OUT()
    memset(byref(frame_out), 0, sizeof(frame_out))
    ret = cam.MV_CC_GetImageBuffer(frame_out, int(timeout_ms))
    if ret != 0:
        raise RuntimeError(f"MV_CC_GetImageBuffer failed: {_ret_hex(ret)}")

    try:
        info = frame_out.stFrameInfo
        width = int(info.nWidth)
        height = int(info.nHeight)
        frame_len = int(info.nFrameLen)
        pixel_type = int(info.enPixelType)
        if width <= 0 or height <= 0 or frame_len <= 0:
            raise RuntimeError(
                f"Invalid frame metadata: width={width}, height={height}, len={frame_len}"
            )
        raw = np.ctypeslib.as_array(frame_out.pBufAddr, shape=(frame_len,)).copy()
        image, pixel_name = _convert_frame(mvs, cam, raw, width, height, pixel_type, frame_len)
        print(
            f"[FRAME] width={width} height={height} len={frame_len} "
            f"pixel={pixel_name} frame_num={int(getattr(info, 'nFrameNum', 0))}"
        )
        return image
    finally:
        try:
            cam.MV_CC_FreeImageBuffer(frame_out)
        except Exception:
            pass


def _pixel_constants(mvs) -> dict[str, int]:
    return {
        "Mono8": int(getattr(mvs, "PixelType_Gvsp_Mono8", 0x01080001)),
        "BayerGR8": int(getattr(mvs, "PixelType_Gvsp_BayerGR8", 0x01080008)),
        "BayerRG8": int(getattr(mvs, "PixelType_Gvsp_BayerRG8", 0x01080009)),
        "BayerGB8": int(getattr(mvs, "PixelType_Gvsp_BayerGB8", 0x0108000A)),
        "BayerBG8": int(getattr(mvs, "PixelType_Gvsp_BayerBG8", 0x0108000B)),
        "RGB8": int(getattr(mvs, "PixelType_Gvsp_RGB8_Packed", 0x02180014)),
        "BGR8": int(getattr(mvs, "PixelType_Gvsp_BGR8_Packed", 0x02180015)),
    }


def _convert_frame(mvs, cam, raw, width: int, height: int, pixel_type: int, frame_len: int):
    constants = _pixel_constants(mvs)
    if pixel_type == constants["Mono8"]:
        return raw.reshape((height, width)), "Mono8"
    if pixel_type == constants["RGB8"]:
        rgb = raw.reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "RGB8"
    if pixel_type == constants["BGR8"]:
        return raw.reshape((height, width, 3)), "BGR8"

    bayer_codes = {
        constants["BayerRG8"]: ("BayerRG8", cv2.COLOR_BayerRG2BGR),
        constants["BayerGB8"]: ("BayerGB8", cv2.COLOR_BayerGB2BGR),
        constants["BayerGR8"]: ("BayerGR8", cv2.COLOR_BayerGR2BGR),
        constants["BayerBG8"]: ("BayerBG8", cv2.COLOR_BayerBG2BGR),
    }
    if pixel_type in bayer_codes:
        name, code = bayer_codes[pixel_type]
        return cv2.cvtColor(raw.reshape((height, width)), code), name

    converted = _convert_with_mvs(mvs, cam, raw, width, height, pixel_type, frame_len)
    if converted is not None:
        return converted, f"MVSConverted:{_ret_hex(pixel_type)}"
    raise RuntimeError(f"Unsupported pixel type: {_ret_hex(pixel_type)}")


def _convert_frame_direct(dll, handle, raw, width: int, height: int, pixel_type: int, frame_len: int):
    if pixel_type == PIXEL_MONO8:
        return raw.reshape((height, width)), "Mono8"
    if pixel_type == PIXEL_RGB8:
        rgb = raw.reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "RGB8"
    if pixel_type == PIXEL_BGR8:
        return raw.reshape((height, width, 3)), "BGR8"

    bayer_codes = {
        PIXEL_BAYER_RG8: ("BayerRG8", cv2.COLOR_BayerRG2BGR),
        PIXEL_BAYER_GB8: ("BayerGB8", cv2.COLOR_BayerGB2BGR),
        PIXEL_BAYER_GR8: ("BayerGR8", cv2.COLOR_BayerGR2BGR),
        PIXEL_BAYER_BG8: ("BayerBG8", cv2.COLOR_BayerBG2BGR),
    }
    if pixel_type in bayer_codes:
        name, code = bayer_codes[pixel_type]
        return cv2.cvtColor(raw.reshape((height, width)), code), name

    converted = _convert_with_mvs_direct(dll, handle, raw, width, height, pixel_type, frame_len)
    if converted is not None:
        return converted, f"MVSConverted:{_ret_hex(pixel_type)}"
    raise RuntimeError(f"Unsupported pixel type: {_ret_hex(pixel_type)}")


def _convert_with_mvs(mvs, cam, raw, width: int, height: int, pixel_type: int, frame_len: int):
    convert_param_type = getattr(mvs, "MV_CC_PIXEL_CONVERT_PARAM", None)
    if convert_param_type is None:
        return None
    dst_size = width * height * 3
    dst = (c_ubyte * dst_size)()
    src = np.ascontiguousarray(raw)
    param = convert_param_type()
    param.nWidth = width
    param.nHeight = height
    param.pSrcData = src.ctypes.data_as(POINTER(c_ubyte))
    param.nSrcDataLen = int(frame_len)
    param.enSrcPixelType = int(pixel_type)
    param.enDstPixelType = int(_pixel_constants(mvs)["BGR8"])
    param.pDstBuffer = dst
    param.nDstBufferSize = dst_size
    ret = cam.MV_CC_ConvertPixelType(param)
    if ret != 0:
        print(f"[CONVERT][WARN] MV_CC_ConvertPixelType failed: {_ret_hex(ret)}")
        return None
    out = np.ctypeslib.as_array(dst, shape=(dst_size,)).copy()
    return out.reshape((height, width, 3))


def _convert_with_mvs_direct(dll, handle, raw, width: int, height: int, pixel_type: int, frame_len: int):
    if not hasattr(dll, "MV_CC_ConvertPixelType"):
        return None
    dst_size = width * height * 3
    dst = (c_ubyte * dst_size)()
    src = np.ascontiguousarray(raw)
    param = DirectPixelConvertParam()
    param.nWidth = int(width)
    param.nHeight = int(height)
    param.pSrcData = src.ctypes.data_as(POINTER(c_ubyte))
    param.nSrcDataLen = int(frame_len)
    param.enSrcPixelType = int(pixel_type)
    param.enDstPixelType = PIXEL_BGR8
    param.pDstBuffer = dst
    param.nDstBufferSize = int(dst_size)
    ret = dll.MV_CC_ConvertPixelType(handle, byref(param))
    if ret != 0:
        print(f"[CONVERT][DIRECT][WARN] MV_CC_ConvertPixelType failed: {_ret_hex(ret)}")
        return None
    out_len = int(param.nDstLen or dst_size)
    out = np.ctypeslib.as_array(dst, shape=(out_len,)).copy()
    return out[:dst_size].reshape((height, width, 3))


def _handle_frame(image, title: str, save_path: str, no_window: bool, wait_ms: int) -> bool:
    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output), image)
        if not ok:
            raise RuntimeError(f"Failed to save image: {output}")
        print(f"[SAVE] {output}")

    if no_window:
        return True

    cv2.imshow(title, image)
    key = cv2.waitKey(max(1, int(wait_ms))) & 0xFF
    return key not in (27, ord("q"), ord("Q"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture and display Hikrobot MVS frames via ctypes.")
    parser.add_argument("--sdk-path", default="", help="MVS SDK root, MvImport path, Runtime path, or DLL path.")
    parser.add_argument("--index", type=int, default=0, help="Camera index from enumeration.")
    parser.add_argument("--timeout-ms", type=int, default=3000, help="Frame timeout in milliseconds.")
    parser.add_argument("--buffer-mb", type=int, default=64, help="Direct DLL capture buffer size.")
    parser.add_argument("--save", default="", help="Optional output image path, for example data/hikrobot_snapshot.png.")
    parser.add_argument("--no-window", action="store_true", help="Capture/save only; do not show OpenCV window.")
    parser.add_argument("--direct-only", action="store_true", help="Use MvCameraControl.dll directly.")
    parser.add_argument("--interval-sec", type=float, default=1.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Capture one frame and exit.")
    args = parser.parse_args()

    image = None
    wait_ms = max(1, int(max(0.001, args.interval_sec) * 1000))
    if not args.direct_only:
        try:
            mvs = _load_mvs(args.sdk_path)
            devices = _enum_devices(mvs)
            if args.index < 0 or args.index >= len(devices):
                raise RuntimeError(f"Camera index out of range: {args.index}, available=0..{len(devices) - 1}")

            cam = _open_camera(mvs, devices[args.index])
            try:
                ret = cam.MV_CC_StartGrabbing()
                if ret != 0:
                    raise RuntimeError(f"MV_CC_StartGrabbing failed: {_ret_hex(ret)}")
                print("[GRAB] started")
                print("[DISPLAY] Updating every %.3g second(s). Press q or Esc to exit." % args.interval_sec)
                while True:
                    image = _capture_one(mvs, cam, args.timeout_ms)
                    if not _handle_frame(image, "Hikrobot MVS ctypes live", args.save, args.no_window, wait_ms):
                        break
                    if args.once or args.no_window:
                        break
            finally:
                try:
                    cam.MV_CC_StopGrabbing()
                except Exception:
                    pass
                try:
                    cam.MV_CC_CloseDevice()
                except Exception:
                    pass
                try:
                    cam.MV_CC_DestroyHandle()
                except Exception:
                    pass
        except Exception as exc:
            print(f"[SDK][PYTHON][WARN] {exc}")
            print("[SDK][DIRECT] Falling back to MvCameraControl.dll direct ctypes mode.")

    if image is None:
        dll = _load_direct_dll(args.sdk_path)
        devices = _enum_devices_direct(dll)
        if args.index < 0 or args.index >= len(devices):
            raise RuntimeError(f"Camera index out of range: {args.index}, available=0..{len(devices) - 1}")

        handle = _open_camera_direct(dll, devices[args.index])
        try:
            ret = dll.MV_CC_StartGrabbing(handle)
            if ret != 0:
                raise RuntimeError(f"MV_CC_StartGrabbing failed: {_ret_hex(ret)}")
            print("[GRAB][DIRECT] started")
            print("[DISPLAY] Updating every %.3g second(s). Press q or Esc to exit." % args.interval_sec)
            while True:
                image = _capture_one_direct(dll, handle, args.timeout_ms, args.buffer_mb)
                if not _handle_frame(image, "Hikrobot MVS ctypes live", args.save, args.no_window, wait_ms):
                    break
                if args.once or args.no_window:
                    break
        finally:
            try:
                dll.MV_CC_StopGrabbing(handle)
            except Exception:
                pass
            try:
                dll.MV_CC_CloseDevice(handle)
            except Exception:
                pass
            try:
                dll.MV_CC_DestroyHandle(handle)
            except Exception:
                pass

    if not args.no_window:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

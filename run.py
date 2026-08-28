from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from collections.abc import Sequence


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

FRONTEND_REQUIRED_MODULES = {
    "cv2": "opencv-contrib-python",
    "numpy": "numpy",
    "PySide6": "PySide6",
    "serial": "pyserial",
}
VISION_REQUIRED_MODULES = {
    "cv2": "opencv-contrib-python",
    "numpy": "numpy",
}


def ensure_project_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def missing_runtime_packages(required_modules: dict[str, str] | None = None) -> list[str]:
    missing: list[str] = []
    for module_name, package_name in (required_modules or FRONTEND_REQUIRED_MODULES).items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def print_dependency_help(missing: Sequence[str]) -> None:
    packages = " ".join(missing)
    print("当前 Python 环境缺少项目运行依赖：", file=sys.stderr)
    print(f"  {packages}", file=sys.stderr)
    print("", file=sys.stderr)
    print("任选一种方式安装后再启动：", file=sys.stderr)
    print("  python -m pip install -r requirements.lock", file=sys.stderr)
    print("  conda install -c conda-forge numpy opencv pyserial", file=sys.stderr)
    print("  uv sync", file=sys.stderr)
    print("  uv run python run.py", file=sys.stderr)


def run_frontend() -> None:
    ensure_project_root_on_path()

    missing = missing_runtime_packages(FRONTEND_REQUIRED_MODULES)
    if missing:
        print_dependency_help(missing)
        raise SystemExit(1)

    from frontend.app import main as frontend_main

    frontend_main()


def run_tuning(argv: Sequence[str]) -> None:
    ensure_project_root_on_path()

    missing = missing_runtime_packages()
    if missing:
        print_dependency_help(missing)
        raise SystemExit(1)

    from frontend.vision_tuning import main as tuning_main

    parser = argparse.ArgumentParser(description="Launch the standalone droplet detection tuning workbench.")
    parser.add_argument("--video", default="", help="Optional local video path")
    args = parser.parse_args(list(argv))
    tuning_main(args.video)


def run_vision(argv: Sequence[str]) -> None:
    ensure_project_root_on_path()

    missing = missing_runtime_packages(VISION_REQUIRED_MODULES)
    if missing:
        print_dependency_help(missing)
        raise SystemExit(1)

    from backend.vision.run_vision import main as vision_main

    forwarded_args = list(argv)
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(PROJECT_ROOT / "backend" / "vision" / "run_vision.py"), *forwarded_args]
        vision_main()
    finally:
        sys.argv = old_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Microfluidic control system launcher.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("frontend", help="启动 PySide6 前端界面（默认）")

    vision_parser = subparsers.add_parser("vision", help="独立运行后端视觉流水线")
    vision_parser.add_argument(
        "vision_args",
        nargs=argparse.REMAINDER,
        help="传递给 backend.vision.run_vision 的参数",
    )
    tune_parser = subparsers.add_parser("tune", help="启动液滴识别算法调参工作台")
    tune_parser.add_argument("--video", default="", help="可选的本地视频路径")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    if not raw_args:
        run_frontend()
        return

    if raw_args[0] in {"-h", "--help"}:
        build_parser().print_help()
        return

    if raw_args[0] == "frontend":
        if len(raw_args) > 1:
            build_parser().parse_args(raw_args)
        run_frontend()
        return

    if raw_args[0] == "vision":
        run_vision(raw_args[1:])
        return

    if raw_args[0] == "tune":
        run_tuning(raw_args[1:])
        return

    build_parser().parse_args(raw_args)


if __name__ == "__main__":
    main()

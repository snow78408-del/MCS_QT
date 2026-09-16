from __future__ import annotations

import pathlib
import sys


def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from run import run_frontend



if __name__ == "__main__":
    main()

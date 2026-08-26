from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class FrontendImportTests(unittest.TestCase):
    def test_qt_and_transport_imports_do_not_require_tk(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = """
import builtins

real_import = builtins.__import__
def import_without_tk(name, *args, **kwargs):
    if name == "tkinter" or name.startswith("tkinter.") or name == "_tkinter":
        raise ModuleNotFoundError("No module named '_tkinter'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_tk
import frontend.app
import frontend.pages
import frontend.components
import frontend.video_process
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

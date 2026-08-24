"""PySide6 frontend entry point (kept for launcher compatibility)."""

from .qt_app import FrontendApp, main

__all__ = ["FrontendApp", "main"]

if __name__ == "__main__":
    main()

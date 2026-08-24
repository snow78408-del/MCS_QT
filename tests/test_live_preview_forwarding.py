from __future__ import annotations

import unittest

from backend.orchestrator.models import FrameSnapshot
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.vision_adapter import GenericVisionAdapter


class _PreviewVisionService:
    def __init__(self) -> None:
        self.frame = FrameSnapshot(
            frame_id=7,
            timestamp=123.0,
            width=720,
            height=540,
            valid=True,
            frame_jpeg=b"jpeg-preview-data",
        )

    def get_frame_snapshot(self) -> FrameSnapshot:
        return self.frame


class LivePreviewForwardingTests(unittest.TestCase):
    def test_generic_adapter_forwards_independent_preview(self) -> None:
        source = _PreviewVisionService()

        frame = GenericVisionAdapter(source).get_frame_snapshot()

        self.assertIs(frame, source.frame)
        self.assertEqual(frame.frame_jpeg, b"jpeg-preview-data")

    def test_orchestrator_returns_preview_bytes_instead_of_analysis_fallback(self) -> None:
        service = OrchestratorService(vision_service=_PreviewVisionService())

        frame = service.get_video_frame_snapshot()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_id, 7)
        self.assertEqual(frame.frame_jpeg, b"jpeg-preview-data")


if __name__ == "__main__":
    unittest.main()

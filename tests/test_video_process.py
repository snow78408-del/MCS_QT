from __future__ import annotations

import unittest

import cv2
import numpy as np

from frontend.video_process import _jpeg_to_tk_payload


class VideoProcessTransportTests(unittest.TestCase):
    def test_jpeg_is_decoded_to_renderer_ppm(self) -> None:
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[:, :, 1] = 180
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        payload = _jpeg_to_tk_payload(encoded.tobytes())

        self.assertTrue(payload.startswith(b"P6\n16 12\n255\n"))
        self.assertEqual(len(payload), len(b"P6\n16 12\n255\n") + 12 * 16 * 3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from backend.vision.camera_profiles import normalize_camera_parameters, resolve_camera_defaults


class CameraProfileTests(unittest.TestCase):
    def test_hikrobot_cs_defaults_are_complete_and_typed(self) -> None:
        name, defaults = resolve_camera_defaults(
            {
                "backend_name": "hikrobot",
                "manufacturer": "HIKROBOT",
                "model": "CS-Series",
            }
        )
        self.assertIn("CS", name)
        self.assertEqual(defaults["width"], 720)
        self.assertEqual(defaults["height"], 540)
        self.assertEqual(defaults["frame_rate"], 100.0)

    def test_user_values_are_normalized_before_downlink(self) -> None:
        values = normalize_camera_parameters(
            {
                "exposure": "4500",
                "gain": "1.5",
                "frame_rate": "40",
                "width": "720",
                "height": "540",
            }
        )
        self.assertEqual(values["exposure"], 4500.0)
        self.assertEqual(values["gain"], 1.5)
        self.assertEqual(values["frame_rate"], 40.0)
        self.assertIsInstance(values["width"], int)

    def test_invalid_camera_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_camera_parameters({"frame_rate": "0"})


if __name__ == "__main__":
    unittest.main()

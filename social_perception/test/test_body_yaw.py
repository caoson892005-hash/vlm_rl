"""Regression tests for camera-derived pedestrian heading."""

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'social_vlm_perception.py'
SPEC = importlib.util.spec_from_file_location('perception_yaw_under_test', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BodyYawTest(unittest.TestCase):
    """Keep heading inference tied to observed RGB-D keypoints only."""

    def setUp(self):
        self.node = SimpleNamespace(
            NOSE=0,
            LEFT_SHOULDER=5,
            RIGHT_SHOULDER=6,
            LEFT_HIP=11,
            RIGHT_HIP=12,
            LEFT_EYE=1,
            RIGHT_EYE=2,
            LEFT_EAR=3,
            RIGHT_EAR=4,
            orientation_keypoint_confidence=0.50,
            orientation_nose_confidence=0.60,
            orientation_face_confidence=0.40,
            minimum_shoulder_width=0.15,
            maximum_shoulder_width=0.60,
            minimum_hip_width=0.15,
            maximum_hip_width=0.50,
            minimum_face_direction=0.04,
            minimum_face_alignment=0.45,
            orientation_hold_frames=-1,
            orientation_flip_frames=4,
            body_yaw_smoothing=0.30,
        )

    def test_nose_to_torso_fallback_handles_side_on_projection(self):
        """A collapsed shoulder width still yields the visually observed facing."""
        points = {
            # Shoulder separation collapses in the ground plane for a side-on
            # person, as commonly happens with a single RGB-D camera.
            5: (4.270, 2.666, 1.30, 0.99),
            6: (4.274, 2.669, 1.30, 0.91),
            # The nose is clearly in front of the torso centre in odom.
            0: (4.471, 2.754, 1.55, 0.83),
        }

        yaw, flip_votes, held_frames = MODULE.SocialVlmPerception.body_yaw_from_keypoints(
            self.node, points, None)

        expected = math.atan2(2.754 - 0.5 * (2.666 + 2.669),
                              4.471 - 0.5 * (4.270 + 4.274))
        self.assertAlmostEqual(yaw, expected)
        self.assertEqual(flip_votes, 0)
        self.assertEqual(held_frames, 0)


if __name__ == '__main__':
    unittest.main()

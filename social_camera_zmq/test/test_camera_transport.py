"""Deterministic transport checks without a camera or live ROS graph."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from sensor_msgs.msg import CameraInfo, Image


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sender = load_script('zmq_camera_sender')
receiver = load_script('zmq_camera_receiver')


class CameraTransportTest(unittest.TestCase):
    def test_rgb_row_padding_and_channel_order(self):
        message = Image()
        message.width = 2
        message.height = 1
        message.step = 8
        message.encoding = 'rgb8'
        message.data = bytes([255, 0, 0, 0, 255, 0, 9, 9])
        bgr = sender.ros_rgb_to_bgr(message)
        self.assertEqual(bgr.shape, (1, 2, 3))
        self.assertEqual(bgr[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(bgr[0, 1].tolist(), [0, 255, 0])

    def test_depth_row_padding_and_big_endian_values(self):
        message = Image()
        message.width = 2
        message.height = 1
        message.step = 6
        message.encoding = '16UC1'
        message.is_bigendian = True
        message.data = bytes([0x03, 0xE8, 0x07, 0xD0, 0, 0])
        depth = sender.ros_depth_to_uint16(message)
        self.assertEqual(depth.tolist(), [[1000, 2000]])

    def test_camera_info_round_trip(self):
        info = CameraInfo()
        info.width = 640
        info.height = 480
        info.header.frame_id = 'camera_color_optical_frame'
        info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
        info.d = [0.1, 0.2, 0.0, 0.0, 0.0]
        data = sender.camera_info_to_dict(info)
        restored = receiver.camera_info_from_dict(data)
        self.assertEqual((restored.width, restored.height), (640, 480))
        self.assertEqual(list(restored.k), list(info.k))
        self.assertEqual(list(restored.d), list(info.d))

    def test_rgbd_requires_matching_fresh_timestamps(self):
        published = []
        params = {'max_pair_delta_s': SimpleNamespace(value=0.15)}
        fake = SimpleNamespace(
            pending_rgb=(np.zeros((2, 3, 3), np.uint8),
                         {'capture_time_ns': 1_000_000_000}),
            pending_depth=(np.zeros((2, 3), np.uint16),
                           {'capture_time_ns': 1_300_000_000}),
            color_info=SimpleNamespace(width=3, height=2),
            depth_info=SimpleNamespace(width=3, height=2),
            last_pair=None,
            last_pair_time=None,
            rgb_pub=None,
            depth_pub=None,
            color_info_pub=None,
            depth_info_pub=None,
            get_parameter=lambda key: params[key],
            _publish=lambda *args: published.append(args),
        )
        receiver.ZmqCameraReceiver._publish_rgbd_pair(fake)
        self.assertEqual(published, [])
        fake.pending_depth[1]['capture_time_ns'] = 1_050_000_000
        receiver.ZmqCameraReceiver._publish_rgbd_pair(fake)
        self.assertEqual(len(published), 2)
        receiver.ZmqCameraReceiver._publish_rgbd_pair(fake)
        self.assertEqual(len(published), 2)

    def test_webcam_is_reopened_after_disconnect(self):
        attempts = []
        params = {'reopen_interval_s': SimpleNamespace(value=2.0)}
        fake = SimpleNamespace(
            capture=None,
            last_open_attempt=-1_000_000.0,
            get_parameter=lambda key: params[key],
            _open_webcam=lambda: attempts.append(True),
        )
        sender.ZmqCameraSender._capture_webcam_frame(fake)
        self.assertEqual(attempts, [True])


if __name__ == '__main__':
    unittest.main()

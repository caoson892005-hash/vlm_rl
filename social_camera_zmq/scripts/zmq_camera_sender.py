#!/usr/bin/env python3
"""Capture RGB locally or relay ROS camera topics through a bounded ZeroMQ PUB socket."""

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import zmq


PROTOCOL_VERSION = 1


def time_to_ns(stamp):
    """Convert a ROS stamp to nanoseconds without changing its clock domain."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def camera_info_to_dict(message):
    """Keep the CameraInfo fields required to reconstruct an identical ROS message."""
    return {
        'header': {
            'stamp_ns': time_to_ns(message.header.stamp),
            'frame_id': message.header.frame_id,
        },
        'height': message.height,
        'width': message.width,
        'distortion_model': message.distortion_model,
        'd': list(message.d),
        'k': list(message.k),
        'r': list(message.r),
        'p': list(message.p),
        'binning_x': message.binning_x,
        'binning_y': message.binning_y,
        'roi': {
            'x_offset': message.roi.x_offset,
            'y_offset': message.roi.y_offset,
            'height': message.roi.height,
            'width': message.roi.width,
            'do_rectify': message.roi.do_rectify,
        },
    }


def ros_rgb_to_bgr(message):
    """Respect ROS Image row padding and RGB/BGR order before JPEG encoding."""
    if message.encoding not in ('bgr8', 'rgb8'):
        raise ValueError(f'unsupported RGB encoding {message.encoding}')
    if message.step < message.width * 3 or len(message.data) < message.height * message.step:
        raise ValueError('invalid RGB Image step or data length')
    rows = np.frombuffer(message.data, dtype=np.uint8,
                         count=message.height * message.step).reshape(
                             message.height, message.step)
    image = rows[:, :message.width * 3].reshape(message.height, message.width, 3)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if message.encoding == 'rgb8' else image


def ros_depth_to_uint16(message):
    """Retain millimetre depth values across ROS row padding and byte order."""
    if message.encoding != '16UC1':
        raise ValueError(f'expected 16UC1 depth, got {message.encoding}')
    if message.step < message.width * 2 or message.step % 2 or len(message.data) < message.height * message.step:
        raise ValueError('invalid depth Image step or data length')
    byte_order = '>' if message.is_bigendian else '<'
    return np.frombuffer(message.data, dtype=byte_order + 'u2',
                         count=message.height * message.step // 2).reshape(
        message.height, message.step // 2)[:, :message.width]


class ZmqCameraSender(Node):
    """Publish independent JPEG RGB frames and optional lossless 16-bit depth frames."""

    def __init__(self):
        super().__init__('zmq_camera_sender')
        self._declare_parameters()
        self.source = self.get_parameter('source').value
        self.camera_mode = self.get_parameter('camera_mode').value
        self.frame_id = self.get_parameter('frame_id').value
        self.sequence = 0
        self.sent_messages = 0
        self.dropped_messages = 0
        self.capture = None
        self.last_open_attempt = 0.0

        if self.source not in ('v4l2', 'ros'):
            raise ValueError("source must be 'v4l2' or 'ros'")
        if self.camera_mode not in ('rgb', 'rgbd'):
            raise ValueError("camera_mode must be 'rgb' or 'rgbd'")
        if self.source == 'v4l2' and self.camera_mode != 'rgb':
            raise ValueError(
                'v4l2 mode is RGB-only. Use source:=ros with a calibrated RGB-D driver; '
                'the package will not invent depth from a webcam.')
        if self.camera_mode == 'rgbd' and not all(
                self.get_parameter(name).value for name in
                ('ros_depth_topic', 'ros_depth_camera_info_topic')):
            raise ValueError('RGB-D needs explicit aligned depth image and CameraInfo topics')
        if not 1 <= self.get_parameter('jpeg_quality').value <= 100:
            raise ValueError('jpeg_quality must be in [1, 100]')
        if self.get_parameter('fps').value <= 0:
            raise ValueError('fps must be positive')
        if self.get_parameter('reopen_interval_s').value <= 0:
            raise ValueError('reopen_interval_s must be positive')

        self.zmq_context = zmq.Context.instance()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, self.get_parameter('send_hwm').value)
        self.socket.setsockopt(zmq.LINGER, 0)
        endpoint = self.get_parameter('bind_endpoint').value
        self.socket.bind(endpoint)
        self.get_logger().info(f'ZeroMQ publisher bound to {endpoint} ({self.source}, {self.camera_mode})')

        self.status_timer = self.create_timer(
            self.get_parameter('status_period_s').value, self._publish_status)
        if self.source == 'v4l2':
            self._start_v4l2_capture()
        else:
            self._start_ros_relay()

    def _declare_parameters(self):
        self.declare_parameter('source', 'v4l2')
        self.declare_parameter('camera_mode', 'rgb')
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('reopen_interval_s', 2.0)
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        self.declare_parameter('bind_endpoint', 'tcp://0.0.0.0:5555')
        self.declare_parameter('rgb_topic', 'camera.rgb.v1')
        self.declare_parameter('depth_topic', 'camera.depth.v1')
        self.declare_parameter('camera_info_topic', 'camera.info.v1')
        self.declare_parameter('status_topic', 'camera.status.v1')
        self.declare_parameter('send_hwm', 2)
        self.declare_parameter('status_period_s', 1.0)
        self.declare_parameter('ros_rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('ros_depth_topic', '')
        self.declare_parameter('ros_camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('ros_depth_camera_info_topic', '')

    def _start_v4l2_capture(self):
        self.capture_timer = self.create_timer(1.0 / self.get_parameter('fps').value,
                                               self._capture_webcam_frame)
        self._open_webcam()

    def _open_webcam(self):
        self.last_open_attempt = time.monotonic()
        device = self.get_parameter('device').value
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            self.get_logger().warning(f'Cannot open webcam {device}; retrying')
            return
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.get_parameter('width').value)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.get_parameter('height').value)
        capture.set(cv2.CAP_PROP_FPS, self.get_parameter('fps').value)
        self.capture = capture

        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f'Opened {device} at {actual_width}x{actual_height}, requested {actual_fps:.1f} FPS')

    def _start_ros_relay(self):
        # Vendor drivers retain calibration and capture stamps, so this adapter stays
        # independent of whether the later RGB-D camera is RealSense, ZED, or OAK-D.
        self.color_info = None
        self.depth_info = None
        self.create_subscription(
            Image, self.get_parameter('ros_rgb_topic').value,
            self._relay_rgb_image, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter('ros_camera_info_topic').value,
            self._relay_color_camera_info, qos_profile_sensor_data)
        if self.camera_mode == 'rgbd':
            self.create_subscription(
                Image, self.get_parameter('ros_depth_topic').value,
                self._relay_depth_image, qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, self.get_parameter('ros_depth_camera_info_topic').value,
                self._relay_depth_camera_info, qos_profile_sensor_data)

    def _capture_webcam_frame(self):
        if self.capture is None:
            if time.monotonic() - self.last_open_attempt >= self.get_parameter('reopen_interval_s').value:
                self._open_webcam()
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.dropped_messages += 1
            self.capture.release()
            self.capture = None
            self.last_open_attempt = time.monotonic()
            self.get_logger().warning('Webcam capture failed; retrying the USB camera')
            return
        self._send_rgb(frame, time.time_ns(), self.frame_id)

    def _relay_rgb_image(self, message):
        try:
            image = ros_rgb_to_bgr(message)
            if self.color_info is not None:
                self._send_camera_info('color', self.color_info)
            self._send_rgb(image, time_to_ns(message.header.stamp), message.header.frame_id)
        except Exception as error:  # cv_bridge errors differ among camera drivers.
            self.dropped_messages += 1
            self.get_logger().error(f'Cannot relay RGB image: {error}')

    def _relay_depth_image(self, message):
        try:
            depth = ros_depth_to_uint16(message)
            if self.depth_info is not None:
                self._send_camera_info('depth', self.depth_info)
            ok, encoded = cv2.imencode('.png', depth)
            if not ok:
                raise RuntimeError('OpenCV failed to encode depth PNG')
            header = self._header(message.header.frame_id, time_to_ns(message.header.stamp),
                                  depth.shape[1], depth.shape[0], '16UC1; png')
            self._send(self.get_parameter('depth_topic').value, header, encoded.tobytes())
        except Exception as error:
            self.dropped_messages += 1
            self.get_logger().error(f'Cannot relay depth image: {error}')

    def _relay_color_camera_info(self, message):
        self.color_info = message
        self._send_camera_info('color', message)

    def _relay_depth_camera_info(self, message):
        self.depth_info = message
        self._send_camera_info('depth', message)

    def _send_camera_info(self, stream, message):
        header = {
            'version': PROTOCOL_VERSION,
            'stream': stream,
            'camera_info': camera_info_to_dict(message),
        }
        self._send(self.get_parameter('camera_info_topic').value, header, b'')

    def _send_rgb(self, image, stamp_ns, frame_id):
        quality = self.get_parameter('jpeg_quality').value
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            self.dropped_messages += 1
            self.get_logger().error('OpenCV failed to encode RGB JPEG')
            return
        header = self._header(frame_id or self.frame_id, stamp_ns,
                              image.shape[1], image.shape[0], 'bgr8; jpeg')
        self._send(self.get_parameter('rgb_topic').value, header, encoded.tobytes())

    def _header(self, frame_id, stamp_ns, width, height, encoding):
        header = {
            'version': PROTOCOL_VERSION,
            'sequence': self.sequence,
            'capture_time_ns': stamp_ns,
            'frame_id': frame_id,
            'width': width,
            'height': height,
            'encoding': encoding,
        }
        self.sequence += 1
        return header

    def _send(self, topic, header, payload):
        try:
            self.socket.send_multipart([
                topic.encode('utf-8'),
                json.dumps(header, separators=(',', ':')).encode('utf-8'),
                payload,
            ], flags=zmq.NOBLOCK)
            self.sent_messages += 1
        except zmq.Again:
            # Live video deliberately drops under pressure instead of accumulating latency.
            self.dropped_messages += 1
        except zmq.ZMQError as error:
            self.dropped_messages += 1
            self.get_logger().error(f'ZeroMQ send failed: {error}')

    def _publish_status(self):
        header = {
            'version': PROTOCOL_VERSION,
            'capture_time_ns': time.time_ns(),
            'source': self.source,
            'camera_mode': self.camera_mode,
            'sent_messages': self.sent_messages,
            'dropped_messages': self.dropped_messages,
        }
        self._send(self.get_parameter('status_topic').value, header, b'')

    def destroy_node(self):
        if self.capture is not None:
            self.capture.release()
        self.socket.close(linger=0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ZmqCameraSender()
        rclpy.spin(node)
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f'zmq_camera_sender failed: {error}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

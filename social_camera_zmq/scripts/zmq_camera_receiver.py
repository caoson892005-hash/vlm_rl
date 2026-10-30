#!/usr/bin/env python3
"""Restore standard ROS camera topics from the Pi's ZeroMQ image stream."""

import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import yaml
import zmq


def set_stamp(stamp, nanoseconds):
    stamp.sec, stamp.nanosec = divmod(int(nanoseconds), 1_000_000_000)


def camera_info_from_dict(data):
    info = CameraInfo()
    info.width = int(data['width'])
    info.height = int(data['height'])
    info.distortion_model = data.get('distortion_model', '')
    info.d = data.get('d', [])
    info.k = data.get('k', [0.0] * 9)
    info.r = data.get('r', [0.0] * 9)
    info.p = data.get('p', [0.0] * 12)
    info.binning_x = int(data.get('binning_x', 0))
    info.binning_y = int(data.get('binning_y', 0))
    roi = data.get('roi', {})
    info.roi.x_offset = int(roi.get('x_offset', 0))
    info.roi.y_offset = int(roi.get('y_offset', 0))
    info.roi.width = int(roi.get('width', 0))
    info.roi.height = int(roi.get('height', 0))
    info.roi.do_rectify = bool(roi.get('do_rectify', False))
    return info


def load_calibration(path):
    """Read the usual camera_calibration YAML exported by ROS camera tools."""
    with Path(path).expanduser().open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    return camera_info_from_dict({
        'width': data['image_width'],
        'height': data['image_height'],
        'distortion_model': data.get('distortion_model', ''),
        'd': data.get('distortion_coefficients', {}).get('data', []),
        'k': data.get('camera_matrix', {}).get('data', [0.0] * 9),
        'r': data.get('rectification_matrix', {}).get('data', [0.0] * 9),
        'p': data.get('projection_matrix', {}).get('data', [0.0] * 12),
    })


class ZmqCameraReceiver(Node):
    """Publish current RGB frames, or synchronized RGB-D pairs when required."""

    def __init__(self):
        super().__init__('zmq_camera_receiver')
        self._declare_parameters()
        self.require_depth = bool(self.get_parameter('require_depth').value)
        if self.require_depth and not self.get_parameter('receive_depth').value:
            raise ValueError('require_depth needs receive_depth:=true')
        self.color_info = self._load_local_calibration()
        self.depth_info = None
        self.pending_rgb = None
        self.pending_depth = None
        self.last_pair = None
        self.last_pair_time = None
        self.last_rgb_time = None
        self.last_warning_time = 0.0

        self.rgb_pub = self.create_publisher(
            Image, self.get_parameter('output_rgb_topic').value, qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(
            Image, self.get_parameter('output_depth_topic').value, qos_profile_sensor_data)
        self.color_info_pub = self.create_publisher(
            CameraInfo, self.get_parameter('output_camera_info_topic').value,
            qos_profile_sensor_data)
        self.depth_info_pub = self.create_publisher(
            CameraInfo, self.get_parameter('output_depth_camera_info_topic').value,
            qos_profile_sensor_data)

        self.zmq_context = zmq.Context.instance()
        self.socket = self.zmq_context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, self.get_parameter('receive_hwm').value)
        self.socket.setsockopt(zmq.LINGER, 0)
        for parameter in ('rgb_topic', 'camera_info_topic', 'status_topic'):
            self.socket.setsockopt(zmq.SUBSCRIBE,
                                   self.get_parameter(parameter).value.encode('utf-8'))
        if self.get_parameter('receive_depth').value:
            self.socket.setsockopt(
                zmq.SUBSCRIBE, self.get_parameter('depth_topic').value.encode('utf-8'))
        endpoint = self.get_parameter('connect_endpoint').value
        self.socket.connect(endpoint)
        self.get_logger().info(f'Connected to {endpoint}')
        self.create_timer(max(0.001, self.get_parameter('poll_period_ms').value / 1000),
                          self._poll)
        self.create_timer(0.5, self._warn_if_stale)

    def _declare_parameters(self):
        self.declare_parameter('connect_endpoint', 'tcp://192.168.1.50:5555')
        self.declare_parameter('rgb_topic', 'camera.rgb.v1')
        self.declare_parameter('depth_topic', 'camera.depth.v1')
        self.declare_parameter('camera_info_topic', 'camera.info.v1')
        self.declare_parameter('status_topic', 'camera.status.v1')
        self.declare_parameter('receive_hwm', 2)
        self.declare_parameter('poll_period_ms', 5)
        self.declare_parameter('stale_timeout_s', 1.0)
        self.declare_parameter('max_pair_delta_s', 0.15)
        self.declare_parameter('max_payload_bytes', 8 * 1024 * 1024)
        self.declare_parameter('receive_depth', False)
        self.declare_parameter('require_depth', False)
        self.declare_parameter('output_rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('output_depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('output_camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_depth_camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('camera_info_file', '')
        self.declare_parameter('frame_id', 'camera_color_optical_frame')

    def _load_local_calibration(self):
        path = self.get_parameter('camera_info_file').value
        if not path:
            self.get_logger().warning('No webcam calibration file; CameraInfo is absent '
                                      'until a sender provides one.')
            return None
        info = load_calibration(path)
        self.get_logger().info(f'Loaded camera calibration: {path}')
        return info

    def _poll(self):
        # Keep only the newest packet of each type. PUB/SUB is for live frames;
        # replaying a backlog would make person positions stale during motion.
        latest = {}
        for _ in range(32):
            try:
                parts = self.socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) != 3 or len(parts[2]) > self.get_parameter('max_payload_bytes').value:
                continue
            try:
                topic = parts[0].decode('utf-8')
                header = json.loads(parts[1].decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if header.get('version') != 1:
                continue
            if topic == self.get_parameter('camera_info_topic').value:
                stream = header.get('stream')
                latest[('info', stream)] = (header, parts[2])
            else:
                latest[topic] = (header, parts[2])

        for key, (header, _) in latest.items():
            if isinstance(key, tuple) and key[0] == 'info':
                self._receive_info(header)
        rgb = latest.get(self.get_parameter('rgb_topic').value)
        depth = latest.get(self.get_parameter('depth_topic').value)
        if rgb is not None:
            self.pending_rgb = self._decode(rgb, 'rgb')
            if self.pending_rgb is not None:
                self.last_rgb_time = time.monotonic()
        if depth is not None:
            self.pending_depth = self._decode(depth, 'depth')

        if self.require_depth:
            self._publish_rgbd_pair()
        else:
            if rgb is not None and self.pending_rgb is not None:
                self._publish(self.pending_rgb, self.color_info,
                              self.rgb_pub, self.color_info_pub)
            if depth is not None and self.pending_depth is not None:
                self._publish(self.pending_depth, self.depth_info,
                              self.depth_pub, self.depth_info_pub)

    def _receive_info(self, header):
        try:
            info = camera_info_from_dict(header['camera_info'])
        except (KeyError, TypeError, ValueError) as error:
            self.get_logger().warning(f'Invalid CameraInfo packet: {error}')
            return
        if header.get('stream') == 'depth':
            self.depth_info = info
        elif header.get('stream') == 'color':
            self.color_info = info

    def _decode(self, packet, stream):
        header, payload = packet
        expected = 'bgr8; jpeg' if stream == 'rgb' else '16UC1; png'
        if header.get('encoding') != expected:
            return None
        flags = cv2.IMREAD_COLOR if stream == 'rgb' else cv2.IMREAD_UNCHANGED
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
        if image is None:
            return None
        if stream == 'depth' and (image.dtype != np.uint16 or image.ndim != 2):
            return None
        if image.shape[1] != header.get('width') or image.shape[0] != header.get('height'):
            return None
        stamp_ns = int(header.get('capture_time_ns', 0))
        if stamp_ns <= 0:
            return None
        return image, header

    def _publish_rgbd_pair(self):
        if self.pending_rgb is None or self.pending_depth is None:
            return
        if self.color_info is None or self.depth_info is None:
            return
        rgb, rgb_header = self.pending_rgb
        depth, depth_header = self.pending_depth
        rgb_stamp = int(rgb_header['capture_time_ns'])
        depth_stamp = int(depth_header['capture_time_ns'])
        pair = rgb_stamp, depth_stamp
        if pair == self.last_pair:
            return
        if self.last_pair is not None and (rgb_stamp <= self.last_pair[0] or
                                           depth_stamp <= self.last_pair[1]):
            return
        if abs(rgb_stamp - depth_stamp) > self.get_parameter('max_pair_delta_s').value * 1e9:
            return
        # Depth must be registered to RGB pixels; equal dimensions are necessary
        # but the camera driver must also perform the geometric alignment.
        if rgb.shape[:2] != depth.shape or (self.color_info.width != rgb.shape[1] or
                                           self.color_info.height != rgb.shape[0]):
            self.get_logger().warning('RGB and aligned depth dimensions/calibration differ')
            return
        if self.depth_info.width != depth.shape[1] or self.depth_info.height != depth.shape[0]:
            return
        self._publish((depth, depth_header), self.depth_info,
                      self.depth_pub, self.depth_info_pub)
        self._publish((rgb, rgb_header), self.color_info,
                      self.rgb_pub, self.color_info_pub)
        self.last_pair = pair
        self.last_pair_time = time.monotonic()

    @staticmethod
    def _publish(frame, info, image_publisher, info_publisher):
        image, header = frame
        message = Image()
        message.header.frame_id = header.get('frame_id', '')
        set_stamp(message.header.stamp, header['capture_time_ns'])
        message.width = image.shape[1]
        message.height = image.shape[0]
        message.encoding = 'bgr8' if image.ndim == 3 else '16UC1'
        message.is_bigendian = False
        message.step = int(image.strides[0])
        message.data = np.ascontiguousarray(image).tobytes()
        if image.ndim == 2:
            message.step = image.shape[1] * 2
        else:
            message.step = image.shape[1] * 3
        image_publisher.publish(message)
        if info is not None and info.width == message.width and info.height == message.height:
            info.header = message.header
            info_publisher.publish(info)

    def _warn_if_stale(self):
        if self.last_rgb_time is None:
            return
        now = time.monotonic()
        age = now - self.last_rgb_time
        timeout = self.get_parameter('stale_timeout_s').value
        pair_age = (now - self.last_pair_time if self.last_pair_time is not None
                    else now - self.last_rgb_time)
        if now - self.last_warning_time <= timeout:
            return
        if age > timeout:
            self.last_warning_time = now
            self.get_logger().warning(f'RGB stream stale for {age:.1f}s; old frames are not replayed')
        elif self.require_depth and pair_age > timeout:
            self.last_warning_time = now
            self.get_logger().warning('RGB-D pair unavailable: check aligned depth, '
                                      'CameraInfo and capture timestamps')

    def destroy_node(self):
        self.socket.close(linger=0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ZmqCameraReceiver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is None:
            print(f'zmq_camera_receiver failed: {error}')
        else:
            node.get_logger().fatal(str(error))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

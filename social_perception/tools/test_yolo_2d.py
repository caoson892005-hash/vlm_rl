#!/usr/bin/env python3
"""
Test YOLOv8 2D detection on live camera topic (/camera/color/image_raw).

Designed for 2D webcams streaming without Depth / CameraInfo.
Subscribes with sensor_data QoS (Best Effort) to match ZMQ camera stream.
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO


class Yolo2DTestNode(Node):
    """Subscribes to 2D camera topic, runs YOLO inference, and displays/publishes results."""

    def __init__(self, model_path: str, conf: float, input_topic: str, output_topic: str, show_window: bool):
        super().__init__('yolo_2d_test_node')

        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path)
        self.conf = conf
        self.show_window = show_window

        # Publisher for annotated image
        self.annotated_pub = self.create_publisher(Image, output_topic, 10)

        # Subscriber using sensor_data QoS (compatible with Best Effort ZMQ receiver)
        self.sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.last_time = time.time()
        self.fps = 0.0
        self.get_logger().info(f'Listening to: {input_topic}')
        self.get_logger().info(f'Publishing debug to: {output_topic}')
        self.get_logger().info('Ready! Press "q" in the display window or Ctrl+C in terminal to stop.')

    @staticmethod
    def imgmsg_to_cv2(msg: Image) -> np.ndarray:
        """Convert ROS Image message to OpenCV BGR format without requiring cv_bridge."""
        channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4}.get(msg.encoding)
        if channels is None:
            raise ValueError(f'Unsupported encoding: {msg.encoding}')

        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)

        if msg.encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if msg.encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if msg.encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    @staticmethod
    def cv2_to_imgmsg(image: np.ndarray, header) -> Image:
        """Convert OpenCV BGR image back to ROS Image message."""
        output = Image()
        output.header = header
        output.height, output.width = image.shape[:2]
        output.encoding = 'bgr8'
        output.is_bigendian = 0
        output.step = output.width * 3
        output.data = image.tobytes()
        return output

    def image_callback(self, msg: Image):
        # Calculate processing FPS
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        try:
            cv_img = self.imgmsg_to_cv2(msg)
        except Exception as err:
            self.get_logger().error(f'Failed to decode image: {err}', throttle_duration_sec=2.0)
            return

        # Run YOLO inference
        results = self.model(cv_img, conf=self.conf, verbose=False)
        annotated = results[0].plot()

        # Overlay FPS on the top-left corner
        cv2.putText(
            annotated,
            f'FPS: {self.fps:.1f}',
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Publish annotated image
        annotated_msg = self.cv2_to_imgmsg(annotated, msg.header)
        self.annotated_pub.publish(annotated_msg)

        # Show OpenCV window if GUI support is available
        if self.show_window:
            try:
                cv2.imshow('YOLOv8 2D Detection', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info('Quit requested by user.')
                    rclpy.shutdown()
            except cv2.error:
                self.get_logger().warning(
                    'OpenCV GUI is not supported (headless mode). Window display disabled. '
                    'View stream via: ros2 run rqt_image_view rqt_image_view',
                    throttle_duration_sec=10.0
                )
                self.show_window = False


def main():
    parser = argparse.ArgumentParser(description='Test YOLOv8 on 2D camera topic')
    default_model = '/home/hung/ninorobot2/yolov8n.pt'
    if not os.path.exists(default_model):
        default_model = 'yolov8n.pt'

    parser.add_argument('--model', default=default_model, help='Path to YOLO weights (.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--input', default='/camera/color/image_raw', help='ROS input image topic')
    parser.add_argument('--output', default='/social_perception/annotated_image', help='ROS output debug topic')
    parser.add_argument('--no-window', action='store_true', help='Disable GUI window display')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = Yolo2DTestNode(
        model_path=args.model,
        conf=args.conf,
        input_topic=args.input,
        output_topic=args.output,
        show_window=not args.no_window
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

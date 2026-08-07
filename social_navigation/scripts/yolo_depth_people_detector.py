#!/usr/bin/env python3
"""Detect people in RGB images and localize them with registered depth."""

import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from social_navigation.msg import People, Person


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def rotate_vector(vector, quaternion):
    """Rotate a vector by a geometry_msgs Quaternion."""
    x, y, z = vector
    qx, qy, qz, qw = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + qy * tz - qz * ty,
        y + qw * ty + qz * tx - qx * tz,
        z + qw * tz + qx * ty - qy * tx,
    )


class YoloDepthPeopleDetector(Node):
    def __init__(self):
        super().__init__('yolo_depth_people_detector')
        self.declare_parameter('model_path', 'yolo11n.pt')
        self.declare_parameter('rgb_topic', '/overhead_camera/color/image_raw')
        self.declare_parameter('depth_topic', '/overhead_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/overhead_camera/color/camera_info')
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('maximum_depth_age', 0.15)
        self.declare_parameter('minimum_depth', 0.2)
        self.declare_parameter('maximum_depth', 20.0)
        self.declare_parameter('people_topic', '/yolo/people')

        model_path = str(self.get_parameter('model_path').value)
        if not Path(model_path).is_file():
            self.get_logger().warn(
                f'Model {model_path} is not a local file; Ultralytics may try to download it')
        self.model = YOLO(model_path)
        self.confidence = float(self.get_parameter('confidence').value)
        self.image_size = int(self.get_parameter('image_size').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.max_depth_age = float(self.get_parameter('maximum_depth_age').value)
        self.min_depth = float(self.get_parameter('minimum_depth').value)
        self.max_depth = float(self.get_parameter('maximum_depth').value)

        self.depth_msg = None
        self.camera_info = None
        self.rgb_frames = 0
        self.depth_frames = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.next_track_id = 1
        self.tracks = {}

        rgb_topic = str(self.get_parameter('rgb_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        info_topic = str(self.get_parameter('camera_info_topic').value)
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, depth_topic, self.depth_callback, camera_qos)
        self.create_subscription(CameraInfo, info_topic, self.info_callback, camera_qos)
        self.create_subscription(Image, rgb_topic, self.rgb_callback, camera_qos)
        self.detections_pub = self.create_publisher(
            Detection2DArray, '/people/detections_2d', 10)
        self.annotated_pub = self.create_publisher(Image, '/people/yolo_image', 10)
        self.depth_visual_pub = self.create_publisher(
            Image, '/people/depth_visualization', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/people/yolo_markers', 10)
        self.people_pub = self.create_publisher(
            People, str(self.get_parameter('people_topic').value), 10)
        self.get_logger().info(
            f'YOLO depth detector ready: RGB={rgb_topic}, depth={depth_topic}, '
            f'target={self.target_frame}, model={model_path}')
        self.create_timer(5.0, self.report_status)

    def depth_callback(self, message):
        self.depth_msg = message
        self.depth_frames += 1

    def info_callback(self, message):
        self.camera_info = message

    def report_status(self):
        self.get_logger().info(
            f'Camera input status: RGB frames={self.rgb_frames}, '
            f'depth frames={self.depth_frames}, '
            f'CameraInfo={"yes" if self.camera_info is not None else "no"}')

    @staticmethod
    def rgb_array(message):
        channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4}.get(message.encoding)
        if channels is None:
            raise ValueError(f'unsupported RGB encoding: {message.encoding}')
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = rows[:, :message.width * channels].reshape(message.height, message.width, channels)
        if message.encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if message.encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if message.encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    @staticmethod
    def depth_array(message):
        if message.encoding == '32FC1':
            dtype, scale = np.dtype('>f4' if message.is_bigendian else '<f4'), 1.0
        elif message.encoding in ('16UC1', 'mono16'):
            dtype, scale = np.dtype('>u2' if message.is_bigendian else '<u2'), 0.001
        else:
            raise ValueError(f'unsupported depth encoding: {message.encoding}')
        row_items = message.step // dtype.itemsize
        rows = np.frombuffer(message.data, dtype=dtype).reshape(message.height, row_items)
        return rows[:, :message.width].astype(np.float32) * scale

    def median_depth(self, depth, x1, y1, x2, y2):
        # Use the central torso region; borders often contain the floor/wall.
        width, height = x2 - x1, y2 - y1
        rx1, rx2 = int(x1 + 0.30 * width), int(x2 - 0.30 * width)
        ry1, ry2 = int(y1 + 0.20 * height), int(y1 + 0.70 * height)
        roi = depth[max(0, ry1):min(depth.shape[0], ry2),
                    max(0, rx1):min(depth.shape[1], rx2)]
        valid = roi[np.isfinite(roi) & (roi >= self.min_depth) & (roi <= self.max_depth)]
        return float(np.median(valid)) if valid.size >= 8 else None

    def point_in_target(self, u, v, depth, source_frame, stamp):
        info = self.camera_info
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        if fx <= 0.0 or fy <= 0.0:
            return None
        optical = ((u - cx) * depth / fx, (v - cy) * depth / fy, depth)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, source_frame, rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(f'TF {source_frame} -> {self.target_frame}: {error}',
                                   throttle_duration_sec=2.0)
            return None
        rotated = rotate_vector(optical, transform.transform.rotation)
        translation = transform.transform.translation
        return (rotated[0] + translation.x,
                rotated[1] + translation.y,
                rotated[2] + translation.z)

    def assign_track(self, point, used_ids):
        best_id, best_distance = None, 1.0
        for track_id, previous in self.tracks.items():
            if track_id in used_ids:
                continue
            distance = math.hypot(point[0] - previous[0], point[1] - previous[1])
            if distance < best_distance:
                best_id, best_distance = track_id, distance
        if best_id is None:
            best_id = self.next_track_id
            self.next_track_id += 1
        return best_id

    def rgb_callback(self, rgb_msg):
        self.rgb_frames += 1
        if self.depth_msg is None or self.camera_info is None:
            self.get_logger().warn('Waiting for depth image and CameraInfo',
                                   throttle_duration_sec=2.0)
            return
        if abs(stamp_seconds(rgb_msg.header.stamp) -
               stamp_seconds(self.depth_msg.header.stamp)) > self.max_depth_age:
            self.get_logger().warn('RGB and depth timestamps are not synchronized',
                                   throttle_duration_sec=2.0)
            return
        try:
            image = self.rgb_array(rgb_msg)
            depth_image = self.depth_array(self.depth_msg)
        except (ValueError, TypeError) as error:
            self.get_logger().error(str(error), throttle_duration_sec=2.0)
            return

        finite = np.isfinite(depth_image) & (depth_image >= self.min_depth) & (
            depth_image <= self.max_depth)
        normalized = np.zeros(depth_image.shape, dtype=np.uint8)
        if np.any(finite):
            near = float(np.percentile(depth_image[finite], 2.0))
            far = float(np.percentile(depth_image[finite], 98.0))
            span = max(0.1, far - near)
            normalized[finite] = np.clip(
                255.0 * (depth_image[finite] - near) / span, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        depth_output = Image()
        depth_output.header = self.depth_msg.header
        depth_output.height, depth_output.width = depth_color.shape[:2]
        depth_output.encoding = 'bgr8'
        depth_output.is_bigendian = 0
        depth_output.step = depth_output.width * 3
        depth_output.data = depth_color.tobytes()
        self.depth_visual_pub.publish(depth_output)

        result = self.model.predict(
            source=image, classes=[0], conf=self.confidence,
            imgsz=self.image_size, verbose=False)[0]
        detections = Detection2DArray()
        detections.header = rgb_msg.header
        people = People()
        people.header.stamp = rgb_msg.header.stamp
        people.header.frame_id = self.target_frame
        markers = MarkerArray()
        annotated = image.copy()
        new_tracks, used_ids = {}, set()

        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
                score = float(box.conf[0].cpu())
                detection = Detection2D()
                detection.header = rgb_msg.header
                detection.bbox.center.position.x = (x1 + x2) * 0.5
                detection.bbox.center.position.y = (y1 + y2) * 0.5
                detection.bbox.size_x = x2 - x1
                detection.bbox.size_y = y2 - y1
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = 'person'
                hypothesis.hypothesis.score = score
                detection.results.append(hypothesis)
                detections.detections.append(detection)

                z = self.median_depth(depth_image, x1, y1, x2, y2)
                point = None if z is None else self.point_in_target(
                    (x1 + x2) * 0.5, y2 - 0.08 * (y2 - y1), z,
                    self.depth_msg.header.frame_id, rgb_msg.header.stamp)
                color = (0, 200, 255) if point is not None else (0, 0, 255)
                label = f'person {score:.2f}' + (f' {z:.1f}m' if z is not None else ' no-depth')
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if point is None:
                    continue
                track_id = self.assign_track(point, used_ids)
                used_ids.add(track_id)
                new_tracks[track_id] = point
                detection.id = f'yolo_person_{track_id}'
                person = Person()
                person.id = detection.id
                person.pose.position.x, person.pose.position.y, person.pose.position.z = point
                person.pose.orientation.w = 1.0
                people.people.append(person)
                marker = Marker()
                marker.header = people.header
                marker.ns = 'yolo_people_3d'
                marker.id = track_id
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.pose = person.pose
                marker.pose.position.z = max(0.9, point[2])
                marker.scale.x, marker.scale.y, marker.scale.z = 0.55, 0.55, 1.8
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.1, 0.55
                marker.lifetime = Duration(seconds=0.4).to_msg()
                markers.markers.append(marker)

        self.tracks = new_tracks
        self.detections_pub.publish(detections)
        self.people_pub.publish(people)
        self.markers_pub.publish(markers)
        output = Image()
        output.header = rgb_msg.header
        output.height, output.width = annotated.shape[:2]
        output.encoding = 'bgr8'
        output.is_bigendian = 0
        output.step = output.width * 3
        output.data = annotated.tobytes()
        self.annotated_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthPeopleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

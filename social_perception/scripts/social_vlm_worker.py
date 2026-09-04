#!/usr/bin/env python3
"""Answer VLM questions for a perception node running on another machine.

This is the workstation half of the split pipeline. The robot carries the
camera and runs social_vlm_perception.py with `vlm_remote: true`: YOLO, depth,
tracking and the social regions all stay there, in the frame the costmap reads.
Only the crop of a candidate pair crosses the network, and only the model's
sentence comes back.

    robot        /social_perception/vlm_request   ->  workstation (this node)
    workstation  /social_perception/vlm_response  ->  robot

Deliberately stateless. It holds no track ids, no cache and no idea which pair
it just looked at, because every rule about what an answer means already lives
in the perception node. Restarting this node therefore costs one inference, not
a rebuilt world model.

    ros2 launch social_perception social_vlm_worker.launch.py \
        config_file:=<profile.yaml>
"""

import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool

from social_perception.msg import VlmRequest, VlmResponse


class SocialVlmWorker(Node):
    def __init__(self):
        # Same contract as the perception node: parameters come from the
        # profile on the command line, and a missing one raises here instead of
        # running on a default nobody can see in the config file.
        super().__init__('social_vlm_worker',
                         automatically_declare_parameters_from_overrides=True)

        self.log_results = bool(self.get_parameter('log_vlm_results').value)

        # Latched, so a perception node that starts later still learns the
        # model is up rather than waiting for the next state change.
        self.ready_pub = self.create_publisher(
            Bool, '/social_perception/vlm_worker_ready',
            QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.response_pub = self.create_publisher(
            VlmResponse, '/social_perception/vlm_response', 10)

        self.backend = None
        self.backend_error = None
        self.load_done = threading.Event()
        # Loading takes ~2 minutes and must not block the executor: requests
        # arriving meanwhile are answered with an explicit failure rather than
        # being silently dropped, which would cost the robot a full timeout
        # each time.
        threading.Thread(target=self.load_backend, name='vlm-load',
                         daemon=True).start()

        self.create_subscription(
            VlmRequest, '/social_perception/vlm_request', self.handle_request,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                       reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        self.get_logger().info(
            'VLM worker: đang nạp model, chưa nhận câu hỏi nào')

    def load_backend(self):
        # Imported here so `ros2 param` and a config error surface before the
        # ML import tree is paid for.
        from vlm_backend import VlmBackend

        adapter_value = str(self.get_parameter('vlm_adapter_path').value)
        adapter_path = self._resolve_file(adapter_value)
        merged_value = str(self.get_parameter('vlm_merged_path').value)
        merged_path = self._resolve_file(merged_value) if merged_value else None
        if merged_path is not None and not (merged_path / 'config.json').is_file():
            self.get_logger().warn(
                f'Ignoring vlm_merged_path {merged_value}: no config.json there')
            merged_path = None
        if merged_path is None and (
                adapter_path is None
                or not (adapter_path / 'adapter_config.json').is_file()):
            self.backend_error = f'VLM adapter not found at {adapter_value}'
            self.get_logger().error(self.backend_error)
            self.load_done.set()
            return
        try:
            self.backend = VlmBackend(
                adapter_path,
                str(self.get_parameter('vlm_base_model').value),
                bool(self.get_parameter('vlm_load_in_4bit').value),
                bool(self.get_parameter('vlm_require_cuda').value),
                int(self.get_parameter('vlm_max_new_tokens').value),
                int(self.get_parameter('vlm_min_pixels').value),
                int(self.get_parameter('vlm_max_pixels').value),
                self.get_logger(),
                # Nothing else on this machine competes for the model, but the
                # backend expects a lock and one request is served at a time.
                threading.Lock(),
                bool(self.get_parameter('vlm_force_answer_prefix').value),
                bool(self.get_parameter('vlm_offline').value),
                merged_path)
        except Exception as error:
            self.backend_error = f'{type(error).__name__}: {error}'
            self.get_logger().error(f'Unable to load VLM ({self.backend_error})')
            self.load_done.set()
            return
        self.load_done.set()
        self.ready_pub.publish(Bool(data=True))
        self.get_logger().info(
            'SẴN SÀNG: VLM đã nạp xong, đang đợi ảnh cắt từ robot')

    @staticmethod
    def _resolve_file(value):
        candidate = Path(value).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.extend([
                Path.cwd() / candidate,
                Path(__file__).resolve().parents[2] / candidate,
            ])
            try:
                candidates.append(Path(get_package_share_directory(
                    'social_perception')) / candidate)
            except Exception:  # Package index is unavailable before a build.
                pass
        for path in candidates:
            if path.exists():
                return path.resolve()
        return None

    def handle_request(self, request):
        response = VlmResponse()
        response.request_id = request.request_id
        response.header.stamp = self.get_clock().now().to_msg()

        if self.backend is None:
            response.failed = True
            response.error = (self.backend_error if self.load_done.is_set()
                              else 'VLM đang nạp, chưa trả lời được')
            self.response_pub.publish(response)
            return

        crop = cv2.imdecode(
            np.frombuffer(request.crop.data, dtype=np.uint8),
            cv2.IMREAD_COLOR)
        if crop is None or crop.size == 0:
            response.failed = True
            response.error = 'không giải mã được ảnh cắt'
            self.response_pub.publish(response)
            return

        pair = tuple(request.member_ids)
        self.get_logger().info(
            f'>>> VLM BẮT ĐẦU suy luận {pair} '
            f'(ảnh cắt {crop.shape[1]}x{crop.shape[0]}, '
            f'{len(request.crop.data) / 1024.0:.1f} KB qua mạng)')
        started = time.monotonic()
        try:
            response.raw_response = self.backend.infer(
                crop, request.prompt, pair)
        except Exception as error:
            response.failed = True
            response.error = f'{type(error).__name__}: {error}'
            self.get_logger().error(f'VLM inference failed for {pair}: '
                                    f'{response.error}')
            self.response_pub.publish(response)
            return
        elapsed = time.monotonic() - started
        prepare, lock, generate, tokens = getattr(
            self.backend, 'last_timing', (0.0, 0.0, 0.0, 0))
        response.prepare_seconds = float(prepare)
        response.lock_seconds = float(lock)
        response.generate_seconds = float(generate)
        response.token_count = int(tokens)
        self.response_pub.publish(response)
        if self.log_results:
            self.get_logger().info(
                f'<<< VLM XONG {pair}: {elapsed:.2f}s '
                f'[tien_xu_ly={prepare:.2f}s model={generate:.2f}s '
                f'tokens={tokens}] -> {response.raw_response!r}')


def main(args=None):
    rclpy.init(args=args)
    node = SocialVlmWorker()
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

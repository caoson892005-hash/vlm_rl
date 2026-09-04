#!/usr/bin/env python3
"""Localize people with RGB-D and use a Qwen2-VL LoRA to detect conversations."""

import array
import copy
import contextlib
import json
import math
import os
import queue
import re
import threading
import time
import unicodedata
import warnings
from enum import IntEnum
from pathlib import Path

# Keep third-party cosmetic warnings from hiding the two operator-facing
# messages that matter: model-load progress and the conversation decision.
warnings.filterwarnings('ignore', message='Unable to import Axes3D.*')
warnings.filterwarnings('ignore', message='`max_length` is ignored.*')
warnings.filterwarnings('ignore', message='You passed `quantization_config`.*')

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Empty
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from social_perception.msg import (
    Group,
    Groups,
    People,
    Person,
    VlmRequest,
    VlmResponse,
    TalkingInteraction,
    TalkingInteractions,
)


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def rotate_vector(vector, quaternion):
    """Rotate a 3-D vector by a geometry_msgs Quaternion."""
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


# cos(45 degrees), the half-angle of the head-on cone. Must stay equal to
# kHeadOnCosine in linorobot2_gazebo/src/animated_people_release.cpp: a policy
# trained on one definition of "walking towards the robot" and run on another
# is being lied to about the situation it is in.
HEAD_ON_COSINE = 0.7071067811865476


def fill_prediction(person, times, still_speed):
    """Block B's short-horizon trajectory, constant velocity.

    Mirrors FillPrediction in the Gazebo plugin. The first sample is t = 0.0
    and repeats the present pose, so block D reads the instant it charges the
    reward at out of the same array as the instants it rasterises.
    """
    vx = person.velocity.linear.x
    vy = person.velocity.linear.y
    moving = math.hypot(vx, vy) >= still_speed
    # Direction of travel once somebody is walking; whatever facing the
    # producer reported while they stand still. This node has no facing
    # estimate, so a standing person keeps orientation 0 -- see the comment on
    # facing in the detection loop.
    heading = (math.atan2(vy, vx) if moving
               else quaternion_yaw(person.pose.orientation))
    for horizon in times:
        pose = Pose()
        pose.position.x = person.pose.position.x + (vx * horizon if moving else 0.0)
        pose.position.y = person.pose.position.y + (vy * horizon if moving else 0.0)
        pose.orientation.z = math.sin(heading / 2.0)
        pose.orientation.w = math.cos(heading / 2.0)
        person.prediction_times.append(float(horizon))
        person.predicted_poses.append(pose)


def fill_relative_motion(person, robot, still_speed):
    """Block B's motion labels, measured against the robot.

    Mirrors FillRelativeMotion in the Gazebo plugin. `robot` is (x, y, yaw) in
    the same frame the person is expressed in.
    """
    vx = person.velocity.linear.x
    vy = person.velocity.linear.y
    speed = math.hypot(vx, vy)
    if speed < still_speed:
        person.radial_velocity = 0.0
        person.motion_type = 'static'
        return

    robot_x, robot_y, robot_yaw = robot
    dx = person.pose.position.x - robot_x
    dy = person.pose.position.y - robot_y
    # A person on top of the robot has no defined direction; the guard keeps
    # the division finite, and at that range there is nothing to decide.
    distance = max(math.hypot(dx, dy), 1e-6)
    radial = (vx * dx + vy * dy) / distance
    person.radial_velocity = float(radial)

    along = radial / speed
    if along < -HEAD_ON_COSINE:
        person.motion_type = 'head_on'
    elif along > HEAD_ON_COSINE:
        person.motion_type = 'receding'
    else:
        # Sideways component in the ROBOT's frame, where +y is its left.
        sideways = -math.sin(robot_yaw) * vx + math.cos(robot_yaw) * vy
        person.motion_type = 'left_to_right' if sideways < 0.0 else 'right_to_left'


def apply_scene_ruling(person, rulings):
    """Write block C's verdict onto a person, or leave the neutral default.

    Absent ruling means block C said nothing about this person, which is not
    the same as it saying "nothing social is happening" -- both land on an
    empty scene_type, but only the second is a measurement, and neither gives
    block D anything but its neutral region shape.
    """
    ruling = rulings.get(person.id)
    if ruling is None:
        return
    person.scene_type = ruling[0]
    person.scene_confidence = ruling[1]
    # Copied, not referenced: assigning a message field stores the object
    # itself, and this stamp belongs to a cached decision later frames read.
    person.scene_stamp = copy.deepcopy(ruling[2])


def quaternion_yaw(orientation):
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2))


def normalized_text(value):
    text = unicodedata.normalize('NFD', str(value).lower())
    text = ''.join(character for character in text
                   if unicodedata.category(character) != 'Mn')
    return text.replace('đ', 'd')


def talking_from_response(response):
    """Parse the model's Vietnamese/English JSON, conservatively."""
    keyed_values = []
    match = re.search(r'\{.*\}', response, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))

            def collect(item, key=''):
                if isinstance(item, dict):
                    for child_key, child in item.items():
                        collect(child, str(child_key))
                elif isinstance(item, list):
                    for child in item:
                        collect(child, key)
                elif any(token in normalized_text(key) for token in (
                        'talk', 'noi chuyen', 'tro chuyen', 'interaction', 'state')):
                    keyed_values.append((normalized_text(key), item))

            collect(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    # A dedicated talking field is authoritative. Generic state/interaction
    # fields in the training JSON may describe an individual, not the pair.
    talking_values = [value for key, value in keyed_values
                      if any(token in key for token in (
                          'talk', 'noi chuyen', 'tro chuyen'))]
    values = talking_values or [value for _, value in keyed_values] or [response]

    for value in values:
        if value is False or (isinstance(value, (int, float)) and value == 0):
            return False, 0.95
        text = normalized_text(value)
        if any(token in text for token in (
                'khong', 'false', 'not talking', 'no conversation', 'khong xac dinh')):
            return False, 0.90
        # A bare "no", mirroring the bare "co" accepted below. It cannot be a
        # substring test: "no" also sits inside "noi chuyen", which is a yes.
        if text.strip() in ('no', 'not'):
            return False, 0.90
    for value in values:
        if value is True or (isinstance(value, (int, float)) and value == 1):
            return True, 0.95
        text = normalized_text(value)
        if any(token in text for token in (
                'dang noi chuyen', 'dang tro chuyen', 'co noi chuyen',
                'talking', 'conversation', 'true', 'yes')) or text.strip() == 'co':
            return True, 0.90
    return False, 0.0


class RemoteVlmBackend:
    """Ask another machine the question a local VlmBackend would answer.

    Deliberately the same call signature as VlmBackend.infer, and blocking in
    the same way, because vlm_worker is what enforces every rule around the
    answer: which pair is worth asking about, what a newer camera frame
    invalidates, how much a contrary reply is worth against a region already on
    the costmap. None of that may be duplicated on the other machine, so only
    the sentence "run the model on this crop" crosses the network.

    The exchange is deliberately one question at a time. vlm_worker is a single
    thread that blocks here until the answer arrives, so a request id is enough
    to recognise a late reply to a question already given up on.
    """

    def __init__(self, node, request_pub, timeout, jpeg_quality):
        self.node = node
        self.request_pub = request_pub
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        # Read by the camera thread for the latency overlay, exactly as the
        # local backend's is. The three parts are measured on the workstation;
        # the gap between their sum and the round trip vlm_worker measures is
        # what the network and the JPEG cost.
        self.last_timing = (0.0, 0.0, 0.0, 0)
        self.lock = threading.Lock()
        self.pending_id = None
        self.pending_response = None
        self.answered = threading.Event()
        self.sequence = 0

    def handle_response(self, message):
        with self.lock:
            if message.request_id != self.pending_id:
                # A reply to a question this side already timed out on. Acting
                # on it would attach an answer to whatever pair is being asked
                # about now.
                return
            self.pending_response = message
        self.answered.set()

    def infer(self, bgr_image, prompt, pair=None):
        encoded_ok, encoded = cv2.imencode(
            '.jpg', bgr_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not encoded_ok:
            raise RuntimeError('failed to JPEG-encode the pair crop')
        request = VlmRequest()
        request.header.stamp = self.node.get_clock().now().to_msg()
        request.crop.header = request.header
        request.crop.format = 'jpeg'
        request.crop.data = encoded.tobytes()
        # The prompt travels with the crop so the wording lives in exactly one
        # config file. Two copies drift, and a drifted prompt is invisible: the
        # model keeps answering, just to a different question.
        request.prompt = prompt
        request.member_ids = [str(member) for member in (pair or ())]
        with self.lock:
            self.sequence += 1
            request.request_id = str(self.sequence)
            self.pending_id = request.request_id
            self.pending_response = None
        self.answered.clear()
        self.request_pub.publish(request)
        if not self.answered.wait(self.timeout):
            with self.lock:
                self.pending_id = None
            raise TimeoutError(
                f'no VLM answer within {self.timeout:.1f}s; is '
                'social_vlm_worker.py running on the workstation?')
        with self.lock:
            response = self.pending_response
            self.pending_id = None
        self.last_timing = (response.prepare_seconds, response.lock_seconds,
                            response.generate_seconds, response.token_count)
        if response.failed:
            raise RuntimeError(f'remote VLM failed: {response.error}')
        return response.raw_response


class SocialVlmPerception(Node):
    def __init__(self):
        # Every parameter comes from the profile passed on the command line
        # (social_vlm_perception.yaml for the simulation,
        # social_vlm_perception_real.yaml for the camera on the workstation).
        # Declaring defaults here as well meant the same 46 values existed in
        # two places, and they had already drifted apart: yolo_device read
        # 'cpu' here while both profiles run it on cuda:0. A parameter left out
        # of the profile now raises at startup instead of silently running on
        # a value nobody can see in the config file.
        super().__init__('social_vlm_perception',
                         automatically_declare_parameters_from_overrides=True)

        self.confidence = float(self.get_parameter('yolo_confidence').value)
        self.image_size = int(self.get_parameter('yolo_image_size').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.max_depth_age = float(self.get_parameter('maximum_depth_age').value)
        self.max_frame_age = float(self.get_parameter('maximum_frame_age').value)
        self.min_depth = float(self.get_parameter('minimum_depth').value)
        self.max_depth = float(self.get_parameter('maximum_depth').value)
        self.track_distance = float(self.get_parameter('tracking_max_distance').value)
        self.duplicate_distance = float(
            self.get_parameter('duplicate_merge_distance').value)
        self.track_timeout = float(self.get_parameter('tracking_timeout').value)
        self.reidentify_timeout = float(
            self.get_parameter('reidentify_timeout').value)
        self.minimum_track_hits = int(
            self.get_parameter('minimum_track_hits').value)
        # How many frames each live id has been SEEN in, as opposed to coasted.
        # Kept beside self.tracks rather than inside it so the count survives a
        # trip through lost_tracks: somebody who walks back into frame reclaims
        # their id and their confirmation with it, and is published again the
        # same frame instead of serving the waiting period twice.
        self.track_hits = {}
        self.velocity_alpha = float(self.get_parameter('velocity_smoothing').value)
        # Block B's prediction window. These have to agree with
        # observation.constraint_field.prediction_times in the social_rl config
        # that trained the policy, which is what sizes the CNN.
        horizon = float(self.get_parameter('prediction_horizon').value)
        step = float(self.get_parameter('prediction_dt').value)
        self.prediction_times = [round(index * step, 6) for index
                                 in range(int(round(horizon / step)) + 1)]
        # Below this a person is standing, not walking. Extrapolating tracker
        # noise on somebody standing still would drag their predicted position
        # metres down the corridor by the end of the horizon.
        self.prediction_still_speed = float(
            self.get_parameter('prediction_still_speed').value)
        # Whose pose radial_velocity and motion_type are measured against.
        # Looked up in target_frame, so the same TF chain the people already
        # travel through.
        self.robot_frame = str(self.get_parameter('robot_frame').value)
        self.yolo_device = str(self.get_parameter('yolo_device').value)
        self.vlm_enabled = bool(self.get_parameter('enable_vlm').value)
        # Where the model actually runs. false keeps it in this process, which
        # is what the simulation and the workstation-camera setup do. true
        # sends each crop to social_vlm_worker.py over
        # /social_perception/vlm_request, so a robot carrying the camera needs
        # ultralytics for YOLO but neither transformers nor a GPU.
        self.vlm_remote = bool(self.get_parameter('vlm_remote').value)
        self.vlm_request_timeout = float(
            self.get_parameter('vlm_request_timeout').value)
        # The crop is the model's only evidence, so this is a quality knob, not
        # just a bandwidth one. Compare answers before lowering it.
        self.vlm_crop_jpeg_quality = int(
            self.get_parameter('vlm_crop_jpeg_quality').value)
        self.vlm_interval = float(self.get_parameter('vlm_inference_interval').value)
        self.vlm_refresh_interval = float(
            self.get_parameter('vlm_refresh_interval').value)
        self.vlm_position_threshold = float(
            self.get_parameter('vlm_position_change_threshold').value)
        self.max_pair_distance = float(self.get_parameter('maximum_talking_distance').value)
        self.max_vlm_pairs = int(self.get_parameter('maximum_vlm_pairs').value)
        self.negatives_to_clear = max(
            1, int(self.get_parameter('negative_answers_to_clear').value))
        self.interaction_timeout = float(self.get_parameter('interaction_timeout').value)
        self.crop_margin = float(self.get_parameter('vlm_crop_margin').value)
        self.latency_overlay = bool(
            self.get_parameter('show_vlm_latency_overlay').value)
        self.log_vlm_results = bool(self.get_parameter('log_vlm_results').value)
        self.group_o_min_radius = float(
            self.get_parameter('group_o_space_min_radius').value)
        self.group_p_margin = float(self.get_parameter('group_p_space_margin').value)
        self.group_r_margin = float(self.get_parameter('group_r_space_margin').value)
        self.last_vlm_enqueue = 0.0
        # Serializes VLM inferences against each other. It deliberately does
        # NOT cover YOLO.
        #
        # It used to. With yolo_device: cuda:0 that made every RGB frame wait
        # out the whole of Qwen's generate(), measured at 11-13 s, so nothing
        # reached /social_perception/annotated_image for that long (the RViz
        # camera panel froze) and /people was published carrying the stamp of
        # the frame received before the wait -- 15 s stale, past the costmap's
        # TF buffer, which Nav2 reported as "SocialLayer transform failed:
        # Lookup would require extrapolation into the past".
        #
        # Letting the two share the GPU costs the VLM some throughput and buys
        # back a detector that never stops. YOLOv8n at imgsz 640 is ~50 MiB of
        # the 4 GiB card against Qwen's ~2.5 GiB, so this is contention for
        # compute, not for memory.
        self.ml_execution_lock = threading.Lock()

        yolo_path = self._resolve_file(str(self.get_parameter('yolo_model_path').value))
        if yolo_path is None:
            yolo_path = str(self.get_parameter('yolo_model_path').value)
            self.get_logger().warn(
                f'YOLO model {yolo_path} is not local; Ultralytics may download it')
        self.yolo = YOLO(str(yolo_path))
        self._warm_up_yolo()

        self.depth_msg = None
        self.camera_info = None
        self.rgb_frames = 0
        self.depth_frames = 0
        self.next_track_id = 1
        self.tracks = {}
        # Where tracks were last seen, so a person who drops out and comes
        # back keeps the identity every cached VLM decision is keyed on.
        self.lost_tracks = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state_lock = threading.Lock()
        self.latest_people = {}
        self.latest_header = None
        self.latest_observation_sequence = 0
        self.scene_generation = 0
        self.interaction_cache = {}
        self.vlm_inflight_pairs = {}
        self.last_vlm_scenes = {}
        self.last_logged_decisions = {}
        # Wall-clock accounting for a single inference, measured from the
        # moment the crop is handed to Qwen until the answer comes back. The
        # VLM thread writes it and the camera thread reads it to draw the
        # overlay, so it gets its own lock instead of sharing the busy scene
        # lock a long generation would then be holding up.
        self.latency_lock = threading.Lock()
        self.vlm_running_since = {}
        self.vlm_last_latency = None
        self.vlm_latency_count = 0
        self.vlm_latency_total = 0.0
        self.last_interactions_signature = None
        self.last_interactions_publish = 0.0
        self.work_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.vlm_thread = None

        rgb_topic = str(self.get_parameter('rgb_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        info_topic = str(self.get_parameter('camera_info_topic').value)
        # YOLO inference is intentionally isolated from the lightweight depth
        # callbacks. A single-threaded executor made depth wait behind YOLO,
        # producing artificial RGB/depth timestamp mismatches.
        self.rgb_callback_group = MutuallyExclusiveCallbackGroup()
        self.depth_callback_group = MutuallyExclusiveCallbackGroup()
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, depth_topic, self.depth_callback,
                                 camera_qos,
                                 callback_group=self.depth_callback_group)
        self.create_subscription(CameraInfo, info_topic, self.info_callback,
                                 camera_qos,
                                 callback_group=self.depth_callback_group)
        self.create_subscription(Image, rgb_topic, self.rgb_callback,
                                 camera_qos,
                                 callback_group=self.rgb_callback_group)
        clear_topic = str(self.get_parameter('clear_interactions_topic').value)
        if clear_topic:
            self.create_subscription(
                Empty, clear_topic, self.clear_interactions_callback, 10)

        self.detections_pub = self.create_publisher(
            Detection2DArray, '/social_perception/detections_2d', 10)
        self.annotated_pub = self.create_publisher(
            Image, '/social_perception/annotated_image', 10)
        self.depth_visual_pub = self.create_publisher(
            Image, '/social_perception/depth_visualization', 10)
        self.markers_pub = self.create_publisher(
            MarkerArray, '/social_perception/person_markers', 10)
        self.people_pub = self.create_publisher(
            People, str(self.get_parameter('people_topic').value), 10)
        self.groups_pub = self.create_publisher(
            Groups, str(self.get_parameter('social_regions_topic').value), 10)
        self.interactions_pub = self.create_publisher(
            TalkingInteractions,
            str(self.get_parameter('interactions_topic').value), 10)
        self.social_markers_pub = self.create_publisher(
            MarkerArray, '/social_spaces', 10)

        # The remote half of the pipeline, wired up here rather than inside the
        # worker thread so every endpoint exists before the executor spins.
        self.remote_backend = None
        self.remote_worker_ready = threading.Event()
        if self.vlm_enabled and self.vlm_remote:
            # Its own group: an answer arriving must not queue behind a depth
            # frame, and the depth path must not wait on network traffic.
            self.remote_callback_group = MutuallyExclusiveCallbackGroup()
            # Depth 1 and BEST_EFFORT: one question is in flight at a time, and
            # a crop that could not be delivered is worthless a second later --
            # the pair will simply be asked about again from a fresher frame.
            request_pub = self.create_publisher(
                VlmRequest, '/social_perception/vlm_request',
                QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                           reliability=ReliabilityPolicy.BEST_EFFORT,
                           durability=DurabilityPolicy.VOLATILE))
            self.remote_backend = RemoteVlmBackend(
                self, request_pub, self.vlm_request_timeout,
                self.vlm_crop_jpeg_quality)
            self.create_subscription(
                VlmResponse, '/social_perception/vlm_response',
                self.remote_backend.handle_response, 10,
                callback_group=self.remote_callback_group)
            # Latched on the worker's side, so this learns the model is up even
            # if the workstation was started first.
            self.create_subscription(
                Bool, '/social_perception/vlm_worker_ready',
                self.remote_worker_ready_callback,
                QoSProfile(
                    depth=1,
                    history=HistoryPolicy.KEEP_LAST,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL),
                callback_group=self.remote_callback_group)
        # Latched, so anything that starts later still learns the pipeline is
        # up. It fires once YOLO has processed a real frame and the VLM has
        # finished loading, which is the moment people can actually be seen.
        self.ready_pub = self.create_publisher(
            Bool, '/social_perception/ready',
            QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.ready_lock = threading.Lock()
        self.ready_published = False
        self.first_frame_done = False
        # A disabled VLM is "loaded" by definition; a failed one must not hold
        # the signal back forever, so the worker reports either way.
        self.vlm_load_done = not self.vlm_enabled
        self.vlm_load_ok = not self.vlm_enabled

        self.create_timer(0.2, self.publish_social_outputs)
        if self.vlm_enabled:
            self.vlm_thread = threading.Thread(
                target=self.vlm_worker, name='social-vlm-worker', daemon=True)
            self.vlm_thread.start()
        self.get_logger().info(
            f'Social perception ready: RGB={rgb_topic}, depth={depth_topic}, '
            f'target={self.target_frame}, YOLO={yolo_path} on {self.yolo_device}, '
            f'VLM={self.vlm_enabled}')

    def _warm_up_yolo(self):
        """Build the YOLO inference backend before the VLM thread can exist.

        Ultralytics defers the real work to the first predict(): it fuses
        Conv+BatchNorm, and `fuse_conv_and_bn` calls `register_parameter` to
        install the fused bias. Meanwhile Transformers loads the VLM under
        accelerate's `init_on_device`, which patches `nn.Module.
        register_parameter` process-wide -- not per-thread -- to divert new
        parameters to the meta device. A first frame arriving inside that
        window therefore built a YOLO whose fused biases had no storage, and
        the following `.to('cuda:0')` died with 'Cannot copy out of meta
        tensor', taking the node down with it.

        Doing the fuse here, before the worker thread starts, closes the
        window: later predicts reuse this backend and register no parameters.
        """
        started_at = time.monotonic()
        try:
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            self.yolo.predict(source=blank, classes=[0], conf=self.confidence,
                              imgsz=self.image_size, device=self.yolo_device,
                              verbose=False)
        except Exception as error:
            # Leave the node running: the first camera frame will retry this,
            # and warning here is what explains a later meta-tensor crash.
            self.get_logger().warn(
                f'YOLO warm-up failed ({type(error).__name__}: {error}); the '
                'first frame will build the backend instead, which can race '
                'the VLM load')
            return
        self.get_logger().info(
            f'YOLO ready on {self.yolo_device} in '
            f'{time.monotonic() - started_at:.2f}s')

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

    def remote_worker_ready_callback(self, message):
        if message.data:
            self.remote_worker_ready.set()

    def wait_for_remote_worker(self):
        """Block until the workstation reports its model is loaded.

        No deadline on purpose. The robot may well be powered on before the
        workstation, and giving up would leave a node that looks alive while
        silently never asking anything. The heartbeat is what tells an operator
        which of the two machines they are still waiting for.
        """
        waited = 0.0
        while not self.stop_event.is_set():
            if self.remote_worker_ready.wait(5.0):
                self.get_logger().info(
                    f'VLM từ xa đã sẵn sàng sau {waited:.0f}s')
                return True
            waited += 5.0
            self.get_logger().info(
                f'Đang đợi social_vlm_worker.py trên máy trạm ({waited:.0f}s). '
                'Kiểm tra ROS_DOMAIN_ID và ROS_LOCALHOST_ONLY nếu chờ quá lâu.')
        return False

    def depth_callback(self, message):
        self.depth_msg = message
        self.depth_frames += 1

    def info_callback(self, message):
        self.camera_info = message

    def clear_interactions_callback(self, _message):
        """Immediately invalidate simulated people when they are hidden."""
        with self.state_lock:
            had_result = any(
                result['state'] != 'processing'
                for result in self.interaction_cache.values())
            self.latest_people = {}
            self.latest_observation_sequence += 1
            # Invalidate queued and in-flight work. CUDA generation cannot be
            # interrupted safely, but its answer must never enter the cache.
            self.scene_generation += 1
            self.interaction_cache.clear()
            self.vlm_inflight_pairs.clear()
            self.last_vlm_scenes.clear()
            self.last_logged_decisions.clear()
        self.discard_queued_vlm_work()
        if had_result:
            self.get_logger().info(
                'TRẠNG THÁI: KHÔNG CÓ CẶP NGƯỜI TRONG CAMERA')

    def mark_first_frame_done(self):
        with self.ready_lock:
            self.first_frame_done = True
        self.publish_ready_if_complete()

    def mark_vlm_load_done(self, loaded):
        with self.ready_lock:
            self.vlm_load_done = True
            self.vlm_load_ok = loaded
        self.publish_ready_if_complete()

    def publish_ready_if_complete(self):
        """Announce readiness once, after YOLO and the VLM have both settled."""
        with self.ready_lock:
            if self.ready_published or not (
                    self.first_frame_done and self.vlm_load_done):
                return
            self.ready_published = True
            vlm_ok = self.vlm_load_ok
        message = Bool()
        message.data = bool(vlm_ok)
        self.ready_pub.publish(message)
        if not self.vlm_enabled:
            state = 'đã tắt (enable_vlm=false)'
        elif vlm_ok:
            state = 'đã nạp xong'
        else:
            state = 'KHÔNG khả dụng, chỉ có định vị người'
        self.get_logger().info(f'SẴN SÀNG: camera + YOLO hoạt động, VLM {state}')

    def report_status(self):
        with self.state_lock:
            interaction_count = len(self.interaction_cache)
        self.get_logger().info(
            f'Input: RGB={self.rgb_frames}, depth={self.depth_frames}, '
            f'CameraInfo={"yes" if self.camera_info else "no"}, '
            f'cached interactions={interaction_count}')

    @staticmethod
    def rgb_array(message):
        channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4}.get(
            message.encoding)
        if channels is None:
            raise ValueError(f'unsupported RGB encoding: {message.encoding}')
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.step)
        image = rows[:, :message.width * channels].reshape(
            message.height, message.width, channels)
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
        rows = np.frombuffer(message.data, dtype=dtype).reshape(
            message.height, row_items)
        return rows[:, :message.width].astype(np.float32) * scale

    # Fraction of the bounding box sampled for distance: the torso, which is
    # the one part of a person that is neither background seen past an arm nor
    # floor seen past the feet.
    TORSO_SIDE, TORSO_TOP, TORSO_BOTTOM = 0.30, 0.20, 0.70

    # Bề dày một thân người, dùng để tách người khỏi nền trong ROI. Rộng hơn
    # thân thật (~0.30 m) để chừa chỗ cho người đứng chếch và cho nhiễu depth.
    PERSON_DEPTH = 0.5

    @classmethod
    def torso_center(cls, box):
        """Pixel that median_depth's distance actually belongs to.

        A pinhole ray is only valid at the pixel its depth was measured at.
        Sampling the torso but casting through the feet placed every person
        several tens of centimetres towards the camera, because the two rays
        diverge once the camera is tilted.
        """
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5,
                y1 + 0.5 * (cls.TORSO_TOP + cls.TORSO_BOTTOM) * (y2 - y1))

    def median_depth(self, depth, box, rgb_shape):
        x1, y1, x2, y2 = box
        scale_x = depth.shape[1] / rgb_shape[1]
        scale_y = depth.shape[0] / rgb_shape[0]
        width, height = x2 - x1, y2 - y1
        rx1 = (x1 + self.TORSO_SIDE * width) * scale_x
        rx2 = (x2 - self.TORSO_SIDE * width) * scale_x
        ry1 = (y1 + self.TORSO_TOP * height) * scale_y
        ry2 = (y1 + self.TORSO_BOTTOM * height) * scale_y
        roi = depth[max(0, int(ry1)):min(depth.shape[0], int(ry2)),
                    max(0, int(rx1)):min(depth.shape[1], int(rx2))]
        valid = roi[np.isfinite(roi) & (roi >= self.min_depth) &
                    (roi <= self.max_depth)]
        if valid.size < 8:
            return None
        # Trung vị của MẶT GẦN NHẤT, không phải của cả ROI.
        #
        # Trung vị cả ROI trả về bức tường ngay khi quá nửa ROI là nền - và
        # điều đó xảy ra thường xuyên hơn ta tưởng: người đứng chếch, bị che
        # một phần, hoặc khung YOLO rộng hơn thân. Đo 29-08-2026 trên
        # lirs_test.world, 705 lần đo trong 80 s: 54 lần (8%) lệch quá 0.5 m,
        # tệ nhất là 8.50 m thay vì 1.44 m khi 54% ROI là nền. Hậu quả không
        # phải "mất người" mà là ĐẶT NGƯỜI SAI CHỖ - một chấm nổi trên tường,
        # cùng phương vị nhưng xa gấp hơn hai lần, và khối D dựng nguyên một
        # vùng xã hội quanh nó.
        #
        # Người LUÔN ở trước nền, nên mặt gần nhất trong ROI là người. Lấy
        # phân vị 10 thay vì min để một vài điểm nhiễu quá gần không kéo lệch,
        # rồi giữ mọi điểm trong một bề dày thân người tính từ đó.
        nearest = float(np.percentile(valid, 10))
        front = valid[valid <= nearest + self.PERSON_DEPTH]
        return float(np.median(front))

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
            self.get_logger().warn(
                f'TF {source_frame} -> {self.target_frame}: {error}',
                throttle_duration_sec=2.0)
            return None
        rotated = rotate_vector(optical, transform.transform.rotation)
        translation = transform.transform.translation
        return (rotated[0] + translation.x,
                rotated[1] + translation.y,
                rotated[2] + translation.z)

    def robot_pose(self, stamp):
        """(x, y, yaw) of the robot in target_frame, or None if TF is not up.

        Only radial_velocity and motion_type need this. A failure here must not
        cost the frame: the positions and velocities are already valid without
        it, and dropping them would be a worse trade than publishing two empty
        labels -- which is exactly what an empty motion_type means.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.robot_frame,
                rclpy.time.Time.from_msg(stamp), timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(
                f'TF {self.robot_frame} -> {self.target_frame}: {error}',
                throttle_duration_sec=5.0)
            return None
        translation = transform.transform.translation
        return (translation.x, translation.y,
                quaternion_yaw(transform.transform.rotation))

    def scene_rulings(self):
        """Block C's verdict per person id: {id: (scene_type, confidence, stamp)}.

        Reads the decision cache the social-region publisher already maintains
        rather than asking the model again. The cache is keyed by pair and
        refreshed every few seconds; this runs at frame rate, so re-deriving it
        here would be asking the same question hundreds of times per answer.

        Only a positive `talking` ruling produces a scene_type. "not_talking"
        is a real answer, but block D carries no region for it -- its neutral
        shape already IS "nothing known socially", so inventing a value its
        table does not hold would fall back to that same shape while looking
        like information. Which way somebody is moving is block B's half of the
        judgement and arrives on motion_type either way.
        """
        rulings = {}
        with self.state_lock:
            cached = list(self.interaction_cache.items())
        for pair, result in cached:
            if result['state'] != 'talking':
                continue
            stamp = result['inference_stamp']
            for member_id in pair:
                previous = rulings.get(member_id)
                # Somebody can sit in two confirmed pairs. The newest ruling is
                # the one that says how recently this was reconfirmed.
                if (previous is None or
                        stamp_seconds(stamp) > stamp_seconds(previous[2])):
                    rulings[member_id] = (
                        'talking', float(result['confidence']), stamp)
        return rulings

    def assign_track(self, point, stamp_value, used_ids):
        best_id, best_distance = None, self.track_distance
        for track_id, state in self.tracks.items():
            if track_id in used_ids:
                continue
            distance = math.hypot(point[0] - state['point'][0],
                                  point[1] - state['point'][1])
            if distance < best_distance:
                best_id, best_distance = track_id, distance
        if best_id is None:
            # Before minting a new id, check whether this is somebody who was
            # tracked until moments ago and is standing where they were left.
            #
            # A new id is not a cosmetic detail here. Every VLM decision is
            # cached under the pair of ids it was made about, so reissuing one
            # renames the pair, orphans the decision and deletes the social
            # region -- for a full inference, which is tens of seconds during
            # which the two people are still talking and nothing keeps the
            # robot out from between them. Handing the same person their id
            # back is what makes the region survive being briefly lost.
            for track_id, state in self.lost_tracks.items():
                if track_id in used_ids:
                    continue
                if stamp_value - state['stamp'] > self.reidentify_timeout:
                    continue
                distance = math.hypot(point[0] - state['point'][0],
                                      point[1] - state['point'][1])
                if distance < best_distance:
                    best_id, best_distance = track_id, distance
            if best_id is not None:
                self.lost_tracks.pop(best_id, None)
                # Velocity across an unobserved gap is not measurable, and a
                # position difference divided by that gap is not a speed.
                return best_id, (0.0, 0.0, 0.0)

        if best_id is None:
            best_id = self.next_track_id
            self.next_track_id += 1
            velocity = (0.0, 0.0, 0.0)
        else:
            previous = self.tracks[best_id]
            dt = stamp_value - previous['stamp']
            if dt <= 1e-3:
                velocity = (0.0, 0.0, 0.0)
            else:
                measured = tuple((point[index] - previous['point'][index]) / dt
                                 for index in range(3))
                velocity = tuple(
                    self.velocity_alpha * measured[index] +
                    (1.0 - self.velocity_alpha) * previous['velocity'][index]
                    for index in range(3))
        return best_id, velocity

    def rgb_callback(self, rgb_msg):
        self.rgb_frames += 1
        # A hide/clear callback may run while YOLO is processing this frame.
        # Capturing the generation here lets us reject that stale frame later.
        with self.state_lock:
            frame_generation = self.scene_generation
        # Decoded before the depth checks below, and separately from depth, so
        # that every path from here on has a frame to show. Turning the RGB
        # message into an array needs nothing from depth, and the camera view
        # is how a person checks the pipeline is alive -- it must not go dark
        # because depth is late, which is exactly when someone looks at it.
        try:
            image = self.rgb_array(rgb_msg)
        except (ValueError, TypeError) as error:
            # The one case with nothing to publish.
            self.get_logger().error(str(error), throttle_duration_sec=2.0)
            return

        # Keep one immutable reference throughout this inference even if the
        # depth callback receives a newer frame on the second executor thread.
        depth_msg = self.depth_msg
        if depth_msg is None or self.camera_info is None:
            self.get_logger().warn('Waiting for depth image and CameraInfo',
                                   throttle_duration_sec=2.0)
            self.publish_camera_view(image, rgb_msg.header)
            return
        if abs(stamp_seconds(rgb_msg.header.stamp) -
               stamp_seconds(depth_msg.header.stamp)) > self.max_depth_age:
            self.get_logger().warn('RGB and depth timestamps are not synchronized',
                                   throttle_duration_sec=2.0)
            self.publish_camera_view(image, rgb_msg.header)
            return
        # /people is stamped with this frame's time, and the costmap layer
        # transforms it through a TF buffer that only keeps a few seconds. A
        # frame this callback picked up late would be published under a stamp
        # TF has already forgotten, which Nav2 reports as "extrapolation into
        # the past" and which silently costs the social layer its input. Drop
        # it here instead. A clock that has not started yet reads as a negative
        # age and is left alone.
        frame_age = (stamp_seconds(self.get_clock().now().to_msg()) -
                     stamp_seconds(rgb_msg.header.stamp))
        if frame_age > self.max_frame_age:
            self.get_logger().warn(
                f'Bỏ khung hình cũ {frame_age:.2f}s (ngưỡng '
                f'{self.max_frame_age:.2f}s): pipeline không theo kịp camera',
                throttle_duration_sec=2.0)
            self.publish_camera_view(image, rgb_msg.header)
            return
        try:
            depth_image = self.depth_array(depth_msg)
        except (ValueError, TypeError) as error:
            self.get_logger().error(str(error), throttle_duration_sec=2.0)
            self.publish_camera_view(image, rgb_msg.header)
            return

        self.publish_depth_visualization(depth_image, depth_msg.header)
        # Runs unsynchronized with the VLM on purpose; see ml_execution_lock.
        result = self.yolo.predict(
            source=image, classes=[0], conf=self.confidence,
            imgsz=self.image_size, device=self.yolo_device,
            verbose=False)[0]
        detections = Detection2DArray()
        detections.header = rgb_msg.header
        people = People()
        people.header.stamp = rgb_msg.header.stamp
        people.header.frame_id = self.target_frame
        # Block B's two robot-relative fields need the robot in the same frame
        # the people end up in. None means the chain is not up yet, and
        # radial_velocity/motion_type stay at their "not known" values rather
        # than being filled with a guess that reads like a measurement.
        robot = self.robot_pose(rgb_msg.header.stamp)
        # Block C's standing verdicts, read once per frame rather than per
        # person: the cache is shared and taking its lock inside the detection
        # loop would serialise every box against the inference thread.
        rulings = self.scene_rulings()
        markers = MarkerArray()
        annotated = image.copy()
        candidates = []
        new_tracks, used_ids = {}, set()
        stamp_value = stamp_seconds(rgb_msg.header.stamp)
        # Every point already accepted for tracking this frame.
        #
        # YOLO's NMS only drops boxes that overlap heavily in the image; two
        # boxes on the same person -- torso and whole body, say -- survive it
        # and land within a few centimetres of each other once projected. Both
        # reaching assign_track mints a SECOND id for that person, because
        # used_ids forbids the later box from taking the id the first one just
        # claimed, and the loser is then coasted for tracking_timeout seconds
        # as a person in its own right. Measured 31-08-2026 with two people in
        # the room: 28 ids in 40 s, up to 7 people in one message, and only
        # 23.6% of messages reporting the right count.
        #
        # The coasting loop below already refuses to republish a track sitting
        # on top of somebody published this frame. This is the same test, at
        # the same distance, applied to the detections themselves.
        frame_points = []

        if result.boxes is not None:
            # Highest score first, so the box that survives a merge is the
            # confident one. Ultralytics already returns NMS output in this
            # order; sorting makes that a property of this code rather than of
            # the version installed.
            for box in sorted(result.boxes,
                              key=lambda item: float(item.conf[0]),
                              reverse=True):
                bbox = tuple(float(value) for value in box.xyxy[0].cpu().tolist())
                score = float(box.conf[0].cpu())
                detection = self.make_detection(rgb_msg, bbox, score)
                detections.detections.append(detection)
                z = self.median_depth(depth_image, bbox, image.shape)
                x1, y1, x2, y2 = bbox
                source_frame = depth_msg.header.frame_id or self.camera_info.header.frame_id
                point = None if z is None else self.point_in_target(
                    *self.torso_center(bbox), z,
                    source_frame, rgb_msg.header.stamp)
                color = (0, 200, 255) if point is not None else (0, 0, 255)
                label = f'person {score:.2f}' + (
                    f' {z:.1f}m' if z is not None else ' no-depth')
                cv2.rectangle(annotated, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if point is None:
                    continue
                if any(math.hypot(point[0] - x, point[1] - y)
                       < self.duplicate_distance for x, y in frame_points):
                    cv2.putText(annotated, 'merged',
                                (int(x1), min(image.shape[0] - 8, int(y2) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    continue
                frame_points.append((point[0], point[1]))

                track_id, velocity = self.assign_track(
                    point, stamp_value, used_ids)
                used_ids.add(track_id)
                new_tracks[track_id] = {
                    'point': point, 'velocity': velocity, 'stamp': stamp_value}
                # One frame of evidence is not a person.
                #
                # Measured 01-09-2026 with two people in the room: a spurious
                # box seen ONCE at 7.6 m produced a track that was then coasted
                # for the full tracking_timeout -- 49 messages of /people, six
                # seconds, from a single frame -- and social_rl's own people
                # memory held it for seconds longer still. A second phantom
                # came from two consecutive frames and cost another 49. Raising
                # yolo_confidence does not reach them: those boxes scored 0.87
                # and 0.53/0.31, straddling any threshold a real person also
                # has to pass.
                #
                # So gate on evidence rather than on score. An unconfirmed
                # track stays in self.tracks and keeps counting, but reaches
                # neither /people nor the coasting loop below. The cost is that
                # somebody entering frame is published (minimum_track_hits - 1)
                # frames late, 0.25 s at the 8 Hz measured here; a person who
                # was already confirmed keeps their count through lost_tracks
                # and pays it only once.
                hits = self.track_hits.get(track_id, 0) + 1
                self.track_hits[track_id] = hits
                if hits < self.minimum_track_hits:
                    cv2.putText(annotated,
                                f'new {hits}/{self.minimum_track_hits}',
                                (int(x1), min(image.shape[0] - 8, int(y2) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    continue
                detection.id = f'person_{track_id}'
                person = Person()
                person.id = detection.id
                person.pose.position.x, person.pose.position.y, person.pose.position.z = point
                person.pose.orientation.w = 1.0
                (person.velocity.linear.x, person.velocity.linear.y,
                 person.velocity.linear.z) = velocity
                fill_prediction(person, self.prediction_times,
                                self.prediction_still_speed)
                if robot is not None:
                    fill_relative_motion(person, robot,
                                         self.prediction_still_speed)
                apply_scene_ruling(person, rulings)
                people.people.append(person)
                candidates.append({'person': person, 'bbox': bbox})
                markers.markers.append(self.person_marker(people.header, person, track_id))
                cv2.putText(annotated, person.id,
                            (int(x1), min(image.shape[0] - 8, int(y2) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # A person the robot is standing in front of is still a person.
        #
        # Retaining the track used to protect only the id counter: the occluded
        # person still dropped out of /people, so the pair key stopped matching
        # and the finished VLM decision was deleted along with the social
        # region. The robot approaching to pass between two people is itself
        # what occludes them, so the region protecting them disappeared exactly
        # when it was needed and a fresh inference (tens of seconds) was needed
        # to earn it back.
        #
        # Republishing the track at its last known position keeps the pair
        # intact, and keeps the hidden person in the costmap while the robot
        # cannot see them, which is the conservative reading either way.
        published_points = [(person.pose.position.x, person.pose.position.y)
                            for person in people.people]
        for track_id, state in self.tracks.items():
            age = stamp_value - state['stamp']
            if track_id in new_tracks:
                continue
            if not 0.0 <= age <= self.track_timeout:
                # Past coasting range, but remember where they were so the
                # same person can reclaim this id instead of being treated as
                # a stranger the moment they are seen again.
                self.lost_tracks[track_id] = state
                continue
            # Never confirmed, and now not seen either: the blip is over.
            # Coasting it is what turns one stray box into six seconds of
            # person. Park it where a reappearance can still reclaim the id.
            if self.track_hits.get(track_id, 0) < self.minimum_track_hits:
                self.lost_tracks[track_id] = state
                continue
            # A track sitting on top of somebody already published this frame
            # is a second id for that same person, left over from a frame where
            # association picked the other one. Coasting it would put one
            # person in /people twice, and the closest-pair search would then
            # hand the VLM a person paired with themselves instead of the two
            # people actually talking. Let it lapse.
            if any(math.hypot(state['point'][0] - x, state['point'][1] - y)
                   < self.duplicate_distance for x, y in published_points):
                continue
            new_tracks[track_id] = state
            published_points.append((state['point'][0], state['point'][1]))
            coasted = Person()
            coasted.id = f'person_{track_id}'
            (coasted.pose.position.x, coasted.pose.position.y,
             coasted.pose.position.z) = state['point']
            coasted.pose.orientation.w = 1.0
            (coasted.velocity.linear.x, coasted.velocity.linear.y,
             coasted.velocity.linear.z) = state['velocity']
            fill_prediction(coasted, self.prediction_times,
                            self.prediction_still_speed)
            if robot is not None:
                fill_relative_motion(coasted, robot,
                                     self.prediction_still_speed)
            apply_scene_ruling(coasted, rulings)
            people.people.append(coasted)
            markers.markers.append(
                self.person_marker(people.header, coasted, track_id))
        self.tracks = new_tracks
        # Hit counts outlive self.tracks but not lost_tracks, which is what a
        # reclaimed id needs and all it needs.
        self.track_hits = {track_id: hits
                           for track_id, hits in self.track_hits.items()
                           if track_id in new_tracks or track_id in self.lost_tracks}
        # `annotated` is finished at this point -- every box and label is drawn
        # by the two loops above, and nothing below touches the pixels. Sending
        # it now rather than at the end of the callback means the scene-cleared
        # return below cannot blank the RViz camera panel, and keeps the topic
        # off the one lock this callback still takes.
        self.publish_camera_view(annotated, rgb_msg.header)
        visible_ids = {item.id for item in people.people}
        removed_completed_result = False
        with self.state_lock:
            if frame_generation != self.scene_generation:
                # This RGB message started before a hide/clear event. Publishing
                # its people would make old actors visible again and could
                # validate an obsolete VLM answer. The camera view is just what
                # the lens saw, so it went out above regardless.
                return
            # Somebody leaving the frame invalidates work about *them*, not the
            # whole scene.
            #
            # This used to bump scene_generation, which every queued and running
            # inference is tagged with, so one person dropping out threw away an
            # in-flight answer about a different pair entirely. Tracking loses a
            # person routinely -- YOLO runs at ~6.6 fps on the CPU and a walking
            # person is missed for a few frames at a time -- so with people
            # moving, inferences were being discarded and restarted faster than
            # they could finish, and no result ever reached the topic.
            #
            # Dropping the per-pair markers is enough: the worker re-checks
            # membership before inferring, and pair_visible_after re-checks it
            # again before publishing. Clearing the markers here also lets an
            # affected pair be re-enqueued immediately instead of waiting out
            # vlm_refresh_interval.
            vanished_ids = set(self.latest_people) - visible_ids
            dropped_inflight = []
            if vanished_ids:
                for registry in (self.vlm_inflight_pairs, self.last_vlm_scenes):
                    for pair in [pair for pair in registry
                                 if not vanished_ids.isdisjoint(pair)]:
                        if registry is self.vlm_inflight_pairs:
                            dropped_inflight.append(pair)
                        registry.pop(pair, None)
            self.latest_people = {item.id: item for item in people.people}
            self.latest_header = people.header
            self.latest_observation_sequence += 1
            stale_pairs = [
                pair for pair in self.interaction_cache
                if not all(member_id in visible_ids for member_id in pair)
            ]
            for pair in stale_pairs:
                removed = self.interaction_cache.pop(pair)
                if removed['state'] != 'processing':
                    removed_completed_result = True
                self.last_logged_decisions.pop(pair, None)
        if dropped_inflight:
            # If this appears repeatedly while two people are talking, tracking
            # is losing them faster than the VLM can answer: raise
            # tracking_timeout or tracking_max_distance rather than blaming the
            # model.
            self.get_logger().warn(
                'MẤT DẤU giữa lúc suy luận, bỏ kết quả VLM của '
                f'{dropped_inflight} (biến mất: {sorted(vanished_ids)})',
                throttle_duration_sec=5.0)
        if removed_completed_result and len(visible_ids) < 2:
            self.get_logger().info(
                'TRẠNG THÁI: KHÔNG CÓ CẶP NGƯỜI TRONG CAMERA')
        self.mark_first_frame_done()
        self.detections_pub.publish(detections)
        self.people_pub.publish(people)
        self.markers_pub.publish(markers)
        self.enqueue_vlm_scene(image, candidates, people.header)

    @staticmethod
    def make_detection(rgb_msg, bbox, score):
        x1, y1, x2, y2 = bbox
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
        return detection

    @staticmethod
    def person_marker(header, person, track_id):
        marker = Marker()
        marker.header = header
        marker.ns = 'rgbd_people'
        marker.id = track_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        # A message field assignment stores the object itself, so the raised
        # marker height below would otherwise be written back into the Person
        # already queued for /people.
        marker.pose = copy.deepcopy(person.pose)
        marker.pose.position.z = max(0.9, person.pose.position.z)
        marker.scale.x, marker.scale.y, marker.scale.z = 0.55, 0.55, 1.8
        marker.color.r, marker.color.g = 0.0, 1.0
        marker.color.b, marker.color.a = 0.1, 0.55
        marker.lifetime = Duration(seconds=0.4).to_msg()
        return marker

    @staticmethod
    def cv_image_message(image, header):
        output = Image()
        output.header = header
        output.height, output.width = image.shape[:2]
        output.encoding = 'bgr8'
        output.is_bigendian = 0
        output.step = output.width * 3
        # array.array('B', ...) rather than bytes: the generated setter for a
        # uint8[] field returns immediately for an array.array, but validates a
        # bytes object element by element -- two Python loops over all 921 600
        # bytes of a 640x480 frame. Measured on this machine: 257 ms per call
        # against 0.34 ms. Two calls per frame (annotated view and depth
        # colormap) were 880 ms of every 1215 ms rgb_callback spent, which held
        # /people to 0.82 Hz and made its stamps older than people_timeout.
        output.data = array.array('B', image.tobytes())
        return output

    def publish_camera_view(self, image, header):
        """Put a frame on the annotated topic, whichever way rgb_callback exits.

        The RViz camera panel is the one place a person looks to decide whether
        this pipeline is alive, so it stays fed even on the paths that have no
        people to report: depth not up yet, depth out of sync, a frame arriving
        too late to stamp, or the scene cleared mid-frame. On those paths the
        image is the plain camera view with no boxes, which is the honest
        picture -- nothing was detected because nothing was run.

        The overlay is drawn here rather than at the call sites so the latency
        readout appears on every published frame, not only on the complete ones.
        """
        if self.latency_overlay:
            self.draw_latency_overlay(image)
        self.annotated_pub.publish(self.cv_image_message(image, header))

    def publish_depth_visualization(self, depth_image, depth_header):
        finite = np.isfinite(depth_image) & (depth_image >= self.min_depth) & (
            depth_image <= self.max_depth)
        normalized = np.zeros(depth_image.shape, dtype=np.uint8)
        if np.any(finite):
            near = float(np.percentile(depth_image[finite], 2.0))
            far = float(np.percentile(depth_image[finite], 98.0))
            normalized[finite] = np.clip(
                255.0 * (depth_image[finite] - near) / max(0.1, far - near),
                0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        self.depth_visual_pub.publish(
            self.cv_image_message(depth_color, depth_header))

    def enqueue_vlm_scene(self, image, candidates, observation_header):
        if not self.vlm_enabled or len(candidates) < 2:
            self.discard_queued_vlm_work()
            return
        now = time.monotonic()
        if now - self.last_vlm_enqueue < self.vlm_interval:
            return
        pair_candidates = []
        for first_index in range(len(candidates)):
            for second_index in range(first_index + 1, len(candidates)):
                first, second = candidates[first_index], candidates[second_index]
                distance = math.hypot(
                    first['person'].pose.position.x - second['person'].pose.position.x,
                    first['person'].pose.position.y - second['person'].pose.position.y)
                if distance <= self.max_pair_distance:
                    pair_candidates.append((distance, first, second))
        pair_candidates.sort(key=lambda item: item[0])
        with self.state_lock:
            observation_sequence = self.latest_observation_sequence
            scene_generation = self.scene_generation
        work = []
        # Walk every pair and stop once enough work is collected, rather than
        # only ever looking at the closest `max_vlm_pairs`. Truncating first
        # meant the nearest pair permanently occupied the only slot: while its
        # decision was still fresh the loop skipped it and ended, so a second
        # conversation further away was never once put to the model and could
        # not form a region of its own.
        for _, first, second in pair_candidates:
            if len(work) >= self.max_vlm_pairs:
                break
            pair = tuple(sorted((first['person'].id, second['person'].id)))
            positions = {
                first['person'].id: (
                    first['person'].pose.position.x,
                    first['person'].pose.position.y),
                second['person'].id: (
                    second['person'].pose.position.x,
                    second['person'].pose.position.y),
            }
            with self.state_lock:
                previous = self.last_vlm_scenes.get(pair)
                in_flight = pair in self.vlm_inflight_pairs
            moved = previous is None
            if previous is not None:
                moved = any(math.hypot(
                    positions[member_id][0] - previous['positions'][member_id][0],
                    positions[member_id][1] - previous['positions'][member_id][1],
                ) >= self.vlm_position_threshold for member_id in pair)
            refresh_due = (previous is None or
                           now - previous['time'] >= self.vlm_refresh_interval)
            if in_flight or not (moved or refresh_due):
                continue
            crop = self.pair_crop(image, first['bbox'], second['bbox'])
            work.append({
                'first_id': first['person'].id,
                'second_id': second['person'].id,
                'people': [copy.deepcopy(first['person']),
                           copy.deepcopy(second['person'])],
                'observation_header': copy.deepcopy(observation_header),
                'observation_sequence': observation_sequence,
                'scene_generation': scene_generation,
                'positions': positions,
                'crop': crop,
            })
        if not work:
            return
        self.discard_queued_vlm_work()
        with self.state_lock:
            if scene_generation != self.scene_generation:
                return
            for item in work:
                pair = tuple(sorted((item['first_id'], item['second_id'])))
                self.vlm_inflight_pairs[pair] = scene_generation
                self.last_vlm_scenes[pair] = {
                    'positions': item['positions'],
                    'time': now,
                    'scene_generation': scene_generation,
                }
        try:
            self.work_queue.put_nowait(work)
        except queue.Full:
            with self.state_lock:
                for item in work:
                    pair = tuple(sorted((item['first_id'], item['second_id'])))
                    if self.vlm_inflight_pairs.get(pair) == scene_generation:
                        self.vlm_inflight_pairs.pop(pair, None)
            return
        self.last_vlm_enqueue = now

    def pair_visible_after(self, pair, sequence, scene_generation, timeout=5.0):
        """Validate an inference against a camera frame captured after it."""
        deadline = time.monotonic() + timeout
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            with self.state_lock:
                current_sequence = self.latest_observation_sequence
                visible_ids = set(self.latest_people)
                current_generation = self.scene_generation
            if current_generation != scene_generation:
                return False
            if current_sequence > sequence:
                return all(member_id in visible_ids for member_id in pair)
            self.stop_event.wait(0.05)
        return False

    def discard_queued_vlm_work(self):
        """Drop camera crops that are no longer the newest observation."""
        discarded_pairs = []
        while True:
            try:
                work = self.work_queue.get_nowait()
                discarded_pairs.extend(
                    (tuple(sorted((item['first_id'], item['second_id']))),
                     item['scene_generation']) for item in work)
            except queue.Empty:
                break
        if discarded_pairs:
            with self.state_lock:
                for pair, generation in discarded_pairs:
                    if self.vlm_inflight_pairs.get(pair) == generation:
                        self.vlm_inflight_pairs.pop(pair, None)

    def pair_crop(self, image, first_box, second_box):
        height, width = image.shape[:2]
        x1 = min(first_box[0], second_box[0])
        y1 = min(first_box[1], second_box[1])
        x2 = max(first_box[2], second_box[2])
        y2 = max(first_box[3], second_box[3])
        margin_x = (x2 - x1) * self.crop_margin
        margin_y = (y2 - y1) * self.crop_margin
        left, top = max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))
        right, bottom = min(width, int(x2 + margin_x)), min(height, int(y2 + margin_y))
        return image[top:bottom, left:right].copy()

    def finish_vlm_item(self, pair, scene_generation, keep_scene):
        """Release one queued/running marker without touching newer work."""
        with self.state_lock:
            if self.vlm_inflight_pairs.get(pair) == scene_generation:
                self.vlm_inflight_pairs.pop(pair, None)
            scene = self.last_vlm_scenes.get(pair)
            if scene is not None and scene['scene_generation'] == scene_generation:
                if keep_scene:
                    scene['time'] = time.monotonic()
                else:
                    self.last_vlm_scenes.pop(pair, None)

    def vlm_worker(self):
        backend = self.load_vlm_backend()
        if backend is None:
            return
        prompt = str(self.get_parameter('talking_prompt').value)
        self.run_vlm_loop(backend, prompt)

    def load_vlm_backend(self):
        """Return the object vlm_worker calls .infer() on, or None to give up.

        Both answer the same call, so everything after this point is identical
        whether the model sits in this process or on another machine.
        """
        if self.vlm_remote:
            if not self.wait_for_remote_worker():
                self.mark_vlm_load_done(False)
                return None
            self.mark_vlm_load_done(True)
            return self.remote_backend

        # Imported here, not at module scope, so a robot running vlm_remote
        # never needs transformers, peft or bitsandbytes installed at all.
        from vlm_backend import VlmBackend

        adapter_value = str(self.get_parameter('vlm_adapter_path').value)
        adapter_path = self._resolve_file(adapter_value)

        # A merged checkpoint is preferred when one is present: it loads without
        # PEFT and answers identically (22/22 on the recorded crops). Missing or
        # half-written, the adapter path below still works, so this stays an
        # optimisation rather than a new requirement. Build one with
        # social_perception/scripts/merge_vlm_adapter.py.
        merged_value = str(self.get_parameter('vlm_merged_path').value)
        merged_path = self._resolve_file(merged_value) if merged_value else None
        if merged_path is not None and not (merged_path / 'config.json').is_file():
            self.get_logger().warn(
                f'Ignoring vlm_merged_path {merged_value}: no config.json there')
            merged_path = None

        if merged_path is None and (
                adapter_path is None
                or not (adapter_path / 'adapter_config.json').is_file()):
            self.get_logger().error(
                f'VLM adapter not found at {adapter_value}; people localization remains active')
            self.mark_vlm_load_done(False)
            return None
        try:
            backend = VlmBackend(
                adapter_path,
                str(self.get_parameter('vlm_base_model').value),
                bool(self.get_parameter('vlm_load_in_4bit').value),
                bool(self.get_parameter('vlm_require_cuda').value),
                int(self.get_parameter('vlm_max_new_tokens').value),
                int(self.get_parameter('vlm_min_pixels').value),
                int(self.get_parameter('vlm_max_pixels').value),
                self.get_logger(),
                self.ml_execution_lock,
                bool(self.get_parameter('vlm_force_answer_prefix').value),
                bool(self.get_parameter('vlm_offline').value),
                merged_path)
        except Exception as error:  # Keep YOLO/depth available when ML deps are absent.
            self.get_logger().error(
                f'Unable to load VLM ({type(error).__name__}: {error}); '
                'people localization remains active')
            self.mark_vlm_load_done(False)
            return None
        self.mark_vlm_load_done(True)
        return backend

    def run_vlm_loop(self, backend, prompt):
        while not self.stop_event.is_set():
            try:
                work = self.work_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            for item in work:
                first_id = item['first_id']
                second_id = item['second_id']
                crop = item['crop']
                pair = tuple(sorted((first_id, second_id)))
                item_generation = item['scene_generation']
                with self.state_lock:
                    current_sequence = self.latest_observation_sequence
                    visible_ids = set(self.latest_people)
                    current_generation = self.scene_generation
                if (current_generation != item_generation or
                        (current_sequence > item['observation_sequence'] and
                        not all(member_id in visible_ids for member_id in pair))):
                    self.finish_vlm_item(pair, item_generation, keep_scene=False)
                    continue
                # Publish an explicit in-progress record immediately. This
                # prevents an empty topic from looking like a stalled node
                # during a long Qwen generation call.
                started_stamp = self.get_clock().now().to_msg()
                with self.state_lock:
                    start_is_current = (
                        self.scene_generation == item_generation and
                        all(member_id in self.latest_people for member_id in pair))
                    # Keep the last completed answer visible while refreshing
                    # it. Only publish "processing" before the first answer.
                    if start_is_current and pair not in self.interaction_cache:
                        self.interaction_cache[pair] = {
                            'state': 'processing',
                            'talking': False,
                            'confidence': 0.0,
                            'response': 'VLM inference in progress',
                            'people': item['people'],
                            'observation_header': item['observation_header'],
                            'inference_stamp': started_stamp,
                            'time': time.monotonic(),
                            'scene_generation': item_generation,
                        }
                if not start_is_current:
                    self.finish_vlm_item(pair, item_generation, keep_scene=False)
                    continue
                self.latency_started(pair)
                self.get_logger().info(
                    f'>>> VLM BẮT ĐẦU suy luận {pair} '
                    f'(ảnh cắt {crop.shape[1]}x{crop.shape[0]})')
                inference_started = time.monotonic()
                try:
                    response = backend.infer(crop, prompt, pair)
                    talking, confidence = talking_from_response(response)
                except Exception as error:
                    self.get_logger().error(
                        f'VLM inference failed for {pair}: {type(error).__name__}: {error}')
                    response, talking, confidence = str(error), False, 0.0
                inference_elapsed = time.monotonic() - inference_started
                average, samples = self.latency_finished(
                    pair, inference_elapsed)
                prep, wait, gen, ntok = getattr(
                    backend, 'last_timing', (0.0, 0.0, 0.0, 0))
                self.get_logger().info(
                    f'<<< VLM XONG {pair}: {inference_elapsed:.2f}s '
                    f'(từ lúc bắt đầu tới khi có kết quả) | '
                    f'tb {average:.2f}s, số lần {samples}')
                self.get_logger().info(
                    f'    chi tiết {pair}: crop={crop.shape[1]}x{crop.shape[0]}, '
                    f'tokens={ntok} [tien_xu_ly={prep:.2f}s '
                    f'cho_lock={wait:.2f}s model={gen:.2f}s]')

                # Do not publish a result from an old image. Require a newer
                # camera frame to confirm that both members are still visible
                # in the same scene generation.
                with self.state_lock:
                    validation_sequence = self.latest_observation_sequence
                if not self.pair_visible_after(
                        pair, validation_sequence, item_generation):
                    with self.state_lock:
                        cached = self.interaction_cache.get(pair)
                        if (cached is not None and
                                cached.get('scene_generation') == item_generation):
                            self.interaction_cache.pop(pair, None)
                            self.last_logged_decisions.pop(pair, None)
                    self.finish_vlm_item(pair, item_generation, keep_scene=False)
                    continue

                inference_stamp = self.get_clock().now().to_msg()
                state = ('unknown' if confidence <= 0.0 else
                         'talking' if talking else 'not_talking')
                cached_result = {
                    'state': state,
                    'talking': talking,
                    'confidence': confidence,
                    'response': response,
                    'people': item['people'],
                    'observation_header': item['observation_header'],
                    'inference_stamp': inference_stamp,
                    'time': time.monotonic(),
                    'scene_generation': item_generation,
                }
                with self.state_lock:
                    if (self.scene_generation != item_generation or
                            not all(member_id in self.latest_people
                                    for member_id in pair)):
                        if self.vlm_inflight_pairs.get(pair) == item_generation:
                            self.vlm_inflight_pairs.pop(pair, None)
                        continue
                    settled = self.settled_result(
                        self.interaction_cache.get(pair), cached_result)
                    self.interaction_cache[pair] = settled
                    if self.vlm_inflight_pairs.get(pair) == item_generation:
                        self.vlm_inflight_pairs.pop(pair, None)
                    scene = self.last_vlm_scenes.get(pair)
                    if (scene is not None and
                            scene['scene_generation'] == item_generation):
                        scene['time'] = time.monotonic()
                    # Log while holding the generation lock so a clear event
                    # cannot be printed before this older decision. Report the
                    # settled decision, not the raw answer: printing "KHÔNG NÓI
                    # CHUYỆN" for an answer that was overruled would contradict
                    # the region still on screen.
                    if self.log_vlm_results:
                        self.log_vlm_result(pair, settled)

    def settled_result(self, previous, fresh):
        """Decide what a new answer does to an already-confirmed conversation.

        The model is asked a borderline question and answers in free text, and
        it does not answer in one stable format: a single session here produced
        `{"talking": "có"}`, the unterminated `{"talking": "có}`, and
        `{"talking":true}`. An answer that cannot be read carries no
        information about the conversation, yet it used to overwrite the cache
        exactly like a confident "no" and delete the region outright.

        Nothing then stopped the robot until a whole new inference confirmed
        the conversation again -- tens of seconds later, and the people were
        still talking the entire time. A single contrary answer is likewise
        weak evidence against a region the robot is actively keeping clear, so
        it takes `negative_answers_to_clear` of them in a row.

        This stickiness is bounded, and deliberately so: publish_social_outputs
        drops the region the moment either person stops being tracked, which is
        the condition that actually means the conversation is over.
        """
        if previous is None or previous['state'] != 'talking':
            return fresh
        if fresh['state'] == 'talking':
            return fresh

        negatives = previous.get('negatives', 0)
        if fresh['state'] == 'not_talking':
            negatives += 1
            if negatives >= self.negatives_to_clear:
                fresh['negatives'] = negatives
                return fresh
            reason = (f'phủ định {negatives}/{self.negatives_to_clear}, '
                      'chưa đủ để xoá')
        else:
            reason = 'không đọc được câu trả lời'
        self.get_logger().info(
            f'GIỮ VÙNG XÃ HỘI ({reason}): {fresh["response"]!r}')
        held = dict(previous)
        held['negatives'] = negatives
        held['time'] = fresh['time']
        held['inference_stamp'] = fresh['inference_stamp']
        return held

    def publish_social_outputs(self):
        now = time.monotonic()
        with self.state_lock:
            people = dict(self.latest_people)
            header = self.latest_header
            # A conversation ends when the model says it ended, not when a
            # stopwatch runs out: two people talking for longer than the
            # timeout had their region deleted underneath them. Zero disables
            # the stopwatch entirely, which is the default. What still bounds
            # the decision is the membership test below -- a region cannot
            # outlive the people it belongs to.
            expired = [
                pair for pair, result in self.interaction_cache.items()
                if (self.interaction_timeout > 0.0 and
                    result['state'] != 'processing' and
                    now - result['time'] > self.interaction_timeout)
            ]
            for pair in expired:
                del self.interaction_cache[pair]
            # A cached decision is meaningful only while every member of that
            # pair is present in the latest successful camera observation.
            results = {
                pair: result for pair, result in self.interaction_cache.items()
                if all(member_id in people for member_id in pair)
            }
        if header is None:
            return

        interactions = TalkingInteractions()
        interactions.header = header
        positive_pairs = []
        for pair, result in sorted(results.items()):
            interaction = TalkingInteraction()
            interaction.id = f'talking_{pair[0]}_{pair[1]}'
            interaction.observation_header = result['observation_header']
            interaction.inference_stamp = result['inference_stamp']
            interaction.member_ids = list(pair)
            interaction.people = result['people']
            interaction.state = result['state']
            interaction.talking = bool(result['talking'])
            interaction.confidence = float(result['confidence'])
            interaction.center = self.center_of(result['people'])
            interaction.raw_response = result['response']
            interactions.interactions.append(interaction)
            if (interaction.talking and
                    all(member_id in people for member_id in pair)):
                positive_pairs.append(pair)

        groups = Groups()
        groups.header = header
        social_markers = MarkerArray()
        delete_all = Marker()
        delete_all.header = header
        delete_all.action = Marker.DELETEALL
        social_markers.markers.append(delete_all)
        marker_id = 0
        for group_index, member_ids in enumerate(self.connected_components(positive_pairs)):
            members = [people[member_id] for member_id in member_ids]
            center = self.center_of(members)
            member_radius = max(math.hypot(
                person.pose.position.x - center.x,
                person.pose.position.y - center.y) for person in members)
            # Stamp each region from the decisions that actually built it. A
            # component can rest on several confirmed pairs, so the newest
            # ruling is what says how recently this region was reconfirmed.
            supporting = [results[pair] for pair in positive_pairs
                          if set(pair) <= member_ids]
            newest = max(supporting,
                         key=lambda item: stamp_seconds(item['inference_stamp']))
            group = Group()
            # Copied, not referenced: assigning a message field stores the
            # object itself, and these two belong to the cached decision that
            # later inferences still read.
            group.inference_stamp = copy.deepcopy(newest['inference_stamp'])
            group.observation_stamp = copy.deepcopy(
                newest['observation_header'].stamp)
            group.confidence = float(newest['confidence'])
            group.id = f'vlm_social_region_{group_index}'
            group.member_ids = sorted(member_ids)
            group.center = center
            group.o_radius = max(self.group_o_min_radius, member_radius * 0.5)
            group.p_radius = max(group.o_radius + 0.15,
                                 member_radius + self.group_p_margin)
            group.r_radius = group.p_radius + self.group_r_margin
            groups.groups.append(group)
            # A translucent disc first, so the region reads as an area from any
            # camera angle. The outlines on top keep the three radii readable.
            social_markers.markers.append(self.social_disc_marker(
                header, group, marker_id))
            marker_id += 1
            for radius, color, label in (
                    (group.o_radius, (1.0, 0.0, 0.0, 0.90), 'O'),
                    (group.p_radius, (1.0, 0.45, 0.0, 0.75), 'P'),
                    (group.r_radius, (1.0, 0.9, 0.0, 0.55), 'R')):
                social_markers.markers.append(self.social_circle_marker(
                    header, group, marker_id, radius, color, label))
                marker_id += 1
            social_markers.markers.append(self.social_label_marker(
                header, group, marker_id, len(members)))
            marker_id += 1
        interaction_signature = tuple(
            (item.id, item.state, item.talking, round(item.confidence, 3),
             item.raw_response, item.inference_stamp.sec,
             item.inference_stamp.nanosec)
            for item in interactions.interactions)
        # Publish decisions immediately when they change. An unchanged state
        # gets only a slow heartbeat so `ros2 topic echo` remains readable.
        if (interaction_signature != self.last_interactions_signature or
                now - self.last_interactions_publish >= 10.0):
            self.interactions_pub.publish(interactions)
            self.last_interactions_signature = interaction_signature
            self.last_interactions_publish = now
        self.groups_pub.publish(groups)
        self.social_markers_pub.publish(social_markers)

    def latency_started(self, pair):
        """Record that this pair is now waiting on the model."""
        with self.latency_lock:
            self.vlm_running_since[pair] = time.monotonic()

    def latency_finished(self, pair, elapsed):
        """Fold one completed inference into the running statistics."""
        with self.latency_lock:
            self.vlm_running_since.pop(pair, None)
            self.vlm_last_latency = elapsed
            self.vlm_latency_count += 1
            self.vlm_latency_total += elapsed
            return (self.vlm_latency_total / self.vlm_latency_count,
                    self.vlm_latency_count)

    def latency_overlay_text(self):
        """One status line for the annotated image, or None when idle.

        cv2's Hershey fonts carry no Vietnamese diacritics, so the text is
        deliberately unaccented -- the alternative renders as boxes.
        """
        now = time.monotonic()
        with self.latency_lock:
            running = dict(self.vlm_running_since)
            last = self.vlm_last_latency
            count = self.vlm_latency_count
            total = self.vlm_latency_total
        if running:
            pair, started_at = min(running.items(), key=lambda item: item[1])
            waited = now - started_at
            label = ','.join(member.replace('person_', '') for member in pair)
            return f'VLM: dang suy luan cap ({label}) {waited:4.1f}s'
        if count:
            return (f'VLM: lan cuoi {last:.2f}s | tb {total / count:.2f}s '
                    f'| n={count}')
        return None

    def draw_latency_overlay(self, annotated):
        text = self.latency_overlay_text()
        if text is None:
            return
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (width, height), baseline = cv2.getTextSize(
            text, font, scale, thickness)
        # A filled plate behind the text keeps it readable over the camera
        # image, which is often bright wall or floor exactly here.
        cv2.rectangle(annotated, (6, 6), (14 + width, 18 + height + baseline),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(annotated, text, (10, 12 + height), font, scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)

    def log_vlm_result(self, pair, result):
        if result['state'] == 'talking':
            decision = 'CÓ NÓI CHUYỆN'
        elif result['state'] == 'not_talking':
            decision = 'KHÔNG NÓI CHUYỆN'
        else:
            decision = 'KHÔNG XÁC ĐỊNH'
        # Continuous monitoring may produce the same answer repeatedly. Print
        # only transitions so the decision remains easy to spot in a terminal.
        if self.last_logged_decisions.get(pair) == decision:
            return
        self.last_logged_decisions[pair] = decision
        self.get_logger().info(f'KẾT QUẢ: {decision}')

    @staticmethod
    def center_of(people):
        center = Point()
        count = float(len(people))
        center.x = sum(person.pose.position.x for person in people) / count
        center.y = sum(person.pose.position.y for person in people) / count
        center.z = sum(person.pose.position.z for person in people) / count
        return center

    @staticmethod
    def connected_components(pairs):
        graph = {}
        for first, second in pairs:
            graph.setdefault(first, set()).add(second)
            graph.setdefault(second, set()).add(first)
        components = []
        remaining = set(graph)
        while remaining:
            seed = remaining.pop()
            component, stack = {seed}, [seed]
            while stack:
                current = stack.pop()
                for neighbor in graph[current] & remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
            components.append(component)
        return components

    # Every social marker outlives one publish cycle only. Without this a
    # crashed or stopped node leaves a region frozen on screen, which reads
    # exactly like a live detection and hides the fact that nothing is running.
    MARKER_LIFETIME = 0.6

    @classmethod
    def social_circle_marker(cls, header, group, marker_id, radius, color, label):
        marker = Marker()
        marker.header = header
        marker.ns = f'vlm_social_region_{label}'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position = copy.deepcopy(group.center)
        marker.pose.position.z = 0.04
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.06
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.lifetime = Duration(seconds=cls.MARKER_LIFETIME).to_msg()
        for index in range(65):
            angle = 2.0 * math.pi * index / 64.0
            point = Point()
            point.x = radius * math.cos(angle)
            point.y = radius * math.sin(angle)
            marker.points.append(point)
        return marker

    @classmethod
    def social_disc_marker(cls, header, group, marker_id):
        """Filled R-space, drawn flat on the floor under the outlines."""
        marker = Marker()
        marker.header = header
        marker.ns = 'vlm_social_region_fill'
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = copy.deepcopy(group.center)
        marker.pose.position.z = 0.01
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2.0 * group.r_radius
        marker.scale.y = 2.0 * group.r_radius
        marker.scale.z = 0.02
        marker.color.r, marker.color.g = 1.0, 0.75
        marker.color.b, marker.color.a = 0.0, 0.22
        marker.lifetime = Duration(seconds=cls.MARKER_LIFETIME).to_msg()
        return marker

    @classmethod
    def social_label_marker(cls, header, group, marker_id, member_count):
        """Floating caption so the VLM decision is visible in the 3D view."""
        marker = Marker()
        marker.header = header
        marker.ns = 'vlm_social_region_label'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = copy.deepcopy(group.center)
        marker.pose.position.z = 2.1
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.28
        marker.color.r = marker.color.g = marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = (f'VLM: dang noi chuyen ({member_count} nguoi)\n'
                       f'R={group.r_radius:.2f} m')
        marker.lifetime = Duration(seconds=cls.MARKER_LIFETIME).to_msg()
        return marker

    def destroy_node(self):
        self.stop_event.set()
        # Print the session total on the way out: the per-inference lines have
        # usually scrolled away by the time a run ends.
        with self.latency_lock:
            samples = self.vlm_latency_count
            total = self.vlm_latency_total
        if samples:
            self.get_logger().info(
                f'TỔNG KẾT VLM: {samples} lần suy luận, '
                f'tb {total / samples:.2f}s')
        if self.vlm_thread is not None:
            try:
                self.vlm_thread.join(timeout=1.0)
            except KeyboardInterrupt:
                # A second SIGINT from ros2 launch can arrive while the daemon
                # worker is leaving its queue wait. Shutdown must stay clean.
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SocialVlmPerception()
    # Three mutually-exclusive callback groups need three threads to run
    # concurrently: RGB (YOLO), depth, and the default group that carries the
    # 0.2 s social-output timer. At two threads a slow RGB frame left depth and
    # the timer sharing one thread, which is what produced the "RGB and depth
    # timestamps are not synchronized" warnings the depth group was split out
    # to prevent. The fourth is headroom for service/parameter callbacks.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

"""Continuously ground tracker + VLM output into the deploy ConstraintField."""

from dataclasses import replace
import math
import os

from builtin_interfaces.msg import Time as TimeMsg
import rclpy
from rclpy.node import Node
from social_perception.msg import PeopleObservations
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray
import yaml

from social_rl.field_markers import FieldMarkerRenderer
from social_rl.field_transport import field_to_json
from social_rl.ground_truth import GroundTruthPeople
from social_rl.observation import ObservationConfig
from social_rl.ros_interface import (EnvConfig, PerceptionBridge,
                                     inverse_transform_point,
                                     quaternion_to_yaw, transform_point)


ZONE_RETENTION_SECONDS = 100.0


def _config_path(model_path: str, configured_path: str) -> str:
    path = os.path.expanduser(configured_path)
    if path:
        return path
    path = os.path.join(os.path.dirname(model_path), 'env_config.yaml')
    if os.path.exists(path):
        return path
    return os.path.join(
        os.path.dirname(os.path.dirname(model_path)), 'env_config.yaml')


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _transform_field(field, transform, target_frame: str, *, inverse: bool):
    """Move zone geometry between the robot and fixed odometry frames."""
    yaw = quaternion_to_yaw(transform.rotation)
    angle_delta = -yaw if inverse else yaw
    zones = []
    for zone in field.zones:
        samples = []
        for sample in zone.trajectory_of_zone:
            center = (inverse_transform_point(transform, *sample.center)
                      if inverse else
                      transform_point(transform, *sample.center))
            samples.append(replace(
                sample,
                center=center,
                orientation=_wrap_angle(sample.orientation + angle_delta)))
        zones.append(replace(zone, trajectory_of_zone=tuple(samples)))
    return replace(field, frame=target_frame, zones=tuple(zones))


class LockedTalkingField:
    """Keep a talking zone fixed after either member leaves the camera."""

    def __init__(self, goal_frame: str, robot_frame: str,
                 retention_seconds: float = ZONE_RETENTION_SECONDS):
        self._goal_frame = goal_frame
        self._robot_frame = robot_frame
        self._retention_seconds = retention_seconds
        self._locked = None
        self._last_observed_at = None

    @property
    def locked(self) -> bool:
        return self._locked is not None

    def try_lock(self, field, goal_to_robot, now: float,
                 observed_at=None) -> bool:
        """Store only the first hard talking zone in the fixed goal frame."""
        if self.locked:
            return False
        talking_zone = next((
            zone for zone in field.zones
            if zone.scene_type == 'talking' and zone.hardness == 'hard'), None)
        if talking_zone is None:
            return False
        if observed_at is not None:
            # Memory can still contain a pair after its hold expires. Require
            # both IDs to have been seen recently before accepting it again.
            if any(str(track_id) not in observed_at
                   for track_id in talking_zone.track_ids):
                return False
            age = now - self._pair_last_seen(talking_zone, observed_at, now)
            if not -0.25 <= age < self._retention_seconds:
                return False
        talking_field = replace(field, zones=(talking_zone,))
        self._locked = _transform_field(
            talking_field, goal_to_robot, self._goal_frame, inverse=True)
        self._last_observed_at = self._pair_last_seen(
            talking_zone, observed_at, now)
        return True

    @staticmethod
    def _pair_last_seen(zone, observed_at, now):
        # The pair stops being fully visible when its first member disappears.
        if observed_at is None:
            return float(now)
        stamps = [observed_at.get(str(track_id), now)
                  for track_id in zone.track_ids]
        return min(float(now), *(float(stamp) for stamp in stamps))

    def note_observations(self, observed_at, now: float) -> None:
        """Extend the hold only while both original track IDs reach the camera."""
        if (self.locked and all(str(track_id) in observed_at
                                for track_id in self._locked.zones[0].track_ids)):
            last_seen = self._pair_last_seen(
                self._locked.zones[0], observed_at, now)
            self._last_observed_at = max(self._last_observed_at, last_seen)

    def expire_if_due(self, now: float) -> bool:
        """Clear the zone after the configured camera-loss interval."""
        if not self.locked:
            return False
        elapsed = float(now) - self._last_observed_at
        if 0.0 <= elapsed < self._retention_seconds:
            return False
        self._locked = None
        self._last_observed_at = None
        return True

    def for_robot(self, goal_to_robot):
        """Express the fixed world zone in the moving robot frame for RL."""
        if not self.locked:
            raise RuntimeError('no talking constraint has been locked yet')
        return _transform_field(
            self._locked, goal_to_robot, self._robot_frame, inverse=False)

    def for_visualization(self):
        """Return the same locked zone in its stable odometry frame."""
        if not self.locked:
            raise RuntimeError('no talking constraint has been locked yet')
        return self._locked


def waiting_only_field(field, observed_at=None, now=None,
                       retention_seconds=ZONE_RETENTION_SECONDS):
    """Keep VLM-confirmed waiting zones until the camera-loss interval ends.

    A talking constraint is latched because it protects the space between a
    pair.  A waiting zone belongs to one tracked person, so it remains live and
    follows the tracker instead of being locked in the odometry frame.
    """
    def retained(zone):
        if zone.scene_type != 'waiting':
            return False
        if observed_at is None:
            return True
        stamp = observed_at.get(str(zone.track_ids[0])) if zone.track_ids else None
        if stamp is None:
            return False
        # Gazebo camera stamps can run slightly ahead of the last /clock tick.
        age = float(now) - float(stamp)
        return -0.25 <= age < retention_seconds

    return replace(field, zones=tuple(zone for zone in field.zones
                                      if retained(zone)))


class DeploymentConstraintFieldNode(Node):
    """Own the live deploy field independently of goal and policy state."""

    def __init__(self):
        super().__init__('social_rl_constraint_field')
        self.declare_parameter('model_path', '')
        self.declare_parameter('env_config', '')
        self.declare_parameter('people_topic_override', '')
        self.declare_parameter(
            'field_topic', '/social_rl/constraint_field')
        self.declare_parameter('publish_zone_markers', True)
        self.declare_parameter(
            'zone_markers_topic', '/social_rl/zone_markers')
        # Seconds after the last RGB observation; finite and > 0, default 100.
        # This holds waiting/talking zones for navigation and RViz, not /people.
        self.declare_parameter('zone_retention_seconds', ZONE_RETENTION_SECONDS)
        # Nonempty ROS topic name; default /people_observations. Only actual
        # detections here reset the hold clock, never coasted /people tracks.
        self.declare_parameter('observations_topic', '/people_observations')

        self._retention_seconds = float(
            self.get_parameter('zone_retention_seconds').value)
        if not math.isfinite(self._retention_seconds) or self._retention_seconds <= 0.0:
            raise ValueError('zone_retention_seconds must be finite and > 0 s')
        observations_topic = str(
            self.get_parameter('observations_topic').value).strip()
        if not observations_topic:
            raise ValueError('observations_topic must be a nonempty ROS topic')
        self._observed_at = {}

        model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value))
        configured_path = str(self.get_parameter('env_config').value)
        config_path = _config_path(model_path, configured_path)
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(
                f'model_path "{model_path}" does not exist')
        if not os.path.exists(config_path):
            raise RuntimeError(
                f'no env_config.yaml found for model "{model_path}"')

        with open(config_path, 'r') as handle:
            saved = yaml.safe_load(handle) or {}
        env_config = EnvConfig.from_dict(saved.get('env', {}))
        people_topic_override = str(
            self.get_parameter('people_topic_override').value)
        if people_topic_override:
            env_config.people_topic = people_topic_override
        observation_config = ObservationConfig.from_dict(
            saved.get('observation', {}))

        # ``people:=ground_truth`` is an explicitly simulation-only launch
        # mode.  It must follow the same Gazebo scenario labels and visibility
        # rules as training/evaluation; routing that topic through the normal
        # VLM cache leaves every ``waiting`` label empty.  The real deployment
        # path remains tracker + VLM below.
        self._simulation_ground_truth = (
            people_topic_override == env_config.ground_truth_people_topic)
        if not self._simulation_ground_truth:
            # /people includes coasted tracks. Memory must outlive that coast,
            # while /people_observations sets the actual camera-loss clock.
            env_config.people_memory_time = self._retention_seconds
            env_config.people_memory_time_still = self._retention_seconds
            self.create_subscription(
                PeopleObservations, observations_topic,
                self._on_observations, 10)
        self._ground_truth = (
            GroundTruthPeople(self, env_config)
            if self._simulation_ground_truth else None)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._bridge = PerceptionBridge(
            self, self._tf_buffer, env_config, observation_config,
            people_provider=self._ground_truth,
            # The ground-truth path needs the scan only for the same occlusion
            # filtering used by training.  The real robot field remains
            # independent of motion messages.
            subscribe_motion=self._simulation_ground_truth,
            subscribe_social=not self._simulation_ground_truth)
        self._field_latch = LockedTalkingField(
            env_config.goal_frame, env_config.robot_frame,
            self._retention_seconds)
        self._field_pub = self.create_publisher(
            String, str(self.get_parameter('field_topic').value), 10)
        self._marker_pub = None
        if bool(self.get_parameter('publish_zone_markers').value):
            self._marker_pub = self.create_publisher(
                MarkerArray,
                str(self.get_parameter('zone_markers_topic').value), 10)
        self._marker_renderer = FieldMarkerRenderer()
        self.create_timer(env_config.control_period, self._publish_field)
        self.get_logger().info(
            ('ground-truth field mirrors training: '
             f'{env_config.ground_truth_people_topic} + Gazebo scenario -> '
             f'{self.get_parameter("field_topic").value}'
             if self._simulation_ground_truth else
             f'waiting/talking zones retained for '
             f'{self._retention_seconds:.1f} s after camera loss: '
             f'{env_config.people_topic} + '
             f'{env_config.vlm_states_topic} -> '
             f'{self.get_parameter("field_topic").value}'))

    def _on_observations(self, message: PeopleObservations):
        # Unlike /people, this message contains detections from the RGB frame
        # only. Use its camera stamp so coasting cannot restart the 100 s hold.
        stamp = (float(message.header.stamp.sec)
                 + float(message.header.stamp.nanosec) * 1e-9)
        for person in message.people:
            self._observed_at[person.id] = stamp

    def _publish_field(self):
        try:
            now = self.get_clock().now().nanoseconds * 1e-9
            if not self._simulation_ground_truth:
                self._field_latch.note_observations(self._observed_at, now)
            if self._field_latch.expire_if_due(now):
                # Keep unrelated waiting memory. try_lock() rejects this old
                # pair until the camera actually sees both IDs again.
                self.get_logger().info(
                    'talking constraint expired after camera-loss retention')
            if self._simulation_ground_truth:
                # Training does not discard soft zones: a ``waiting`` person
                # is a soft Gaussian, but it is still part of the CNN input.
                # Publishing the full field here keeps the simulation test
                # faithful to the checkpoint rather than to the real VLM
                # deployment policy below.
                field = self._bridge.deployment_constraint_field()
            elif not self._field_latch.locked:
                candidate = self._bridge.deployment_constraint_field()
                goal_to_robot = self._bridge.goal_transform()
                if not self._field_latch.try_lock(
                        candidate, goal_to_robot, now, self._observed_at):
                    # A single waiting person has a soft personal zone. Keep
                    # it live for safe navigation and RViz; only a talking
                    # pair needs the fixed-world latch above.
                    field = waiting_only_field(
                        candidate, self._observed_at, now,
                        self._retention_seconds)
                else:
                    self.get_logger().info(
                        'locked new talking constraint; holding it until '
                        f'{self._retention_seconds:.1f} s after the pair leaves '
                        'the camera')
                    field = self._field_latch.for_robot(goal_to_robot)
            else:
                # Only TF remains live: the zone stays fixed in odom while the
                # robot-frame coordinates change as the robot moves.
                field = self._field_latch.for_robot(
                    self._bridge.goal_transform())
        except RuntimeError as error:
            self.get_logger().warn(str(error), throttle_duration_sec=5.0)
            return

        message = String()
        message.data = field_to_json(field)
        self._field_pub.publish(message)
        if self._marker_pub is not None:
            marker_field = (
                self._field_latch.for_visualization()
                if self._field_latch.locked else field)
            # The locked geometry is already fixed in odom. A zero stamp asks
            # RViz for the latest map -> odom transform instead of requiring
            # an exact transform at this timer tick, which avoids extrapolation
            # drops while leaving the RL field and policy input unchanged.
            markers = self._marker_renderer.render(
                marker_field, TimeMsg())
            self._marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DeploymentConstraintFieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

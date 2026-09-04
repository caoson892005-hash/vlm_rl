#!/usr/bin/env python3
"""Block F: the CBF/QP safety shield between the policy and the wheels.

    /cmd_vel                nominal (v, w) from block E
    /social_rl/constraint_field   block D's zones
    /people                 block B: where people are and where they are going
    /scan                   block H: what the lidar can see
        -> /cmd_vel_safe    the closest command to the nominal one that keeps
                            every barrier from being crossed

This is NOT social_velocity_filter. That one scales the command down as a
person gets nearer and stops below a threshold, which can only ever slow the
robot along the line it was already driving. A shield built on control barrier
functions can also STEER: the constraint is on the rate of change of a barrier,
so going around a region at speed satisfies it exactly where driving into it
slowly does not. Going round a conversation rather than creeping into it is the
behaviour this whole stack exists to produce, and a scalar speed limiter cannot
express it.

WHY THE QP IS SOLVED HERE INSTEAD OF BY A LIBRARY

There are two decision variables. The feasible set is an intersection of half
planes in a plane, so the optimum is the unconstrained point, or its projection
onto one constraint, or the intersection of two -- an exhaustive check over a
few hundred candidates, exact and with no iteration count to tune. osqp,
qpsolvers, cvxpy and quadprog are all absent from this machine and everything
here has to run offline, so pulling one in was not on the table anyway; but
even with one available, a 2-variable QP is a case where the enumeration is
both simpler and more predictable than a general solver.

THE UNICYCLE PROBLEM, AND THE LOOK-AHEAD POINT

The point at the wheel axis moves at (v, 0) in the robot frame -- w does not
move it at all, instantaneously. A barrier written on that point therefore has
no term in w, and the QP cannot steer: it can only brake. The standard fix is
to protect a point a short distance in front of the axis instead, which moves
at (v, lookahead * w), so both controls appear. The cost is that the shield
guards a point ahead of the robot rather than its centre, which is why
`lookahead` should stay small relative to the clearances below.
"""

import math

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import LaserScan
from social_perception.msg import ConstraintField
from tf2_ros import Buffer, TransformException, TransformListener


def ellipse_reach(forward, sideways, front, side, rear):
    """Distance from the centre to the boundary, in one direction.

    The same piecewise ellipse block D draws with: which semi-axis applies
    along the region's own forward axis depends on which side of it you are.
    """
    along = front if forward >= 0.0 else rear
    distance = math.hypot(forward, sideways)
    if distance < 1e-9:
        return min(along, side)
    cos_angle = forward / distance
    sin_angle = sideways / distance
    denominator = (cos_angle / along) ** 2 + (sin_angle / side) ** 2
    return 1.0 / math.sqrt(max(denominator, 1e-12))


def zone_barrier(point, center, orientation, size, core):
    """How far outside a zone's solid core `point` is, in metres.

    Positive outside, zero on the contour, negative inside. The contour is the
    core, not the outer reach: the outer ellipse is what a step INTO the region
    costs the policy, while the core is the part there is no acceptable reason
    to be in. Charging a shield with the outer one would have it refuse to
    drive past anybody at all.
    """
    delta_x = point[0] - center[0]
    delta_y = point[1] - center[1]
    cos_o, sin_o = math.cos(orientation), math.sin(orientation)
    forward = delta_x * cos_o + delta_y * sin_o
    sideways = -delta_x * sin_o + delta_y * cos_o
    reach = ellipse_reach(forward, sideways, size[0], size[1], size[2])
    return math.hypot(forward, sideways) - min(core, reach)


class SocialSafetyShield(Node):
    def __init__(self):
        super().__init__('social_safety_shield')
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/cmd_vel_safe')
        self.declare_parameter('constraint_field_topic',
                               '/social_rl/constraint_field')
        self.declare_parameter('scan_topic', '/scan')

        # Barrier gain. h_dot >= -alpha * h, so alpha is how fast the shield is
        # willing to let a margin shrink: at alpha = 2.0 a 0.5 m margin may
        # close at up to 1.0 m/s, which is the whole speed range of this base.
        # Lower is more conservative and starts intervening further out.
        self.declare_parameter('alpha_hard', 2.0)
        self.declare_parameter('alpha_soft', 1.0)
        self.declare_parameter('alpha_obstacle', 2.0)

        # Whether the personal space of individuals is enforced at all. False
        # is the diagram's dashed arrow: the policy is already paid to respect
        # those, and a shield that also refuses to enter them turns every
        # narrow passage into a deadlock. The o-space of a conversation --
        # hardness `hard` -- is enforced either way.
        self.declare_parameter('enforce_soft_zones', False)

        self.declare_parameter('lookahead', 0.25)
        self.declare_parameter('robot_radius', 0.30)
        self.declare_parameter('maximum_linear_speed', 0.5)
        self.declare_parameter('maximum_angular_speed', 1.0)
        self.declare_parameter('allow_reverse', False)
        # How many lidar returns become constraints. The nearest ones are the
        # only ones that can bind, and every extra point squares into the pair
        # enumeration below.
        self.declare_parameter('scan_constraints', 12)
        self.declare_parameter('scan_maximum_range', 3.0)
        # Returns nearer than this are the robot looking at itself. Measured on
        # lirs_test.world: 170 of 360 beams come back under 0.30 m, and the
        # pattern is mirror-symmetric about the robot's centreline -- the world
        # is not, so that is the chassis, not an obstacle. Left in, every one
        # of them is a barrier already violated by 0.2 m, the QP is infeasible
        # from the first message, and the shield brakes to a stop in an empty
        # room. Above the observed 0.10-0.30 m self-hit band by a margin.
        self.declare_parameter('scan_minimum_range', 0.35)
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('transform_timeout', 0.05)
        self.declare_parameter('field_timeout', 1.0)
        self.declare_parameter('scan_timeout', 0.5)
        self.declare_parameter('input_command_timeout', 0.35)
        self.declare_parameter('stop_broadcast_time', 1.0)
        self.declare_parameter('rate', 20.0)
        # Turning is cheaper to give up than driving: the two enter the cost
        # in different units, and weighting them equally makes a 1 rad/s
        # correction look as expensive as abandoning half the linear speed.
        self.declare_parameter('linear_weight', 1.0)
        self.declare_parameter('angular_weight', 0.25)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.alpha_hard = float(self.get_parameter('alpha_hard').value)
        self.alpha_soft = float(self.get_parameter('alpha_soft').value)
        self.alpha_obstacle = float(self.get_parameter('alpha_obstacle').value)
        self.enforce_soft = bool(self.get_parameter('enforce_soft_zones').value)
        self.lookahead = max(1e-3, float(self.get_parameter('lookahead').value))
        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.max_linear = float(self.get_parameter('maximum_linear_speed').value)
        self.max_angular = float(self.get_parameter('maximum_angular_speed').value)
        self.allow_reverse = bool(self.get_parameter('allow_reverse').value)
        self.scan_constraints = int(self.get_parameter('scan_constraints').value)
        self.scan_max_range = float(self.get_parameter('scan_maximum_range').value)
        self.scan_min_range = float(self.get_parameter('scan_minimum_range').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)
        self.transform_timeout = float(
            self.get_parameter('transform_timeout').value)
        self.field_timeout = float(self.get_parameter('field_timeout').value)
        self.scan_timeout = float(self.get_parameter('scan_timeout').value)
        self.command_timeout = float(
            self.get_parameter('input_command_timeout').value)
        self.stop_broadcast_time = max(
            0.0, float(self.get_parameter('stop_broadcast_time').value))
        self.weights = np.array(
            [float(self.get_parameter('linear_weight').value),
             float(self.get_parameter('angular_weight').value)], dtype=float)

        sensor_qos = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.field = None
        self.field_time = None
        self.scan = None
        self.scan_time = None
        self.command = None
        self.command_time = None
        self.blocked_since = None

        self.create_subscription(
            Twist, self.input_topic, self._on_command, 10)
        self.create_subscription(
            ConstraintField,
            str(self.get_parameter('constraint_field_topic').value),
            self._on_field, 10)
        self.create_subscription(
            LaserScan, str(self.get_parameter('scan_topic').value),
            self._on_scan, sensor_qos)
        self.publisher = self.create_publisher(
            Twist, str(self.get_parameter('output_topic').value), 10)

        rate = max(1.0, float(self.get_parameter('rate').value))
        self.create_timer(1.0 / rate, self._step)
        self.get_logger().info(
            f'khối F sẵn sàng: {self.input_topic} -> '
            f'{self.get_parameter("output_topic").value}, '
            f'lookahead {self.lookahead:.2f} m, '
            f'vùng soft {"CÓ" if self.enforce_soft else "KHÔNG"} ràng buộc')

    # ------------------------------------------------------------- callbacks

    def _on_command(self, message):
        self.command = message
        self.command_time = self.get_clock().now()

    def _on_field(self, message):
        self.field = message
        self.field_time = self.get_clock().now()

    def _on_scan(self, message):
        self.scan = message
        self.scan_time = self.get_clock().now()

    def _age(self, stamp):
        if stamp is None:
            return math.inf
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    # ---------------------------------------------------------- constraints

    def zone_constraints(self):
        """One CBF row per zone, from the t = 0.0 sample and its own velocity.

        A zone's velocity is read off its own trajectory rather than from the
        people underneath it, because a group zone has no single person to
        borrow one from: it is the midpoint of two, and it moves when either of
        them does.
        """
        rows, bounds, labels = [], [], []
        if self.field is None or self._age(self.field_time) > self.field_timeout:
            return rows, bounds, labels
        point = (self.lookahead, 0.0)
        for zone in self.field.zones:
            if zone.hardness != 'hard' and not self.enforce_soft:
                continue
            if not zone.trajectory_of_zone:
                continue
            now = zone.trajectory_of_zone[0]
            center = (float(now.center[0]), float(now.center[1]))
            size = (float(now.size[0]), float(now.size[1]), float(now.size[2]))
            barrier = zone_barrier(point, center, now.orientation, size, now.core)
            gradient = self._barrier_gradient(
                point, center, now.orientation, size, now.core)
            velocity = self._zone_velocity(zone)
            alpha = (self.alpha_hard if zone.hardness == 'hard'
                     else self.alpha_soft)
            rows.append([-gradient[0], -self.lookahead * gradient[1]])
            bounds.append(alpha * barrier
                          - (gradient[0] * velocity[0] + gradient[1] * velocity[1]))
            labels.append(f'{zone.zone_id}({zone.hardness})')
        return rows, bounds, labels

    def _barrier_gradient(self, point, center, orientation, size, core):
        """Central difference. The analytic form of the piecewise-ellipse reach
        needs a case split on the sign of `forward` and another on whether the
        core or the reach is the nearer contour, and a wrong sign in one branch
        is a shield that pushes the robot INTO a region. A 1 mm difference costs
        four evaluations of a closed-form expression and cannot get that wrong.
        """
        step = 1e-3
        gradient = []
        for axis in (0, 1):
            ahead = list(point)
            behind = list(point)
            ahead[axis] += step
            behind[axis] -= step
            gradient.append(
                (zone_barrier(ahead, center, orientation, size, core)
                 - zone_barrier(behind, center, orientation, size, core))
                / (2.0 * step))
        return gradient

    def _zone_velocity(self, zone):
        samples = zone.trajectory_of_zone
        if len(samples) < 2:
            return (0.0, 0.0)
        span = float(samples[1].t) - float(samples[0].t)
        if span <= 1e-9:
            return (0.0, 0.0)
        return ((float(samples[1].center[0]) - float(samples[0].center[0])) / span,
                (float(samples[1].center[1]) - float(samples[0].center[1])) / span)

    def obstacle_constraints(self):
        """One CBF row per nearby lidar return. Block H's half of the shield.

        Static, so the relative velocity term drops out. People are already
        covered by the zones and would be double counted here, except that the
        actors in lirs_test.world carry no collision proxy, so /scan passes
        straight through them -- which is exactly why the zones have to exist.
        """
        rows, bounds, labels = [], [], []
        if self.scan is None or self._age(self.scan_time) > self.scan_timeout:
            return rows, bounds, labels
        points = self.scan_points()
        if points is None or points.shape[0] == 0:
            return rows, bounds, labels
        offsets = points - np.array([self.lookahead, 0.0])
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        nearest = np.argsort(distances)[:self.scan_constraints]
        for index in nearest:
            distance = float(distances[index])
            barrier = distance - self.robot_radius
            # Unit vector from the obstacle towards the protected point, which
            # is the direction the barrier grows in.
            gradient = (float(offsets[index, 0]) / max(distance, 1e-6),
                        float(offsets[index, 1]) / max(distance, 1e-6))
            rows.append([-gradient[0], -self.lookahead * gradient[1]])
            bounds.append(self.alpha_obstacle * barrier)
            labels.append(f'scan@{distance:.2f}m')
        return rows, bounds, labels

    def scan_points(self):
        """Usable laser returns as (N, 2) points in the robot frame.

        Carried across TF rather than assumed to start at the robot origin: the
        lidar sits forward of the wheel axis, and reading its ranges as if they
        began at base_link puts every obstacle that offset closer than it is.
        Measured 0.007 m in this simulation, which is nothing -- and exactly
        why it has to be a lookup rather than a constant, because the same code
        runs on a base where the mount is somewhere else entirely.
        """
        scan = self.scan
        ranges = np.asarray(scan.ranges, dtype=float)
        if ranges.size == 0:
            return None
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
        # The lower bound is scan_minimum_range, not scan.range_min: the sensor
        # reports what it can measure, which here includes the robot's own
        # body. See the parameter's comment for the measurement.
        usable = (np.isfinite(ranges)
                  & (ranges >= max(self.scan_min_range, scan.range_min))
                  & (ranges <= min(self.scan_max_range, scan.range_max)))
        if not np.any(usable):
            return None
        ranges = ranges[usable]
        angles = angles[usable]
        local = np.stack([ranges * np.cos(angles), ranges * np.sin(angles)], 1)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame, scan.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.transform_timeout))
        except TransformException as error:
            # Dropping the obstacle constraints entirely would leave the shield
            # guarding people and nothing else, which is worse than useless as
            # a last line of defence, so this is loud rather than quiet.
            self.get_logger().warn(
                f'TF {scan.header.frame_id} -> {self.robot_frame}: {error}. '
                f'Bỏ ràng buộc vật cản ở chu kỳ này.',
                throttle_duration_sec=5.0)
            return None
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y ** 2 + rotation.z ** 2))
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        translation = transform.transform.translation
        return np.stack([
            translation.x + cos_yaw * local[:, 0] - sin_yaw * local[:, 1],
            translation.y + sin_yaw * local[:, 0] + cos_yaw * local[:, 1]], 1)

    # ------------------------------------------------------------------ QP

    def solve(self, nominal, rows, bounds):
        """min ||u - nominal||^2_W subject to A u <= b and the box limits.

        Exhaustive over the only places a 2-D optimum can sit: the
        unconstrained point, the projection onto each single constraint, and
        every pair intersection. Returns None when the feasible set is empty.
        """
        minimum_linear = -self.max_linear if self.allow_reverse else 0.0
        box_rows = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        box_bounds = [self.max_linear, -minimum_linear,
                      self.max_angular, self.max_angular]
        matrix = np.array(rows + box_rows, dtype=float)
        limits = np.array(bounds + box_bounds, dtype=float)

        candidates = [nominal]
        inverse = 1.0 / self.weights
        for index in range(matrix.shape[0]):
            row = matrix[index]
            scale = float(row @ (inverse * row))
            if scale < 1e-12:
                continue
            # Projection of the nominal command onto {u : row @ u = limit},
            # measured in the weighted norm the cost uses.
            offset = (float(row @ nominal) - limits[index]) / scale
            candidates.append(nominal - offset * inverse * row)
        for first in range(matrix.shape[0]):
            for second in range(first + 1, matrix.shape[0]):
                pair = matrix[[first, second]]
                determinant = float(pair[0, 0] * pair[1, 1]
                                    - pair[0, 1] * pair[1, 0])
                if abs(determinant) < 1e-12:
                    continue
                candidates.append(
                    np.linalg.solve(pair, limits[[first, second]]))

        best, best_cost = None, math.inf
        for candidate in candidates:
            # A tolerance, not zero: a pair intersection sits exactly ON its
            # two constraints, and floating point puts it a few ulps outside
            # half the time.
            if np.any(matrix @ candidate > limits + 1e-9):
                continue
            error = candidate - nominal
            cost = float(error @ (self.weights * error))
            if cost < best_cost:
                best, best_cost = candidate, cost
        return best

    # ---------------------------------------------------------------- loop

    def _step(self):
        if self.command is None:
            # Nothing has ever driven through this shield, so it owns nothing.
            # Publishing zero here would put the shield on output_topic from
            # the second it launches, on top of whoever else writes that topic.
            return
        if self._age(self.command_time) > self.command_timeout:
            # Nobody is driving any more. Publishing zero rather than nothing is
            # what makes this a shield: a policy that dies mid-episode must not
            # leave its last command running on the wheels. The burst ends after
            # stop_broadcast_time -- long enough to stop the wheels, short
            # enough that a policy started once and closed does not leave the
            # shield overwriting the next process to write output_topic.
            if (self._age(self.command_time)
                    <= self.command_timeout + self.stop_broadcast_time):
                self.publisher.publish(Twist())
            return

        nominal = np.array([self.command.linear.x, self.command.angular.z],
                           dtype=float)
        rows, bounds, labels = self.zone_constraints()
        obstacle_rows, obstacle_bounds, obstacle_labels = self.obstacle_constraints()
        rows += obstacle_rows
        bounds += obstacle_bounds
        labels += obstacle_labels

        solution = self.solve(nominal, rows, bounds)
        command = Twist()
        if solution is None:
            # Every barrier cannot be held at once -- somebody walked into the
            # robot, or it is already inside a region. Stopping is the correct
            # answer for THIS base rather than a generic one: it has no rear
            # sensor, allow_reverse is false in every profile, so backing out
            # of the situation would be reversing blind.
            if self.blocked_since is None:
                self.blocked_since = self.get_clock().now()
            self.get_logger().warn(
                f'không có lệnh nào thoả mọi ràng buộc ({len(rows)} ràng '
                f'buộc), dừng tại chỗ. Gần nhất: {labels[:3]}',
                throttle_duration_sec=2.0)
        else:
            self.blocked_since = None
            command.linear.x = float(solution[0])
            command.angular.z = float(solution[1])
            change = float(np.hypot(*(solution - nominal)))
            if change > 1e-3:
                self.get_logger().info(
                    f'khối F sửa lệnh ({nominal[0]:+.3f}, {nominal[1]:+.3f}) '
                    f'-> ({solution[0]:+.3f}, {solution[1]:+.3f}) '
                    f'vì {len(rows)} ràng buộc',
                    throttle_duration_sec=2.0)
        self.publisher.publish(command)


    def stop(self):
        """Leave the wheels stopped, not running the last command forwarded."""
        self.publisher.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = SocialSafetyShield()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C and a launch shutdown both reach here with the context already
        # torn down, and publishing then raises out of the handler instead of
        # stopping the wheels. The zero that matters was already sent by the
        # watchdog burst; this is the clean-exit case only.
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        # SIGTERM from a launch shutdown gets here with the context already
        # torn down, and calling shutdown twice raises out of the handler.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

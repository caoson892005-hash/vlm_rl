#!/usr/bin/env python3
"""Move Gazebo person models along arbitrary time-stamped waypoint paths."""

import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
import yaml


def shortest_angle(start, end):
    return start + math.atan2(math.sin(end - start), math.cos(end - start))


class PeopleMotionController(Node):
    def __init__(self):
        super().__init__('people_motion_controller')
        default_file = str(Path(get_package_share_directory('social_navigation')) /
                           'config' / 'people_paths.yaml')
        self.declare_parameter('trajectory_file', default_file)
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('face_group_members', True)
        self.declare_parameter('group_distance', 1.5)
        trajectory_file = self.get_parameter('trajectory_file').value
        with open(trajectory_file, 'r', encoding='utf-8') as stream:
            document = yaml.safe_load(stream) or {}
        self.paths = document.get('people', {})
        self.pending = set()
        self.loop = bool(self.get_parameter('loop').value)
        self.face_group_members = bool(
            self.get_parameter('face_group_members').value)
        self.group_distance = max(
            0.0, float(self.get_parameter('group_distance').value))
        self.client = self.create_client(SetEntityState, '/set_entity_state')
        self.start_time = self.get_clock().now()
        rate = max(1.0, float(self.get_parameter('update_rate').value))
        self.timer = self.create_timer(1.0 / rate, self.update)
        self.warned = False
        self.get_logger().info(f'Loaded trajectories for: {", ".join(self.paths)}')

    def update(self):
        if not self.client.service_is_ready():
            if not self.warned:
                self.get_logger().warn('/set_entity_state is unavailable; load libgazebo_ros_state.so')
                self.warned = True
            return
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        states = {}
        for name, data in self.paths.items():
            waypoints = sorted(data.get('waypoints', []), key=lambda item: float(item['time']))
            if not waypoints:
                continue
            duration = float(waypoints[-1]['time'])
            path_time = elapsed % duration if self.loop and duration > 0.0 else min(elapsed, duration)
            states[name] = self.interpolate(waypoints, path_time)

        if self.face_group_members:
            self.orient_groups(states, self.group_distance)

        for name, (pose, velocity) in states.items():
            if name in self.pending:
                continue
            state = EntityState()
            state.name = name
            state.reference_frame = 'world'
            state.pose.position.x, state.pose.position.y, state.pose.position.z = pose[:3]
            half_yaw = pose[3] * 0.5
            state.pose.orientation.z = math.sin(half_yaw)
            state.pose.orientation.w = math.cos(half_yaw)
            state.twist.linear.x, state.twist.linear.y = velocity
            request = SetEntityState.Request()
            request.state = state
            self.pending.add(name)
            future = self.client.call_async(request)
            future.add_done_callback(
                lambda _, entity=name: self.pending.discard(entity))

    @staticmethod
    def orient_groups(states, group_distance):
        """Turn every member of a proximity group toward its group center."""
        names = list(states)
        unvisited = set(names)
        while unvisited:
            seed = unvisited.pop()
            component = {seed}
            queue = [seed]
            while queue:
                current = queue.pop()
                cx, cy = states[current][0][:2]
                neighbours = {
                    name for name in unvisited
                    if math.hypot(states[name][0][0] - cx,
                                  states[name][0][1] - cy) <= group_distance
                }
                unvisited.difference_update(neighbours)
                component.update(neighbours)
                queue.extend(neighbours)

            if len(component) < 2:
                continue
            center_x = sum(states[name][0][0] for name in component) / len(component)
            center_y = sum(states[name][0][1] for name in component) / len(component)
            for name in component:
                pose, velocity = states[name]
                yaw = math.atan2(center_y - pose[1], center_x - pose[0])
                states[name] = (pose[:3] + (yaw,), velocity)

    @staticmethod
    def interpolate(waypoints, current_time):
        if len(waypoints) == 1:
            point = waypoints[0]
            return (float(point['x']), float(point['y']), float(point.get('z', 0.0)),
                    float(point.get('yaw', 0.0))), (0.0, 0.0)
        end_index = next((i for i, point in enumerate(waypoints)
                          if float(point['time']) >= current_time), len(waypoints) - 1)
        end_index = max(1, end_index)
        start, end = waypoints[end_index - 1], waypoints[end_index]
        start_time, end_time = float(start['time']), float(end['time'])
        segment_time = max(end_time - start_time, 1e-6)
        ratio = min(1.0, max(0.0, (current_time - start_time) / segment_time))
        sx, sy = float(start['x']), float(start['y'])
        ex, ey = float(end['x']), float(end['y'])
        yaw_start = float(start.get('yaw', math.atan2(ey - sy, ex - sx)))
        yaw_end = shortest_angle(yaw_start, float(end.get('yaw', yaw_start)))
        pose = (sx + ratio * (ex - sx), sy + ratio * (ey - sy),
                float(start.get('z', 0.0)) + ratio *
                (float(end.get('z', 0.0)) - float(start.get('z', 0.0))),
                yaw_start + ratio * (yaw_end - yaw_start))
        return pose, ((ex - sx) / segment_time, (ey - sy) / segment_time)


def main(args=None):
    rclpy.init(args=args)
    node = PeopleMotionController()
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

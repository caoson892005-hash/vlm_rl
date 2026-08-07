#!/usr/bin/env python3
"""Detect proximity groups and publish O-P-R visualization markers."""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from social_navigation.msg import Group, Groups, People
from visualization_msgs.msg import Marker, MarkerArray


class PeopleGroupDetector(Node):
    def __init__(self):
        super().__init__('people_group_detector')
        self.declare_parameter('group_distance', 1.5)
        self.declare_parameter('formation_prediction_horizon', 9.0)
        self.declare_parameter('formation_distance', 1.6)
        self.declare_parameter('minimum_closing_speed', 0.1)
        self.declare_parameter('forming_p_radius', 0.9)
        self.declare_parameter('forming_r_radius', 1.7)
        self.declare_parameter('minimum_group_size', 2)
        self.declare_parameter('o_space_min_radius', 0.45)
        self.declare_parameter('p_space_margin', 0.45)
        self.declare_parameter('r_space_margin', 1.0)
        self.declare_parameter('individual_o_radius', 0.38)
        self.declare_parameter('individual_p_front_radius', 0.9)
        self.declare_parameter('individual_p_side_radius', 0.65)
        self.declare_parameter('individual_p_rear_radius', 0.5)
        self.declare_parameter('individual_r_front_radius', 1.5)
        self.declare_parameter('individual_r_side_radius', 1.0)
        self.declare_parameter('individual_r_rear_radius', 0.75)
        self.distance = float(self.get_parameter('group_distance').value)
        self.prediction_horizon = float(
            self.get_parameter('formation_prediction_horizon').value)
        self.formation_distance = float(self.get_parameter('formation_distance').value)
        self.minimum_closing_speed = float(
            self.get_parameter('minimum_closing_speed').value)
        self.forming_p_radius = float(self.get_parameter('forming_p_radius').value)
        self.forming_r_radius = float(self.get_parameter('forming_r_radius').value)
        self.minimum_size = int(self.get_parameter('minimum_group_size').value)
        self.o_min = float(self.get_parameter('o_space_min_radius').value)
        self.p_margin = float(self.get_parameter('p_space_margin').value)
        self.r_margin = float(self.get_parameter('r_space_margin').value)
        self.individual_o = float(self.get_parameter('individual_o_radius').value)
        self.p_front = float(self.get_parameter('individual_p_front_radius').value)
        self.p_side = float(self.get_parameter('individual_p_side_radius').value)
        self.p_rear = float(self.get_parameter('individual_p_rear_radius').value)
        self.r_front = float(self.get_parameter('individual_r_front_radius').value)
        self.r_side = float(self.get_parameter('individual_r_side_radius').value)
        self.r_rear = float(self.get_parameter('individual_r_rear_radius').value)
        self.group_pub = self.create_publisher(Groups, '/people_groups', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/social_spaces', 10)
        self.subscription = self.create_subscription(People, '/people', self.people_callback, 10)

    def people_callback(self, people_msg):
        components = self.connected_components(people_msg.people)
        groups_msg = Groups()
        groups_msg.header = people_msg.header
        markers = MarkerArray()
        marker_id = 0
        grouped_indices = {
            index for component in components if len(component) >= self.minimum_size
            for index in component
        }

        # Always draw individual O-P-R regions. SocialLayer keeps avoiding every
        # person internally, but hide member markers while a shared group exists
        # to keep RViz readable. They reappear as soon as the group disperses.
        for index, person in enumerate(people_msg.people):
            if index in grouped_indices:
                continue
            for front, side, rear, color, label in (
                    (self.individual_o, self.individual_o, self.individual_o,
                     (1.0, 0.0, 0.0, 0.95), 'O'),
                    (self.p_front, self.p_side, self.p_rear,
                     (1.0, 0.45, 0.0, 0.85), 'P'),
                    (self.r_front, self.r_side, self.r_rear,
                     (1.0, 0.9, 0.0, 0.70), 'R')):
                markers.markers.append(self.individual_marker(
                    people_msg.header, person, marker_id, front, side, rear, color, label))
                marker_id += 1

        group_number = 0
        for indices in components:
            if len(indices) < self.minimum_size:
                continue
            members = [people_msg.people[index] for index in indices]
            predicted_centers = []
            currently_close = False
            for first_index in range(len(members)):
                for second_index in range(first_index + 1, len(members)):
                    first, second = members[first_index], members[second_index]
                    separation = math.hypot(
                        first.pose.position.x - second.pose.position.x,
                        first.pose.position.y - second.pose.position.y)
                    currently_close = currently_close or separation <= self.distance
                    predicted = self.predicted_interaction(first, second)
                    if predicted is not None:
                        predicted_centers.append(predicted)

            forming = not currently_close and bool(predicted_centers)
            if forming:
                center_x = sum(item[0] for item in predicted_centers) / len(predicted_centers)
                center_y = sum(item[1] for item in predicted_centers) / len(predicted_centers)
            else:
                center_x = sum(item.pose.position.x for item in members) / len(members)
                center_y = sum(item.pose.position.y for item in members) / len(members)
            member_radius = max(math.hypot(item.pose.position.x - center_x,
                                           item.pose.position.y - center_y)
                                for item in members)
            group = Group()
            group.id = f'group_{group_number}'
            group.member_ids = [item.id for item in members]
            group.center.x, group.center.y, group.center.z = center_x, center_y, 0.03
            if forming:
                # A predicted interaction must trigger a detour early without
                # trapping a robot that is already crossing the future center.
                group.o_radius = 0.0
                group.p_radius = self.forming_p_radius
                group.r_radius = self.forming_r_radius
            else:
                group.o_radius = max(self.o_min, member_radius * 0.5)
                group.p_radius = max(group.o_radius + 0.15, member_radius + self.p_margin)
                group.r_radius = group.p_radius + self.r_margin
            groups_msg.groups.append(group)
            for radius, color, label in (
                    (group.o_radius, (1.0, 0.0, 0.0, 0.9), 'O'),
                    (group.p_radius, (1.0, 0.45, 0.0, 0.75), 'P'),
                    (group.r_radius, (1.0, 0.9, 0.0, 0.55), 'R')):
                markers.markers.append(self.circle_marker(
                    people_msg.header, group, marker_id, radius, color, label))
                marker_id += 1
            group_number += 1

        delete_old = Marker()
        delete_old.header = people_msg.header
        delete_old.action = Marker.DELETEALL
        # DELETEALL must precede the current marker set in the same publication.
        markers.markers.insert(0, delete_old)
        self.group_pub.publish(groups_msg)
        self.marker_pub.publish(markers)

    def connected_components(self, people):
        remaining = set(range(len(people)))
        components = []
        while remaining:
            seed = remaining.pop()
            component = [seed]
            queue = [seed]
            while queue:
                current = queue.pop()
                near = [index for index in remaining
                        if self.people_are_connected(people[current], people[index])]
                for index in near:
                    remaining.remove(index)
                    component.append(index)
                    queue.append(index)
            components.append(component)
        return components

    def people_are_connected(self, first, second):
        separation = math.hypot(
            first.pose.position.x - second.pose.position.x,
            first.pose.position.y - second.pose.position.y)
        return separation <= self.distance or self.predicted_interaction(first, second) is not None

    def predicted_interaction(self, first, second):
        """Return the future midpoint when two people are converging, else None."""
        rx = second.pose.position.x - first.pose.position.x
        ry = second.pose.position.y - first.pose.position.y
        vx = second.velocity.linear.x - first.velocity.linear.x
        vy = second.velocity.linear.y - first.velocity.linear.y
        separation = math.hypot(rx, ry)
        relative_speed_sq = vx * vx + vy * vy
        if separation < 1e-6 or relative_speed_sq < 1e-6:
            return None

        closing_speed = -(rx * vx + ry * vy) / separation
        if closing_speed < self.minimum_closing_speed:
            return None

        closest_time = -(rx * vx + ry * vy) / relative_speed_sq
        if closest_time < 0.0 or closest_time > self.prediction_horizon:
            return None

        first_x = first.pose.position.x + first.velocity.linear.x * closest_time
        first_y = first.pose.position.y + first.velocity.linear.y * closest_time
        second_x = second.pose.position.x + second.velocity.linear.x * closest_time
        second_y = second.pose.position.y + second.velocity.linear.y * closest_time
        predicted_separation = math.hypot(second_x - first_x, second_y - first_y)
        if predicted_separation > self.formation_distance:
            return None
        return ((first_x + second_x) * 0.5, (first_y + second_y) * 0.5)

    def individual_marker(self, header, person, marker_id, front, side, rear, color, label):
        marker = Marker()
        marker.header = header
        marker.ns = f'individual_{label}'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose = person.pose
        marker.pose.position.z = 0.04
        marker.scale.x = 0.055
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        for index in range(49):
            angle = 2.0 * math.pi * index / 48.0
            longitudinal = front if math.cos(angle) >= 0.0 else rear
            point = Point()
            point.x = longitudinal * math.cos(angle)
            point.y = side * math.sin(angle)
            marker.points.append(point)
        return marker

    @staticmethod
    def circle_marker(header, group, marker_id, radius, color, label):
        marker = Marker()
        marker.header = header
        marker.ns = f'group_{label}'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position = group.center
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.06
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        for index in range(65):
            angle = 2.0 * math.pi * index / 64.0
            point = Point()
            point.x = radius * math.cos(angle)
            point.y = radius * math.sin(angle)
            marker.points.append(point)
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = PeopleGroupDetector()
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

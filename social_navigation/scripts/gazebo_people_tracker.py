#!/usr/bin/env python3
"""Publish ground-truth Gazebo poses for configured person models."""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
from social_navigation.msg import People, Person


class GazeboPeopleTracker(Node):
    def __init__(self):
        super().__init__('gazebo_people_tracker')
        self.declare_parameter('people_names', ['person_1', 'person_2', 'person_3'])
        self.declare_parameter('gazebo_reference_frame', 'linorobot2')
        self.declare_parameter('output_frame', 'base_link')
        self.declare_parameter('publish_rate', 10.0)
        self.names = list(self.get_parameter('people_names').value)
        self.gazebo_reference_frame = self.get_parameter('gazebo_reference_frame').value
        self.output_frame = self.get_parameter('output_frame').value
        rate = max(0.5, float(self.get_parameter('publish_rate').value))
        self.client = self.create_client(GetEntityState, '/get_entity_state')
        self.publisher = self.create_publisher(People, '/people', 10)
        self.states = {}
        self.pending = set()
        self.timer = self.create_timer(1.0 / rate, self.update)
        self.get_logger().info('Tracking Gazebo people: ' + ', '.join(self.names))

    def update(self):
        if not self.client.service_is_ready():
            self.get_logger().warn('/get_entity_state is unavailable; load libgazebo_ros_state.so',
                                   throttle_duration_sec=5.0)
            return
        for name in self.names:
            if name in self.pending:
                continue
            request = GetEntityState.Request()
            request.name = name
            request.reference_frame = self.gazebo_reference_frame
            future = self.client.call_async(request)
            self.pending.add(name)
            future.add_done_callback(lambda result, model=name: self.got_state(model, result))
        self.publish_people()

    def got_state(self, name, future):
        self.pending.discard(name)
        try:
            response = future.result()
            if response.success:
                self.states[name] = response.state
        except Exception as error:  # service transport error
            self.get_logger().error(f'Cannot read {name}: {error}')

    def publish_people(self):
        message = People()
        # A zero stamp means "use the latest available TF". Gazebo state and
        # odom TF are published on adjacent simulation ticks (usually 10 ms
        # apart); stamping with now would intermittently request future TF and
        # make the social layer / RViz markers flicker.
        message.header.frame_id = self.output_frame
        for name in self.names:
            state = self.states.get(name)
            if state is None:
                continue
            person = Person()
            person.id = name
            person.pose = state.pose
            person.velocity = state.twist
            message.people.append(person)
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboPeopleTracker()
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

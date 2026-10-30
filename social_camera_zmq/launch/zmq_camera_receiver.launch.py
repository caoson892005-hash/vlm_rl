"""Run the laptop-side ZeroMQ-to-ROS camera receiver."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('social_camera_zmq'), 'config', 'laptop_receiver.yaml'
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='ROS parameter YAML for the laptop camera receiver'),
        Node(
            package='social_camera_zmq',
            executable='zmq_camera_receiver.py',
            name='zmq_camera_receiver',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])

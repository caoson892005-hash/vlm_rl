"""Run the Pi-side ZeroMQ camera sender with a parameter YAML file."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('social_camera_zmq'), 'config', 'pi_realsense_d435i.yaml'
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='ROS parameter YAML for the Pi camera sender'),
        Node(
            package='social_camera_zmq',
            executable='zmq_camera_sender.py',
            name='zmq_camera_sender',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])

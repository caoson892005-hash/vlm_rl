from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare('social_navigation'), 'config', 'yolo_depth.yaml'])
    return LaunchDescription([
        Node(
            package='social_navigation',
            executable='yolo_depth_people_detector.py',
            name='yolo_depth_people_detector',
            output='screen',
            parameters=[config],
        )
    ])

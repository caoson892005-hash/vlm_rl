from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    package_share = FindPackageShare('social_navigation')
    model_file = PathJoinSubstitution([package_share, 'models', 'person', 'model.sdf'])
    config_file = PathJoinSubstitution([package_share, 'config', 'social_navigation.yaml'])
    trajectory_file = PathJoinSubstitution([package_share, 'config', 'people_paths.yaml'])
    spawn_people = LaunchConfiguration('spawn_people')
    move_people = LaunchConfiguration('move_people')

    nodes = [
        DeclareLaunchArgument('spawn_people', default_value='true'),
        DeclareLaunchArgument('move_people', default_value='true'),
    ]

    for name, x, y, yaw in (
            ('person_1', '-0.6', '4.5', '0.0'),
            ('person_2', '0.6', '4.5', '3.14159'),
            ('person_3', '-2.2', '3.0', '0.7'),
            ('person_4', '-2.2', '-3.5', '2.4'),
            ('person_5', '0.0', '0.5', '0.0'),
            ('person_6', '5.4', '0.5', '3.14159')):
        nodes.append(Node(
            package='gazebo_ros', executable='spawn_entity.py',
            name=f'spawn_{name}', output='screen', condition=IfCondition(spawn_people),
            arguments=['-entity', name, '-file', model_file,
                       '-x', x, '-y', y, '-z', '0.0', '-Y', yaw]))

    nodes.append(TimerAction(period=3.0, actions=[
        Node(package='social_navigation', executable='gazebo_people_tracker.py',
             name='gazebo_people_tracker', output='screen', parameters=[config_file]),
        Node(package='social_navigation', executable='people_group_detector.py',
             name='people_group_detector', output='screen', parameters=[config_file]),
        Node(package='social_navigation', executable='people_motion_controller.py',
             name='people_motion_controller', output='screen', condition=IfCondition(move_people),
             parameters=[config_file, {'trajectory_file': trajectory_file}]),
        Node(package='social_navigation', executable='social_velocity_filter.py',
             name='social_velocity_filter', output='screen', parameters=[config_file]),
    ]))
    return LaunchDescription(nodes)

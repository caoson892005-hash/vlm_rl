"""Block F: the CBF/QP safety shield, between the RL policy and the wheels.

This is the CBF replacement for social_safety.launch.py, not an addition to it.
Both write /cmd_vel_safe, so running the two together means whichever published
last wins and neither guarantee holds. Pick one.

    social_safety.launch.py    social_velocity_filter -- scales the command
                               down near people. Only ever slows the robot
                               along the line it was already driving.
    social_shield.launch.py    this -- constrains the RATE OF CHANGE of a
                               barrier, so it can steer as well as brake.
                               Consumes block D's zones, so it knows the
                               difference between the middle of a conversation
                               and somebody's back.

Gazebo
    policy -> /cmd_vel -> shield -> /cmd_vel_safe -> gazebo_ros_diff_drive

        ros2 launch social_navigation social_shield.launch.py sim:=true

    Needs block D on the wire. That comes from whichever process compiles the
    zones -- the RL agent node in deployment, or the trainer during a run --
    on /social_rl/constraint_field. With no field published the shield still
    enforces the lidar constraints and passes everything else through, which
    is a real degradation and is logged, not silently absorbed.

Real robot
    micro-ROS firmware owns /cmd_vel, so the shield takes that name and the
    policy is pushed off it:

        ros2 launch social_navigation social_shield.launch.py \\
            input_topic:=/cmd_vel_policy output_topic:=/cmd_vel

    and the agent node is launched with cmd_vel_topic:=/cmd_vel_policy.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('social_navigation'),
        'config',
        'social_safety_shield.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'sim',
            default_value='true',
            description='Set to true when running against Gazebo'),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to the shield ROS parameter file'),
        DeclareLaunchArgument(
            'input_topic',
            default_value='/cmd_vel',
            description='Nominal velocity from the RL policy (block E)'),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/cmd_vel_safe',
            description='Velocity command the robot base actually consumes'),
        DeclareLaunchArgument(
            'constraint_field_topic',
            default_value='/social_rl/constraint_field',
            description="Block D's compiled zones"),
        DeclareLaunchArgument(
            'enforce_soft_zones',
            default_value='false',
            description='Also constrain individual personal space, not just '
                        'the o-space of conversations'),
        Node(
            package='social_navigation',
            executable='social_safety_shield.py',
            name='social_safety_shield',
            output='screen',
            # Tuning lives in the parameter file; wiring is appended after it
            # so a launch argument always wins.
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('sim'), value_type=bool),
                    'input_topic': LaunchConfiguration('input_topic'),
                    'output_topic': LaunchConfiguration('output_topic'),
                    'constraint_field_topic': LaunchConfiguration(
                        'constraint_field_topic'),
                    'enforce_soft_zones': ParameterValue(
                        LaunchConfiguration('enforce_soft_zones'),
                        value_type=bool),
                },
            ],
        ),
    ])

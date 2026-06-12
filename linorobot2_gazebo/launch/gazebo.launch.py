# Copyright (c) 2021 Juan Miguel Jimeno
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    use_sim_time = True

    ekf_config_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_base"), "config", "ekf.yaml"]
    )

    world_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_gazebo"), "worlds", "map2.world"]
    )

    robot_base = os.getenv('LINOROBOT2_BASE')
    urdf_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_description"), "urdf/robots", f"{robot_base}.urdf.xacro"]
    )

    description_launch_path = PathJoinSubstitution(
        [FindPackageShare('linorobot2_description'), 'launch', 'description.launch.py']
    )

    rviz_config_path = PathJoinSubstitution(
            [FindPackageShare("linorobot2_gazebo"), "rviz", "trajectory_view.rviz"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            name='paused', 
            default_value='false',
            description='Start Gazebo paused'
        ),

        DeclareLaunchArgument(
            name='rviz', 
            default_value='false', # Mặc định là bật, đổi thành 'false' nếu muốn mặc định tắt
            description='Launch RViz'
        ),

        DeclareLaunchArgument(
            name='urdf', 
            default_value=urdf_path,
            description='URDF path'
        ),

        DeclareLaunchArgument(
            name='odom_topic', 
            default_value='/odom',
            description='EKF out odometry topic'
        ),

        DeclareLaunchArgument(
            name='world', 
            default_value=world_path,
            description='Gazebo world'
        ),

        DeclareLaunchArgument(
            name='spawn_x', 
            default_value='4.0',
            description='Robot spawn position in X axis'
        ),

        DeclareLaunchArgument(
            name='spawn_y', 
            default_value='-2.0',
            description='Robot spawn position in Y axis'
        ),

        DeclareLaunchArgument(
            name='spawn_z', 
            default_value='0.0',
            description='Robot spawn position in Z axis'
        ),
            
        DeclareLaunchArgument(
            name='spawn_yaw', 
            default_value='-1.7',
            description='Robot spawn heading'
        ),

        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so',  '-s', 'libgazebo_ros_init.so', LaunchConfiguration('world')],
            output='screen'
        ),

        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='urdf_spawner',
            output='screen',
            arguments=[
                '-topic', 'robot_description', 
                '-entity', 'linorobot2', 
                '-x', LaunchConfiguration('spawn_x'),
                '-y', LaunchConfiguration('spawn_y'),
                '-z', LaunchConfiguration('spawn_z'),
                '-Y', LaunchConfiguration('spawn_yaw'),
            ]
        ),

        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    condition=IfCondition(LaunchConfiguration('paused')),
                    cmd=['ros2', 'service', 'call', '/pause_physics', 'std_srvs/srv/Empty', '{}'],
                    output='screen'
                )
            ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                LaunchConfiguration('spawn_x'),   # X
                LaunchConfiguration('spawn_y'),   # Y
                LaunchConfiguration('spawn_z'),   # Z
                LaunchConfiguration('spawn_yaw'), # Yaw
                '0',                             # Pitch
                '0',                             # Roll
                'world',                         # Parent Frame
                'odom'                           # Child Frame
            ],
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            condition=IfCondition(LaunchConfiguration("rviz")), # <--- QUAN TRỌNG NHẤT
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        # Node(
        #     package='linorobot2_gazebo',
        #     executable='command_timeout.py',
        #     name='command_timeout'
        # ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time}, 
                ekf_config_path
            ],
            remappings=[("odometry/filtered", LaunchConfiguration("odom_topic"))]
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(description_launch_path),
            launch_arguments={
                'rviz': 'false',
                'use_sim_time': str(use_sim_time),
                'publish_joints': 'false',
                'urdf': LaunchConfiguration('urdf')
            }.items()
        )
    ])

#sources: 
#https://navigation.ros.org/setup_guides/index.html#
#https://answers.ros.org/question/374976/ros2-launch-gazebolaunchpy-from-my-own-launch-file/
#https://github.com/ros2/rclcpp/issues/940

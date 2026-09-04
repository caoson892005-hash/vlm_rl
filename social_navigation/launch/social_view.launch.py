"""Open RViz on the social layer only, for Gazebo or for real hardware.

The view is deliberately separate from the Nav2 RViz config. On hardware the
camera and the VLM run on the workstation while Nav2 may run on the robot, so
the person watching the social regions is not necessarily sitting in front of
the machine that owns the navigation stack.

    ros2 launch social_navigation social_view.launch.py              # Gazebo
    ros2 launch social_navigation social_view.launch.py sim:=false   # hardware

Fixed frame is `map` in both cases. In Gazebo the perception node publishes in
`world`, which reaches `map` through the static transform that
gazebo.launch.py already provides.

`sim` defaults to true here to match social_bringup.launch.py, which is how
this view is normally opened (`rviz:=true`) and which passes `sim` down
explicitly. The two files disagreeing was a trap: getting use_sim_time wrong
does not fail loudly, it leaves RViz holding a clock that never matches the
timestamps on TF, so every display that needs a transform quietly switches
off and the window looks broken for no stated reason. Both directions fail
that way, so the default belongs on the case that occurs more often.
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
        'rviz',
        'social_navigation.rviz',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'sim',
            default_value='true',
            description='true for Gazebo, false for a real camera on this '
                        'machine; must match social_bringup.launch.py'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_config,
            description='RViz configuration to open'),
        DeclareLaunchArgument(
            'rviz_log_level',
            default_value='ERROR',
            description='RViz log level reaching the terminal; '
                        'raise to INFO to debug RViz itself'),
        Node(
            package='rviz2',
            executable='rviz2',
            name='social_rviz',
            # This terminal exists to show VLM decisions. RViz narrates its
            # OpenGL and Ogre setup on every start and repeats frame warnings
            # while a transform is briefly missing, which buries them.
            #
            # Two separate streams have to be handled. Ogre and Qt write
            # directly to stdout, so that stream goes to the launch log file
            # instead of the screen. Everything RViz logs through ROS goes to
            # stderr, which stays on screen but is filtered by level below, so
            # a real RViz failure is still the one thing that shows up.
            output={'stdout': 'log', 'stderr': 'screen'},
            arguments=['-d', LaunchConfiguration('rviz_config')],
            ros_arguments=['--log-level',
                           LaunchConfiguration('rviz_log_level')],
            parameters=[{
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('sim'), value_type=bool),
            }],
        ),
    ])

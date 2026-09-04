"""Launch RGB-D person localization and VLM conversation perception."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('social_perception'),
        'config',
        'social_vlm_perception.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to the social perception ROS parameter file'),
        DeclareLaunchArgument(
            'hf_offline',
            default_value='1',
            # The base checkpoint already sits in ~/.cache/huggingface. Left
            # online, every start pays a hub revision lookup that measured
            # 3.2 s against 1.2 s offline, and hangs far longer when the
            # network is reachable but slow. Set to 0 only when switching
            # vlm_base_model to a checkpoint that is not cached yet.
            description='Load the VLM from the local Hugging Face cache only'),
        SetEnvironmentVariable('HF_HUB_OFFLINE',
                               LaunchConfiguration('hf_offline')),
        SetEnvironmentVariable('TRANSFORMERS_OFFLINE',
                               LaunchConfiguration('hf_offline')),
        Node(
            package='social_perception',
            executable='social_vlm_perception.py',
            name='social_vlm_perception',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])

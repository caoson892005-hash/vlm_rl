"""Launch the VLM half of the split pipeline on the workstation.

Pair it with social_vlm_perception.py running with `vlm_remote: true` on the
robot. Both read the same profile file, so the prompt and the checkpoint paths
cannot drift apart between the two machines.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('social_perception'),
        'config',
        'social_vlm_perception_real.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Parameter file holding the social_vlm_worker block'),
        DeclareLaunchArgument(
            'hf_offline',
            default_value='1',
            description='Load the VLM from the local Hugging Face cache only'),
        SetEnvironmentVariable('HF_HUB_OFFLINE',
                               LaunchConfiguration('hf_offline')),
        SetEnvironmentVariable('TRANSFORMERS_OFFLINE',
                               LaunchConfiguration('hf_offline')),
        Node(
            package='social_perception',
            executable='social_vlm_worker.py',
            name='social_vlm_worker',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])

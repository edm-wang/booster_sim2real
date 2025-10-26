#!/usr/bin/env python3

"""
Launch file for Booster Policy Server
Runs the policy inference node that publishes motor commands
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='T1.yaml',
        description='Configuration file name'
    )
    
    policy_interval_arg = DeclareLaunchArgument(
        'policy_interval',
        default_value='0.02',
        description='Policy inference interval in seconds'
    )
    
    use_random_commands_arg = DeclareLaunchArgument(
        'use_random_commands',
        default_value='true',
        description='Use random velocity commands'
    )
    
    max_velocity_arg = DeclareLaunchArgument(
        'max_velocity',
        default_value='1.0',
        description='Maximum velocity for commands'
    )
    
    max_angular_velocity_arg = DeclareLaunchArgument(
        'max_angular_velocity',
        default_value='1.0',
        description='Maximum angular velocity for commands'
    )
    
    # Booster Policy Node
    booster_policy_node = Node(
        package='booster_policy_server',
        executable='booster_policy',
        name='booster_policy',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'policy_interval': LaunchConfiguration('policy_interval'),
            'use_random_commands': LaunchConfiguration('use_random_commands'),
            'max_velocity': LaunchConfiguration('max_velocity'),
            'max_angular_velocity': LaunchConfiguration('max_angular_velocity'),
        }],
        remappings=[
            ('/booster/sensor_data', '/booster/sensor_data'),
            ('/booster/motor_cmd', '/booster/motor_cmd'),
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        policy_interval_arg,
        use_random_commands_arg,
        max_velocity_arg,
        max_angular_velocity_arg,
        booster_policy_node,
    ])




#!/usr/bin/env python3

"""
Booster Policy Server Launch File
Launches the complete policy server system for 5090
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for Booster Policy Server"""
    
    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='T1.yaml',
        description='Configuration file name'
    )
    
    policy_interval_arg = DeclareLaunchArgument(
        'policy_interval',
        default_value='0.02',
        description='Policy inference interval in seconds (20ms)'
    )
    
    control_interval_arg = DeclareLaunchArgument(
        'control_interval',
        default_value='0.002',
        description='Control loop interval in seconds (2ms)'
    )
    
    # Policy Server Node - Main inference and motor command publishing
    policy_server_node = Node(
        package='booster_policy_server',
        executable='booster_policy_server.py',
        name='booster_policy_server',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'policy_interval': LaunchConfiguration('policy_interval'),
            'control_interval': LaunchConfiguration('control_interval'),
        }]
    )
    
    # Remote Control Node - User input handling
    remote_control_node = Node(
        package='booster_policy_server',
        executable='booster_remote_control.py',
        name='booster_remote_control',
        output='screen'
    )
    
    return LaunchDescription([
        config_file_arg,
        policy_interval_arg,
        control_interval_arg,
        policy_server_node,
        remote_control_node,
    ])


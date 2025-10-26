#!/usr/bin/env python3

"""
Launch file for Booster ROSBag Policy Processor
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'config_file',
            default_value='T1_alt.yaml',
            description='Configuration file name'
        ),
        DeclareLaunchArgument(
            'policy_interval',
            default_value='0.02',
            description='Policy inference interval in seconds'
        ),
        DeclareLaunchArgument(
            'use_random_commands',
            default_value='true',
            description='Use random velocity commands'
        ),
        DeclareLaunchArgument(
            'max_velocity',
            default_value='1.0',
            description='Maximum velocity for random commands'
        ),
        DeclareLaunchArgument(
            'max_angular_velocity',
            default_value='1.0',
            description='Maximum angular velocity for random commands'
        ),
        
        # Booster ROSBag Policy Processor Node
        Node(
            package='booster_policy_server',
            executable='booster_rosbag_policy_processor.py',
            name='booster_rosbag_policy_processor',
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
            ]
        ),
    ])

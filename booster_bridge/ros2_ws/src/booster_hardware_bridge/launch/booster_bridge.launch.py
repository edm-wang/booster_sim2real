#!/usr/bin/env python3

"""
Launch file for Booster Hardware Bridge
Runs the hardware bridge node that interfaces with the robot
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='/home/romela5090/Han/booster_sim2real/deploy_booster/configs/T1.yaml',
        description='Path to configuration file'
    )
    
    # Booster Hardware Bridge Node
    booster_bridge_node = Node(
        package='booster_hardware_bridge',
        executable='booster_bridge',
        name='booster_bridge',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
        }],
        remappings=[
            ('/booster/sensor_data', '/booster/sensor_data'),
            ('/booster/motor_cmd', '/booster/motor_cmd'),
            ('/booster/robot_mode', '/booster/robot_mode'),
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        booster_bridge_node,
    ])

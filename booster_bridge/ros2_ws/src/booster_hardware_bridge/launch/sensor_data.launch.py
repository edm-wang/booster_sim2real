#!/usr/bin/env python3

"""
Launch file for Booster Sensor Data Node
Publishes robot sensor data to ROS2 topics
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='/home/booster/Workspace/booster_sim2real/deploy_booster/configs/T1_alt.yaml',
        description='Path to configuration file'
    )
    
    # Booster Sensor Data Node
    sensor_data_node = Node(
        package='booster_hardware_bridge',
        executable='booster_sensor_data',
        name='booster_sensor_data',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
        }],
        remappings=[
            ('/booster/sensor_data', '/booster/sensor_data'),
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        sensor_data_node,
    ])

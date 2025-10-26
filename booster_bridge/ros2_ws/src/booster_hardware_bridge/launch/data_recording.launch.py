#!/usr/bin/env python3

"""
Launch file for Booster Data Recording and Plotting
Launches sensor data node and data plotter for real-time visualization
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
    
    # Booster Data Plotter Node
    data_plotter_node = Node(
        package='booster_hardware_bridge',
        executable='booster_data_plotter',
        name='booster_data_plotter',
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
        data_plotter_node,
    ])

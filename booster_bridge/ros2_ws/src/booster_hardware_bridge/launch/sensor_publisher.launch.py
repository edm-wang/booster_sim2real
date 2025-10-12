#!/usr/bin/env python3

"""
Sensor Publisher Launch File
Launches only the sensor data publisher for testing
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    """Generate launch description for sensor publisher"""
    
    # Launch arguments
    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='true',
        description='Run in simulation mode (no Booster SDK)'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='100.0',
        description='Sensor data publish rate in Hz'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'config', 
            'robot_config.yaml'
        ),
        description='Path to robot configuration file'
    )
    
    # Get launch configurations
    use_simulation = LaunchConfiguration('use_simulation')
    publish_rate = LaunchConfiguration('publish_rate')
    config_file = LaunchConfiguration('config_file')
    
    # Sensor Publisher Node
    sensor_publisher_node = Node(
        package='booster_hardware_bridge',
        executable='booster_sensor_publisher.py',
        name='booster_sensor_publisher',
        output='screen',
        parameters=[{
            'use_simulation': use_simulation,
            'publish_rate': publish_rate
        }],
        arguments=['--config', config_file]
    )
    
    # Log info
    log_info = LogInfo(
        msg="Starting Booster sensor data publisher for testing"
    )
    
    return LaunchDescription([
        use_simulation_arg,
        publish_rate_arg,
        config_file_arg,
        
        log_info,
        
        # Sensor publisher
        sensor_publisher_node,
    ])

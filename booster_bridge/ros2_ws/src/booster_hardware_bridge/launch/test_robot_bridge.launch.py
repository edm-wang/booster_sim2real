#!/usr/bin/env python3

"""
Test Robot Bridge Launch File
Launches robot bridge in simulation mode for testing
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    """Generate launch description for testing robot bridge"""
    
    # Launch arguments
    use_monitor_arg = DeclareLaunchArgument(
        'use_monitor',
        default_value='true',
        description='Enable robot monitor node'
    )
    
    # Get launch configurations
    use_monitor = LaunchConfiguration('use_monitor')
    
    # Hardware Bridge Node (simulation mode)
    hardware_bridge_node = Node(
        package='booster_hardware_bridge',
        executable='booster_hardware_bridge.py',
        name='booster_hardware_bridge',
        output='screen',
        parameters=[{
            'use_simulation': True
        }]
    )
    
    # Robot Monitor Node
    robot_monitor_node = Node(
        package='booster_hardware_bridge',
        executable='booster_robot_monitor.py',
        name='booster_robot_monitor',
        output='screen',
        condition=IfCondition(use_monitor)
    )
    
    # Log info
    log_info = LogInfo(
        msg="Starting Booster robot bridge test (simulation mode)"
    )
    
    return LaunchDescription([
        use_monitor_arg,
        
        log_info,
        
        # Hardware bridge (simulation)
        hardware_bridge_node,
        
        # Robot monitor
        robot_monitor_node,
    ])


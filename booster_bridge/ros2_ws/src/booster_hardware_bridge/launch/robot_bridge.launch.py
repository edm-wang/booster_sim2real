#!/usr/bin/env python3

"""
Robot Bridge Launch File
Launches the complete robot hardware bridge system
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
import os


def generate_launch_description():
    """Generate launch description for robot bridge"""
    
    # Launch arguments
    use_monitor_arg = DeclareLaunchArgument(
        'use_monitor',
        default_value='true',
        description='Enable robot monitor node'
    )
    
    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='false',
        description='Run in simulation mode (no Booster SDK)'
    )
    
    test_mode_arg = DeclareLaunchArgument(
        'test_mode',
        default_value='false',
        description='Run in test mode (receive commands but don\'t move robot)'
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
    use_monitor = LaunchConfiguration('use_monitor')
    use_simulation = LaunchConfiguration('use_simulation')
    test_mode = LaunchConfiguration('test_mode')
    config_file = LaunchConfiguration('config_file')
    
    # Hardware Bridge Node
    hardware_bridge_node = Node(
        package='booster_hardware_bridge',
        executable='booster_hardware_bridge.py',
        name='booster_hardware_bridge',
        output='screen',
        parameters=[{
            'use_simulation': use_simulation,
            'test_mode': test_mode
        }],
        arguments=['--config', config_file]
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
        msg="Starting Booster robot hardware bridge system"
    )
    
    return LaunchDescription([
        use_monitor_arg,
        use_simulation_arg,
        test_mode_arg,
        config_file_arg,
        
        log_info,
        
        # Hardware bridge
        hardware_bridge_node,
        
        # Robot monitor
        robot_monitor_node,
    ])


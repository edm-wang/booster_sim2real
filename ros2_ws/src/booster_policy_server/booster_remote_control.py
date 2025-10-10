#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time

# Import ROS2 message types
from booster_msgs.msg import BoosterControlCmd

# Import existing remote control service
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../deploy_booster'))
from utils.remote_control_service import RemoteControlService


class BoosterRemoteControl(Node):
    """
    ROS2 Remote Control Node for Booster Robot
    Publishes control commands based on joystick/keyboard input
    """
    
    def __init__(self):
        super().__init__('booster_remote_control')
        
        # Initialize remote control service
        self.remote_control = RemoteControlService()
        
        # ROS2 QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Publisher for control commands
        self.control_cmd_publisher = self.create_publisher(
            BoosterControlCmd,
            'booster/control_cmd',
            qos_profile
        )
        
        # Timer for publishing control commands
        self.control_timer = self.create_timer(0.01, self.publish_control_commands)  # 100Hz
        
        # State variables
        self.last_custom_mode_request = False
        self.last_rl_gait_request = False
        
        self.get_logger().info("Booster Remote Control initialized")
        self.get_logger().info(f"Operation hints: {self.remote_control.get_operation_hint()}")
    
    def publish_control_commands(self):
        """Publish control commands based on remote input"""
        # Create control command message
        control_cmd = BoosterControlCmd()
        
        # Get velocity commands
        control_cmd.vx = self.remote_control.get_vx_cmd()
        control_cmd.vy = self.remote_control.get_vy_cmd()
        control_cmd.vyaw = self.remote_control.get_vyaw_cmd()
        
        # Check for mode switching requests
        current_custom_mode_request = self.remote_control.start_custom_mode()
        current_rl_gait_request = self.remote_control.start_rl_gait()
        
        # Only send mode requests on state change
        control_cmd.start_custom_mode = (current_custom_mode_request and 
                                       not self.last_custom_mode_request)
        control_cmd.start_rl_gait = (current_rl_gait_request and 
                                   not self.last_rl_gait_request)
        
        self.last_custom_mode_request = current_custom_mode_request
        self.last_rl_gait_request = current_rl_gait_request
        
        # Set timestamp
        control_cmd.timestamp = self.get_clock().now().to_msg()
        
        # Publish
        self.control_cmd_publisher.publish(control_cmd)
        
        # Log commands occasionally
        if abs(control_cmd.vx) > 0.01 or abs(control_cmd.vy) > 0.01 or abs(control_cmd.vyaw) > 0.01:
            self.get_logger().debug(
                f"Control: vx={control_cmd.vx:.2f}, vy={control_cmd.vy:.2f}, vyaw={control_cmd.vyaw:.2f}"
            )
    
    def cleanup(self):
        """Cleanup resources"""
        self.remote_control.close()
        self.get_logger().info("Booster Remote Control cleanup completed")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        remote_control = BoosterRemoteControl()
        
        # Handle shutdown gracefully
        def shutdown_handler():
            remote_control.cleanup()
            rclpy.shutdown()
        
        import signal
        signal.signal(signal.SIGINT, lambda sig, frame: shutdown_handler())
        signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_handler())
        
        rclpy.spin(remote_control)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


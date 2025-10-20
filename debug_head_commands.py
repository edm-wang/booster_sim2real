#!/usr/bin/env python3

"""
Debug script to test head movement commands directly.
This will help verify the exact values being sent and received.
"""

import rclpy
from rclpy.node import Node
from booster_msgs.msg import BoosterControlCmd
import time

class HeadCommandDebugger(Node):
    """
    Debug node to test head movement commands
    """
    
    def __init__(self):
        super().__init__('head_command_debugger')
        
        # Create publisher for head movement commands
        self.head_cmd_publisher = self.create_publisher(
            BoosterControlCmd,
            '/booster/head_control_cmd',
            10
        )
        
        self.get_logger().info('Head Command Debugger started')
        self.get_logger().info('Publishing test commands...')
        
        # Create timer to send test commands
        self.test_timer = self.create_timer(3.0, self.send_test_commands)
        self.test_count = 0
        
    def send_test_commands(self):
        """Send test head movement commands"""
        if self.test_count >= 2:  # Only send 2 commands
            self.get_logger().info('Test completed, shutting down...')
            self.destroy_timer(self.test_timer)
            return
            
        if self.test_count == 0:
            # Test head up command
            self.get_logger().info('Sending HEAD UP command (vy=-0.3)...')
            cmd = self.create_head_up_command()
            self.head_cmd_publisher.publish(cmd)
            
        elif self.test_count == 1:
            # Test head down command
            self.get_logger().info('Sending HEAD DOWN command (vy=1.0)...')
            cmd = self.create_head_down_command()
            self.head_cmd_publisher.publish(cmd)
            
        self.test_count += 1
        
    def create_head_up_command(self):
        """Create head up command"""
        cmd = BoosterControlCmd()
        cmd.timestamp = self.get_clock().now().to_msg()
        cmd.vx = 0.0
        cmd.vy = -0.3  # Negative pitch for head up
        cmd.vyaw = 0.0
        cmd.start_custom_mode = False
        cmd.start_rl_gait = False
        return cmd
        
    def create_head_down_command(self):
        """Create head down command"""
        cmd = BoosterControlCmd()
        cmd.timestamp = self.get_clock().now().to_msg()
        cmd.vx = 0.0
        cmd.vy = 1.0  # Positive pitch for head down
        cmd.vyaw = 0.0
        cmd.start_custom_mode = False
        cmd.start_rl_gait = False
        return cmd

def main(args=None):
    rclpy.init(args=args)
    
    try:
        debugger = HeadCommandDebugger()
        
        # Run for a short time to send test commands
        rclpy.spin_once(debugger, timeout_sec=1.0)
        rclpy.spin_once(debugger, timeout_sec=1.0)
        rclpy.spin_once(debugger, timeout_sec=1.0)
        rclpy.spin_once(debugger, timeout_sec=1.0)
        
        print("Debug commands sent!")
        
    except KeyboardInterrupt:
        print('\nDebug interrupted by user')
    except Exception as e:
        print(f'Debug error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

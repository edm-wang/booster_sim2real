#!/usr/bin/env python3

"""
Test script for the updated head movement system.
This script demonstrates how to send head movement commands directly.

Usage:
1. Start the head receiver: python3 booster_head_receiver.py
2. Run this script: python3 test_head_movement.py
"""

import rclpy
from rclpy.node import Node
from booster_msgs.msg import BoosterControlCmd
import time
import sys

class HeadMovementTester(Node):
    """
    Test node that sends head movement commands directly.
    """
    
    def __init__(self):
        super().__init__('head_movement_tester')
        
        # Create publisher for head movement commands
        self.head_cmd_publisher = self.create_publisher(
            BoosterControlCmd,
            '/booster/head_control_cmd',
            10
        )
        
        self.get_logger().info('Head Movement Tester started')
        self.get_logger().info('Sending test head movement commands...')
        
        # Wait a moment for the publisher to be ready
        time.sleep(1.0)
        
        # Test different head movements
        self.test_head_movements()
    
    def test_head_movements(self):
        """Test various head movement commands"""
        try:
            # Test 1: Head up
            self.get_logger().info('Testing head up movement...')
            self.send_head_command(True, -0.3, 0.0)  # Look up
            time.sleep(2.0)
            
            # Test 2: Head down
            self.get_logger().info('Testing head down movement...')
            self.send_head_command(True, 1.0, 0.0)   # Look down
            time.sleep(2.0)
            
            # Test 3: Head straight
            self.get_logger().info('Testing head straight movement...')
            self.send_head_command(True, 0.0, 0.0)    # Look straight
            time.sleep(2.0)
            
            # Test 4: Head left
            self.get_logger().info('Testing head left movement...')
            self.send_head_command(True, 0.0, 0.5)   # Look left
            time.sleep(2.0)
            
            # Test 5: Head right
            self.get_logger().info('Testing head right movement...')
            self.send_head_command(True, 0.0, -0.5)   # Look right
            time.sleep(2.0)
            
            # Test 6: Return to center
            self.get_logger().info('Returning head to center...')
            self.send_head_command(True, 0.0, 0.0)   # Look straight
            time.sleep(2.0)
            
            self.get_logger().info('✅ All head movement tests completed!')
            
        except Exception as e:
            self.get_logger().error(f'Error during head movement testing: {str(e)}')
    
    def send_head_command(self, head_control, pitch, yaw):
        """Send a head movement command"""
        try:
            cmd = BoosterControlCmd()
            cmd.timestamp = self.get_clock().now().to_msg()
            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.vyaw = 0.0
            cmd.start_custom_mode = False
            cmd.start_rl_gait = False
            cmd.head_control = head_control
            cmd.head_pitch = pitch
            cmd.head_yaw = yaw
            
            self.head_cmd_publisher.publish(cmd)
            self.get_logger().info(f'Sent head command: control={head_control}, pitch={pitch:.2f}, yaw={yaw:.2f}')
            
        except Exception as e:
            self.get_logger().error(f'Error sending head command: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tester = HeadMovementTester()
        
        # Run for a short time to complete tests
        rclpy.spin_once(tester, timeout_sec=15.0)
        
    except KeyboardInterrupt:
        print('\nShutting down head movement tester...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()


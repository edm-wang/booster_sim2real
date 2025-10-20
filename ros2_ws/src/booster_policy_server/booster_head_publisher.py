#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from booster_msgs.msg import BoosterControlCmd
from booster_msgs.srv import BoosterHeadControl
import time

class BoosterHeadPublisher(Node):
    """
    Simple head publisher that sends head movement commands via service calls.
    Uses only the two predefined SDK movements: head up and head down.
    """
    
    def __init__(self):
        super().__init__('booster_head_publisher')
        
        # Create publisher for head movement commands
        self.head_cmd_publisher = self.create_publisher(
            BoosterControlCmd,
            '/booster/head_control_cmd',
            10
        )
        
        # Create service server for head control
        self.head_control_service = self.create_service(
            BoosterHeadControl,
            '/booster/head_control',
            self.head_control_callback
        )
        
        self.get_logger().info('Booster Head Publisher started')
        self.get_logger().info('Available commands: "up" or "down"')
        
    def head_control_callback(self, request, response):
        """Handle head control service requests"""
        try:
            command = request.command.lower()
            
            if command == "up":
                # Head up movement from SDK: yaw=0.0, pitch=-0.3
                cmd = self.create_head_up_command()
                self.head_cmd_publisher.publish(cmd)
                response.success = True
                response.message = "Head moved up"
                self.get_logger().info('Head moved up')
                
            elif command == "down":
                # Head down movement from SDK: yaw=0.0, pitch=1.0
                cmd = self.create_head_down_command()
                self.head_cmd_publisher.publish(cmd)
                response.success = True
                response.message = "Head moved down"
                self.get_logger().info('Head moved down')
                
            else:
                response.success = False
                response.message = f"Unknown command: {command}. Use 'up' or 'down'"
                self.get_logger().warn(f'Unknown command: {command}')
            
        except Exception as e:
            self.get_logger().error(f'Error in head control service: {str(e)}')
            response.success = False
            response.message = f"Error: {str(e)}"
        
        return response
    
    def create_head_up_command(self):
        """Create head up command - EXACT from SDK 'hu' command"""
        cmd = BoosterControlCmd()
        cmd.timestamp = self.get_clock().now().to_msg()
        cmd.vx = 0.0
        cmd.vy = -0.3  # Negative pitch for head up
        cmd.vyaw = 0.0
        cmd.start_custom_mode = False
        cmd.start_rl_gait = False
        # Head control data
        cmd.head_control = True
        cmd.head_pitch = -0.3  # Head up from SDK
        cmd.head_yaw = 0.0
        return cmd
        
    def create_head_down_command(self):
        """Create head down command - EXACT from SDK 'hd' command"""
        cmd = BoosterControlCmd()
        cmd.timestamp = self.get_clock().now().to_msg()
        cmd.vx = 0.0
        cmd.vy = 1.0  # Positive pitch for head down
        cmd.vyaw = 0.0
        cmd.start_custom_mode = False
        cmd.start_rl_gait = False
        # Head control data
        cmd.head_control = True
        cmd.head_pitch = 1.0   # Head down from SDK
        cmd.head_yaw = 0.0
        return cmd

def main(args=None):
    rclpy.init(args=args)
    
    try:
        publisher = BoosterHeadPublisher()
        
        # Run the publisher
        rclpy.spin(publisher)
        
    except KeyboardInterrupt:
        print('\nShutting down head publisher...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
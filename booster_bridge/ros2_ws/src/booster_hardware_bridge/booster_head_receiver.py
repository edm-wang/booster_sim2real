#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from booster_msgs.msg import BoosterControlCmd
from booster_robotics_sdk_python import ChannelFactory, B1LocoClient
import time
import sys

class BoosterHeadReceiver(Node):
    """
    Simple head receiver that receives head movement commands and executes them on the robot.
    Uses only the two predefined SDK movements: head up and head down.
    """
    
    def __init__(self):
        super().__init__('booster_head_receiver')
        
        # Initialize the SDK
        try:
            # Initialize with network interface (you may need to adjust this)
            ChannelFactory.Instance().Init(0, "eth0")  # Adjust network interface as needed
            
            # Create the locomotion client for head control
            self.client = B1LocoClient()
            self.client.Init()
            
            self.get_logger().info('SDK initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize SDK: {str(e)}')
            sys.exit(1)
        
        # Create subscriber for head movement commands
        self.head_cmd_subscriber = self.create_subscription(
            BoosterControlCmd,
            '/booster/head_control_cmd',
            self.head_command_callback,
            10
        )
        
        self.get_logger().info('Booster Head Receiver started')
        self.get_logger().info('Waiting for head movement commands...')
        
    def head_command_callback(self, msg):
        """Handle incoming head movement commands"""
        try:
            self.get_logger().info(f'Received head command: vx={msg.vx}, vy={msg.vy}, vyaw={msg.vyaw}')
            
            # Execute head movement based on command values
            # Head up: vy < 0 (negative pitch)
            # Head down: vy > 0 (positive pitch)
            
            # More specific detection for the exact values being sent
            if msg.vy <= -0.2:  # Head up movement (covers -0.3)
                self.get_logger().info(f'Executing head up movement (vy={msg.vy})...')
                # Use SDK head up command
                self.client.SendCommand("hu")  # SDK head up command
                self.get_logger().info('Head up command sent to robot')
                
            elif msg.vy >= 0.5:  # Head down movement (covers 1.0)
                self.get_logger().info(f'Executing head down movement (vy={msg.vy})...')
                # Use SDK head down command
                self.client.SendCommand("hd")  # SDK head down command
                self.get_logger().info('Head down command sent to robot')
                
            else:
                self.get_logger().info(f'No significant head movement detected (vy={msg.vy} not in range)')
            
        except Exception as e:
            self.get_logger().error(f'Error processing head command: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        receiver = BoosterHeadReceiver()
        
        # Run the receiver
        rclpy.spin(receiver)
        
    except KeyboardInterrupt:
        print('\nShutting down head receiver...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

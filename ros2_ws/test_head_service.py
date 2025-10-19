#!/usr/bin/env python3

"""
Simple test script for head control service.
This script demonstrates how to call head up/down movements via service calls.

Usage:
1. Start the publisher: python3 booster_head_publisher.py
2. Start the robot receiver: python3 booster_head_receiver.py
3. Run this script: python3 test_head_service.py
"""

import rclpy
from rclpy.node import Node
from booster_msgs.srv import BoosterHeadControl
import time
import sys

class HeadServiceTester(Node):
    """
    Simple test node for head control service calls.
    """
    
    def __init__(self):
        super().__init__('head_service_tester')
        
        # Create service client
        self.head_control_client = self.create_client(
            BoosterHeadControl,
            '/booster/head_control'
        )
        
        # Wait for service to be available
        while not self.head_control_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for head control service...')
        
        self.get_logger().info('Head Service Tester started')
        self.get_logger().info('Service is available!')
        
    def call_head_command(self, command):
        """Call a head movement command"""
        try:
            # Create service request
            request = BoosterHeadControl.Request()
            request.command = command
            
            # Call the service
            future = self.head_control_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            # Get response
            response = future.result()
            
            if response.success:
                self.get_logger().info(f'✅ Success: {response.message}')
            else:
                self.get_logger().error(f'❌ Failed: {response.message}')
                
            return response.success
            
        except Exception as e:
            self.get_logger().error(f'Error calling service: {str(e)}')
            return False

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tester = HeadServiceTester()
        
        print("\n" + "="*50)
        print("HEAD CONTROL SERVICE TESTER")
        print("="*50)
        print("This script tests the head control service.")
        print("Available commands: 'up' or 'down'")
        print("="*50)
        
        if len(sys.argv) > 1:
            # Command line mode
            command = sys.argv[1]
            
            print(f"Testing command: {command}")
            
            success = tester.call_head_command(command)
            
            if success:
                print(f"✅ {command} executed successfully")
            else:
                print(f"❌ {command} failed")
        else:
            # Interactive mode
            print("Starting interactive mode...")
            print("Type 'up' or 'down' to test head movements.")
            print("Type 'quit' to exit.")
            
            while True:
                try:
                    command = input('> ').strip().lower()
                    
                    if command == 'quit' or command == 'exit':
                        break
                    elif command in ['up', 'down']:
                        success = tester.call_head_command(command)
                        if success:
                            print(f"✅ {command} executed successfully")
                        else:
                            print(f"❌ {command} failed")
                    else:
                        print("Available commands: 'up', 'down', 'quit'")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f'Error: {str(e)}')
        
    except KeyboardInterrupt:
        print('\nTest stopped by user.')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

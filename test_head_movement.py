#!/usr/bin/env python3

"""
Test script to verify head movement functionality.
This script will test the complete head movement pipeline.

Usage:
1. Make sure the head receiver is running: python3 booster_head_receiver.py
2. Make sure the head publisher is running: python3 booster_head_publisher.py  
3. Run this test: python3 test_head_movement.py
"""

import rclpy
from rclpy.node import Node
from booster_msgs.srv import BoosterHeadControl
import time
import sys

class HeadMovementTester(Node):
    """
    Test node for head movement functionality
    """
    
    def __init__(self):
        super().__init__('head_movement_tester')
        
        # Create service client
        self.head_control_client = self.create_client(
            BoosterHeadControl,
            '/booster/head_control'
        )
        
        self.get_logger().info('Head Movement Tester started')
        
    def test_head_movements(self):
        """Test both head up and head down movements"""
        # Wait for service to be available
        if not self.head_control_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Head control service not available!')
            self.get_logger().error('Make sure booster_head_publisher.py is running')
            return False
        
        self.get_logger().info('✅ Head control service is available')
        
        # Test head up movement
        self.get_logger().info('Testing head UP movement...')
        success_up = self.call_head_command("up")
        
        if success_up:
            self.get_logger().info('✅ Head UP test passed')
        else:
            self.get_logger().error('❌ Head UP test failed')
        
        # Wait a bit between movements
        time.sleep(2.0)
        
        # Test head down movement
        self.get_logger().info('Testing head DOWN movement...')
        success_down = self.call_head_command("down")
        
        if success_down:
            self.get_logger().info('✅ Head DOWN test passed')
        else:
            self.get_logger().error('❌ Head DOWN test failed')
        
        # Final result
        if success_up and success_down:
            self.get_logger().info('🎉 All head movement tests passed!')
            return True
        else:
            self.get_logger().error('❌ Some head movement tests failed!')
            return False
    
    def call_head_command(self, command):
        """Call a head movement command"""
        try:
            # Create service request
            request = BoosterHeadControl.Request()
            request.command = command
            
            # Call the service
            future = self.head_control_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            # Get response
            response = future.result()
            
            if response.success:
                self.get_logger().info(f'✅ Success: {response.message}')
                return True
            else:
                self.get_logger().error(f'❌ Failed: {response.message}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error calling service: {str(e)}')
            return False

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tester = HeadMovementTester()
        
        # Run the test
        success = tester.test_head_movements()
        
        if success:
            print("\n🎉 HEAD MOVEMENT TEST PASSED!")
            print("The head should have moved up and down.")
        else:
            print("\n❌ HEAD MOVEMENT TEST FAILED!")
            print("Check the logs above for error details.")
        
        return success
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        return False
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

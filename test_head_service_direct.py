#!/usr/bin/env python3

"""
Direct test of the head service without going through the publisher.
This will help isolate whether the issue is in the service or the receiver.
"""

import rclpy
from rclpy.node import Node
from booster_msgs.srv import BoosterHeadControl
import time

class DirectHeadServiceTester(Node):
    """
    Direct test of head service
    """
    
    def __init__(self):
        super().__init__('direct_head_service_tester')
        
        # Create service client
        self.head_control_client = self.create_client(
            BoosterHeadControl,
            '/booster/head_control'
        )
        
        self.get_logger().info('Direct Head Service Tester started')
        
    def test_head_service(self):
        """Test the head service directly"""
        # Wait for service to be available
        if not self.head_control_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Head control service not available!')
            self.get_logger().error('Make sure booster_head_publisher.py is running')
            return False
        
        self.get_logger().info('✅ Head control service is available')
        
        # Test head up
        self.get_logger().info('Testing head UP via service...')
        success_up = self.call_head_command("up")
        
        if success_up:
            self.get_logger().info('✅ Head UP service call successful')
        else:
            self.get_logger().error('❌ Head UP service call failed')
        
        # Wait between commands
        time.sleep(2.0)
        
        # Test head down
        self.get_logger().info('Testing head DOWN via service...')
        success_down = self.call_head_command("down")
        
        if success_down:
            self.get_logger().info('✅ Head DOWN service call successful')
        else:
            self.get_logger().error('❌ Head DOWN service call failed')
        
        return success_up and success_down
    
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
                self.get_logger().info(f'✅ Service Success: {response.message}')
                return True
            else:
                self.get_logger().error(f'❌ Service Failed: {response.message}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error calling service: {str(e)}')
            return False

def main(args=None):
    rclpy.init(args=args)
    
    try:
        tester = DirectHeadServiceTester()
        
        # Run the test
        success = tester.test_head_service()
        
        if success:
            print("\n🎉 HEAD SERVICE TEST PASSED!")
            print("The service calls were successful.")
        else:
            print("\n❌ HEAD SERVICE TEST FAILED!")
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
    exit(0 if success else 1)

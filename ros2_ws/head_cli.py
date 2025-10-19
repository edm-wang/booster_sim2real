#!/usr/bin/env python3

"""
Simple command-line interface for head control service.
Quick way to trigger head up/down movements via service calls.

Usage:
    python3 head_cli.py up
    python3 head_cli.py down
"""

import rclpy
from rclpy.node import Node
from booster_msgs.srv import BoosterHeadControl
import sys

def call_head_command(command):
    """Call a head movement service"""
    rclpy.init()
    
    try:
        # Create node
        node = Node('head_cli')
        
        # Create service client
        client = node.create_client(BoosterHeadControl, '/booster/head_control')
        
        # Wait for service
        if not client.wait_for_service(timeout_sec=5.0):
            print("❌ Service not available. Make sure the publisher is running.")
            return False
        
        # Create request
        request = BoosterHeadControl.Request()
        request.command = command
        
        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future)
        
        # Get response
        response = future.result()
        
        if response.success:
            print(f"✅ {response.message}")
            return True
        else:
            print(f"❌ {response.message}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    finally:
        if rclpy.ok():
            rclpy.shutdown()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 head_cli.py <command>")
        print("Available commands: up, down")
        print("Example: python3 head_cli.py up")
        return
    
    command = sys.argv[1]
    
    print(f"Calling head command: {command}")
    success = call_head_command(command)
    
    if success:
        print(f"✅ {command} executed successfully")
    else:
        print(f"❌ {command} failed")

if __name__ == '__main__':
    main()

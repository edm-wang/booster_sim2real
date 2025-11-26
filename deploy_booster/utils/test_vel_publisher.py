#!/usr/bin/env python3
"""
Simple test publisher for velocity commands.
Publishes geometry_msgs/Twist messages to the "vel_cmd" topic.
Edit the velocity values below to change the command.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class VelCmdPublisher(Node):
    """Simple publisher for velocity commands."""

    def __init__(self):
        super().__init__('vel_cmd_publisher')
        self.publisher = self.create_publisher(Twist, 'vel_cmd', 10)
        self.timer = self.create_timer(0.1, self.publish_callback)  # 10 Hz
        self.get_logger().info('Publishing to topic "vel_cmd"')

    def publish_callback(self):
        """Publish a velocity command."""
        msg = Twist()
        
        # Edit these values to change the velocity command
        msg.linear.x = 0.0   # vx
        msg.linear.y = 0.0   # vy
        msg.angular.z = 0.0  # vyaw
        
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = VelCmdPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        # Publish zero velocity before shutting down
        msg = Twist()
        node.publisher.publish(msg)
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

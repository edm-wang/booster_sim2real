#!/usr/bin/env python3

"""
Test Robot Bridge Script
Tests the robot hardware bridge system
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time
import numpy as np
from typing import Optional

# Import ROS2 message types
from booster_msgs.msg import BoosterSensorData, BoosterMotorCmd, BoosterControlCmd
from booster_msgs.srv import BoosterRobotMode


class RobotBridgeTester(Node):
    """
    Robot Bridge Test Node
    Tests the robot hardware bridge system
    """
    
    def __init__(self):
        super().__init__('robot_bridge_tester')
        
        # ROS2 QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscribers for testing
        self.sensor_data_subscriber = self.create_subscription(
            BoosterSensorData, 'booster/sensor_data', self.sensor_data_callback, qos_profile
        )
        
        # Publishers for testing
        self.motor_cmd_publisher = self.create_publisher(
            BoosterMotorCmd, 'booster/motor_cmd', qos_profile
        )
        
        self.control_cmd_publisher = self.create_publisher(
            BoosterControlCmd, 'booster/control_cmd', qos_profile
        )
        
        # Service client for testing
        self.robot_mode_client = self.create_client(BoosterRobotMode, 'booster/robot_mode')
        
        # Test state
        self.sensor_data_count = 0
        self.motor_cmd_count = 0
        self.test_start_time = time.time()
        
        # Test timer
        self.test_timer = self.create_timer(1.0, self.test_callback)
        
        self.get_logger().info("Robot Bridge Tester initialized")
    
    def sensor_data_callback(self, msg: BoosterSensorData):
        """Monitor sensor data"""
        self.sensor_data_count += 1
        
        # Validate sensor data
        if len(msg.joint_positions) != 23:
            self.get_logger().error(f"Invalid joint positions length: {len(msg.joint_positions)}")
        
        if len(msg.joint_velocities) != 23:
            self.get_logger().error(f"Invalid joint velocities length: {len(msg.joint_velocities)}")
        
        if len(msg.joint_torques) != 23:
            self.get_logger().error(f"Invalid joint torques length: {len(msg.joint_torques)}")
        
        # Log sensor data occasionally
        if self.sensor_data_count % 50 == 0:
            self.get_logger().info(f"Received sensor data #{self.sensor_data_count}")
    
    def test_callback(self):
        """Main test callback"""
        current_time = time.time()
        test_duration = current_time - self.test_start_time
        
        # Calculate frequencies
        sensor_freq = self.sensor_data_count / test_duration if test_duration > 0 else 0
        
        # Log test status
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"ROBOT BRIDGE TEST - Duration: {test_duration:.1f}s")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Sensor data: {self.sensor_data_count} messages ({sensor_freq:.1f} Hz)")
        self.get_logger().info(f"Motor commands sent: {self.motor_cmd_count}")
        
        # Test motor command publishing
        if test_duration > 5.0 and self.motor_cmd_count == 0:
            self._test_motor_command_publishing()
        
        # Test robot mode service
        if test_duration > 10.0:
            self._test_robot_mode_service()
        
        # Test control command publishing
        if test_duration > 15.0:
            self._test_control_command_publishing()
        
        self.get_logger().info("=" * 50)
    
    def _test_motor_command_publishing(self):
        """Test motor command publishing"""
        if self.motor_cmd_count > 0:
            return
        
        self.get_logger().info("Testing motor command publishing...")
        
        # Create test motor command
        motor_cmd = BoosterMotorCmd()
        motor_cmd.cmd_type = 0  # SERIAL
        
        # Set test joint positions
        for i in range(23):
            motor_cmd.joint_positions[i] = 0.1 * np.sin(time.time() + i)
            motor_cmd.joint_velocities[i] = 0.0
            motor_cmd.joint_torques[i] = 0.0
            motor_cmd.joint_kp[i] = 10.0
            motor_cmd.joint_kd[i] = 1.0
        
        # Set timestamp
        motor_cmd.timestamp = self.get_clock().now().to_msg()
        
        # Publish
        self.motor_cmd_publisher.publish(motor_cmd)
        self.motor_cmd_count += 1
        
        self.get_logger().info("Motor command published successfully")
    
    def _test_robot_mode_service(self):
        """Test robot mode service"""
        if not self.robot_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Robot mode service not available")
            return
        
        self.get_logger().info("Testing robot mode service...")
        
        # Test mode switching
        request = BoosterRobotMode.Request()
        request.mode = 1  # kCustom
        
        future = self.robot_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info("✅ Robot mode service working")
            else:
                self.get_logger().error(f"❌ Robot mode service failed: {response.message}")
        else:
            self.get_logger().error("❌ Robot mode service timeout")
    
    def _test_control_command_publishing(self):
        """Test control command publishing"""
        self.get_logger().info("Testing control command publishing...")
        
        # Create test control command
        control_cmd = BoosterControlCmd()
        control_cmd.vx = 0.5
        control_cmd.vy = 0.2
        control_cmd.vyaw = 0.1
        control_cmd.start_custom_mode = True
        control_cmd.start_rl_gait = False
        
        # Publish
        self.control_cmd_publisher.publish(control_cmd)
        
        self.get_logger().info("Control command published successfully")
    
    def run_comprehensive_test(self, duration: float = 30.0):
        """Run comprehensive test"""
        self.get_logger().info(f"Starting comprehensive robot bridge test for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Final analysis
        total_time = time.time() - self.test_start_time
        sensor_freq = self.sensor_data_count / total_time if total_time > 0 else 0
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("COMPREHENSIVE TEST RESULTS")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Total test time: {total_time:.1f}s")
        self.get_logger().info(f"Total sensor data messages: {self.sensor_data_count}")
        self.get_logger().info(f"Total motor command messages: {self.motor_cmd_count}")
        self.get_logger().info(f"Average sensor frequency: {sensor_freq:.1f} Hz")
        
        # Success criteria
        success = True
        if sensor_freq < 400:
            self.get_logger().error("❌ Sensor frequency too low")
            success = False
        else:
            self.get_logger().info("✅ Sensor frequency adequate")
        
        if self.sensor_data_count == 0:
            self.get_logger().error("❌ No sensor data received")
            success = False
        else:
            self.get_logger().info("✅ Sensor data received")
        
        if success:
            self.get_logger().info("🎉 ROBOT BRIDGE TEST PASSED!")
        else:
            self.get_logger().error("❌ ROBOT BRIDGE TEST FAILED!")
        
        return success


def main(args=None):
    rclpy.init(args=args)
    
    try:
        tester = RobotBridgeTester()
        
        # Run comprehensive test
        success = tester.run_comprehensive_test(duration=30.0)
        
        if success:
            print("\n🎉 ROBOT BRIDGE TEST PASSED!")
        else:
            print("\n❌ ROBOT BRIDGE TEST FAILED!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


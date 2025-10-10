#!/usr/bin/env python3

"""
Booster Robot Monitor Node
Monitors robot health and system status
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time
import numpy as np
from typing import Optional

# Import ROS2 message types
from booster_msgs.msg import BoosterSensorData, BoosterMotorCmd, BoosterControlCmd


class BoosterRobotMonitor(Node):
    """
    Robot Monitor Node
    Monitors robot health and system status
    """
    
    def __init__(self):
        super().__init__('booster_robot_monitor')
        
        # ROS2 QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscribers for monitoring
        self.sensor_data_subscriber = self.create_subscription(
            BoosterSensorData, 'booster/sensor_data', self.sensor_data_callback, qos_profile
        )
        
        self.motor_cmd_subscriber = self.create_subscription(
            BoosterMotorCmd, 'booster/motor_cmd', self.motor_cmd_callback, qos_profile
        )
        
        self.control_cmd_subscriber = self.create_subscription(
            BoosterControlCmd, 'booster/control_cmd', self.control_cmd_callback, qos_profile
        )
        
        # Monitoring variables
        self.sensor_data_count = 0
        self.motor_cmd_count = 0
        self.control_cmd_count = 0
        self.last_sensor_time = 0.0
        self.last_motor_cmd_time = 0.0
        self.start_time = time.time()
        
        # Health monitoring
        self.sensor_timeout_threshold = 0.1  # 100ms
        self.motor_cmd_timeout_threshold = 0.2  # 200ms
        
        # Statistics
        self.latest_sensor_data: Optional[BoosterSensorData] = None
        self.latest_motor_cmd: Optional[BoosterMotorCmd] = None
        
        # Monitor timer
        self.monitor_timer = self.create_timer(1.0, self.monitor_callback)
        
        self.get_logger().info("Booster Robot Monitor initialized")
    
    def sensor_data_callback(self, msg: BoosterSensorData):
        """Monitor sensor data"""
        self.sensor_data_count += 1
        self.latest_sensor_data = msg
        self.last_sensor_time = time.time()
        
        # Health checks
        self._check_sensor_health(msg)
    
    def motor_cmd_callback(self, msg: BoosterMotorCmd):
        """Monitor motor commands"""
        self.motor_cmd_count += 1
        self.latest_motor_cmd = msg
        self.last_motor_cmd_time = time.time()
        
        # Health checks
        self._check_motor_cmd_health(msg)
    
    def control_cmd_callback(self, msg: BoosterControlCmd):
        """Monitor control commands"""
        self.control_cmd_count += 1
        
        # Log control commands
        self.get_logger().info(f"Control command: vx={msg.vx:.2f}, vy={msg.vy:.2f}, vyaw={msg.vyaw:.2f}")
    
    def _check_sensor_health(self, msg: BoosterSensorData):
        """Check sensor data health"""
        # Check for reasonable values
        if any(abs(pos) > 10.0 for pos in msg.joint_positions):
            self.get_logger().warn("Joint positions out of reasonable range")
        
        if any(abs(vel) > 50.0 for vel in msg.joint_velocities):
            self.get_logger().warn("Joint velocities out of reasonable range")
        
        if any(abs(tau) > 100.0 for tau in msg.joint_torques):
            self.get_logger().warn("Joint torques out of reasonable range")
        
        # Check IMU values
        if any(abs(rpy) > 3.14 for rpy in msg.imu_rpy):
            self.get_logger().warn("IMU RPY values out of reasonable range")
    
    def _check_motor_cmd_health(self, msg: BoosterMotorCmd):
        """Check motor command health"""
        # Check for reasonable values
        if any(abs(pos) > 10.0 for pos in msg.joint_positions):
            self.get_logger().warn("Motor command positions out of reasonable range")
        
        if any(kp < 0.0 or kp > 100.0 for kp in msg.joint_kp):
            self.get_logger().warn("Motor command Kp values out of reasonable range")
        
        if any(kd < 0.0 or kd > 100.0 for kd in msg.joint_kd):
            self.get_logger().warn("Motor command Kd values out of reasonable range")
    
    def monitor_callback(self):
        """Main monitoring callback"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Check for timeouts
        sensor_timeout = current_time - self.last_sensor_time > self.sensor_timeout_threshold
        motor_cmd_timeout = current_time - self.last_motor_cmd_time > self.motor_cmd_timeout_threshold
        
        # Calculate frequencies
        sensor_freq = self.sensor_data_count / uptime if uptime > 0 else 0
        motor_cmd_freq = self.motor_cmd_count / uptime if uptime > 0 else 0
        
        # Log status
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"ROBOT MONITOR STATUS - Uptime: {uptime:.1f}s")
        self.get_logger().info("=" * 60)
        
        # Data flow status
        self.get_logger().info(f"Sensor Data: {self.sensor_data_count} messages ({sensor_freq:.1f} Hz)")
        self.get_logger().info(f"Motor Commands: {self.motor_cmd_count} messages ({motor_cmd_freq:.1f} Hz)")
        self.get_logger().info(f"Control Commands: {self.control_cmd_count} messages")
        
        # Timeout warnings
        if sensor_timeout:
            self.get_logger().warn("⚠️  SENSOR DATA TIMEOUT - No data received recently")
        else:
            self.get_logger().info("✅ Sensor data flowing normally")
        
        if motor_cmd_timeout:
            self.get_logger().warn("⚠️  MOTOR COMMANDS TIMEOUT - No commands received recently")
        else:
            self.get_logger().info("✅ Motor commands flowing normally")
        
        # Frequency analysis
        if sensor_freq > 0:
            if 400 <= sensor_freq <= 600:
                self.get_logger().info("✅ Sensor data frequency is good (400-600 Hz)")
            else:
                self.get_logger().warn(f"⚠️  Sensor data frequency unusual: {sensor_freq:.1f} Hz")
        
        if motor_cmd_freq > 0:
            if 40 <= motor_cmd_freq <= 60:
                self.get_logger().info("✅ Motor command frequency is good (40-60 Hz)")
            else:
                self.get_logger().warn(f"⚠️  Motor command frequency unusual: {motor_cmd_freq:.1f} Hz")
        
        # Latest data analysis
        if self.latest_sensor_data is not None:
            self.get_logger().info(f"Latest sensor data: {len(self.latest_sensor_data.joint_positions)} joints")
            self.get_logger().info(f"IMU RPY: [{self.latest_sensor_data.imu_rpy[0]:.3f}, {self.latest_sensor_data.imu_rpy[1]:.3f}, {self.latest_sensor_data.imu_rpy[2]:.3f}]")
        
        if self.latest_motor_cmd is not None:
            self.get_logger().info(f"Latest motor command: {len(self.latest_motor_cmd.joint_positions)} joints")
            avg_kp = np.mean(self.latest_motor_cmd.joint_kp)
            self.get_logger().info(f"Average Kp: {avg_kp:.2f}")
        
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        monitor = BoosterRobotMonitor()
        rclpy.spin(monitor)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


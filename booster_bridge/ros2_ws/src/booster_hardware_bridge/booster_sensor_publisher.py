#!/usr/bin/env python3

"""
Simplified Booster Sensor Data Publisher
Runs on the robot to collect and publish sensor data for testing
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import time
import logging
import yaml
import os
from typing import Optional

# Import ROS2 message types
from booster_msgs.msg import BoosterSensorData

# Import Booster SDK (ON THE ROBOT)
try:
    from booster_robotics_sdk_python import (
        ChannelFactory,
        B1LocoClient,
        B1LowStateSubscriber,
        LowState,
        B1JointCnt,
    )
    BOOSTER_SDK_AVAILABLE = True
except ImportError:
    BOOSTER_SDK_AVAILABLE = False
    print("Warning: Booster SDK not available. Running in simulation mode.")


class BoosterSensorPublisher(Node):
    """
    Simplified sensor data publisher for testing
    Only publishes sensor data, no motor commands
    """
    
    def __init__(self, config_file: str = None):
        super().__init__('booster_sensor_publisher')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Declare parameters
        self.declare_parameter('use_simulation', False)
        self.declare_parameter('publish_rate', 100.0)  # Hz
        
        # Get parameters
        self.use_simulation = self.get_parameter('use_simulation').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Load configuration
        self.cfg = self._load_config(config_file)
        
        # ROS2 QoS profile for real-time communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Initialize state variables
        self.running = True
        
        # Initialize sensor data arrays
        self._init_sensor_data_arrays()
        
        # Statistics
        self.sensor_data_count = 0
        self.last_stats_time = 0.0
        
        # Initialize Booster SDK or simulation
        if BOOSTER_SDK_AVAILABLE and not self.use_simulation:
            self._init_booster_sdk()
        else:
            self._init_simulation_mode()
        
        # Initialize ROS2 communication
        self._init_ros2_communication(qos_profile)
        
        # Sensor data publishing timer
        self.sensor_timer = self.create_timer(1.0/self.publish_rate, self.publish_sensor_data)
        
        self.logger.info("Booster Sensor Publisher initialized")
        if BOOSTER_SDK_AVAILABLE and not self.use_simulation:
            self.logger.info("Running with Booster SDK")
        else:
            self.logger.info("Running in simulation mode")
    
    def _load_config(self, config_file: str = None):
        """Load configuration file"""
        if config_file is None:
            config_file = os.path.join(
                os.path.dirname(__file__), 
                "config", 
                "robot_config.yaml"
            )
        
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.logger.info(f"Loaded configuration from {config_file}")
            return cfg
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_file} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            "common": {
                "dt": 0.002,
                "stiffness": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 200, 200, 200, 200, 200, 50, 50, 200, 200, 200, 200, 50, 50],
                "damping": [0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5, 5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 3, 3],
                "default_qpos": [0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5, 0, -0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0],
                "torque_limit": [7, 7, 10, 10, 10, 10, 10, 10, 10, 10, 30, 60, 25, 30, 60, 24, 15, 60, 25, 30, 60, 24, 15]
            }
        }
    
    def _init_sensor_data_arrays(self):
        """Initialize sensor data arrays"""
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(23, dtype=np.float32)
        self.dof_vel = np.zeros(23, dtype=np.float32)
        self.dof_pos_latest = np.zeros(23, dtype=np.float32)
    
    def _init_booster_sdk(self):
        """Initialize Booster SDK"""
        try:
            # Initialize SDK
            ChannelFactory.Instance().Init(0)
            
            # Create SDK objects
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            
            # Initialize channels
            self.low_state_subscriber.InitChannel()
            
            self.logger.info("Booster SDK initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Booster SDK: {e}")
            raise
    
    def _init_simulation_mode(self):
        """Initialize simulation mode"""
        self.logger.info("Initializing simulation mode")
        self.simulation_mode = True
        self.sim_time = 0.0
        self._init_sim_sensor_data()
    
    def _init_sim_sensor_data(self):
        """Initialize simulated sensor data"""
        # Default joint positions (standing pose)
        self.default_positions = [
            0.0, 0.0, 0.2, -1.35, 0.0, -0.5, 0.2, 1.35, 0.0, 0.5, 0.0,
            -0.1638, 0.0483, 0.0486, 0.4370, -0.2337, -0.0083, -0.2310,
            -0.0460, -0.0498, 0.3586, -0.2728, 0.0013
        ]
    
    def _init_ros2_communication(self, qos_profile):
        """Initialize ROS2 communication"""
        # ROS2 Publisher for sensor data
        self.sensor_data_publisher = self.create_publisher(
            BoosterSensorData, 'booster/sensor_data', qos_profile
        )
    
    def _low_state_handler(self, low_state_msg: LowState):
        """Process sensor data from Booster SDK"""
        # Safety check
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            return
        
        # Update latest joint positions
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        
        # Update sensor data arrays
        self.projected_gravity[:] = self._rotate_vector_inverse_rpy(
            low_state_msg.imu_state.rpy[0],
            low_state_msg.imu_state.rpy[1],
            low_state_msg.imu_state.rpy[2],
            np.array([0.0, 0.0, -1.0]),
        )
        self.base_ang_vel[:] = low_state_msg.imu_state.gyro
        
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos[i] = motor.q
            self.dof_vel[i] = motor.dq
    
    def _rotate_vector_inverse_rpy(self, roll, pitch, yaw, vector):
        """Rotate vector by inverse RPY (SAME AS deploy.py utils/rotate.py)"""
        # Implement the exact same rotation as deploy.py
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Rotation matrices
        R_x = np.array([[1, 0, 0],
                       [0, cos_roll, -sin_roll],
                       [0, sin_roll, cos_roll]])
        
        R_y = np.array([[cos_pitch, 0, sin_pitch],
                       [0, 1, 0],
                       [-sin_pitch, 0, cos_pitch]])
        
        R_z = np.array([[cos_yaw, -sin_yaw, 0],
                       [sin_yaw, cos_yaw, 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix (inverse)
        return (R_z @ R_y @ R_x).T @ vector
    
    def publish_sensor_data(self):
        """Publish sensor data"""
        if not self.running:
            return
        
        # Create sensor data message
        sensor_data = BoosterSensorData()
        
        if BOOSTER_SDK_AVAILABLE and not self.use_simulation:
            # Use real sensor data from SDK
            # This would be populated by the _low_state_handler
            # For now, use default values
            sensor_data.imu_rpy = [0.0, 0.0, 0.0]
            sensor_data.imu_gyro = [0.0, 0.0, 0.0]
            sensor_data.imu_acc = [0.0, 0.0, -9.81]
            sensor_data.joint_positions = self.dof_pos.tolist()
            sensor_data.joint_velocities = self.dof_vel.tolist()
            sensor_data.joint_torques = [0.0] * 23
        else:
            # Use simulated sensor data (SAME FORMAT AS deploy.py)
            self.sim_time += 1.0 / self.publish_rate
            
            # Simulate IMU data with small oscillations
            imu_rpy = np.array([
                0.05 * np.sin(self.sim_time * 0.5),
                0.02 * np.cos(self.sim_time * 0.3),
                self.sim_time * 0.1
            ])
            imu_gyro = np.array([
                0.01 * np.sin(self.sim_time * 0.7),
                0.01 * np.cos(self.sim_time * 0.4),
                0.01 * np.sin(self.sim_time * 0.6)
            ])
            imu_acc = np.array([0.0, 0.0, -9.81])
            
            # Calculate projected gravity (SAME AS deploy.py)
            self.projected_gravity[:] = self._rotate_vector_inverse_rpy(
                imu_rpy[0], imu_rpy[1], imu_rpy[2],
                np.array([0.0, 0.0, -1.0])
            )
            self.base_ang_vel[:] = imu_gyro
            
            # Simulate joint data with small movements
            for i in range(23):
                # Add small oscillations to default positions
                self.dof_pos[i] = self.default_positions[i] + 0.01 * np.sin(self.sim_time + i * 0.1)
                self.dof_vel[i] = 0.01 * np.cos(self.sim_time + i * 0.1)
            
            # Set sensor data message (SAME FORMAT AS deploy.py)
            sensor_data.imu_rpy = imu_rpy.tolist()
            sensor_data.imu_gyro = imu_gyro.tolist()
            sensor_data.imu_acc = imu_acc.tolist()
            sensor_data.joint_positions = self.dof_pos.tolist()
            sensor_data.joint_velocities = self.dof_vel.tolist()
            sensor_data.joint_torques = [0.0] * 23
        
        # Set timestamp
        sensor_data.timestamp = self.get_clock().now().to_msg()
        
        # Publish
        self.sensor_data_publisher.publish(sensor_data)
        self.sensor_data_count += 1
        
        # Log statistics occasionally
        current_time = time.time()
        if current_time - self.last_stats_time > 10.0:  # Every 10 seconds
            self._log_statistics()
            self.last_stats_time = current_time
    
    def _log_statistics(self):
        """Log system statistics"""
        self.logger.info("=" * 50)
        self.logger.info("SENSOR PUBLISHER STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Sensor data messages published: {self.sensor_data_count}")
        self.logger.info(f"Publish rate: {self.publish_rate} Hz")
        self.logger.info(f"SDK available: {BOOSTER_SDK_AVAILABLE}")
        self.logger.info(f"Simulation mode: {self.use_simulation}")
        self.logger.info("=" * 50)
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.logger.info("Booster Sensor Publisher cleanup completed")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Parse command line arguments (only non-ROS2 args)
        import argparse
        import sys
        
        # Filter out ROS2 arguments
        ros2_args = []
        other_args = []
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] in ['--ros-args', '-r', '--params-file', '-p']:
                # Skip ROS2 arguments
                if sys.argv[i] in ['--ros-args', '--params-file']:
                    i += 1
                elif sys.argv[i] in ['-r', '-p']:
                    i += 2
                else:
                    i += 1
            else:
                other_args.append(sys.argv[i])
                i += 1
        
        parser = argparse.ArgumentParser(description='Booster Sensor Publisher')
        parser.add_argument('--config', type=str, help='Path to configuration file')
        parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
        parser.add_argument('--rate', type=float, default=100.0, help='Publish rate in Hz')
        parsed_args = parser.parse_args(other_args[1:])  # Skip script name
        
        publisher = BoosterSensorPublisher(config_file=parsed_args.config)
        
        # Update parameters directly (like booster_hardware_bridge.py does)
        publisher.use_simulation = parsed_args.simulation
        publisher.publish_rate = parsed_args.rate
        # Update the timer with new rate
        publisher.sensor_timer.cancel()
        publisher.sensor_timer = publisher.create_timer(1.0/publisher.publish_rate, publisher.publish_sensor_data)
        
        # Handle shutdown gracefully
        def shutdown_handler():
            publisher.cleanup()
            rclpy.shutdown()
        
        import signal
        signal.signal(signal.SIGINT, lambda sig, frame: shutdown_handler())
        signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_handler())
        
        rclpy.spin(publisher)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

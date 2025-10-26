#!/usr/bin/env python3

"""
Booster ROSBag Policy Processor
Processes rosbag sensor data and runs policy inference
Publishes observation data and commands for PlotJuggler visualization
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import time
import yaml
import logging
import threading
from typing import Optional
import random

# Import ROS2 message types
from booster_msgs.msg import BoosterSensorData, BoosterMotorCmd, BoosterControlCmd
from std_msgs.msg import Float64MultiArray, Float64, Header

# Import existing booster utilities
import sys
import os
sys.path.append('/home/romela5090/Han/booster_sim2real/deploy_booster')
from utils.policy import Policy
from utils.rotate import rotate_vector_inverse_rpy


class BoosterRosbagPolicyProcessor(Node):
    """
    Booster ROSBag Policy Processor Node
    Processes rosbag sensor data and runs policy inference
    """
    
    def __init__(self):
        super().__init__('booster_rosbag_policy_processor')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize parameters
        self.declare_parameter('config_file', 'T1.yaml')
        self.declare_parameter('policy_interval', 0.02)  # 20ms for policy inference
        self.declare_parameter('use_random_commands', True)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('max_angular_velocity', 1.0)
        
        # Load configuration
        config_file = self.get_parameter('config_file').value
        config_path = f'/home/romela5090/Han/booster_sim2real/deploy_booster/configs/{config_file}'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {e}")
            raise
        
        # Fix policy path to use absolute path
        if 'policy' in self.cfg and 'policy_path' in self.cfg['policy']:
            original_path = self.cfg['policy']['policy_path']
            if original_path.startswith('./'):
                # Convert relative path to absolute path
                self.cfg['policy']['policy_path'] = f'/home/romela5090/Han/booster_sim2real/deploy_booster/{original_path[2:]}'
                self.get_logger().info(f"Updated policy path: {self.cfg['policy']['policy_path']}")
        
        # Initialize policy
        try:
            self.policy = Policy(cfg=self.cfg)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize policy: {e}")
            raise
        
        # Initialize state variables
        self._init_state_variables()
        
        # Setup QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.sensor_subscription = self.create_subscription(
            BoosterSensorData,
            '/booster/sensor_data',
            self.sensor_callback,
            qos_profile
        )
        
        # Publishers for observation data
        self.projected_gravity_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/projected_gravity',
            qos_profile
        )
        
        self.base_ang_vel_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/base_ang_vel',
            qos_profile
        )
        
        self.velocity_commands_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/velocity_commands',
            qos_profile
        )
        
        self.gait_phase_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/gait_phase',
            qos_profile
        )
        
        self.joint_positions_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/joint_positions',
            qos_profile
        )
        
        self.joint_velocities_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/joint_velocities',
            qos_profile
        )
        
        self.previous_actions_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/previous_actions',
            qos_profile
        )
        
        # Publisher for full observation vector
        self.observation_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/observation',
            qos_profile
        )
        
        # Publishers for policy outputs
        self.actions_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/actions',
            qos_profile
        )
        
        self.joint_targets_pub = self.create_publisher(
            Float64MultiArray,
            '/plotjuggler/joint_targets',
            qos_profile
        )
        
        # Timer for policy inference
        self.policy_interval = self.get_parameter('policy_interval').value
        self.policy_timer = self.create_timer(
            self.policy_interval,
            self.policy_callback
        )
        
        # Timer for command generation
        self.use_random_commands = self.get_parameter('use_random_commands').value
        if self.use_random_commands:
            self.command_timer = self.create_timer(
                1.0,  # Update commands every second
                self.generate_random_commands
            )
        
        self.get_logger().info("Booster ROSBag Policy Processor initialized")
        self.get_logger().info(f"Policy interval: {self.policy_interval}s")
        self.get_logger().info(f"Using constant forward commands: {self.use_random_commands}")
    
    def _init_state_variables(self):
        """Initialize state variables"""
        # Sensor data
        self.dof_pos = np.zeros(23, dtype=np.float32)
        self.dof_vel = np.zeros(23, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        
        # Commands
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        
        # Policy state
        self.actions = np.zeros(12, dtype=np.float32)
        self.observation = np.zeros(47, dtype=np.float32)
        
        # Timing
        self.last_policy_time = 0.0
        self.current_time = 0.0
        
        # Command generation
        self.max_vel = self.get_parameter('max_velocity').value
        self.max_ang_vel = self.get_parameter('max_angular_velocity').value
        self.command_change_time = 0.0
    
    def sensor_callback(self, msg: BoosterSensorData):
        """Process incoming sensor data"""
        try:
            # Extract sensor data
            self.dof_pos = np.array(msg.joint_positions, dtype=np.float32)
            self.dof_vel = np.array(msg.joint_velocities, dtype=np.float32)
            
            # Process IMU data
            imu_rpy = np.array(msg.imu_rpy, dtype=np.float32)
            imu_gyro = np.array(msg.imu_gyro, dtype=np.float32)
            
            # Compute projected gravity
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                imu_rpy[0],  # roll
                imu_rpy[1],  # pitch
                imu_rpy[2],  # yaw
                np.array([0.0, 0.0, -1.0])
            )
            
            # Store angular velocity
            self.base_ang_vel[:] = imu_gyro
            
            # Update current time
            self.current_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error processing sensor data: {e}")
    
    def generate_random_commands(self):
        """Generate constant forward velocity commands"""
        if not self.use_random_commands:
            return
        
        # Set constant forward velocity
        self.vx = 0.5  # Forward velocity
        self.vy = 0.0  # No lateral movement
        self.vyaw = 0.0  # No rotation
        
        self.get_logger().info(f"Set constant commands: vx={self.vx:.2f}, vy={self.vy:.2f}, vyaw={self.vyaw:.2f}")
    
    def policy_callback(self):
        """Run policy inference"""
        try:
            # Update current time for policy inference
            self.current_time = time.time()
            
            # Check if we have recent sensor data
            if self.current_time - self.last_policy_time < self.policy_interval:
                return
              
            # Validate data before policy inference
            if np.any(np.isnan(self.dof_pos)) or np.any(np.isinf(self.dof_pos)):
                self.get_logger().warn("Invalid dof_pos data, skipping policy inference")
                return
            if np.any(np.isnan(self.dof_vel)) or np.any(np.isinf(self.dof_vel)):
                self.get_logger().warn("Invalid dof_vel data, skipping policy inference")
                return
            if np.any(np.isnan(self.base_ang_vel)) or np.any(np.isinf(self.base_ang_vel)):
                self.get_logger().warn("Invalid base_ang_vel data, skipping policy inference")
                return
            if np.any(np.isnan(self.projected_gravity)) or np.any(np.isinf(self.projected_gravity)):
                self.get_logger().warn("Invalid projected_gravity data, skipping policy inference")
                return
            
            # Run policy inference
            joint_targets = self.policy.inference(
                time_now=self.current_time,
                dof_pos=self.dof_pos,
                dof_vel=self.dof_vel,
                base_ang_vel=self.base_ang_vel,
                projected_gravity=self.projected_gravity,
                vx=self.vx,
                vy=self.vy,
                vyaw=self.vyaw
            )
            
            # Get observation data from policy
            self.observation = self.policy.obs.copy()
            self.actions = self.policy.actions.copy()
            
            # Publish observation components
            self.publish_observation_data()
            
            # Publish policy outputs
            self.publish_policy_outputs(joint_targets)
            
            self.last_policy_time = self.current_time
            
        except Exception as e:
            self.get_logger().error(f"Error in policy inference: {e}")
    
    def publish_observation_data(self):
        """Publish observation data components"""
        try:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            
            # Check for invalid data and log warnings
            if np.any(np.isnan(self.projected_gravity)) or np.any(np.isinf(self.projected_gravity)):
                self.get_logger().warn("WARNING: Invalid projected_gravity data detected!")
                self.get_logger().warn(f"projected_gravity values: {self.projected_gravity}")
            
            # Projected gravity
            msg = Float64MultiArray()
            msg.data = self.projected_gravity.tolist()
            self.projected_gravity_pub.publish(msg)
            
            # Check for invalid data and log warnings
            if np.any(np.isnan(self.base_ang_vel)) or np.any(np.isinf(self.base_ang_vel)):
                self.get_logger().warn("WARNING: Invalid base_ang_vel data detected!")
                self.get_logger().warn(f"base_ang_vel values: {self.base_ang_vel}")
            
            # Base angular velocity
            msg = Float64MultiArray()
            msg.data = self.base_ang_vel.tolist()
            self.base_ang_vel_pub.publish(msg)
            
            # Velocity commands
            msg = Float64MultiArray()
            msg.data = [self.vx, self.vy, self.vyaw]
            self.velocity_commands_pub.publish(msg)
            
            # Check for invalid observation data
            if len(self.observation) > 0 and (np.any(np.isnan(self.observation)) or np.any(np.isinf(self.observation))):
                self.get_logger().warn("WARNING: Invalid observation data detected!")
                self.get_logger().warn(f"observation values: {self.observation}")
            
            # Gait phase (from observation) - validate first
            if len(self.observation) >= 11 and not np.any(np.isnan(self.observation[9:11])) and not np.any(np.isinf(self.observation[9:11])):
                msg = Float64MultiArray()
                # Convert numpy float32 to Python float64
                msg.data = [float(self.observation[9]), float(self.observation[10])]  # cos, sin
                self.gait_phase_pub.publish(msg)
            else:
                msg = Float64MultiArray()
                msg.data = [0.0, 0.0]  # Default values
                self.gait_phase_pub.publish(msg)
                self.get_logger().warn("NOTIFICATION: Using default gait phase values [0.0, 0.0]")
        
            # Joint positions (filtered) - validate first
            if len(self.observation) >= 23 and not np.any(np.isnan(self.observation[11:23])) and not np.any(np.isinf(self.observation[11:23])):
                msg = Float64MultiArray()
                # Convert numpy float32 to Python float64
                msg.data = [float(x) for x in self.observation[11:23]]  # Joint positions 11-22
                self.joint_positions_pub.publish(msg)
            else:
                msg = Float64MultiArray()
                msg.data = [0.0] * 12  # Default values
                self.joint_positions_pub.publish(msg)
                self.get_logger().warn("NOTIFICATION: Using default joint positions [0.0] * 12")
            
            # Joint velocities (filtered) - validate first
            if len(self.observation) >= 35 and not np.any(np.isnan(self.observation[23:35])) and not np.any(np.isinf(self.observation[23:35])):
                msg = Float64MultiArray()
                # Convert numpy float32 to Python float64
                msg.data = [float(x) for x in self.observation[23:35]]  # Joint velocities 11-22
                self.joint_velocities_pub.publish(msg)
            else:
                msg = Float64MultiArray()
                msg.data = [0.0] * 12  # Default values
                self.joint_velocities_pub.publish(msg)
                self.get_logger().warn("NOTIFICATION: Using default joint velocities [0.0] * 12")
            
            # Previous actions - validate first
            if len(self.observation) >= 47 and not np.any(np.isnan(self.observation[35:47])) and not np.any(np.isinf(self.observation[35:47])):
                msg = Float64MultiArray()
                # Convert numpy float32 to Python float64
                msg.data = [float(x) for x in self.observation[35:47]]  # Previous actions
                self.previous_actions_pub.publish(msg)
            else:
                msg = Float64MultiArray()
                msg.data = [0.0] * 12  # Default values
                self.previous_actions_pub.publish(msg)
                self.get_logger().warn("NOTIFICATION: Using default previous actions [0.0] * 12")
            
            # Full observation vector - validate first
            if len(self.observation) >= 47 and not np.any(np.isnan(self.observation)) and not np.any(np.isinf(self.observation)):
                msg = Float64MultiArray()
                # Convert numpy float32 to Python float64
                msg.data = [float(x) for x in self.observation]
                self.observation_pub.publish(msg)
            else:
                msg = Float64MultiArray()
                msg.data = [0.0] * 47  # Default values
                self.observation_pub.publish(msg)
                self.get_logger().warn("NOTIFICATION: Using default observation vector [0.0] * 47")
                
        except Exception as e:
            self.get_logger().error(f"Error publishing observation data: {e}")
    
    def publish_policy_outputs(self, joint_targets):
        """Publish policy outputs"""
        try:
            # Check for invalid actions data
            if len(self.actions) > 0 and (np.any(np.isnan(self.actions)) or np.any(np.isinf(self.actions))):
                self.get_logger().warn("WARNING: Invalid actions data detected!")
                self.get_logger().warn(f"actions values: {self.actions}")
            
            # Actions - validate first
            if len(self.actions) >= 12 and not np.any(np.isnan(self.actions)) and not np.any(np.isinf(self.actions)):
                msg = Float64MultiArray()
                # Convert numpy float32 to Python float64
                msg.data = [float(x) for x in self.actions]
                self.actions_pub.publish(msg)
            else:
                msg = Float64MultiArray()
                msg.data = [0.0] * 12  # Default values
                self.actions_pub.publish(msg)
                self.get_logger().warn("NOTIFICATION: Using default actions [0.0] * 12")
            
            # Check for invalid joint targets data
            if len(joint_targets) > 0 and (np.any(np.isnan(joint_targets)) or np.any(np.isinf(joint_targets))):
                self.get_logger().warn("WARNING: Invalid joint_targets data detected!")
                self.get_logger().warn(f"joint_targets values: {joint_targets}")
            
            # Joint targets - validate first
            if len(joint_targets) >= 23 and not np.any(np.isnan(joint_targets)) and not np.any(np.isinf(joint_targets)):
                msg = Float64MultiArray()
                # Convert numpy float32 to Python float64
                msg.data = [float(x) for x in joint_targets]
                self.joint_targets_pub.publish(msg)
            else:
                msg = Float64MultiArray()
                msg.data = [0.0] * 23  # Default values
                self.joint_targets_pub.publish(msg)
                self.get_logger().warn("NOTIFICATION: Using default joint targets [0.0] * 23")
                
        except Exception as e:
            self.get_logger().error(f"Error publishing policy outputs: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        processor = BoosterRosbagPolicyProcessor()
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

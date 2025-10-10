#!/usr/bin/env python3

"""
Enhanced Booster Policy Server Node
Runs complete policy inference loop with motor command publishing
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

# Import ROS2 message types
from booster_msgs.msg import BoosterSensorData, BoosterMotorCmd, BoosterControlCmd
from booster_msgs.srv import BoosterRobotMode

# Import existing booster utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../deploy_booster'))
from utils.policy import Policy
from utils.remote_control_service import RemoteControlService
from utils.timer import TimerConfig, Timer
from utils.rotate import rotate_vector_inverse_rpy


class BoosterPolicyServer(Node):
    """
    Booster Policy Server Node
    Runs complete policy inference and motor command publishing
    """
    
    def __init__(self):
        super().__init__('booster_policy_server')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize parameters
        self.declare_parameter('config_file', 'T1.yaml')
        self.declare_parameter('policy_interval', 0.02)  # 20ms for policy inference
        self.declare_parameter('control_interval', 0.002)  # 2ms for control loop
        
        # Load configuration
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        config_path = os.path.join(os.path.dirname(__file__), '../../../deploy_booster/configs', config_file)
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # Initialize components
        self.policy = Policy(cfg=self.cfg)
        self.remote_control = RemoteControlService()
        
        # Initialize state variables
        self._init_state_variables()
        self._init_timer()
        
        # ROS2 QoS profile for real-time communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # ROS2 Publishers and Subscribers
        self.motor_cmd_publisher = self.create_publisher(
            BoosterMotorCmd, 
            'booster/motor_cmd', 
            qos_profile
        )
        
        self.sensor_data_subscriber = self.create_subscription(
            BoosterSensorData,
            'booster/sensor_data',
            self.sensor_data_callback,
            qos_profile
        )
        
        # Control command subscriber (for remote control)
        self.control_cmd_subscriber = self.create_subscription(
            BoosterControlCmd,
            'booster/control_cmd',
            self.control_cmd_callback,
            qos_profile
        )
        
        # Robot mode service client
        self.robot_mode_client = self.create_client(BoosterRobotMode, 'booster/robot_mode')
        
        # Timers for control loops
        self.policy_timer = self.create_timer(
            self.get_parameter('policy_interval').get_parameter_value().double_value,
            self.policy_inference_callback
        )
        
        # State variables for control
        self.latest_sensor_data: Optional[BoosterSensorData] = None
        self.latest_control_cmd: Optional[BoosterControlCmd] = None
        self.dof_targets = np.zeros(23, dtype=np.float32)
        self.filtered_dof_targets = np.zeros(23, dtype=np.float32)
        self.running = True
        
        # Statistics
        self.inference_count = 0
        self.motor_cmd_count = 0
        self.last_stats_time = 0.0
        
        self.logger.info("Enhanced Booster Policy Server initialized")
        self.logger.info(f"Policy interval: {self.get_parameter('policy_interval').get_parameter_value().double_value}s")
        self.logger.info(f"Control interval: {self.get_parameter('control_interval').get_parameter_value().double_value}s")
    
    def _init_state_variables(self):
        """Initialize state variables"""
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(23, dtype=np.float32)
        self.dof_vel = np.zeros(23, dtype=np.float32)
        self.dof_pos_latest = np.zeros(23, dtype=np.float32)
    
    def _init_timer(self):
        """Initialize timing system"""
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_policy_time = self.timer.get_time()
        self.next_control_time = self.timer.get_time()
    
    def sensor_data_callback(self, msg: BoosterSensorData):
        """Process incoming sensor data from robot simulator"""
        self.latest_sensor_data = msg
        
        # Update timer (simulate time progression)
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        
        # Safety check for IMU values
        if abs(msg.imu_rpy[0]) > 1.0 or abs(msg.imu_rpy[1]) > 1.0:
            self.logger.warning(f"IMU base rpy values are too large: {msg.imu_rpy}")
            self.running = False
            return
        
        # Update latest joint positions
        for i in range(23):
            self.dof_pos_latest[i] = msg.joint_positions[i]
        
        # Update state variables for policy inference (only when needed)
        if time_now >= self.next_policy_time:
            # Calculate projected gravity
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                msg.imu_rpy[0], msg.imu_rpy[1], msg.imu_rpy[2],
                np.array([0.0, 0.0, -1.0])
            )
            self.base_ang_vel[:] = msg.imu_gyro
            
            # Update joint states
            self.dof_pos[:] = msg.joint_positions
            self.dof_vel[:] = msg.joint_velocities
    
    def control_cmd_callback(self, msg: BoosterControlCmd):
        """Process control commands (velocity commands)"""
        self.latest_control_cmd = msg
        
        # Handle mode switching requests
        if msg.start_custom_mode:
            self.request_robot_mode(1)  # kCustom
        elif msg.start_rl_gait:
            self.request_robot_mode(2)  # kStand (placeholder)
    
    def policy_inference_callback(self):
        """Execute policy inference and publish motor commands"""
        if not self.running or self.latest_sensor_data is None:
            return
        
        time_now = self.timer.get_time()
        if time_now < self.next_policy_time:
            return
        
        self.next_policy_time += self.policy.get_policy_interval()
        
        # Get control commands
        vx = vy = vyaw = 0.0
        if self.latest_control_cmd is not None:
            vx = self.latest_control_cmd.vx
            vy = self.latest_control_cmd.vy
            vyaw = self.latest_control_cmd.vyaw
        else:
            # Fallback to remote control service
            vx = self.remote_control.get_vx_cmd()
            vy = self.remote_control.get_vy_cmd()
            vyaw = self.remote_control.get_vyaw_cmd()
        
        # Execute policy inference
        try:
            start_time = time.perf_counter()
            
            self.dof_targets[:] = self.policy.inference(
                time_now=time_now,
                dof_pos=self.dof_pos,
                dof_vel=self.dof_vel,
                base_ang_vel=self.base_ang_vel,
                projected_gravity=self.projected_gravity,
                vx=vx,
                vy=vy,
                vyaw=vyaw
            )
            
            inference_time = time.perf_counter() - start_time
            
            # Apply filtering (same as original code)
            self.filtered_dof_targets = (self.filtered_dof_targets * 0.8 + 
                                       self.dof_targets * 0.2)
            
            # Publish motor commands
            self.publish_motor_commands()
            
            # Update statistics
            self.inference_count += 1
            
            # Log performance occasionally
            if self.inference_count % 50 == 0:  # Every 50 inferences
                self.logger.info(f"Policy inference #{self.inference_count}: {inference_time*1000:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Policy inference failed: {e}")
            self.running = False
    
    def publish_motor_commands(self):
        """Publish motor commands to robot simulator"""
        if not self.running:
            return
        
        # Create motor command message
        motor_cmd = BoosterMotorCmd()
        motor_cmd.cmd_type = 0  # SERIAL
        
        # Set joint commands
        for i in range(23):
            motor_cmd.joint_positions[i] = self.filtered_dof_targets[i]
            motor_cmd.joint_kp[i] = self.cfg["common"]["stiffness"][i]
            motor_cmd.joint_kd[i] = self.cfg["common"]["damping"][i]
            motor_cmd.joint_torques[i] = 0.0  # Default to position control
            
            # Handle parallel mechanism joints (torque control)
            if i in self.cfg["mech"]["parallel_mech_indexes"]:
                motor_cmd.joint_positions[i] = self.dof_pos_latest[i]
                motor_cmd.joint_torques[i] = np.clip(
                    (self.filtered_dof_targets[i] - self.dof_pos_latest[i]) * 
                    self.cfg["common"]["stiffness"][i],
                    -self.cfg["common"]["torque_limit"][i],
                    self.cfg["common"]["torque_limit"][i]
                )
                motor_cmd.joint_kp[i] = 0.0
        
        # Set timestamp
        motor_cmd.timestamp = self.get_clock().now().to_msg()
        
        # Publish
        self.motor_cmd_publisher.publish(motor_cmd)
        self.motor_cmd_count += 1
        
        # Log motor commands occasionally
        if self.motor_cmd_count % 250 == 0:  # Every 250 commands (5 seconds at 50Hz)
            self.logger.info(f"Published motor command #{self.motor_cmd_count}")
    
    def request_robot_mode(self, mode: int):
        """Request robot mode change"""
        if not self.robot_mode_client.wait_for_service(timeout_sec=1.0):
            self.logger.warning("Robot mode service not available")
            return
        
        request = BoosterRobotMode.Request()
        request.mode = mode
        
        future = self.robot_mode_client.call_async(request)
        # Note: In a real implementation, you might want to handle the response
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.remote_control.close()
        self.logger.info("Enhanced Booster Policy Server cleanup completed")
        self.logger.info(f"Total inferences: {self.inference_count}")
        self.logger.info(f"Total motor commands: {self.motor_cmd_count}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        policy_server = BoosterPolicyServerEnhanced()
        
        # Handle shutdown gracefully
        def shutdown_handler():
            policy_server.cleanup()
            rclpy.shutdown()
        
        import signal
        signal.signal(signal.SIGINT, lambda sig, frame: shutdown_handler())
        signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_handler())
        
        rclpy.spin(policy_server)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()




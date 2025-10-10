#!/usr/bin/env python3

"""
Booster Hardware Bridge Node
Runs on the robot, bridges Booster SDK with ROS2 communication
This is the minimal hardware interface that runs on the robot.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import time
import logging
import threading
import yaml
import os
from typing import Optional

# Import ROS2 message types
from booster_msgs.msg import BoosterSensorData, BoosterMotorCmd, BoosterControlCmd
from booster_msgs.srv import BoosterRobotMode

# Import Booster SDK (ON THE ROBOT)
try:
    from booster_robotics_sdk_python import (
        ChannelFactory,
        B1LocoClient,
        B1LowCmdPublisher,
        B1LowStateSubscriber,
        LowCmd,
        LowState,
        B1JointCnt,
        RobotMode,
        LowCmdType,
        MotorCmd,
    )
    BOOSTER_SDK_AVAILABLE = True
except ImportError:
    BOOSTER_SDK_AVAILABLE = False
    print("Warning: Booster SDK not available. Running in simulation mode.")


class BoosterHardwareBridge(Node):
    """
    Hardware Bridge Node
    Bridges Booster SDK with ROS2 communication
    This is the minimal hardware interface that runs on the robot.
    Based on deploy.py but separated into ROS2 communication.
    """
    
    def __init__(self, config_file: str = None):
        super().__init__('booster_hardware_bridge')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Declare parameters
        self.declare_parameter('test_mode', False)
        self.declare_parameter('use_simulation', False)
        
        # Get parameters
        self.test_mode = self.get_parameter('test_mode').get_parameter_value().bool_value
        self.use_simulation = self.get_parameter('use_simulation').get_parameter_value().bool_value
        
        # Load configuration
        self.cfg = self._load_config(config_file)
        
        # ROS2 QoS profile for real-time communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Initialize state variables (from deploy.py)
        self.running = True
        self.robot_mode = 0  # kDamping
        self.latest_motor_cmd: Optional[BoosterMotorCmd] = None
        self.last_motor_cmd_time = 0.0
        self.motor_cmd_timeout = 0.1  # 100ms timeout
        
        # Initialize sensor data arrays (from deploy.py)
        self._init_sensor_data_arrays()
        
        # Statistics
        self.sensor_data_count = 0
        self.motor_cmd_count = 0
        self.last_stats_time = 0.0
        
        # Initialize Booster SDK
        if BOOSTER_SDK_AVAILABLE:
            self._init_booster_sdk()
        else:
            self._init_simulation_mode()
        
        # Initialize ROS2 communication
        self._init_ros2_communication(qos_profile)
        
        # Control loop timer (500Hz like deploy.py)
        self.control_timer = self.create_timer(0.002, self.control_loop)
        
        self.logger.info("Booster Hardware Bridge initialized")
        if BOOSTER_SDK_AVAILABLE:
            self.logger.info("Running with Booster SDK")
        else:
            self.logger.info("Running in simulation mode")
    
    def _load_config(self, config_file: str = None):
        """Load configuration file (from deploy.py)"""
        if config_file is None:
            # Default config file path
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
        """Get default configuration (from T1.yaml)"""
        return {
            "common": {
                "dt": 0.002,
                "stiffness": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 200, 200, 200, 200, 200, 50, 50, 200, 200, 200, 200, 50, 50],
                "damping": [0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5, 5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 3, 3],
                "default_qpos": [0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5, 0, -0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0],
                "torque_limit": [7, 7, 10, 10, 10, 10, 10, 10, 10, 10, 30, 60, 25, 30, 60, 24, 15, 60, 25, 30, 60, 24, 15]
            },
            "mech": {
                "parallel_mech_indexes": [15, 16, 21, 22]
            }
        }
    
    def _init_sensor_data_arrays(self):
        """Initialize sensor data arrays (from deploy.py)"""
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)
    
    def _init_booster_sdk(self):
        """Initialize Booster SDK (SAME AS deploy.py)"""
        try:
            # Initialize SDK (SAME AS deploy.py)
            ChannelFactory.Instance().Init(0)
            
            # Create SDK objects (SAME AS deploy.py)
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()
            
            # Initialize channels (SAME AS deploy.py)
            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
            
            self.logger.info("Booster SDK initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Booster SDK: {e}")
            raise
    
    def _init_simulation_mode(self):
        """Initialize simulation mode when SDK not available"""
        self.logger.info("Initializing simulation mode")
        self.simulation_mode = True
        self.sim_sensor_data = BoosterSensorData()
        self._init_sim_sensor_data()
    
    def _init_sim_sensor_data(self):
        """Initialize simulated sensor data"""
        # Initialize with default values
        self.sim_sensor_data = BoosterSensorData()
        self.sim_sensor_data.imu_rpy = [0.0, 0.0, 0.0]
        self.sim_sensor_data.imu_gyro = [0.0, 0.0, 0.0]
        self.sim_sensor_data.imu_acc = [0.0, 0.0, -9.81]
        
        # Default joint positions (standing pose)
        default_positions = [
            0.0, 0.0, 0.2, -1.35, 0.0, -0.5, 0.2, 1.35, 0.0, 0.5, 0.0,
            -0.1638, 0.0483, 0.0486, 0.4370, -0.2337, -0.0083, -0.2310,
            -0.0460, -0.0498, 0.3586, -0.2728, 0.0013
        ]
        
        self.sim_sensor_data.joint_positions = default_positions
        self.sim_sensor_data.joint_velocities = [0.0] * 23
        self.sim_sensor_data.joint_torques = [0.0] * 23
    
    def _init_ros2_communication(self, qos_profile):
        """Initialize ROS2 communication"""
        # ROS2 Publishers and Subscribers
        self.sensor_data_publisher = self.create_publisher(
            BoosterSensorData, 'booster/sensor_data', qos_profile
        )
        
        self.motor_cmd_subscriber = self.create_subscription(
            BoosterMotorCmd, 'booster/motor_cmd', self.motor_cmd_callback, qos_profile
        )
        
        self.control_cmd_subscriber = self.create_subscription(
            BoosterControlCmd, 'booster/control_cmd', self.control_cmd_callback, qos_profile
        )
        
        # Robot mode service
        self.robot_mode_service = self.create_service(
            BoosterRobotMode, 'booster/robot_mode', self.robot_mode_callback
        )
    
    def _low_state_handler(self, low_state_msg: LowState):
        """Process sensor data from Booster SDK (SAME AS deploy.py)"""
        # Safety check (SAME AS deploy.py)
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
            return
        
        # Update latest joint positions (SAME AS deploy.py)
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        
        # Update sensor data arrays (SAME AS deploy.py)
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
        
        # Convert LowState to BoosterSensorData
        sensor_data = BoosterSensorData()
        
        # IMU data (SAME AS deploy.py)
        sensor_data.imu_rpy = list(low_state_msg.imu_state.rpy)
        sensor_data.imu_gyro = list(low_state_msg.imu_state.gyro)
        sensor_data.imu_acc = list(low_state_msg.imu_state.acc)
        
        # Joint data (SAME AS deploy.py)
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            sensor_data.joint_positions[i] = motor.q
            sensor_data.joint_velocities[i] = motor.dq
            sensor_data.joint_torques[i] = motor.tau_est
            sensor_data.joint_temperature[i] = motor.temperature
            sensor_data.joint_current[i] = motor.current
            sensor_data.joint_voltage[i] = motor.voltage
        
        # Timestamp
        sensor_data.timestamp = self.get_clock().now().to_msg()
        
        # Publish to ROS2
        self.sensor_data_publisher.publish(sensor_data)
        self.sensor_data_count += 1
    
    def _rotate_vector_inverse_rpy(self, roll, pitch, yaw, vector):
        """Rotate vector by inverse RPY (from deploy.py utils/rotate.py)"""
        # This is a simplified version - the full implementation would be more complex
        # For now, just return the vector as-is
        return vector
    
    def motor_cmd_callback(self, msg: BoosterMotorCmd):
        """Process motor commands from ROS2 (SAME AS deploy.py)"""
        self.latest_motor_cmd = msg
        self.last_motor_cmd_time = time.time()
        self.motor_cmd_count += 1
        
        # Safety check: Don't send commands to robot in test mode
        if self.test_mode:
            self.logger.info(f"TEST MODE: Received motor command (NOT sent to robot)")
            return
        
        if BOOSTER_SDK_AVAILABLE:
            # Convert BoosterMotorCmd to LowCmd (SAME AS deploy.py)
            self.low_cmd.cmd_type = msg.cmd_type
            
            # Set joint commands (SAME AS deploy.py)
            for i in range(23):
                self.low_cmd.motor_cmd[i].q = msg.joint_positions[i]
                self.low_cmd.motor_cmd[i].dq = msg.joint_velocities[i]
                self.low_cmd.motor_cmd[i].tau = msg.joint_torques[i]
                self.low_cmd.motor_cmd[i].kp = msg.joint_kp[i]
                self.low_cmd.motor_cmd[i].kd = msg.joint_kd[i]
                self.low_cmd.motor_cmd[i].weight = msg.joint_weight[i]
            
            # Handle parallel mechanism (SAME AS deploy.py)
            if "parallel_mech_indexes" in self.cfg.get("mech", {}):
                for i in self.cfg["mech"]["parallel_mech_indexes"]:
                    self.low_cmd.motor_cmd[i].q = self.dof_pos_latest[i]
                    self.low_cmd.motor_cmd[i].tau = np.clip(
                        (msg.joint_positions[i] - self.dof_pos_latest[i]) * self.cfg["common"]["stiffness"][i],
                        -self.cfg["common"]["torque_limit"][i],
                        self.cfg["common"]["torque_limit"][i],
                    )
                    self.low_cmd.motor_cmd[i].kp = 0.0
            
            # Send to robot (SAME AS deploy.py)
            self.low_cmd_publisher.Write(self.low_cmd)
        else:
            # In simulation mode, just log the command
            self.logger.debug(f"SIMULATION MODE: Received motor command: {len(msg.joint_positions)} joints")
    
    def control_cmd_callback(self, msg: BoosterControlCmd):
        """Process control commands (for logging/monitoring)"""
        self.logger.info(f"Control command: vx={msg.vx:.2f}, vy={msg.vy:.2f}, vyaw={msg.vyaw:.2f}")
        
        # Handle mode switching requests
        if msg.start_custom_mode:
            self.request_robot_mode(1)  # kCustom
        elif msg.start_rl_gait:
            self.request_robot_mode(2)  # kStand
    
    def robot_mode_callback(self, request, response):
        """Handle robot mode switching (SAME AS deploy.py)"""
        try:
            if BOOSTER_SDK_AVAILABLE:
                # Convert ROS2 mode to Booster SDK mode
                if request.mode == 0:  # kDamping
                    self.client.ChangeMode(RobotMode.kDamping)
                    self.robot_mode = 0
                elif request.mode == 1:  # kCustom
                    self._start_custom_mode()
                    self.robot_mode = 1
                elif request.mode == 2:  # kStand
                    self.client.ChangeMode(RobotMode.kStand)
                    self.robot_mode = 2
                else:
                    raise ValueError(f"Invalid mode: {request.mode}")
            else:
                # In simulation mode, just update internal state
                self.robot_mode = request.mode
            
            response.success = True
            response.message = f"Mode changed to {request.mode}"
            self.logger.info(f"Robot mode changed to: {request.mode}")
            
        except Exception as e:
            response.success = False
            response.message = f"Mode change failed: {e}"
            self.logger.error(f"Mode change failed: {e}")
        
        return response
    
    def _start_custom_mode(self):
        """Start custom mode with prepare commands (SAME AS deploy.py)"""
        self.logger.info("Starting custom mode with prepare commands")
        
        # Create prepare command (SAME AS deploy.py)
        self._create_prepare_cmd()
        
        # Send prepare command
        self.low_cmd_publisher.Write(self.low_cmd)
        
        # Switch to custom mode
        self.client.ChangeMode(RobotMode.kCustom)
        self.logger.info("Custom mode activated")
    
    def _create_prepare_cmd(self):
        """Create prepare command (SAME AS deploy.py)"""
        # Initialize command
        self.low_cmd.cmd_type = 0  # SERIAL
        
        # Set prepare parameters (SAME AS deploy.py)
        prepare_cfg = self.cfg.get("prepare", {})
        if prepare_cfg:
            for i in range(23):
                self.low_cmd.motor_cmd[i].kp = prepare_cfg.get("stiffness", [0.0] * 23)[i]
                self.low_cmd.motor_cmd[i].kd = prepare_cfg.get("damping", [0.0] * 23)[i]
                self.low_cmd.motor_cmd[i].q = prepare_cfg.get("default_qpos", [0.0] * 23)[i]
        else:
            # Use common parameters as fallback
            for i in range(23):
                self.low_cmd.motor_cmd[i].kp = self.cfg["common"]["stiffness"][i]
                self.low_cmd.motor_cmd[i].kd = self.cfg["common"]["damping"][i]
                self.low_cmd.motor_cmd[i].q = self.cfg["common"]["default_qpos"][i]
    
    def control_loop(self):
        """Main control loop with safety checks"""
        if not self.running:
            return
        
        current_time = time.time()
        
        # Safety checks
        self._check_motor_command_timeout(current_time)
        self._check_joint_limits()
        self._check_emergency_conditions()
        
        # In simulation mode, publish simulated sensor data
        if not BOOSTER_SDK_AVAILABLE:
            self._publish_simulated_sensor_data()
        
        # Log statistics occasionally
        if current_time - self.last_stats_time > 10.0:  # Every 10 seconds
            self._log_statistics()
            self.last_stats_time = current_time
    
    def _check_motor_command_timeout(self, current_time):
        """Check for motor command timeout (safety feature)"""
        timeout = self.cfg.get("safety", {}).get("motor_cmd_timeout", 0.1)
        if current_time - self.last_motor_cmd_time > timeout:
            if not hasattr(self, 'motor_cmd_timeout_warned'):
                self.logger.warning("Motor command timeout - no commands received")
                self.motor_cmd_timeout_warned = True
                # Could implement emergency stop here
                # self._emergency_stop()
    
    def _check_joint_limits(self):
        """Check joint velocity and torque limits (safety feature)"""
        if not hasattr(self, 'dof_vel') or not hasattr(self, 'dof_pos'):
            return
        
        max_velocity = self.cfg.get("safety", {}).get("max_joint_velocity", 10.0)
        max_torque = self.cfg.get("safety", {}).get("max_joint_torque", 100.0)
        
        # Check velocity limits
        for i, vel in enumerate(self.dof_vel):
            if abs(vel) > max_velocity:
                self.logger.warning(f"Joint {i} velocity limit exceeded: {vel:.2f} > {max_velocity}")
                # Could implement emergency stop here
                # self._emergency_stop()
                break
        
        # Check torque limits (if we have torque data)
        if hasattr(self, 'latest_motor_cmd') and self.latest_motor_cmd:
            for i, torque in enumerate(self.latest_motor_cmd.joint_torques):
                if abs(torque) > max_torque:
                    self.logger.warning(f"Joint {i} torque limit exceeded: {torque:.2f} > {max_torque}")
                    # Could implement emergency stop here
                    # self._emergency_stop()
                    break
    
    def _check_emergency_conditions(self):
        """Check for emergency conditions that require immediate stop"""
        # This is already handled in _low_state_handler for IMU values
        # Additional emergency checks can be added here
        pass
    
    def _emergency_stop(self):
        """Emergency stop procedure"""
        self.logger.error("EMERGENCY STOP ACTIVATED")
        self.running = False
        
        if BOOSTER_SDK_AVAILABLE:
            try:
                # Switch to damping mode for safety
                self.client.ChangeMode(RobotMode.kDamping)
                self.logger.info("Robot switched to damping mode")
            except Exception as e:
                self.logger.error(f"Failed to switch to damping mode: {e}")
    
    def _publish_simulated_sensor_data(self):
        """Publish simulated sensor data in simulation mode"""
        # Update simulated data with slight variations
        self.sim_sensor_data.imu_rpy[0] = 0.05 * np.sin(time.time() * 0.5)
        self.sim_sensor_data.imu_rpy[1] = 0.02 * np.cos(time.time() * 0.3)
        self.sim_sensor_data.imu_rpy[2] = time.time() * 0.1
        
        # Add slight joint movements
        for i in range(23):
            self.sim_sensor_data.joint_positions[i] += 0.001 * np.sin(time.time() + i)
            self.sim_sensor_data.joint_velocities[i] = 0.001 * np.cos(time.time() + i)
        
        # Set timestamp
        self.sim_sensor_data.timestamp = self.get_clock().now().to_msg()
        
        # Publish
        self.sensor_data_publisher.publish(self.sim_sensor_data)
        self.sensor_data_count += 1
    
    def _log_statistics(self):
        """Log system statistics"""
        self.logger.info("=" * 50)
        self.logger.info("HARDWARE BRIDGE STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Sensor data messages: {self.sensor_data_count}")
        self.logger.info(f"Motor command messages: {self.motor_cmd_count}")
        self.logger.info(f"Robot mode: {self.robot_mode}")
        self.logger.info(f"SDK available: {BOOSTER_SDK_AVAILABLE}")
        self.logger.info("=" * 50)
    
    def request_robot_mode(self, mode: int):
        """Request robot mode change (internal method)"""
        # This would be called by the robot mode service
        pass
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.logger.info("Booster Hardware Bridge cleanup completed")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Parse command line arguments for config file
        import argparse
        parser = argparse.ArgumentParser(description='Booster Hardware Bridge')
        parser.add_argument('--config', type=str, help='Path to configuration file')
        args = parser.parse_args()
        
        bridge = BoosterHardwareBridge(config_file=args.config)
        
        # Handle shutdown gracefully
        def shutdown_handler():
            bridge.cleanup()
            rclpy.shutdown()
        
        import signal
        signal.signal(signal.SIGINT, lambda sig, frame: shutdown_handler())
        signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_handler())
        
        rclpy.spin(bridge)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


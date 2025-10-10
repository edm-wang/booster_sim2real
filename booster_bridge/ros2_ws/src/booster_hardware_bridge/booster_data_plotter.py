#!/usr/bin/env python3

"""
Booster Robot Data Plotter
Real-time plotting for Booster robot sensor data and motor commands
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
from typing import Optional

# Import ROS2 message types
from booster_msgs.msg import BoosterSensorData, BoosterMotorCmd, BoosterControlCmd


class BoosterDataPlotter(Node):
    """Real-time data plotting for Booster robot"""
    
    def __init__(self, max_points=200):
        super().__init__('booster_data_plotter')
        
        # Data buffers
        self.max_points = max_points
        self.time_data = deque(maxlen=max_points)
        self.imu_rpy_data = deque(maxlen=max_points)
        self.imu_gyro_data = deque(maxlen=max_points)
        self.imu_acc_data = deque(maxlen=max_points)
        self.joint_pos_data = deque(maxlen=max_points)
        self.joint_vel_data = deque(maxlen=max_points)
        self.joint_torque_data = deque(maxlen=max_points)
        self.motor_cmd_pos_data = deque(maxlen=max_points)
        self.motor_cmd_vel_data = deque(maxlen=max_points)
        self.motor_cmd_torque_data = deque(maxlen=max_points)
        self.control_cmd_data = deque(maxlen=max_points)
        
        # Current data
        self.latest_sensor_data: Optional[BoosterSensorData] = None
        self.latest_motor_cmd: Optional[BoosterMotorCmd] = None
        self.latest_control_cmd: Optional[BoosterControlCmd] = None
        
        # ROS2 QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscribers
        self.sensor_data_subscriber = self.create_subscription(
            BoosterSensorData,
            'booster/sensor_data',
            self.sensor_data_callback,
            qos_profile
        )
        
        self.motor_cmd_subscriber = self.create_subscription(
            BoosterMotorCmd,
            'booster/motor_cmd',
            self.motor_cmd_callback,
            qos_profile
        )
        
        self.control_cmd_subscriber = self.create_subscription(
            BoosterControlCmd,
            'booster/control_cmd',
            self.control_cmd_callback,
            qos_profile
        )
        
        # Plot setup
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 12))
        self.fig.suptitle('Booster Robot Real-time Data Monitoring', fontsize=16)
        
        # Initialize plots
        self._setup_plots()
        
        # Animation
        self.ani = None
        self.running = True
        
        # Start animation
        self.start_plotting()
        
        self.get_logger().info("Booster Data Plotter initialized")
    
    def _setup_plots(self):
        """Setup the subplot structure"""
        # IMU Roll, Pitch, Yaw
        self.ax_imu_rpy = self.axes[0, 0]
        self.ax_imu_rpy.set_title('IMU RPY (rad)')
        self.ax_imu_rpy.set_ylabel('Angle')
        self.ax_imu_rpy.legend(['Roll', 'Pitch', 'Yaw'])
        self.ax_imu_rpy.grid(True)
        
        # IMU Gyroscope
        self.ax_imu_gyro = self.axes[0, 1]
        self.ax_imu_gyro.set_title('IMU Gyro (rad/s)')
        self.ax_imu_gyro.set_ylabel('Angular Velocity')
        self.ax_imu_gyro.legend(['X', 'Y', 'Z'])
        self.ax_imu_gyro.grid(True)
        
        # IMU Accelerometer
        self.ax_imu_acc = self.axes[0, 2]
        self.ax_imu_acc.set_title('IMU Acc (m/s²)')
        self.ax_imu_acc.set_ylabel('Linear Acceleration')
        self.ax_imu_acc.legend(['X', 'Y', 'Z'])
        self.ax_imu_acc.grid(True)
        
        # Joint Positions (mean/std)
        self.ax_joint_pos = self.axes[1, 0]
        self.ax_joint_pos.set_title('Joint Positions (rad)')
        self.ax_joint_pos.set_ylabel('Position')
        self.ax_joint_pos.legend(['Mean', 'Std'])
        self.ax_joint_pos.grid(True)
        
        # Joint Velocities (mean/std)
        self.ax_joint_vel = self.axes[1, 1]
        self.ax_joint_vel.set_title('Joint Velocities (rad/s)')
        self.ax_joint_vel.set_ylabel('Velocity')
        self.ax_joint_vel.legend(['Mean', 'Std'])
        self.ax_joint_vel.grid(True)
        
        # Joint Torques (mean/std)
        self.ax_joint_torque = self.axes[1, 2]
        self.ax_joint_torque.set_title('Joint Torques (Nm)')
        self.ax_joint_torque.set_ylabel('Torque')
        self.ax_joint_torque.legend(['Mean', 'Std'])
        self.ax_joint_torque.grid(True)
        
        # Motor Commands (mean/std)
        self.ax_motor_cmd = self.axes[2, 0]
        self.ax_motor_cmd.set_title('Motor Commands')
        self.ax_motor_cmd.set_ylabel('Command')
        self.ax_motor_cmd.legend(['Pos Mean', 'Vel Mean', 'Torque Mean'])
        self.ax_motor_cmd.grid(True)
        
        # Control Commands
        self.ax_control_cmd = self.axes[2, 1]
        self.ax_control_cmd.set_title('Control Commands')
        self.ax_control_cmd.set_ylabel('Velocity')
        self.ax_control_cmd.legend(['Vx', 'Vy', 'Vyaw'])
        self.ax_control_cmd.grid(True)
        
        # Data Summary
        self.ax_summary = self.axes[2, 2]
        self.ax_summary.set_title('Data Summary')
        self.ax_summary.set_ylabel('Count')
        self.ax_summary.grid(True)
        
        plt.tight_layout()
    
    def sensor_data_callback(self, msg: BoosterSensorData):
        """Process sensor data"""
        self.latest_sensor_data = msg
        current_time = time.time()
        
        # Add to time series
        self.time_data.append(current_time)
        
        # IMU data
        self.imu_rpy_data.append([msg.imu_rpy[0], msg.imu_rpy[1], msg.imu_rpy[2]])
        self.imu_gyro_data.append([msg.imu_gyro[0], msg.imu_gyro[1], msg.imu_gyro[2]])
        self.imu_acc_data.append([msg.imu_acc[0], msg.imu_acc[1], msg.imu_acc[2]])
        
        # Joint data (mean/std for visualization)
        joint_pos = np.array(msg.joint_positions)
        joint_vel = np.array(msg.joint_velocities)
        joint_torque = np.array(msg.joint_torques)
        
        self.joint_pos_data.append([np.mean(joint_pos), np.std(joint_pos)])
        self.joint_vel_data.append([np.mean(joint_vel), np.std(joint_vel)])
        self.joint_torque_data.append([np.mean(joint_torque), np.std(joint_torque)])
    
    def motor_cmd_callback(self, msg: BoosterMotorCmd):
        """Process motor commands"""
        self.latest_motor_cmd = msg
        current_time = time.time()
        
        # Motor command data (mean/std for visualization)
        motor_pos = np.array(msg.joint_positions)
        motor_vel = np.array(msg.joint_velocities)
        motor_torque = np.array(msg.joint_torques)
        
        self.motor_cmd_pos_data.append([np.mean(motor_pos), np.std(motor_pos)])
        self.motor_cmd_vel_data.append([np.mean(motor_vel), np.std(motor_vel)])
        self.motor_cmd_torque_data.append([np.mean(motor_torque), np.std(motor_torque)])
    
    def control_cmd_callback(self, msg: BoosterControlCmd):
        """Process control commands"""
        self.latest_control_cmd = msg
        current_time = time.time()
        
        # Control command data
        self.control_cmd_data.append([msg.vx, msg.vy, msg.vyaw])
    
    def _animate(self, frame):
        """Animation function for real-time plotting"""
        if not self.running or len(self.time_data) < 2:
            return
        
        # Convert to numpy arrays
        times = np.array(self.time_data)
        
        # Clear and replot
        self.ax_imu_rpy.clear()
        self.ax_imu_gyro.clear()
        self.ax_imu_acc.clear()
        self.ax_joint_pos.clear()
        self.ax_joint_vel.clear()
        self.ax_joint_torque.clear()
        self.ax_motor_cmd.clear()
        self.ax_control_cmd.clear()
        self.ax_summary.clear()
        
        # Re-setup plots
        self._setup_plots()
        
        # Plot IMU RPY
        if len(self.imu_rpy_data) > 0:
            imu_rpy = np.array(self.imu_rpy_data)
            self.ax_imu_rpy.plot(times, imu_rpy[:, 0], 'r-', label='Roll')
            self.ax_imu_rpy.plot(times, imu_rpy[:, 1], 'g-', label='Pitch')
            self.ax_imu_rpy.plot(times, imu_rpy[:, 2], 'b-', label='Yaw')
        
        # Plot IMU Gyro
        if len(self.imu_gyro_data) > 0:
            imu_gyro = np.array(self.imu_gyro_data)
            self.ax_imu_gyro.plot(times, imu_gyro[:, 0], 'r-', label='X')
            self.ax_imu_gyro.plot(times, imu_gyro[:, 1], 'g-', label='Y')
            self.ax_imu_gyro.plot(times, imu_gyro[:, 2], 'b-', label='Z')
        
        # Plot IMU Acc
        if len(self.imu_acc_data) > 0:
            imu_acc = np.array(self.imu_acc_data)
            self.ax_imu_acc.plot(times, imu_acc[:, 0], 'r-', label='X')
            self.ax_imu_acc.plot(times, imu_acc[:, 1], 'g-', label='Y')
            self.ax_imu_acc.plot(times, imu_acc[:, 2], 'b-', label='Z')
        
        # Plot Joint Positions
        if len(self.joint_pos_data) > 0:
            joint_pos = np.array(self.joint_pos_data)
            self.ax_joint_pos.plot(times, joint_pos[:, 0], 'r-', label='Mean')
            self.ax_joint_pos.plot(times, joint_pos[:, 1], 'g--', label='Std')
        
        # Plot Joint Velocities
        if len(self.joint_vel_data) > 0:
            joint_vel = np.array(self.joint_vel_data)
            self.ax_joint_vel.plot(times, joint_vel[:, 0], 'r-', label='Mean')
            self.ax_joint_vel.plot(times, joint_vel[:, 1], 'g--', label='Std')
        
        # Plot Joint Torques
        if len(self.joint_torque_data) > 0:
            joint_torque = np.array(self.joint_torque_data)
            self.ax_joint_torque.plot(times, joint_torque[:, 0], 'r-', label='Mean')
            self.ax_joint_torque.plot(times, joint_torque[:, 1], 'g--', label='Std')
        
        # Plot Motor Commands
        if len(self.motor_cmd_pos_data) > 0:
            motor_pos = np.array(self.motor_cmd_pos_data)
            motor_vel = np.array(self.motor_cmd_vel_data)
            motor_torque = np.array(self.motor_cmd_torque_data)
            self.ax_motor_cmd.plot(times, motor_pos[:, 0], 'r-', label='Pos Mean')
            self.ax_motor_cmd.plot(times, motor_vel[:, 0], 'g-', label='Vel Mean')
            self.ax_motor_cmd.plot(times, motor_torque[:, 0], 'b-', label='Torque Mean')
        
        # Plot Control Commands
        if len(self.control_cmd_data) > 0:
            control_cmd = np.array(self.control_cmd_data)
            self.ax_control_cmd.plot(times, control_cmd[:, 0], 'r-', label='Vx')
            self.ax_control_cmd.plot(times, control_cmd[:, 1], 'g-', label='Vy')
            self.ax_control_cmd.plot(times, control_cmd[:, 2], 'b-', label='Vyaw')
        
        # Plot Summary
        self.ax_summary.bar(['Sensor', 'Motor', 'Control'], 
                          [len(self.time_data), len(self.motor_cmd_pos_data), len(self.control_cmd_data)])
        
        # Update legends
        for ax in [self.ax_imu_rpy, self.ax_imu_gyro, self.ax_imu_acc, 
                  self.ax_joint_pos, self.ax_joint_vel, self.ax_joint_torque,
                  self.ax_motor_cmd, self.ax_control_cmd]:
            ax.legend()
    
    def start_plotting(self):
        """Start the real-time plotting"""
        self.ani = animation.FuncAnimation(self.fig, self._animate, interval=100, blit=False)
        plt.show(block=False)
        self.get_logger().info("Real-time plotting started")
    
    def stop_plotting(self):
        """Stop the real-time plotting"""
        self.running = False
        if self.ani:
            self.ani.event_source.stop()
        plt.close(self.fig)
        self.get_logger().info("Real-time plotting stopped")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        plotter = BoosterDataPlotter()
        
        # Spin the node
        rclpy.spin(plotter)
        
    except KeyboardInterrupt:
        print("\nPlotting interrupted by user")
    except Exception as e:
        print(f"Plotting error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()





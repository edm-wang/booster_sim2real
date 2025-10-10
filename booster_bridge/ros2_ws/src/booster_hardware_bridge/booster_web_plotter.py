#!/usr/bin/env python3

"""
Booster Robot Web-Based Data Plotter
Web-based real-time plotting for Booster robot data
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
import threading
import time
import json
import os
import argparse
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
import signal
import sys
from typing import Optional

# Import ROS2 message types
from booster_msgs.msg import BoosterSensorData, BoosterMotorCmd, BoosterControlCmd


class BoosterWebPlotter(Node):
    """Web-based real-time data plotting for Booster robot"""
    
    def __init__(self, max_points=200, port=8080, plot_dir="/tmp/booster_plots"):
        super().__init__('booster_web_plotter')
        
        # Configuration
        self.max_points = max_points
        self.port = port
        self.plot_dir = plot_dir
        
        # Data buffers
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
        
        # Statistics
        self.update_count = 0
        self.last_data_time = 0.0
        self.running = True
        
        # Create plot directory
        os.makedirs(self.plot_dir, exist_ok=True)
        
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
        
        # Start web server
        self.start_web_server()
        
        # Start plotting thread
        self.plot_thread = threading.Thread(target=self._plotting_loop, daemon=True)
        self.plot_thread.start()
        
        self.get_logger().info(f"Booster Web Plotter initialized on port {self.port}")
        self.get_logger().info(f"Plots saved to: {self.plot_dir}")
        self.get_logger().info(f"Web interface: http://localhost:{self.port}")
    
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
        
        self.update_count += 1
    
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
    
    def _plotting_loop(self):
        """Main plotting loop"""
        while self.running:
            try:
                self._create_plots()
                time.sleep(1.0)  # Update plots every second
            except Exception as e:
                self.get_logger().error(f"Error in plotting loop: {e}")
                time.sleep(1.0)
    
    def _create_plots(self):
        """Create and save all plots"""
        try:
            if len(self.time_data) < 2:
                return
            
            # Convert to numpy arrays
            times = np.array(self.time_data)
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle('Booster Robot Real-time Data Monitoring', fontsize=16)
            
            # IMU RPY
            ax = axes[0, 0]
            if len(self.imu_rpy_data) > 0:
                imu_rpy = np.array(self.imu_rpy_data)
                ax.plot(times, imu_rpy[:, 0], 'r-', label='Roll')
                ax.plot(times, imu_rpy[:, 1], 'g-', label='Pitch')
                ax.plot(times, imu_rpy[:, 2], 'b-', label='Yaw')
            ax.set_title('IMU RPY (rad)')
            ax.set_ylabel('Angle')
            ax.legend()
            ax.grid(True)
            
            # IMU Gyro
            ax = axes[0, 1]
            if len(self.imu_gyro_data) > 0:
                imu_gyro = np.array(self.imu_gyro_data)
                ax.plot(times, imu_gyro[:, 0], 'r-', label='X')
                ax.plot(times, imu_gyro[:, 1], 'g-', label='Y')
                ax.plot(times, imu_gyro[:, 2], 'b-', label='Z')
            ax.set_title('IMU Gyro (rad/s)')
            ax.set_ylabel('Angular Velocity')
            ax.legend()
            ax.grid(True)
            
            # IMU Acc
            ax = axes[0, 2]
            if len(self.imu_acc_data) > 0:
                imu_acc = np.array(self.imu_acc_data)
                ax.plot(times, imu_acc[:, 0], 'r-', label='X')
                ax.plot(times, imu_acc[:, 1], 'g-', label='Y')
                ax.plot(times, imu_acc[:, 2], 'b-', label='Z')
            ax.set_title('IMU Acc (m/s²)')
            ax.set_ylabel('Linear Acceleration')
            ax.legend()
            ax.grid(True)
            
            # Joint Positions
            ax = axes[1, 0]
            if len(self.joint_pos_data) > 0:
                joint_pos = np.array(self.joint_pos_data)
                ax.plot(times, joint_pos[:, 0], 'r-', label='Mean')
                ax.plot(times, joint_pos[:, 1], 'g--', label='Std')
            ax.set_title('Joint Positions (rad)')
            ax.set_ylabel('Position')
            ax.legend()
            ax.grid(True)
            
            # Joint Velocities
            ax = axes[1, 1]
            if len(self.joint_vel_data) > 0:
                joint_vel = np.array(self.joint_vel_data)
                ax.plot(times, joint_vel[:, 0], 'r-', label='Mean')
                ax.plot(times, joint_vel[:, 1], 'g--', label='Std')
            ax.set_title('Joint Velocities (rad/s)')
            ax.set_ylabel('Velocity')
            ax.legend()
            ax.grid(True)
            
            # Joint Torques
            ax = axes[1, 2]
            if len(self.joint_torque_data) > 0:
                joint_torque = np.array(self.joint_torque_data)
                ax.plot(times, joint_torque[:, 0], 'r-', label='Mean')
                ax.plot(times, joint_torque[:, 1], 'g--', label='Std')
            ax.set_title('Joint Torques (Nm)')
            ax.set_ylabel('Torque')
            ax.legend()
            ax.grid(True)
            
            # Motor Commands
            ax = axes[2, 0]
            if len(self.motor_cmd_pos_data) > 0:
                motor_pos = np.array(self.motor_cmd_pos_data)
                motor_vel = np.array(self.motor_cmd_vel_data)
                motor_torque = np.array(self.motor_cmd_torque_data)
                ax.plot(times, motor_pos[:, 0], 'r-', label='Pos Mean')
                ax.plot(times, motor_vel[:, 0], 'g-', label='Vel Mean')
                ax.plot(times, motor_torque[:, 0], 'b-', label='Torque Mean')
            ax.set_title('Motor Commands')
            ax.set_ylabel('Command')
            ax.legend()
            ax.grid(True)
            
            # Control Commands
            ax = axes[2, 1]
            if len(self.control_cmd_data) > 0:
                control_cmd = np.array(self.control_cmd_data)
                ax.plot(times, control_cmd[:, 0], 'r-', label='Vx')
                ax.plot(times, control_cmd[:, 1], 'g-', label='Vy')
                ax.plot(times, control_cmd[:, 2], 'b-', label='Vyaw')
            ax.set_title('Control Commands')
            ax.set_ylabel('Velocity')
            ax.legend()
            ax.grid(True)
            
            # Data Summary
            ax = axes[2, 2]
            ax.bar(['Sensor', 'Motor', 'Control'], 
                  [len(self.time_data), len(self.motor_cmd_pos_data), len(self.control_cmd_data)])
            ax.set_title('Data Summary')
            ax.set_ylabel('Count')
            ax.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.plot_dir, 'booster_data.png')
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Save data as JSON
            data_file = os.path.join(self.plot_dir, 'booster_data.json')
            data = {
                'timestamp': time.time(),
                'update_count': self.update_count,
                'sensor_data_count': len(self.time_data),
                'motor_cmd_count': len(self.motor_cmd_pos_data),
                'control_cmd_count': len(self.control_cmd_data),
                'latest_sensor_data': {
                    'imu_rpy': self.latest_sensor_data.imu_rpy if self.latest_sensor_data else None,
                    'imu_gyro': self.latest_sensor_data.imu_gyro if self.latest_sensor_data else None,
                    'imu_acc': self.latest_sensor_data.imu_acc if self.latest_sensor_data else None,
                } if self.latest_sensor_data else None,
                'latest_motor_cmd': {
                    'joint_positions': self.latest_motor_cmd.joint_positions[:5] if self.latest_motor_cmd else None,
                    'joint_velocities': self.latest_motor_cmd.joint_velocities[:5] if self.latest_motor_cmd else None,
                    'joint_torques': self.latest_motor_cmd.joint_torques[:5] if self.latest_motor_cmd else None,
                } if self.latest_motor_cmd else None,
                'latest_control_cmd': {
                    'vx': self.latest_control_cmd.vx if self.latest_control_cmd else None,
                    'vy': self.latest_control_cmd.vy if self.latest_control_cmd else None,
                    'vyaw': self.latest_control_cmd.vyaw if self.latest_control_cmd else None,
                } if self.latest_control_cmd else None,
            }
            
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            self.get_logger().error(f"Error creating plots: {e}")
    
    def start_web_server(self):
        """Start the web server"""
        class BoosterPlotHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Booster Robot Data Monitor</title>
                        <meta http-equiv="refresh" content="2">
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            .container {{ max-width: 1200px; margin: 0 auto; }}
                            .header {{ text-align: center; margin-bottom: 20px; }}
                            .plot {{ text-align: center; margin: 20px 0; }}
                            .status {{ background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>🤖 Booster Robot Data Monitor</h1>
                                <p>Real-time monitoring of robot sensor data and motor commands</p>
                            </div>
                            
                            <div class="plot">
                                <h2>Real-time Data Plots</h2>
                                <img src="/plot" alt="Booster Data Plot" style="max-width: 100%; height: auto;">
                            </div>
                            
                            <div class="status">
                                <h3>System Status</h3>
                                <p><strong>Update Count:</strong> {self.update_count}</p>
                                <p><strong>Sensor Data Points:</strong> {len(self.time_data)}</p>
                                <p><strong>Motor Commands:</strong> {len(self.motor_cmd_pos_data)}</p>
                                <p><strong>Control Commands:</strong> {len(self.control_cmd_data)}</p>
                                <p><strong>Last Update:</strong> {time.time() - self.last_data_time:.1f}s ago</p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode())
                
                elif self.path == '/plot':
                    plot_file = os.path.join(self.plot_dir, 'booster_data.png')
                    if os.path.exists(plot_file):
                        self.send_response(200)
                        self.send_header('Content-type', 'image/png')
                        self.end_headers()
                        with open(plot_file, 'rb') as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                elif self.path == '/data':
                    data_file = os.path.join(self.plot_dir, 'booster_data.json')
                    if os.path.exists(data_file):
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        with open(data_file, 'rb') as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress log messages
        
        # Start web server in a separate thread
        def run_server():
            try:
                server = HTTPServer(('', self.port), BoosterPlotHandler)
                self.get_logger().info(f"Web server started on port {self.port}")
                server.serve_forever()
            except Exception as e:
                self.get_logger().error(f"Web server error: {e}")
        
        self.web_thread = threading.Thread(target=run_server, daemon=True)
        self.web_thread.start()
    
    def stop_plotting(self):
        """Stop the plotting"""
        self.running = False
        self.get_logger().info("Web plotting stopped")


def main(args=None):
    parser = argparse.ArgumentParser(description='Booster Robot Web Plotter')
    parser.add_argument('--port', type=int, default=8080, help='Web server port')
    parser.add_argument('--plot-dir', type=str, default='/tmp/booster_plots', help='Plot directory')
    parser.add_argument('--max-points', type=int, default=200, help='Maximum data points')
    
    args = parser.parse_args()
    
    rclpy.init(args=args)
    
    try:
        plotter = BoosterWebPlotter(
            max_points=args.max_points,
            port=args.port,
            plot_dir=args.plot_dir
        )
        
        # Spin the node
        rclpy.spin(plotter)
        
    except KeyboardInterrupt:
        print("\nWeb plotting interrupted by user")
    except Exception as e:
        print(f"Web plotting error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()





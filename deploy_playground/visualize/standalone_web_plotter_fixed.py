#!/usr/bin/env python3
"""
Fixed Standalone Web-Based Observation Plotter
==============================================

This is a simplified, more robust version of the web plotter that handles
dimension mismatches and other errors gracefully.
"""

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

class StandaloneWebPlotter:
    """Standalone web-based real-time observation plotting."""
    
    def __init__(self, max_points=200, port=8080, data_file="/tmp/robot_obs.json"):
        self.max_points = max_points
        self.port = port
        self.data_file = data_file
        
        # Data buffers
        self.time_data = deque(maxlen=max_points)
        self.linear_vel_data = deque(maxlen=max_points)
        self.angular_vel_data = deque(maxlen=max_points)
        self.gravity_data = deque(maxlen=max_points)
        self.commands_data = deque(maxlen=max_points)
        self.gait_phase_data = deque(maxlen=max_points)
        self.joint_pos_data = deque(maxlen=max_points)
        self.joint_vel_data = deque(maxlen=max_points)
        self.actions_data = deque(maxlen=max_points)
        
        # Individual joint data buffers (23 joints each)
        self.individual_joint_pos_data = [deque(maxlen=max_points) for _ in range(23)]
        self.individual_joint_vel_data = [deque(maxlen=max_points) for _ in range(23)]
        self.individual_actions_data = [deque(maxlen=max_points) for _ in range(23)]
        
        # Current observation
        self.current_obs = None
        self.current_time = 0.0
        self.last_update_time = 0.0
        
        # Plot directory
        self.plot_dir = "/tmp/robot_plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # HTTP server
        self.server = None
        self.server_thread = None
        self.running = False
        
        # Data monitoring thread
        self.monitor_thread = None
        
        # Get robot IP for web access
        self.robot_ip = self._get_robot_ip()
        
        # Statistics
        self.update_count = 0
        self.last_data_time = 0.0
        
    def _get_robot_ip(self):
        """Get the robot's IP address for web access."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"
    
    def _load_observation_data(self):
        """Load observation data from the data file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                if 'observation' in data and 'timestamp' in data:
                    obs = np.array(data['observation'], dtype=np.float32)
                    timestamp = data['timestamp']
                    self._update_buffers(obs, timestamp)
                    return True
        except Exception as e:
            print(f"⚠️  Error loading data: {e}")
        return False
    
    def _update_buffers(self, obs, timestamp):
        """Update internal data buffers with new observation."""
        self.current_obs = obs
        self.current_time = timestamp
        self.last_update_time = time.time()
        self.update_count += 1
        
        # Add to time series
        self.time_data.append(timestamp)
        
        # Extract components (assuming 85-dim observation)
        if len(obs) >= 85:
            self.linear_vel_data.append(obs[0:3])
            self.angular_vel_data.append(obs[3:6])
            self.gravity_data.append(obs[6:9])
            # Commands are actually joint commands at indices 62-85, not 9-12
            # For plotting, we'll show the first 3 joint commands as representative
            self.commands_data.append(obs[62:65])
            self.gait_phase_data.append(obs[12:16])
            
            # Joint positions (23)
            joint_pos = obs[16:39]
            self.joint_pos_data.append([np.mean(joint_pos), np.std(joint_pos)])
            for i in range(23):
                self.individual_joint_pos_data[i].append(joint_pos[i])
            
            # Joint velocities (23)
            joint_vel = obs[39:62]
            self.joint_vel_data.append([np.mean(joint_vel), np.std(joint_vel)])
            for i in range(23):
                self.individual_joint_vel_data[i].append(joint_vel[i])
            
            # Actions (23)
            actions = obs[62:85]
            self.actions_data.append([np.mean(actions), np.std(actions)])
            for i in range(23):
                self.individual_actions_data[i].append(actions[i])
    
    def _monitor_data_file(self):
        """Monitor the data file for updates."""
        print("📊 Starting data file monitoring...")
        last_mtime = 0
        
        while self.running:
            try:
                if os.path.exists(self.data_file):
                    current_mtime = os.path.getmtime(self.data_file)
                    if current_mtime > last_mtime:
                        if self._load_observation_data():
                            last_mtime = current_mtime
                            self.last_data_time = time.time()
                            if self.update_count % 10 == 0:
                                print(f"📊 Updated {self.update_count} times (last: {time.time() - self.last_data_time:.1f}s ago)")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️  Error in data monitoring: {e}")
                time.sleep(1.0)
    
    def _create_plots(self):
        """Create and save all plots with robust error handling."""
        try:
            if len(self.time_data) < 2:
                return
            
            # Get consistent data lengths and convert to relative time in seconds
            times = np.array(list(self.time_data))
            if len(times) > 0:
                # Convert to relative time in seconds (time since first data point)
                times = times - times[0]
            min_length = len(times)
            
            # Create figure with expanded layout for individual joint plots
            # We need: 3 basic plots + 23 individual joint plots = 26 total plots
            # Use a 6x5 grid (30 plots) to accommodate all
            fig, axes = plt.subplots(6, 5, figsize=(25, 20))
            fig.suptitle(f'Robot Observation Monitor - {time.strftime("%H:%M:%S")}', fontsize=16)
            
            # Helper function to safely plot data
            def safe_plot(ax, times, data, labels, title, ylabel):
                try:
                    if len(data) > 0 and len(data) == len(times):
                        data_array = np.array(data)
                        colors = ['r-', 'g-', 'b-', 'm-']
                        for i, (label, color) in enumerate(zip(labels, colors)):
                            if i < data_array.shape[1]:
                                ax.plot(times, data_array[:, i], color, label=label, linewidth=2)
                        ax.set_title(title)
                        ax.set_ylabel(ylabel)
                        ax.legend()
                        ax.grid(True)
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(title)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title)
            
            # Helper function to plot individual joint positions with proper y-axis scaling
            def safe_plot_single_joint_pos(ax, times, joint_pos_data, action_data, joint_idx, joint_names):
                try:
                    # Clear the axis first to prevent mixing data
                    ax.clear()
                    
                    if len(times) > 0 and len(joint_pos_data) > 0:
                        # Convert deque to list and trim to match times length
                        pos_data = list(joint_pos_data)[-len(times):]
                        act_data = list(action_data)[-len(times):]
                        
                        # Plot joint position (actual) - single value per time point
                        if len(pos_data) == len(times):
                            ax.plot(times, pos_data, 'b-', label='Actual Position', linewidth=2)
                        
                        # Plot action command (input) - single value per time point
                        if len(act_data) == len(times):
                            ax.plot(times, act_data, 'r--', label='Command', linewidth=2)
                        
                        # Set y-axis limits to adapt to actual data range
                        all_data = pos_data + act_data
                        if all_data:
                            data_min = min(all_data)
                            data_max = max(all_data)
                            # Add 10% padding above and below the data range
                            padding = (data_max - data_min) * 0.1
                            ax.set_ylim(data_min - padding, data_max + padding)
                        
                        joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f'J{joint_idx}'
                        ax.set_title(f'{joint_name} - Position', fontsize=10)
                        ax.set_ylabel('Position (rad)')
                        if any(len(data) == len(times) for data in [pos_data, act_data]):
                            ax.legend(loc='upper left', fontsize=8)
                        ax.grid(True, alpha=0.3)
                    else:
                        joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f'J{joint_idx}'
                        ax.text(0.5, 0.5, f'{joint_name}\nNo data\n({len(joint_pos_data)} pts)', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{joint_name}', fontsize=10)
                        # Set reasonable default y-axis when no data
                        ax.set_ylim(-0.1, 0.1)
                except Exception as e:
                    joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f'J{joint_idx}'
                    ax.text(0.5, 0.5, f'{joint_name}\nError: {str(e)[:20]}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{joint_name}', fontsize=10)
                    # Set reasonable default y-axis even on error
                    ax.set_ylim(-0.1, 0.1)
            
            # Joint names for better plot titles
            joint_names = [
                'HeadYaw', 'HeadPitch',
                'LShoulderPitch', 'LShoulderRoll', 'LElbowPitch', 'LElbowYaw',
                'RShoulderPitch', 'RShoulderRoll', 'RElbowPitch', 'RElbowYaw',
                'Waist',
                'LHipPitch', 'LHipRoll', 'LHipYaw', 'LKneePitch', 'CrankUpLeft', 'CrankDownLeft',
                'RHipPitch', 'RHipRoll', 'RHipYaw', 'RKneePitch', 'CrankUpRight', 'CrankDownRight'
            ]
            
            # Plot basic data (first row)
            safe_plot(axes[0, 0], times, list(self.linear_vel_data), ['X', 'Y', 'Z'], 
                     'Linear Velocity (m/s)', 'Velocity')
            safe_plot(axes[0, 1], times, list(self.angular_vel_data), ['X', 'Y', 'Z'], 
                     'Angular Velocity (rad/s)', 'Angular Velocity')
            safe_plot(axes[0, 2], times, list(self.gravity_data), ['X', 'Y', 'Z'], 
                     'Gravity Vector', 'Gravity')
            safe_plot(axes[0, 3], times, list(self.commands_data), ['Joint 0', 'Joint 1', 'Joint 2'], 
                     'Joint Commands (First 3)', 'Command Value')
            safe_plot(axes[0, 4], times, list(self.gait_phase_data), ['Left Cos', 'Left Sin', 'Right Cos', 'Right Sin'], 
                     'Gait Phase', 'Phase Value')
            
            # Plot aggregated joint data (second row)
            safe_plot(axes[1, 0], times, list(self.joint_pos_data), ['Mean', 'Std'], 
                     'Joint Positions (rad)', 'Position')
            safe_plot(axes[1, 1], times, list(self.joint_vel_data), ['Mean', 'Std'], 
                     'Joint Velocities (rad/s)', 'Velocity')
            safe_plot(axes[1, 2], times, list(self.actions_data), ['Mean', 'Std'], 
                     'Joint Actions (rad)', 'Action')
            # Leave axes[1, 3] and axes[1, 4] empty for spacing
            
            # Plot individual joint positions (command vs actual) - 23 joints
            plot_idx = 10  # Start from third row (after 2 rows of 5 plots each)
            for joint_idx in range(23):
                row = plot_idx // 5
                col = plot_idx % 5
                if row < 6 and col < 5:  # Make sure we don't exceed grid bounds
                    safe_plot_single_joint_pos(
                        axes[row, col], 
                        times, 
                        self.individual_joint_pos_data[joint_idx],
                        self.individual_actions_data[joint_idx],
                        joint_idx,
                        joint_names
                    )
                plot_idx += 1
            
            # Set time axis (show last 10 seconds) and ensure proper scaling
            for ax in axes.flat:
                if len(times) > 0:
                    # Ensure we show a proper 10-second window
                    current_time = times[-1]
                    start_time = max(0, current_time - 10)
                    end_time = current_time + 1
                    ax.set_xlim(start_time, end_time)
                    ax.set_xlabel('Time (seconds)')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plot_dir, "observation_plot.png")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")
            # Create error plot
            try:
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                fig.suptitle(f'Error in Plot Generation - {time.strftime("%H:%M:%S")}', fontsize=16)
                for ax in axes.flat:
                    ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)
                plot_path = os.path.join(self.plot_dir, "observation_plot.png")
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
            except:
                pass
    
    def _start_http_server(self):
        """Start HTTP server to serve plots."""
        plotter_instance = self
        
        class PlotHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    time_since_update = time.time() - plotter_instance.last_data_time if plotter_instance.last_data_time > 0 else float('inf')
                    readable_time = "N/A"
                    
                    try:
                        if os.path.exists(plotter_instance.data_file):
                            with open(plotter_instance.data_file, 'r') as f:
                                data = json.load(f)
                                readable_time = data.get('readable_time', 'N/A')
                    except:
                        pass
                    
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Robot Observation Monitor</title>
                        <meta http-equiv="refresh" content="1">
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            .header {{ text-align: center; margin-bottom: 20px; }}
                            .plot {{ text-align: center; margin: 20px 0; }}
                            .info {{ background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                            .timestamp {{ color: #666; font-size: 0.9em; }}
                            .status {{ padding: 5px; border-radius: 3px; }}
                            .status.active {{ background: #d4edda; color: #155724; }}
                            .status.inactive {{ background: #f8d7da; color: #721c24; }}
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <h1>🤖 Robot Observation Monitor</h1>
                            <p>Real-time monitoring of robot state and policy observations</p>
                        </div>
                        
                        <div class="info">
                            <h3>📊 Current Status</h3>
                            <p><strong>Robot IP:</strong> {plotter_instance.robot_ip}</p>
                            <p><strong>Port:</strong> {plotter_instance.port}</p>
                            <p><strong>Data Points:</strong> {len(plotter_instance.time_data)}</p>
                            <p><strong>Updates:</strong> {plotter_instance.update_count}</p>
                            <p><strong>Data Source:</strong> {plotter_instance.data_file}</p>
                            <p><strong>Status:</strong> 
                                <span class="status {'active' if time_since_update < 5 else 'inactive'}">
                                    {'🟢 Active' if time_since_update < 5 else '🔴 Inactive'}
                                </span>
                            </p>
                            <p><strong>Last Update:</strong> 
                                <span class="timestamp">
                                    {readable_time} 
                                    ({time_since_update:.1f}s ago)
                                </span>
                            </p>
                        </div>
                        
                        <div class="plot">
                            <h3>📈 Real-time Plots</h3>
                            <img src="/plot.png" alt="Observation Plot" style="max-width: 100%; height: auto;">
                            <p><em>Plot updates every 1 second</em></p>
                        </div>
                        
                        <div class="info">
                            <h3>🔗 Access Information</h3>
                            <p>This page automatically refreshes every 1 second.</p>
                            <p>To access from your remote computer, open:</p>
                            <p><code>http://{plotter_instance.robot_ip}:{plotter_instance.port}</code></p>
                            <p><strong>Data File:</strong> <code>{plotter_instance.data_file}</code></p>
                        </div>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode())
                    
                elif self.path == '/plot.png':
                    plot_path = os.path.join(plotter_instance.plot_dir, "observation_plot.png")
                    if os.path.exists(plot_path):
                        self.send_response(200)
                        self.send_header('Content-type', 'image/png')
                        self.end_headers()
                        with open(plot_path, 'rb') as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_response(404)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass
        
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), PlotHandler)
            print(f"🌐 Web plotter server started on http://{self.robot_ip}:{self.port}")
            self.server.serve_forever()
        except Exception as e:
            print(f"❌ Failed to start web server: {e}")
    
    def start(self):
        """Start the web-based plotting."""
        print("📊 Starting standalone web-based observation plotting...")
        print(f"📁 Monitoring data file: {self.data_file}")
        self.running = True
        
        # Start data monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_data_file, daemon=True)
        self.monitor_thread.start()
        
        # Start HTTP server in separate thread
        self.server_thread = threading.Thread(target=self._start_http_server, daemon=True)
        self.server_thread.start()
        
        time.sleep(1)
        
        print(f"🌐 Web plotter accessible at: http://{self.robot_ip}:{self.port}")
        print(f"📱 Open this URL in your browser on your remote computer")
        print(f"📊 Waiting for data from: {self.data_file}")
        
        # Create initial empty plot
        self._create_plots()
    
    def stop(self):
        """Stop the web-based plotting."""
        print("📊 Stopping web plotter...")
        self.running = False
        if self.server:
            self.server.shutdown()
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def run_forever(self):
        """Run the plotter forever, updating plots periodically."""
        try:
            while self.running:
                self._create_plots()
                time.sleep(0.1)  # Update every 0.1 seconds (10Hz)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.stop()

def signal_handler(sig, frame):
    print("\n🛑 Shutting down web plotter...")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Fixed Standalone Web-Based Robot Observation Plotter')
    parser.add_argument('--port', type=int, default=8080, help='Port for web server (default: 8080)')
    parser.add_argument('--data-file', type=str, default='/tmp/robot_obs.json', 
                       help='Data file to monitor for observations (default: /tmp/robot_obs.json)')
    parser.add_argument('--max-points', type=int, default=200, 
                       help='Maximum number of data points to display (default: 200)')
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting Fixed Standalone Web-Based Robot Observation Plotter")
    print("=" * 60)
    print(f"📊 Port: {args.port}")
    print(f"📁 Data file: {args.data_file}")
    print(f"📈 Max points: {args.max_points}")
    print("=" * 60)
    
    plotter = StandaloneWebPlotter(
        max_points=args.max_points,
        port=args.port,
        data_file=args.data_file
    )
    
    try:
        plotter.start()
        plotter.run_forever()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    finally:
        plotter.stop()
        print("✅ Web plotter stopped")

if __name__ == "__main__":
    main()

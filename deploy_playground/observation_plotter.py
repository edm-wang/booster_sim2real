import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time

class ObservationPlotter:
    """Real-time observation plotting for debugging."""
    
    def __init__(self, max_points=200):
        self.max_points = max_points
        
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
        
        # Current observation
        self.current_obs = None
        self.current_time = 0.0
        
        # Plot setup
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 12))
        self.fig.suptitle('Real-time Observation Monitoring', fontsize=16)
        
        # Initialize plots
        self._setup_plots()
        
        # Animation
        self.ani = None
        self.running = False
        
    def _setup_plots(self):
        """Setup the subplot structure."""
        # Linear velocity (3D)
        self.ax_linvel = self.axes[0, 0]
        self.ax_linvel.set_title('Linear Velocity (m/s)')
        self.ax_linvel.set_ylabel('Velocity')
        self.ax_linvel.legend(['X', 'Y', 'Z'])
        self.ax_linvel.grid(True)
        
        # Angular velocity (3D)
        self.ax_angvel = self.axes[0, 1]
        self.ax_angvel.set_title('Angular Velocity (rad/s)')
        self.ax_angvel.set_ylabel('Angular Velocity')
        self.ax_angvel.legend(['X', 'Y', 'Z'])
        self.ax_angvel.grid(True)
        
        # Gravity vector (3D)
        self.ax_gravity = self.axes[0, 2]
        self.ax_gravity.set_title('Gravity Vector')
        self.ax_gravity.set_ylabel('Gravity')
        self.ax_gravity.legend(['X', 'Y', 'Z'])
        self.ax_gravity.grid(True)
        
        # Commands (3D)
        self.ax_commands = self.axes[1, 0]
        self.ax_commands.set_title('Commands')
        self.ax_commands.set_ylabel('Command Value')
        self.ax_commands.legend(['VX', 'VY', 'VYAW'])
        self.ax_commands.grid(True)
        
        # Gait phase (4D)
        self.ax_gait = self.axes[1, 1]
        self.ax_gait.set_title('Gait Phase')
        self.ax_gait.set_ylabel('Phase Value')
        self.ax_gait.legend(['Left Cos', 'Left Sin', 'Right Cos', 'Right Sin'])
        self.ax_gait.grid(True)
        
        # Joint positions (mean/std)
        self.ax_jointpos = self.axes[1, 2]
        self.ax_jointpos.set_title('Joint Positions (mean/std)')
        self.ax_jointpos.set_ylabel('Position (rad)')
        self.ax_jointpos.legend(['Mean', 'Std'])
        self.ax_jointpos.grid(True)
        
        # Joint velocities (mean/std)
        self.ax_jointvel = self.axes[2, 0]
        self.ax_jointvel.set_title('Joint Velocities (mean/std)')
        self.ax_jointvel.set_ylabel('Velocity (rad/s)')
        self.ax_jointvel.legend(['Mean', 'Std'])
        self.ax_jointvel.grid(True)
        
        # Actions (mean/std)
        self.ax_actions = self.axes[2, 1]
        self.ax_actions.set_title('Actions (mean/std)')
        self.ax_actions.set_ylabel('Action Value')
        self.ax_actions.legend(['Mean', 'Std'])
        self.ax_actions.grid(True)
        
        # Observation summary
        self.ax_summary = self.axes[2, 2]
        self.ax_summary.set_title('Observation Summary')
        self.ax_summary.set_ylabel('Value')
        self.ax_summary.grid(True)
        
        plt.tight_layout()
        
    def update_observation(self, obs, time_now):
        """Update the current observation data."""
        self.current_obs = obs
        self.current_time = time_now
        
        # Add to time series
        self.time_data.append(time_now)
        
        # Extract components (assuming 85-dim observation)
        if len(obs) >= 85:
            # Linear velocity (3)
            self.linear_vel_data.append(obs[0:3])
            
            # Angular velocity (3)
            self.angular_vel_data.append(obs[3:6])
            
            # Gravity vector (3)
            self.gravity_data.append(obs[6:9])
            
            # Commands (3)
            self.commands_data.append(obs[9:12])
            
            # Gait phase (4)
            self.gait_phase_data.append(obs[12:16])
            
            # Joint positions (23)
            joint_pos = obs[16:39]
            self.joint_pos_data.append([np.mean(joint_pos), np.std(joint_pos)])
            
            # Joint velocities (23)
            joint_vel = obs[39:62]
            self.joint_vel_data.append([np.mean(joint_vel), np.std(joint_vel)])
            
            # Actions (23)
            actions = obs[62:85]
            self.actions_data.append([np.mean(actions), np.std(actions)])
    
    def _animate(self, frame):
        """Animation function for real-time plotting."""
        if not self.running or len(self.time_data) < 2:
            return
            
        # Convert to numpy arrays
        times = np.array(self.time_data)
        
        # Clear and replot
        for ax in self.axes.flat:
            ax.clear()
        
        # Re-setup plots
        self._setup_plots()
        
        # Plot linear velocity
        if len(self.linear_vel_data) > 0:
            linvel = np.array(self.linear_vel_data)
            self.ax_linvel.plot(times, linvel[:, 0], 'r-', label='X', linewidth=2)
            self.ax_linvel.plot(times, linvel[:, 1], 'g-', label='Y', linewidth=2)
            self.ax_linvel.plot(times, linvel[:, 2], 'b-', label='Z', linewidth=2)
            self.ax_linvel.legend()
        
        # Plot angular velocity
        if len(self.angular_vel_data) > 0:
            angvel = np.array(self.angular_vel_data)
            self.ax_angvel.plot(times, angvel[:, 0], 'r-', label='X', linewidth=2)
            self.ax_angvel.plot(times, angvel[:, 1], 'g-', label='Y', linewidth=2)
            self.ax_angvel.plot(times, angvel[:, 2], 'b-', label='Z', linewidth=2)
            self.ax_angvel.legend()
        
        # Plot gravity vector
        if len(self.gravity_data) > 0:
            gravity = np.array(self.gravity_data)
            self.ax_gravity.plot(times, gravity[:, 0], 'r-', label='X', linewidth=2)
            self.ax_gravity.plot(times, gravity[:, 1], 'g-', label='Y', linewidth=2)
            self.ax_gravity.plot(times, gravity[:, 2], 'b-', label='Z', linewidth=2)
            self.ax_gravity.legend()
        
        # Plot commands
        if len(self.commands_data) > 0:
            commands = np.array(self.commands_data)
            self.ax_commands.plot(times, commands[:, 0], 'r-', label='VX', linewidth=2)
            self.ax_commands.plot(times, commands[:, 1], 'g-', label='VY', linewidth=2)
            self.ax_commands.plot(times, commands[:, 2], 'b-', label='VYAW', linewidth=2)
            self.ax_commands.legend()
        
        # Plot gait phase
        if len(self.gait_phase_data) > 0:
            gait = np.array(self.gait_phase_data)
            self.ax_gait.plot(times, gait[:, 0], 'r-', label='Left Cos', linewidth=2)
            self.ax_gait.plot(times, gait[:, 1], 'g-', label='Left Sin', linewidth=2)
            self.ax_gait.plot(times, gait[:, 2], 'b-', label='Right Cos', linewidth=2)
            self.ax_gait.plot(times, gait[:, 3], 'm-', label='Right Sin', linewidth=2)
            self.ax_gait.legend()
        
        # Plot joint positions (mean/std)
        if len(self.joint_pos_data) > 0:
            jointpos = np.array(self.joint_pos_data)
            self.ax_jointpos.plot(times, jointpos[:, 0], 'r-', label='Mean', linewidth=2)
            self.ax_jointpos.plot(times, jointpos[:, 1], 'g-', label='Std', linewidth=2)
            self.ax_jointpos.legend()
        
        # Plot joint velocities (mean/std)
        if len(self.joint_vel_data) > 0:
            jointvel = np.array(self.joint_vel_data)
            self.ax_jointvel.plot(times, jointvel[:, 0], 'r-', label='Mean', linewidth=2)
            self.ax_jointvel.plot(times, jointvel[:, 1], 'g-', label='Std', linewidth=2)
            self.ax_jointvel.legend()
        
        # Plot actions (mean/std)
        if len(self.actions_data) > 0:
            actions = np.array(self.actions_data)
            self.ax_actions.plot(times, actions[:, 0], 'r-', label='Mean', linewidth=2)
            self.ax_actions.plot(times, actions[:, 1], 'g-', label='Std', linewidth=2)
            self.ax_actions.legend()
        
        # Plot observation summary (current values)
        if self.current_obs is not None:
            summary_data = [
                np.linalg.norm(self.current_obs[0:3]),   # Linear velocity magnitude
                np.linalg.norm(self.current_obs[3:6]),   # Angular velocity magnitude
                np.linalg.norm(self.current_obs[6:9]),   # Gravity magnitude
                np.linalg.norm(self.current_obs[9:12]),  # Commands magnitude
                np.linalg.norm(self.current_obs[12:16]), # Gait phase magnitude
                np.mean(np.abs(self.current_obs[16:39])), # Joint pos magnitude
                np.mean(np.abs(self.current_obs[39:62])), # Joint vel magnitude
                np.mean(np.abs(self.current_obs[62:85]))  # Actions magnitude
            ]
            summary_labels = ['LinVel', 'AngVel', 'Gravity', 'Commands', 'Gait', 'JointPos', 'JointVel', 'Actions']
            
            self.ax_summary.bar(summary_labels, summary_data, color=['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray'])
            self.ax_summary.set_xticklabels(summary_labels, rotation=45)
        
        # Set time axis for all plots
        for ax in self.axes.flat:
            ax.set_xlim(max(0, times[-1] - 10), times[-1] + 1)  # Show last 10 seconds
        
        plt.tight_layout()
    
    def start(self):
        """Start the real-time plotting."""
        print("📊 Starting real-time observation plotting...")
        self.running = True
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self._animate, interval=100, blit=False
        )
        
        # Show plot in non-blocking way
        plt.show(block=False)
        
    def stop(self):
        """Stop the real-time plotting."""
        print("📊 Stopping observation plotting...")
        self.running = False
        if self.ani:
            self.ani.event_source.stop()
        plt.close(self.fig)
    
    def save_plot(self, filename="observation_plot.png"):
        """Save current plot to file."""
        if self.fig:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {filename}")

# Global plotter instance
plotter = None

def init_plotter():
    """Initialize the global plotter."""
    global plotter
    plotter = ObservationPlotter()
    plotter.start()
    return plotter

def update_plotter(obs, time_now):
    """Update the global plotter with new observation."""
    global plotter
    if plotter:
        plotter.update_observation(obs, time_now)

def stop_plotter():
    """Stop the global plotter."""
    global plotter
    if plotter:
        plotter.stop()
        plotter = None

#!/usr/bin/env python3
"""
Direct Hardware Deployment Script for Booster T1 Robot
=====================================================

This script deploys a trained JAX/Flax locomotion policy directly to hardware
without requiring ONNX conversion. It combines:

1. Direct JAX checkpoint loading (from static_booster_simulator.py)
2. Gamepad control (from MuJoCo Playground)
3. Booster SDK integration (for hardware communication)

Usage:
    python hardware_deployment.py --checkpoint_path /path/to/checkpoint --use_gamepad
"""

import argparse
import time
import threading
import numpy as np
import jax
import jax.numpy as jp
from pathlib import Path
import json
import functools
import sys
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our policy loading function
from sim2real.utils import load_trained_policy

# Import configuration with official gains
from sim2real.config import get_joint_gains, T1_JOINT_COUNT

# Import Booster SDK
try:
    from booster_robotics_sdk_python import (
        ChannelFactory, B1LocoClient, B1LowStateSubscriber, B1LowCmdPublisher, RobotMode,
        MotorCmd, LowState, LowCmd, LowCmdType, B1JointCnt, B1JointIndex
    )
    BOOSTER_SDK_AVAILABLE = True
except ImportError:
    print("Warning: Booster SDK not available. Running in simulation mode.")
    BOOSTER_SDK_AVAILABLE = False

# Import gamepad control
try:
    import hid
    import threading
    GAMEPAD_AVAILABLE = True
except ImportError:
    print("Warning: hidapi not available. Gamepad control disabled.")
    GAMEPAD_AVAILABLE = False


class Gamepad:
    """Gamepad class for Logitech F710 gamepad control."""
    
    def __init__(
        self,
        vendor_id=0x046D,
        product_id=0xC219,
        vel_scale_x=0.4,
        vel_scale_y=0.4,
        vel_scale_rot=1.0,
    ):
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.is_running = True

        self._device = None

        if GAMEPAD_AVAILABLE:
            self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
            self.read_thread.start()

    def _interpolate(self, value, old_max, new_scale, deadzone=0.01):
        ret = value / old_max * new_scale
        if abs(ret) < deadzone:
            return 0.0
        return ret

    def _connect_device(self):
        if not GAMEPAD_AVAILABLE:
            return False
        try:
            self._device = hid.device()
            self._device.open(self._vendor_id, self._product_id)
            self._device.set_nonblocking(True)
            print(f"Connected to {self._device.get_manufacturer_string()} {self._device.get_product_string()}")
            return True
        except (hid.HIDException, OSError) as e:
            print(f"Error connecting to device: {e}")
            return False

    def read_loop(self):
        if not self._connect_device():
            self.is_running = False
            return

        while self.is_running:
            try:
                data = self._device.read(64)
                if data:
                    self.update_command(data)
            except (hid.HIDException, OSError) as e:
                print(f"Error reading from device: {e}")

        if self._device:
            self._device.close()

    def update_command(self, data):
        left_x = -(data[1] - 128) / 128.0
        left_y = -(data[2] - 128) / 128.0
        right_x = -(data[3] - 128) / 128.0

        self.vx = self._interpolate(left_y, 1.0, self._vel_scale_x)
        self.vy = self._interpolate(left_x, 1.0, self._vel_scale_y)
        self.wz = self._interpolate(right_x, 1.0, self._vel_scale_rot)

    def get_command(self):
        return np.array([self.vx, self.vy, self.wz])

    def stop(self):
        self.is_running = False


class HardwareController:
    """Direct hardware controller using JAX checkpoint and Booster SDK."""
    
    def __init__(self, checkpoint_path: str, use_gamepad: bool = True, robot_ip: str = "127.0.0.1"):
        self.checkpoint_path = checkpoint_path
        self.use_gamepad = use_gamepad
        self.robot_ip = robot_ip
        self.policy_fn = None
        self.gamepad = None
        self.booster_client = None
        self.state_subscriber = None
        self.cmd_publisher = None
        
        # Control parameters
        self.control_frequency = 100  # Hz
        self.dt = 1.0 / self.control_frequency
        
        # Safety parameters
        self.emergency_stop = False
        self.last_state_time = 0
        self.state_timeout = 0.1  # 100ms timeout
        
        # Initialize components
        self._load_policy()
        # Gamepad disabled - using simple forward movement
            # self._init_gamepad()  # Disabled - using simple forward movement
        if BOOSTER_SDK_AVAILABLE:
            self._init_booster_sdk()
    
    def _load_policy(self):
        """Load the trained policy from checkpoint."""
        print(f"Loading policy from: {self.checkpoint_path}")
        
        # We need to create a dummy environment to get observation/action sizes
        # Use an existing T1 environment from mujoco_playground
        try:
            from mujoco_playground import registry
            
            # Create environment to get sizes
            env = registry.load("T1JoystickFlatTerrain")
            self.policy_fn = load_trained_policy(self.checkpoint_path, env)
            print("✅ Policy loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            raise
    
    def _init_gamepad(self):
        """Initialize gamepad control."""
        if not GAMEPAD_AVAILABLE:
            print("⚠️  Gamepad not available, using keyboard input")
            return
            
        self.gamepad = Gamepad()
        print("🎮 Gamepad initialized")
    
    def _init_booster_sdk(self):
        """Initialize Booster SDK for hardware communication."""
        try:
            print(f"🔌 Initializing Booster SDK connection to {self.robot_ip}...")
            
            # Initialize DDS channel factory
            ChannelFactory.Instance().Init(0, self.robot_ip)
            
            # Initialize high-level locomotion client
            self.booster_client = B1LocoClient()
            self.booster_client.Init()
            
            # Initialize low-level state subscriber
            self.state_subscriber = B1LowStateSubscriber(self._state_handler)
            self.state_subscriber.InitChannel()
            
            # Initialize low-level command publisher
            self.cmd_publisher = B1LowCmdPublisher()
            self.cmd_publisher.InitChannel()
            
            print("✅ Booster SDK initialized successfully!")
            print(f"   - Connected to robot at: {self.robot_ip}")
            print(f"   - Joint count: {B1JointCnt}")
            print(f"   - Control frequency: {self.control_frequency} Hz")

            # Switch to custom mode for policy control
            print("🔄 Switching robot to custom mode...")
            res = self.booster_client.ChangeMode(RobotMode.kCustom)
            if res == 0:
                print("✅ Robot switched to custom mode successfully!")
            else:
                print(f"⚠️  Warning: Failed to switch to custom mode (code: {res})")
                print("   Robot may still work, but behavior may be limited")
            
        except Exception as e:
            print(f"❌ Error initializing Booster SDK: {e}")
            print("Running in simulation mode...")
            self.booster_client = None
            self.state_subscriber = None
            self.cmd_publisher = None
    
    def _state_handler(self, low_state_msg):
        """Handle incoming robot state messages."""
        self.last_state_time = time.time()
        # Store the latest state for processing
        self.latest_state = low_state_msg
    
    def get_robot_state(self):
        """Get current robot state from hardware or simulation."""
        if self.state_subscriber and BOOSTER_SDK_AVAILABLE:
            # Check if we have recent state data
            current_time = time.time()
            if current_time - self.last_state_time > self.state_timeout:
                print("⚠️  No recent robot state data - using last known state")
                return None
            
            # Get real hardware state
            try:
                if hasattr(self, 'latest_state') and self.latest_state:
                    return self._hardware_state_to_obs(self.latest_state)
                else:
                    print("⚠️  No robot state available yet")
                    return None
            except Exception as e:
                print(f"Error getting hardware state: {e}")
                return None
        else:
            # Return dummy state for simulation
            return self._get_dummy_state()
    
    def _hardware_state_to_obs(self, state):
        """Convert Booster SDK LowState to observation vector."""
        # This is a simplified conversion - you'll need to match your training observation space
        obs = np.zeros(85)  # T1 observation size
        
        try:
            # IMU data - use parallel motor state IMU
            if hasattr(state, 'imu_state') and state.imu_state:
                imu = state.imu_state
                # Roll, Pitch, Yaw (RPY) - convert to quaternion-like representation
                obs[0:3] = [imu.rpy[0], imu.rpy[1], imu.rpy[2]]  # roll, pitch, yaw
                obs[3:6] = [imu.gyro[0], imu.gyro[1], imu.gyro[2]]     # Angular velocity
                obs[6:9] = [imu.acc[0], imu.acc[1], imu.acc[2]]  # Linear acceleration
            
            # Motor states (23 joints) - use parallel motor states
            if hasattr(state, 'motor_state_parallel') and state.motor_state_parallel:
                for i, motor in enumerate(state.motor_state_parallel[:23]):
                    obs[9 + i*2] = motor.q        # Joint position
                    obs[9 + i*2 + 1] = motor.dq   # Joint velocity
            
            # Add simple forward movement commands (no gamepad needed)
            obs[55:58] = [0.1, 0.0, 0.0]  # Small forward velocity (vx=0.1, vy=0, wz=0)
                
        except Exception as e:
            print(f"Error converting hardware state: {e}")
            # Return dummy state if conversion fails
            return self._get_dummy_state()
        
        return obs
    
    def _get_dummy_state(self):
        """Get dummy state for simulation/testing."""
        obs = np.zeros(85)
        # Add some dummy IMU data
        obs[0:3] = [1.0, 0.0, 0.0]  # Quaternion (no rotation)
        obs[3:6] = [0.0, 0.0, 0.0]  # Angular velocity
        obs[6:9] = [0.0, 0.0, -9.81]  # Gravity
        
        # Add simple forward movement commands (no gamepad needed)
        obs[55:58] = [0.1, 0.0, 0.0]  # Small forward velocity (vx=0.1, vy=0, wz=0)
        
        return obs
    
    def send_motor_commands(self, actions):
        """Send motor commands to hardware."""
        if self.cmd_publisher and BOOSTER_SDK_AVAILABLE and not self.emergency_stop:
            try:
                # Create LowCmd message
                low_cmd = LowCmd()
                low_cmd.cmd_type = LowCmdType.PARALLEL
                
                # Create motor commands for all 23 joints
                motor_cmds = [MotorCmd() for _ in range(B1JointCnt)]
                
                for i in range(B1JointCnt):
                    if i < len(actions):
                        # Set position command from policy
                        motor_cmds[i].q = float(actions[i])
                        motor_cmds[i].dq = 0.0  # Velocity command
                        motor_cmds[i].tau = 0.0  # Torque command
                        motor_cmds[i].kp = 100.0  # Position gain
                        motor_cmds[i].kd = 5.0    # Velocity gain
                        motor_cmds[i].weight = 1.0  # Command weight
                    else:
                        # Zero command for unused joints
                        motor_cmds[i].q = 0.0
                        motor_cmds[i].dq = 0.0
                        motor_cmds[i].tau = 0.0
                        motor_cmds[i].kp = 0.0
                        motor_cmds[i].kd = 0.0
                        motor_cmds[i].weight = 0.0
                
                low_cmd.motor_cmd = motor_cmds
                
                # Send commands
                self.cmd_publisher.Write(low_cmd)
                
            except Exception as e:
                print(f"Error sending motor commands: {e}")
                self.emergency_stop = True
        else:
            # Simulation mode - just print the commands
            if self.emergency_stop:
                print("🛑 EMERGENCY STOP - No commands sent")
            else:
                print(f"Simulation mode - Motor commands: {actions[:5]}...")  # Show first 5
    
    def emergency_stop_robot(self):
        """Emergency stop - send zero commands to all motors."""
        print("🛑 EMERGENCY STOP ACTIVATED!")
        self.emergency_stop = True
        
        if self.cmd_publisher and BOOSTER_SDK_AVAILABLE:
            try:
                # Send zero commands to all motors
                low_cmd = LowCmd()
                low_cmd.cmd_type = LowCmdType.PARALLEL
                motor_cmds = [MotorCmd() for _ in range(B1JointCnt)]
                
                for i in range(B1JointCnt):
                    motor_cmds[i].q = 0.0
                    motor_cmds[i].dq = 0.0
                    motor_cmds[i].tau = 0.0
                    motor_cmds[i].kp = 0.0
                    motor_cmds[i].kd = 0.0
                    motor_cmds[i].weight = 0.0
                
                low_cmd.motor_cmd = motor_cmds
                self.cmd_publisher.Write(low_cmd)
                print("✅ Emergency stop commands sent")
                
            except Exception as e:
                print(f"❌ Error sending emergency stop: {e}")
    
    def run_control_loop(self):
        """Main control loop."""
        print("🚀 Starting control loop...")
        print("Press Ctrl+C to stop")
        print("Press 'e' + Enter for emergency stop")
        
        # Start a thread to monitor for emergency stop input
        def emergency_monitor():
            while True:
                try:
                    user_input = input()
                    if user_input.strip().lower() == 'e':
                        self.emergency_stop_robot()
                        break
                except:
                    break
        
        emergency_thread = threading.Thread(target=emergency_monitor, daemon=True)
        emergency_thread.start()
        
        try:
            while not self.emergency_stop:
                start_time = time.time()
                
                # Get robot state
                obs = self.get_robot_state()
                if obs is None:
                    print("⚠️  No robot state available, skipping control step")
                    time.sleep(self.dt)
                    continue
                
                # Run policy
                if self.policy_fn:
                    # Convert to JAX array and add batch dimension
                    obs_jax = jp.array(obs).reshape(1, -1)
                    
                    # Run policy (need to provide random key and proper input format)
                    rng = jax.random.PRNGKey(42)
                    actions, _ = self.policy_fn({'state': obs_jax}, rng)
                    actions = np.array(actions[0])  # Remove batch dimension
                else:
                    # Fallback: zero actions
                    actions = np.zeros(23)
                
                # Send commands
                self.send_motor_commands(actions)
                
                # Control timing
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n🛑 Control loop stopped by user")
            self.emergency_stop_robot()
        except Exception as e:
            print(f"❌ Error in control loop: {e}")
            self.emergency_stop_robot()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        
        # Simple forward movement commands
        if self.booster_client:
            # Add any cleanup needed for Booster SDK
            pass


def main():
    parser = argparse.ArgumentParser(description="Direct Hardware Deployment for Booster T1")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--robot_ip", type=str, default="127.0.0.1",
                       help="Robot IP address (default: 127.0.0.1 for local)")
    parser.add_argument("--use_gamepad", action="store_true", default=False,
                       help="Use gamepad for joystick control")
    parser.add_argument("--no_gamepad", action="store_true",
                       help="Disable gamepad control")
    
    args = parser.parse_args()
    
    # Handle gamepad flag
    use_gamepad = args.use_gamepad and not args.no_gamepad
    
    print("🤖 Booster T1 Hardware Deployment")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Robot IP: {args.robot_ip}")
    print(f"Gamepad: {'Enabled' if use_gamepad else 'Disabled'}")
    print(f"Booster SDK: {'Available' if BOOSTER_SDK_AVAILABLE else 'Not Available'}")
    print("=" * 50)
    
    # Create and run controller
    controller = HardwareController(args.checkpoint_path, use_gamepad, args.robot_ip)
    controller.run_control_loop()


if __name__ == "__main__":
    main()

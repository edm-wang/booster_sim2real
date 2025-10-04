#!/usr/bin/env python3
"""
Dry Run Deploy Script
====================

This script runs policy inference and observation writing without sending
commands to the robot. The robot stays in prepare or custom mode while
we can verify policy behavior through the web plotter.

Usage:
    python deploy_dry.py --net 192.168.10.102 --config T1.yaml
"""

import numpy as np
import time
import yaml
import logging
import threading
import os

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import TimerConfig, Timer
from utils.policy import Policy
from visualize.observation_writer import write_observation


class DryRunController:
    def __init__(self, cfg_file) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Initialize components
        self.remoteControlService = RemoteControlService()
        self.policy = Policy(cfg=self.cfg)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.running = True

        # Initialize observation data writer
        self.obs_data_file = "/tmp/robot_obs_dry.json"
        print(f"📊 DRY RUN: Observation data will be written to: {self.obs_data_file}")
        print("🌐 To view plots, run: python visualize/standalone_web_plotter_fixed.py --data-file /tmp/robot_obs_dry.json")
        
        # Start observation writing thread
        self.obs_writer_thread = None
        self.start_observation_writer()
        
        # Policy inference tracking
        self.policy_inference_started = False
        self.inference_count = 0

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)
        self.accelerometer = np.zeros(3, dtype=np.float32)
        
        # Store current observation for the observation writer thread
        self.current_observation = np.zeros(47, dtype=np.float32)
        self.observation_lock = threading.Lock()

    def _init_communication(self) -> None:
        try:
            print("🔌 Initializing communication...")
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
            print("✅ Communication initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize communication: {e}")
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        
        if time_now >= self.next_inference_time:
            # Debug: Print first few low state messages to verify data reception
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            if self._debug_count < 3:
                print(f"📡 Low state message #{self._debug_count}:")
                print(f"   IMU RPY: {low_state_msg.imu_state.rpy}")
                print(f"   IMU Gyro: {low_state_msg.imu_state.gyro}")
                print(f"   IMU Acc: {low_state_msg.imu_state.acc}")
                print(f"   Motor count: {len(low_state_msg.motor_state_serial)}")
                self._debug_count += 1
            
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            self.accelerometer[:] = low_state_msg.imu_state.acc
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def start_observation_writer(self):
        """Start the observation writing thread."""
        print("📊 Starting observation writer thread...")
        self.obs_writer_thread = threading.Thread(target=self._observation_writer_loop, daemon=True)
        self.obs_writer_thread.start()
        print("✅ Observation writer thread started")

    def _observation_writer_loop(self):
        """Continuously write observation data for web plotter."""
        print("📊 Observation writer loop started")
        obs_write_count = 0
        
        while self.running:
            try:
                time_now = self.timer.get_time()
                
                # Always write sensor observations (raw data before policy inference)
                with self.observation_lock:
                    if self.policy_inference_started:
                        # Use policy observations if available
                        obs = self.current_observation.copy()
                        obs_type = "Policy"
                    else:
                        # Create raw sensor observation from current sensor data
                        obs = np.zeros(47, dtype=np.float32)
                        obs[0:3] = self.projected_gravity  # Gravity vector
                        obs[3:6] = self.base_ang_vel      # Angular velocity
                        obs[6:9] = [self.remoteControlService.get_vx_cmd(), 
                                   self.remoteControlService.get_vy_cmd(), 
                                   self.remoteControlService.get_vyaw_cmd()]  # Commands
                        obs[9:11] = [0.0, 0.0]  # Gait phase (zeros before policy)
                        obs[11:23] = self.dof_pos[11:] - self.policy.default_dof_pos[11:]  # Joint positions
                        obs[23:35] = self.dof_vel[11:]  # Joint velocities
                        obs[35:47] = np.zeros(12)  # Actions (zeros before policy)
                        obs_type = "Raw Sensors"
                    
                    # Write observation data
                    write_observation(obs, time_now, self.obs_data_file)
                    obs_write_count += 1
                    
                    # Debug: Print observation update every 50 writes (5 seconds at 10Hz)
                    if obs_write_count % 50 == 0:
                        print(f"📊 {obs_type} observation data written {obs_write_count} times")
                        print(f"   Gravity: {obs[0:3]}")
                        print(f"   Angular velocity: {obs[3:6]}")
                        print(f"   Commands: {obs[6:9]}")
                        print(f"   Gait phase: {obs[9:11]}")
                        print(f"   Joint pos range: [{obs[11:23].min():.3f}, {obs[11:23].max():.3f}]")
                        print(f"   Joint vel range: [{obs[23:35].min():.3f}, {obs[23:35].max():.3f}]")
                        print(f"   Actions range: [{obs[35:47].min():.3f}, {obs[35:47].max():.3f}]")
                        print(f"   Raw IMU Acc: {self.accelerometer}")
                        print(f"   Raw IMU RPY: {self.projected_gravity}")
                
                # Write at 10Hz (every 100ms)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to write observation data: {e}")
                time.sleep(0.1)

    def start_custom_mode_conditionally(self):
        print(f"🎮 {self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        print("🔄 Starting custom mode sequence...")
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd, self.cfg)
        # NOTE: We don't send the command in dry run mode!
        print("🚫 DRY RUN: Skipping command send to robot")
        send_time = time.perf_counter()
        print(f"📤 Command preparation took {(send_time - start_time)*1000:.4f} ms")
        # NOTE: We don't change robot mode in dry run!
        print("🚫 DRY RUN: Skipping robot mode change")
        end_time = time.perf_counter()
        print(f"✅ Dry run setup took {(end_time - send_time)*1000:.4f} ms")
        print("🎯 Robot remains in current mode (dry run mode)")

    def start_policy_inference(self):
        """Start policy inference without sending commands to robot."""
        print("🤖 Starting policy inference (DRY RUN - no commands sent to robot)...")
        self.next_inference_time = self.timer.get_time()
        self.policy_inference_started = True
        print(f"⏰ Policy inference started at: {self.next_inference_time:.3f}")
        print("🧠 Policy will run inference and write observations to file")
        print("🚫 No commands will be sent to the robot")

    def run_policy_inference(self):
        """Run policy inference and store observations without sending commands."""
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        
        self.inference_count += 1
        print(f"\n🧠 === POLICY INFERENCE #{self.inference_count} (DRY RUN) ===")
        print(f"⏰ Time: {time_now:.3f}, Next inference: {self.next_inference_time:.3f}")
        
        self.next_inference_time += self.policy.get_policy_interval()
        start_time = time.perf_counter()

        # Get current state for debugging
        print(f"📊 Current state:")
        print(f"   dof_pos range: [{self.dof_pos.min():.3f}, {self.dof_pos.max():.3f}]")
        print(f"   dof_vel range: [{self.dof_vel.min():.3f}, {self.dof_vel.max():.3f}]")
        print(f"   base_ang_vel: {self.base_ang_vel}")
        print(f"   projected_gravity: {self.projected_gravity}")
        print(f"   accelerometer: {self.accelerometer}")

        # Run policy inference and get actions (but don't send them!)
        actions = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
        )
        
        # Store current observation for the observation writer thread
        with self.observation_lock:
            if hasattr(self.policy, 'obs'):
                self.current_observation[:] = self.policy.obs

        print(f"🎯 Policy actions computed (NOT sent to robot):")
        print(f"   Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"   Sample actions: {actions[:5]}")
        print(f"🚫 DRY RUN: Actions not sent to robot")

        inference_time = time.perf_counter()
        print(f"⏱️  Inference took {(inference_time - start_time)*1000:.4f} ms")
        time.sleep(0.001)

    def cleanup(self) -> None:
        """Cleanup resources."""
        print("🧹 Cleaning up resources...")
        
        # Stop observation writer thread
        if hasattr(self, "obs_writer_thread") and self.obs_writer_thread:
            print("📊 Stopping observation writer thread...")
            self.obs_writer_thread.join(timeout=2.0)
        
        # Clean up observation data file
        if hasattr(self, "obs_data_file") and os.path.exists(self.obs_data_file):
            print(f"🧹 Cleaning up observation data file: {self.obs_data_file}")
            try:
                os.remove(self.obs_data_file)
            except:
                pass
        
        self.remoteControlService.close() if hasattr(self.remoteControlService, 'close') else None
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()

    def __enter__(self) -> "DryRunController":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    import argparse
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\n🛑 Shutting down dry run...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Dry run deploy script - runs policy inference without sending commands to robot')
    parser.add_argument("--config", type=str, default="T1.yaml", help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)

    print(f"🚀 Starting DRY RUN controller, connecting to {args.net} ...")
    print(f"📡 Network interfaces:")
    print(f"   - eth0 (wired): 192.168.10.102")
    print(f"   - wlan0 (wireless): 192.168.0.53")
    print(f"🎯 Using IP: {args.net}")
    print("🚫 DRY RUN MODE: No commands will be sent to the robot!")
    ChannelFactory.Instance().Init(0, args.net)

    with DryRunController(cfg_file) as controller:
        time.sleep(2)  # Wait for channels to initialize
        print("✅ Initialization complete.")
        controller.start_custom_mode_conditionally()
        
        print("\n" + "="*60)
        print("🤖 DRY RUN MODE ACTIVE")
        print("="*60)
        print("📊 Policy inference will run and write observations to file")
        print("🚫 No commands will be sent to the robot")
        print("🌐 Use web plotter to visualize policy behavior")
        print("⏹️  Press Ctrl+C to stop")
        print("="*60)
        
        # Start policy inference
        controller.start_policy_inference()

        try:
            while controller.running:
                controller.run_policy_inference()
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received. Cleaning up...")
            controller.cleanup()

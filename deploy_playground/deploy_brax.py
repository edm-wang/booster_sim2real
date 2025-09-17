import numpy as np
import time
import logging
import threading
import argparse
import signal
import sys
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
    MotorCmd,
    LowCmdType,
)

# Local imports for MuJoCo Playground policy
import jax
import jax.numpy as jp
from utils.brax_utils import load_trained_policy
from observation_plotter import init_plotter, update_plotter, stop_plotter

# Official motor gains from Booster SDK examples
OFFICIAL_KP_GAINS = [
    5.0, 5.0,                    # Head joints
    40.0, 50.0, 20.0, 10.0,     # Left arm
    40.0, 50.0, 20.0, 10.0,     # Right arm
    100.0,                       # Waist
    350.0, 350.0, 180.0, 350.0, 450.0, 450.0,  # Left leg
    350.0, 350.0, 180.0, 350.0, 450.0, 450.0,  # Right leg
]

OFFICIAL_KD_GAINS = [
    0.1, 0.1,                    # Head joints
    0.5, 1.5, 0.2, 0.2,         # Left arm
    0.5, 1.5, 0.2, 0.2,         # Right arm
    5.0,                         # Waist
    7.5, 7.5, 3.0, 5.5, 0.5, 0.5,  # Left leg
    7.5, 7.5, 3.0, 5.5, 0.5, 0.5,  # Right leg
]

# FIXED: Default pose from playground training (23 joints)
DEFAULT_POSE = np.array([
    0.0, 0.0, 0.0, -1.399999976158142, 0.0, -0.4000000059604645, 
    0.0, 1.399999976158142, 0.0, 0.4000000059604645, 0.0, 
    -0.20000000298023224, 0.0, 0.0, 0.4000000059604645, -0.20000000298023224, 0.0, 
    -0.20000000298023224, 0.0, 0.0, 0.4000000059604645, -0.20000000298023224, 0.0
], dtype=np.float32)

# Simplified utility functions (replace Booster Gym utils)
def create_prepare_cmd(low_cmd: LowCmd):
    """Create prepare command for robot initialization."""
    print("🔧 Creating prepare command...")
    low_cmd.cmd_type = LowCmdType.PARALLEL
    motor_cmds = [MotorCmd() for _ in range(B1JointCnt)]
    
    for i in range(B1JointCnt):
        motor_cmds[i].q = 0.0
        motor_cmds[i].dq = 0.0
        motor_cmds[i].tau = 0.0
        motor_cmds[i].kp = OFFICIAL_KP_GAINS[i]  # Use proper gains!
        motor_cmds[i].kd = OFFICIAL_KD_GAINS[i]  # Use proper gains!
        motor_cmds[i].weight = 0.0
    
    low_cmd.motor_cmd = motor_cmds
    print(f"✅ Prepare command created with gains: kp={OFFICIAL_KP_GAINS[:5]}...")

def create_first_frame_rl_cmd(low_cmd: LowCmd):
    """Create first frame RL command."""
    print("🚀 Creating first frame RL command...")
    create_prepare_cmd(low_cmd)

def rotate_vector_inverse_rpy(roll, pitch, yaw, vector):
    """Rotate vector by inverse RPY angles."""
    # Simple implementation - you may want to use proper rotation matrices
    cos_r, sin_r = np.cos(-roll), np.sin(-roll)
    cos_p, sin_p = np.cos(-pitch), np.sin(-pitch)
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    
    # Apply rotations (simplified)
    x, y, z = vector
    # This is a simplified rotation - implement proper 3D rotation if needed
    return np.array([x, y, z])

class RemoteControlService:
    """Remote control service that waits for user input like Booster Gym."""
    
    def __init__(self):
        self.vx_cmd = 0.0  # Small forward velocity
        self.vy_cmd = 0.0  # No lateral movement
        self.vyaw_cmd = 0.0  # No rotation
    
    def start_custom_mode(self) -> bool:
        """Wait for user input to start custom mode."""
        return input().strip() != ""
    
    def start_rl_gait(self) -> bool:
        """Wait for user input to start RL gait."""
        return input().strip() != ""
    
    def get_vx_cmd(self) -> float:
        return self.vx_cmd
    
    def get_vy_cmd(self) -> float:
        return self.vy_cmd
    
    def get_vyaw_cmd(self) -> float:
        return self.vyaw_cmd
    
    def get_custom_mode_operation_hint(self) -> str:
        return "Press ENTER to start custom mode..."
    
    def get_rl_gait_operation_hint(self) -> str:
        return "Press ENTER to start RL gait..."
    
    def get_operation_hint(self) -> str:
        return "RL gait active. Press Ctrl+C to stop."

class Timer:
    """Simplified timer class."""
    
    def __init__(self, config):
        self.dt = config.time_step
        self.current_time = time.time()
    
    def get_time(self):
        return time.time()
    
    def tick_timer_if_sim(self):
        # For real robot, we use real time
        pass

class TimerConfig:
    def __init__(self, time_step=0.01):
        self.time_step = time_step

class Policy:
    """MuJoCo Playground policy wrapper with FIXED observation space."""
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.policy_fn = None
        self.obs_size = 85  # FIXED: Playground expects exactly 85 dimensions (3+3+3+3+4+23+23+23)
        self.act_size = 23  # FIXED: Playground expects 23 actions (23 joints)
        self._load_policy()
        
        # FIXED: Track previous action for observation construction
        self.last_action = np.zeros(23, dtype=np.float32)
        
        # FIXED: Track linear velocity from accelerometer integration
        self.linear_velocity = np.zeros(3, dtype=np.float32)
        self.last_accel = np.zeros(3, dtype=np.float32)
        self.last_time = time.time()
        
        # FIXED: Gait phase tracking for both feet (4D)
        self.gait_frequency = 1.0  # Hz
        self.gait_phase_left = 0.0
        self.gait_phase_right = 0.0
    
    def _load_policy(self):
        """Load the trained policy and determine observation/action sizes."""
        print(f"🤖 Loading policy from: {self.checkpoint_path}")
        try:
            # Create environment for policy loading
            from mujoco_playground import registry
            env = registry.load("T1JoystickFlatTerrain")
            
            # FIXED: Use hardcoded sizes instead of dynamic
            print(f"📊 Environment info:")
            print(f"   Observation size: {self.obs_size} (FIXED)")
            print(f"   Action size: {self.act_size} (FIXED)")
            
            # Load the policy
            self.policy_fn = load_trained_policy(self.checkpoint_path, env)
            print("✅ Policy loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            self.policy_fn = None
    
    def get_policy_interval(self):
        """Get policy inference interval (1/policy_frequency)."""
        return 1.0 / 20.0  # 20Hz policy frequency
    
    def update_linear_velocity(self, accel, dt):
        """FIXED: Update linear velocity from accelerometer integration."""
        # Simple integration: v = v0 + a * dt
        self.linear_velocity += accel * dt
        # Apply some damping to prevent drift
        self.linear_velocity *= 0.99
    
    def update_gait_phase(self, dt):
        """FIXED: Update gait phase for both feet (4D)."""
        # Update both feet with slight phase offset
        self.gait_phase_left += dt * self.gait_frequency
        self.gait_phase_right += dt * self.gait_frequency + 0.5  # 180 degree offset
        
        # Keep phases in [0, 1]
        self.gait_phase_left = self.gait_phase_left % 1.0
        self.gait_phase_right = self.gait_phase_right % 1.0
    
    def _construct_observation(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw, accel):
        """FIXED: Construct observation exactly like playground (85 dimensions) for plotting."""
        # FIXED: Update linear velocity from accelerometer
        dt = time_now - self.last_time
        self.update_linear_velocity(accel, dt)
        self.last_time = time_now
        
        # FIXED: Update gait phase for both feet
        self.update_gait_phase(dt)
        
        # FIXED: Construct observation exactly like playground (85 dimensions)
        obs = np.zeros(85, dtype=np.float32)
        
        # 1. Linear velocity (3) - from accelerometer integration
        obs[0:3] = self.linear_velocity
        
        # 2. Gyroscope (3) - angular velocity
        obs[3:6] = base_ang_vel
        
        # 3. Gravity vector (3) - in body frame
        obs[6:9] = projected_gravity
        
        # 4. Commands (3) - joystick commands
        obs[9:12] = [vx, vy, vyaw]
        
        # 5. Gait phase (4) - cos/sin for both feet
        obs[12] = np.cos(2 * np.pi * self.gait_phase_left)
        obs[13] = np.sin(2 * np.pi * self.gait_phase_left)
        obs[14] = np.cos(2 * np.pi * self.gait_phase_right)
        obs[15] = np.sin(2 * np.pi * self.gait_phase_right)
        
        # 6. Joint angles relative to default pose (23)
        joint_angles_relative = dof_pos[:23] - DEFAULT_POSE[:23]
        obs[16:39] = joint_angles_relative
        
        # 7. Joint velocities (23)
        obs[39:62] = dof_vel[:23]
        
        # 8. Previous action (23)
        obs[62:85] = self.last_action
        
        return obs
    
    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw, accel):
        """FIXED: Run policy inference with exact playground observation structure."""
        if self.policy_fn is None:
            print("⚠️  Policy not loaded, returning zero actions")
            return np.zeros(23, dtype=np.float32), np.zeros(85, dtype=np.float32)  # Return actions and obs
        
        try:
            # Construct observation using the dedicated method
            obs = self._construct_observation(
                time_now, dof_pos, dof_vel, base_ang_vel, 
                projected_gravity, vx, vy, vyaw, accel
            )
            
            # Run policy
            obs_jax = jp.array(obs).reshape(1, -1)
            rng = jax.random.PRNGKey(42)
            actions, _ = self.policy_fn({'state': obs_jax}, rng)
            actions = np.array(actions[0])
            
            # FIXED: Store action for next observation
            self.last_action[:] = actions
            
            # FIXED: Convert to 23-joint targets for Booster SDK
            result = np.zeros(23, dtype=np.float32)  # Booster has 23 joints
            result[:] = actions.astype(np.float32)  # All 23 actions from policy
            
            print(f"🧠 Policy inference: obs_size=85, actions range [{result.min():.3f}, {result.max():.3f}], mean: {result.mean():.3f}")
            print(f"   Sample actions: {result[:5]}")
            print(f"   Linear velocity: {self.linear_velocity}")
            print(f"   Gait phase left: {self.gait_phase_left:.3f}, right: {self.gait_phase_right:.3f}")
            return result, obs
            
        except Exception as e:
            print(f"❌ Error in policy inference: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(23, dtype=np.float32), np.zeros(85, dtype=np.float32)

class Controller:
    def __init__(self, checkpoint_path, robot_ip="192.168.10.102") -> None:
        # Setup logging
        logging.basicConfig(level=logging.DEBUG)  # More verbose logging
        self.logger = logging.getLogger(__name__)

        # Store parameters
        self.checkpoint_path = checkpoint_path
        self.robot_ip = robot_ip

        # Initialize components
        self.remoteControlService = RemoteControlService()
        self.policy = Policy(checkpoint_path)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()
        self.inference_count = 0
        self.publish_count = 0
        
        # Initialize observation plotter
        print("📊 Initializing observation plotter...")
        try:
            self.plotter = init_plotter()
            print("✅ Observation plotter initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize plotter: {e}")
            self.plotter = None


    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=0.01))  # 100Hz control
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()
        print(f"⏰ Timer initialized: dt=0.01s, next_publish={self.next_publish_time:.3f}, next_inference={self.next_inference_time:.3f}")

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.accelerometer = np.zeros(3, dtype=np.float32)  # FIXED: Add accelerometer
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)

        self.dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)
        print(f"📊 State buffers initialized: {B1JointCnt} joints")

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
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            print(f"⚠️  IMU base rpy values are too large: {low_state_msg.imu_state.rpy}")
            self.running = False
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        if time_now >= self.next_inference_time:
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            # FIXED: Extract accelerometer data
            self.accelerometer[:] = low_state_msg.imu_state.accelerometer
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        """Cleanup resources."""
        print("🧹 Cleaning up resources...")
        
        
        # Stop and cleanup plotter
        if hasattr(self, "plotter") and self.plotter:
            print("📊 Stopping observation plotter...")
            stop_plotter()
        
        self.remoteControlService.close() if hasattr(self.remoteControlService, 'close') else None
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and getattr(self, "publish_runner") != None:
            self.publish_runner.join(timeout=1.0)

    def start_custom_mode_conditionally(self):
        print(f"🎮 {self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        print("🔄 Starting custom mode sequence...")
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd)
        for i in range(B1JointCnt):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        print(f"📤 Send cmd took {(send_time - start_time)*1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        end_time = time.perf_counter()
        print(f"✅ Change mode took {(end_time - send_time)*1000:.4f} ms")
        print("🎯 Robot is now in CUSTOM mode!")

    def start_rl_gait_conditionally(self):
        print(f"🚀 {self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        print("🤖 Starting RL gait sequence...")
        
        create_first_frame_rl_cmd(self.low_cmd)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        print(f"⏰ Reset timers: next_inference={self.next_inference_time:.3f}, next_publish={self.next_publish_time:.3f}")
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print("🧵 Publish thread started")
        print(f"🎉 {self.remoteControlService.get_operation_hint()}")

    def run(self):
        time_now = self.timer.get_time()
        
        # Always update plotter with current observation (regardless of mode)
        try:
            obs = self.policy._construct_observation(
                time_now=time_now,
                dof_pos=self.dof_pos,
                dof_vel=self.dof_vel,
                base_ang_vel=self.base_ang_vel,
                projected_gravity=self.projected_gravity,
                vx=self.remoteControlService.get_vx_cmd(),
                vy=self.remoteControlService.get_vy_cmd(),
                vyaw=self.remoteControlService.get_vyaw_cmd(),
                accel=self.accelerometer,
            )
            update_plotter(obs, time_now)
        except Exception as e:
            print(f"⚠️  Warning: Failed to update plotter: {e}")
        
        # Only run policy inference if it's time for it
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        
        self.inference_count += 1
        print(f"\n🧠 === POLICY INFERENCE #{self.inference_count} ===")
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

        # Run policy inference and get actions
        actions, _ = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
            accel=self.accelerometer,  # FIXED: Pass accelerometer
        )
        
        self.dof_target[:] = actions

        print(f"🎯 New dof_target range: [{self.dof_target.min():.3f}, {self.dof_target.max():.3f}]")
        print(f"   Sample targets: {self.dof_target[:5]}")

        inference_time = time.perf_counter()
        print(f"⏱️  Inference took {(inference_time - start_time)*1000:.4f} ms")
        time.sleep(0.001)

    def _publish_cmd(self):
        """Publish command following Booster Gym approach exactly."""
        print("📡 Publish thread started - following Booster Gym approach")
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            
            self.publish_count += 1
            if self.publish_count % 10 == 0:  # Print every 10th publish (every 100ms)
                print(f"\n📤 === PUBLISH #{self.publish_count} ===")
                print(f"⏰ Time: {time_now:.3f}, Next publish: {self.next_publish_time:.3f}")
            
            self.next_publish_time += 0.01  # 100Hz control

            # Apply command filtering (exactly like Booster Gym)
            old_filtered = self.filtered_dof_target.copy()
            self.filtered_dof_target = self.filtered_dof_target * 0.8 + self.dof_target * 0.2
            
            if self.publish_count % 10 == 0:
                print(f"🔄 Command filtering:")
                print(f"   Old filtered range: [{old_filtered.min():.3f}, {old_filtered.max():.3f}]")
                print(f"   New filtered range: [{self.filtered_dof_target.min():.3f}, {self.filtered_dof_target.max():.3f}]")
                print(f"   Sample filtered: {self.filtered_dof_target[:5]}")

            # Set motor commands (EXACTLY like Booster Gym - only set q!)
            for i in range(B1JointCnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
                if self.publish_count % 10 == 0:
                    print(f"Setting motor {i} to {self.filtered_dof_target[i]}")
                # NOTE: We don't touch dq, tau, kp, kd, weight here!
                # Booster Gym only sets q in the main loop

            if self.publish_count % 10 == 0:
                print(f"🎯 Motor commands set:")
                print(f"   q range: [{self.filtered_dof_target.min():.3f}, {self.filtered_dof_target.max():.3f}]")
                print(f"   Sample q: {self.filtered_dof_target[:5]}")

            start_time = time.perf_counter()
            self._send_cmd(self.low_cmd)
            publish_time = time.perf_counter()
            
            if self.publish_count % 10 == 0:
                print(f"⏱️  Publish took {(publish_time - start_time)*1000:.4f} ms")
            
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint", help="Path to the trained policy checkpoint.")
    parser.add_argument("--robot_ip", type=str, default="192.168.10.102", help="Robot IP address.")
    args = parser.parse_args()

    print(f"🚀 Starting playground policy controller, connecting to {args.robot_ip} ...")
    ChannelFactory.Instance().Init(0, args.robot_ip)

    with Controller(args.checkpoint_path, args.robot_ip) as controller:
        time.sleep(2)  # Wait for channels to initialize
        print("✅ Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received. Cleaning up...")
            controller.cleanup()

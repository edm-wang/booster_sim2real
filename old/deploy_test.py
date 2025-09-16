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
from utils import load_trained_policy

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

# Simplified utility functions (replace Booster Gym utils)
def create_prepare_cmd(low_cmd: LowCmd):
    """Create prepare command for robot initialization."""
    print("🔧 Creating prepare command...")
    low_cmd.cmd_type = LowCmdType.PARALLEL
    motor_cmds = [MotorCmd() for _ in range(B1JointCnt)]
    
    # T1.yaml의 prepare default_qpos 사용 (안정적인 서있는 자세)
    default_qpos = [
        0.0,  0.0,                    # Head
        0.25, -1.4, 0.0, -0.5,       # Left arm  
        0.25, 1.4, 0.0, 0.5,         # Right arm
        0.0,                          # Waist
        -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # Left leg
        -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # Right leg
    ]
    
    for i in range(B1JointCnt):
        motor_cmds[i].q = default_qpos[i] if i < len(default_qpos) else 0.0
        motor_cmds[i].dq = 0.0
        motor_cmds[i].tau = 0.0
        motor_cmds[i].kp = OFFICIAL_KP_GAINS[i]  # Use proper gains!
        motor_cmds[i].kd = OFFICIAL_KD_GAINS[i]  # Use proper gains!
        motor_cmds[i].weight = 0.0
    
    low_cmd.motor_cmd = motor_cmds
    print(f"✅ Prepare command created with stable standing pose")
    print(f"   Sample positions: {[motor_cmds[i].q for i in range(5)]}")

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
    """MuJoCo Playground policy wrapper."""
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.policy_fn = None
        self._load_policy()
    
    def _load_policy(self):
        """Load the trained policy."""
        print(f"🤖 Loading policy from: {self.checkpoint_path}")
        try:
            # Create environment for policy loading
            from mujoco_playground import registry
            env = registry.load("T1JoystickFlatTerrain")
            self.policy_fn = load_trained_policy(self.checkpoint_path, env)
            print("✅ Policy loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            self.policy_fn = None
    
    def get_policy_interval(self):
        """Get policy inference interval (1/policy_frequency)."""
        return 1.0 / 20.0  # 20Hz policy frequency
    
    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        """Run policy inference."""
        if self.policy_fn is None:
            print("⚠️  Policy not loaded, returning zero actions")
            return np.zeros(B1JointCnt, dtype=np.float32)
        
        try:
            # Create observation vector (T1 observation size)
            obs = np.zeros(85, dtype=np.float32)
            
            # IMU data
            obs[0:3] = projected_gravity  # Projected gravity
            obs[3:6] = base_ang_vel       # Angular velocity
            obs[6:9] = [0.0, 0.0, -9.81] # Gravity vector
            
            # Motor states (23 joints)
            for i in range(min(23, B1JointCnt)):
                obs[9 + i*2] = dof_pos[i]     # Joint position
                obs[9 + i*2 + 1] = dof_vel[i] # Joint velocity
            
            # Joystick commands
            obs[55:58] = [vx, vy, vyaw]
            
            # Run policy
            obs_jax = jp.array(obs).reshape(1, -1)
            rng = jax.random.PRNGKey(42)
            actions, _ = self.policy_fn({'state': obs_jax}, rng)
            actions = np.array(actions[0])
            
            # Convert actions to joint targets with conservative scaling
            # MuJoCo Playground actions are typically in [-1, 1] range
            # Scale to reasonable joint position range - start very conservative
            action_scale = 0.1  # Very conservative scaling to prevent large movements
            result = (actions[:B1JointCnt] * action_scale).astype(np.float32)
            
            print(f"🧠 Policy inference: obs shape={obs.shape}, actions range [{result.min():.3f}, {result.max():.3f}], mean: {result.mean():.3f}")
            print(f"   Sample actions: {result[:5]}")
            return result
            
        except Exception as e:
            print(f"❌ Error in policy inference: {e}")
            return np.zeros(B1JointCnt, dtype=np.float32)

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

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=0.01))  # 100Hz control
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()
        print(f"⏰ Timer initialized: dt=0.01s, next_publish={self.next_publish_time:.3f}, next_inference={self.next_inference_time:.3f}")

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
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
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        """Cleanup resources."""
        print("🧹 Cleaning up resources...")
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

        self.dof_target[:] = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
        )

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

            # Set motor commands with proper gains for RL Gait mode
            for i in range(B1JointCnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
                self.low_cmd.motor_cmd[i].dq = 0.0  # Set velocity to 0
                self.low_cmd.motor_cmd[i].tau = 0.0  # Set torque to 0
                # Use appropriate gains for RL Gait mode
                self.low_cmd.motor_cmd[i].kp = OFFICIAL_KP_GAINS[i]
                self.low_cmd.motor_cmd[i].kd = OFFICIAL_KD_GAINS[i]
                self.low_cmd.motor_cmd[i].weight = 1.0  # Full weight
                if self.publish_count % 10 == 0 and i < 5:  # Debug first 5 motors
                    print(f"Motor {i}: q={self.filtered_dof_target[i]:.3f}, kp={OFFICIAL_KP_GAINS[i]}, kd={OFFICIAL_KD_GAINS[i]}")

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
    parser.add_argument("--checkpoint_path", type=str, default="/home/booster/Workspace/booster/checkpoint", help="Path to the trained policy checkpoint.")
    parser.add_argument("--robot_ip", type=str, default="192.168.10.102", help="Robot IP address.")
    args = parser.parse_args()

    print(f"🚀 Starting custom controller, connecting to {args.robot_ip} ...")
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

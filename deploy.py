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

# Simplified utility functions (replace Booster Gym utils)
def create_prepare_cmd(low_cmd: LowCmd):
    """Create prepare command for robot initialization."""
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

def create_first_frame_rl_cmd(low_cmd: LowCmd):
    """Create first frame RL command."""
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
    """Simplified remote control service for joystick commands."""
    
    def __init__(self):
        self.vx_cmd = 0.1  # Small forward velocity
        self.vy_cmd = 0.0  # No lateral movement
        self.vyaw_cmd = 0.0  # No rotation
    
    def start_custom_mode(self) -> bool:
        """Always allow custom mode for now."""
        return True
    
    def start_rl_gait(self) -> bool:
        """Always allow RL gait for now."""
        return True
    
    def get_vx_cmd(self) -> float:
        return self.vx_cmd
    
    def get_vy_cmd(self) -> float:
        return self.vy_cmd
    
    def get_vyaw_cmd(self) -> float:
        return self.vyaw_cmd
    
    def get_custom_mode_operation_hint(self) -> str:
        return "Press any key to start custom mode..."
    
    def get_rl_gait_operation_hint(self) -> str:
        return "Press any key to start RL gait..."
    
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
        # Create environment for policy loading
        from mujoco_playground import registry
        env = registry.load("T1JoystickFlatTerrain")
        """Load the trained policy."""
        print(f"Loading policy from: {self.checkpoint_path}")
        try:
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
            
            # Convert actions to joint targets
            return actions[:B1JointCnt].astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error in policy inference: {e}")
            return np.zeros(B1JointCnt, dtype=np.float32)

class Controller:
    def __init__(self, checkpoint_path, robot_ip="192.168.10.102") -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
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

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=0.01))  # 100Hz control
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)

        self.dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)

    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
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
        self.remoteControlService.close() if hasattr(self.remoteControlService, 'close') else None
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and getattr(self, "publish_runner") != None:
            self.publish_runner.join(timeout=1.0)

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd)
        for i in range(B1JointCnt):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        self.logger.debug(f"Send cmd took {(send_time - start_time)*1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        end_time = time.perf_counter()
        self.logger.debug(f"Change mode took {(end_time - send_time)*1000:.4f} ms")

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        create_first_frame_rl_cmd(self.low_cmd)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()

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

        inference_time = time.perf_counter()
        self.logger.debug(f"Inference took {(inference_time - start_time)*1000:.4f} ms")
        time.sleep(0.001)

    def _publish_cmd(self):
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += 0.01  # 100Hz control
            self.logger.debug(f"Next publish time: {self.next_publish_time}")

            self.filtered_dof_target = self.filtered_dof_target * 0.8 + self.dof_target * 0.2

            for i in range(B1JointCnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]

            # Simplified command sending (no parallel mechanism handling for now)
            for i in range(B1JointCnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.0
                self.low_cmd.motor_cmd[i].weight = 0.0

            start_time = time.perf_counter()
            self._send_cmd(self.low_cmd)
            publish_time = time.perf_counter()
            self.logger.debug(f"Publish took {(publish_time - start_time)*1000:.4f} ms")
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="/home/booster/Workspace/booster/checkpoint", type=str, help="Path to the trained policy checkpoint.")
    parser.add_argument("--robot_ip", type=str, default="192.168.10.102", help="Robot IP address.")
    args = parser.parse_args()

    print(f"Starting custom controller, connecting to {args.robot_ip} ...")
    ChannelFactory.Instance().Init(0, args.robot_ip)

    with Controller(args.checkpoint_path, args.robot_ip) as controller:
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()

import numpy as np
import onnxruntime as rt


class Policy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            # Load ONNX model
            self.policy = rt.InferenceSession(
                self.cfg["policy"]["policy_path"],
                providers=["CPUExecutionProvider"]
            )
            # Get input/output names
            self._input_name = self.policy.get_inputs()[0].name
            self._output_name = self.policy.get_outputs()[0].name
        except Exception as e:
            print(f"Failed to load ONNX policy: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        
        # Phase tracking (2 phases for left and right feet)
        self.phase = np.array([0.0, np.pi], dtype=np.float32)
        self.phase_dt = 2 * np.pi * self.gait_frequency * self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        
        # Observation and action tracking
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.last_action = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        
        self.dof_targets = np.copy(self.default_dof_pos)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        
        # For computing local linear velocity (we'll need base orientation)
        self.base_linvel_global = np.zeros(3, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w, x, y, z

    def _compute_local_linvel(self, base_linvel_global, base_rpy):
        """Convert global linear velocity to local frame."""
        # Simple rotation using RPY angles
        roll, pitch, yaw = base_rpy[0], base_rpy[1], base_rpy[2]
        
        # Rotation matrix for yaw (rotation around z-axis)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R_z = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # For pitch (rotation around y-axis)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        R_y = np.array([
            [cos_pitch, 0, -sin_pitch],
            [0, 1, 0],
            [sin_pitch, 0, cos_pitch]
        ])
        
        # For roll (rotation around x-axis)
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        R_x = np.array([
            [1, 0, 0],
            [0, cos_roll, sin_roll],
            [0, -sin_roll, cos_roll]
        ])
        
        # Combined rotation: R = R_z @ R_y @ R_x
        R = R_z @ R_y @ R_x
        local_linvel = R.T @ base_linvel_global
        return local_linvel.astype(np.float32)

    def _compute_gravity_vector(self, base_rpy):
        """Compute gravity vector in local frame from base orientation."""
        roll, pitch, yaw = base_rpy[0], base_rpy[1], base_rpy[2]
        
        # Gravity in world frame
        gravity_world = np.array([0.0, 0.0, -1.0])
        
        # Rotation matrices (same as above)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R_z = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        R_y = np.array([
            [cos_pitch, 0, -sin_pitch],
            [0, 1, 0],
            [sin_pitch, 0, cos_pitch]
        ])
        
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        R_x = np.array([
            [1, 0, 0],
            [0, cos_roll, sin_roll],
            [0, -sin_roll, cos_roll]
        ])
        
        # Rotate gravity to local frame
        R = R_z @ R_y @ R_x
        gravity_local = R.T @ gravity_world
        return gravity_local.astype(np.float32)

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw, base_linvel=None, base_rpy=None):
        """
        Perform policy inference.
        
        Args:
            time_now: Current time
            dof_pos: Joint positions (all joints)
            dof_vel: Joint velocities (all joints)
            base_ang_vel: Base angular velocity (gyro) [wx, wy, wz]
            projected_gravity: Gravity vector in local frame [gx, gy, gz]
            vx: Desired forward velocity
            vy: Desired lateral velocity
            vyaw: Desired yaw angular velocity
            base_linvel: Base linear velocity in global frame (optional, for local_linvel computation)
            base_rpy: Base roll-pitch-yaw angles (optional, for local_linvel computation)
        """
        # Update gait process
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        
        # Update commands
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        
        # Smooth commands
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        # Update gait frequency based on command magnitude
        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = 0.0
            # When stopped, set phase to pi (stance phase)
            self.phase = np.ones(2) * np.pi
        else:
            self.gait_frequency = self.cfg["policy"]["gait_frequency"]
            # Update phase
            phase_tp1 = self.phase + self.phase_dt
            self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

        # Compute local linear velocity
        if base_linvel is not None and base_rpy is not None:
            local_linvel = self._compute_local_linvel(base_linvel, base_rpy)
        else:
            # Fallback: use zero or estimate from dof_vel if available
            local_linvel = np.zeros(3, dtype=np.float32)
        
        # Construct observation (matching play_t1_joystick.py format exactly)
        # The ONNX model expects 82D observation: 3+3+3+23+23+23+4 = 82 (removed local linvel)

        # 1. Gyroscope / angular velocity (3D) - indices 0:3
        self.obs[0:3] = base_ang_vel

        # 2. Gravity vector (3D) - indices 3:6
        self.obs[3:6] = projected_gravity

        # 3. Command (3D) - indices 6:9
        command = self.smoothed_commands.copy()
        if np.linalg.norm(command) < 0.01:
            command = np.zeros(3)
        self.obs[6:9] = command

        # 4. Joint angles relative to default (23D) - indices 9:32
        # Use ALL 23 joints (matching play_t1_joystick.py which uses qpos[7:])
        # Zero the first 2 joints (head joints: AAHead_yaw, Head_pitch)
        joint_angles_rel = (dof_pos - self.default_dof_pos).astype(np.float32)
        joint_angles_rel[:2] = 0.0  # Zero head joints (indices 0-1)
        self.obs[9:32] = joint_angles_rel

        # 5. Joint velocities (23D) - indices 32:55
        # Use ALL 23 joint velocities (matching play_t1_joystick.py which uses qvel[6:])
        # Zero the first 2 joint velocities (head joints)
        joint_vel = dof_vel.astype(np.float32)
        joint_vel[:2] = 0.0  # Zero head joint velocities (indices 0-1)
        self.obs[32:55] = joint_vel

        # 6. Last action (23D) - indices 55:78
        # The model outputs 23D actions (all joints)
        self.obs[55:78] = self.last_action

        # 7. Phase (4D) - indices 78:82
        # Use cos/sin for both left and right phases (matching play_t1_joystick.py)
        ph = self.phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
        phase_encoding = np.concatenate([np.cos(ph), np.sin(ph)]).astype(np.float32)
        self.obs[78:82] = phase_encoding

        # Run ONNX inference
        onnx_input = {self._input_name: self.obs.reshape(1, -1).astype(np.float32)}
        onnx_output = self.policy.run([self._output_name], onnx_input)
        self.actions[:] = onnx_output[0][0]
        
        # Clip actions
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        
        # Update last action for next step
        self.last_action[:] = self.actions.copy()
        
        # Compute DOF targets: default + scaled action
        # The model outputs 23D actions (all joints), so apply to all joints
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets

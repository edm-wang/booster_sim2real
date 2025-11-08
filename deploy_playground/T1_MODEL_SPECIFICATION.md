# T1 Robot Model Input/Output Specification

Based on analysis of `mujoco_playground/_src/locomotion/t1/joystick.py` and `base.py`.

## Observation Structure (Input: 85 dimensions)

The observation vector `obs` is constructed in `_get_obs()` method (lines 461-470) as:

```python
state = jp.hstack([
    noisy_linvel,      # 3 dims
    noisy_gyro,       # 3 dims
    noisy_gravity,     # 3 dims
    info["command"],  # 3 dims
    noisy_joint_angles - self._default_pose,  # 23 dims
    noisy_joint_vel,   # 23 dims
    info["last_act"], # 23 dims
    phase,            # 4 dims
])
```

### Detailed Breakdown:

1. **Noisy Linear Velocity** (3 dims, indices 0-2)
   - Local frame linear velocity: `[vx, vy, vz]`
   - Source: `get_local_linvel(data)` (line 452)
   - Noise: Uniform noise with scale `noise_config.scales.linvel = 0.1` (line 47)
   - Units: m/s

2. **Noisy Gyroscope** (3 dims, indices 3-5)
   - Angular velocity from gyroscope: `[wx, wy, wz]`
   - Source: `get_gyro(data)` (line 412)
   - Noise: Uniform noise with scale `noise_config.scales.gyro = 0.2` (line 48)
   - Units: rad/s

3. **Noisy Gravity Vector** (3 dims, indices 6-8)
   - Projected gravity vector in local frame: `[gx, gy, gz]`
   - Source: `data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])` (line 421)
   - Noise: Uniform noise with scale `noise_config.scales.gravity = 0.05` (line 46)
   - Normalized vector (magnitude ≈ 1.0)

4. **Command** (3 dims, indices 9-11)
   - Desired velocity commands: `[lin_vel_x, lin_vel_y, ang_vel_yaw]`
   - Source: `info["command"]` (line 465)
   - Range: 
     - `lin_vel_x`: [-1.0, 1.0] m/s (line 94)
     - `lin_vel_y`: [-0.8, 0.8] m/s (line 95)
     - `ang_vel_yaw`: [-1.0, 1.0] rad/s (line 96)
   - Updated every 500 steps or when command changes (lines 382-390)

5. **Joint Angles Relative to Default** (23 dims, indices 12-34)
   - Joint positions relative to default pose: `joint_angles - self._default_pose`
   - Source: `data.qpos[7:]` (line 430) - excludes free joint (position + orientation)
   - Noise: Uniform noise with scale `noise_config.scales.joint_pos = 0.03` (line 44)
   - Units: radians
   - The default pose is the "home" keyframe from the MuJoCo model (line 124)

6. **Joint Velocities** (23 dims, indices 35-57)
   - Joint velocities: `[dq1, dq2, ..., dq23]`
   - Source: `data.qvel[6:]` (line 439) - excludes free joint velocities
   - Noise: Uniform noise with scale `noise_config.scales.joint_vel = 1.5` (line 45)
   - Units: rad/s

7. **Previous Action** (23 dims, indices 58-80)
   - Last action taken: `info["last_act"]` (line 468)
   - Initialized to zeros (line 268)
   - Updated after each step (line 380)
   - Range: [-1, 1] (tanh-scaled)

8. **Gait Phase** (4 dims, indices 81-84)
   - Phase encoding for both feet: `[cos(phase_left), sin(phase_left), cos(phase_right), sin(phase_right)]`
   - Source: `jp.concatenate([cos, sin])` where `cos = jp.cos(info["phase"])` and `sin = jp.sin(info["phase"])` (lines 448-450)
   - `info["phase"]` has 2 elements (one per foot) (line 250)
   - Phase frequency: U(1.25, 1.75) Hz (line 248)
   - Phase updates: `phase_tp1 = phase + phase_dt` where `phase_dt = 2 * π * dt * freq` (lines 372-373)

### Total: 3 + 3 + 3 + 3 + 23 + 23 + 23 + 4 = **85 dimensions**

## Action Structure (Output: 23 dimensions)

The action vector has 23 dimensions, one for each actuator/joint.

### Details:

- **Size**: 23 (from `self._mjx_model.nu` in `base.py` line 106)
- **Range**: [-1, 1] (tanh activation applied in the policy network)
- **Application**: 
  ```python
  motor_targets = self._default_pose + action * self._config.action_scale
  ```
  where `action_scale = 1.0` (line 38 in `joystick.py`)
- **Units**: Offset from default pose in radians (scaled by action_scale)
- **Interpretation**: Actions are relative offsets from the default/home pose

### Joint Order:

The 23 joints correspond to the T1 robot's actuators, excluding the free joint (base position/orientation). Based on the model structure:
- Head joints (2)
- Left arm joints (4)
- Right arm joints (4)
- Waist joint (1)
- Left leg joints (6)
- Right leg joints (6)

**Total: 2 + 4 + 4 + 1 + 6 + 6 = 23 joints**

## Key Implementation Notes

1. **Noise**: During training, noise is added to observations (lines 413-446). For deployment, you may want to set `noise_config.level = 0.0` or use clean sensor readings.

2. **Normalization**: The policy network uses `running_statistics.normalize` for observation normalization. The ONNX model includes this normalization (mean and std are baked into the model).

3. **Action Scaling**: Actions are already tanh-scaled to [-1, 1] in the network output. The `action_scale` parameter (default 1.0) further scales these actions before adding to the default pose.

4. **Command Updates**: Commands are sampled randomly during training (line 751-771) and updated every 500 steps or when episode resets (lines 382-390). For deployment, you'll provide real joystick/command inputs.

5. **Phase Tracking**: The gait phase is automatically tracked and updated. For deployment, you'll need to maintain phase state or reconstruct it from foot contact sensors.

## Deployment Checklist

When deploying on hardware, ensure:

- [ ] Observation vector is constructed in the exact order shown above (85 dims)
- [ ] All sensor readings are in the correct units (m/s, rad/s, etc.)
- [ ] Joint angles are relative to the default/home pose
- [ ] Previous action is tracked and included
- [ ] Gait phase is tracked (or reconstructed from contact sensors)
- [ ] Commands are provided in the correct range
- [ ] Actions are applied as: `motor_targets = default_pose + actions * action_scale`
- [ ] Noise is either disabled or matches training noise levels


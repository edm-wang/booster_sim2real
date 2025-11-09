# Deploy ONNX Policy for T1 Robot

This directory contains the deployment code for running an ONNX policy on the T1 robot hardware.

## Changes Made

### 1. `utils/policy.py`
- **Replaced PyTorch JIT loading with ONNX Runtime**
- **Updated observation construction** to match `play_t1_joystick.py` format (50 dimensions):
  - Local linear velocity (3D)
  - Gyroscope/angular velocity (3D)
  - Gravity vector (3D)
  - Velocity commands (3D)
  - Joint angles relative to default (12D, lower body only)
  - Joint velocities (12D, lower body only)
  - Last action (12D)
  - Gait phase encoding (2D: cos, sin)
- **Added phase tracking** for gait synchronization
- **Added last_action tracking** for policy input

### 2. `deploy.py`
- **Added base_rpy tracking** from IMU state
- **Added base_linvel** (currently zeros, as not directly available from SDK)
- **Updated inference call** to pass base_rpy and base_linvel to policy

### 3. `configs/T1.yaml`
- **Updated policy_path** to point to ONNX model:
  `/home/romela5090/Han/booster_sim2real/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/t1_policy.onnx`
- **Updated num_observations** from 47 to 50
- **Updated gait_frequency** from 1.0 to 1.5 Hz

### 4. `requirements.txt`
- **Replaced torch with onnxruntime**

## Observation Structure (50 dimensions)

The observation vector is constructed as follows:

1. **Local linear velocity** (3D): `[vx, vy, vz]` in robot's local frame
2. **Gyroscope** (3D): `[wx, wy, wz]` angular velocity
3. **Gravity vector** (3D): `[gx, gy, gz]` projected gravity in local frame
4. **Velocity commands** (3D): `[vx_cmd, vy_cmd, vyaw_cmd]`
5. **Joint angles** (12D): Lower body joint positions relative to default (indices 11:23)
   - First 2 joints (ankles) are set to 0
6. **Joint velocities** (12D): Lower body joint velocities (indices 11:23)
   - First 2 joints (ankles) are set to 0
7. **Last action** (12D): Previous policy output
8. **Phase encoding** (2D): `[cos(phase), sin(phase)]` for gait synchronization

## Action Structure (12 dimensions)

The policy outputs 12 actions corresponding to lower body joints (indices 11:23).
Actions are applied as:
```python
dof_targets[11:] = default_dof_pos[11:] + action_scale * actions
```

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Update the ONNX model path** in `configs/T1.yaml` if needed

3. **Run the deployment:**
   ```bash
   python deploy.py --config=T1.yaml --net=<robot_ip>
   ```

## Notes

- **Local linear velocity**: Currently set to zeros as it's not directly available from the Booster SDK. The policy should still work, but performance may be slightly reduced. If you have access to base velocity estimation, you can update `base_linvel` in `deploy.py`.

- **Joint indexing**: The policy uses lower body joints (indices 11:23), which correspond to the leg joints. The first 2 of these (ankle joints) are set to 0 in the observation as they may be controlled separately.

- **Phase tracking**: The gait phase is automatically tracked and updated based on the gait frequency. When commands are zero, the phase is set to π (stance phase).

- **Action scaling**: Actions are scaled by `action_scale` (default 1.0) before being added to the default joint positions.

## Troubleshooting

- **Observation size mismatch**: Ensure `num_observations: 50` in the config matches the ONNX model input size
- **Action size mismatch**: Ensure `num_actions: 12` matches the ONNX model output size
- **Model loading errors**: Verify the ONNX model path is correct and the file exists
- **Performance issues**: Consider using GPU execution provider if available:
  ```python
  providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
  ```


# T1 Robot Joint Mapping from XML Analysis

## Complete Joint Structure (from t1_mjx_feetonly.xml)

### MuJoCo qpos Structure:
- **qpos[0:3]**: Base position (x, y, z)
- **qpos[3:7]**: Base orientation (quaternion: w, x, y, z)
- **qpos[7:30]**: All 23 joints (after free joint)

### Joint Order in qpos[7:] (matches actuator order in XML):

| Index in qpos[7:] | Index in full qpos | Joint Name | Joint Type | Notes |
|-------------------|-------------------|------------|------------|-------|
| 0 | 7 | AAHead_yaw | Head | **ZEROED in play_t1_joystick.py** |
| 1 | 8 | Head_pitch | Head | **ZEROED in play_t1_joystick.py** |
| 2 | 9 | Left_Shoulder_Pitch | Left Arm | |
| 3 | 10 | Left_Shoulder_Roll | Left Arm | |
| 4 | 11 | Left_Elbow_Pitch | Left Arm | |
| 5 | 12 | Left_Elbow_Yaw | Left Arm | |
| 6 | 13 | Right_Shoulder_Pitch | Right Arm | |
| 7 | 14 | Right_Shoulder_Roll | Right Arm | |
| 8 | 15 | Right_Elbow_Pitch | Right Arm | |
| 9 | 16 | Right_Elbow_Yaw | Right Arm | |
| 10 | 17 | Waist | Waist | |
| 11 | 18 | Left_Hip_Pitch | Left Leg | Lower body starts here |
| 12 | 19 | Left_Hip_Roll | Left Leg | |
| 13 | 20 | Left_Hip_Yaw | Left Leg | |
| 14 | 21 | Left_Knee_Pitch | Left Leg | |
| 15 | 22 | Left_Ankle_Pitch | Left Leg | **Parallel mechanism** |
| 16 | 23 | Left_Ankle_Roll | Left Leg | **Parallel mechanism** |
| 17 | 24 | Right_Hip_Pitch | Right Leg | |
| 18 | 25 | Right_Hip_Roll | Right Leg | |
| 19 | 26 | Right_Hip_Yaw | Right Leg | |
| 20 | 27 | Right_Knee_Pitch | Right Leg | |
| 21 | 28 | Right_Ankle_Pitch | Right Leg | **Parallel mechanism** |
| 22 | 29 | Right_Ankle_Roll | Right Leg | **Parallel mechanism** |

## Key Findings:

### 1. Joint Zeroing in play_t1_joystick.py:
- **Lines 92-93**: `joint_angles[:2] *= 0.0` and `joint_velocities[:2] *= 0.0`
- This zeros **indices 0-1** of `qpos[7:]`, which are:
  - **Index 0**: AAHead_yaw (Head joint)
  - **Index 1**: Head_pitch (Head joint)
- **NOT ankle joints** - these are head joints!

### 2. Lower Body Joints (Policy Output):
- Lower body joints start at **index 11** in `qpos[7:]` (Left_Hip_Pitch)
- Lower body includes: **12 joints** (indices 11-22)
  - Left leg: Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll (6 joints)
  - Right leg: Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll (6 joints)

### 3. Parallel Mechanism Joints:
- **Indices 15-16**: Left_Ankle_Pitch, Left_Ankle_Roll
- **Indices 21-22**: Right_Ankle_Pitch, Right_Ankle_Roll
- These are controlled via **torque** in hardware (not position)
- In Booster SDK: `parallel_mech_indexes: [15, 16, 21, 22]` (in full joint array, so indices 15, 16, 21, 22)

### 4. Observation Structure:

**Training code (joystick.py)** uses:
- `qpos[7:]` → all 23 joints (NO zeroing)
- Observation includes all 23 joints

**Deployment code (play_t1_joystick.py)** uses:
- `qpos[7:]` → all 23 joints
- **Zeros first 2** (head joints) → still 23 elements, but first 2 are zero
- Observation includes all 23 joints (with head joints zeroed)

### 5. Action Space:
- Policy outputs **12 actions** (lower body only: indices 11-22 in qpos[7:])
- Actions correspond to: Left_Hip_Pitch through Right_Ankle_Roll

## Critical Mismatch Found and Fixed:

**Previous issue in deploy_onnx/utils/policy.py:**
- Was using `dof_pos[11:]` which correctly gets indices 11-22 (12 lower body joints) ✓
- But was incorrectly zeroing `joint_angles_rel[:2]` which zeroed Left_Hip_Pitch and Left_Hip_Roll ✗

**Correct approach (now fixed):**
- Use `dof_pos[11:]` to get lower body joints (indices 11-22: 12 joints) ✓
- Do NOT zero anything - the zeroing in `play_t1_joystick.py` is for head joints (indices 0-1), which we're not including since we only use lower body joints ✓
- This matches the 50D observation format expected by the ONNX model

## Mapping to Booster SDK:

The Booster SDK uses a different indexing:
- **B1JointCnt = 23** joints total
- **parallel_mech_indexes: [15, 16, 21, 22]** in the SDK's joint array
- These correspond to ankle joints in the SDK's indexing

**SDK Joint Mapping (from b1_api_const.hpp):**
- 0-1: Head (AAHead_yaw, Head_pitch)
- 2-5: Left arm
- 6-9: Right arm  
- 10: Waist
- 11-16: Left leg (Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, CrankUpLeft, CrankDownLeft)
- 17-22: Right leg (Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, CrankUpRight, CrankDownRight)

**SDK parallel_mech_indexes [15, 16, 21, 22]** = Left and Right ankle joints (CrankUp/Down)

## Recommendation:

For ONNX deployment, match `play_t1_joystick.py` exactly:
1. Use **all 23 joints** in observation (not just 12)
2. Zero **indices 0-1** (head joints: AAHead_yaw, Head_pitch)
3. Observation size should account for all 23 joints (with first 2 zeroed)


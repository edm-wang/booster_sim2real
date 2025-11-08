# ONNX Model Deployment Guide

## Model Information

**Model File:** `checkpoint/policy.onnx`

### Inputs
- **Name:** `obs`
- **Shape:** `[1, 85]` (batch_size=1, observation_size=85)
- **Type:** `float32`
- **Description:** Observation vector containing:
  - Linear velocity (3)
  - Angular velocity (3)
  - Gravity vector (3)
  - Commands (3)
  - Gait phase (4)
  - Joint positions relative to default (23)
  - Joint velocities (23)
  - Previous actions (23)

### Outputs
- **Name:** `continuous_actions`
- **Shape:** `[1, 23]` (batch_size=1, action_size=23)
- **Type:** `float32`
- **Description:** Action values for 23 joints (already tanh-scaled, range: [-1, 1])

## Deployment Options

### 1. Python Deployment (ONNX Runtime)

**Installation:**
```bash
pip install onnxruntime  # CPU only
# OR
pip install onnxruntime-gpu  # GPU support
```

**Basic Usage:**
```python
import numpy as np
import onnxruntime as rt

# Load model
session = rt.InferenceSession("checkpoint/policy.onnx", providers=['CPUExecutionProvider'])

# Prepare observation (85 dimensions)
obs = np.random.randn(85).astype(np.float32).reshape(1, 85)

# Run inference
actions = session.run(None, {"obs": obs})[0]
print(f"Actions: {actions[0]}")  # Shape: (23,)
```

**With GPU:**
```python
session = rt.InferenceSession(
    "checkpoint/policy.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### 2. C++ Deployment

**Installation:**
```bash
# Download ONNX Runtime C++ library from:
# https://github.com/microsoft/onnxruntime/releases
```

**Example Code:**
```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

// Initialize ONNX Runtime
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PolicyInference");
Ort::SessionOptions session_options;
Ort::Session session(env, "checkpoint/policy.onnx", session_options);

// Prepare input
std::vector<float> obs(85, 0.0f);  // Your observation data
std::vector<int64_t> input_shape = {1, 85};

Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator, OrtMemTypeDefault);

Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, obs.data(), obs.size(), input_shape.data(), 2);

// Run inference
const char* input_names[] = {"obs"};
const char* output_names[] = {"continuous_actions"};

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names, &input_tensor, 1,
    output_names, 1
);

// Get output
float* actions = output_tensors[0].GetTensorMutableData<float>();
// actions is now a pointer to 23 float values
```

### 3. ROS 2 Deployment

Create a ROS 2 node that:
1. Subscribes to sensor topics (IMU, joint states, etc.)
2. Constructs the 85-dim observation vector
3. Runs ONNX inference
4. Publishes actions to joint command topic

**Example Node Structure:**
```python
import rclpy
from rclpy.node import Node
import onnxruntime as rt
import numpy as np

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        # Load ONNX model
        self.session = rt.InferenceSession("checkpoint/policy.onnx")
        
        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # Publisher
        self.action_pub = self.create_publisher(JointCommand, '/joint_commands', 10)
        
        # State storage
        self.obs = np.zeros(85, dtype=np.float32)
    
    def imu_callback(self, msg):
        # Extract IMU data and update observation
        pass
    
    def joint_callback(self, msg):
        # Extract joint data, construct full observation, run inference
        obs = self.construct_observation(msg)
        actions = self.run_policy(obs)
        self.publish_actions(actions)
    
    def run_policy(self, obs):
        obs_batch = obs.reshape(1, 85)
        outputs = self.session.run(None, {"obs": obs_batch})
        return outputs[0][0]  # Return (23,) array
```

### 4. Embedded Deployment (Jetson, Raspberry Pi, etc.)

**For NVIDIA Jetson:**
```bash
# Install ONNX Runtime for Jetson
# Follow: https://github.com/microsoft/onnxruntime/blob/main/BUILD.md#jetson
```

**For ARM devices:**
```bash
# Use ONNX Runtime Mobile or build from source
# https://onnxruntime.ai/docs/build/inferencing.html
```

### 5. Isaac Sim/Isaac Lab Deployment

Since the model uses opset 11 (compatible with Isaac Lab):

```python
from isaaclab import sim
import onnxruntime as rt

# Load model
session = rt.InferenceSession("checkpoint/policy.onnx")

# In your simulation loop:
obs = get_observation()  # Shape: (85,)
obs_batch = obs.reshape(1, 85)
actions = session.run(None, {"obs": obs_batch})[0][0]
apply_actions(actions)  # Shape: (23,)
```

## Performance Considerations

1. **Batch Processing:** The model accepts batch dimension, but for real-time control, use batch_size=1
2. **Latency:** ONNX Runtime is optimized for inference. Typical latency: <1ms on CPU, <0.5ms on GPU
3. **Memory:** Model size is small (~few MB), suitable for embedded deployment

## Integration with Your Robot

Based on your `deploy_brax.py`, you'll need to:

1. **Construct Observation (85 dims):**
   ```python
   obs = np.zeros(85, dtype=np.float32)
   obs[0:3] = linear_velocity      # From accelerometer integration
   obs[3:6] = angular_velocity     # From gyroscope
   obs[6:9] = projected_gravity     # From IMU
   obs[9:12] = [vx, vy, vyaw]       # Commands
   obs[12:16] = gait_phase          # Cos/sin for both feet
   obs[16:39] = joint_pos_relative  # Joint angles - default pose
   obs[39:62] = joint_velocities    # Joint velocities
   obs[62:85] = previous_actions     # Previous action values
   ```

2. **Run Inference:**
   ```python
   obs_batch = obs.reshape(1, 85)
   actions = session.run(None, {"obs": obs_batch})[0][0]
   ```

3. **Apply Actions:**
   ```python
   # Actions are already in [-1, 1] range (tanh applied)
   # Scale if needed for your robot
   joint_targets = actions * action_scale + default_pose
   ```

## Testing

Use the provided `deploy_onnx_example.py` to test the model:
```bash
python deploy_onnx_example.py
```


# ROSBag Policy Processor

This system processes rosbag sensor data and runs the booster policy inference, publishing observation data and commands for PlotJuggler visualization.

## Overview

The `booster_rosbag_policy_processor.py` node:
1. Subscribes to `/booster/sensor_data` from your rosbag
2. Processes sensor data (IMU, joint positions/velocities)
3. Runs policy inference with random velocity commands
4. Publishes observation components and policy outputs for visualization

## Files Created

- `booster_rosbag_policy_processor.py` - Main processing node
- `launch/rosbag_policy_processor.launch.py` - Launch file
- `plotjuggler_rosbag_processor_config.xml` - PlotJuggler configuration
- `ROSBAG_POLICY_PROCESSOR_README.md` - This documentation

## Usage

### Step 1: Build the ROS2 Workspace

```bash
cd /home/romela5090/Han/booster_sim2real/ros2_ws
colcon build --packages-select booster_policy_server
source install/setup.bash
```

### Step 2: Play Your ROSBag

```bash
# In Terminal 1 - Play your rosbag
cd /home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Play your rosbag (replace with your bag file)
ros2 bag play rosbag2_2025_10_15-11_05_42/
```

### Step 3: Launch the Policy Processor

```bash
# In Terminal 2 - Launch the processor
cd /home/romela5090/Han/booster_sim2real/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch with default settings
ros2 launch booster_policy_server rosbag_policy_processor.launch.py

# Or with custom parameters
ros2 launch booster_policy_server rosbag_policy_processor.launch.py \
    config_file:=T1.yaml \
    policy_interval:=0.02 \
    use_random_commands:=true \
    max_velocity:=1.0 \
    max_angular_velocity:=1.0
```

### Step 4: Launch PlotJuggler

```bash
# In Terminal 3 - Launch PlotJuggler with configuration
cd /home/romela5090/Han/booster_sim2real/ros2_ws
source /opt/ros/humble/setup.bash
plotjuggler --layout plotjuggler_rosbag_processor_config.xml
```

## Published Topics

### Observation Components
- `/plotjuggler/projected_gravity` - 3D gravity vector in robot frame
- `/plotjuggler/base_ang_vel` - 3D angular velocity
- `/plotjuggler/velocity_commands` - [vx, vy, vyaw] commands
- `/plotjuggler/gait_phase` - [cos(phase), sin(phase)] gait timing
- `/plotjuggler/joint_positions` - Joint positions (filtered, 12 joints)
- `/plotjuggler/joint_velocities` - Joint velocities (filtered, 12 joints)
- `/plotjuggler/previous_actions` - Previous policy actions (12D)
- `/plotjuggler/observation` - Full 47D observation vector

### Policy Outputs
- `/plotjuggler/actions` - Policy actions (12D)
- `/plotjuggler/joint_targets` - Joint target positions (23D)

## PlotJuggler Visualization

The configuration file provides 8 plot areas:

1. **Raw IMU Data** - Original sensor data from rosbag
2. **Processed Observation Data** - Computed gravity vector and angular velocity
3. **Commands and Gait** - Velocity commands and gait phase
4. **Joint Positions (Filtered)** - Joint positions used by policy (joints 11-22)
5. **Joint Velocities (Filtered)** - Joint velocities used by policy (joints 11-22)
6. **Policy Actions** - Generated policy actions
7. **Joint Targets** - Computed joint target positions
8. **Full Observation Vector** - Complete 47D observation vector

## Configuration Parameters

- `config_file`: Policy configuration file (default: T1.yaml)
- `policy_interval`: Policy inference interval in seconds (default: 0.02)
- `use_random_commands`: Use random velocity commands (default: true)
- `max_velocity`: Maximum velocity for random commands (default: 1.0)
- `max_angular_velocity`: Maximum angular velocity for random commands (default: 1.0)

## Data Flow

```
ROSBag → /booster/sensor_data → Policy Processor → PlotJuggler Topics
   ↓              ↓                    ↓
Raw Data    Observation Vector    Visualization
```

## Troubleshooting

### No Data in PlotJuggler
1. Check if rosbag is playing: `ros2 topic hz /booster/sensor_data`
2. Check if processor is running: `ros2 node list`
3. Check processor logs: `ros2 run booster_policy_server booster_rosbag_policy_processor.py`

### Policy Not Loading
1. Check if config file exists: `/home/romela5090/Han/booster_sim2real/deploy_booster/configs/T1.yaml`
2. Check if model file exists: `/home/romela5090/Han/booster_sim2real/deploy_booster/models/T1.pt`

### No Random Commands
1. Check parameter: `ros2 param get /booster_rosbag_policy_processor use_random_commands`
2. Commands change every second by default

## Expected Results

When working correctly, you should see:
- ✅ Raw sensor data flowing from rosbag
- ✅ Processed observation data being published
- ✅ Policy actions being generated
- ✅ Random velocity commands changing every second
- ✅ All data visible in PlotJuggler

## Customization

### Modify Random Commands
Edit the `generate_random_commands()` method to change command generation:
- Change frequency: modify timer interval
- Change limits: modify `max_velocity` and `max_angular_velocity` parameters
- Add custom patterns: replace random generation with custom logic

### Add Manual Commands
Replace random commands with manual input:
1. Set `use_random_commands:=false`
2. Modify the `vx`, `vy`, `vyaw` variables in the code
3. Or add a subscriber for manual command input

### Modify Observation Processing
Edit the `publish_observation_data()` method to change what data is published for visualization.


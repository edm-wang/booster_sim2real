# Booster Bridge - Robot Hardware Interface

This repository contains the **minimal hardware interface** that runs on the Booster robot. It bridges the Booster SDK with ROS2 communication, allowing the robot to receive motor commands and send sensor data over the network.

## Architecture Overview

### Original Architecture (deploy.py)
The original `deploy.py` was a monolithic controller that ran everything on the robot:
- Policy inference (neural network)
- Remote control handling (joystick/keyboard)
- Robot communication (Booster SDK)
- Motor command generation
- Sensor data processing

### New Separated Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   5090 Computer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Policy Server   │  │ Remote Control  │  │ Monitoring  │ │
│  │ (Neural Net)    │  │ (Joystick)     │  │ & Logging   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│           │                    │                    │        │
│           └────────────────────┼────────────────────┘        │
│                                │                            │
│                    ROS2 Network Communication              │
│                                │                            │
└────────────────────────────────┼────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────┐
│                    Booster Robot                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Hardware Bridge (This Repository)            │ │
│  │  • Minimal sensor data collection                     │ │
│  │  • Motor command execution                            │ │
│  │  • Safety checks                                      │ │
│  │  • ROS2 communication                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                │                            │
│                    Booster SDK Communication               │
│                                │                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                Booster Robot Hardware                  │ │
│  │  • Motors, IMU, Sensors                               │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Hardware Bridge (`booster_hardware_bridge.py`)
- **Purpose**: Minimal hardware interface running on the robot
- **Responsibilities**:
  - Receive motor commands via ROS2
  - Send sensor data via ROS2
  - Handle robot mode switching
  - Implement safety checks
  - Manage Booster SDK communication

### 2. ROS2 Messages
- **`BoosterSensorData`**: Complete sensor data from robot
- **`BoosterMotorCmd`**: Motor commands to robot
- **`BoosterControlCmd`**: High-level control commands
- **`BoosterRobotMode`**: Robot mode switching service

### 3. Configuration (`config/robot_config.yaml`)
- Robot-specific settings
- Safety limits
- Control parameters
- ROS2 communication settings

## Installation and Setup

### Prerequisites
- ROS2 (Humble or later)
- Python 3.8+
- Booster SDK (on robot hardware)

### Build the Package
```bash
cd ros2_ws
colcon build --packages-select booster_hardware_bridge booster_msgs
source install/setup.bash
```

### Run the Hardware Bridge
```bash
# With default configuration
ros2 launch booster_hardware_bridge robot_bridge.launch.py

# With custom configuration
ros2 launch booster_hardware_bridge robot_bridge.launch.py config_file:=/path/to/config.yaml

# In simulation mode (no Booster SDK)
ros2 launch booster_hardware_bridge robot_bridge.launch.py use_simulation:=true
```

## ROS2 Topics and Services

### Published Topics
- **`/booster/sensor_data`** (`booster_msgs/msg/BoosterSensorData`)
  - Complete sensor data from robot
  - Published at 500Hz
  - Includes IMU, joint positions, velocities, torques

### Subscribed Topics
- **`/booster/motor_cmd`** (`booster_msgs/msg/BoosterMotorCmd`)
  - Motor commands from policy server
  - Includes positions, velocities, torques, gains

- **`/booster/control_cmd`** (`booster_msgs/msg/BoosterControlCmd`)
  - High-level control commands
  - Includes velocity commands and mode switching

### Services
- **`/booster/robot_mode`** (`booster_msgs/srv/BoosterRobotMode`)
  - Switch robot operating modes
  - Modes: kDamping=0, kCustom=1, kStand=2

## Configuration

The hardware bridge uses a YAML configuration file with the following sections:

```yaml
common:
  dt: 0.002                    # Control loop frequency
  stiffness: [20, 20, ...]     # Joint stiffness values
  damping: [0.2, 0.2, ...]     # Joint damping values
  default_qpos: [0, 0, ...]   # Default joint positions
  torque_limit: [7, 7, ...]   # Joint torque limits

mech:
  parallel_mech_indexes: [15, 16, 21, 22]  # Parallel mechanism joints

safety:
  max_imu_rpy: 1.0             # Maximum IMU values before emergency stop
  motor_cmd_timeout: 0.1       # Motor command timeout (seconds)
  max_joint_velocity: 10.0     # Maximum joint velocity (rad/s)
  max_joint_torque: 100.0      # Maximum joint torque (Nm)

ros2:
  sensor_data_topic: "booster/sensor_data"
  motor_cmd_topic: "booster/motor_cmd"
  control_cmd_topic: "booster/control_cmd"
  robot_mode_service: "booster/robot_mode"
```

## Safety Features

The hardware bridge implements several safety mechanisms from the original `deploy.py`:

1. **IMU Safety Check**: Emergency stop if IMU values exceed limits
2. **Motor Command Timeout**: Stop robot if no commands received
3. **Torque Limits**: Enforce maximum joint torques
4. **Velocity Limits**: Enforce maximum joint velocities
5. **Parallel Mechanism Handling**: Special handling for parallel joints

## Development

### Adding New Features
1. Update ROS2 messages in `booster_msgs/`
2. Modify hardware bridge in `booster_hardware_bridge.py`
3. Update configuration file if needed
4. Test with simulation mode first

### Testing
```bash
# Test in simulation mode
ros2 launch booster_hardware_bridge robot_bridge.launch.py use_simulation:=true

# Monitor topics
ros2 topic echo /booster/sensor_data
ros2 topic echo /booster/motor_cmd

# Test robot mode switching
ros2 service call /booster/robot_mode booster_msgs/srv/BoosterRobotMode "{mode: 1}"
```

## Differences from deploy.py

### What was removed (moved to 5090 computer):
- Policy inference (neural network)
- Remote control handling (joystick/keyboard)
- High-level control logic
- Gait generation

### What was kept (on robot):
- Booster SDK communication
- Sensor data collection
- Motor command execution
- Safety checks
- Robot mode management

### What was added:
- ROS2 communication layer
- Configuration file handling
- Simulation mode support
- Enhanced logging and monitoring

## Troubleshooting

### Common Issues
1. **SDK Import Error**: Install Booster SDK or use simulation mode
2. **Config File Not Found**: Check file path in launch arguments
3. **Topic Connection Issues**: Verify ROS2 network configuration
4. **Safety Stop**: Check IMU values and motor commands

### Debug Commands
```bash
# Check node status
ros2 node list
ros2 node info /booster_hardware_bridge

# Monitor topics
ros2 topic list
ros2 topic hz /booster/sensor_data

# Check services
ros2 service list
ros2 service type /booster/robot_mode
```

## Contributing

When contributing to this repository:
1. Follow the existing code structure
2. Add appropriate logging
3. Update configuration files if needed
4. Test in simulation mode first
5. Update documentation

## License

This project follows the same license as the original Booster Gym repository.
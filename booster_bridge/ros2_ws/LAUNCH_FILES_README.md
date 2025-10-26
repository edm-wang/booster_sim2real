# Booster Bridge Launch Files

This directory contains launch files for the Booster robot bridge system.

## Available Launch Files

### 1. `sensor_data.launch.py`
**Purpose**: Launches only the sensor data publisher
**Use Case**: When you only need to collect sensor data from the robot
```bash
ros2 launch booster_hardware_bridge sensor_data.launch.py
```

### 2. `booster_bridge.launch.py` 
**Purpose**: Launches the main hardware bridge
**Use Case**: Full robot control and communication
```bash
ros2 launch booster_hardware_bridge booster_bridge.launch.py
```

### 3. `complete_bridge.launch.py`
**Purpose**: Launches both sensor data and main bridge
**Use Case**: Complete robot system with data collection
```bash
ros2 launch booster_hardware_bridge complete_bridge.launch.py
```

### 4. `data_recording.launch.py`
**Purpose**: Launches sensor data and data plotter
**Use Case**: Real-time data visualization and recording
```bash
ros2 launch booster_hardware_bridge data_recording.launch.py
```

## Configuration

All launch files accept a `config_file` parameter:
```bash
ros2 launch booster_hardware_bridge sensor_data.launch.py config_file:=/path/to/your/config.yaml
```

## Topics

### Published Topics
- `/booster/sensor_data` - Robot sensor data (IMU, joint positions, velocities, torques)

### Subscribed Topics  
- `/booster/motor_cmd` - Motor commands to robot
- `/booster/robot_mode` - Robot mode control

## Usage Examples

### Basic Sensor Data Collection
```bash
# Terminal 1: Start sensor data collection
ros2 launch booster_hardware_bridge sensor_data.launch.py

# Terminal 2: Record data to rosbag
ros2 bag record /booster/sensor_data
```

### Complete Robot Control
```bash
# Start complete bridge system
ros2 launch booster_hardware_bridge complete_bridge.launch.py
```

### Real-time Data Visualization
```bash
# Start data recording and plotting
ros2 launch booster_hardware_bridge data_recording.launch.py
```

## Requirements

- Booster SDK must be available
- Robot must be connected and powered on
- Conda environment `lambo` must be activated

## Troubleshooting

1. **SDK not available**: Make sure conda environment is activated
2. **Config file not found**: Check the path in the launch file
3. **Robot not responding**: Verify robot connection and power

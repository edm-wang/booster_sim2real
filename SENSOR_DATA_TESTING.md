# Booster Sensor Data Collection and PlotJuggler Integration

This document provides instructions for testing the Booster robot sensor data collection system and integrating it with PlotJuggler for data visualization.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Booster Robot                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Sensor Publisher Node                         │ │
│  │  • Collects sensor data from Booster SDK              │ │
│  │  • Publishes to /booster/sensor_data topic            │ │
│  │  • Runs at 100Hz (configurable)                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                │                            │
│                    ROS2 Network Communication              │
│                                │                            │
└────────────────────────────────┼────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────┐
│                    5090 Computer                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │           Data Plotter Node                              │ │
│  │  • Receives /booster/sensor_data                         │ │
│  │  • Converts to PlotJuggler-compatible format           │ │
│  │  • Publishes to plotjuggler/* topics                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                │                            │
│                    PlotJuggler Visualization                │
│                                │                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           PlotJuggler GUI                              │ │
│  │  • Real-time data visualization                        │ │
│  │  • Multiple plot types                                │ │
│  │  • Data recording and playback                        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### On the Booster Robot
- ROS2 Humble or later
- Booster SDK (for real robot) or simulation mode
- Python 3.8+

### On the 5090 Computer
- ROS2 Humble or later
- Python 3.8+
- PlotJuggler (for visualization)

## Installation

### 1. Install PlotJuggler (on 5090 computer)
```bash
sudo apt update
sudo apt install plotjuggler
```

### 2. Build the Workspaces

#### Build booster_bridge workspace (on robot):
```bash
cd /home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws
colcon build --packages-select booster_hardware_bridge booster_msgs
source install/setup.bash
```

#### Build ros2_ws workspace (on 5090 computer):
```bash
cd /home/romela5090/Han/booster_sim2real/ros2_ws
colcon build --packages-select booster_policy_server booster_msgs
source install/setup.bash
```

## Usage Instructions

### Step 1: Launch Sensor Publisher (on Booster Robot)

#### For Real Robot (with Booster SDK):
```bash
cd /home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws
source install/setup.bash
ros2 launch booster_hardware_bridge sensor_publisher.launch.py
```

#### For Simulation Mode (without robot):
```bash
cd /home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws
source install/setup.bash
ros2 launch booster_hardware_bridge sensor_publisher.launch.py use_simulation:=true
```

### Step 2: Launch Data Plotter (on 5090 Computer)
```bash
cd /home/romela5090/Han/booster_sim2real/ros2_ws
source install/setup.bash
ros2 launch booster_policy_server data_plotter.launch.py
```

### Step 3: Launch PlotJuggler (on 5090 Computer)
```bash
plotjuggler
```

### Step 4: Configure PlotJuggler

1. In PlotJuggler, click "Add" to add data sources
2. Select "ROS2" as the data source
3. Add the following topics:
   - `plotjuggler/imu_rpy` - IMU roll, pitch, yaw
   - `plotjuggler/imu_gyro` - IMU gyroscope data
   - `plotjuggler/imu_acc` - IMU accelerometer data
   - `plotjuggler/joint_positions` - All joint positions
   - `plotjuggler/joint_velocities` - All joint velocities
   - `plotjuggler/joint_torques` - All joint torques
   - `plotjuggler/imu_roll` - Individual IMU roll
   - `plotjuggler/imu_pitch` - Individual IMU pitch
   - `plotjuggler/imu_yaw` - Individual IMU yaw

4. Create plots by dragging topics to the plot area
5. Start recording data by clicking the "Record" button

## Monitoring and Debugging

### Check System Status
```bash
# List all nodes
ros2 node list

# Check node information
ros2 node info /booster_sensor_publisher
ros2 node info /booster_data_plotter

# List all topics
ros2 topic list

# Check topic frequency
ros2 topic hz /booster/sensor_data
ros2 topic hz /plotjuggler/imu_rpy

# Monitor sensor data
ros2 topic echo /booster/sensor_data
```

### Verify Data Flow
```bash
# Check if sensor data is being published
ros2 topic hz /booster/sensor_data

# Check if PlotJuggler data is being published
ros2 topic hz /plotjuggler/imu_rpy
ros2 topic hz /plotjuggler/joint_positions
```

## Configuration

### Sensor Publisher Parameters
- `use_simulation`: Set to `true` for simulation mode
- `publish_rate`: Sensor data publish rate in Hz (default: 100.0)
- `config_file`: Path to robot configuration file

### Data Plotter Parameters
- `log_level`: Logging level (default: info)

## Troubleshooting

### Common Issues

1. **No sensor data received**
   - Check if the sensor publisher is running
   - Verify ROS2 network configuration
   - Check if topics are being published: `ros2 topic list`

2. **PlotJuggler not receiving data**
   - Ensure the data plotter is running
   - Check if plotjuggler/* topics are being published
   - Verify PlotJuggler is connected to the correct ROS2 domain

3. **Build errors**
   - Ensure all dependencies are installed
   - Check if ROS2 is properly sourced
   - Verify Python packages are installed

4. **Network connectivity issues**
   - Check if both computers are on the same network
   - Verify ROS2 domain configuration
   - Check firewall settings

### Debug Commands
```bash
# Check ROS2 domain
echo $ROS_DOMAIN_ID

# Check network connectivity
ping <robot_ip>

# Check ROS2 discovery
ros2 node list
ros2 topic list
ros2 service list
```

## Data Structure

### BoosterSensorData Message
```yaml
# IMU data
imu_rpy: [roll, pitch, yaw]          # radians
imu_gyro: [x, y, z]                  # rad/s
imu_acc: [x, y, z]                   # m/s²

# Joint data (23 joints)
joint_positions: [pos0, pos1, ...]  # radians
joint_velocities: [vel0, vel1, ...]  # rad/s
joint_torques: [torque0, torque1, ...]  # Nm

# Timestamp
timestamp: builtin_interfaces/Time
```

### PlotJuggler Topics
- `plotjuggler/imu_rpy` - Float64MultiArray with [roll, pitch, yaw]
- `plotjuggler/imu_gyro` - Float64MultiArray with [x, y, z]
- `plotjuggler/imu_acc` - Float64MultiArray with [x, y, z]
- `plotjuggler/joint_positions` - Float64MultiArray with 23 joint positions
- `plotjuggler/joint_velocities` - Float64MultiArray with 23 joint velocities
- `plotjuggler/joint_torques` - Float64MultiArray with 23 joint torques
- `plotjuggler/joint_XX_position` - Float64 for individual joint positions
- `plotjuggler/joint_XX_velocity` - Float64 for individual joint velocities
- `plotjuggler/joint_XX_torque` - Float64 for individual joint torques

## Performance Considerations

- **Publish Rate**: Default 100Hz for sensor data, adjust based on system performance
- **Network Bandwidth**: Monitor network usage with high-frequency data
- **PlotJuggler Performance**: Reduce number of plots for better performance
- **Memory Usage**: Monitor memory usage with long data recordings

## Next Steps

1. **Data Validation**: Verify sensor data accuracy and consistency
2. **Calibration**: Perform sensor calibration if needed
3. **Data Recording**: Set up automated data recording for analysis
4. **Real-time Monitoring**: Create custom dashboards for specific metrics
5. **Integration**: Integrate with existing monitoring systems

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review ROS2 and PlotJuggler documentation
3. Check system logs for error messages
4. Verify network connectivity and ROS2 configuration

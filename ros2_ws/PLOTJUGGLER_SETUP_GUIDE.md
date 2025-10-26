# PlotJuggler Setup Guide for Booster Sensor Data

This guide shows you how to set up PlotJuggler to visualize Booster robot sensor data.

## Prerequisites

- Conda environment named "robotics" 
- ROS2 Humble installed
- Booster sensor data system running

## Quick Start

### 1. Run the Setup Script
```bash
cd /home/romela5090/Han/booster_sim2real/ros2_ws
./setup_and_test.sh
```

### 2. Manual Setup (if needed)

#### Activate Environment and Source ROS2
```bash
# Activate conda environment
conda activate robotics

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace
cd /home/romela5090/Han/booster_sim2real/ros2_ws
source install/setup.bash
```

## Step-by-Step Testing

### Step 1: Start Sensor Publisher (Robot/Simulation)
```bash
# Terminal 1 - Robot side
conda activate robotics
source /opt/ros/humble/setup.bash
cd /home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws
source install/setup.bash
ros2 launch booster_hardware_bridge sensor_publisher.launch.py use_simulation:=true
```

### Step 2: Start Data Plotter (This Computer)
```bash
# Terminal 2 - This computer
conda activate robotics
source /opt/ros/humble/setup.bash
cd /home/romela5090/Han/booster_sim2real/ros2_ws
source install/setup.bash
ros2 launch booster_policy_server data_plotter.launch.py
```

### Step 3: Test Sensor Data (Optional)
```bash
# Terminal 3 - Test subscriber
conda activate robotics
source /opt/ros/humble/setup.bash
cd /home/romela5090/Han/booster_sim2real/ros2_ws
source install/setup.bash
ros2 launch booster_policy_server sensor_test.launch.py
```

### Step 4: Launch PlotJuggler
```bash
# Terminal 4 - PlotJuggler
conda activate robotics
source /opt/ros/humble/setup.bash
plotjuggler
```

## PlotJuggler Configuration

### 1. Connect to ROS2
- Click the **"Add"** button in PlotJuggler
- Select **"ROS2"** as the data source
- Click **"Connect"**

### 2. Add Topics
Add these topics to PlotJuggler:

#### IMU Data
- `/plotjuggler/imu_rpy` - Roll, pitch, yaw array
- `/plotjuggler/imu_gyro` - Gyroscope array  
- `/plotjuggler/imu_acc` - Accelerometer array
- `/plotjuggler/imu_roll` - Individual roll
- `/plotjuggler/imu_pitch` - Individual pitch
- `/plotjuggler/imu_yaw` - Individual yaw

#### Joint Data
- `/plotjuggler/joint_positions` - All joint positions
- `/plotjuggler/joint_velocities` - All joint velocities
- `/plotjuggler/joint_torques` - All joint torques

#### Statistics
- `/plotjuggler/joint_pos_mean` - Joint position mean
- `/plotjuggler/joint_pos_std` - Joint position std
- `/plotjuggler/joint_vel_mean` - Joint velocity mean
- `/plotjuggler/joint_vel_std` - Joint velocity std

### 3. Create Plots
Drag topics to the plot area to create visualizations:

#### Recommended Plot Layout
1. **IMU RPY Plot**: `/plotjuggler/imu_roll`, `/plotjuggler/imu_pitch`, `/plotjuggler/imu_yaw`
2. **IMU Gyro Plot**: `/plotjuggler/imu_gyro_x`, `/plotjuggler/imu_gyro_y`, `/plotjuggler/imu_gyro_z`
3. **IMU Acc Plot**: `/plotjuggler/imu_acc_x`, `/plotjuggler/imu_acc_y`, `/plotjuggler/imu_acc_z`
4. **Joint Positions Plot**: `/plotjuggler/joint_pos_mean`, `/plotjuggler/joint_pos_std`
5. **Joint Velocities Plot**: `/plotjuggler/joint_vel_mean`, `/plotjuggler/joint_vel_std`
6. **Joint Torques Plot**: `/plotjuggler/joint_torque_mean`, `/plotjuggler/joint_torque_std`

### 4. Start Recording
- Click the **"Record"** button to start recording data
- Data will be saved and can be played back later

## Monitoring Commands

### Check System Status
```bash
# List all topics
ros2 topic list

# Check sensor data frequency
ros2 topic hz /booster/sensor_data

# Check PlotJuggler data frequency
ros2 topic hz /plotjuggler/imu_rpy

# Monitor sensor data
ros2 topic echo /booster/sensor_data

# Monitor PlotJuggler data
ros2 topic echo /plotjuggler/imu_rpy
```

### Check Node Status
```bash
# List running nodes
ros2 node list

# Check node information
ros2 node info /booster_sensor_publisher
ros2 node info /booster_data_plotter
```

## Troubleshooting

### Common Issues

1. **No topics visible in PlotJuggler**
   - Check if data plotter is running: `ros2 node list`
   - Check if topics exist: `ros2 topic list | grep plotjuggler`

2. **No sensor data received**
   - Check if sensor publisher is running: `ros2 topic hz /booster/sensor_data`
   - Check ROS2 network connectivity

3. **PlotJuggler not connecting to ROS2**
   - Make sure ROS2 is sourced: `echo $ROS_DISTRO`
   - Check if ROS2 domain is set: `echo $ROS_DOMAIN_ID`

4. **Build errors**
   - Make sure conda environment is activated
   - Make sure ROS2 is sourced
   - Check if all dependencies are installed

### Debug Commands
```bash
# Check environment
echo $CONDA_DEFAULT_ENV
echo $ROS_DISTRO

# Check workspace
cd /home/romela5090/Han/booster_sim2real/ros2_ws
source install/setup.bash
ros2 pkg list | grep booster

# Check topics
ros2 topic list
ros2 topic hz /booster/sensor_data
ros2 topic hz /plotjuggler/imu_rpy
```

## Expected Results

When everything is working correctly, you should see:

✅ **Sensor data flowing at ~100Hz**
✅ **PlotJuggler topics publishing**
✅ **Real-time plots updating in PlotJuggler**
✅ **Data recording working**
✅ **Smooth data visualization**

## Data Structure

### BoosterSensorData (Raw)
```yaml
imu_rpy: [roll, pitch, yaw]          # radians
imu_gyro: [x, y, z]                  # rad/s
imu_acc: [x, y, z]                   # m/s²
joint_positions: [pos0, pos1, ...]  # radians (23 joints)
joint_velocities: [vel0, vel1, ...] # rad/s (23 joints)
joint_torques: [torque0, torque1, ...] # Nm (23 joints)
```

### PlotJuggler Topics (Converted)
```yaml
/plotjuggler/imu_rpy: Float64MultiArray [roll, pitch, yaw]
/plotjuggler/imu_gyro: Float64MultiArray [x, y, z]
/plotjuggler/imu_acc: Float64MultiArray [x, y, z]
/plotjuggler/joint_positions: Float64MultiArray [23 joint positions]
/plotjuggler/joint_velocities: Float64MultiArray [23 joint velocities]
/plotjuggler/joint_torques: Float64MultiArray [23 joint torques]
```

## Next Steps

1. **Data Analysis**: Use PlotJuggler's analysis tools to examine sensor data
2. **Data Recording**: Record data for offline analysis
3. **Custom Plots**: Create custom plot layouts for specific analysis
4. **Data Export**: Export data to CSV or other formats
5. **Real-time Monitoring**: Set up dashboards for continuous monitoring






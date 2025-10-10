# Booster sensor data message
# Contains all sensor data from the booster robot

# IMU data
float32[3] imu_rpy              # Roll, pitch, yaw (radians)
float32[3] imu_gyro             # Angular velocity (rad/s)
float32[3] imu_acc              # Linear acceleration (m/s²)

# Motor/joint data (23 joints for B1 robot)
float32[23] joint_positions     # Joint positions (radians)
float32[23] joint_velocities    # Joint velocities (rad/s)
float32[23] joint_torques       # Estimated torques (Nm)

# Timestamp
builtin_interfaces/Time timestamp


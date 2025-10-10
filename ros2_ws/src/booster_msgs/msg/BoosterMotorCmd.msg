# Booster motor command message
# Contains motor commands to send to the booster robot

# Command type (SERIAL=0, PARALLEL=1)
uint8 cmd_type

# Motor/joint commands (23 joints for B1 robot)
float32[23] joint_positions     # Target positions (radians)
float32[23] joint_velocities    # Target velocities (rad/s)
float32[23] joint_torques       # Torque commands (Nm)
float32[23] joint_kp            # Position gains
float32[23] joint_kd            # Velocity gains

# Timestamp
builtin_interfaces/Time timestamp


# Booster control command message
# Contains high-level control commands (velocity commands)

# Velocity commands
float32 vx                      # Forward velocity (m/s)
float32 vy                      # Lateral velocity (m/s)
float32 vyaw                    # Yaw velocity (rad/s)

# Control flags
bool start_custom_mode          # Request to start custom control mode
bool start_rl_gait              # Request to start RL gait

# Timestamp
builtin_interfaces/Time timestamp


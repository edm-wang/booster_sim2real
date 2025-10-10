# Booster Robot Control Command Message
# This message contains high-level control commands
# Published by: booster_remote_control (on 5090)
# Subscribed by: booster_hardware_bridge (for logging/monitoring)

# Velocity commands (matches RemoteControlService)
float32 vx                      # Forward velocity (m/s)
float32 vy                      # Lateral velocity (m/s)
float32 vyaw                    # Yaw velocity (rad/s)

# Mode switching flags (matches deploy.py logic)
bool start_custom_mode          # Switch to custom mode
bool start_rl_gait              # Start RL gait

# Timestamp
builtin_interfaces/Time timestamp


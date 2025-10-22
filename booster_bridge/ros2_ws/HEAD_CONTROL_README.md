# Booster Robot Head Control System

This system enables real-time head movement control for the Booster robot using ROS2 and the Booster Robotics SDK.

## Overview

The head control system consists of:
- **booster_head_receiver.py**: Receives head movement commands and executes them on the robot using the SDK
- **booster_head_publisher.py**: Publishes head movement commands via service calls
- **test_head_movement.py**: Direct testing script for head movements

## Features

✅ **Real SDK Integration**: Actually moves the robot's head using the Booster Robotics SDK  
✅ **Multiple Control Methods**: Tries different SDK methods for maximum compatibility  
✅ **Comprehensive Logging**: Detailed logs for debugging and monitoring  
✅ **Error Handling**: Robust error handling with fallback methods  
✅ **Service Interface**: Easy-to-use service calls for head control  
✅ **Direct Testing**: Direct command testing without service layer  

## Quick Start

### 1. Start the Head Receiver
```bash
cd /home/booster/Workspace/booster_sim2real/booster_bridge/ros2_ws/src/booster_hardware_bridge/
python3 booster_head_receiver.py
```

### 2. Test Head Movements
```bash
cd /home/booster/Workspace/booster_sim2real/booster_bridge/ros2_ws/
python3 test_head_movement.py
```

### 3. Use Service Interface (Optional)
```bash
# Start the publisher
python3 booster_head_publisher.py

# In another terminal, use the CLI
python3 head_cli.py up
python3 head_cli.py down
```

## Message Format

The system uses the `BoosterControlCmd` message with these head control fields:

```python
bool head_control      # Enable/disable head control
float32 head_pitch     # Head pitch angle (radians)
float32 head_yaw       # Head yaw angle (radians)
```

### Angle Conventions
- **Pitch**: Negative = look up, Positive = look down, 0 = straight
- **Yaw**: Negative = look right, Positive = look left, 0 = straight

## SDK Integration

The system tries multiple SDK methods in order:

1. **Direct Head Control**: `SetHeadPitch()`, `SetHeadYaw()`, `SetHead()`
2. **Command Interface**: `SendCommand('hu')`, `SendCommand('hd')`, `SendCommand('hs')`
3. **Locomotion Interface**: `SetLocomotion()` with head parameters

## Available Commands

### Service Commands
- `up`: Move head up (pitch = -0.3)
- `down`: Move head down (pitch = 1.0)

### Direct Commands
- Any pitch/yaw combination in radians
- Head control can be enabled/disabled per command

## Troubleshooting

### SDK Connection Issues
- Check network interface (default: "eth0")
- Verify robot is connected and powered on
- Check SDK initialization logs

### Head Not Moving
- Check if `head_control` flag is set to `True`
- Verify SDK methods are available (check logs)
- Try different network interfaces if needed

### Debug Information
The system logs:
- Available SDK methods
- Head movement attempts
- Success/failure status
- Error messages with details

## Example Usage

### Python Code
```python
import rclpy
from rclpy.node import Node
from booster_msgs.msg import BoosterControlCmd

# Create head movement command
cmd = BoosterControlCmd()
cmd.head_control = True
cmd.head_pitch = -0.3  # Look up
cmd.head_yaw = 0.0     # No yaw rotation

# Publish command
publisher.publish(cmd)
```

### Service Calls
```bash
# Move head up
ros2 service call /booster/head_control booster_msgs/srv/BoosterHeadControl "{command: 'up'}"

# Move head down  
ros2 service call /booster/head_control booster_msgs/srv/BoosterHeadControl "{command: 'down'}"
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Test Script   │───▶│  ROS2 Publisher  │───▶│ Head Receiver   │
│                 │    │                 │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  Booster SDK    │
                                                │  (B1LocoClient) │
                                                └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  Robot Hardware │
                                                │  (Head Motors)  │
                                                └─────────────────┘
```

## Files

- `booster_head_receiver.py`: Main head control receiver
- `booster_head_publisher.py`: Service-based head control publisher  
- `test_head_movement.py`: Direct testing script
- `head_cli.py`: Command-line interface
- `test_head_service.py`: Service testing script

## Notes

- The system automatically detects available SDK methods
- Multiple fallback methods ensure compatibility
- All angles are in radians (ROS2 standard)
- The system logs detailed information for debugging
- Network interface may need adjustment based on your setup


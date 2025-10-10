#!/bin/bash

# Booster Robot Bridge Build Script
# Builds the ROS2 workspace for the robot hardware bridge

set -e

echo "Building Booster Robot Bridge..."

# Check if ROS2 is sourced
if [[ -z "$ROS_DISTRO" ]]; then
    echo "Warning: ROS2 not sourced"
    echo "Please run: source /opt/ros/humble/setup.bash"
    echo "Then run this script again"
    exit 1
fi

echo "Environment check passed:"
echo "  ROS2 distribution: $ROS_DISTRO"

# Navigate to workspace
cd ros2_ws

# Build workspace
colcon build

# Source workspace
source install/setup.bash

echo "Build completed successfully!"
echo "To use the workspace, run: source install/setup.bash"

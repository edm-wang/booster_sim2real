#!/bin/bash

# Setup script for Booster ROS2 workspace
# This script activates the conda environment and sources the ROS2 workspace

echo "🚀 Setting up Booster ROS2 environment..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lambo

# Set NumPy include paths (needed for future builds)
export CPLUS_INCLUDE_PATH=/home/booster/miniconda3/envs/lambo/lib/python3.10/site-packages/numpy/core/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/home/booster/miniconda3/envs/lambo/lib/python3.10/site-packages/numpy/core/include:$C_INCLUDE_PATH

# Source the ROS2 workspace
source install/setup.bash

echo "✅ Booster ROS2 environment ready!"
echo ""
echo "📋 Available launch files:"
echo "  ros2 launch booster_hardware_bridge sensor_data.launch.py"
echo "  ros2 launch booster_hardware_bridge complete_bridge.launch.py"
echo "  ros2 launch booster_hardware_bridge data_recording.launch.py"
echo ""
echo "🔧 To rebuild: ./build_booster.sh"





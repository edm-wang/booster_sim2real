#!/bin/bash

# Build script for Booster ROS2 workspace
# This script sets up the proper environment and builds the workspace

echo "🔧 Setting up Booster ROS2 build environment..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lambo

# Set NumPy include paths for ROS2 message generation
export CPLUS_INCLUDE_PATH=/home/booster/miniconda3/envs/lambo/lib/python3.10/site-packages/numpy/core/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/home/booster/miniconda3/envs/lambo/lib/python3.10/site-packages/numpy/core/include:$C_INCLUDE_PATH

echo "✅ Environment configured"
echo "📦 Building ROS2 workspace..."

# Build the workspace
colcon build

echo "✅ Build complete!"
echo ""
echo "🚀 To use the workspace, run:"
echo "source install/setup.bash"





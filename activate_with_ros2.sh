#!/bin/bash

# Script to activate conda environment with ROS2 Python packages
# Usage: source activate_with_ros2.sh <conda_env_name>

if [ $# -eq 0 ]; then
    echo "Usage: source activate_with_ros2.sh <conda_env_name>"
    echo "Available environments:"
    conda env list
    return 1
fi

ENV_NAME=$1

# Activate the conda environment
conda activate $ENV_NAME

# Add ROS2 Python packages to PYTHONPATH
export PYTHONPATH="/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"

# Source ROS2 setup if available
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

echo "Activated conda environment: $ENV_NAME"
echo "ROS2 Python packages added to PYTHONPATH"
echo "Current PYTHONPATH: $PYTHONPATH"

#!/bin/bash

# Setup script for running the interactive T1 play script
# This script sets up the proper environment and runs the interactive play

# Set Isaac Gym path
ISAAC_GYM_PATH="/home/romela/Jun/isaacgym/IsaacGym_Preview_4_Package/isaacgym/python"

# Check if Isaac Gym exists
if [ ! -d "$ISAAC_GYM_PATH" ]; then
    echo "Error: Isaac Gym not found at $ISAAC_GYM_PATH"
    echo "Please update the ISAAC_GYM_PATH variable in this script to point to your Isaac Gym installation"
    exit 1
fi

# Set Python path
export PYTHONPATH="$ISAAC_GYM_PATH:$PYTHONPATH"

# Change to the booster_gym directory
cd "$(dirname "$0")"

# Check if YAML config file exists
YAML_FILE="envs/T1.yaml"
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: YAML config file $YAML_FILE not found"
    exit 1
fi

# Extract checkpoint path from YAML file
CHECKPOINT_PATH=$(grep "checkpoint:" "$YAML_FILE" | head -1 | sed 's/.*checkpoint:[[:space:]]*//' | sed 's/[[:space:]]*#.*//')

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: Could not find checkpoint path in $YAML_FILE"
    exit 1
fi

# Check if the checkpoint file exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    echo "Please update the checkpoint path in $YAML_FILE or ensure the file exists"
    exit 1
fi

MODEL_PATH="$CHECKPOINT_PATH"

echo "Starting interactive T1 play script..."
echo "Isaac Gym path: $ISAAC_GYM_PATH"
echo "Model: $MODEL_PATH"
echo ""

# Run the interactive play script (single robot)
python play_interactive.py --task T1

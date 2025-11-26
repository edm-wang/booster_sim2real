#!/usr/bin/env python3

import numpy as np
import yaml
from utils.policy import Policy

def test_policy_inference():
    # Load config
    with open("configs/T1.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Initialize policy
    policy = Policy(cfg)

    # Create dummy inputs similar to what would come from the robot
    time_now = 0.0
    dof_pos = np.zeros(23, dtype=np.float32)  # 23 joints total
    dof_pos[11:] = cfg["common"]["default_qpos"][11:]  # Set lower body to default
    dof_vel = np.zeros(23, dtype=np.float32)
    base_ang_vel = np.array([0.0, 0.0, 0.1], dtype=np.float32)  # Small yaw rotation
    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Gravity down
    vx, vy, vyaw = 0.5, 0.0, 0.0  # Forward command
    base_linvel = np.zeros(3, dtype=np.float32)  # Not available from SDK
    base_rpy = np.zeros(3, dtype=np.float32)  # Assume level

    print("Testing policy inference with dummy data...")
    print(f"Input commands: vx={vx}, vy={vy}, vyaw={vyaw}")
    print(f"Base angular velocity: {base_ang_vel}")
    print(f"Projected gravity: {projected_gravity}")
    print()

    # Run inference
    joint_commands = policy.inference(
        time_now=time_now,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        base_ang_vel=base_ang_vel,
        projected_gravity=projected_gravity,
        vx=vx,
        vy=vy,
        vyaw=vyaw,
        base_linvel=base_linvel,
        base_rpy=base_rpy,
    )

    print("Joint Commands (dof_targets):")
    print(f"Shape: {joint_commands.shape}")
    print(f"Values: {joint_commands}")
    print()

    # Print joint names and values for clarity (based on the mapping)
    # All 23 joints: head (2), arms (8), waist (1), legs (12)
    joint_names = [
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"
    ]

    print("Joint Commands by Name:")
    for i, (name, value) in enumerate(zip(joint_names, joint_commands)):
        print(f"{name:<20}: {value:8.3f}")

    print()
    print("Statistics:")
    print(f"Min: {np.min(joint_commands):.4f}")
    print(f"Max: {np.max(joint_commands):.4f}")
    print(f"Mean: {np.mean(joint_commands):.4f}")
    print(f"Std: {np.std(joint_commands):.4f}")

    # Test with zero commands (should be close to default positions)
    print("\n" + "="*50)
    print("Testing with zero commands (should be close to default)...")
    zero_commands = policy.inference(
        time_now=time_now,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        base_ang_vel=np.zeros(3),
        projected_gravity=projected_gravity,
        vx=0.0, vy=0.0, vyaw=0.0,
        base_linvel=base_linvel,
        base_rpy=base_rpy,
    )

    print(f"Zero command joint commands: {zero_commands}")
    print(f"Difference from default: {zero_commands - cfg['common']['default_qpos']}")

if __name__ == "__main__":
    test_policy_inference()

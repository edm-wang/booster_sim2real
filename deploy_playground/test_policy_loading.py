#!/usr/bin/env python3
"""
Test script to validate playground policy loading and basic inference.
This runs without hardware to verify the policy integration works correctly.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.brax_utils import load_trained_policy
from mujoco_playground import registry

def test_policy_loading():
    """Test loading the playground policy and basic inference."""
    print("🧪 Testing playground policy loading...")
    
    # Check if checkpoint exists
    checkpoint_path = "./checkpoint"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint path {checkpoint_path} does not exist!")
        return False
    
    try:
        # Load environment
        print("📦 Loading T1JoystickFlatTerrain environment...")
        env = registry.load("T1JoystickFlatTerrain")
        
        # Get environment info
        obs_size = env.observation_size["state"]
        act_size = env.action_size
        
        # Convert tuple to int if needed
        if isinstance(obs_size, tuple):
            obs_size = obs_size[0]
        
        print(f"📊 Environment info:")
        print(f"   Observation size: {obs_size}")
        print(f"   Action size: {act_size}")
        
        # Load policy
        print(f"🤖 Loading policy from {checkpoint_path}...")
        policy_fn = load_trained_policy(checkpoint_path, env)
        print("✅ Policy loaded successfully!")
        
        # Test inference with dummy observations
        print("🧠 Testing policy inference...")
        
        # Create dummy observation
        dummy_obs = np.random.randn(obs_size).astype(np.float32)
        print(f"   Dummy observation shape: {dummy_obs.shape}")
        print(f"   Dummy observation range: [{dummy_obs.min():.3f}, {dummy_obs.max():.3f}]")
        
        # Run inference
        import jax
        import jax.numpy as jp
        obs_jax = jp.array(dummy_obs).reshape(1, -1)
        rng = jax.random.PRNGKey(42)
        
        actions, _ = policy_fn({'state': obs_jax}, rng)
        actions = np.array(actions[0])
        
        print(f"✅ Policy inference successful!")
        print(f"   Action shape: {actions.shape}")
        print(f"   Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"   Action mean: {actions.mean():.3f}")
        print(f"   Sample actions: {actions[:5]}")
        
        # Test multiple inferences for timing
        print("⏱️  Testing inference timing...")
        import time
        
        num_tests = 10
        times = []
        
        for i in range(num_tests):
            start_time = time.perf_counter()
            actions, _ = policy_fn({'state': obs_jax}, rng)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        print(f"   Average inference time: {avg_time:.2f} ms")
        print(f"   Min inference time: {min_time:.2f} ms")
        print(f"   Max inference time: {max_time:.2f} ms")
        
        # Check if timing is acceptable for real-time (should be <50ms for 20Hz)
        if avg_time < 50:
            print("✅ Inference timing is acceptable for real-time operation")
        else:
            print("⚠️  Inference timing may be too slow for real-time operation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during policy testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_observation_construction():
    """Test observation construction similar to what the deployment will do."""
    print("\n🔍 Testing observation construction...")
    
    try:
        # Load environment to get observation size
        env = registry.load("T1JoystickFlatTerrain")
        obs_size = env.observation_size["state"]
        # Convert tuple to int if needed
        if isinstance(obs_size, tuple):
            obs_size = obs_size[0]
        
        # Simulate hardware sensor data
        # Use hardcoded joint count since SDK may not be available in test environment
        B1JointCnt = 23  # Booster T1 has 23 joints
        
        # Create simulated sensor data
        dof_pos = np.random.randn(B1JointCnt).astype(np.float32) * 0.1  # Small joint positions
        dof_vel = np.random.randn(B1JointCnt).astype(np.float32) * 0.5  # Joint velocities
        base_ang_vel = np.random.randn(3).astype(np.float32) * 0.1      # Angular velocity
        projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Gravity
        vx, vy, vyaw = 0.1, 0.0, 0.0  # Velocity commands
        
        # Construct observation similar to deployment
        obs = np.zeros(obs_size, dtype=np.float32)
        
        # Basic observation construction (adapt based on actual playground training)
        obs[0:3] = projected_gravity  # Projected gravity
        obs[3:6] = base_ang_vel       # Angular velocity
        
        # Motor states (joint positions and velocities)
        joint_start_idx = 6
        for i in range(min(B1JointCnt, (obs_size - joint_start_idx) // 2)):
            if joint_start_idx + i*2 + 1 < obs_size:
                obs[joint_start_idx + i*2] = dof_pos[i]     # Joint position
                obs[joint_start_idx + i*2 + 1] = dof_vel[i] # Joint velocity
        
        # Commands (velocity commands)
        cmd_start_idx = joint_start_idx + min(B1JointCnt, (obs_size - joint_start_idx) // 2) * 2
        if cmd_start_idx + 2 < obs_size:
            obs[cmd_start_idx:cmd_start_idx+3] = [vx, vy, vyaw]
        
        print(f"✅ Observation construction successful!")
        print(f"   Constructed observation shape: {obs.shape}")
        print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"   Non-zero elements: {np.count_nonzero(obs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during observation construction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting playground policy validation tests...")
    print("=" * 60)
    
    # Test 1: Policy loading and basic inference
    success1 = test_policy_loading()
    
    # Test 2: Observation construction
    success2 = test_observation_construction()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! Policy is ready for deployment testing.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

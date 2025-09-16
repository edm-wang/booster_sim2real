#!/usr/bin/env python3
"""
Safety test script for hardware deployment
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_simulation_mode():
    """Test deployment in simulation mode."""
    print("🧪 Testing simulation mode...")
    
    try:
        from hardware_deployment import HardwareController
        
        # Test with localhost (simulation mode)
        controller = HardwareController(
            checkpoint_path="/home/romela5090/Admond/LAMBO/locomotion/checkpoints/T1JoystickFlatTerrain/20250713_170520",
            use_gamepad=False,
            robot_ip="127.0.0.1"
        )
        
        print("✅ Simulation mode test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Simulation mode test failed: {e}")
        return False

def test_policy_safety():
    """Test policy outputs for safety."""
    print("🧪 Testing policy safety...")
    
    try:
        from hardware_deployment import HardwareController
        
        controller = HardwareController(
            checkpoint_path="/home/romela5090/Admond/LAMBO/locomotion/checkpoints/T1JoystickFlatTerrain/20250713_170520",
            use_gamepad=False,
            robot_ip="127.0.0.1"
        )
        
        # Test policy with dummy state
        obs = controller._get_dummy_state()
        obs_jax = controller.policy_fn({'state': obs.reshape(1, -1)}, controller.policy_fn.key)
        actions = obs_jax[0]
        
        # Check if actions are reasonable
        max_action = max(abs(a) for a in actions)
        if max_action > 10.0:  # Arbitrary safety limit
            print(f"⚠️  Warning: Large action values detected (max: {max_action})")
            return False
        
        print(f"✅ Policy safety test passed! Max action: {max_action}")
        return True
        
    except Exception as e:
        print(f"❌ Policy safety test failed: {e}")
        return False

def test_emergency_stop():
    """Test emergency stop functionality."""
    print("🧪 Testing emergency stop...")
    
    try:
        from hardware_deployment import HardwareController
        
        controller = HardwareController(
            checkpoint_path="/home/romela5090/Admond/LAMBO/locomotion/checkpoints/T1JoystickFlatTerrain/20250713_170520",
            use_gamepad=False,
            robot_ip="127.0.0.1"
        )
        
        # Test emergency stop
        controller.emergency_stop_robot()
        
        if controller.emergency_stop:
            print("✅ Emergency stop test passed!")
            return True
        else:
            print("❌ Emergency stop test failed!")
            return False
        
    except Exception as e:
        print(f"❌ Emergency stop test failed: {e}")
        return False

def main():
    print("🛡️ Safety Test Suite")
    print("=" * 40)
    
    tests = [
        ("Simulation Mode", test_simulation_mode),
        ("Policy Safety", test_policy_safety),
        ("Emergency Stop", test_emergency_stop),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("📊 Safety Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'✅ ALL SAFETY TESTS PASSED' if all_passed else '❌ SOME SAFETY TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Ready for hardware deployment!")
        print("Next steps:")
        print("1. Test robot connection: python test_connection.py <ROBOT_IP>")
        print("2. Start with conservative gains in config.py")
        print("3. Deploy with short test duration first")
    else:
        print("\n🔧 Please fix failed tests before hardware deployment")
    
    return all_passed

if __name__ == "__main__":
    main()


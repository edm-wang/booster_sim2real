#!/usr/bin/env python3
"""
Test script for hardware deployment
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sim2real.hardware_deployment import HardwareController
from sim2real.config import DEFAULT_CHECKPOINT_PATH

def test_policy_loading():
    """Test if the policy can be loaded correctly."""
    print("🧪 Testing policy loading...")
    
    try:
        controller = HardwareController(DEFAULT_CHECKPOINT_PATH, use_gamepad=False)
        print("✅ Policy loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Policy loading failed: {e}")
        return False

def test_gamepad():
    """Test gamepad functionality."""
    print("🧪 Testing gamepad...")
    
    try:
        from sim2real.hardware_deployment import Gamepad
        gamepad = Gamepad()
        time.sleep(1)  # Let it initialize
        
        cmd = gamepad.get_command()
        print(f"✅ Gamepad working! Command: {cmd}")
        gamepad.stop()
        return True
    except Exception as e:
        print(f"❌ Gamepad test failed: {e}")
        return False

def test_booster_sdk():
    """Test Booster SDK availability."""
    print("🧪 Testing Booster SDK...")
    
    try:
        from booster_robotics_sdk_python import B1LocoClient, B1JointCnt
        print("✅ Booster SDK available!")
        print(f"   - Joint count: {B1JointCnt}")
        return True
    except ImportError:
        print("⚠️  Booster SDK not available (expected in simulation)")
        return False

def main():
    print("🧪 Hardware Deployment Test Suite")
    print("=" * 40)
    
    tests = [
        ("Policy Loading", test_policy_loading),
        ("Gamepad", test_gamepad),
        ("Booster SDK", test_booster_sdk),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    import time
    main()

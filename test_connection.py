#!/usr/bin/env python3
"""
Test script for Booster SDK connection
"""

import sys
import time
from pathlib import Path
import threading

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def test_booster_sdk_import():
    """Test if Booster SDK can be imported."""
    print("🧪 Testing Booster SDK import...")
    try:
        from booster_robotics_sdk_python import (
            ChannelFactory, B1LocoClient, B1LowStateSubscriber, B1LowCmdPublisher,
            MotorCmd, LowState, LowCmd, LowCmdType, B1JointCnt, B1JointIndex
        )
        print("✅ Booster SDK imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Booster SDK import failed: {e}")
        return False


def test_dds_connection(robot_ip="127.0.0.1"):
    """Test DDS connection to robot by performing a real API call (GetMode)."""
    print(f"🧪 Testing DDS connection to {robot_ip}...")
    try:
        from booster_robotics_sdk_python import ChannelFactory, B1LocoClient, GetModeResponse
        
        # Initialize DDS on the provided network interface/IP string
        ChannelFactory.Instance().Init(0, robot_ip)
        print("✅ DDS channel factory initialized")
        
        # Test high-level client
        client = B1LocoClient()
        client.Init()
        print("✅ High-level client initialized")
        
        # Perform an RPC that requires a live server
        resp = GetModeResponse()
        ret = client.GetMode(resp)
        if ret != 0:
            print(f"❌ GetMode RPC failed with code: {ret}")
            return False
        print("✅ GetMode RPC succeeded")
        return True
        
    except Exception as e:
        print(f"❌ DDS connection failed: {e}")
        return False


def test_low_level_communication(robot_ip="127.0.0.1"):
    """Require receiving at least one low-level state message within timeout, and publish a no-op command."""
    print(f"🧪 Testing low-level communication to {robot_ip}...")
    try:
        from booster_robotics_sdk_python import (
            ChannelFactory, B1LowStateSubscriber, B1LowCmdPublisher,
            LowCmd, LowCmdType, MotorCmd, B1JointCnt
        )
        
        # Initialize DDS
        ChannelFactory.Instance().Init(0, robot_ip)
        
        # State reception synchronization
        state_received = threading.Event()
        
        def state_handler(state):
            try:
                _ = len(state.motor_state_parallel)
            except Exception:
                pass
            state_received.set()
        
        # Initialize state subscriber
        state_sub = B1LowStateSubscriber(state_handler)
        state_sub.InitChannel()
        print("✅ State subscriber initialized")
        
        # Initialize command publisher
        cmd_pub = B1LowCmdPublisher()
        cmd_pub.InitChannel()
        print("✅ Command publisher initialized")
        
        # Send a safe no-op command
        low_cmd = LowCmd()
        low_cmd.cmd_type = LowCmdType.PARALLEL
        motor_cmds = [MotorCmd() for _ in range(B1JointCnt)]
        for i in range(B1JointCnt):
            motor_cmds[i].q = 0.0
            motor_cmds[i].dq = 0.0
            motor_cmds[i].tau = 0.0
            motor_cmds[i].kp = 0.0
            motor_cmds[i].kd = 0.0
            motor_cmds[i].weight = 0.0
        low_cmd.motor_cmd = motor_cmds
        cmd_pub.Write(low_cmd)
        print("✅ Test command sent successfully")
        
        # Wait for at least one state message
        print("⏳ Waiting for state data (5s timeout)...")
        if not state_received.wait(5.0):
            print("❌ No state received within timeout")
            return False
        print("✅ Received state data")
        return True
        
    except Exception as e:
        print(f"❌ Low-level communication failed: {e}")
        return False


def main():
    print("🔌 Booster SDK Connection Test")
    print("=" * 40)
    
    # Get robot IP/NIC from command line or use default
    robot_ip = "127.0.0.1"
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]
    
    print(f"Testing connection to: {robot_ip}")
    print()
    
    tests = [
        ("SDK Import", test_booster_sdk_import),
        ("DDS Connection", lambda: test_dds_connection(robot_ip)),
        ("Low-level Communication", lambda: test_low_level_communication(robot_ip)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
        if not result:
            print(f"❌ {test_name} failed - stopping tests")
            break
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Ready for hardware deployment!")
        print(f"Run: python sim2real/hardware_deployment.py --robot_ip {robot_ip} --checkpoint_path /path/to/checkpoint")
    else:
        print("\n🔧 Please fix the failed tests before deploying to hardware")
    
    return all_passed

if __name__ == "__main__":
    main()


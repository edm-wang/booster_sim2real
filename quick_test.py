#!/usr/bin/env python3

"""
Quick Test Script for Booster Sensor Data System
This script provides quick commands to test the system
"""

import subprocess
import time
import os


def run_command(cmd, description, timeout=5):
    """Run a command with timeout"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print("❌ FAILED")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("🚀 Booster Sensor Data System - Quick Test")
    print("=" * 50)
    
    # Test 1: Check ROS2 installation
    print("\n1. Checking ROS2 installation...")
    if 'ROS_DISTRO' not in os.environ:
        print("❌ ROS2 not sourced. Please run:")
        print("   source /opt/ros/humble/setup.bash")
        return
    else:
        print(f"✅ ROS2 {os.environ['ROS_DISTRO']} detected")
    
    # Test 2: Build booster_bridge workspace
    print("\n2. Building booster_bridge workspace...")
    os.chdir('/home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws')
    success = run_command("colcon build --packages-select booster_hardware_bridge booster_msgs", 
                         "Building booster_bridge workspace", timeout=30)
    if not success:
        print("❌ Build failed. Check errors above.")
        return
    
    # Test 3: Build ros2_ws workspace
    print("\n3. Building ros2_ws workspace...")
    os.chdir('/home/romela5090/Han/booster_sim2real/ros2_ws')
    success = run_command("colcon build --packages-select booster_policy_server booster_msgs", 
                         "Building ros2_ws workspace", timeout=30)
    if not success:
        print("❌ Build failed. Check errors above.")
        return
    
    print("\n✅ All builds completed successfully!")
    
    # Test 4: Check if PlotJuggler is installed
    print("\n4. Checking PlotJuggler installation...")
    success = run_command("which plotjuggler", "Checking PlotJuggler installation")
    if not success:
        print("⚠️  PlotJuggler not found. Install with:")
        print("   sudo apt install plotjuggler")
    else:
        print("✅ PlotJuggler is installed")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("🎯 QUICK START INSTRUCTIONS")
    print("="*60)
    
    print("\n📋 Terminal Commands to Run:")
    
    print("\n🔧 ON THE ROBOT (booster_bridge folder):")
    print("   cd /home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws")
    print("   source install/setup.bash")
    print("   ros2 launch booster_hardware_bridge sensor_publisher.launch.py use_simulation:=true")
    
    print("\n💻 ON THIS COMPUTER (ros2_ws folder):")
    print("   cd /home/romela5090/Han/booster_sim2real/ros2_ws")
    print("   source install/setup.bash")
    print("   ros2 launch booster_policy_server data_plotter.launch.py")
    
    print("\n📊 FOR PLOTJUGGLER:")
    print("   plotjuggler")
    print("   # Add topics: plotjuggler/imu_rpy, plotjuggler/joint_positions, etc.")
    
    print("\n🔍 MONITORING:")
    print("   ros2 topic list")
    print("   ros2 topic hz /booster/sensor_data")
    print("   ros2 topic hz /plotjuggler/imu_rpy")
    
    print("\n" + "="*60)
    print("🎉 System is ready for testing!")
    print("="*60)


if __name__ == '__main__':
    main()

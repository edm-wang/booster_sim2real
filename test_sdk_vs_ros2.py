#!/usr/bin/env python3

"""
Test Script: Compare SDK Raw Data vs ROS2 Data
This helps verify that your sensor data conversion is working correctly
"""

import subprocess
import time
import threading
import json
from datetime import datetime


def run_sdk_subscriber():
    """Run the SDK subscriber to get raw robot data"""
    print("🔧 Running SDK subscriber to get raw robot data...")
    
    try:
        # Run the SDK subscriber
        process = subprocess.Popen([
            "python3", 
            "/home/romela5090/Han/booster_sim2real/external/booster_robotics_sdk/example/low_level/low_level_subscriber.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Let it run for 5 seconds
        time.sleep(5)
        
        # Terminate and get output
        process.terminate()
        stdout, stderr = process.communicate()
        
        print("✅ SDK subscriber completed")
        print("Raw data output:")
        print(stdout)
        
        if stderr:
            print("SDK errors:")
            print(stderr)
            
        return stdout, stderr
        
    except Exception as e:
        print(f"❌ SDK subscriber failed: {e}")
        return None, str(e)


def run_ros2_monitor():
    """Run ROS2 topic monitoring"""
    print("🔧 Running ROS2 topic monitor...")
    
    try:
        # Run ROS2 topic echo
        process = subprocess.Popen([
            "ros2", "topic", "echo", "/booster/sensor_data", "--once"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for output
        stdout, stderr = process.communicate(timeout=10)
        
        print("✅ ROS2 topic monitor completed")
        print("ROS2 data output:")
        print(stdout)
        
        if stderr:
            print("ROS2 errors:")
            print(stderr)
            
        return stdout, stderr
        
    except Exception as e:
        print(f"❌ ROS2 monitor failed: {e}")
        return None, str(e)


def check_ros2_topics():
    """Check if ROS2 topics are available"""
    print("🔧 Checking ROS2 topics...")
    
    try:
        # List topics
        result = subprocess.run([
            "ros2", "topic", "list"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ ROS2 topics available:")
            print(result.stdout)
            
            # Check if our topic exists
            if "/booster/sensor_data" in result.stdout:
                print("✅ /booster/sensor_data topic found")
                return True
            else:
                print("❌ /booster/sensor_data topic not found")
                return False
        else:
            print("❌ Failed to list ROS2 topics")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ ROS2 topic check failed: {e}")
        return False


def main():
    print("🚀 SDK vs ROS2 Data Comparison Test")
    print("=" * 50)
    
    # Step 1: Check ROS2 topics
    print("\n1. Checking ROS2 topics...")
    if not check_ros2_topics():
        print("❌ ROS2 topics not available. Make sure your sensor publisher is running.")
        print("Run: ros2 launch booster_hardware_bridge sensor_publisher.launch.py")
        return
    
    # Step 2: Test SDK subscriber
    print("\n2. Testing SDK subscriber...")
    sdk_stdout, sdk_stderr = run_sdk_subscriber()
    
    # Step 3: Test ROS2 monitor
    print("\n3. Testing ROS2 monitor...")
    ros2_stdout, ros2_stderr = run_ros2_monitor()
    
    # Step 4: Compare results
    print("\n4. Comparison Results:")
    print("=" * 50)
    
    if sdk_stdout and ros2_stdout:
        print("✅ Both SDK and ROS2 data received")
        print("\n📊 Data Comparison:")
        print("- SDK shows raw robot sensor data")
        print("- ROS2 shows converted sensor data")
        print("- Compare the values to verify conversion is correct")
        
        # Check for specific data
        if "imu:" in sdk_stdout:
            print("✅ SDK IMU data detected")
        if "joint_positions" in ros2_stdout:
            print("✅ ROS2 joint data detected")
            
    elif sdk_stdout and not ros2_stdout:
        print("⚠️  SDK data available but ROS2 data not received")
        print("Check if your sensor publisher is running")
        
    elif not sdk_stdout and ros2_stdout:
        print("⚠️  ROS2 data available but SDK data not received")
        print("Check if robot connection is working")
        
    else:
        print("❌ No data received from either source")
        print("Check robot connection and ROS2 setup")
    
    print("\n🎯 Next Steps:")
    print("1. If SDK data works but ROS2 doesn't: Check your sensor publisher")
    print("2. If ROS2 data works but SDK doesn't: Check robot connection")
    print("3. If both work: Compare the data values to verify conversion")
    print("4. If neither works: Check robot connection and ROS2 setup")


if __name__ == '__main__':
    main()

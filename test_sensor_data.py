#!/usr/bin/env python3

"""
Test Script for Booster Sensor Data Collection and PlotJuggler Integration
This script provides commands to test the sensor data collection system
"""

import subprocess
import time
import os
import sys


def run_command(cmd, description):
    """Run a command and print the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False


def check_ros2_installation():
    """Check if ROS2 is properly installed"""
    print("Checking ROS2 installation...")
    
    # Check if ROS2 is sourced
    if 'ROS_DISTRO' not in os.environ:
        print("❌ ROS2 not sourced. Please run: source /opt/ros/humble/setup.bash")
        return False
    
    print(f"✅ ROS2 {os.environ['ROS_DISTRO']} detected")
    return True


def build_workspaces():
    """Build both ROS2 workspaces"""
    print("\nBuilding ROS2 workspaces...")
    
    # Build booster_bridge workspace
    print("\n1. Building booster_bridge workspace...")
    os.chdir('/home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws')
    success = run_command("colcon build --packages-select booster_hardware_bridge booster_msgs", 
                         "Building booster_bridge workspace")
    
    if not success:
        return False
    
    # Build ros2_ws workspace
    print("\n2. Building ros2_ws workspace...")
    os.chdir('/home/romela5090/Han/booster_sim2real/ros2_ws')
    success = run_command("colcon build --packages-select booster_policy_server booster_msgs", 
                         "Building ros2_ws workspace")
    
    return success


def test_sensor_publisher():
    """Test the sensor publisher in simulation mode"""
    print("\nTesting sensor publisher (simulation mode)...")
    
    os.chdir('/home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws')
    
    # Source the workspace
    run_command("source install/setup.bash", "Sourcing booster_bridge workspace")
    
    # Launch sensor publisher in simulation mode
    print("\nLaunching sensor publisher...")
    print("This will run for 10 seconds to test data publishing...")
    
    try:
        # Launch in background
        process = subprocess.Popen([
            "ros2", "launch", "booster_hardware_bridge", "sensor_publisher.launch.py",
            "--ros-args", "-p", "use_simulation:=true", "-p", "publish_rate:=50.0"
        ])
        
        # Wait for 10 seconds
        time.sleep(10)
        
        # Terminate the process
        process.terminate()
        process.wait()
        
        print("✅ Sensor publisher test completed")
        return True
        
    except Exception as e:
        print(f"❌ Sensor publisher test failed: {e}")
        return False


def test_data_receiver():
    """Test the data receiver"""
    print("\nTesting data receiver...")
    
    os.chdir('/home/romela5090/Han/booster_sim2real/ros2_ws')
    
    # Source the workspace
    run_command("source install/setup.bash", "Sourcing ros2_ws workspace")
    
    # Check if topics are available
    print("\nChecking available topics...")
    run_command("ros2 topic list", "List all topics")
    run_command("ros2 topic hz /booster/sensor_data", "Check sensor data frequency")
    
    return True


def test_plotjuggler_integration():
    """Test PlotJuggler integration"""
    print("\nTesting PlotJuggler integration...")
    
    os.chdir('/home/romela5090/Han/booster_sim2real/ros2_ws')
    
    # Source the workspace
    run_command("source install/setup.bash", "Sourcing ros2_ws workspace")
    
    # Launch data plotter
    print("\nLaunching data plotter for PlotJuggler...")
    print("This will run for 10 seconds to test data publishing...")
    
    try:
        # Launch in background
        process = subprocess.Popen([
            "ros2", "launch", "booster_policy_server", "data_plotter.launch.py"
        ])
        
        # Wait for 10 seconds
        time.sleep(10)
        
        # Terminate the process
        process.terminate()
        process.wait()
        
        print("✅ PlotJuggler integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ PlotJuggler integration test failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*80)
    print("BOOSTER SENSOR DATA TESTING - USAGE INSTRUCTIONS")
    print("="*80)
    
    print("\n1. ON THE ROBOT (booster_bridge folder):")
    print("   cd /home/romela5090/Han/booster_sim2real/booster_bridge/ros2_ws")
    print("   source install/setup.bash")
    print("   ros2 launch booster_hardware_bridge sensor_publisher.launch.py")
    print("   # For simulation mode:")
    print("   ros2 launch booster_hardware_bridge sensor_publisher.launch.py use_simulation:=true")
    
    print("\n2. ON THIS COMPUTER (ros2_ws folder):")
    print("   cd /home/romela5090/Han/booster_sim2real/ros2_ws")
    print("   source install/setup.bash")
    print("   ros2 launch booster_policy_server data_plotter.launch.py")
    
    print("\n3. FOR PLOTJUGGLER:")
    print("   # Install PlotJuggler if not already installed:")
    print("   sudo apt install plotjuggler")
    print("   # Launch PlotJuggler:")
    print("   plotjuggler")
    print("   # In PlotJuggler, add ROS2 topics starting with 'plotjuggler/'")
    
    print("\n4. MONITORING COMMANDS:")
    print("   # Check topics:")
    print("   ros2 topic list")
    print("   ros2 topic echo /booster/sensor_data")
    print("   ros2 topic hz /booster/sensor_data")
    print("   ros2 topic hz /plotjuggler/imu_rpy")
    
    print("\n5. TROUBLESHOOTING:")
    print("   # Check if nodes are running:")
    print("   ros2 node list")
    print("   ros2 node info /booster_sensor_publisher")
    print("   ros2 node info /booster_data_plotter")
    
    print("\n6. SIMULATION MODE (for testing without robot):")
    print("   # On robot computer:")
    print("   ros2 launch booster_hardware_bridge sensor_publisher.launch.py use_simulation:=true")
    print("   # On this computer:")
    print("   ros2 launch booster_policy_server data_plotter.launch.py")
    
    print("\n" + "="*80)


def main():
    """Main test function"""
    print("Booster Sensor Data Collection Test Script")
    print("="*50)
    
    # Check ROS2 installation
    if not check_ros2_installation():
        return
    
    # Build workspaces
    if not build_workspaces():
        print("❌ Build failed. Please check the errors above.")
        return
    
    print("\n✅ All builds completed successfully!")
    
    # Print usage instructions
    print_usage_instructions()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run the sensor publisher on the robot (or in simulation mode)")
    print("2. Run the data plotter on this computer")
    print("3. Launch PlotJuggler and add the plotjuggler/* topics")
    print("4. Monitor the data flow and verify sensor readings")


if __name__ == '__main__':
    main()

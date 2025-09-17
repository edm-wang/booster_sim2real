#!/usr/bin/env python3
"""
Observation Data Writer
======================

Simple utility to write observation data to a file that can be read
by the standalone web plotter.

Usage:
    from observation_writer import write_observation
    
    # Write observation data
    write_observation(obs_array, timestamp, "/tmp/robot_obs.json")
"""

import json
import time
import numpy as np

def write_observation(observation, timestamp=None, data_file="/tmp/robot_obs.json"):
    """
    Write observation data to a file for the web plotter to read.
    
    Args:
        observation: numpy array or list of observation data
        timestamp: timestamp (defaults to current time)
        data_file: path to the data file
    """
    if timestamp is None:
        timestamp = time.time()
    
    # Convert numpy array to list if needed
    if isinstance(observation, np.ndarray):
        observation = observation.tolist()
    
    # Create human-readable timestamp
    readable_time = time.strftime("%H:%M:%S", time.localtime(timestamp))
    
    # Create data structure
    data = {
        "observation": observation,
        "timestamp": timestamp,
        "readable_time": readable_time,
        "data_file": data_file
    }
    
    try:
        # Write to file atomically
        temp_file = data_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic move
        import os
        os.rename(temp_file, data_file)
        
    except Exception as e:
        print(f"⚠️  Error writing observation data: {e}")

def write_test_data(data_file="/tmp/robot_obs.json", duration=30):
    """
    Write test observation data for testing the web plotter.
    
    Args:
        data_file: path to the data file
        duration: how long to write test data (seconds)
    """
    print(f"🧪 Writing test observation data to {data_file}")
    print(f"⏰ Duration: {duration} seconds")
    
    start_time = time.time()
    counter = 0
    
    try:
        while time.time() - start_time < duration:
            # Create fake observation data (85 dimensions)
            obs = np.random.randn(85).astype(np.float32)
            
            # Make some data more realistic
            obs[0:3] = np.sin(counter * 0.1) * 0.5  # Linear velocity
            obs[3:6] = np.cos(counter * 0.1) * 0.2  # Angular velocity
            obs[6:9] = [0, 0, -1]  # Gravity vector
            obs[9:12] = [0.1, 0, 0]  # Commands
            obs[12:16] = [np.cos(counter * 0.2), np.sin(counter * 0.2), 
                         np.cos(counter * 0.2 + 1), np.sin(counter * 0.2 + 1)]  # Gait phase
            obs[16:39] = np.sin(counter * 0.05) * 0.3  # Joint positions
            obs[39:62] = np.cos(counter * 0.05) * 0.1  # Joint velocities
            obs[62:85] = np.sin(counter * 0.03) * 0.2  # Actions
            
            # Write observation
            write_observation(obs, time.time(), data_file)
            
            counter += 1
            time.sleep(0.1)  # 10Hz update rate
            
            if counter % 50 == 0:
                print(f"📊 Written {counter} observations... (time: {time.time() - start_time:.1f}s)")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    print(f"✅ Test data writing completed! ({counter} observations)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Write test observation data')
    parser.add_argument('--data-file', type=str, default='/tmp/robot_obs.json', 
                       help='Data file to write to (default: /tmp/robot_obs.json)')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    write_test_data(args.data_file, args.duration)

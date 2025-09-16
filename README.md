# Sim2Real Hardware Deployment

This directory contains scripts for deploying trained JAX/Flax locomotion policies directly to the Booster T1 robot hardware, **without requiring ONNX conversion**.

## 🚀 Quick Start

### 1. Test the Setup
```bash
cd /home/romela5090/Admond/LAMBO
conda activate robotics
python sim2real/test_deployment.py
```

### 2. Run Hardware Deployment
```bash
# With gamepad control
python sim2real/hardware_deployment.py --checkpoint_path /path/to/checkpoint --use_gamepad

# Without gamepad (keyboard input)
python sim2real/hardware_deployment.py --checkpoint_path /path/to/checkpoint --no_gamepad
```

## 📁 Files

- **`hardware_deployment.py`**: Main deployment script
- **`test_deployment.py`**: Test script to verify setup
- **`config.py`**: Configuration parameters
- **`README.md`**: This file

## 🔧 Features

### ✅ **Direct JAX Checkpoint Loading**
- Uses the existing `load_trained_policy` function from `static_booster_simulator.py`
- No ONNX conversion required
- Supports all JAX/Flax trained policies

### ✅ **Gamepad Control**
- Logitech F710 gamepad support
- Velocity scaling and deadzone
- Real-time joystick input

### ✅ **Booster SDK Integration**
- Direct hardware communication
- Low-level motor control
- IMU and joint state reading

### ✅ **Safety Features**
- Graceful error handling
- Simulation mode fallback
- Resource cleanup

## 🎮 Gamepad Controls

- **Left Stick**: Forward/backward (vx), left/right (vy)
- **Right Stick**: Yaw rotation (wz)
- **Ctrl+C**: Emergency stop

## 🔧 Dependencies

### Required
- JAX/Flax (for policy inference)
- NumPy
- Booster Robotics SDK (for hardware)

### Optional
- `hidapi` (for gamepad control)
- `mujoco_playground` (for environment registration)

## 🚨 Safety Notes

1. **Always test in simulation first**
2. **Have emergency stop ready**
3. **Start with low control gains**
4. **Monitor robot behavior closely**

## 🐛 Troubleshooting

### Policy Loading Issues
- Check checkpoint path is correct
- Ensure environment is registered
- Verify JAX/Flax versions

### Gamepad Issues
- Install `hidapi`: `pip install hidapi`
- Check device permissions
- Verify gamepad is connected

### Hardware Issues
- Check Booster SDK installation
- Verify robot connection
- Check motor power

## 📊 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gamepad       │    │  JAX Policy      │    │  Booster SDK    │
│   (Joystick)    │───▶│  (Checkpoint)    │───▶│  (Hardware)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Velocity      │    │   Observations   │    │   Motor         │
│   Commands      │    │   (85 dim)       │    │   Commands      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎯 Next Steps

1. **Test with your checkpoint**
2. **Calibrate gamepad scaling**
3. **Tune motor gains**
4. **Add safety limits**
5. **Deploy to hardware!**


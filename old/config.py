"""
Configuration file for hardware deployment
"""

# Default checkpoint path
DEFAULT_CHECKPOINT_PATH = "/home/booster/Workspace/booster/checkpoint"

# Network settings
DEFAULT_ROBOT_IP = "192.168.10.102"  # Change to actual robot IP
DDS_DOMAIN_ID = 0

# Control parameters
CONTROL_FREQUENCY = 100  # Hz
CONTROL_DT = 1.0 / CONTROL_FREQUENCY
STATE_TIMEOUT = 0.1  # seconds

# Gamepad settings
GAMEPAD_VENDOR_ID = 0x046D  # Logitech
GAMEPAD_PRODUCT_ID = 0xC219  # F710
VEL_SCALE_X = 0.4
VEL_SCALE_Y = 0.4
VEL_SCALE_ROT = 1.0
GAMEPAD_DEADZONE = 0.01

# Robot parameters
T1_JOINT_COUNT = 23
T1_OBS_SIZE = 85

# Official motor control gains from booster_robotics_sdk examples
# These are the official gains used in the SDK examples

# Joint names for reference (23 joints total):
# 0: HeadYaw, 1: HeadPitch
# 2: LeftShoulderPitch, 3: LeftShoulderRoll, 4: LeftElbowPitch, 5: LeftElbowYaw
# 6: RightShoulderPitch, 7: RightShoulderRoll, 8: RightElbowPitch, 9: RightElbowYaw
# 10: Waist
# 11: LeftHipPitch, 12: LeftHipRoll, 13: LeftHipYaw, 14: LeftKneePitch, 15: CrankUpLeft, 16: CrankDownLeft
# 17: RightHipPitch, 18: RightHipRoll, 19: RightHipYaw, 20: RightKneePitch, 21: CrankUpRight, 22: CrankDownRight

# Official Kp gains from low_level_for_custom_publisher.py (SERIAL mode)
OFFICIAL_KP_GAINS = [
    5.0, 5.0,                    # Head joints
    40.0, 50.0, 20.0, 10.0,     # Left arm
    40.0, 50.0, 20.0, 10.0,     # Right arm
    100.0,                       # Waist
    350.0, 350.0, 180.0, 350.0, 450.0, 450.0,  # Left leg
    350.0, 350.0, 180.0, 350.0, 450.0, 450.0,  # Right leg
]

# Official Kd gains from low_level_for_custom_publisher.py (SERIAL mode)
OFFICIAL_KD_GAINS = [
    0.1, 0.1,                    # Head joints
    0.5, 1.5, 0.2, 0.2,         # Left arm
    0.5, 1.5, 0.2, 0.2,         # Right arm
    5.0,                         # Waist
    7.5, 7.5, 3.0, 5.5, 0.5, 0.5,  # Left leg
    7.5, 7.5, 3.0, 5.5, 0.5, 0.5,  # Right leg
]

# Alternative gains from b1_low_sdk_example.cpp (slightly different)
ALTERNATIVE_KP_GAINS = [
    5.0, 5.0,                    # Head joints
    40.0, 50.0, 20.0, 10.0,     # Left arm
    40.0, 50.0, 20.0, 10.0,     # Right arm
    100.0,                       # Waist
    350.0, 350.0, 180.0, 350.0, 550.0, 550.0,  # Left leg (higher crank gains)
    350.0, 350.0, 180.0, 350.0, 550.0, 550.0,  # Right leg (higher crank gains)
]

ALTERNATIVE_KD_GAINS = [
    0.1, 0.1,                    # Head joints
    0.5, 1.5, 0.2, 0.2,         # Left arm
    0.5, 1.5, 0.2, 0.2,         # Right arm
    5.0,                         # Waist
    7.5, 7.5, 3.0, 5.5, 1.5, 1.5,  # Left leg (higher crank gains)
    7.5, 7.5, 3.0, 5.5, 1.5, 1.5,  # Right leg (higher crank gains)
]

# Default gains (fallback)
DEFAULT_KP = 50.0   # Position gain (start low)
DEFAULT_KD = 2.0    # Velocity gain (start low)
DEFAULT_WEIGHT = 1.0  # Command weight

# Use official gains by default
USE_OFFICIAL_GAINS = True
GAIN_SET = "official"  # "official" or "alternative"

def get_joint_gains():
    """Get the appropriate gain set based on configuration."""
    if USE_OFFICIAL_GAINS:
        if GAIN_SET == "official":
            return OFFICIAL_KP_GAINS, OFFICIAL_KD_GAINS
        elif GAIN_SET == "alternative":
            return ALTERNATIVE_KP_GAINS, ALTERNATIVE_KD_GAINS
    
    # Fallback to uniform gains
    return [DEFAULT_KP] * T1_JOINT_COUNT, [DEFAULT_KD] * T1_JOINT_COUNT

# Safety limits
MAX_JOINT_VELOCITY = 10.0  # rad/s
MAX_JOINT_TORQUE = 50.0    # Nm
MAX_JOINT_POSITION_ERROR = 1.0  # rad

# Emergency stop settings
EMERGENCY_STOP_TIMEOUT = 0.5  # seconds
EMERGENCY_STOP_COMMAND = 'e'

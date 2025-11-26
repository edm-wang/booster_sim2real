import threading
import time
from typing import Optional
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ROSVelSubscriber:
    """ROS2 subscriber for velocity commands from geometry_msgs.msg.Twist."""

    def __init__(self, topic_name: str = "vel_cmd", timeout: float = 0.5):
        """
        Initialize ROS2 velocity subscriber.
        
        Args:
            topic_name: Name of the topic to subscribe to (default: "vel_cmd")
            timeout: Timeout in seconds before velocities default to zero (default: 0.5)
        """
        self.topic_name = topic_name
        self.timeout = timeout
        self._lock = threading.Lock()
        
        # Velocity values (default to zero)
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        
        # Track last message time (0 means no message received yet)
        self.last_message_time = 0.0
        
        # Initialize ROS2 if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create a minimal node for the subscriber
        self.node = Node('vel_cmd_subscriber')
        self.subscription = self.node.create_subscription(
            Twist,
            topic_name,
            self._vel_cmd_callback,
            10  # QoS depth
        )
        
        # Start ROS2 spinning in a separate thread
        self._running = True
        self.ros_thread = threading.Thread(target=self._spin_ros)
        self.ros_thread.daemon = True
        self.ros_thread.start()
        
        # Auto-start flags (for compatibility with RemoteControlService interface)
        self._auto_start_custom_mode = False
        self._auto_start_rl_gait = False
        
        # Set auto-start flags after a short delay
        threading.Timer(1.0, self._enable_auto_start).start()
    
    def _enable_auto_start(self):
        """Enable auto-start after initialization delay."""
        self._auto_start_custom_mode = True
        self._auto_start_rl_gait = True
    
    def _vel_cmd_callback(self, msg: Twist):
        """Callback for velocity command messages."""
        with self._lock:
            self.vx = msg.linear.x
            self.vy = msg.linear.y
            self.vyaw = msg.angular.z
            self.last_message_time = time.time()
    
    def _spin_ros(self):
        """Spin ROS2 node in a separate thread."""
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            # Check for timeout and reset velocities if needed
            current_time = time.time()
            with self._lock:
                if self.last_message_time > 0 and current_time - self.last_message_time > self.timeout:
                    self.vx = 0.0
                    self.vy = 0.0
                    self.vyaw = 0.0
    
    def get_vx_cmd(self) -> float:
        """Get forward velocity command."""
        with self._lock:
            # If no message received yet or timeout occurred, return 0
            if self.last_message_time == 0:
                return 0.0
            if time.time() - self.last_message_time > self.timeout:
                return 0.0
            return self.vx
    
    def get_vy_cmd(self) -> float:
        """Get lateral velocity command."""
        with self._lock:
            # If no message received yet or timeout occurred, return 0
            if self.last_message_time == 0:
                return 0.0
            if time.time() - self.last_message_time > self.timeout:
                return 0.0
            return self.vy
    
    def get_vyaw_cmd(self) -> float:
        """Get yaw velocity command."""
        with self._lock:
            # If no message received yet or timeout occurred, return 0
            if self.last_message_time == 0:
                return 0.0
            if time.time() - self.last_message_time > self.timeout:
                return 0.0
            return self.vyaw
    
    def start_custom_mode(self) -> bool:
        """Check if custom mode should start (auto-start enabled)."""
        return self._auto_start_custom_mode
    
    def start_rl_gait(self) -> bool:
        """Check if RL gait should start (auto-start enabled)."""
        return self._auto_start_rl_gait
    
    def get_operation_hint(self) -> str:
        """Get operation hint message."""
        return "Receiving velocity commands from ROS topic 'vel_cmd' (geometry_msgs/Twist)"
    
    def get_custom_mode_operation_hint(self) -> str:
        """Get custom mode operation hint."""
        return "Auto-starting custom mode..."
    
    def get_rl_gait_operation_hint(self) -> str:
        """Get RL gait operation hint."""
        return "Auto-starting RL gait..."
    
    def close(self):
        """Clean up resources."""
        self._running = False
        if hasattr(self, "node"):
            self.node.destroy_node()
        # Note: We don't shutdown rclpy here as it might be used by other components
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


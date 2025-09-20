import isaacgym
import threading
import time
import sys
import torch
from utils.runner import Runner


class SimpleKeyboardController:
    """Simple keyboard input handler that doesn't use termios"""
    
    def __init__(self):
        self.running = True
        self.commands = [0.0, 0.0, 0.0]  # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        self.max_lin_vel = 1.5
        self.max_ang_vel = 1.5
        self.command_step = 0.2
        
        print("\n=== Keyboard Controls ===")
        print("W/S: Forward/Backward")
        print("A/D: Left/Right")
        print("Q/E: Turn Left/Right")
        print("SPACE: Stop")
        print("ESC: Exit")
        print("========================\n")
        print("Press Enter after typing commands, or type 'quit' to exit")
    
    def get_user_input(self):
        """Get user input in a simple way"""
        try:
            user_input = input("Enter command (w/s/a/d/q/e/space/quit): ").strip().lower()
            
            if user_input == 'quit':
                self.running = False
                return
            
            # Process multiple commands
            for char in user_input:
                if char == 'w':
                    self.commands[0] = min(self.commands[0] + self.command_step, self.max_lin_vel)
                elif char == 's':
                    self.commands[0] = max(self.commands[0] - self.command_step, -self.max_lin_vel)
                elif char == 'a':
                    self.commands[1] = min(self.commands[1] + self.command_step, self.max_lin_vel)
                elif char == 'd':
                    self.commands[1] = max(self.commands[1] - self.command_step, -self.max_lin_vel)
                elif char == 'q':
                    self.commands[2] = min(self.commands[2] + self.command_step, self.max_ang_vel)
                elif char == 'e':
                    self.commands[2] = max(self.commands[2] - self.command_step, -self.max_ang_vel)
                elif char == ' ':
                    self.commands = [0.0, 0.0, 0.0]
            
            print(f"Commands: lin_x={self.commands[0]:.2f}, lin_y={self.commands[1]:.2f}, ang_yaw={self.commands[2]:.2f}")
            
        except (EOFError, KeyboardInterrupt):
            self.running = False
    
    def get_commands(self):
        """Get current command values"""
        return self.commands.copy()


class SingleRobotRunner(Runner):
    """Runner class that spawns only 1 robot for interactive control"""
    
    def __init__(self, test=True):
        # Override the argument parsing to force num_envs=1
        import sys
        original_argv = sys.argv.copy()
        
        # Add --num_envs 1 to the arguments if not already present
        if '--num_envs' not in sys.argv:
            sys.argv.extend(['--num_envs', '1'])
        
        try:
            super().__init__(test=test)
        finally:
            # Restore original argv
            sys.argv = original_argv
        
        self.keyboard_controller = SimpleKeyboardController()
        
        # Override command resampling to use keyboard input
        self.env._resample_commands = self._keyboard_resample_commands
    
    def _keyboard_resample_commands(self):
        """Override the default command resampling with keyboard input"""
        # Get commands from keyboard controller
        commands = self.keyboard_controller.get_commands()
        
        # Set commands for the single environment
        self.env.commands[0, 0] = commands[0]  # lin_vel_x
        self.env.commands[0, 1] = commands[1]  # lin_vel_y  
        self.env.commands[0, 2] = commands[2]  # ang_vel_yaw
        
        # Set gait frequency based on command magnitude
        command_magnitude = (commands[0]**2 + commands[1]**2 + commands[2]**2)**0.5
        if command_magnitude > 0.1:
            self.env.gait_frequency[0] = 2.0  # Active gait
        else:
            self.env.gait_frequency[0] = 0.0  # Standing still
    
    def play(self):
        """Interactive play with keyboard control - single robot"""
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        
        if self.cfg["viewer"]["record_video"]:
            import os
            import imageio
            import signal
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        
        print("Starting interactive control with 1 robot...")
        print("The simulation will run continuously. Use the input prompt to control the robot!")
        
        # Start a thread for user input
        input_thread = threading.Thread(target=self._input_loop, daemon=True)
        input_thread.start()
        
        try:
            step_count = 0
            while self.keyboard_controller.running:
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.loc
                    obs, rew, done, infos = self.env.step(act)
                    obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                
                if self.cfg["viewer"]["record_video"]:
                    record_time -= self.env.dt
                    if record_time < 0:
                        record_time += self.cfg["viewer"]["record_interval"]
                        self.interrupt = False
                        signal.signal(signal.SIGINT, self.interrupt_handler)
                        with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                            for frame in self.env.camera_frames:
                                self.writer.append_data(frame)
                        if self.interrupt:
                            raise KeyboardInterrupt
                        signal.signal(signal.SIGINT, signal.default_int_handler)
                
                step_count += 1
                if step_count % 100 == 0:  # Print status every 100 steps
                    print(f"Step {step_count}, Commands: lin_x={self.keyboard_controller.commands[0]:.2f}, lin_y={self.keyboard_controller.commands[1]:.2f}, ang_yaw={self.keyboard_controller.commands[2]:.2f}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.keyboard_controller.running = False
    
    def _input_loop(self):
        """Input loop running in a separate thread"""
        while self.keyboard_controller.running:
            try:
                self.keyboard_controller.get_user_input()
            except:
                break


if __name__ == "__main__":
    runner = SingleRobotRunner(test=True)
    runner.play()

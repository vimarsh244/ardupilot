import gymnasium as gym
import numpy as np
import time
import subprocess
import math
import pymavlink.mavutil as mavutil
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any

class ArduPilotSITLEnv(gym.Env):
    """
    Gymnasium Environment for ArduPilot SITL simulator to optimize copter parameters using RL.
    
    This environment should allow reinforcement learning algorithms to find optimal parameter sets
    for ArduPilot-based copters by interacting with the SITL simulator.
    """
    
    def __init__(self, 
                 frame_type: str = "quad",
                 vehicle: str = "copter",
                 param_ranges: Dict[str, Tuple[float, float]] = None,
                 mission_file: str = None,
                 sitl_port: int = 5760,
                 max_episode_steps: int = 1000,
                 reward_weights: Dict[str, float] = None):
        """
        Initialize the ArduPilot SITL environment.
        
        Args:
            frame_type: Type of vehicle frame (e.g., "quad", "hexa", "octa")
            vehicle: ArduPilot vehicle type (e.g., "copter", "plane")
            param_ranges: Dictionary mapping parameter names to their min/max value ranges
            mission_file: Path to a mission file to load for evaluation
            sitl_port: SITL connection port : default 5760, but if having multiple environments have a way to dynamically select it
            max_episode_steps: Maximum steps per episode
            reward_weights: Dictionary with weights for different components of the reward function
        """
        super(ArduPilotSITLEnv, self).__init__()
        
        self.frame_type = frame_type
        self.vehicle = vehicle
        self.sitl_port = sitl_port
        self.max_episode_steps = max_episode_steps
        self.mission_file = mission_file
        self.sitl_process = None
        self.mavlink_connection = None
        self.steps = 0
        self.mavlink_port = self.sitl_port + 2
        
        #default parameter ranges if none specified
        self.param_ranges = param_ranges or {
            "PSC_POSXY_P": (0.5, 2.0),
            "PSC_VELXY_P": (0.5, 2.0),
            "PSC_VELXY_I": (0.0, 1.0),
            "PSC_VELXY_D": (0.0, 0.5),
            "PSC_POSZ_P": (0.5, 3.0),
            "PSC_VELZ_P": (1.0, 8.0),
            "PSC_VELZ_I": (0.0, 3.0),
            "INS_ACCEL_FILTER": (5.0, 20.0),
            "INS_GYRO_FILTER": (5.0, 20.0),
            "MOT_THST_HOVER": (0.1, 0.6)
        }
        
        # Default reward weights if none specified
        self.reward_weights = reward_weights or {
            "position_error": -1.0,
            "attitude_error": -1.0,
            "velocity_error": -1.0,
            "power_consumption": -0.5,
            "mission_completion": 5.0,
            "stability": 2.0
        }
        
        # Define action space (one dimension per parameter)
        self.param_keys = list(self.param_ranges.keys())
        self.action_space = spaces.Box(
            low=np.array([0.0] * len(self.param_ranges)),
            high=np.array([1.0] * len(self.param_ranges)),
            dtype=np.float32
        )
        
        # Define observation space
        # Includes telemetry data and current parameter values
        # We'll track position error, attitude error, velocity, battery status, etc.
        num_obs = 15 + len(self.param_ranges)  # 15 basic telemetry values + parameters
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float32
        )
        
        # to track best parameters and performance
        self.best_reward = -np.inf
        self.best_params = {}
        
        # Initialize mission metrics
        self.mission_waypoints = []
        self.current_waypoint_idx = 0
        self.mission_started = False
        self.mission_completed = False
        
        # Telemetry history for evaluating stability
        self.telemetry_history = []
        self.history_length = 50  # Store last 50 telemetry points
    
    def start_sitl(self):
        """Start the SITL simulator as a subprocess."""
        cmd = [
            "sim_vehicle.py",
            "-v", self.vehicle,
            "-f", self.frame_type,
            "--console",
            # "--home 15.3911, 73.8781,0,0",
            # "--custom-location=15.3911,73.8781,0,0", # bits goa map
            "--custom-location=0,0,0,0",
            "--map",
            "-I0",
            "--no-extra-ports"
        ]
        
        self.sitl_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        print("SITL simulator started")
        # Wait for SITL to initialize
        time.sleep(30)
        print(self.sitl_process.stdout.readline())
        # Connect to the simulator via MAVLink
        self.connect_mavlink()

        
        # If mission file provided, load it
        if self.mission_file:
            self.load_mission()
    
    def connect_mavlink(self):
        """Establish MAVLink connection to SITL."""
        print(f"Connecting to SITL on port {self.mavlink_port}")
        connection_string = f"tcp:127.0.0.1:{self.mavlink_port}"
        try:
            
            self.mavlink_connection = mavutil.mavlink_connection(connection_string)
            self.mavlink_connection.wait_heartbeat()
            print("MAVLink connection established")
        except Exception as e:
            print(f"Failed to connect to SITL: {e}")
            self.close()
            raise
    
    def load_mission(self):
        """Load a mission from file."""
        # This should ideally load a mission file from some txt file or something but for now manually setting the waypoints
        self.mission_waypoints = [
            {"lat": 0.0, "lon": 0.0, "alt": 10.0},
            {"lat": 0.0001, "lon": 0.0, "alt": 15.0},
            {"lat": 0.0001, "lon": 0.0001, "alt": 15.0},
            {"lat": 0.0, "lon": 0.0001, "alt": 10.0},
            {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        ]
        print(f"Loaded mission with {len(self.mission_waypoints)} waypoints")
    
    def set_parameters(self, params_dict):
        """Set parameters in the SITL simulator."""
        for param_name, param_value in params_dict.items():
            msg = self.mavlink_connection.mav.param_set_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                param_name.encode('utf-8'),
                float(param_value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
            
            # Wait for ack
            start_time = time.time()
            while time.time() - start_time < 5:
                ack = self.mavlink_connection.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
                if ack is not None and ack.param_id == param_name:
                    break
            else:
                print(f"Warning: No acknowledgement received for parameter {param_name}")
    
    def get_parameters(self):
        """Get current parameter values from SITL."""
        param_values = {}
        
        for param_name in self.param_ranges.keys():
            self.mavlink_connection.mav.param_request_read_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                param_name.encode('utf-8'),
                -1  # -1 for get by name
            )
            
            start_time = time.time()
            while time.time() - start_time < 5:
                param = self.mavlink_connection.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
                if param is not None and param.param_id == param_name:
                    param_values[param_name] = param.param_value
                    break
            else:
                print(f"Warning: Failed to get parameter {param_name}")
                param_values[param_name] = 0.0
        
        return param_values
    
    def get_normalized_parameters(self):
        """Get parameters normalized to [0, 1]."""
        params = self.get_parameters()
        normalized_params = {}
        
        for param_name, param_value in params.items():
            min_val, max_val = self.param_ranges[param_name]
            normalized_params[param_name] = (param_value - min_val) / (max_val - min_val)
        
        return normalized_params
    
    def convert_actions_to_parameters(self, actions):
        """Convert normalized [0,1] actions to actual parameter values."""
        params_dict = {}
        
        for i, param_name in enumerate(self.param_keys):
            min_val, max_val = self.param_ranges[param_name]
            actual_value = min_val + actions[i] * (max_val - min_val)
            params_dict[param_name] = actual_value
        
        return params_dict
    
    def get_telemetry(self):
        """Get current telemetry data from the vehicle."""
        telemetry = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
            "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "battery": {"voltage": 0.0, "current": 0.0, "remaining": 0.0},
            "status": {"armed": False, "mode": ""}
        }
        
        # Request data from SITL
        msgs = []
        start_time = time.time()
        while time.time() - start_time < 0.5:
            msg = self.mavlink_connection.recv_match(blocking=True, timeout=0.1)
            if msg is not None:
                msgs.append(msg)
        
        # Parse messages
        for msg in msgs:
            msg_type = msg.get_type()
            
            if msg_type == "LOCAL_POSITION_NED":
                telemetry["position"]["x"] = msg.x
                telemetry["position"]["y"] = msg.y
                telemetry["position"]["z"] = msg.z
                telemetry["velocity"]["vx"] = msg.vx
                telemetry["velocity"]["vy"] = msg.vy
                telemetry["velocity"]["vz"] = msg.vz
            
            elif msg_type == "ATTITUDE":
                telemetry["attitude"]["roll"] = msg.roll
                telemetry["attitude"]["pitch"] = msg.pitch
                telemetry["attitude"]["yaw"] = msg.yaw
            
            elif msg_type == "SYS_STATUS":
                telemetry["battery"]["voltage"] = msg.voltage_battery / 1000.0  # Convert to volts
                telemetry["battery"]["current"] = msg.current_battery / 100.0   # Convert to amps
                telemetry["battery"]["remaining"] = msg.battery_remaining
            
            elif msg_type == "HEARTBEAT":
                telemetry["status"]["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                telemetry["status"]["mode"] = mavutil.mode_string_v10(msg)
        
        # Add to history for stability calculation
        self.telemetry_history.append(telemetry)
        if len(self.telemetry_history) > self.history_length:
            self.telemetry_history.pop(0)
        
        return telemetry
    
    def get_parameters(self, param_name):
        """Get parameter value from the vehicle."""
        # This is a stub that should be implemented to retrieve actual parameter values 
        # from the MAVLink connection for now just returning
        #  a default value
        return 0.5
    
    def flatten_observation(self, telemetry, normalized_params):
        obs = [
            telemetry["position"]["x"],
            telemetry["position"]["y"],
            telemetry["position"]["z"],
            telemetry["velocity"]["vx"],
            telemetry["velocity"]["vy"],
            telemetry["velocity"]["vz"],
            telemetry["attitude"]["roll"],
            telemetry["attitude"]["pitch"],
            telemetry["attitude"]["yaw"],
            telemetry["battery"]["voltage"],
            telemetry["battery"]["current"],
            telemetry["battery"]["remaining"],
            1.0 if telemetry["status"]["armed"] else 0.0,
            float(self.current_waypoint_idx) / max(1, len(self.mission_waypoints)),
            1.0 if self.mission_completed else 0.0
        ]
        
        # Add normalized parameters
        # for param_name in self.param_keys:
            
        
        # return np.array(obs, dtype=np.float32)
        #    obs.append(normalized_params[param_name])
        
        return np.array(obs, dtype=np.float32)
    
    def calculate_position_error(self, telemetry):
        """Calculate error between current position and target waypoint."""
        if not self.mission_waypoints or self.current_waypoint_idx >= len(self.mission_waypoints):
            return 0.0
        
        target = self.mission_waypoints[self.current_waypoint_idx]
        
        # Simple Euclidean distance for now
        # In a real implementation, have to use and do coordinate conversions
        error = math.sqrt(
            (telemetry["position"]["x"] - target["lat"] * 1e5) ** 2 +
            (telemetry["position"]["y"] - target["lon"] * 1e5) ** 2 +
            (telemetry["position"]["z"] - target["alt"]) ** 2
        )
        
        return error
    
    def calculate_stability(self):
        """Calculate stability based on telemetry history."""
        if len(self.telemetry_history) < 10:
            return 0.0
        
        # Calculate variance of roll, pitch rates
        roll_rates = []
        pitch_rates = []
        
        for i in range(1, len(self.telemetry_history)):
            prev = self.telemetry_history[i-1]
            curr = self.telemetry_history[i]
            dt = 0.1  # Assuming constant time step
            
            roll_rate = (curr["attitude"]["roll"] - prev["attitude"]["roll"]) / dt
            pitch_rate = (curr["attitude"]["pitch"] - prev["attitude"]["pitch"]) / dt
            
            roll_rates.append(roll_rate)
            pitch_rates.append(pitch_rate)
        
        roll_var = np.var(roll_rates)
        pitch_var = np.var(pitch_rates)
        
        # Lower variance means more stable
        stability_score = 1.0 / (1.0 + roll_var + pitch_var)
        return stability_score
    
    def calculate_power_efficiency(self, telemetry):
        """Calculate power efficiency score."""
        current = telemetry["battery"]["current"]
        voltage = telemetry["battery"]["voltage"]
        
        if current <= 0.0 or voltage <= 0.0:
            return 0.0
        
        # Calculate power
        power = current * voltage
        
        # Calculate efficiency (lower power use is better)
        # Normalize to a reasonable range
        power_efficiency = 1.0 / (1.0 + power / 100.0)
        
        return power_efficiency
    
    def calculate_reward(self, telemetry):
        """Calculate reward based on multiple factors."""
        position_error = self.calculate_position_error(telemetry)
        stability = self.calculate_stability()
        power_efficiency = self.calculate_power_efficiency(telemetry)
        
        # Calculate individual reward components
        position_reward = self.reward_weights["position_error"] * position_error
        stability_reward = self.reward_weights["stability"] * stability
        power_reward = self.reward_weights["power_consumption"] * (1.0 - power_efficiency)
        
        mission_reward = 0.0
        if self.mission_completed:
            mission_reward = self.reward_weights["mission_completion"]
        
        # Sum up the components
        total_reward = position_reward + stability_reward + power_reward + mission_reward
        
        return total_reward
    
    def check_mission_progress(self, telemetry):
        """Check if we've reached the current waypoint and update mission progress."""
        if not self.mission_waypoints or self.current_waypoint_idx >= len(self.mission_waypoints):
            return
        
        # Check distance to current waypoint
        target = self.mission_waypoints[self.current_waypoint_idx]
        
        distance = math.sqrt(
            (telemetry["position"]["x"] - target["lat"] * 1e5) ** 2 +
            (telemetry["position"]["y"] - target["lon"] * 1e5) ** 2 +
            (telemetry["position"]["z"] - target["alt"]) ** 2
        )
        
        # If close enough to waypoint, advance to next
        if distance < 2.0:  # 2 meter threshold
            self.current_waypoint_idx += 1
            
            # Check if mission completed
            if self.current_waypoint_idx >= len(self.mission_waypoints):
                self.mission_completed = True
    
    def wait_for_position(self):
        """Wait for the vehicle to get a valid position."""
        print("Waiting for valid GPS position...")
        max_wait_time = 30  # Maximum time to wait in seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check for GPS_RAW_INT which has the fix_type field
            gps_msg = self.mavlink_connection.recv_match(type='GPS_RAW_INT', blocking=False)
            if gps_msg and gps_msg.fix_type >= 3:  # 3D fix or better
                print(f"Valid GPS fix received (fix_type: {gps_msg.fix_type})")
                return True
                
            # Also check position messages as a backup
            pos_msg = self.mavlink_connection.recv_match(type=['GLOBAL_POSITION_INT', 'LOCAL_POSITION_NED'], blocking=True, timeout=1)
            if pos_msg and (pos_msg.get_type() == 'GLOBAL_POSITION_INT' or pos_msg.get_type() == 'LOCAL_POSITION_NED'):
                if pos_msg.get_type() == 'GLOBAL_POSITION_INT' and pos_msg.lat != 0 and pos_msg.lon != 0:
                    print("Valid global position received")
                    return True
                elif pos_msg.get_type() == 'LOCAL_POSITION_NED' and (pos_msg.x != 0 or pos_msg.y != 0):
                    print("Valid local position received")
                    return True
            time.sleep(0.1)
        
        print("Failed to get valid position within timeout")
        return False
        
        
    def reset(self, **kwargs):
        """Reset the environment to initial state."""
        # If SITL is running, stop it
        if self.sitl_process:
            self.close()
        
        # Start SITL
        self.start_sitl()
        
        # Reset mission state
        self.current_waypoint_idx = 0
        self.mission_started = False
        self.mission_completed = False
        self.telemetry_history = []
        self.steps = 0
        
        # Get initial telemetry
        telemetry = self.get_telemetry()
        normalized_params = self.get_normalized_parameters()
        observation = self.flatten_observation(telemetry, normalized_params)
        
        # return observation, {}
        # If not already armed and in guided mode, do so
        if not self.mission_started:
            # Wait for GPS position to be valid
            self.wait_for_position()
            
            # Send arm command
            self.mavlink_connection.arducopter_arm()
            time.sleep(1)
            
            # First set mode to GUIDED to ensure position is being used
            self.mavlink_connection.set_mode("GUIDED")
            time.sleep(2)
            
            # Then set mode to AUTO for mission execution
            self.mavlink_connection.set_mode_auto()
            time.sleep(3)
            
            # Start mission
            self.mavlink_connection.mav.mission_set_current_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                0
            )
            self.mission_started = True
            time.sleep(1)
            time.sleep(1)
            
            # Set mode to GUIDED
            self.mavlink_connection.set_mode_auto()
            time.sleep(5)
            
            # Start mission
            self.mavlink_connection.mav.mission_set_current_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                0
            )
            # self.mavlink_connection.mav.set_mode_send(
            #     self.mavlink_connection.target_system,
            #     mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            #     mavutil.mavlink.MAV_MODE_GUIDED
            # )
            # self.mavlink_connection.waypoint_current_send(0)

            # requires some specific fixing at `mavutils.py`
               # set mode by integer mode number for ArduPilot
                    # # self.mav.command_long_send(self.target_system,
                    # #                            self.target_component,
                    # #                            mavlink.MAV_CMD_DO_SET_MODE,
                    # #                            0,
                    # #                            mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    # #                            mode,
                    # #                            0,
                    # #                            0,
                    # #                            0,
                    # #                            0,
                    # #                            0)
                    # self.mav.set_mode_send(self.target_system, mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode)
            # this has to be modified because on newer python versions the older mavutils is giving issues


            self.mission_started = True
            time.sleep(1)
        
        # Get telemetry data
        telemetry = self.get_telemetry()
        
        # Check mission progress
        self.check_mission_progress(telemetry)
        
        # Calculate reward
        reward = self.calculate_reward(telemetry)
        
        # Get current parameters
        params_dict = {param_name: self.get_parameters(param_name) for param_name in self.param_keys}
        
        # Check if current parameters are better than best known
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params_dict.copy()
        
        # Get normalized parameter values
        normalized_params = {param_name: action[i] for i, param_name in enumerate(self.param_keys)}
        
        # Create observation
        observation = self.flatten_observation(telemetry, normalized_params)
        
        # Check if episode is done
        terminated = self.mission_completed
        truncated = self.steps >= self.max_episode_steps
        
        # Prepare info dict
        info = {
            "mission_progress": self.current_waypoint_idx / max(1, len(self.mission_waypoints)),
            "mission_completed": self.mission_completed,
            "position_error": self.calculate_position_error(telemetry),
            "stability": self.calculate_stability(),
            "power_efficiency": self.calculate_power_efficiency(telemetry),
            "current_params": params_dict,
            "best_params": self.best_params,
            "best_reward": self.best_reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Clean up resources."""
        if self.mavlink_connection:
            try:
                # Disarm if armed
                self.mavlink_connection.arducopter_disarm()
                time.sleep(0.5)
                self.mavlink_connection.close()
            except:
                pass
            self.mavlink_connection = None
        
        if self.sitl_process:
            try:
                self.sitl_process.terminate()
                self.sitl_process.wait(timeout=5)
            except:
                self.sitl_process.kill()
            self.sitl_process = None


#Sample Example usage
if __name__ == "__main__":
    # Create the environment
    env = ArduPilotSITLEnv(
        frame_type="quad",
        vehicle="copter",
        param_ranges={
            "PSC_POSXY_P": (0.5, 2.0),
            "PSC_VELXY_P": (0.5, 2.0),
            "PSC_VELXY_I": (0.0, 1.0),
            "PSC_VELXY_D": (0.0, 0.5),
            "PSC_POSZ_P": (0.5, 3.0),
        }
    )
    
    # Reset the environment
    observation, _ = env.reset()
    
    # Run a simple random action test
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
        print(f"Mission Progress: {info['mission_progress']:.2f}")
        
        if terminated or truncated:
            break
    
    # Print the best parameters found
    print("\nBest Parameters:")
    for param_name, param_value in env.best_params.items():
        print(f"{param_name}: {param_value:.4f}")
    
    # Close the environment
    env.close()
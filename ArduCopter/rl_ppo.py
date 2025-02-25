import os
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import json
import matplotlib.pyplot as plt
from datetime import datetime

#importing the custom env made
from gym import ArduPilotSITLEnv  

class SaveBestParamsCallback(BaseCallback):
    """
    Callback for saving the best parameters found during training.
    """
    def __init__(self, eval_env, log_dir, param_ranges, verbose=1):
        super(SaveBestParamsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.param_ranges = param_ranges
        self.best_mean_reward = -np.inf
        self.best_params = {}
        self.param_history = []
        self.reward_history = []
        
        #logging directory
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        #getting current mean reward from parent callback
        if self.parent is not None and hasattr(self.parent, "best_mean_reward"):
            mean_reward = self.parent.best_mean_reward
            
            #to check if this is the best model so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                #extracting the best parameters from the environment
                env = self._get_eval_env()
                if hasattr(env, "best_params") and env.best_params:
                    self.best_params = env.best_params
                    
                    #logging the best parameters
                    self._log_best_params()
                    
                    #plotting progress
                    self._plot_progress()
        
        return True
    
    def _get_eval_env(self):
        """Get the unwrapped evaluation environment."""
        env = self.eval_env
        
        # Unwrap DummyVecEnv
        if isinstance(env, DummyVecEnv) or isinstance(env, SubprocVecEnv):
            env = env.envs[0]
        
        # Unwrap Monitor
        if isinstance(env, Monitor):
            env = env.env
        
        return env
    
    def _log_best_params(self):
        """Save the best parameters to a file."""
        if not self.best_params:
            return
        
        # save params at enf of cycle to JSON file
        params_file = os.path.join(self.log_dir, "best_params.json")
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        if self.verbose > 0:
            print(f"Saved best parameters to {params_file}")
            print("Best Parameters:")
            for param_name, param_value in self.best_params.items():
                print(f"  {param_name}: {param_value:.4f}")
    
    def _record_params(self, step):
        """Record parameters for history."""
        if not self.best_params:
            return
        
        param_record = {
            "step": step,
            "reward": self.best_mean_reward,
            "params": self.best_params.copy()
        }
        
        self.param_history.append(param_record)
        self.reward_history.append(self.best_mean_reward)
        
        # Save history to JSON file
        history_file = os.path.join(self.log_dir, "param_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.param_history, f, indent=4)
    
    def _plot_progress(self):
        """Plot the training progress."""
        if len(self.reward_history) < 2:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot reward progress
        plt.subplot(1, 2, 1)
        plt.plot(self.reward_history)
        plt.xlabel('Optimization Step')
        plt.ylabel('Mean Reward')
        plt.title('Training Progress')
        
        # Plot parameter evolution
        plt.subplot(1, 2, 2)
        for param_name in self.best_params.keys():
            param_values = [record['params'].get(param_name, 0) for record in self.param_history]
            plt.plot(param_values, label=param_name)
        
        plt.xlabel('Optimization Step')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Evolution')
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_progress.png"))
        plt.close()

def make_env(frame_type, vehicle, param_ranges, mission_file, seed=0):
    """
    Factory function for creating environments with seed setting.
    """
    def _init():
        env = ArduPilotSITLEnv(
            frame_type=frame_type,
            vehicle=vehicle,
            param_ranges=param_ranges,
            mission_file=mission_file
        )
        env = Monitor(env)
        env.seed(seed)
        return env
    
    set_random_seed(seed)
    return _init

def train_ppo_for_parameter_optimization(
    frame_type="quad",
    vehicle="copter",
    param_ranges=None,
    mission_file=None,
    n_envs=1,
    total_timesteps=100000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=4,
    n_epochs=10,
    gamma=0.99,
    seed=0,
    log_dir=None,
    eval_freq=5000,
):
    """
    Train a PPO agent to find optimal parameters for an ArduPilot vehicle.
    
    Args:
        frame_type: Type of vehicle frame (quad, hexa, etc.)
        vehicle: ArduPilot vehicle type (copter, plane, etc.)
        param_ranges: Dictionary of parameter ranges to optimize
        mission_file: Path to a mission file for evaluation
        n_envs: Number of parallel environments
        total_timesteps: Total timesteps for training
        learning_rate: Learning rate for PPO
        n_steps: Steps per environment per update
        batch_size: Batch size for training
        n_epochs: Number of epochs for each update
        gamma: Discount factor
        seed: Random seed
        log_dir: Directory to save logs and models
        eval_freq: Frequency of evaluation in timesteps
        
    Returns:
        Trained PPO model and dictionary of best parameters
    """
    # Default parameter ranges
    if param_ranges is None:
        param_ranges = {
            "PSC_POSXY_P": (0.5, 2.0),
            "PSC_VELXY_P": (0.5, 2.0),
            "PSC_VELXY_I": (0.0, 1.0),
            "PSC_VELXY_D": (0.0, 0.5),
            "PSC_POSZ_P": (0.5, 3.0),
            "PSC_VELZ_P": (1.0, 8.0),
            "PSC_VELZ_I": (0.0, 3.0),
        }
    
    #setting up log directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"./logs/ardupilot_ppo_{timestamp}"
    
    #creating vectorized environment for training
    vec_env = make_vec_env(
        lambda: ArduPilotSITLEnv(
            frame_type=frame_type,
            vehicle=vehicle,
            param_ranges=param_ranges,
            mission_file=mission_file
        ),
        n_envs=n_envs, # the onyl big issue here is that because this is ardupilot SITL
        # essentially cant have more than one environemnt in parallel, technically can have if running different ports and all
        # but way too complex and still limited as it is not a good sim
        seed=seed,
        monitor_dir=os.path.join(log_dir, "train_monitor"),
    )
    
    #evaluation environment
    eval_env = Monitor(
        ArduPilotSITLEnv(
            frame_type=frame_type,
            vehicle=vehicle,
            param_ranges=param_ranges,
            mission_file=mission_file
        ),
        os.path.join(log_dir, "eval_monitor")
    )
    
    #callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False
    )
    
    param_callback = SaveBestParamsCallback(
        eval_env=eval_env,
        log_dir=log_dir,
        param_ranges=param_ranges,
        verbose=1
    )
    
    #creatuing PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard")
    )
    
    #training model
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, param_callback],
        tb_log_name="ppo_ardupilot"
    )
    
    #save model
    model.save(os.path.join(log_dir, "final_model"))
    
    #best parameters
    best_params = param_callback.best_params
    
    print(f"Training completed. Best reward: {param_callback.best_mean_reward:.2f}")
    print("Best parameters:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value:.4f}")
    
    return model, best_params

def visualize_and_evaluate_best_params(
    best_params,
    frame_type="quad",
    vehicle="copter",
    param_ranges=None,
    mission_file=None
):
    """
    Create a visualization of the learned parameters and evaluate their performance.
    
    Args:
        best_params: Dictionary of best parameters
        frame_type: Vehicle frame type
        vehicle: ArduPilot vehicle type
        param_ranges: Parameter ranges for normalization
        mission_file: Mission file for evaluation
    """
    # Create environment
    env = ArduPilotSITLEnv(
        frame_type=frame_type,
        vehicle=vehicle,
        param_ranges=param_ranges,
        mission_file=mission_file
    )
    
    # Reset the environment
    observation, _ = env.reset()
    
    # Prepare the action (convert parameters to normalized actions)
    action = []
    for param_name in env.param_keys:
        min_val, max_val = env.param_ranges[param_name]
        normalized_value = (best_params[param_name] - min_val) / (max_val - min_val)
        action.append(normalized_value)
    
    # Run an evaluation episode
    total_reward = 0
    metrics = {
        "position_error": [],
        "stability": [],
        "power_efficiency": [],
        "rewards": []
    }
    
    print("Starting evaluation of best parameters...")
    
    for step in range(1000):  # Run for 1000 steps max
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Collect metrics
        metrics["position_error"].append(info["position_error"])
        metrics["stability"].append(info["stability"])
        metrics["power_efficiency"].append(info["power_efficiency"])
        metrics["rewards"].append(reward)
        
        print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
        print(f"Position error: {info['position_error']:.2f}, Stability: {info['stability']:.2f}")
        
        if terminated or truncated:
            break
    
    # Close the environment
    env.close()
    
    # Plot metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title("Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics["position_error"])
    plt.title("Position Error")
    plt.xlabel("Step")
    plt.ylabel("Error")
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics["stability"])
    plt.title("Stability")
    plt.xlabel("Step")
    plt.ylabel("Stability Score")
    
    plt.subplot(2, 2, 4)
    plt.plot(metrics["power_efficiency"])
    plt.title("Power Efficiency")
    plt.xlabel("Step")
    plt.ylabel("Efficiency Score")
    
    plt.tight_layout()
    plt.savefig("parameter_evaluation_metrics.png")
    plt.show()
    
    print(f"Evaluation completed. Total reward: {total_reward:.2f}")
    print("Evaluation results saved to parameter_evaluation_metrics.png")
    
    return metrics

def main():
    """
    Main function to run the parameter optimization.
    """
    # Define parameter ranges to optimize
    param_ranges = {
        "PSC_POSXY_P": (0.5, 2.0),
        "PSC_VELXY_P": (0.5, 2.0),
        "PSC_VELXY_I": (0.0, 1.0),
        "PSC_VELXY_D": (0.0, 0.5),
        "PSC_POSZ_P": (0.5, 3.0),
        "PSC_VELZ_P": (1.0, 8.0),
        "PSC_VELZ_I": (0.0, 3.0),
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ardupilot_ppo_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save the parameter ranges
    with open(os.path.join(log_dir, "param_ranges.json"), 'w') as f:
        json.dump(param_ranges, f, indent=4)
    
    # training using PPO
    model, best_params = train_ppo_for_parameter_optimization(
        frame_type="quad",
        vehicle="copter",
        param_ranges=param_ranges,
        mission_file=None,  # Use default mission
        n_envs=1,           # Single environment due to SITL
        total_timesteps=5000,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=4,
        n_epochs=10,
        gamma=0.99,
        seed=42,
        log_dir=log_dir,
        eval_freq=500
    )
    
    #getting best parameters
    metrics = visualize_and_evaluate_best_params(
        best_params=best_params,
        frame_type="quad",
        vehicle="copter",
        param_ranges=param_ranges,
        mission_file=None  # Use default mission
    )
    
    # saving the best parameters with explanation
    with open(os.path.join(log_dir, "optimized_parameters.txt"), 'w') as f:
        f.write("# ArduPilot Optimized Parameters\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total training steps: 50000\n")
        f.write(f"# Final evaluation reward: {sum(metrics['rewards']):.2f}\n\n")
        
        f.write("# Parameter Name, Value, Description\n")
        
        descriptions = {
            "PSC_POSXY_P": "Position control P gain for horizontal position error",
            "PSC_VELXY_P": "Velocity control P gain for horizontal velocity error",
            "PSC_VELXY_I": "Velocity control I gain for horizontal velocity error",
            "PSC_VELXY_D": "Velocity control D gain for horizontal velocity error",
            "PSC_POSZ_P": "Position control P gain for vertical position error",
            "PSC_VELZ_P": "Velocity control P gain for vertical velocity error",
            "PSC_VELZ_I": "Velocity control I gain for vertical velocity error"
        }
        
        for param_name, param_value in best_params.items():
            desc = descriptions.get(param_name, "Unknown parameter")
            f.write(f"{param_name}, {param_value:.4f}, {desc}\n")
    
    print(f"All results saved to {log_dir}")
    
    return model, best_params, metrics

if __name__ == "__main__":
    main()
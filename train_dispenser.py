#!/usr/bin/env python3
"""
Training Script for Satellite Dispenser RL Agent - CORRECTED with CUDA
Uses Stable Baselines3 PPO algorithm
"""
import torch  
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import os
from datetime import datetime

# Import our custom environment
from dispenser_env import SatelliteDispenserEnv


def make_env():
    """Create and wrap the environment"""
    env = SatelliteDispenserEnv()
    env = Monitor(env)  # Track episode statistics
    return env


def train_model(total_timesteps=1000000, save_dir="./models/"):
    """
    Train the RL agent using PPO

    Args:
        total_timesteps: Number of training steps (default: 1M for better learning)
        save_dir: Directory to save models
    """
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "best_model/", exist_ok=True)
    os.makedirs(save_dir + "logs/", exist_ok=True)

    # Use 8 parallel environments for faster training
    n_envs = 8
    
    # PPO hyperparameters
    n_steps = 2048  # Steps per environment before update
    batch_size = 256
    
    # Check CUDA availability and set device
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_info = f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)"
    else:
        device = 'cpu'
        cuda_info = "CUDA not available, using CPU"
    
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Create evaluation environment (single env for evaluation)
    eval_env = DummyVecEnv([make_env])

    # Define PPO model with CORRECTED hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,          # Standard learning rate
        n_steps=n_steps,             # Steps per environment before update
        batch_size=batch_size,       # Batch size for updates
        n_epochs=10,                 # Number of optimization epochs per update
        gamma=0.99,                  # Discount factor
        gae_lambda=0.95,             # GAE lambda for advantage estimation
        clip_range=0.2,              # PPO clipping parameter
        ent_coef=0.05,               # INCREASED exploration coefficient (was 0.01)
        vf_coef=0.5,                 # Value function loss coefficient
        max_grad_norm=0.5,           # Gradient clipping
        policy_kwargs=dict(
            net_arch=[256, 256],     # Two hidden layers with 256 neurons each
            activation_fn=torch.nn.Tanh  # Tanh works better for bounded continuous actions
        ),
        tensorboard_log="./tensorboard_logs/",
        device=device                # Use CUDA if available
    )

    # Callbacks
    # Save checkpoint every 50k steps (adjusted for parallel envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1000),  # Ensure at least 1000 steps
        save_path=save_dir,
        name_prefix="dispenser_ppo"
    )

    # Evaluate every 50k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir + "best_model/",
        log_path=save_dir + "logs/",
        eval_freq=max(50000 // n_envs, 1000),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )

    print("=" * 70)
    print("SATELLITE DISPENSER RL TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device.upper()}")
    print(f"  -> {cuda_info}")
    print(f"Parallel environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Steps per update: {n_steps * n_envs:,}")
    print(f"Batch size: {batch_size}")
    print(f"Save directory: {os.path.abspath(save_dir)}")
    print(f"TensorBoard logs: ./tensorboard_logs/")
    print("=" * 70)
    print("To monitor training, run:")
    print(f"  tensorboard --logdir=./tensorboard_logs/")
    print("=" * 70)

    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        # Save final model
        final_path = os.path.join(save_dir, "dispenser_ppo_final.zip")
        model.save(final_path)
        
        # Also save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(save_dir, f"dispenser_ppo_{timestamp}.zip")
        model.save(backup_path)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Final model saved to: {final_path}")
        print(f"Backup saved to: {backup_path}")
        print(f"Best model saved to: {save_dir}best_model/best_model.zip")
        print("=" * 70)

        env.close()
        eval_env.close()

    return model


def continue_training(model_path, total_timesteps=500000, save_dir="./models/"):
    """
    Continue training from a saved model
    
    Args:
        model_path: Path to saved model
        total_timesteps: Additional timesteps to train
        save_dir: Directory to save models
    """
    print(f"Loading model from: {model_path}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {gpu_name}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Create environments
    n_envs = 8
    
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env])
    
    # Load model with specified device
    model = PPO.load(model_path, env=env, device=device, verbose=1)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1000),
        save_path=save_dir,
        name_prefix="dispenser_ppo_continued"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir + "best_model/",
        log_path=save_dir + "logs/",
        eval_freq=max(50000 // n_envs, 1000),
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    print(f"Continuing training for {total_timesteps:,} timesteps on {device.upper()}...")
    
    # Continue training
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Continue counting
    )
    
    # Save
    final_path = os.path.join(save_dir, "dispenser_ppo_final.zip")
    model.save(final_path)
    print(f"Model saved to: {final_path}")
    
    env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Satellite Dispenser RL Agent')
    parser.add_argument('--timesteps', type=int, default=1000000, 
                       help='Total training timesteps (default: 1000000)')
    parser.add_argument('--save-dir', type=str, default='./models/',
                       help='Directory to save models (default: ./models/)')
    parser.add_argument('--continue-from', type=str, default=None,
                       help='Path to model to continue training from')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    
    # Force CPU if requested
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Forcing CPU usage as requested")
    
    if args.continue_from:
        continue_training(args.continue_from, args.timesteps, args.save_dir)
    else:
        train_model(total_timesteps=args.timesteps, save_dir=args.save_dir)
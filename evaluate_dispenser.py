#!/usr/bin/env python3
"""
Evaluation Script for Trained Dispenser Agent
Loads trained model and runs deployment, outputs all metrics for analysis
"""

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import json

from dispenser_env import SatelliteDispenserEnv


def evaluate_model(model_path, n_episodes=5):
    """
    Evaluate trained model and output all results for analysis
    """
    print("="*70)
    print("SATELLITE DISPENSER MODEL EVALUATION")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None
    
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    env = SatelliteDispenserEnv()
    
    all_episodes_data = []
    success_count = 0
    
    for episode in range(n_episodes):
        print(f"\n--- Episode {episode+1}/{n_episodes} ---")
        
        obs, _ = env.reset()
        done = False
        step = 0
        max_steps = 500
        
        # Storage for this episode
        episode_data = {
            'episode': episode + 1,
            'states': [],
            'actions': [],
            'rewards': [],
            'total_reward': 0,
            'steps': 0,
            'success': False,
            'final_satellite_pos': 0,
            'max_velocity': 0,
            'max_contact_force': 0,
            'deployment_time': 0
        }
        
        while not done and step < max_steps:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Record state before step
            state = {
                'step': step,
                'pusher_pos': float(env.pusher_pos),
                'pusher_vel': float(env.pusher_vel),
                'satellite_pos': float(env.satellite_pos),
                'satellite_vel': float(env.satellite_vel),
                'gate_open': float(env.gate_open),
                'contact_force': float(env.contact_force)
            }
            episode_data['states'].append(state)
            episode_data['actions'].append(action.tolist())
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_data['rewards'].append(float(reward))
            episode_data['total_reward'] += float(reward)
            
            # Track metrics
            if abs(env.satellite_vel) > episode_data['max_velocity']:
                episode_data['max_velocity'] = float(abs(env.satellite_vel))
            if env.contact_force > episode_data['max_contact_force']:
                episode_data['max_contact_force'] = float(env.contact_force)
            
            step += 1
            done = terminated or truncated
        
        # Record final metrics
        episode_data['steps'] = step
        episode_data['final_satellite_pos'] = float(env.satellite_pos)
        episode_data['success'] = info.get('deployed', False)
        episode_data['deployment_time'] = step * 0.02  # dt = 0.02s
        
        if episode_data['success']:
            success_count += 1
        
        all_episodes_data.append(episode_data)
        
        # Print episode summary
        print(f"  Steps: {step}")
        print(f"  Success: {'YES' if episode_data['success'] else 'NO'}")
        print(f"  Final satellite pos: {episode_data['final_satellite_pos']:.3f}")
        print(f"  Max velocity: {episode_data['max_velocity']:.3f} m/s")
        print(f"  Max contact force: {episode_data['max_contact_force']:.3f}")
        print(f"  Total reward: {episode_data['total_reward']:.2f}")
    
    env.close()
    
    # Calculate aggregate statistics
    stats = {
        'model_path': model_path,
        'n_episodes': n_episodes,
        'success_rate': success_count / n_episodes,
        'success_count': success_count,
        'avg_steps': np.mean([ep['steps'] for ep in all_episodes_data]),
        'avg_reward': np.mean([ep['total_reward'] for ep in all_episodes_data]),
        'avg_final_pos': np.mean([ep['final_satellite_pos'] for ep in all_episodes_data]),
        'avg_max_velocity': np.mean([ep['max_velocity'] for ep in all_episodes_data]),
        'avg_max_force': np.mean([ep['max_contact_force'] for ep in all_episodes_data]),
        'episodes': all_episodes_data
    }
    
    # Print final summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Success Rate: {stats['success_rate']*100:.1f}% ({success_count}/{n_episodes})")
    print(f"Average Steps: {stats['avg_steps']:.1f}")
    print(f"Average Reward: {stats['avg_reward']:.2f}")
    print(f"Average Final Position: {stats['avg_final_pos']:.3f}")
    print(f"Average Max Velocity: {stats['avg_max_velocity']:.3f} m/s")
    print(f"Average Max Contact Force: {stats['avg_max_force']:.3f}")
    print("="*70)
    
    # Save results to JSON
    output_file = 'evaluation_results.json'
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        json.dump(stats, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create visualization of best episode
    best_episode = max(all_episodes_data, key=lambda x: x['total_reward'])
    create_analysis_plots(best_episode, 'best_episode_analysis.png')
    
    return stats


def create_analysis_plots(episode_data, save_path):
    """Create detailed plots for analysis"""
    states = episode_data['states']
    steps = [s['step'] for s in states]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Row 1: Positions
    axes[0,0].plot(steps, [s['pusher_pos'] for s in states], 'r-', label='Pusher', linewidth=2)
    axes[0,0].plot(steps, [s['satellite_pos'] for s in states], 'b-', label='Satellite', linewidth=2)
    axes[0,0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Exit')
    axes[0,0].set_ylabel('Position (normalized)')
    axes[0,0].set_title('Positions')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(steps, [s['gate_open'] for s in states], 'g-', linewidth=2)
    axes[0,1].set_ylabel('Gate Open (0-1)')
    axes[0,1].set_title('Gate State')
    axes[0,1].grid(True, alpha=0.3)
    
    # Row 2: Velocities
    axes[1,0].plot(steps, [s['pusher_vel'] for s in states], 'r-', label='Pusher', linewidth=2)
    axes[1,0].plot(steps, [s['satellite_vel'] for s in states], 'b-', label='Satellite', linewidth=2)
    axes[1,0].set_ylabel('Velocity (m/s)')
    axes[1,0].set_title('Velocities')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Cumulative reward
    cumsum_rewards = np.cumsum(episode_data['rewards'])
    axes[1,1].plot(steps, cumsum_rewards, 'purple', linewidth=2)
    axes[1,1].set_ylabel('Cumulative Reward')
    axes[1,1].set_title('Reward Accumulation')
    axes[1,1].grid(True, alpha=0.3)
    
    # Row 3: Contact force and phase diagram
    axes[2,0].plot(steps, [s['contact_force'] for s in states], 'orange', linewidth=2)
    axes[2,0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Safety Limit')
    axes[2,0].set_xlabel('Step')
    axes[2,0].set_ylabel('Contact Force')
    axes[2,0].set_title('Contact Force')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # Phase portrait
    sat_positions = [s['satellite_pos'] for s in states]
    sat_velocities = [s['satellite_vel'] for s in states]
    axes[2,1].plot(sat_positions, sat_velocities, 'b-', linewidth=2)
    axes[2,1].set_xlabel('Satellite Position')
    axes[2,1].set_ylabel('Satellite Velocity')
    axes[2,1].set_title('Phase Portrait')
    axes[2,1].grid(True, alpha=0.3)
    
    fig.suptitle(f"Episode {episode_data['episode']} Analysis | "
                 f"Success: {episode_data['success']} | "
                 f"Reward: {episode_data['total_reward']:.2f}", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    # Default model path
    model_path = "./models/dispenser_ppo_final.zip"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    n_episodes = 5
    if len(sys.argv) > 2:
        n_episodes = int(sys.argv[2])
    
    # Run evaluation
    stats = evaluate_model(model_path, n_episodes)
    
    if stats is None:
        print("\nTo train a model first, run: python train_dispenser.py")
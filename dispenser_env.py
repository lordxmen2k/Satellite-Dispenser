import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

class SatelliteDispenserEnv(gym.Env):
    """
    FIXED Satellite Dispenser Environment for RL Training
    
    State Space (7D):
    - pusher_position: 0.0 to 1.0 (normalized, 0=home, 1=fully extended)
    - pusher_velocity: -1.0 to 1.0 (normalized)
    - gate_open: 0.0 to 1.0 (0=closed, 1=fully open)
    - satellite_position: 0.0 to 1.5 (0=fully retracted, 1.2=clear)
    - satellite_velocity: -1.0 to 1.0 (normalized)
    - contact_force: 0.0 to 1.0 (normalized force)
    - time_step: 0 to 1.0 (normalized progress)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode=None, max_steps=500):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        # Physics constants
        self.dt = 0.02
        self.pusher_max_vel = 0.5               # m/s
        self.pusher_accel = 2.0                 # m/s^2
        self.satellite_mass = 10.0
        self.friction_coeff = 0.05
        self.contact_stiffness = 2000.0         # REDUCED from 5000
        self.gate_actuation_time = int(0.5 / self.dt)
        
        # Success threshold
        self.success_pos = 1.0                  # Match deployed flag

        # State space: 7 continuous values
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0], dtype=np.float32),  # satellite can go to 1.5
            dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # State variables
        self.pusher_pos = 0.0
        self.pusher_vel = 0.0
        self.gate_open = 0.0
        self.gate_target = 0.0
        self.satellite_pos = 0.0
        self.satellite_vel = 0.0
        self.contact_force = 0.0
        self.step_count = 0
        self.deployed = False
        self._prev_vel = 0.0
        self._prev_action = np.array([0.0, 0.0])

        # Rendering
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pusher_pos = 0.0
        self.pusher_vel = 0.0
        self.gate_open = 0.0
        self.gate_target = 0.0
        self.satellite_pos = 0.0
        self.satellite_vel = 0.0
        self.contact_force = 0.0
        self.step_count = 0
        self.deployed = False
        self._prev_vel = 0.0
        self._prev_action = np.array([0.0, 0.0])
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        target_vel_norm, gate_cmd = action
        self.step_count += 1

        # Convert normalized action to physical values
        target_vel = target_vel_norm * self.pusher_max_vel
        self.gate_target = 1.0 if gate_cmd > 0.5 else 0.0

        # Update gate position
        gate_rate = 1.0 / self.gate_actuation_time
        if self.gate_target > self.gate_open:
            self.gate_open = min(self.gate_open + gate_rate, 1.0)
        else:
            self.gate_open = max(self.gate_open - gate_rate, 0.0)

        # Pusher dynamics
        vel_error = target_vel - self.pusher_vel
        vel_delta = np.clip(vel_error, -self.pusher_accel * self.dt, self.pusher_accel * self.dt)
        self.pusher_vel += vel_delta
        self.pusher_vel = np.clip(self.pusher_vel, -self.pusher_max_vel, self.pusher_max_vel)

        # Update pusher position
        self.pusher_pos += self.pusher_vel * self.dt
        self.pusher_pos = np.clip(self.pusher_pos, 0.0, 1.0)

        # ==================== IMPROVED CONTACT LOGIC ====================
        contact_threshold = 0.02
        
        if self.pusher_pos >= self.satellite_pos - contact_threshold:
            penetration = max(0.0, self.pusher_pos - self.satellite_pos)
            # Softer contact model
            self.contact_force = min(penetration * self.contact_stiffness * 0.001, 1.0)
            
            if penetration > 0:
                # IMPROVED: Better velocity transfer with damping
                transfer_efficiency = 0.9  # Increased from 0.8
                
                # Target velocity with smoothing
                target_sat_vel = self.pusher_vel * transfer_efficiency
                
                # Smooth velocity transfer with less damping
                alpha = 0.7  # Faster response
                self.satellite_vel += (target_sat_vel - self.satellite_vel) * alpha
                
                # Allow slight overshoot for momentum
                if self.satellite_vel > self.pusher_vel and self.satellite_vel < self.pusher_vel * 1.1:
                    pass  # Allow small overshoot
                else:
                    self.satellite_vel = min(self.satellite_vel, self.pusher_vel * 1.05)
        else:
            self.contact_force = 0.0
            # Less friction when not in contact
            self.satellite_vel *= 0.98
        
        # Prevent backward motion
        self.satellite_vel = max(self.satellite_vel, 0.0)

        # Update satellite position
        self.satellite_pos += self.satellite_vel * self.dt
        self.satellite_pos = max(self.satellite_pos, 0.0)

        # Check deployment
        if self.satellite_pos >= self.success_pos and not self.deployed:
            self.deployed = True

        # Calculate reward
        reward = self._calculate_reward(target_vel_norm, action)

        # Check termination
        terminated = False
        truncated = False

        # Safety violation
        if self.pusher_vel > 0.1 and self.gate_open < 0.8 and self.satellite_pos < 0.3:
            reward -= 5.0  # Reduced from 10
            
        # Success: Satellite deployed (FIXED: use same threshold)
        if self.satellite_pos >= self.success_pos:
            reward += 500.0  # Much bigger bonus
            terminated = True

        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            'pusher_pos': self.pusher_pos,
            'pusher_vel': self.pusher_vel,
            'satellite_pos': self.satellite_pos,
            'satellite_vel': self.satellite_vel,
            'gate_open': self.gate_open,
            'contact_force': self.contact_force,
            'deployed': self.deployed
        }

        self._prev_action = action.copy()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        return np.array([
            self.pusher_pos,
            self.pusher_vel / self.pusher_max_vel,
            self.gate_open,
            self.satellite_pos / 1.5,  # Normalize to 0-1 range for network
            self.satellite_vel / self.pusher_max_vel,
            self.contact_force,
            self.step_count / self.max_steps
        ], dtype=np.float32)

    def _calculate_reward(self, target_vel_norm, action):
        reward = 0.0

        # 1. Progress reward: Strong reward for satellite position
        # Use shaped reward: exponential to encourage getting closer
        reward += self.satellite_pos * 3.0
        
        # 2. Velocity reward: Encourage positive velocity
        reward += self.satellite_vel * 8.0
        
        # 3. Contact efficiency: Reward for making progress while in contact
        if self.contact_force > 0 and self.satellite_vel > 0.01:
            reward += 1.0
            
        # 4. Penalty for stalling
        if self.satellite_vel < 0.001 and self.step_count > 100:
            reward -= 0.5  # Increased penalty

        # 5. Action smoothness penalty (reduce chattering)
        action_diff = np.abs(action - self._prev_action)
        reward -= 0.1 * action_diff[0]  # Penalize velocity command changes
        
        # 6. Stronger penalty for excessive contact force
        if self.contact_force > 0.5:
            reward -= (self.contact_force - 0.5) * 2.0  # Scales with violation
        if self.contact_force > 0.8:
            reward -= 5.0  # Heavy penalty for dangerous force

        # 7. Gate sequencing reward
        if self.pusher_pos > 0.1:
            if self.gate_open > 0.8:
                reward += 0.5
            else:
                reward -= 1.0  # Stronger penalty
                
        # 8. Small efficiency penalty
        reward -= 0.005 * (target_vel_norm ** 2)

        # 9. Bonus for reaching milestones
        if self.satellite_pos > 0.5 and not hasattr(self, '_halfway_bonus'):
            reward += 10.0
            self._halfway_bonus = True
        if self.satellite_pos > 0.8 and not hasattr(self, '_near_exit_bonus'):
            reward += 20.0
            self._near_exit_bonus = True

        return float(reward)

    def render(self):
        # ... (keep existing render code) ...
        pass

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
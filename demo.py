#!/usr/bin/env python3
"""
Satellite Dispenser Demo - FINAL VERSION
Satellite moves past EXIT arrow, status: GATE OPENING -> PUSHING -> COASTING -> EXITING
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter

def run_demo_simulation():
    """Run simulation with satellite exiting past the arrow"""
    
    dt = 0.02
    max_steps = 600  # Increased to allow full exit
    
    # Coordinate system: 0 = left (pusher), 1 = right (gate/exit)
    gate_open = 0.0       
    pusher_pos = 0.15     
    pusher_vel = 0.0
    pusher_active = True
    
    satellite_pos = 0.35  
    satellite_vel = 0.0
    contact_force = 0.0
    
    history = []
    exited = False  # Track if satellite passed exit point
    
    for step in range(max_steps):
        # Phase 1: Open gate (roll up) at exit (right)
        if step < 80:
            target_gate = 1.0
            target_pusher_vel = 0.0
        # Phase 2: Pusher pushes satellite right toward gate/exit
        elif step < 400:
            target_gate = 1.0
            progress = (step - 80) / 320
            
            # S-curve velocity profile
            if progress < 0.2:
                target_pusher_vel = 0.1 + (progress / 0.2) * 0.3
            elif progress < 0.6:
                target_pusher_vel = 0.4
            else:
                target_pusher_vel = 0.4 - ((progress - 0.6) / 0.4) * 0.3
            
            # Pusher stops at position, satellite continues to exit
            if pusher_pos > 0.6:
                pusher_active = False
                target_pusher_vel = 0.0
        else:
            target_gate = 1.0
            target_pusher_vel = 0.0
        
        # Update gate (roll up)
        gate_rate = 0.015
        gate_open += gate_rate * np.sign(target_gate - gate_open)
        gate_open = np.clip(gate_open, 0.0, 1.0)
        
        # Update pusher
        if pusher_active:
            max_accel = 2.0
            vel_error = target_pusher_vel - pusher_vel
            vel_delta = np.clip(vel_error, -max_accel * dt, max_accel * dt)
            pusher_vel += vel_delta
            pusher_vel = np.clip(pusher_vel, -0.1, 0.5)
            pusher_pos += pusher_vel * dt
            pusher_pos = np.clip(pusher_pos, 0.0, 0.65)
        else:
            pusher_vel = 0.0
        
        # Contact physics: pusher pushes satellite from left
        contact_threshold = 0.08
        if pusher_active and (satellite_pos - pusher_pos) < contact_threshold and satellite_pos > pusher_pos:
            k = 2000
            penetration = contact_threshold - (satellite_pos - pusher_pos)
            force = k * penetration * 0.001
            friction = 0.05 * 4.0 * 9.81 * 0.001
            accel = max(0, force - friction) / 4.0
            satellite_vel = pusher_vel + accel * dt * 10
            contact_force = force
        else:
            # Less friction in space once deployed
            if satellite_pos > 1.0:
                satellite_vel *= 0.999  # Almost no friction in space
            else:
                satellite_vel *= 0.995  # Some friction in dispenser
            contact_force = 0.0
        
        satellite_pos += satellite_vel * dt
        
        # Track when satellite passes exit point (1.0)
        if satellite_pos >= 1.0:
            exited = True
        
        # Determine status based on phase and position
        if step < 80:
            status = "GATE OPENING"
        elif pusher_active and step < 400:
            status = "PUSHING"
        elif not pusher_active and not exited:
            status = "COASTING"
        else:
            status = "EXITING"
        
        history.append({
            'step': step,
            'pusher_pos': pusher_pos,
            'pusher_vel': pusher_vel,
            'satellite_pos': satellite_pos,
            'satellite_vel': satellite_vel,
            'gate_open': gate_open,
            'contact_force': contact_force,
            'target_vel': target_pusher_vel,
            'pusher_active': pusher_active,
            'status': status,
            'exited': exited
        })
        
        # Continue until satellite is well past exit
        if exited and satellite_pos > 1.3:
            print(f"Satellite fully exited at step {step}!")
            break
    
    return history

def create_visualization(history, save_path=None):
    """Create animation with satellite moving past EXIT arrow"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={'width_ratios': [2, 1]})
    
    # Animation area - extended to show satellite exiting
    ax1.set_xlim(-0.1, 1.5)
    ax1.set_ylim(-0.2, 1.0)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('CubeSat Dispenser - Top View', fontsize=12, fontweight='bold')
    
    # Metrics
    ax2.set_title('Deployment Metrics', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Velocity (m/s)')
    line1, = ax2.plot([], [], 'r-', label='Pusher', linewidth=2)
    line2, = ax2.plot([], [], 'b-', label='Satellite', linewidth=2)
    line3, = ax2.plot([], [], 'g--', label='Target', linewidth=1.5, alpha=0.7)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(history))
    ax2.set_ylim(0, 0.6)
    
    steps_data = [h['step'] for h in history]
    pusher_vels = [h['pusher_vel'] for h in history]
    sat_vels = [h['satellite_vel'] for h in history]
    tgt_vels = [h['target_vel'] for h in history]
    
    def init():
        ax1.clear()
        ax1.set_xlim(-0.1, 1.5)
        ax1.set_ylim(-0.2, 1.0)
        ax1.set_aspect('equal')
        ax1.axis('off')
        return []
    
    def update(frame_idx):
        state = history[frame_idx]
        
        ax1.clear()
        ax1.set_xlim(-0.1, 1.5)
        ax1.set_ylim(-0.2, 1.0)
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # Dispenser frame (rails) - only up to exit point
        ax1.plot([0, 1.0], [0.7, 0.7], 'k-', linewidth=3)
        ax1.plot([0, 1.0], [0.2, 0.2], 'k-', linewidth=3)
        
        # BACK/LEFT side
        ax1.plot([0, 0], [0.2, 0.7], 'k-', linewidth=4)
        ax1.text(0, 0.85, 'BACK', fontsize=9, ha='center', fontweight='bold')
        
        # PUSHER (RED) - on left
        pusher_x = state['pusher_pos']
        pusher = Rectangle((pusher_x, 0.25), 0.04, 0.4,
                          facecolor='#e53e3e', edgecolor='darkred', linewidth=3)
        ax1.add_patch(pusher)
        ax1.plot([0, pusher_x], [0.45, 0.45], '#e53e3e', linewidth=8)
        base = Rectangle((-0.08, 0.35), 0.08, 0.2, 
                        facecolor='#c53030', edgecolor='darkred', linewidth=2)
        ax1.add_patch(base)
        
        # SATELLITE (BLUE) - moves past exit arrow
        sat_x = state['satellite_pos']
        sat_width = 0.18
        # Draw satellite even when past exit
        satellite = Rectangle((sat_x - sat_width/2, 0.3), sat_width, 0.3,
                             facecolor='#3182ce', edgecolor='navy', linewidth=2)
        ax1.add_patch(satellite)
        for i in range(2):
            x = sat_x - sat_width/2 + 0.04 + i * 0.07
            ax1.plot([x, x], [0.35, 0.55], 'white', linewidth=3)
        
        # GATE (ROLL-UP ONLY) - on right at position 1.0
        gate_open = state['gate_open']
        
        curtain_height = 0.5 * (1.0 - gate_open)
        if curtain_height > 0.02:
            curtain = Rectangle((0.95, 0.2), 0.05, curtain_height,
                               facecolor='#e53e3e', edgecolor='darkred', 
                               linewidth=2, alpha=0.9)
            ax1.add_patch(curtain)
            for i in range(4):
                y = 0.2 + i * (curtain_height/4)
                ax1.plot([0.95, 1.0], [y, y], 'white', linewidth=1)
        
        # EXIT LABEL AND ARROW - fixed at position 1.0 (gate location)
        ax1.text(1.0, 0.85, 'EXIT', fontsize=11, ha='center', 
                fontweight='bold', color='green')
        ax1.annotate('', xy=(1.15, 0.45), xytext=(1.0, 0.45),
                    arrowprops=dict(arrowstyle='->', color='green', lw=3))
        
        # Dotted line showing exit boundary
        ax1.plot([1.0, 1.0], [0.2, 0.7], 'g--', alpha=0.3, linewidth=1)
        
        # Direction arrow showing push (only while pushing)
        if state['pusher_vel'] > 0.01 and state['pusher_active']:
            ax1.annotate('', xy=(sat_x + 0.2, 0.45), xytext=(sat_x, 0.45),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3))
        
        # Status text with color coding
        status = state['status']
        if status == "GATE OPENING":
            color = '#ed8936'  # Orange
        elif status == "PUSHING":
            color = '#e53e3e'  # Red
        elif status == "COASTING":
            color = '#3182ce'  # Blue
        else:  # EXITING
            color = '#48bb78'  # Green
            
        ax1.text(0.5, 0.95, f'Step {state["step"]} | {status}', 
                fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # Labels
        ax1.text(0.5, -0.1, 'Pusher → Satellite → Gate → EXIT', 
                fontsize=10, ha='center', style='italic')
        ax1.text(0.95, -0.05, f'Gate: {gate_open:.0%}', fontsize=9, ha='center')
        
        # Metrics update
        line1.set_data(steps_data[:frame_idx+1], pusher_vels[:frame_idx+1])
        line2.set_data(steps_data[:frame_idx+1], sat_vels[:frame_idx+1])
        line3.set_data(steps_data[:frame_idx+1], tgt_vels[:frame_idx+1])
        
        return []
    
    anim = FuncAnimation(fig, update, frames=len(history), init_func=init,
                        interval=40, blit=False)
    
    if save_path:
        print(f"Saving to {save_path}...")
        writer = PillowWriter(fps=25)
        anim.save(save_path, writer=writer)
        print("Saved!")
    
    plt.tight_layout()
    plt.show()
    return anim

def create_static_analysis(history):
    """Create analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps = [h['step'] for h in history]
    
    axes[0,0].plot(steps, [h['pusher_pos'] for h in history], 'r-', label='Pusher', linewidth=2)
    axes[0,0].plot(steps, [h['satellite_pos'] for h in history], 'b-', label='Satellite', linewidth=2)
    axes[0,0].axhline(y=0.6, color='r', linestyle='--', alpha=0.3, label='Pusher Stop')
    axes[0,0].axhline(y=1.0, color='g', linestyle='--', alpha=0.3, label='Exit Point')
    axes[0,0].set_title('Positions (0=left/back, 1=exit, >1=exited)')
    axes[0,0].set_ylabel('Position')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(steps, [h['pusher_vel'] for h in history], 'r-', label='Pusher', linewidth=2)
    axes[0,1].plot(steps, [h['satellite_vel'] for h in history], 'b-', label='Satellite', linewidth=2)
    axes[0,1].set_title('Velocities')
    axes[0,1].set_ylabel('m/s')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(steps, [h['gate_open'] for h in history], 'purple', linewidth=2)
    axes[1,0].set_title('Gate Open (0=closed, 1=open)')
    axes[1,0].set_ylim(-0.1, 1.1)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot status over time
    status_map = {'GATE OPENING': 0, 'PUSHING': 1, 'COASTING': 2, 'EXITING': 3}
    status_values = [status_map[h['status']] for h in history]
    axes[1,1].plot(steps, status_values, 'green', linewidth=2)
    axes[1,1].set_title('Mission Phase')
    axes[1,1].set_yticks([0, 1, 2, 3])
    axes[1,1].set_yticklabels(['Gate Open', 'Pushing', 'Coasting', 'Exiting'])
    axes[1,1].set_xlabel('Step')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_analysis.png', dpi=300, bbox_inches='tight')
    print("Analysis saved")
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("SATELLITE DISPENSER - FINAL VERSION")
    print("="*60)
    print("Satellite moves past EXIT arrow")
    print("Status: GATE OPENING → PUSHING → COASTING → EXITING")
    print()
    
    history = run_demo_simulation()
    
    print(f"\nCompleted: {len(history)} steps")
    print(f"Final satellite position: {history[-1]['satellite_pos']:.3f}")
    
    create_visualization(history, 'demo_deployment.gif')
    create_static_analysis(history)
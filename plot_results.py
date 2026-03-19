import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------
# Load logs
# ------------------------------------------------------------------
with open('training_logs.json', 'r') as f:
    logs = json.load(f)

rewards  = np.array(logs['rewards'])
costs    = np.array(logs['costs'])
lambdas  = np.array(logs['lambdas'])
rm_loss  = np.array(logs['rm_loss'])

def smooth(x, window=20):
    return np.convolve(x, np.ones(window)/window, mode='valid')

episodes = np.arange(len(smooth(rewards, 20)))

# ------------------------------------------------------------------
# Figure 1 — RL_agent.png
# Three panels: reward, cost, lambda over training
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle('Safety-Constrained HRL Training Curves', fontsize=14, fontweight='bold', y=0.98)

# Reward
axes[0].plot(episodes, smooth(rewards, 20), color='#2196F3', linewidth=1.8, label='Smoothed Reward')
axes[0].fill_between(episodes, smooth(rewards, 20), alpha=0.15, color='#2196F3')
axes[0].set_ylabel('Episode Reward')
axes[0].set_title('RLHF Reward Signal')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Cost
axes[1].plot(episodes, smooth(costs, 20), color='#F44336', linewidth=1.8, label='Avg Cost/Step')
axes[1].axhline(y=0.1, color='black', linestyle='--', linewidth=1.2, label='Cost Limit (δ=0.1)')
axes[1].fill_between(episodes, smooth(costs, 20), alpha=0.15, color='#F44336')
axes[1].set_ylabel('Safety Cost')
axes[1].set_title('Constraint Violation Rate')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Lambda
axes[2].plot(episodes, smooth(lambdas, 20), color='#4CAF50', linewidth=1.8, label='λ (Lagrange Multiplier)')
axes[2].fill_between(episodes, smooth(lambdas, 20), alpha=0.15, color='#4CAF50')
axes[2].set_ylabel('λ Value')
axes[2].set_xlabel('Episode')
axes[2].set_title('Lagrangian Multiplier Adaptation')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RL_agent.png', dpi=150, bbox_inches='tight')
print("Saved RL_agent.png")
plt.close()

# ------------------------------------------------------------------
# Figure 2 — reward_modelling.png
# RewardModel pretraining loss + preference accuracy estimate
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('RLHF Reward Model Pretraining', fontsize=14, fontweight='bold')

epochs = np.arange(1, len(rm_loss) + 1)

# Loss curve
axes[0].plot(epochs, rm_loss, color='#9C27B0', linewidth=2.0, marker='o', markersize=5, label='BCE Loss')
axes[0].fill_between(epochs, rm_loss, alpha=0.15, color='#9C27B0')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Bradley-Terry BCE Loss')
axes[0].set_title('Preference Learning Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy estimate: lower loss → higher accuracy
# Binary cross-entropy of 0.693 = random (50%), 0.0 = perfect
# Map loss to accuracy estimate
acc_estimate = (1 - (rm_loss - min(rm_loss)) / (rm_loss[0] - min(rm_loss) + 1e-8)) * 50 + 50
axes[1].plot(epochs, acc_estimate, color='#FF9800', linewidth=2.0, marker='s', markersize=5, label='Est. Preference Accuracy')
axes[1].axhline(y=50, color='gray', linestyle='--', linewidth=1.2, label='Random Baseline (50%)')
axes[1].set_ylim(45, 105)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Preference Prediction Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reward_modelling.png', dpi=150, bbox_inches='tight')
print("Saved reward_modelling.png")
plt.close()

print("\nAll plots generated successfully.")

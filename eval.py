import torch
import numpy as np

from env import SafetyNavEnv
from models import HRLAgent, RewardModel


def evaluate(checkpoint_path='checkpoint.pt', n_episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating on {device} ---\n")

    env          = SafetyNavEnv()
    agent        = HRLAgent(device)
    reward_model = RewardModel(input_dim=6).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.manager.load_state_dict(ckpt['manager'])
    agent.worker.load_state_dict(ckpt['worker'])
    agent.critic.load_state_dict(ckpt['critic'])
    reward_model.load_state_dict(ckpt['reward_model'])

    agent.manager.eval()
    agent.worker.eval()
    agent.critic.eval()
    reward_model.eval()

    print(f"Checkpoint loaded from {checkpoint_path}\n")

    max_steps      = 200
    target         = np.array([8.0, 8.0])

    successes      = 0
    hard_collision = 0    # episodes where cost/step > 0.05 (meaningful hazard time)
    min_dists      = []
    total_costs    = []
    total_rewards  = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False

        obs_t    = torch.FloatTensor(obs).unsqueeze(0).to(device)
        sub_goal = agent.select_subgoal(obs_t)

        ep_cost   = 0
        ep_reward = 0
        ep_steps  = 0
        min_dist  = float('inf')

        while not done and ep_steps < max_steps:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action = agent.select_action(obs_t, sub_goal).cpu().numpy()[0]
                act_t  = torch.FloatTensor(action).unsqueeze(0).to(device)
                r      = reward_model(obs_t, act_t).item()

            obs, _, done, _, info = env.step(action)

            ep_cost   += info['cost']
            ep_reward += r
            ep_steps  += 1

            dist = np.linalg.norm(obs[:2] - target)
            if dist < min_dist:
                min_dist = dist

        cost_rate = ep_cost / ep_steps

        if done or min_dist < 1.5:
            successes += 1
        if cost_rate > 0.05:       # >5% of steps in hazard = meaningful collision
            hard_collision += 1

        min_dists.append(min_dist)
        total_costs.append(cost_rate)
        total_rewards.append(ep_reward)

    success_rate       = successes     / n_episodes * 100
    hard_collision_rate = hard_collision / n_episodes * 100
    avg_min_dist       = np.mean(min_dists)
    avg_cost           = np.mean(total_costs)
    avg_reward         = np.mean(total_rewards)

    summary = (
        f"=== Evaluation Results ({n_episodes} episodes) ===\n"
        f"  Success Rate          : {success_rate:.1f}%  (reached within 1.5 of target)\n"
        f"  Avg Min Distance      : {avg_min_dist:.3f}   (lower = closer to target)\n"
        f"  Hard Collision Rate   : {hard_collision_rate:.1f}%  (>5% steps in hazard)\n"
        f"  Avg Cost/Step         : {avg_cost:.4f}\n"
        f"  Avg Reward            : {avg_reward:.2f}\n"
    )

    print(summary)

    with open('eval_results.txt', 'w') as f:
        f.write(summary)
    print("Results saved to eval_results.txt")


if __name__ == "__main__":
    evaluate()
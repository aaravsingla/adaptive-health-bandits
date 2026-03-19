import torch
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Local Imports
from env import SafetyNavEnv
from models import HRLAgent, RewardModel, train_reward_model


def train():
    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on {device} ---\n")

    env          = SafetyNavEnv()
    agent        = HRLAgent(device)
    reward_model = RewardModel(input_dim=6).to(device)

    # Hyperparameters
    episodes   = 2000
    max_steps  = 200
    gamma      = 0.99
    cost_limit = 0.1

    # PPO clip epsilon
    eps_clip = 0.2

    # ------------------------------------------------------------------
    # 2. Pretrain RewardModel (Step 1 — RLHF)
    # ------------------------------------------------------------------
    reward_model, rm_loss_history = train_reward_model(
        reward_model,
        device,
        n_pairs    = 2000,
        epochs     = 10,
        batch_size = 64,
        lr         = 1e-3,
    )
    # Freeze reward model — used only for inference during RL training
    for param in reward_model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # 3. RL Training Loop
    # ------------------------------------------------------------------
    pbar = tqdm(range(episodes), desc="Training")

    # Logging buffers
    ep_rewards  = []
    ep_costs    = []
    ep_lambdas  = []

    moving_avg_reward = 0

    for ep in pbar:
        obs, _ = env.reset()
        done   = False
        traj_data = []

        # Manager selects sub-goal once per episode
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        sub_goal   = agent.select_subgoal(obs_tensor)

        step_count = 0
        while not done and step_count < max_steps:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            # Worker action with exploration noise
            action_mean = agent.select_action(obs_t, sub_goal)
            action      = action_mean.cpu().detach().numpy()[0] + np.random.normal(0, 0.1, size=2)

            next_obs, _, done, _, info = env.step(action)

            # RLHF reward from pretrained reward model
            with torch.no_grad():
                act_t      = torch.FloatTensor(action).unsqueeze(0).to(device)
                rlhf_reward = reward_model(obs_t, act_t).item()

            # Value estimate from critic
            with torch.no_grad():
                value = agent.get_value(obs_t).item()

            traj_data.append({
                's'        : obs_t,
                'a'        : torch.FloatTensor(action).unsqueeze(0).to(device),
                'r'        : rlhf_reward,
                'c'        : info['cost'],
                'sub_goal' : sub_goal,
                'value'    : value,
            })

            obs = next_obs
            step_count += 1

        # --------------------------------------------------------------
        # 4. Compute Returns + Advantages (PPO baseline)
        # --------------------------------------------------------------
        R       = 0
        returns = []

        for step in reversed(traj_data):
            R = step['r'] + gamma * R
            returns.insert(0, R)

        returns_t  = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t   = torch.tensor([s['value'] for s in traj_data], dtype=torch.float32).to(device)

        # Normalize advantages
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        total_cost = sum(s['c'] for s in traj_data)
        avg_cost   = total_cost / len(traj_data)

        # --------------------------------------------------------------
        # 5. PPO + Lagrangian Update
        # --------------------------------------------------------------
        policy_loss  = []
        critic_loss  = []

        for i, step in enumerate(traj_data):
            # --- Actor (PPO clipped surrogate) ---
            action_pred = agent.select_action(step['s'], step['sub_goal'])
            dist        = Normal(action_pred, 0.1)
            log_prob    = dist.log_prob(step['a']).sum()

            # Old log prob (detached — single update per sample, ratio=1 at first)
            with torch.no_grad():
                old_log_prob = dist.log_prob(step['a']).sum()

            ratio      = torch.exp(log_prob - old_log_prob)
            adv        = advantages[i]
            lagrange_val = F.softplus(agent.log_lagrange)

            # Constrained advantage
            constrained_adv = adv - lagrange_val.detach() * (avg_cost - cost_limit)

            surr1 = ratio * constrained_adv
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * constrained_adv
            actor_loss = -torch.min(surr1, surr2)
            policy_loss.append(actor_loss)

            # --- Critic ---
            value_pred = agent.get_value(step['s'])
            c_loss     = F.mse_loss(value_pred.squeeze(), returns_t[i])
            critic_loss.append(c_loss)

        # Lagrangian dual ascent
        lagrange_loss = -agent.log_lagrange * (avg_cost - cost_limit)

        agent.optimizer.zero_grad()
        final_loss = (
            torch.stack(policy_loss).sum() +
            0.5 * torch.stack(critic_loss).sum() +
            lagrange_loss
        )
        final_loss.backward()
        agent.optimizer.step()

        # --------------------------------------------------------------
        # 6. Logging
        # --------------------------------------------------------------
        ep_reward = sum(s['r'] for s in traj_data)
        moving_avg_reward = 0.05 * ep_reward + 0.95 * moving_avg_reward

        ep_rewards.append(ep_reward)
        ep_costs.append(avg_cost)
        ep_lambdas.append(F.softplus(agent.log_lagrange).item())

        if ep % 10 == 0:
            pbar.set_postfix({
                'Rew' : f"{moving_avg_reward:.2f}",
                'Cost': f"{avg_cost:.2f}",
                'Lmb' : f"{F.softplus(agent.log_lagrange).item():.2f}",
            })

    # ------------------------------------------------------------------
    # 7. Save trained agent
    # ------------------------------------------------------------------
    torch.save({
        'manager'     : agent.manager.state_dict(),
        'worker'      : agent.worker.state_dict(),
        'critic'      : agent.critic.state_dict(),
        'log_lagrange': agent.log_lagrange,
        'reward_model': reward_model.state_dict(),
    }, 'checkpoint.pt')
    print("\nCheckpoint saved to checkpoint.pt")

    # ------------------------------------------------------------------
    # 8. Return logs for plotting (Step 4 will use these)
    # ------------------------------------------------------------------
    return ep_rewards, ep_costs, ep_lambdas, rm_loss_history


if __name__ == "__main__":
    ep_rewards, ep_costs, ep_lambdas, rm_loss_history = train()
    print("\nTraining complete. Run plot_results.py to generate figures.")
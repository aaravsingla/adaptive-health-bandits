import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# RLHF Reward Model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


def train_reward_model(reward_model, device, n_pairs=2000, epochs=10, batch_size=64, lr=1e-3):
    """
    Pretrain the RewardModel on synthetic preference data.

    Uses Bradley-Terry cross-entropy loss:
        loss = -[ label * log P(A>B) + (1-label) * log P(B>A) ]
    where P(A>B) = sigmoid(r_A - r_B)

    Args:
        reward_model : RewardModel instance (already on device)
        device       : torch device
        n_pairs      : number of preference pairs to generate
        epochs       : training epochs
        batch_size   : dataloader batch size
        lr           : learning rate

    Returns:
        reward_model : trained in-place, also returned for convenience
        loss_history : list of per-epoch mean losses (for plotting)
    """
    # Import here to avoid circular deps at module level
    from reward_data import PreferenceDataset

    print("--- Pretraining RewardModel on synthetic preferences ---")

    dataset    = PreferenceDataset(n_pairs=n_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = optim.Adam(reward_model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        epoch_losses = []

        for sa, aa, sb, ab, label in dataloader:
            sa, aa = sa.to(device), aa.to(device)
            sb, ab = sb.to(device), ab.to(device)
            label  = label.to(device)          # (B, 1)  1=A preferred, 0=B preferred

            r_a = reward_model(sa, aa)         # (B, 1)
            r_b = reward_model(sb, ab)         # (B, 1)

            # Bradley-Terry: P(A>B) = sigmoid(r_a - r_b)
            # BCE loss: -[y*log(sig) + (1-y)*log(1-sig)]
            loss = F.binary_cross_entropy_with_logits(r_a - r_b, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(mean_loss)
        print(f"  RewardModel Epoch [{epoch+1}/{epochs}]  Loss: {mean_loss:.4f}")

    print("--- RewardModel pretraining complete ---\n")
    return reward_model, loss_history


# ---------------------------------------------------------------------------
# Hierarchical Agent
# ---------------------------------------------------------------------------

class HRLAgent:
    def __init__(self, device):
        self.device = device

        # Manager: State -> Subgoal
        self.manager = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)
        ).to(device)

        # Worker: State + Subgoal -> Action
        self.worker = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 2)
        ).to(device)

        # Value network (Critic) for PPO baseline
        self.critic = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1)
        ).to(device)

        # Lagrangian Multiplier (Learnable Safety Parameter)
        self.log_lagrange = torch.nn.Parameter(
            torch.zeros(1, requires_grad=True).to(device)
        )

        self.optimizer = optim.Adam(
            list(self.manager.parameters()) +
            list(self.worker.parameters()) +
            list(self.critic.parameters()) +
            [self.log_lagrange],
            lr=0.0003
        )

    def select_action(self, state_t, subgoal_t):
        inputs = torch.cat([state_t, subgoal_t], dim=1)
        return self.worker(inputs)

    def select_subgoal(self, state_t):
        return self.manager(state_t)

    def get_value(self, state_t):
        return self.critic(state_t)
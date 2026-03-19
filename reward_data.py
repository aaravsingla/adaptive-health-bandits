import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Environment constants (mirrored from env.py — no import needed)
# ---------------------------------------------------------------------------
TARGET = np.array([8.0, 8.0])
HAZARDS = [
    {'center': np.array([5.0, 5.0]), 'radius': 1.5},
    {'center': np.array([2.0, 8.0]), 'radius': 1.0},
]

# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

def _in_hazard(pos):
    for h in HAZARDS:
        if np.linalg.norm(pos - h['center']) < h['radius']:
            return True
    return False


def _oracle_score(traj):
    """
    Score a trajectory (list of (state_np, action_np) tuples).
    Higher = more preferred.
    Penalizes distance to target and time spent in hazards.
    """
    final_pos  = traj[-1][0][:2]
    final_dist = np.linalg.norm(final_pos - TARGET)
    cost_rate  = sum(1 for (s, _) in traj if _in_hazard(s[:2])) / len(traj)

    # distance weighted 2x more than safety
    return -2.0 * final_dist - 1.5 * cost_rate


def _sample_trajectory(length=20, noise_scale=0.5):
    """
    Roll out a random policy from a random start.
    Returns list of (state_np [4], action_np [2]) tuples.
    """
    pos    = np.random.uniform(0, 10, size=2)
    target = TARGET
    traj   = []

    for _ in range(length):
        state  = np.concatenate([pos, target]).astype(np.float32)
        action = np.random.uniform(-1, 1, size=2).astype(np.float32)
        traj.append((state, action))
        pos = np.clip(pos + action * 0.5, 0, 10)

    return traj


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """
    Each item: (state_a, action_a, state_b, action_b, label)
    label = 1.0 if trajectory A preferred, 0.0 if B preferred.
    Uses Bradley-Terry model to sample soft labels from oracle scores.
    """

    def __init__(self, n_pairs=2000, traj_len=20):
        self.data = []

        for _ in range(n_pairs):
            traj_a = _sample_trajectory(traj_len)
            traj_b = _sample_trajectory(traj_len)

            score_a = _oracle_score(traj_a)
            score_b = _oracle_score(traj_b)

            # Bradley-Terry: P(A > B) = sigmoid(score_a - score_b)
            p_a_wins = 1.0 / (1.0 + np.exp(-(score_a - score_b)))
            label    = 1.0 if np.random.rand() < p_a_wins else 0.0

            # Summarize each trajectory as mean (state, action) pair
            # RewardModel sees a single (state, action) — we use the
            # trajectory mean as a compact representation
            sa_a = self._mean_sa(traj_a)
            sa_b = self._mean_sa(traj_b)

            self.data.append((sa_a[:4], sa_a[4:], sa_b[:4], sa_b[4:], label))

    def _mean_sa(self, traj):
        states  = np.stack([s for s, _ in traj])   # (T, 4)
        actions = np.stack([a for _, a in traj])    # (T, 2)
        return np.concatenate([states.mean(0), actions.mean(0)])  # (6,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sa, aa, sb, ab, label = self.data[idx]
        return (
            torch.FloatTensor(sa),
            torch.FloatTensor(aa),
            torch.FloatTensor(sb),
            torch.FloatTensor(ab),
            torch.FloatTensor([label]),
        )
"""
Action embeddings: per-frame future actions a_{t+1:t+K} -> frame-level
conditioning vectors added to the backbone's time embedding (paper Fig. 2).

MIND actions are four ternary axes [ws, ad, ud, lr] (0 = no-op, 1 = positive,
2 = negative), encoded as an 8-way multi-hot vector.
"""

import torch
import torch.nn as nn

ACTION_ATOM_DIM = 8


def raw_actions_to_multihot(actions):
    """[..., 4] ternary codes -> [..., 8] multi-hot
    ([ws+, ws-, ad+, ad-, ud+, ud-, lr+, lr-])."""
    if actions is None:
        return None
    if actions.shape[-1] != 4:
        raise ValueError(
            f"Expected raw actions with last dim 4, got {tuple(actions.shape)}")
    multihot = torch.zeros(*actions.shape[:-1], ACTION_ATOM_DIM,
                           device=actions.device, dtype=torch.float32)
    for axis in range(4):
        vals = actions[..., axis]
        multihot[..., 2 * axis] = (vals == 1).to(torch.float32)
        multihot[..., 2 * axis + 1] = (vals == 2).to(torch.float32)
    return multihot


class ActionEmbedding(nn.Module):
    """Multi-hot action atoms -> [B, F, dim] conditioning vectors."""

    def __init__(self, dim, num_action_atoms=ACTION_ATOM_DIM):
        super().__init__()
        self.dim = dim
        self.num_action_atoms = num_action_atoms
        self.atom_embedding = nn.Embedding(num_action_atoms, dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        nn.init.normal_(self.atom_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj[1].bias)
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, actions):
        multihot = raw_actions_to_multihot(actions)              # [B, F, 8]
        atom_emb = torch.matmul(multihot, self.atom_embedding.weight)
        return self.proj(atom_emb)


def attach_action_embedding(backbone, num_action_atoms=ACTION_ATOM_DIM):
    if hasattr(backbone, 'action_embedding'):
        return
    backbone.action_embedding = ActionEmbedding(
        dim=backbone.dim, num_action_atoms=num_action_atoms)


def encode_actions(backbone, actions):
    """Returns [B, F, dim] or None when no actions are given."""
    if actions is None:
        return None
    return backbone.action_embedding(actions)


def load_action_embedding(backbone, state):
    backbone.action_embedding.load_state_dict(state)

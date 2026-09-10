"""
Camera-Queryable Geometry Supervision (paper Sec. 3.3) — training only.

This module is *not* used at inference; it is included for completeness so
the released code covers every component of the paper. At training time it
constrains the memory m to store view-consistent geometry:

  1. A history frame i is sampled and its camera c_i is turned into one ray
     per patch of an H_g x W_g grid, rho(c_i)_{u,v} = [o_i, d_{i,u,v}] (Eq. 11).
  2. Rays are embedded by an MLP and added to a learned 2-D grid positional
     embedding e_{u,v} to form the queries q_i (Eq. 12).
  3. The geometry head decodes q_i against the memory:
         G_hat_i = FFN(SelfAttn(CrossAttn(q_i, m)))                 (Eq. 13)
  4. G_hat_i is aligned to the VGGT encoder feature map G_i of the same frame
     with a per-patch cosine loss (Eq. 14):
         L_geo = 1 - mean_{u,v} cos(G_hat_{i,u,v}, G_{i,u,v})
     and the total objective is L = L_FM + lambda * L_geo (Eq. 20).

The VGGT teacher itself is external (https://github.com/facebookresearch/vggt):
its encoder's per-frame patch tokens are precomputed offline and passed in as
`target`. Both the head and the teacher are discarded after training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_ray_map(camera_poses, grid_h, grid_w, focal=1.0):
    """
    One ray per patch centre, in the (relative) world frame (Eq. 11).

    Args:
        camera_poses: [B, T, 12] = [t(3), R_row_major(9)], camera-to-world.
        grid_h, grid_w: geometry feature grid (VGGT patch grid).
        focal: canonical pinhole focal in normalized image coordinates with
               patch centres in [-0.5, 0.5]; the ray MLP and positional
               embedding absorb any offset from the true intrinsics.
    Returns:
        [B, T, H_g*W_g, 6] = (origin_world, direction_world)
    """
    B, T, _ = camera_poses.shape
    t = camera_poses[..., :3]
    R = camera_poses[..., 3:].reshape(B, T, 3, 3)

    ys = torch.linspace(-0.5, 0.5, grid_h, device=camera_poses.device, dtype=camera_poses.dtype)
    xs = torch.linspace(-0.5, 0.5, grid_w, device=camera_poses.device, dtype=camera_poses.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    d_cam = torch.stack([xx, yy, torch.full_like(xx, float(focal))], dim=-1).reshape(-1, 3)
    d_cam = F.normalize(d_cam, p=2, dim=-1, eps=1e-6)

    d_world = torch.einsum('btij,nj->btni', R, d_cam)
    o_world = t.unsqueeze(2).expand(-1, -1, d_world.size(2), -1)
    return torch.cat([o_world, d_world], dim=-1)


class GeometryDecoderBlock(nn.Module):
    """Pre-LN cross-attention (queries -> memory) + self-attention + FFN."""

    def __init__(self, dim, num_heads, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim))

    def forward(self, q, kv):
        nkv = self.cross_norm_kv(kv)
        q = q + self.cross_attn(self.cross_norm_q(q), nkv, nkv, need_weights=False)[0]
        nq = self.self_norm(q)
        q = q + self.self_attn(nq, nq, nq, need_weights=False)[0]
        return q + self.ffn(self.ffn_norm(q))


class GeometryHead(nn.Module):
    """
    Camera-queryable geometry head: (memory tokens, camera) -> G_hat.

    Args:
        memory_dim: dimension of the memory tokens (= backbone hidden dim).
        feature_dim: VGGT encoder feature dimension.
        grid_hw: (H_g, W_g) VGGT patch grid.
        inner_dim / num_heads / num_blocks / ffn_mult: decoder size
            (paper default: 256 / 8 / 1 / 4 — deliberately small so the
            geometric alignment burden stays on the memory, not the head).
        focal: canonical focal for the ray map.
    """

    def __init__(self, memory_dim, feature_dim, grid_hw, inner_dim=256,
                 num_heads=8, num_blocks=1, ffn_mult=4, focal=1.0):
        super().__init__()
        self.grid_h, self.grid_w = int(grid_hw[0]), int(grid_hw[1])
        self.num_patches = self.grid_h * self.grid_w
        self.inner_dim = int(inner_dim)
        self.feature_dim = int(feature_dim)
        self.focal = float(focal)

        self.ray_mlp = nn.Sequential(
            nn.Linear(6, self.inner_dim), nn.GELU(), nn.Linear(self.inner_dim, self.inner_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.inner_dim))
        self.memory_proj = nn.Linear(int(memory_dim), self.inner_dim)
        self.blocks = nn.ModuleList([
            GeometryDecoderBlock(self.inner_dim, num_heads, ffn_mult) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.feature_dim)
        # Zero-init so the head is a no-op at step 0 and does not perturb the
        # memory before it has warmed up.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, memory_tokens, camera_poses):
        """
        Args:
            memory_tokens: [B, N_m, memory_dim]
            camera_poses:  [B, T_q, 12] cameras of the queried history frames
        Returns:
            G_hat: [B, T_q, H_g*W_g, feature_dim]
        """
        B, T_q, _ = camera_poses.shape
        M = memory_tokens.shape[1]

        rays = build_ray_map(camera_poses, self.grid_h, self.grid_w, focal=self.focal)
        q = (self.ray_mlp(rays) + self.pos_embed).reshape(B * T_q, self.num_patches, self.inner_dim)
        kv = self.memory_proj(memory_tokens).unsqueeze(1).expand(-1, T_q, -1, -1)
        kv = kv.reshape(B * T_q, M, self.inner_dim)

        for block in self.blocks:
            q = block(q, kv)
        out = self.out_proj(self.out_norm(q))
        return out.reshape(B, T_q, self.num_patches, self.feature_dim)


def geometry_loss(pred, target, num_valid_frames):
    """
    L_geo (Eq. 14): per-patch cosine distance between decoded and VGGT
    features, averaged over patches, valid frames and the batch.

    Args:
        pred:   [B, T, N_g, D] output of GeometryHead
        target: [B, T, N_g, D] precomputed VGGT encoder tokens (no grad)
        num_valid_frames: [B] number of valid (non-padded) frames per sample
    Returns:
        scalar in [0, 2]
    """
    pred = pred.float()
    target = target.float()
    total = pred.new_zeros(())
    count = 0
    for i in range(pred.shape[0]):
        T_i = int(num_valid_frames[i])
        if T_i <= 0:
            continue
        p = F.normalize(pred[i, :T_i], p=2, dim=-1, eps=1e-6)
        g = F.normalize(target[i, :T_i], p=2, dim=-1, eps=1e-6)
        cos_per_frame = (p * g).sum(dim=-1).mean(dim=-1)      # [T_i]
        total = total + (1.0 - cos_per_frame).sum()
        count += T_i
    return total / count if count else total


def sample_geometry_targets(history_cameras, vggt_tokens, num_history_frames, generator=None):
    """
    Uniformly sample one history frame per sample for the single-frame
    estimator of Sec. 3.3 (unbiased for the all-frame objective while the
    per-step cost stays independent of history length).

    Args:
        history_cameras:    [B, T_max, 12]
        vggt_tokens:        [B, T_max, N_g, D] precomputed teacher features
        num_history_frames: [B]
    Returns:
        cameras [B, 1, 12], targets [B, 1, N_g, D], valid [B]
    """
    B = history_cameras.shape[0]
    idx = torch.stack([
        torch.randint(0, max(int(n), 1), (1,), generator=generator, device=history_cameras.device)
        for n in num_history_frames
    ]).squeeze(1)
    ar = torch.arange(B, device=history_cameras.device)
    return (history_cameras[ar, idx].unsqueeze(1),
            vggt_tokens[ar, idx].unsqueeze(1),
            torch.ones(B, dtype=torch.long, device=history_cameras.device))

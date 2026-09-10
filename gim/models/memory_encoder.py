"""
Implicit Memory Encoder (paper Sec. 3.2).

Maps a variable-length, camera-indexed history to a fixed number of memory
tokens. History latents are patchified by the backbone's patch embedding into
history tokens h_{i,p}; a linear camera embedding E_c(c_i) is added to the
history tokens only (Eq. 6), keeping the memory queries pose-free. N_m
learnable memory queries Q_0 are concatenated with the camera-aware history
into Z_0 = [Q_0; H~] and refined by two MemoryEncoderBlocks, each a compact
self-attention branch followed by a full-resolution feed-forward branch
(Eq. 7-8). The first N_m tokens of the output are the memory m.

Compact / Expand (Eq. 9-10): with compact_stride s > 1 each s x s block of
tokens is flattened through a shared linear layer before attention and
inverted after, so attention runs on (N_m + T H_p W_p) / s^2 tokens while the
residual stream and FFN stay at full resolution. Separate RoPE grids are used
for the query and history segments.

Initialization: memory queries start as the learnable parameters plus a
uniform temporal sample of the raw history, and all output projections are
zero-initialized, so at step 0 the memory equals a uniformly downsampled
history that the backbone already understands.
"""

import torch
import torch.nn as nn

from wan.modules.attention import flash_attention
from wan.modules.model import WanRMSNorm, rope_apply


class MemoryEncoderBlock(nn.Module):
    """Pre-norm compact self-attention + pre-norm FFN with residuals."""

    def __init__(self, dim, num_heads, ffn_dim, compact_stride=1, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.compact_stride = int(compact_stride)

        self.norm_sa = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

        if self.compact_stride > 1:
            s = self.compact_stride
            self.compact_proj = nn.Linear(s * s * dim, dim, bias=True)
            self.expand_proj = nn.Linear(dim, s * s * dim, bias=True)
        else:
            self.compact_proj = None
            self.expand_proj = None

        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        if self.expand_proj is not None:
            nn.init.zeros_(self.expand_proj.bias)

    def compact(self, tokens, H_p, W_p, T):
        """[B, T*H_p*W_p, d] -> [B, T*(H_p/s)*(W_p/s), d]  (Eq. 9)."""
        B, _, d = tokens.shape
        s = self.compact_stride
        H_c, W_c = H_p // s, W_p // s
        x = tokens.view(B, T, H_c, s, W_c, s, d)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, T, H_c, W_c, s * s * d)
        x = self.compact_proj(x)
        return x.view(B, T * H_c * W_c, d)

    def expand(self, tokens_c, H_p, W_p, T):
        """Inverse of compact (Eq. 10)."""
        B, _, d = tokens_c.shape
        s = self.compact_stride
        H_c, W_c = H_p // s, W_p // s
        x = tokens_c.view(B, T, H_c, W_c, d)
        x = self.expand_proj(x)
        x = x.view(B, T, H_c, W_c, s, s, d)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(B, T * H_p * W_p, d)

    def forward(self, z, query_grid, history_grid, freqs, num_query_tokens,
                num_query_tokens_c, seq_lens, H_p, W_p, num_memory_frames,
                T_max):
        """
        Args:
            z:                  [B, N_m*S + T_max*S, d] full-res [queries; history]
            query_grid:         [B, 3] compact-domain RoPE grid of the queries
            history_grid:       [B, 3] compact-domain RoPE grid of the history
            freqs:              backbone RoPE frequencies
            num_query_tokens:   N_m * S (full-res query token count)
            num_query_tokens_c: N_m * S / s^2 (compact-domain count)
            seq_lens:           [B] valid attention length per sample
            H_p, W_p:           full-res patch grid per frame
            num_memory_frames:  temporal extent of the queries
            T_max:              padded history frame count
        """
        n, d = self.num_heads, self.head_dim
        B = z.shape[0]
        M, M_c = num_query_tokens, num_query_tokens_c

        h_norm = self.norm_sa(z)
        if self.compact_proj is not None:
            q_c = self.compact(h_norm[:, :M], H_p, W_p, num_memory_frames)
            h_c = self.compact(h_norm[:, M:], H_p, W_p, T_max)
            h_in = torch.cat([q_c, h_c], dim=1)
        else:
            h_in = h_norm

        L = h_in.shape[1]
        q = self.norm_q(self.q_proj(h_in)).view(B, L, n, d)
        k = self.norm_k(self.k_proj(h_in)).view(B, L, n, d)
        v = self.v_proj(h_in).view(B, L, n, d)

        # Separate rotary grids for the query and history segments.
        q = torch.cat([rope_apply(q[:, :M_c], query_grid, freqs),
                       rope_apply(q[:, M_c:], history_grid, freqs)], dim=1)
        k = torch.cat([rope_apply(k[:, :M_c], query_grid, freqs),
                       rope_apply(k[:, M_c:], history_grid, freqs)], dim=1)

        attn = flash_attention(q=q, k=k, v=v, k_lens=seq_lens)
        attn = self.o_proj(attn.flatten(2))

        if self.expand_proj is not None:
            attn = torch.cat([
                self.expand(attn[:, :M_c], H_p, W_p, num_memory_frames),
                self.expand(attn[:, M_c:], H_p, W_p, T_max),
            ], dim=1)

        z = z + attn
        z = z + self.ffn(self.norm_ffn(z))
        return z


class MemoryEncoder(nn.Module):
    """
    Implicit memory encoder: [Q_0; H~] -> m (fixed size, pose-free).

    Args:
        num_memory_frames: temporal extent of the memory (N_m = frames * S)
        spatial_tokens_per_frame: S = H_p * W_p (880 for 480x832 with 2x2 patches)
        dim: hidden dimension (must match the backbone)
        num_heads: attention heads (must match the backbone for shared RoPE)
        ffn_dim: FFN hidden dim (default 4 * dim)
        num_layers: number of MemoryEncoderBlocks (paper: 2)
        compact_stride: s in Eq. 9-10 (paper default: 2)
    """

    def __init__(self, num_memory_frames=20, spatial_tokens_per_frame=880,
                 dim=1536, num_heads=12, ffn_dim=None, num_layers=2,
                 compact_stride=2):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = dim * 4

        self.num_memory_frames = num_memory_frames
        self.spatial_tokens = spatial_tokens_per_frame
        self.dim = dim
        self.compact_stride = int(compact_stride)
        self.num_memory_tokens = num_memory_frames * spatial_tokens_per_frame

        self.memory_queries = nn.Parameter(
            torch.randn(self.num_memory_tokens, dim) * 0.02)

        # E_c: linear camera embedding added to history tokens only (Eq. 6).
        self.camera_embed = nn.Linear(12, dim, bias=True)
        nn.init.zeros_(self.camera_embed.weight)
        nn.init.zeros_(self.camera_embed.bias)

        self.blocks = nn.ModuleList([
            MemoryEncoderBlock(dim, num_heads, ffn_dim,
                               compact_stride=self.compact_stride)
            for _ in range(num_layers)
        ])

    def init_queries_from_history(self, history_tokens, num_history_frames):
        """Uniform temporal sample of the raw history, one per memory frame."""
        B = history_tokens.shape[0]
        S = self.spatial_tokens
        F = self.num_memory_frames
        samples = []
        for i in range(B):
            T_i = max(int(num_history_frames[i].item()), 1)
            frame_idx = torch.linspace(0, T_i - 1, F).long()
            token_idx = (frame_idx.unsqueeze(1) * S
                         + torch.arange(S).unsqueeze(0)).reshape(-1)
            samples.append(history_tokens[i, token_idx])
        return torch.stack(samples)

    def forward(self, history_tokens, num_history_frames, freqs, H_p, W_p,
                history_cameras=None):
        """
        Args:
            history_tokens:     [B, T_max*S, d] patchified history latents
            num_history_frames: [B] valid history frames per sample
            freqs:              backbone RoPE frequencies
            H_p, W_p:           patch grid of one frame
            history_cameras:    [B, T_max, 12] per-frame camera poses
        Returns:
            memory_tokens: [B, N_m*S, d]
        """
        s = self.compact_stride
        if s > 1 and (H_p % s != 0 or W_p % s != 0):
            raise ValueError(
                f"compact_stride={s} does not divide H_p={H_p} or W_p={W_p}.")

        B = history_tokens.shape[0]
        S = self.spatial_tokens
        assert S == H_p * W_p, f"spatial_tokens={S} != H_p*W_p={H_p * W_p}"
        M = self.num_memory_tokens
        T_max = history_tokens.shape[1] // S

        queries = (self.memory_queries.unsqueeze(0).expand(B, -1, -1)
                   + self.init_queries_from_history(history_tokens,
                                                    num_history_frames))

        if history_cameras is not None:
            cam_emb = torch.zeros_like(history_tokens)
            for i in range(B):
                T_i = int(num_history_frames[i].item())
                if T_i > 0:
                    cam_i = history_cameras[i, :T_i]
                    cam_i = cam_i.unsqueeze(1).expand(-1, S, -1)
                    cam_emb[i, :T_i * S] = self.camera_embed(
                        cam_i.reshape(T_i * S, 12))
            history_tokens = history_tokens + cam_emb

        z = torch.cat([queries, history_tokens], dim=1)

        H_c, W_c = H_p // s, W_p // s
        S_c = H_c * W_c
        M_c = self.num_memory_frames * S_c
        query_grid = torch.tensor(
            [[self.num_memory_frames, H_c, W_c]], dtype=torch.long).expand(B, -1)
        history_grid = torch.stack([
            torch.tensor([int(t.item()), H_c, W_c], dtype=torch.long)
            for t in num_history_frames
        ])
        seq_lens = (M_c + num_history_frames.cpu() * S_c).to(torch.long)

        for block in self.blocks:
            z = block(z, query_grid, history_grid, freqs, M, M_c, seq_lens,
                      H_p, W_p, self.num_memory_frames, T_max)

        return z[:, :M]


# Mapping from training-time checkpoint keys to the released module layout.
LEGACY_KEY_MAP = {
    "queries": "memory_queries",
    "cam_proj.": "camera_embed.",
    "layers.": "blocks.",
    ".compact.": ".compact_proj.",
    ".expand.": ".expand_proj.",
}


def convert_legacy_state_dict(state):
    """Rename keys saved by the training code to the released naming."""
    out = {}
    for k, v in state.items():
        nk = k
        for old, new in LEGACY_KEY_MAP.items():
            if old.endswith(".") and not old.startswith("."):
                if nk.startswith(old):
                    nk = new + nk[len(old):]
            elif old.startswith("."):
                nk = nk.replace(old, new)
            elif nk == old:
                nk = new
        out[nk] = v
    return out

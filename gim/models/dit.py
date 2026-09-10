"""
Memory-conditioned diffusion backbone (Wan2.1 DiT).

The memory tokens m produced by the MemoryEncoder are concatenated along the
temporal axis with the noisy target latents (paper Sec. 3.2), so the backbone
sees [memory tokens | target tokens] as one sequence. Two lightweight
conditioning paths are attached to the pretrained WanModel:

  - camera injection: every DiT block owns a zero-initialized Linear(12, dim)
    that maps the 12-D camera pose of each token's frame into the residual
    stream right after self-attention;
  - action embeddings: per-frame action vectors added to the time embedding
    of the target tokens (see action_embedding.py).

Provides:
  - attach_camera_proj / load_camera_proj
  - patchify_history:      history latents -> history tokens for the encoder
  - forward_with_memory:   one denoising step conditioned on memory tokens
  - generate_chunk:        DPM-Solver++ sampling of the next K latents
"""

import torch
import torch.nn as nn
from torch import amp

from wan.modules.model import WanModel, sinusoidal_embedding_1d

from gim.models.action_embedding import encode_actions


# ---------------------------------------------------------------------------
# Camera injection
# ---------------------------------------------------------------------------

def attach_camera_proj(backbone):
    """Add a zero-initialized Linear(12, dim) to every DiT block."""
    for block in backbone.blocks:
        proj = nn.Linear(12, backbone.dim, bias=True)
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)
        block.camera_proj = proj


def load_camera_proj(backbone, state):
    """Load per-block camera projections from a flat state dict
    ({'blocks.{i}.camera_proj.weight', ...})."""
    for i, block in enumerate(backbone.blocks):
        block.camera_proj.weight.data.copy_(state[f'blocks.{i}.camera_proj.weight'])
        block.camera_proj.bias.data.copy_(state[f'blocks.{i}.camera_proj.bias'])


def block_forward_with_camera(block, x, camera_raw, e0, seq_lens, grid_sizes,
                              freqs, context, context_lens=None):
    """One WanAttentionBlock with the camera embedding added to the residual
    stream after self-attention."""
    assert e0.dtype == torch.float32
    with amp.autocast('cuda', dtype=torch.float32):
        if e0.dim() == 3:
            e = (block.modulation + e0).chunk(6, dim=1)
        else:
            e = (block.modulation.unsqueeze(1) + e0).unbind(dim=2)

    sa_input = block.norm1(x).float() * (1 + e[1]) + e[0]
    y = block.self_attn(sa_input, seq_lens, grid_sizes, freqs)
    with amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2]

    x = x + block.camera_proj(camera_raw).to(x.dtype)

    x = x + block.cross_attn(block.norm3(x), context, context_lens)

    y = block.ffn(block.norm2(x).float() * (1 + e[4]) + e[3])
    with amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[5]
    return x


def head_forward_with_token_time(backbone, x, token_e):
    """Wan output head with token-level time conditioning."""
    assert token_e.dtype == torch.float32
    with amp.autocast('cuda', dtype=torch.float32):
        e = (backbone.head.modulation.unsqueeze(1) + token_e.unsqueeze(2)).unbind(dim=2)
        x = backbone.head.head(backbone.head.norm(x) * (1 + e[1]) + e[0])
    return x


# ---------------------------------------------------------------------------
# History tokens
# ---------------------------------------------------------------------------

def patchify_history(backbone, history_latents, num_history_frames):
    """
    Per-sample patchification of variable-length history, padded to the
    longest history in the batch.

    Args:
        history_latents:    [B, C, T_max, H, W]
        num_history_frames: [B]
    Returns:
        history_tokens: [B, T_max*S, dim]
    """
    B = history_latents.shape[0]
    tokens = []
    for i in range(B):
        T_i = int(num_history_frames[i].item())
        emb = backbone.patch_embedding(history_latents[i:i + 1, :, :T_i])
        tokens.append(emb.flatten(2).transpose(1, 2).squeeze(0))
    max_len = max(t.shape[0] for t in tokens)
    dim = tokens[0].shape[1]
    padded = []
    for t in tokens:
        if t.shape[0] < max_len:
            t = torch.cat([t, t.new_zeros(max_len - t.shape[0], dim)], dim=0)
        padded.append(t)
    return torch.stack(padded)


# ---------------------------------------------------------------------------
# Denoising step conditioned on memory tokens
# ---------------------------------------------------------------------------

def forward_with_memory(
    backbone: WanModel,
    memory_tokens,
    noisy_target,
    t,
    text_embedding,
    target_cameras,
    target_actions=None,
):
    """
    Args:
        backbone:       WanModel with camera_proj / action_embedding attached
        memory_tokens:  [B, N_m*S, dim]
        noisy_target:   [B, C, K, H', W']
        t:              [B] diffusion timesteps
        text_embedding: [B, 512, text_dim]
        target_cameras: [B, K, 12] relative camera poses of the target frames
        target_actions: [B, K, 4] raw action codes (optional)
    Returns:
        velocity prediction for the target latents: [B, C_out, K, H', W']
    """
    device = backbone.patch_embedding.weight.device
    if backbone.freqs.device != device:
        backbone.freqs = backbone.freqs.to(device)

    B, M = memory_tokens.shape[:2]

    target_patched = backbone.patch_embedding(noisy_target)
    K_p, H_p, W_p = target_patched.shape[2:]
    S = H_p * W_p
    target_tokens = target_patched.flatten(2).transpose(1, 2)

    x = torch.cat([memory_tokens.to(target_tokens.dtype), target_tokens], dim=1)
    L = x.shape[1]

    num_mem_frames = M // S
    F_total = num_mem_frames + K_p
    grid_sizes = torch.tensor([[F_total, H_p, W_p]], dtype=torch.long).expand(B, -1)
    seq_lens = torch.full((B,), L, dtype=torch.long)

    # Memory tokens have no camera of their own; tile the target cameras so
    # every token position carries a well-defined pose.
    mem_cam_idx = torch.arange(num_mem_frames, device=target_cameras.device) % K_p
    cam_frames = torch.cat([target_cameras[:, mem_cam_idx], target_cameras], dim=1)
    camera_raw = cam_frames.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, L, 12)

    with amp.autocast('cuda', dtype=torch.float32):
        base_e = backbone.time_embedding(
            sinusoidal_embedding_1d(backbone.freq_dim, t).float())
    action_cond = encode_actions(backbone, target_actions)
    if action_cond is None:
        token_action = base_e.new_zeros(B, L, backbone.dim)
    else:
        tgt = action_cond.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, K_p * S, backbone.dim)
        token_action = torch.cat([tgt.new_zeros(B, M, backbone.dim), tgt], dim=1)

    token_e = base_e.unsqueeze(1).expand(-1, L, -1) + token_action.to(base_e.dtype)
    with amp.autocast('cuda', dtype=torch.float32):
        e0 = backbone.time_projection(token_e).unflatten(-1, (6, backbone.dim))

    context = backbone.text_embedding(text_embedding)

    for block in backbone.blocks:
        x = block_forward_with_camera(block, x, camera_raw, e0, seq_lens,
                                      grid_sizes, backbone.freqs, context)

    x = head_forward_with_token_time(backbone, x, token_e)

    c = backbone.out_dim
    out = []
    for i in range(B):
        u = x[i, :F_total * S].view(F_total, H_p, W_p, *backbone.patch_size, c)
        u = torch.einsum('fhwpqrc->cfphqwr', u)
        u = u.reshape(c, *[a * b for a, b in zip((F_total, H_p, W_p), backbone.patch_size)])
        out.append(u[:, num_mem_frames * backbone.patch_size[0]:].float())
    return torch.stack(out)


# ---------------------------------------------------------------------------
# Chunk sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_chunk(
    backbone,
    memory_encoder,
    history_latents,
    num_history_frames,
    text_embedding,
    camera_poses,
    target_actions=None,
    num_steps=50,
    chunk_latents=20,
    shift=5.0,
    num_timesteps=1000,
    guide_scale=1.0,
):
    """
    Sample the next `chunk_latents` latents given the (pruned) history.

    Args:
        history_latents:    [B, C, T_max, H, W] clean history latents
        num_history_frames: [B]
        text_embedding:     [B, 512, text_dim]
        camera_poses:       [B, T_max + K, 12] relative poses for history + target
        target_actions:     [B, K, 4] or None
        guide_scale:        != 1.0 enables text CFG (memory kept, text dropped)
    Returns:
        generated latents [B, C, K, H, W]
    """
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )

    device = history_latents.device
    B, C, _, H, W = history_latents.shape
    ps = backbone.patch_size
    H_p, W_p = H // ps[1], W // ps[2]
    if backbone.freqs.device != device:
        backbone.freqs = backbone.freqs.to(device)

    history_tokens = patchify_history(backbone, history_latents, num_history_frames)

    max_hist = max(int(n.item()) for n in num_history_frames)
    target_cameras, history_cameras = [], []
    for i in range(B):
        T_i = int(num_history_frames[i].item())
        target_cameras.append(camera_poses[i, T_i:T_i + chunk_latents])
        hc = camera_poses[i, :T_i]
        if T_i < max_hist:
            hc = torch.cat([hc, hc.new_zeros(max_hist - T_i, 12)], dim=0)
        history_cameras.append(hc)
    target_cameras = torch.stack(target_cameras)
    history_cameras = torch.stack(history_cameras)

    # Encode the memory once; it is reused across all denoising steps.
    memory_tokens = memory_encoder(
        history_tokens, num_history_frames, freqs=backbone.freqs,
        H_p=H_p, W_p=W_p, history_cameras=history_cameras)

    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=num_timesteps, shift=1, use_dynamic_shifting=False)
    sigmas = get_sampling_sigmas(num_steps, shift)
    timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sigmas)

    z = torch.randn(B, C, chunk_latents, H, W, device=device)
    text_uncond = torch.zeros_like(text_embedding)

    for t_val in timesteps:
        t_batch = t_val.unsqueeze(0).expand(B)
        v = forward_with_memory(backbone, memory_tokens, z, t_batch,
                                text_embedding, target_cameras, target_actions)
        if guide_scale != 1.0:
            v_uncond = forward_with_memory(backbone, memory_tokens, z, t_batch,
                                           text_uncond, target_cameras, target_actions)
            v = v_uncond + guide_scale * (v - v_uncond)
        z = scheduler.step(v, t_val, z, return_dict=False)[0]

    return z

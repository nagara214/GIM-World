"""
Camera pose utilities.

Poses are 12-D vectors c_i = [x, y, z, R.flatten()] (row-major 3x3 rotation).
MIND provides Unreal Engine poses in centimetres and degrees; positions are
divided by POSITION_SCALE (cm -> m) before entering the model, and every
chunk is expressed relative to its first history frame.

Also contains the mapping between raw video frames and Wan VAE latents:
latent 0 covers raw frame 0, latent j (j >= 1) covers raw frames
[1 + 4(j-1), 4j].
"""

import math

import numpy as np
import torch

POSITION_SCALE = 100.0


# ---------------------------------------------------------------------------
# Euler / pose construction
# ---------------------------------------------------------------------------

def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """RPY (degrees) -> 3x3 rotation, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (math.radians(float(v)) for v in (roll_deg, pitch_deg, yaw_deg))
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]], dtype=np.float32)
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                   [0, 1, 0],
                   [-math.sin(pitch), 0, math.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0],
                   [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def poses_from_action_json(frames, perspective="1st"):
    """
    Per-frame 12-D poses from the `data` list of a MIND-style action.json.

    First-person uses actor_pos / actor_rpy; third-person uses
    camera_pos / camera_rpy when present (falls back to the actor).

    Returns: np.ndarray [T, 12].
    """
    rows = []
    for fr in frames:
        if perspective == "3rd" and "camera_pos" in fr:
            pos, rpy = fr["camera_pos"], fr["camera_rpy"]
        else:
            pos, rpy = fr["actor_pos"], fr["actor_rpy"]
        t = np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])],
                     dtype=np.float32)
        R = euler_to_rotation_matrix(rpy["x"], rpy["y"], rpy["z"])
        rows.append(np.concatenate([t, R.flatten()]))
    return np.stack(rows)


def actions_from_action_json(frames):
    """Per-frame raw action codes [T, 4] = [ws, ad, ud, lr]."""
    return torch.tensor(
        [[int(fr.get("ws", 0)), int(fr.get("ad", 0)),
          int(fr.get("ud", 0)), int(fr.get("lr", 0))] for fr in frames],
        dtype=torch.long)


def convert_to_relative(poses, ref_idx=0):
    """
    Express absolute 12-D poses relative to poses[ref_idx].

    Args:
        poses: [N, 12] tensor
    Returns:
        [N, 12] tensor
    """
    rt = poses.numpy()
    ref = rt[ref_idx]
    R_ref_inv = ref[3:].reshape(3, 3).T
    t_ref_inv = -R_ref_inv @ ref[:3].reshape(3, 1)
    out = []
    for row in rt:
        t = row[:3].reshape(3, 1)
        R = row[3:].reshape(3, 3)
        out.append(np.concatenate([(R_ref_inv @ t + t_ref_inv).flatten(),
                                   (R_ref_inv @ R).flatten()]))
    return torch.tensor(np.stack(out), dtype=torch.float32)


def relative_camera_chunk(camera_poses, history_indices, target_indices):
    """Relative poses for [history | target] latent frames, [1, N, 12]."""
    idx = list(history_indices) + list(target_indices)
    chunk = camera_poses[idx].clone()
    chunk[:, :3] /= POSITION_SCALE
    return convert_to_relative(chunk, ref_idx=0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Raw frame <-> latent alignment (Wan VAE temporal compression 1 + 4(T-1))
# ---------------------------------------------------------------------------

def latent_to_raw_index(num_latents, num_raw_frames=None):
    """Representative raw frame index for each latent."""
    idx = []
    for i in range(num_latents):
        r = 0 if i == 0 else 1 + 4 * (i - 1)
        if num_raw_frames is not None:
            r = min(r, num_raw_frames - 1)
        idx.append(r)
    return idx


def latent_spans(num_latents, num_raw_frames=None):
    """Raw-frame [start, end] covered by each latent."""
    spans = []
    for i in range(num_latents):
        if i == 0:
            s = e = 0
        else:
            s, e = 1 + 4 * (i - 1), 4 * i
        if num_raw_frames is not None:
            s, e = min(s, num_raw_frames - 1), min(e, num_raw_frames - 1)
        spans.append((s, e))
    return spans


def raw_frame_to_latent(frame_idx, num_raw_frames=None):
    """Latent index containing raw frame `frame_idx`."""
    if frame_idx <= 0:
        return 0
    if num_raw_frames is not None:
        frame_idx = min(frame_idx, num_raw_frames - 1)
    return 1 + (frame_idx - 1) // 4


def num_latents_for_frames(num_raw_frames):
    """Latents produced by the Wan VAE for a raw frame count (trailing 1-3
    frames that do not complete a 4-frame group are dropped)."""
    if num_raw_frames <= 0:
        return 0
    return 1 + (num_raw_frames - 1) // 4

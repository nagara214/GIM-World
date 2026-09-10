"""
Video I/O and streaming Wan VAE encode / decode.

The Wan VAE is temporally causal with a 1 + 4(T-1) frame grouping. Both
helpers below keep the VAE's temporal cache alive across the whole clip so
that long videos can be processed with bounded memory and without the
per-call cache reset of the public WanVAE.encode / decode API.
"""

from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# mp4 read / write
# ---------------------------------------------------------------------------

def iter_video_frames(path, max_frames=None):
    """Yield RGB uint8 frames [H, W, 3] from an mp4 via imageio-ffmpeg."""
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        for i, frame in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise RuntimeError(f"Unexpected frame shape {tuple(frame.shape)}")
            yield np.asarray(frame, dtype=np.uint8)
    finally:
        reader.close()


def count_video_frames(path):
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        n = reader.count_frames()
    finally:
        reader.close()
    return int(n)


def resize_frame(rgb, width, height):
    """uint8 RGB -> float32 [H, W, 3] in [-1, 1] at the target size."""
    img = Image.fromarray(rgb, mode="RGB").resize(
        (width, height), resample=Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0


def save_video(video, path, fps=24):
    """video: [3, T, H, W] float in [0, 1] -> H.264 mp4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                output_params=["-crf", "18"])
    try:
        for t in range(video.shape[1]):
            frame = video[:, t].permute(1, 2, 0).cpu().numpy()
            writer.append_data((frame * 255).clip(0, 255).astype(np.uint8))
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Streaming VAE encode
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_video_streaming(vae, frames, device, height, width,
                           buffer_frames=257):
    """
    Encode an iterable of RGB uint8 frames into Wan VAE latents.

    Frames are grouped globally as [0], [1:5], [5:9], ... with the encoder
    cache kept across groups. Trailing 1-3 frames that do not complete a
    group are dropped (they are not representable by the temporal
    downsamplers).

    Returns:
        latents: [C, T_lat, H/8, W/8] float32 on CPU
        num_encoded_frames: raw frames actually covered by the latents
    """
    model = vae.model
    autocast = str(device).startswith("cuda")
    chunks, buf = [], []
    next_group = 1

    def _encode_group(group_frames):
        arr = np.ascontiguousarray(np.stack(group_frames).transpose(3, 0, 1, 2))
        x = torch.from_numpy(arr).unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(dtype=vae.dtype, enabled=autocast):
            model._enc_conv_idx = [0]
            out = model.encoder(x, feat_cache=model._enc_feat_map,
                                feat_idx=model._enc_conv_idx)
            mu, _ = model.conv1(out).chunk(2, dim=1)
            scale = vae.scale
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, model.z_dim, 1, 1, 1)) \
                    * scale[1].view(1, model.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        return mu.float().squeeze(0).cpu()

    def _flush():
        nonlocal next_group
        while len(buf) >= next_group:
            group = buf[:next_group]
            del buf[:next_group]
            chunks.append(_encode_group(group))
            next_group = 4

    model.clear_cache()
    try:
        for rgb in frames:
            buf.append(resize_frame(rgb, width=width, height=height))
            if len(buf) >= max(4, buffer_frames):
                _flush()
        _flush()
    finally:
        model.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not chunks:
        raise RuntimeError("No frames were encoded.")
    latents = torch.cat(chunks, dim=1)
    num_encoded_frames = 1 + 4 * (latents.shape[1] - 1)
    return latents, num_encoded_frames


# ---------------------------------------------------------------------------
# Streaming VAE decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_latents_streaming(vae, latents, device, chunk_size=20):
    """
    Decode [C, T, H', W'] latents to video [3, 1 + 4(T-1), H, W] in [0, 1],
    keeping the decoder cache across latent chunks.
    """
    if latents.ndim != 4 or latents.shape[1] == 0:
        raise ValueError(f"Expected non-empty latents [C, T, H, W], got {tuple(latents.shape)}")
    model = vae.model
    autocast = str(device).startswith("cuda")
    outputs = []

    model.clear_cache()
    try:
        with torch.cuda.amp.autocast(dtype=vae.dtype, enabled=autocast):
            for start in range(0, latents.shape[1], max(1, chunk_size)):
                z = latents[:, start:start + chunk_size].unsqueeze(0).to(device)
                scale = vae.scale
                if isinstance(scale[0], torch.Tensor):
                    z = z / scale[1].view(1, model.z_dim, 1, 1, 1) \
                        + scale[0].view(1, model.z_dim, 1, 1, 1)
                else:
                    z = z / scale[1] + scale[0]
                x = model.conv2(z)
                for i in range(x.shape[2]):
                    model._conv_idx = [0]
                    out = model.decoder(x[:, :, i:i + 1], feat_cache=model._feat_map,
                                        feat_idx=model._conv_idx)
                    outputs.append(out.float().clamp_(-1, 1).squeeze(0).cpu())
                del z, x
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        model.clear_cache()

    video = torch.cat(outputs, dim=1)
    return (video + 1.0).div_(2.0).clamp_(0, 1)

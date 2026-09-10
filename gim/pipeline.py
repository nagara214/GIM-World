"""
End-to-end GIM-World inference.

    pipe = GIMWorldPipeline.from_pretrained(gim_dir, wan_dir, device="cuda")
    video = pipe("path/to/clip_dir")          # video.mp4 + action.json inside

A clip directory follows the MIND layout: `video.mp4` holds at least the
observed prefix (frames [0, mark_time)), and `action.json` holds
`mark_time`, `total_time`, `caption` and one entry per frame with the camera
pose and action codes (see gim/utils/trajectory.py for authoring your own).

Rollout (paper Sec. 3.1 / 3.5): the prefix is encoded by the Wan VAE into
history latents; at each step the history is pruned to `pruning_budget`
latents with information-guided pruning, encoded by the MemoryEncoder into
fixed-size memory tokens, and the backbone samples the next `chunk_latents`
latents conditioned on the memory, the target cameras and the actions. The
generated latents are appended to the history and the loop continues until
`total_time` (or `max_seconds`) is reached.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import load_file

from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE

from gim.models.action_embedding import attach_action_embedding, load_action_embedding
from gim.models.dit import attach_camera_proj, generate_chunk, load_camera_proj
from gim.models.memory_encoder import MemoryEncoder
from gim.utils import camera as cam
from gim.utils.pruning import prune_history
from gim.utils.video import (
    count_video_frames,
    decode_latents_streaming,
    encode_video_streaming,
    iter_video_frames,
)

logger = logging.getLogger(__name__)

DEFAULT_CAPTION = "A video rendered in Unreal Engine."


# ---------------------------------------------------------------------------
# Input clip
# ---------------------------------------------------------------------------

@dataclass
class Clip:
    """A MIND-style input clip parsed from a directory."""
    name: str
    video_path: Path
    perspective: str                 # "1st" | "3rd"
    caption: str
    mark_time: int                   # first frame to generate
    total_time: int                  # frames in the full clip
    camera_poses_raw: torch.Tensor   # [T_json, 12] absolute, cm / degrees
    actions_raw: torch.Tensor        # [T_json, 4]

    @property
    def num_pose_frames(self):
        return int(self.camera_poses_raw.shape[0])

    @classmethod
    def from_dir(cls, clip_dir, perspective="1st", caption=None):
        clip_dir = Path(clip_dir)
        with open(clip_dir / "action.json") as f:
            aj = json.load(f)
        frames = aj["data"]
        return cls(
            name=clip_dir.name,
            video_path=clip_dir / "video.mp4",
            perspective=perspective,
            caption=caption or aj.get("caption", DEFAULT_CAPTION),
            mark_time=int(aj["mark_time"]),
            total_time=int(aj.get("total_time", len(frames))),
            camera_poses_raw=torch.tensor(
                cam.poses_from_action_json(frames, perspective), dtype=torch.float32),
            actions_raw=cam.actions_from_action_json(frames),
        )


@dataclass
class RolloutResult:
    video: torch.Tensor            # [3, T_pred, H, W] in [0, 1], generated frames only
    generated_latents: torch.Tensor
    history_latents: torch.Tensor  # prefix latents fed as initial history
    num_chunks: int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GIMWorldPipeline:

    def __init__(self, backbone, memory_encoder, vae, config, wan_dir, device):
        self.backbone = backbone
        self.memory_encoder = memory_encoder
        self.vae = vae
        self.config = config
        self.wan_dir = Path(wan_dir)
        self.device = device
        self._t5 = None

    # ---- loading ---------------------------------------------------------

    @classmethod
    def from_pretrained(cls, gim_dir, wan_dir, device="cuda"):
        """
        Args:
            gim_dir: a released GIM-World checkpoint folder
                     (e.g. `.../GIM-World/first_person`) containing
                     config.json, transformer/, memory_encoder.safetensors,
                     camera_proj.safetensors, action_embedding.safetensors.
            wan_dir: Wan2.1-T2V-1.3B folder (VAE + umT5 + tokenizer).
        """
        gim_dir = Path(gim_dir)
        with open(gim_dir / "config.json") as f:
            config = json.load(f)

        logger.info(f"Loading backbone from {gim_dir / 'transformer'}")
        backbone = WanModel.from_pretrained(str(gim_dir / "transformer"))
        attach_camera_proj(backbone)
        attach_action_embedding(backbone)
        load_camera_proj(backbone, load_file(str(gim_dir / "camera_proj.safetensors")))
        load_action_embedding(backbone, load_file(str(gim_dir / "action_embedding.safetensors")))
        backbone.to(device).eval().requires_grad_(False)

        H, W = config["resolution"]
        ps = backbone.patch_size
        spatial_tokens = (H // 8 // ps[1]) * (W // 8 // ps[2])
        enc_cfg = config["memory_encoder"]
        memory_encoder = MemoryEncoder(
            num_memory_frames=enc_cfg.get("num_memory_frames", config["chunk_latents"]),
            spatial_tokens_per_frame=spatial_tokens,
            dim=backbone.dim,
            num_heads=enc_cfg["num_heads"],
            num_layers=enc_cfg["num_layers"],
            compact_stride=enc_cfg["compact_stride"],
        )
        memory_encoder.load_state_dict(
            load_file(str(gim_dir / "memory_encoder.safetensors")), strict=True)
        memory_encoder.to(device).eval().requires_grad_(False)

        vae = WanVAE(vae_pth=str(Path(wan_dir) / "Wan2.1_VAE.pth"),
                     dtype=torch.bfloat16, device=device)
        return cls(backbone, memory_encoder, vae, config, wan_dir, device)

    # ---- text --------------------------------------------------------------

    def _load_t5(self):
        if self._t5 is None:
            self._t5 = T5EncoderModel(
                text_len=512, dtype=torch.bfloat16, device=torch.device(self.device),
                checkpoint_path=str(self.wan_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=str(self.wan_dir / "google" / "umt5-xxl"))
        return self._t5

    def release_t5(self):
        """Free the umT5 encoder (11 GB in bf16) once captions are encoded."""
        self._t5 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_text(self, captions: List[str]) -> List[torch.Tensor]:
        """Each caption -> [512, text_dim] (zero padded), on CPU."""
        t5 = self._load_t5()
        out = []
        for emb in t5(captions, torch.device(self.device)):
            emb = emb.float().cpu()
            if emb.shape[0] < 512:
                emb = torch.cat([emb, emb.new_zeros(512 - emb.shape[0], emb.shape[1])])
            out.append(emb[:512])
        return out

    # ---- video -------------------------------------------------------------

    @torch.no_grad()
    def encode_prefix(self, video_path, num_frames):
        """Encode the first `num_frames` raw frames into history latents."""
        H, W = self.config["resolution"]
        latents, covered = encode_video_streaming(
            self.vae, iter_video_frames(video_path, max_frames=num_frames),
            self.device, height=H, width=W)
        return latents, covered

    # ---- rollout -----------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        history_latents: torch.Tensor,     # [C, T0, H', W'] prefix latents
        camera_poses: torch.Tensor,        # [T_total, 12] absolute per-latent poses
        actions: Optional[torch.Tensor],   # [T_total, 4] per-latent codes or None
        text_embedding: torch.Tensor,      # [512, text_dim]
        num_new_latents: int,
        num_steps=50,
        guide_scale=1.0,
        shift=5.0,
        pruning: Optional[str] = None,
        pruning_budget: Optional[int] = None,
        progress=None,
    ):
        cfg = self.config
        chunk = cfg["chunk_latents"]
        pruning = pruning or cfg.get("pruning", "information_guided")
        budget = pruning_budget or cfg.get("pruning_budget", 200)
        prune_kwargs = {}
        if pruning == "information_guided":
            prune_kwargs["max_kernel_size"] = cfg.get("max_kernel_size", 1500)

        text = text_embedding.unsqueeze(0).to(self.device)
        history = history_latents.clone()
        start = history.shape[1]
        end = start + num_new_latents
        generated = []
        current = start
        num_chunks = 0

        while current < end:
            n_target = min(chunk, end - current)
            target_idx = list(range(current, current + n_target))
            pool = list(range(history.shape[1]))
            kept = prune_history(pruning, pool, budget, camera_poses=camera_poses, **prune_kwargs)

            hist = history[:, kept].unsqueeze(0).to(self.device)
            n_hist = torch.tensor([len(kept)], dtype=torch.long, device=self.device)
            cams = cam.relative_camera_chunk(camera_poses, kept, target_idx).to(self.device)
            acts = None
            if actions is not None:
                acts = actions[current:current + n_target].unsqueeze(0).to(self.device)

            z = generate_chunk(
                self.backbone, self.memory_encoder, hist, n_hist, text, cams,
                target_actions=acts, num_steps=num_steps, chunk_latents=n_target,
                shift=shift, num_timesteps=cfg.get("num_train_timesteps", 1000),
                guide_scale=guide_scale)

            z = z[0].detach().cpu()
            generated.append(z)
            history = torch.cat([history, z], dim=1)
            current += n_target
            num_chunks += 1
            if progress is not None:
                progress.update(1)

        return torch.cat(generated, dim=1), num_chunks

    # ---- decode ------------------------------------------------------------

    @torch.no_grad()
    def decode_generated(self, prefix_latents, generated_latents, mark_time,
                         num_pred_frames, warmup_latents=8, chunk_size=20):
        """
        Decode generated latents to frames [mark_time, mark_time + num_pred_frames).

        A few prefix latents are prepended as decoder warmup so the causal VAE
        cache is in a natural state at the first generated latent; the
        warmup frames are cropped away afterwards.
        """
        T0 = prefix_latents.shape[1]
        w_start = max(0, T0 - warmup_latents)
        w_len = T0 - w_start
        decoded = decode_latents_streaming(
            self.vae, torch.cat([prefix_latents[:, w_start:], generated_latents], dim=1),
            self.device, chunk_size=chunk_size)

        span_start = cam.latent_spans(T0 + 1)[T0][0]     # first raw frame of latent T0
        decoded_warmup = 0 if w_len == 0 else 1 + 4 * (w_len - 1)
        crop = decoded_warmup + max(0, mark_time - span_start)
        available = max(0, decoded.shape[1] - crop)
        if available < num_pred_frames:
            logger.warning(f"decoded {available} < requested {num_pred_frames} frames; cropping")
            num_pred_frames = available
        return decoded[:, crop:crop + num_pred_frames]

    # ---- end to end --------------------------------------------------------

    @torch.no_grad()
    def __call__(self, clip, text_embedding=None, max_seconds=None, fps=24.0,
                 num_steps=50, guide_scale=1.0, shift=5.0, pruning=None,
                 pruning_budget=None, warmup_latents=8, progress=None,
                 dry_run=False) -> RolloutResult:
        """
        Args:
            clip: Clip or path to a clip directory.
            text_embedding: precomputed [512, text_dim]; encoded from the
                caption if None (loads umT5 on demand).
            max_seconds: stop the rollout this long after mark_time
                (None or <= 0 -> generate until total_time).
            dry_run: encode, align and prune only; skip the backbone.
        """
        if not isinstance(clip, Clip):
            clip = Clip.from_dir(clip)

        T_json = clip.num_pose_frames
        total_time = min(clip.total_time, T_json)
        if clip.mark_time >= total_time:
            raise ValueError(f"{clip.name}: mark_time={clip.mark_time} >= total_time={total_time}")

        # Latent-aligned cameras / actions over the whole clip.
        total_latents = cam.num_latents_for_frames(total_time)
        lat2raw = cam.latent_to_raw_index(total_latents, total_time)
        camera_poses = clip.camera_poses_raw[lat2raw]
        actions = clip.actions_raw[lat2raw]

        # Prefix -> history latents.
        n_video = count_video_frames(clip.video_path)
        if n_video < clip.mark_time:
            raise ValueError(f"{clip.name}: video has {n_video} frames < mark_time={clip.mark_time}")
        prefix_latents, _ = self.encode_prefix(clip.video_path, clip.mark_time)
        T0 = prefix_latents.shape[1]

        # How far to roll out.
        target_total = total_latents
        if max_seconds is not None and max_seconds > 0:
            budget_frames = int(round(max_seconds * fps))
            spans = cam.latent_spans(total_latents, total_time)
            for j in range(T0, total_latents):
                if spans[j][1] >= clip.mark_time + budget_frames:
                    target_total = j + 1
                    break
        num_new = target_total - T0
        if num_new <= 0:
            raise ValueError(f"{clip.name}: nothing to generate (T0={T0}, target={target_total})")

        logger.info(f"{clip.name}: prefix {clip.mark_time} frames -> {T0} latents; "
                    f"generating {num_new} latents in chunks of {self.config['chunk_latents']}")

        if dry_run:
            pool = list(range(T0))
            kept = prune_history(pruning or self.config.get("pruning", "information_guided"),
                                 pool, pruning_budget or self.config.get("pruning_budget", 200),
                                 camera_poses=camera_poses)
            logger.info(f"[dry_run] history {T0} -> {len(kept)} latents after pruning; "
                        f"camera_poses {tuple(camera_poses.shape)}, actions {tuple(actions.shape)}")
            return RolloutResult(video=torch.zeros(3, 0, 1, 1),
                                 generated_latents=torch.zeros(prefix_latents.shape[0], 0, 1, 1),
                                 history_latents=prefix_latents, num_chunks=0)

        if text_embedding is None:
            text_embedding = self.encode_text([clip.caption])[0]

        generated, num_chunks = self.rollout(
            prefix_latents, camera_poses, actions, text_embedding, num_new,
            num_steps=num_steps, guide_scale=guide_scale, shift=shift,
            pruning=pruning, pruning_budget=pruning_budget, progress=progress)

        # Frames actually covered by the generated latents.
        gen_end_raw = cam.latent_spans(T0 + generated.shape[1], total_time)[-1][1]
        num_pred = max(0, min(total_time, gen_end_raw + 1) - clip.mark_time)
        video = self.decode_generated(prefix_latents, generated, clip.mark_time,
                                      num_pred, warmup_latents=warmup_latents)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return RolloutResult(video=video, generated_latents=generated,
                             history_latents=prefix_latents, num_chunks=num_chunks)

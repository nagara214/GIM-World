"""
GIM-World inference entry point.

Single clip (a directory with video.mp4 + action.json):

    python generate.py --input assets/examples/demo_scene --output outputs/demo \
        --gim_ckpt checkpoints/GIM-World/first_person --wan_dir checkpoints/Wan2.1-T2V-1.3B

Batch over a MIND split (every sub-directory is a clip), sharded across GPUs:

    torchrun --nproc_per_node=8 generate.py \
        --input_root MIND-Data/1st_data/test/mem_test \
        --output outputs/gim_world/1st_data/mem_test \
        --gim_ckpt checkpoints/GIM-World/first_person --wan_dir checkpoints/Wan2.1-T2V-1.3B

Outputs `{output}/{clip_name}/video.mp4` containing the generated frames
(the layout expected by the MIND evaluator). Add `--include_prefix` to
prepend the observed frames for viewing.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gim import Clip, GIMWorldPipeline  # noqa: E402
from gim.utils.video import iter_video_frames, resize_frame, save_video  # noqa: E402

logger = logging.getLogger("gim")


def parse_args():
    p = argparse.ArgumentParser(description="GIM-World autoregressive rollout")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="clip directory (video.mp4 + action.json)")
    src.add_argument("--input_root", type=str, help="directory of clip directories")
    p.add_argument("--output", type=str, required=True, help="output root")
    p.add_argument("--gim_ckpt", type=str, required=True,
                   help="GIM-World checkpoint folder (first_person / third_person)")
    p.add_argument("--wan_dir", type=str, required=True, help="Wan2.1-T2V-1.3B folder")
    p.add_argument("--perspective", choices=["1st", "3rd"], default=None,
                   help="which pose to read from action.json; defaults to the checkpoint's")
    p.add_argument("--caption", type=str, default=None, help="override the clip caption")

    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--guide_scale", type=float, default=1.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--pruning", choices=["information_guided", "uniform"], default=None,
                   help="history pruning rule (default: from config.json)")
    p.add_argument("--pruning_budget", type=int, default=None,
                   help="max history latents kept before memory encoding (default: config)")
    p.add_argument("--max_seconds", type=float, default=20.0,
                   help="stop this many seconds after mark_time; <=0 -> until total_time")
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--include_prefix", action="store_true",
                   help="prepend the observed prefix frames to the saved video")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="encode + prune only, skip the diffusion backbone")
    return p.parse_args()


def rank_info():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local


def discover_clips(root):
    root = Path(root)
    clips = sorted(d for d in root.rglob("*") if d.is_dir() and (d / "action.json").exists()
                   and (d / "video.mp4").exists())
    return clips


def load_prefix_frames(clip, resolution):
    H, W = resolution
    frames = [torch.from_numpy(resize_frame(f, W, H)).permute(2, 0, 1)
              for f in iter_video_frames(clip.video_path, max_frames=clip.mark_time)]
    return (torch.stack(frames, dim=1) + 1.0) / 2.0     # [3, T, H, W] in [0, 1]


def main():
    args = parse_args()
    rank, world, local_rank = rank_info()
    device = args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO,
                        format=f"[%(asctime)s][rank{rank}] %(levelname)s: %(message)s")
    torch.manual_seed(args.seed + rank)
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)

    pipe = GIMWorldPipeline.from_pretrained(args.gim_ckpt, args.wan_dir, device=device)
    perspective = args.perspective or (
        "3rd" if pipe.config.get("perspective") == "third_person" else "1st")

    clip_dirs = [Path(args.input)] if args.input else discover_clips(args.input_root)
    clip_dirs = clip_dirs[rank::world]
    logger.info(f"{len(clip_dirs)} clip(s) assigned to this rank")

    out_root = Path(args.output)
    todo = []
    for d in clip_dirs:
        save_path = out_root / d.name / "video.mp4"
        if save_path.exists() and not args.overwrite:
            logger.info(f"skip existing {save_path}")
            continue
        todo.append((Clip.from_dir(d, perspective=perspective, caption=args.caption), save_path))
    if not todo:
        return

    # Encode all captions with one umT5 load, then free it before the DiT runs.
    text_embeddings = None
    if not args.dry_run:
        text_embeddings = pipe.encode_text([c.caption for c, _ in todo])
        pipe.release_t5()

    for i, (clip, save_path) in enumerate(todo):
        logger.info(f"[{i + 1}/{len(todo)}] {clip.name} -> {save_path}")
        result = pipe(
            clip,
            text_embedding=None if text_embeddings is None else text_embeddings[i],
            max_seconds=args.max_seconds, fps=args.fps,
            num_steps=args.num_steps, guide_scale=args.guide_scale, shift=args.shift,
            pruning=args.pruning, pruning_budget=args.pruning_budget,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        video = result.video
        if args.include_prefix:
            video = torch.cat([load_prefix_frames(clip, pipe.config["resolution"]), video], dim=1)
        save_video(video, save_path, fps=int(args.fps))
        logger.info(f"saved {video.shape[1]} frames ({result.num_chunks} chunks) to {save_path}")

    logger.info("done")


if __name__ == "__main__":
    main()

"""
Download the models required by GIM-World from ModelScope.

    python download_models.py                  # Wan2.1 VAE / umT5 + GIM-World checkpoints
    python download_models.py --with_example   # also fetch one MIND clip for the quick start

Downloads
  1. Wan-AI/Wan2.1-T2V-1.3B  (VAE, umT5 text encoder, tokenizer)
  2. nagara214/GIM-World     (first_person/, third_person/)

and prints the paths to pass to generate.py / run.sh. The example clip comes
from the MIND dataset on Hugging Face (CSU-JPG/MIND).
"""

import argparse
import os
import shutil

MS_WAN = "Wan-AI/Wan2.1-T2V-1.3B"
MS_GIM = "nagara214/GIM-World"

MIND_DATASET = "CSU-JPG/MIND"
EXAMPLE_CLIP = "1st_data/test/mem_test/data-25"
EXAMPLE_DIR = os.path.join("assets", "examples", "demo_scene")

# Only the VAE / text encoder / tokenizer are needed from the base model;
# the DiT weights are shipped inside the GIM-World checkpoint.
WAN_PATTERNS = ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/*"]


def download_models(local_dir, skip_wan, skip_gim):
    from modelscope import snapshot_download
    paths = {}
    if not skip_wan:
        paths["wan_dir"] = snapshot_download(
            MS_WAN, local_dir=os.path.join(local_dir, "Wan2.1-T2V-1.3B"),
            allow_patterns=WAN_PATTERNS)
    if not skip_gim:
        paths["gim_dir"] = snapshot_download(
            MS_GIM, local_dir=os.path.join(local_dir, "GIM-World"))
    return paths


def download_example(dest=EXAMPLE_DIR):
    """One first-person MIND mem_test clip (video.mp4 + action.json)."""
    from huggingface_hub import hf_hub_download
    os.makedirs(dest, exist_ok=True)
    for name in ("video.mp4", "action.json"):
        src = hf_hub_download(MIND_DATASET, f"{EXAMPLE_CLIP}/{name}", repo_type="dataset")
        shutil.copyfile(src, os.path.join(dest, name))
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", default="checkpoints")
    p.add_argument("--skip_wan", action="store_true")
    p.add_argument("--skip_gim", action="store_true")
    p.add_argument("--with_example", action="store_true",
                   help="also download one MIND clip to assets/examples/demo_scene")
    args = p.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    paths = download_models(args.local_dir, args.skip_wan, args.skip_gim)
    if args.with_example:
        paths["example"] = download_example()

    print("\nDone. Use these paths in run.sh / generate.py:")
    if "wan_dir" in paths:
        print(f"  WAN_DIR={paths['wan_dir']}")
    if "gim_dir" in paths:
        print(f"  GIM_CKPT={paths['gim_dir']}/first_person   # or third_person")
    if "example" in paths:
        print(f"  INPUT={paths['example']}")


if __name__ == "__main__":
    main()

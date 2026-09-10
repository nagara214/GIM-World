"""
Download the models required by GIM-World.

    python download_models.py                         # from Hugging Face
    python download_models.py --source modelscope     # from ModelScope (CN mirror)

Downloads
  1. Wan-AI/Wan2.1-T2V-1.3B  (VAE, umT5 text encoder, tokenizer)
  2. GIM-World checkpoints   (first_person/, third_person/)

and prints the paths to pass to generate.py / run.sh. Add `--with_example`
to also fetch one MIND mem_test clip into assets/examples/demo_scene for the
quick start.
"""

import argparse
import os
import shutil

HF_WAN = "Wan-AI/Wan2.1-T2V-1.3B"
HF_GIM = "WeiZhengxuan/GIM-World"
MS_WAN = "Wan-AI/Wan2.1-T2V-1.3B"
MS_GIM = "nagara214/GIM-World"

MIND_DATASET = "CSU-JPG/MIND"
EXAMPLE_CLIP = "1st_data/test/mem_test/data-25"
EXAMPLE_DIR = os.path.join("assets", "examples", "demo_scene")

# Only the VAE / text encoder / tokenizer are needed from the base model;
# the DiT weights are shipped inside the GIM-World checkpoint.
WAN_PATTERNS = ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/*"]


def download_hf(local_dir, skip_wan, skip_gim):
    from huggingface_hub import snapshot_download
    paths = {}
    if not skip_wan:
        paths["wan_dir"] = snapshot_download(
            HF_WAN, local_dir=os.path.join(local_dir, "Wan2.1-T2V-1.3B"),
            allow_patterns=WAN_PATTERNS)
    if not skip_gim:
        paths["gim_dir"] = snapshot_download(
            HF_GIM, local_dir=os.path.join(local_dir, "GIM-World"))
    return paths


def download_modelscope(local_dir, skip_wan, skip_gim):
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
    p.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    p.add_argument("--local_dir", default="checkpoints")
    p.add_argument("--skip_wan", action="store_true")
    p.add_argument("--skip_gim", action="store_true")
    p.add_argument("--with_example", action="store_true",
                   help="also download one MIND clip to assets/examples/demo_scene")
    args = p.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    fn = download_hf if args.source == "hf" else download_modelscope
    paths = fn(args.local_dir, args.skip_wan, args.skip_gim)
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

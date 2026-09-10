"""
Repack training checkpoints into the released GIM-World weight layout.

Pulls only the inference-relevant files of `gim_1st/` and `gim_3rd/` from
the ModelScope repo `nagara214/cam_ssm_ckpts` (optimizer states, RNG states
and the training-only geometry head are skipped), converts the small `.pt`
state dicts to safetensors with the released key names, writes a
`config.json` per perspective, verifies that the released modules load them
with strict=True, and optionally uploads to ModelScope / Hugging Face.

    python scripts/repack_weights.py --work_dir /tmp/gim_weights
    python scripts/repack_weights.py --work_dir /tmp/gim_weights --upload

Requires: torch (CPU is fine), safetensors, modelscope, huggingface_hub.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gim.models.memory_encoder import MemoryEncoder, convert_legacy_state_dict  # noqa: E402

SRC_REPO = "nagara214/cam_ssm_ckpts"
MS_DST = "nagara214/GIM-World"
HF_DST = "WeiZhengxuan/GIM-World"

PERSPECTIVES = {"first_person": "gim_1st", "third_person": "gim_3rd"}

SRC_FILES = [
    "model_weights/config.json",
    "model_weights/diffusion_pytorch_model.safetensors",
    "compressor.pt",
    "cam_proj.pt",
    "action_condition.pt",
]

BASE_CONFIG = {
    "model_type": "gim-world",
    "base_model": "Wan-AI/Wan2.1-T2V-1.3B",
    "resolution": [480, 832],
    "chunk_latents": 20,
    "pruning": "information_guided",
    "pruning_budget": 200,
    "max_kernel_size": 1500,
    "num_train_timesteps": 1000,
    "memory_encoder": {
        "num_layers": 2,
        "num_heads": 12,
        "compact_stride": 2,
        "num_memory_frames": 20,
    },
}


def fetch(src_subdir, work_dir):
    from modelscope.hub.file_download import model_file_download
    out = Path(work_dir) / "src" / src_subdir
    out.mkdir(parents=True, exist_ok=True)
    for rel in SRC_FILES:
        dst = out / rel
        if dst.exists():
            continue
        print(f"  downloading {src_subdir}/{rel}")
        path = model_file_download(SRC_REPO, f"{src_subdir}/{rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
    return out


def to_safetensors(pt_path, out_path, key_map=None):
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    if key_map is not None:
        state = key_map(state)
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, str(out_path))
    return state


def rename_camera_proj(state):
    return {k.replace(".cam_proj.", ".camera_proj."): v for k, v in state.items()}


def repack(perspective, src_dir, dst_root):
    dst = Path(dst_root) / perspective
    dst.mkdir(parents=True, exist_ok=True)

    tr = dst / "transformer"
    tr.mkdir(exist_ok=True)
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        if not (tr / name).exists():
            shutil.copyfile(src_dir / "model_weights" / name, tr / name)

    enc_state = to_safetensors(src_dir / "compressor.pt",
                               dst / "memory_encoder.safetensors",
                               key_map=convert_legacy_state_dict)
    to_safetensors(src_dir / "cam_proj.pt", dst / "camera_proj.safetensors",
                   key_map=rename_camera_proj)
    to_safetensors(src_dir / "action_condition.pt", dst / "action_embedding.safetensors")

    cfg = json.loads(json.dumps(BASE_CONFIG))
    cfg["perspective"] = perspective
    with open(dst / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Verify the memory encoder loads strictly with the released module.
    with open(tr / "config.json") as f:
        dit_cfg = json.load(f)
    H, W = cfg["resolution"]
    spatial = (H // 8 // 2) * (W // 8 // 2)
    enc = MemoryEncoder(
        num_memory_frames=cfg["memory_encoder"]["num_memory_frames"],
        spatial_tokens_per_frame=spatial, dim=dit_cfg["dim"],
        num_heads=cfg["memory_encoder"]["num_heads"],
        num_layers=cfg["memory_encoder"]["num_layers"],
        compact_stride=cfg["memory_encoder"]["compact_stride"])
    enc.load_state_dict(load_file(str(dst / "memory_encoder.safetensors")), strict=True)
    cam_state = load_file(str(dst / "camera_proj.safetensors"))
    assert all(k.startswith("blocks.") and ".camera_proj." in k for k in cam_state), \
        "unexpected camera_proj keys"
    assert len(cam_state) == 2 * dit_cfg["num_layers"]
    print(f"  {perspective}: memory_encoder {sum(v.numel() for v in enc_state.values()) / 1e6:.1f}M "
          f"params, {len(cam_state)} camera_proj tensors, verified.")
    return dst


MODEL_CARD = """---
license: apache-2.0
base_model: Wan-AI/Wan2.1-T2V-1.3B
pipeline_tag: image-to-video
tags:
- world-model
- video-generation
- memory
---

# GIM-World: Geometry-Aware Implicit Memory for Video World Models

Inference checkpoints for **GIM-World** (SIGGRAPH Asia 2026).
Code: https://github.com/nagara214/GIM-World · Paper: https://arxiv.org/abs/2606.02436 · Project page: https://gim-world.github.io/

| Folder | Trained on | Notes |
|---|---|---|
| `first_person/` | MIND first-person split | camera = actor pose |
| `third_person/` | MIND third-person split | camera = chase camera pose |

Each folder contains

```
config.json                    # inference hyper-parameters read by GIMWorldPipeline
transformer/                   # Wan2.1-1.3B DiT fine-tuned with memory conditioning
memory_encoder.safetensors     # implicit memory encoder (2 blocks, compact stride 2)
camera_proj.safetensors        # per-block camera injection
action_embedding.safetensors   # action embedding
```

The Wan2.1 VAE and umT5 text encoder are loaded from `Wan-AI/Wan2.1-T2V-1.3B`.

```python
from gim import GIMWorldPipeline
pipe = GIMWorldPipeline.from_pretrained("GIM-World/first_person", "Wan2.1-T2V-1.3B")
result = pipe("path/to/clip_dir")   # video.mp4 + action.json
```

```bibtex
@article{wei2026gim,
  title   = {Geometry-Aware Implicit Memory for Video World Models},
  author  = {Wei, Zhengxuan and Guo, Xu and Li, Xinghui and Xiang, Xunzhi and Wei, Min and Zhu, Yiran and Wang, Qiulin and Wang, Xintao and Wan, Pengfei and Hou, Xiangwang and Fan, Qi},
  journal = {arXiv preprint arXiv:2606.02436},
  year    = {2026}
}
```
"""


def upload(dst_root, to_modelscope=True, to_hf=True):
    if to_hf:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_DST, repo_type="model", exist_ok=True)
        api.upload_large_folder(repo_id=HF_DST, repo_type="model", folder_path=str(dst_root))
        print(f"uploaded to https://huggingface.co/{HF_DST}")
    if to_modelscope:
        from modelscope.hub.api import HubApi
        api = HubApi()
        try:
            api.create_model(MS_DST, visibility=5, license="Apache License 2.0")
        except Exception as e:  # already exists
            print(f"  (create_model skipped: {e})")
        api.upload_folder(repo_id=MS_DST, folder_path=str(dst_root), repo_type="model",
                          commit_message="Release GIM-World inference weights")
        print(f"uploaded to https://www.modelscope.cn/models/{MS_DST}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", default="weights_repack")
    p.add_argument("--perspectives", nargs="+", default=list(PERSPECTIVES),
                   choices=list(PERSPECTIVES))
    p.add_argument("--upload", action="store_true")
    p.add_argument("--no_hf", action="store_true")
    p.add_argument("--no_modelscope", action="store_true")
    args = p.parse_args()

    dst_root = Path(args.work_dir) / "GIM-World"
    dst_root.mkdir(parents=True, exist_ok=True)
    for persp in args.perspectives:
        print(f"== {persp}")
        src = fetch(PERSPECTIVES[persp], args.work_dir)
        repack(persp, src, dst_root)
    with open(dst_root / "README.md", "w") as f:
        f.write(MODEL_CARD)
    print(f"repacked to {dst_root}")

    if args.upload:
        upload(dst_root, to_modelscope=not args.no_modelscope, to_hf=not args.no_hf)


if __name__ == "__main__":
    main()

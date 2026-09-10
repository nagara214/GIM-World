# Example clip

Run `python download_models.py --with_example` to place a first-person MIND
`mem_test` clip here:

```
demo_scene/
├── video.mp4      # 1080p, 24 fps; frames [0, mark_time) are the observed prefix
└── action.json    # mark_time / total_time / per-frame pose + action codes
```

Then `bash run.sh` rolls the scene out from `mark_time` following the recorded
camera trajectory and actions.

To roll out your own footage, put a `video.mp4` in a folder and author the
trajectory with `make_trajectory.py` (see the top-level README).

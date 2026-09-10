"""
Author an action.json for a custom clip.

Given a prefix video and a pose string describing the camera motion to
generate, writes `<clip_dir>/action.json` compatible with generate.py.

    python make_trajectory.py --clip_dir my_clip --prefix_frames 97 \
        --pose "w-60, right-20, w-40" --caption "A stone courtyard at dusk."

The prefix is assumed to be a static camera at the origin unless
`--prefix_pose_json` provides per-frame poses (a JSON list of
{"pos": [x, y, z], "rpy": [roll, pitch, yaw]} in cm / degrees). The
generated part starts from the last prefix pose.

Pose string actions: w/s/a/d (move), up/down/left/right (look), stay.
Each item is `<action>-<frames>`; speeds default to 5 cm/frame and
1 deg/frame (MIND-like) and can be changed with --move_speed / --turn_speed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gim.utils import trajectory as traj  # noqa: E402
from gim.utils.video import count_video_frames  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_dir", required=True, help="folder containing video.mp4")
    p.add_argument("--pose", required=True, help='e.g. "w-60, right-20, w-40"')
    p.add_argument("--caption", default="A video rendered in Unreal Engine.")
    p.add_argument("--prefix_frames", type=int, default=None,
                   help="frames of video.mp4 to treat as observed prefix (default: all)")
    p.add_argument("--prefix_pose_json", default=None,
                   help="optional per-frame prefix poses; static origin otherwise")
    p.add_argument("--move_speed", type=float, default=traj.DEFAULT_MOVE_SPEED)
    p.add_argument("--turn_speed", type=float, default=traj.DEFAULT_TURN_SPEED)
    args = p.parse_args()

    clip_dir = Path(args.clip_dir)
    n_video = count_video_frames(clip_dir / "video.mp4")
    n_prefix = args.prefix_frames or n_video
    if n_prefix > n_video:
        raise ValueError(f"--prefix_frames {n_prefix} > video frames {n_video}")

    if args.prefix_pose_json:
        with open(args.prefix_pose_json) as f:
            poses = json.load(f)
        if len(poses) < n_prefix:
            raise ValueError("prefix_pose_json has fewer entries than prefix frames")
        prefix = [traj._frame_entry(pp["pos"], pp["rpy"], {"ws": 0, "ad": 0, "ud": 0, "lr": 0})
                  for pp in poses[:n_prefix]]
    else:
        prefix = traj.static_prefix_frames(n_prefix)

    last = prefix[-1]
    start_pos = [last["actor_pos"][k] for k in "xyz"]
    start_rpy = [last["actor_rpy"][k] for k in "xyz"]
    future = traj.frames_from_pose_string(start_pos, start_rpy, args.pose,
                                          move_speed=args.move_speed,
                                          turn_speed=args.turn_speed)

    aj = traj.build_action_json(prefix, future, args.caption)
    out = clip_dir / "action.json"
    traj.save_action_json(aj, out)
    print(f"wrote {out}: mark_time={aj['mark_time']}, total_time={aj['total_time']}")


if __name__ == "__main__":
    main()

"""
Build MIND-style action.json files for custom rollouts.

An action.json describes the whole clip: the observed prefix (frames
[0, mark_time)) and the frames to be generated (frames [mark_time,
total_time)). Every frame carries a pose (Unreal Engine convention:
position in centimetres, roll/pitch/yaw in degrees) and four ternary action
codes [ws, ad, ud, lr] (0 = none, 1 = positive, 2 = negative).

Two ways to author the generated part:

  1. Pose string, e.g. "w-40, right-10, d-20": a comma-separated list of
     `<action>-<frames>` items applied sequentially from the last prefix pose.
  2. Keyframes: a list of {"frame": int, "pos": [x, y, z], "rpy": [r, p, y]}
     that is linearly interpolated between keyframes.

Supported pose-string actions:
  w / s          move forward / backward       (ws = 1 / 2)
  a / d          move left / right             (ad = 1 / 2)
  up / down      look up / look down           (ud = 1 / 2)
  left / right   look left / look right        (lr = 1 / 2)
  stay           hold pose                     (all zero)
"""

import json
import math

import numpy as np

MOVE_ACTIONS = {"w": ("ws", 1), "s": ("ws", 2), "a": ("ad", 1), "d": ("ad", 2)}
LOOK_ACTIONS = {"up": ("ud", 1), "down": ("ud", 2), "left": ("lr", 1), "right": ("lr", 2)}

DEFAULT_MOVE_SPEED = 5.0    # cm per frame
DEFAULT_TURN_SPEED = 1.0    # degrees per frame


def _forward_right(yaw_deg):
    yaw = math.radians(yaw_deg)
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return fwd, right


def parse_pose_string(pose_string):
    """'w-40, right-10' -> [('w', 40), ('right', 10)]"""
    items = []
    for tok in pose_string.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, n = tok.rsplit("-", 1)
        name = name.strip().lower()
        if name not in MOVE_ACTIONS and name not in LOOK_ACTIONS and name != "stay":
            raise ValueError(f"Unknown action {name!r} in pose string")
        items.append((name, int(n)))
    return items


def frames_from_pose_string(start_pos, start_rpy, pose_string,
                            move_speed=DEFAULT_MOVE_SPEED,
                            turn_speed=DEFAULT_TURN_SPEED):
    """Integrate a pose string into per-frame entries starting from a pose."""
    pos = np.array(start_pos, dtype=np.float64)
    roll, pitch, yaw = (float(v) for v in start_rpy)
    frames = []
    for name, n in parse_pose_string(pose_string):
        for _ in range(n):
            action = {"ws": 0, "ad": 0, "ud": 0, "lr": 0}
            if name in MOVE_ACTIONS:
                key, code = MOVE_ACTIONS[name]
                action[key] = code
                fwd, right = _forward_right(yaw)
                if name == "w":
                    pos = pos + move_speed * fwd
                elif name == "s":
                    pos = pos - move_speed * fwd
                elif name == "a":
                    pos = pos - move_speed * right
                else:
                    pos = pos + move_speed * right
            elif name in LOOK_ACTIONS:
                key, code = LOOK_ACTIONS[name]
                action[key] = code
                if name == "up":
                    pitch += turn_speed
                elif name == "down":
                    pitch -= turn_speed
                elif name == "left":
                    yaw -= turn_speed
                else:
                    yaw += turn_speed
            frames.append(_frame_entry(pos, (roll, pitch, yaw), action))
    return frames


def frames_from_keyframes(keyframes, num_frames):
    """Linearly interpolate {frame, pos, rpy} keyframes into per-frame entries
    (actions are left at zero)."""
    kf = sorted(keyframes, key=lambda k: k["frame"])
    if not kf:
        raise ValueError("keyframes must not be empty")
    fr = np.array([k["frame"] for k in kf], dtype=np.float64)
    pos = np.array([k["pos"] for k in kf], dtype=np.float64)
    rpy = np.array([k["rpy"] for k in kf], dtype=np.float64)
    t = np.arange(num_frames, dtype=np.float64)
    out = []
    for i in range(num_frames):
        p = [np.interp(t[i], fr, pos[:, j]) for j in range(3)]
        r = [np.interp(t[i], fr, rpy[:, j]) for j in range(3)]
        out.append(_frame_entry(p, r, {"ws": 0, "ad": 0, "ud": 0, "lr": 0}))
    return out


def _frame_entry(pos, rpy, action):
    return {
        **action,
        "actor_pos": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
        "actor_rpy": {"x": float(rpy[0]), "y": float(rpy[1]), "z": float(rpy[2])},
    }


def build_action_json(prefix_frames, future_frames, caption):
    """Assemble prefix + future frame entries into an action.json dict."""
    data = []
    for i, fr in enumerate(list(prefix_frames) + list(future_frames)):
        data.append({"time": i, **fr})
    return {
        "mark_time": len(prefix_frames),
        "total_time": len(data),
        "caption": caption,
        "data": data,
    }


def static_prefix_frames(num_frames, pos=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    """Prefix entries for a clip whose camera does not move."""
    return [_frame_entry(pos, rpy, {"ws": 0, "ad": 0, "ud": 0, "lr": 0})
            for _ in range(num_frames)]


def save_action_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

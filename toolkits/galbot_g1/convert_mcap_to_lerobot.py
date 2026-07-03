# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Galbot G1 teleop SYNC MCAP episodes to a LeRobot v2.1 dataset.

Produces the dataset consumed by the ``pi05_galbot_g1`` OpenPI train config
(see ``rlinf/models/embodiment/openpi/dataconfig/galbot_g1_dataconfig.py``):

- ``image``: head camera RGB (front_head left color), resized to 640x480.
- ``extra_view_image``: right-wrist camera RGB, native 640x360.
- ``state``: 8-dim float32 — 7 measured right-arm joint positions (rad) +
  gripper open fraction in [0, 1] (1 = fully open = 0.12 m).
- ``actions``: 8-dim float32 — zero-order hold of the teleop joint-space
  command targets, same layout/units as ``state``.
- ``task``: fixed language prompt.

Requires the same pinned lerobot as RLinf training (dataset format v2.1):
``pip install "git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"``
plus ``mcap mcap-protobuf-support opencv-python``.

Example:
    python toolkits/galbot_g1/convert_mcap_to_lerobot.py \\
        --data-dir /home/galbot/1105_1696 \\
        --output-root /home/galbot/lerobot_data \\
        --prompt "put the cola bottle into the basket"
"""

from __future__ import annotations

import argparse
import functools
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from toolkits.galbot_g1.galbot_mcap import (  # noqa: E402
    ARM_JOINT_NAMES,
    EpisodeStreams,
    assemble_state_action,
    decimate_indices,
    nearest_indices,
    read_episode,
    zoh_indices,
)

SOURCE_FPS = 30.0
HEAD_SIZE_WH = (640, 480)
STATE_NAMES = [*ARM_JOINT_NAMES, "right_gripper_open"]

# QA thresholds (ns). Exceeding these prints a warning but does not abort;
# inspect the episode before training on it. Set to roughly one source-frame
# period: the recorder occasionally drops a packet/frame (observed gaps up to
# ~65 ms), which is harmless, so only flag mismatches beyond that.
MAX_WRIST_MISMATCH_NS = 70_000_000
MAX_SENSOR_MISMATCH_NS = 70_000_000
MAX_FRAME_GAP_FACTOR = 2.5


def _decode_rgb(jpeg: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode jpeg frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    import cv2

    if (img.shape[1], img.shape[0]) == size_wh:
        return img
    return cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA)


def _episode_qa(
    name: str, streams: EpisodeStreams, sel: np.ndarray, fps: float
) -> list[str]:
    """Return human-readable warnings for alignment/timing anomalies."""
    warnings = []
    frame_ts = streams.head_ts[sel]

    gaps = np.diff(frame_ts)
    max_gap_ns = 1e9 / fps * MAX_FRAME_GAP_FACTOR
    if len(gaps) and gaps.max() > max_gap_ns:
        warnings.append(f"{name}: head-frame gap {gaps.max() / 1e6:.0f} ms")

    wi = nearest_indices(frame_ts, streams.wrist_ts)
    wrist_err = np.abs(streams.wrist_ts[wi] - frame_ts).max()
    if wrist_err > MAX_WRIST_MISMATCH_NS:
        warnings.append(f"{name}: wrist-head mismatch {wrist_err / 1e6:.0f} ms")

    si = nearest_indices(frame_ts, streams.sensor_ts)
    sensor_err = np.abs(streams.sensor_ts[si] - frame_ts).max()
    if sensor_err > MAX_SENSOR_MISMATCH_NS:
        warnings.append(f"{name}: sensor-frame mismatch {sensor_err / 1e6:.0f} ms")

    ai = zoh_indices(frame_ts, streams.arm_target_ts)
    held = ai >= 0
    if held.any():
        age = (frame_ts[held] - streams.arm_target_ts[ai[held]]).max()
        if age > 1e9 / fps * 2:
            warnings.append(f"{name}: arm command age {age / 1e6:.0f} ms")
    return warnings


def convert(
    data_dir: Path,
    output_root: Path,
    repo_id: str,
    fps: float,
    prompt: str,
    vcodec: str | None,
    max_episodes: int | None,
    overwrite: bool,
) -> None:
    from lerobot.common.datasets import lerobot_dataset as lds

    if vcodec is not None:
        lds.encode_video_frames = functools.partial(
            lds.encode_video_frames, vcodec=vcodec
        )

    episodes = sorted(data_dir.glob("*.SYNC.mcap"))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise FileNotFoundError(f"no *.SYNC.mcap files under {data_dir}")

    root = output_root / repo_id
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{root} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(root)

    probe = _decode_rgb(read_episode(episodes[0]).wrist_jpeg[0])
    wrist_h, wrist_w = probe.shape[:2]

    features = {
        "image": {
            "dtype": "video",
            "shape": (HEAD_SIZE_WH[1], HEAD_SIZE_WH[0], 3),
            "names": ["height", "width", "channel"],
        },
        "extra_view_image": {
            "dtype": "video",
            "shape": (wrist_h, wrist_w, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "actions": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
    }
    dataset = lds.LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        root=root,
        robot_type="galbot_g1",
        features=features,
        image_writer_threads=8,
    )

    all_warnings: list[str] = []
    total_frames = 0
    for ep_path in episodes:
        name = ep_path.name.replace(".SYNC.mcap", "")
        streams = read_episode(ep_path)
        sel = decimate_indices(len(streams.head_ts), SOURCE_FPS, fps)
        frame_ts = streams.head_ts[sel]

        state, actions = assemble_state_action(
            frame_ts,
            streams.sensor_ts,
            streams.sensor_arm,
            streams.sensor_gripper,
            streams.arm_target_ts,
            streams.arm_targets,
            streams.gripper_target_ts,
            streams.gripper_targets,
        )
        wrist_sel = nearest_indices(frame_ts, streams.wrist_ts)

        warnings = _episode_qa(name, streams, sel, fps)
        all_warnings.extend(warnings)

        for i, src_idx in enumerate(sel):
            dataset.add_frame(
                {
                    "image": _resize(
                        _decode_rgb(streams.head_jpeg[src_idx]), HEAD_SIZE_WH
                    ),
                    "extra_view_image": _decode_rgb(
                        streams.wrist_jpeg[wrist_sel[i]]
                    ),
                    "state": state[i],
                    "actions": actions[i],
                    "task": prompt,
                }
            )
        dataset.save_episode()

        total_frames += len(sel)
        dur = (frame_ts[-1] - frame_ts[0]) / 1e9
        arm_span = np.ptp(state[:, :7], axis=0).max()
        grip_min = state[:, 7].min()
        status = " | ".join(warnings) if warnings else "ok"
        print(
            f"{name}: {len(sel)} frames, {dur:.1f}s, "
            f"max joint span {arm_span:.2f} rad, min grip {grip_min:.2f} "
            f"[{status}]"
        )

    print(f"\ndone: {len(episodes)} episodes, {total_frames} frames -> {root}")
    if all_warnings:
        print(f"{len(all_warnings)} QA warning(s) above — inspect before training.")
    print(
        "next: compute norm stats on the training node with\n"
        f"  export HF_LEROBOT_HOME={output_root}\n"
        "  python toolkits/lerobot/calculate_norm_stats.py "
        f"--config-name pi05_galbot_g1 --repo-id {repo_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="galbot_g1_cola_basket")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--prompt", default="put the cola bottle into the basket"
    )
    parser.add_argument(
        "--vcodec",
        default=None,
        help="override video codec (e.g. libx264) if libsvtav1 is unavailable",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert(
        data_dir=args.data_dir,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        prompt=args.prompt,
        vcodec=args.vcodec,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

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
"""Utilities to read Galbot G1 teleop MCAP recordings.

Galbot's data-collection stack records one ``*.FIN.mcap`` (raw) and one
``*.SYNC.mcap`` (time-aligned) file per episode, containing protobuf-encoded
messages (schemas are embedded in the MCAP, decoded via
``mcap-protobuf-support``). Conversion uses the SYNC files.

Relevant topics (single right-arm tabletop task):

- ``/front_head_camera/left_color/image_raw``: head RGB, 30 Hz, jpeg 1280x960.
- ``/right_arm_camera/color/image_raw``: right-wrist RGB, 30 Hz, jpeg 640x360.
- ``singorix/wbcs/sensor``: whole-body joint sensors, ~123 Hz. Per-group map;
  ``right_arm`` has 7 joint positions (rad), ``right_gripper`` one position in
  **percent open** (0 = closed, 100 = fully open — not meters).
- ``singorix/wbcs/target``: teleop joint-space command targets, ~105 Hz for the
  arm; gripper commands are sparse. Same units as the sensors.

The pure alignment helpers in this module operate on int64 nanosecond
timestamps and are unit-tested without any MCAP files.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

HEAD_IMAGE_TOPIC = "/front_head_camera/left_color/image_raw"
WRIST_IMAGE_TOPIC = "/right_arm_camera/color/image_raw"
SENSOR_TOPIC = "singorix/wbcs/sensor"
TARGET_TOPIC = "singorix/wbcs/target"

ARM_GROUP = "right_arm"
GRIPPER_GROUP = "right_gripper"
ARM_JOINT_NAMES = tuple(f"right_arm_joint{i}" for i in range(1, 8))
GRIPPER_FULL_OPEN_PCT = 100.0


def decimate_indices(num_frames: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """Return frame indices that decimate ``src_fps`` down to ``dst_fps``.

    Uses a constant integer stride so the selected frames stay uniformly
    spaced (required by LeRobot's per-frame ``frame_index / fps`` timestamps).

    Args:
        num_frames: Number of source frames.
        src_fps: Source frame rate.
        dst_fps: Desired frame rate; must divide ``src_fps`` into an integer
            stride when rounded.

    Returns:
        Int64 array of selected source-frame indices.
    """
    step = int(round(src_fps / dst_fps))
    if step < 1:
        raise ValueError(f"dst_fps {dst_fps} exceeds src_fps {src_fps}")
    return np.arange(0, num_frames, step, dtype=np.int64)


def nearest_indices(query_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp, return the index of the nearest ref timestamp.

    Args:
        query_ts: Query timestamps, any sorted or unsorted 1-D array.
        ref_ts: Reference timestamps, sorted ascending, non-empty.

    Returns:
        Int64 array of indices into ``ref_ts``, same length as ``query_ts``.
    """
    if len(ref_ts) == 0:
        raise ValueError("ref_ts must be non-empty")
    if len(ref_ts) == 1:
        return np.zeros(len(query_ts), dtype=np.int64)
    pos = np.searchsorted(ref_ts, query_ts)
    pos = np.clip(pos, 1, len(ref_ts) - 1)
    left_dist = np.abs(query_ts - ref_ts[pos - 1])
    right_dist = np.abs(ref_ts[pos] - query_ts)
    return np.where(left_dist <= right_dist, pos - 1, pos).astype(np.int64)


def zoh_indices(query_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    """Zero-order hold: index of the latest ref timestamp <= each query time.

    Args:
        query_ts: Query timestamps.
        ref_ts: Reference timestamps, sorted ascending (may be empty).

    Returns:
        Int64 array of indices into ``ref_ts``; ``-1`` where no reference
        sample exists at or before the query time.
    """
    return (np.searchsorted(ref_ts, query_ts, side="right") - 1).astype(np.int64)


def gripper_pct_to_norm(pct: np.ndarray) -> np.ndarray:
    """Convert gripper percent-open (0-100) to normalized open fraction [0, 1].

    At deploy time the inverse mapping to the galbot_sdk gripper command is
    ``width_m = norm * 0.12`` (G1 gripper full-open width).
    """
    return np.clip(np.asarray(pct, dtype=np.float64) / GRIPPER_FULL_OPEN_PCT, 0.0, 1.0)


def assemble_state_action(
    frame_ts: np.ndarray,
    sensor_ts: np.ndarray,
    sensor_arm: np.ndarray,
    sensor_gripper: np.ndarray,
    arm_target_ts: np.ndarray,
    arm_targets: np.ndarray,
    gripper_target_ts: np.ndarray,
    gripper_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-frame 8-dim state and action arrays for the selected frames.

    State  = [7 measured arm joint positions (rad), gripper open fraction].
    Action = [7 commanded arm joint positions (rad), commanded open fraction],
    using zero-order hold of the most recent teleop command at each frame
    time. Before the first command of a stream, the measured value is used
    (the teleop had not issued a command yet, so "hold current" is the
    faithful label).

    Args:
        frame_ts: Selected frame timestamps (ns), shape (T,).
        sensor_ts: Joint-sensor timestamps (ns), sorted, shape (N,).
        sensor_arm: Measured arm joint positions, shape (N, 7).
        sensor_gripper: Measured gripper percent-open, shape (N,).
        arm_target_ts: Arm command timestamps (ns), sorted, shape (M,).
        arm_targets: Commanded arm joint positions, shape (M, 7).
        gripper_target_ts: Gripper command timestamps (ns), sorted, shape (K,).
        gripper_targets: Commanded gripper percent-open, shape (K,).

    Returns:
        ``(state, actions)`` float32 arrays of shape (T, 8).
    """
    si = nearest_indices(frame_ts, sensor_ts)
    state_arm = sensor_arm[si]
    state_grip = gripper_pct_to_norm(sensor_gripper[si])

    if len(arm_target_ts) == 0:
        action_arm = state_arm
    else:
        ai = zoh_indices(frame_ts, arm_target_ts)
        action_arm = np.where(
            (ai >= 0)[:, None], arm_targets[np.maximum(ai, 0)], state_arm
        )
    if len(gripper_target_ts) == 0:
        action_grip = state_grip
    else:
        gi = zoh_indices(frame_ts, gripper_target_ts)
        action_grip = np.where(
            gi >= 0,
            gripper_pct_to_norm(gripper_targets[np.maximum(gi, 0)]),
            state_grip,
        )

    state = np.concatenate([state_arm, state_grip[:, None]], axis=1)
    actions = np.concatenate([action_arm, action_grip[:, None]], axis=1)
    return state.astype(np.float32), actions.astype(np.float32)


@dataclasses.dataclass
class EpisodeStreams:
    """Raw per-episode streams extracted from one SYNC MCAP file.

    All timestamps are int64 nanoseconds on the recording's device clock and
    sorted ascending. Images are kept jpeg-compressed until frame selection.
    """

    head_ts: np.ndarray
    head_jpeg: list[bytes]
    wrist_ts: np.ndarray
    wrist_jpeg: list[bytes]
    sensor_ts: np.ndarray
    sensor_arm: np.ndarray
    sensor_gripper: np.ndarray
    arm_target_ts: np.ndarray
    arm_targets: np.ndarray
    gripper_target_ts: np.ndarray
    gripper_targets: np.ndarray


def _header_time_ns(proto, log_time: int) -> int:
    """Message time from the proto header, falling back to the MCAP log time."""
    ts = proto.header.timestamp
    ns = int(ts.sec) * 1_000_000_000 + int(ts.nanosec)
    return ns if ns > 0 else int(log_time)


def _sorted_by_ts(ts_list: list[int], *columns: list):
    """Sort parallel columns by timestamp; returns (ts_array, *columns)."""
    ts = np.asarray(ts_list, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    sorted_cols = []
    for col in columns:
        if isinstance(col, list):
            sorted_cols.append([col[i] for i in order])
        else:
            sorted_cols.append(np.asarray(col)[order])
    return (ts[order], *sorted_cols)


def read_episode(path: str | Path) -> EpisodeStreams:
    """Read one SYNC MCAP episode into aligned-ready stream arrays.

    Verifies arm joint naming/order against :data:`ARM_JOINT_NAMES` so a
    firmware-side reordering cannot silently corrupt the dataset.
    """
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    head_ts, head_jpeg = [], []
    wrist_ts, wrist_jpeg = [], []
    sensor_ts, sensor_arm, sensor_grip = [], [], []
    arm_tgt_ts, arm_tgt = [], []
    grip_tgt_ts, grip_tgt = [], []

    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, proto in reader.iter_decoded_messages(
            topics=[HEAD_IMAGE_TOPIC, WRIST_IMAGE_TOPIC, SENSOR_TOPIC, TARGET_TOPIC]
        ):
            t = _header_time_ns(proto, message.log_time)
            topic = channel.topic
            if topic == HEAD_IMAGE_TOPIC:
                head_ts.append(t)
                head_jpeg.append(bytes(proto.data))
            elif topic == WRIST_IMAGE_TOPIC:
                wrist_ts.append(t)
                wrist_jpeg.append(bytes(proto.data))
            elif topic == SENSOR_TOPIC:
                arm = proto.joint_sensor_map.get(ARM_GROUP)
                grip = proto.joint_sensor_map.get(GRIPPER_GROUP)
                if arm is None or grip is None:
                    continue
                if tuple(arm.name) != ARM_JOINT_NAMES:
                    raise ValueError(
                        f"{path}: unexpected arm joint order {tuple(arm.name)}, "
                        f"expected {ARM_JOINT_NAMES}"
                    )
                sensor_ts.append(t)
                sensor_arm.append(list(arm.position))
                sensor_grip.append(grip.position[0])
            elif topic == TARGET_TOPIC:
                for group, traj in proto.target_group_trajectory_map.items():
                    if not traj.group_commands:
                        continue
                    positions = [
                        jc.position for jc in traj.group_commands[0].joint_commands
                    ]
                    if group == ARM_GROUP:
                        if tuple(traj.joint_names) != ARM_JOINT_NAMES:
                            raise ValueError(
                                f"{path}: unexpected target joint order "
                                f"{tuple(traj.joint_names)}"
                            )
                        arm_tgt_ts.append(t)
                        arm_tgt.append(positions)
                    elif group == GRIPPER_GROUP:
                        grip_tgt_ts.append(t)
                        grip_tgt.append(positions[0])

    if not head_ts:
        raise ValueError(f"{path}: no head-camera frames on {HEAD_IMAGE_TOPIC}")
    if not sensor_ts:
        raise ValueError(f"{path}: no joint sensor messages on {SENSOR_TOPIC}")
    if not arm_tgt_ts:
        raise ValueError(f"{path}: no arm targets on {TARGET_TOPIC}")

    head_ts, head_jpeg = _sorted_by_ts(head_ts, head_jpeg)
    wrist_ts, wrist_jpeg = _sorted_by_ts(wrist_ts, wrist_jpeg)
    sensor_ts, sensor_arm, sensor_grip = _sorted_by_ts(
        sensor_ts, np.asarray(sensor_arm, dtype=np.float64), sensor_grip
    )
    arm_tgt_ts, arm_tgt = _sorted_by_ts(
        arm_tgt_ts, np.asarray(arm_tgt, dtype=np.float64)
    )
    if grip_tgt_ts:
        grip_tgt_ts, grip_tgt = _sorted_by_ts(
            grip_tgt_ts, np.asarray(grip_tgt, dtype=np.float64)
        )
    else:
        grip_tgt_ts = np.empty(0, dtype=np.int64)
        grip_tgt = np.empty(0, dtype=np.float64)

    return EpisodeStreams(
        head_ts=head_ts,
        head_jpeg=head_jpeg,
        wrist_ts=wrist_ts,
        wrist_jpeg=wrist_jpeg,
        sensor_ts=sensor_ts,
        sensor_arm=sensor_arm,
        sensor_gripper=np.asarray(sensor_grip, dtype=np.float64),
        arm_target_ts=arm_tgt_ts,
        arm_targets=arm_tgt,
        gripper_target_ts=grip_tgt_ts,
        gripper_targets=np.asarray(grip_tgt, dtype=np.float64),
    )

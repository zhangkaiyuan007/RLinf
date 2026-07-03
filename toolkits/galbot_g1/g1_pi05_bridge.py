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
"""Galbot G1 client for the pi0.5 cola-into-basket policy.

Runs on the machine connected to the robot LAN. Inference runs remotely via
openpi's websocket protocol — start ``serve_g1_policy.py`` on the GPU node
and point ``--host`` at it (direct LAN preferred; SSH forward also works).

Modes:
    --dry-run (default): read real observations, query the policy, print the
        actions. The robot NEVER moves.
    --execute: move the robot (y/n safety gate, joint-delta guards).

Execution is pipelined: while a chunk executes, the observation for the next
chunk is captured ``--latency-budget`` seconds before the chunk ends and
inference runs in parallel; the actions of the new chunk that correspond to
the already-elapsed time are skipped. Gripper open/close events interrupt the
pipeline (arm stops, gripper actuates blocking, pipeline restarts) — matching
the demos, where the arm holds still while the gripper moves.

Contract (must match training data):
    obs   = head-left RGB (resized 640x480) + right-wrist RGB
            + [7 right-arm joints (rad), gripper open fraction 0-1]
    action chunk = (10, 8) absolute joint targets at 15 Hz; grip 1=open.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

PROMPT = "put the cola bottle into the basket"
CONTROL_HZ = 15.0
HORIZON = 10
HEAD_SIZE_WH = (640, 480)
GRIPPER_FULL_OPEN_M = 0.12
ARM_GROUP = "right_arm"
GRIPPER_NAME = "right_gripper"

# Whole-body posture measured from the training episodes (static throughout
# collection). Leg/head define the head-camera viewpoint — they MUST match,
# otherwise the policy sees out-of-distribution images.
HOME_LEG = [0.5505, 1.7068, 1.1645, 0.0002, -0.0039]
HOME_HEAD = [0.0, 0.0]
# Median first-frame arm pose across all 50 training episodes (NOT episode 0,
# which started from an atypical posture ~3 sigma off on joints 2/5).
HOME_RIGHT_ARM = [-2.007, 1.312, 0.613, 1.726, 0.272, 0.730, -0.030]

# Safety guards: max joint jump between 1/15 s points inside a chunk, and for
# the transition from the current pose to a re-planned chunk (which gets extra
# travel time, bounded by TRANSIT_SPEED_RAD_S).
MAX_STEP_RAD = 0.20
MAX_TRANSIT_RAD = 0.40
TRANSIT_SPEED_RAD_S = 0.4

# The teleop gripper commands in the training data are binary (0/100), so the
# bridge drives the gripper as open/closed with hysteresis, timed to the step
# where the predicted profile crosses the threshold.
GRIP_CLOSE_BELOW = 0.45
GRIP_OPEN_ABOVE = 0.65
GRIP_VELOCITY_MPS = 0.2
GRIP_EFFORT = 50


def decode_rgb(compressed: dict) -> np.ndarray:
    """Decode a galbot_sdk compressed image dict to an RGB uint8 array."""
    arr = np.frombuffer(compressed["data"], np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise RuntimeError("camera frame decode failed")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def chunk_is_safe(current: np.ndarray, chunk: np.ndarray) -> tuple[bool, str]:
    """Reject re-plan transitions and within-chunk steps beyond their limits."""
    transit = np.abs(chunk[0, :7] - current).max()
    if transit > MAX_TRANSIT_RAD:
        return False, f"re-plan transition {transit:.3f} rad > {MAX_TRANSIT_RAD}"
    if len(chunk) > 1:
        step = np.abs(np.diff(chunk[:, :7], axis=0)).max()
        if step > MAX_STEP_RAD:
            return False, f"per-step joint delta {step:.3f} rad > {MAX_STEP_RAD}"
    return True, ""


def build_trajectory(gm, current: np.ndarray, chunk: np.ndarray, dt: float):
    """Build a right-arm Trajectory: current pose, transition, then the chunk."""
    transit = float(np.abs(chunk[0, :7] - current).max())
    transit_time = max(dt, transit / TRANSIT_SPEED_RAD_S)

    traj = gm.Trajectory()
    traj.joint_groups = [ARM_GROUP]
    traj.joint_names = []
    points = []
    t = 0.05
    path = np.vstack([current, chunk[:, :7]])
    for i, q in enumerate(path):
        pt = gm.TrajectoryPoint()
        pt.time_from_start_second = t
        t += transit_time if i == 0 else dt
        cmds = []
        for pos in q:
            c = gm.JointCommand()
            c.position = float(pos)
            cmds.append(c)
        pt.joint_command_vec = cmds
        points.append(pt)
    traj.points = points
    return traj


def confirm_or_exit(message: str) -> None:
    while True:
        key = input(f"{message} (y/n): ").strip().lower()
        if key == "y":
            return
        if key == "n":
            raise SystemExit("aborted by operator")


class G1PolicyRunner:
    """Owns the robot/policy handles and the capture→infer→execute cycle."""

    def __init__(self, robot, gm, policy, args):
        self.robot = robot
        self.gm = gm
        self.policy = policy
        self.args = args
        self.dt = (1.0 / CONTROL_HZ) * args.slow
        # Actions to skip on a pipelined chunk = time between obs capture and
        # execution start, in executed-step units. Capped so that at least
        # 3 steps per chunk remain executable.
        self.skip = min(math.ceil(args.latency_budget / self.dt), HORIZON - 3)
        self.grip_closed = False
        self.did_grasp = False
        self.still_chunks = 0
        self.obs_dir = None
        if args.save_obs:
            from pathlib import Path

            self.obs_dir = Path(args.save_obs).expanduser()
            self.obs_dir.mkdir(parents=True, exist_ok=True)

    # ---- robot I/O -------------------------------------------------------

    def capture(self, tag: str | None = None) -> tuple[dict, np.ndarray]:
        from galbot_sdk.g1 import SensorType

        head = decode_rgb(self.robot.get_rgb_data(SensorType.HEAD_LEFT_CAMERA))
        head = cv2.resize(head, HEAD_SIZE_WH, interpolation=cv2.INTER_AREA)
        wrist = decode_rgb(self.robot.get_rgb_data(SensorType.RIGHT_ARM_CAMERA))
        arm_q = np.array(
            self.robot.get_joint_positions([ARM_GROUP], []), dtype=np.float32
        )
        grip = float(
            np.clip(
                self.robot.get_gripper_state(GRIPPER_NAME).width
                / GRIPPER_FULL_OPEN_M,
                0.0,
                1.0,
            )
        )
        if self.obs_dir is not None and tag is not None:
            cv2.imwrite(
                str(self.obs_dir / f"{tag}_head.jpg"),
                cv2.cvtColor(head, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(self.obs_dir / f"{tag}_wrist.jpg"),
                cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR),
            )
        obs = {
            "observation/image": head,
            "observation/extra_view_image": wrist,
            "observation/state": np.concatenate([arm_q, [grip]]).astype(np.float32),
            "prompt": self.args.prompt,
        }
        return obs, arm_q

    def infer(self, obs: dict) -> np.ndarray:
        t0 = time.monotonic()
        chunk = np.asarray(self.policy.infer(obs)["actions"])[:HORIZON]
        self.last_latency = time.monotonic() - t0
        return chunk

    def arm_pos(self) -> np.ndarray:
        return np.array(
            self.robot.get_joint_positions([ARM_GROUP], []), dtype=np.float32
        )

    def set_gripper(self, close: bool) -> None:
        width = 0.0 if close else GRIPPER_FULL_OPEN_M
        self.robot.set_gripper_command(
            GRIPPER_NAME, width, GRIP_VELOCITY_MPS, GRIP_EFFORT, True
        )
        self.grip_closed = close
        if close:
            self.did_grasp = True

    # ---- execution -------------------------------------------------------

    def find_grip_event(self, chunk: np.ndarray) -> tuple[int, str] | None:
        profile = chunk[:, 7]
        if not self.grip_closed:
            idx = np.nonzero(profile < GRIP_CLOSE_BELOW)[0]
            if len(idx):
                return int(idx[0]), "close"
        else:
            idx = np.nonzero(profile > GRIP_OPEN_ABOVE)[0]
            if len(idx):
                return int(idx[0]), "open"
        return None

    def execute_arm(self, chunk: np.ndarray, blocking: bool = True):
        """Execute arm targets; returns the ControlStatus."""
        arm_q = self.arm_pos()
        traj = build_trajectory(self.gm, arm_q, chunk, self.dt)
        return self.robot.execute_joint_trajectory(traj, blocking)

    def episode_done(self, chunk: np.ndarray, arm_q: np.ndarray) -> bool:
        """After a grasp cycle, sustained hold-position predictions = done."""
        if self.did_grasp and not self.grip_closed:
            if np.abs(chunk[:, :7] - arm_q).max() < 0.05:
                self.still_chunks += 1
            else:
                self.still_chunks = 0
            return self.still_chunks >= 4
        return False

    def run_episode(self, executor: ThreadPoolExecutor) -> None:
        from galbot_sdk.g1 import ControlStatus

        obs, arm_q = self.capture(tag="chunk000")
        chunk = self.infer(obs)
        trim = 0  # first chunk: arm was still during inference, keep all steps

        for chunk_idx in range(self.args.max_chunks):
            body = chunk[trim:]
            print(
                f"[chunk {chunk_idx}] infer {self.last_latency * 1e3:.0f} ms | "
                f"trim {trim} | action[0] {np.round(body[0], 3)} | "
                f"grip {body[0, 7]:.2f}→{body[-1, 7]:.2f}"
            )

            ok, reason = chunk_is_safe(arm_q, body)
            if not ok:
                print(f"❌ unsafe chunk, stopping: {reason}")
                break
            if self.episode_done(body, arm_q):
                print("✅ policy settled after grasp cycle — episode complete")
                break
            if not self.args.execute:
                obs, arm_q = self.capture()
                chunk, trim = self.infer(obs), 0
                continue

            event = self.find_grip_event(body)
            if event is not None:
                # Move to the event step, actuate blocking, restart pipeline.
                cut, action = event
                if cut > 0:
                    status = self.execute_arm(body[:cut])
                    if status != ControlStatus.SUCCESS:
                        print(f"❌ trajectory execution failed: {status}")
                        break
                self.set_gripper(close=(action == "close"))
                print(f"  → gripper {action.upper()} at step {cut}")
                obs, arm_q = self.capture(tag=f"chunk{chunk_idx + 1:03d}")
                chunk, trim = self.infer(obs), 0
                continue

            # Pipelined: start (non-blocking-by-thread) execution, capture the
            # next observation latency_budget before the chunk ends, infer in
            # parallel, then join.
            exec_future = executor.submit(self.execute_arm, body)
            exec_duration = 0.05 + len(body) * self.dt
            time.sleep(max(0.0, exec_duration - self.args.latency_budget))
            obs, _ = self.capture(tag=f"chunk{chunk_idx + 1:03d}")
            next_chunk = self.infer(obs)
            status = exec_future.result()
            if status != ControlStatus.SUCCESS:
                print(f"❌ trajectory execution failed: {status}")
                break
            if self.last_latency > self.args.latency_budget:
                print(
                    f"  ⚠ inference {self.last_latency * 1e3:.0f} ms exceeded "
                    f"budget {self.args.latency_budget * 1e3:.0f} ms (brief pause)"
                )
            chunk, trim = next_chunk, self.skip
            arm_q = self.arm_pos()

        print("episode finished (operator judges success)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--execute", action="store_true", help="move the robot (default: dry-run)"
    )
    parser.add_argument(
        "--go-home", action="store_true", help="restore data-collection posture first"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=90, help="episode length limit"
    )
    parser.add_argument(
        "--slow",
        type=float,
        default=1.0,
        help="time-stretch factor for execution (2.0 = half speed)",
    )
    parser.add_argument(
        "--latency-budget",
        type=float,
        default=0.35,
        help="seconds reserved for capture+inference inside each chunk; "
        "must exceed the observed inference latency",
    )
    parser.add_argument(
        "--save-obs",
        default=None,
        metavar="DIR",
        help="save the head/wrist images fed to the policy for inspection",
    )
    args = parser.parse_args()

    import galbot_sdk.g1 as gm
    from galbot_sdk.g1 import ControlStatus, GalbotRobot, SensorType
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"policy server metadata: {policy.get_server_metadata()}")

    if args.execute:
        print(
            "⚠️  EXECUTE mode: ensure the E-stop is released, the workspace is "
            "clear, and one hand is on the E-stop during the whole run."
        )
        confirm_or_exit("confirm safety conditions")

    robot = None
    try:
        robot = GalbotRobot()
        sensors = {SensorType.HEAD_LEFT_CAMERA, SensorType.RIGHT_ARM_CAMERA}
        if not robot.init(sensors):
            raise SystemExit("GalbotRobot init failed")
        time.sleep(3)  # let sensor streams start

        if args.go_home:
            if args.execute:
                confirm_or_exit(
                    "move to data-collection posture? "
                    f"leg {HOME_LEG}, head {HOME_HEAD}, arm {HOME_RIGHT_ARM}"
                )
                for group, target in [
                    ("leg", HOME_LEG),
                    ("head", HOME_HEAD),
                    (ARM_GROUP, HOME_RIGHT_ARM),
                ]:
                    status = robot.set_joint_positions(
                        target,
                        joint_groups=[group],
                        joint_names=[],
                        is_blocking=True,
                        speed_rad_s=0.15,
                        timeout_s=40.0,
                    )
                    if status != ControlStatus.SUCCESS:
                        raise SystemExit(f"go-home failed on {group}: {status}")
                robot.set_gripper_command(
                    GRIPPER_NAME, GRIPPER_FULL_OPEN_M, GRIP_VELOCITY_MPS, 30, True
                )
            else:
                print(
                    f"[dry-run] would move leg {HOME_LEG}, head {HOME_HEAD}, "
                    f"arm {HOME_RIGHT_ARM} and open gripper"
                )

        runner = G1PolicyRunner(robot, gm, policy, args)
        runner.grip_closed = (
            robot.get_gripper_state(GRIPPER_NAME).width / GRIPPER_FULL_OPEN_M < 0.5
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            runner.run_episode(executor)
    finally:
        if robot is not None:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()


if __name__ == "__main__":
    main()

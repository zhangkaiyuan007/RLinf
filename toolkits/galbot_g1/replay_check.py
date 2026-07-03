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
"""Offline replay check: run the packaged policy on training frames.

Feeds recorded observations to the policy and compares predicted action
chunks against the teleop ground truth. This validates weight loading,
transforms, and normalization end-to-end before touching the real robot.
Expect small errors on training data (the model has seen it); large or
structured errors indicate a broken pipeline, not a bad policy.

Run inside the RLinf venv on the training node:

    python toolkits/galbot_g1/replay_check.py \\
        --checkpoint-dir ~/ckpts/pi05_g1_sft_step3000 \\
        --dataset ~/lerobot_data/galbot_g1_cola_basket \\
        --episodes 0 5 25 --out-dir ~/replay_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

HORIZON = 10
TRAIN_PROMPT = "put the cola bottle into the basket"
DIM_NAMES = [f"joint{i}" for i in range(1, 8)] + ["gripper"]


def to_hwc_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 10, 25, 40])
    parser.add_argument("--stride", type=int, default=5, help="frame subsampling")
    parser.add_argument("--out-dir", type=Path, default=Path("~/replay_check"))
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies import policy_config as _policy_config

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = _policy_config.create_trained_policy(
        get_openpi_config("pi05_galbot_g1"),
        args.checkpoint_dir.expanduser(),
        default_prompt=TRAIN_PROMPT,
    )

    dataset = LeRobotDataset(
        "galbot_g1_cola_basket",
        root=args.dataset.expanduser(),
        delta_timestamps={"actions": [i / 15 for i in range(HORIZON)]},
    )

    all_err = []
    for ep in args.episodes:
        i0 = dataset.episode_data_index["from"][ep].item()
        i1 = dataset.episode_data_index["to"][ep].item()
        frames = range(i0, i1 - HORIZON, args.stride)

        gt_all, pred_all, err = [], [], []
        for i in frames:
            item = dataset[i]
            obs = {
                "observation/image": to_hwc_uint8(item["image"]),
                "observation/extra_view_image": to_hwc_uint8(
                    item["extra_view_image"]
                ),
                "observation/state": item["state"].numpy().astype(np.float32),
                "prompt": TRAIN_PROMPT,
            }
            pred = np.asarray(policy.infer(obs)["actions"])[:HORIZON]
            gt = item["actions"].numpy()[:HORIZON]
            gt_all.append(gt)
            pred_all.append(pred)
            err.append(np.abs(pred - gt))

        err = np.stack(err)  # (n, HORIZON, 8)
        all_err.append(err.reshape(-1, 8))
        mae = err.mean(axis=(0, 1))
        print(f"episode {ep}: {err.shape[0]} frames")
        for name, m in zip(DIM_NAMES, mae):
            print(f"  {name:<8} MAE {m:.4f}")

        # Overlay plot: GT vs predicted first-step action per frame.
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gt0 = np.stack([g[0] for g in gt_all])
            pr0 = np.stack([p[0] for p in pred_all])
            fig, axes = plt.subplots(8, 1, figsize=(10, 14), sharex=True)
            for d, ax in enumerate(axes):
                ax.plot(gt0[:, d], label="teleop GT", color="g")
                ax.plot(pr0[:, d], label="policy", color="r", ls="--")
                ax.set_ylabel(DIM_NAMES[d])
            axes[0].legend()
            axes[0].set_title(f"episode {ep}: first-step action, GT vs policy")
            fig.savefig(out_dir / f"replay_ep{ep}.png", dpi=100)
            plt.close(fig)
        except ImportError:
            pass

    total = np.concatenate(all_err)
    print("\noverall MAE per dim:")
    for name, m in zip(DIM_NAMES, total.mean(axis=0)):
        print(f"  {name:<8} {m:.4f}")
    print(
        f"\narm MAE {total[:, :7].mean():.4f} rad "
        f"(rule of thumb: <0.05 pipeline OK, >0.3 something is wrong)"
    )
    print(f"plots saved to {out_dir}")


if __name__ == "__main__":
    main()

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
"""Serve the Galbot G1 pi0.5 policy over openpi's websocket protocol.

Run inside the RLinf venv on a GPU node, with a checkpoint packaged by
``package_sft_checkpoint.py``:

    python toolkits/galbot_g1/serve_g1_policy.py \\
        --checkpoint-dir ~/ckpts/pi05_g1_sft_step3000 --port 8000

Client obs dict (see ``g1_pi05_bridge.py``):

- ``observation/image``: head RGB uint8 HWC
- ``observation/extra_view_image``: right-wrist RGB uint8 HWC
- ``observation/state``: float (8,) — 7 right-arm joints (rad) + gripper [0,1]
- ``prompt``: task string (defaults to the training prompt if omitted)

Response: ``{"actions": (10, 8) float}`` — absolute joint targets + gripper.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

TRAIN_PROMPT = "put the cola bottle into the basket"


def self_test(policy) -> None:
    """Run one inference on a synthetic observation and report shape/latency."""
    obs = {
        "observation/image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/extra_view_image": np.random.randint(
            256, size=(360, 640, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(8).astype(np.float32),
        "prompt": TRAIN_PROMPT,
    }
    policy.infer(obs)  # warmup (compilation / cudnn autotune)
    start = time.monotonic()
    result = policy.infer(obs)
    latency_ms = (time.monotonic() - start) * 1e3
    actions = np.asarray(result["actions"])
    print(f"self-test ok: actions {actions.shape}, {latency_ms:.0f} ms/chunk")
    print(f"  first action: {np.round(actions[0], 3)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="run one synthetic inference and exit without serving",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server

    train_config = get_openpi_config("pi05_galbot_g1")
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir.expanduser(),
        default_prompt=TRAIN_PROMPT,
    )

    self_test(policy)
    if args.self_test_only:
        return

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={"robot": "galbot_g1", "prompt": TRAIN_PROMPT},
    )
    print(f"serving G1 pi0.5 policy on port {args.port} ...")
    server.serve_forever()


if __name__ == "__main__":
    main()

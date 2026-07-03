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
"""Package an RLinf SFT checkpoint into an openpi-format checkpoint directory.

RLinf's FSDP SFT saves ``.../global_step_N/actor/model_state_dict/full_weights.pt``
(a plain full state dict). openpi's ``create_trained_policy`` expects:

    <out_dir>/
      model.safetensors
      assets/<asset_id>/norm_stats.json

Run inside the RLinf venv on the training node:

    python toolkits/galbot_g1/package_sft_checkpoint.py \\
        --full-weights ~/RLinf/logs/.../global_step_3000/actor/model_state_dict/full_weights.pt \\
        --norm-stats   ~/ckpts/pi05_base_pytorch/galbot_g1_cola_basket/norm_stats.json \\
        --out-dir      ~/ckpts/pi05_g1_sft_step3000
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ASSET_ID = "galbot_g1_cola_basket"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-weights", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-safetensors",
        type=Path,
        required=True,
        help="model.safetensors of the converted base checkpoint; the packaged "
        "state dict is filtered to exactly this key set (the training-side "
        "module registers tied duplicates, e.g. embed_tokens vs lm_head, that "
        "the inference-side PI0Pytorch does not expect)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.full_weights} ...")
    state_dict = torch.load(
        args.full_weights.expanduser(), map_location="cpu", weights_only=True
    )
    if not isinstance(state_dict, dict):
        raise TypeError(f"expected a state dict, got {type(state_dict)}")

    # safetensors requires contiguous, non-shared storage.
    seen_ptrs: dict[int, str] = {}
    clean: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.contiguous()
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs and tensor.numel() > 0:
            tensor = tensor.clone()
        seen_ptrs[ptr] = key
        clean[key] = tensor

    from safetensors import safe_open

    with safe_open(
        args.reference_safetensors.expanduser(), framework="pt"
    ) as ref:
        ref_keys = set(ref.keys())

    missing = sorted(ref_keys - set(clean))
    if missing:
        raise KeyError(
            f"SFT state dict is missing {len(missing)} keys expected by the "
            f"inference model, e.g. {missing[:5]} — wrong checkpoint?"
        )
    for key in sorted(set(clean) - ref_keys):
        tensor = clean.pop(key)
        twin = next(
            (
                k
                for k, v in clean.items()
                if v.shape == tensor.shape and torch.equal(v, tensor)
            ),
            None,
        )
        if twin is not None:
            print(f"dropping extra key {key} (tied duplicate of {twin})")
        else:
            print(
                f"WARNING: dropping extra key {key} with no identical twin — "
                "verify the replay check carefully"
            )

    dtypes = {str(t.dtype) for t in clean.values()}
    n_params = sum(t.numel() for t in clean.values())
    print(f"{len(clean)} tensors, {n_params / 1e9:.2f}B params, dtypes={dtypes}")

    save_file(clean, out_dir / "model.safetensors")

    assets_dir = out_dir / "assets" / ASSET_ID
    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.norm_stats.expanduser(), assets_dir / "norm_stats.json")

    print(f"packaged openpi checkpoint at {out_dir}")


if __name__ == "__main__":
    main()

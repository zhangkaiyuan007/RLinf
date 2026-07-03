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
"""OpenPI input/output transforms for Galbot G1 single-arm joint-space data.

Contract (matches ``toolkits/galbot_g1/convert_mcap_to_lerobot.py``):

- ``observation/image``: head camera RGB (front_head left color).
- ``observation/extra_view_image``: right-wrist camera RGB.
- ``observation/state``: 8-dim — 7 right-arm joint positions (rad) + gripper
  open fraction in [0, 1] (1 = fully open = 0.12 m on the real G1).
- ``actions``: (N, 8) joint-space command targets, same layout as the state.

The wrist image is fed to the model's ``left_wrist_0_rgb`` slot; the slot name
is just a position in the π₀/π₀.₅ input layout, and training/inference both
use this mapping consistently.
"""

import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model

GALBOT_G1_STATE_DIM = 8
GALBOT_G1_ACTION_DIM = 8


def make_galbot_g1_example() -> dict:
    """Creates a random input example for the Galbot G1 policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/extra_view_image": np.random.randint(
            256, size=(360, 640, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(GALBOT_G1_STATE_DIM),
        "prompt": "put the cola bottle into the basket",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GalbotG1Outputs(transforms.DataTransformFn):
    """Converts model outputs back to the 8-dim G1 joint-space action format."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :GALBOT_G1_ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class GalbotG1Inputs(transforms.DataTransformFn):
    """Converts Galbot G1 dataset samples to OpenPI inputs."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        assert data["observation/state"].shape == (GALBOT_G1_STATE_DIM,), (
            f"Expected state shape ({GALBOT_G1_STATE_DIM},), "
            f"got {data['observation/state'].shape}"
        )

        if isinstance(data["observation/state"], np.ndarray):
            data["observation/state"] = torch.from_numpy(
                data["observation/state"]
            ).float()

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(
            data.get("observation/extra_view_image", np.zeros_like(base_image))
        )

        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type: {self.model_type}")

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (np.True_, np.True_, np.False_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            assert (
                len(data["actions"].shape) == 2
                and data["actions"].shape[-1] == GALBOT_G1_ACTION_DIM
            ), (
                f"Expected actions shape (N, {GALBOT_G1_ACTION_DIM}), "
                f"got {data['actions'].shape}"
            )
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs

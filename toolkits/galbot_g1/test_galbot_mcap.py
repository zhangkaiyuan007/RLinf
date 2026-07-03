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
"""Unit tests for the pure alignment helpers in galbot_mcap.py.

Run from the repo root: ``pytest toolkits/galbot_g1/test_galbot_mcap.py``.
"""

import numpy as np
import pytest

from toolkits.galbot_g1.galbot_mcap import (
    assemble_state_action,
    decimate_indices,
    gripper_pct_to_norm,
    nearest_indices,
    zoh_indices,
)

NS = 1_000_000_000


class TestDecimateIndices:
    def test_30_to_15_takes_every_second_frame(self):
        np.testing.assert_array_equal(
            decimate_indices(7, 30.0, 15.0), [0, 2, 4, 6]
        )

    def test_30_to_10_takes_every_third_frame(self):
        np.testing.assert_array_equal(decimate_indices(7, 30.0, 10.0), [0, 3, 6])

    def test_same_rate_keeps_all(self):
        np.testing.assert_array_equal(decimate_indices(3, 30.0, 30.0), [0, 1, 2])

    def test_upsampling_rejected(self):
        with pytest.raises(ValueError):
            decimate_indices(10, 30.0, 90.0)


class TestNearestIndices:
    def test_picks_nearest_on_both_sides(self):
        ref = np.array([0, 10, 20], dtype=np.int64)
        query = np.array([1, 6, 14, 19, 25], dtype=np.int64)
        np.testing.assert_array_equal(nearest_indices(query, ref), [0, 1, 1, 2, 2])

    def test_tie_prefers_earlier(self):
        ref = np.array([0, 10], dtype=np.int64)
        query = np.array([5], dtype=np.int64)
        np.testing.assert_array_equal(nearest_indices(query, ref), [0])

    def test_queries_outside_range_clamp_to_ends(self):
        ref = np.array([10, 20], dtype=np.int64)
        query = np.array([-100, 100], dtype=np.int64)
        np.testing.assert_array_equal(nearest_indices(query, ref), [0, 1])

    def test_single_reference(self):
        ref = np.array([10], dtype=np.int64)
        query = np.array([0, 10, 99], dtype=np.int64)
        np.testing.assert_array_equal(nearest_indices(query, ref), [0, 0, 0])

    def test_empty_reference_rejected(self):
        with pytest.raises(ValueError):
            nearest_indices(np.array([1]), np.array([], dtype=np.int64))


class TestZohIndices:
    def test_holds_latest_at_or_before(self):
        ref = np.array([10, 20, 30], dtype=np.int64)
        query = np.array([5, 10, 15, 30, 99], dtype=np.int64)
        np.testing.assert_array_equal(zoh_indices(query, ref), [-1, 0, 0, 2, 2])

    def test_empty_reference_gives_minus_one(self):
        query = np.array([1, 2], dtype=np.int64)
        np.testing.assert_array_equal(
            zoh_indices(query, np.empty(0, dtype=np.int64)), [-1, -1]
        )


class TestGripperPctToNorm:
    def test_maps_percent_to_fraction(self):
        np.testing.assert_allclose(
            gripper_pct_to_norm(np.array([0.0, 54.0, 100.0])), [0.0, 0.54, 1.0]
        )

    def test_clips_out_of_range(self):
        np.testing.assert_allclose(
            gripper_pct_to_norm(np.array([-5.0, 120.0])), [0.0, 1.0]
        )


class TestAssembleStateAction:
    def _streams(self):
        # Sensors at t = 0, 1s, 2s; arm targets at 0.5s, 1.5s; gripper cmd at 1.5s.
        sensor_ts = np.array([0, 1 * NS, 2 * NS], dtype=np.int64)
        sensor_arm = np.arange(21, dtype=np.float64).reshape(3, 7)
        sensor_grip = np.array([100.0, 100.0, 50.0])
        arm_tgt_ts = np.array([NS // 2, 3 * NS // 2], dtype=np.int64)
        arm_tgt = np.stack(
            [np.full(7, 100.0), np.full(7, 200.0)]
        )
        grip_tgt_ts = np.array([3 * NS // 2], dtype=np.int64)
        grip_tgt = np.array([0.0])
        return (
            sensor_ts,
            sensor_arm,
            sensor_grip,
            arm_tgt_ts,
            arm_tgt,
            grip_tgt_ts,
            grip_tgt,
        )

    def test_state_uses_nearest_sensor(self):
        frame_ts = np.array([0, 2 * NS], dtype=np.int64)
        state, _ = assemble_state_action(frame_ts, *self._streams())
        assert state.shape == (2, 8)
        np.testing.assert_allclose(state[0, :7], np.arange(7))
        np.testing.assert_allclose(state[0, 7], 1.0)  # 100 pct -> 1.0
        np.testing.assert_allclose(state[1, :7], np.arange(14, 21))
        np.testing.assert_allclose(state[1, 7], 0.5)  # 50 pct -> 0.5

    def test_action_holds_latest_command(self):
        frame_ts = np.array([1 * NS, 2 * NS], dtype=np.int64)
        _, actions = assemble_state_action(frame_ts, *self._streams())
        np.testing.assert_allclose(actions[0, :7], 100.0)  # cmd from t=0.5s
        np.testing.assert_allclose(actions[1, :7], 200.0)  # cmd from t=1.5s
        np.testing.assert_allclose(actions[1, 7], 0.0)  # close cmd at 1.5s

    def test_action_falls_back_to_state_before_first_command(self):
        frame_ts = np.array([0], dtype=np.int64)
        state, actions = assemble_state_action(frame_ts, *self._streams())
        np.testing.assert_allclose(actions[0, :7], state[0, :7])
        np.testing.assert_allclose(actions[0, 7], 1.0)  # sensor 100 pct

    def test_empty_gripper_commands_fall_back_to_sensor(self):
        (s_ts, s_arm, s_grip, a_ts, a_tgt, _, _) = self._streams()
        frame_ts = np.array([2 * NS], dtype=np.int64)
        _, actions = assemble_state_action(
            frame_ts,
            s_ts,
            s_arm,
            s_grip,
            a_ts,
            a_tgt,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
        np.testing.assert_allclose(actions[0, 7], 0.5)

    def test_outputs_are_float32(self):
        frame_ts = np.array([0], dtype=np.int64)
        state, actions = assemble_state_action(frame_ts, *self._streams())
        assert state.dtype == np.float32
        assert actions.dtype == np.float32

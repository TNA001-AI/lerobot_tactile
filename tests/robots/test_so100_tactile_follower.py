#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.so100_tactile_follower import SO100TactileFollower, SO100TactileFollowerConfig
from lerobot.utils.constants import OBS_TACTILE


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the tactile follower."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture
def tactile_follower():
    bus_mock = _make_bus_mock()
    tactile_sensor_mock = MagicMock(name="TactileSensorMock")
    tactile_sensor_mock.read_data.return_value = np.ones((16, 32), dtype=np.float32)

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)
        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch(
            "lerobot.robots.so100_tactile_follower.so100_tactile_follower.TactileSensor",
            return_value=tactile_sensor_mock,
        ),
        patch.object(SO100TactileFollower, "configure", lambda self: None),
    ):
        cfg = SO100TactileFollowerConfig(port="/dev/null", tactile_port="/dev/null", tactile_auto_calibrate=False)
        robot = SO100TactileFollower(cfg)
        yield robot, tactile_sensor_mock
        if robot.is_connected:
            robot.disconnect()


def test_tactile_follower_observation_includes_tactile_data(tactile_follower):
    robot, tactile_sensor_mock = tactile_follower
    robot.connect()

    observation = robot.get_observation()
    assert OBS_TACTILE in observation
    assert observation[OBS_TACTILE].shape == (16, 32)
    tactile_sensor_mock.read_data.assert_called()


def test_tactile_follower_disconnect_closes_sensor(tactile_follower):
    robot, tactile_sensor_mock = tactile_follower
    robot.connect()
    robot.disconnect()

    tactile_sensor_mock.disconnect.assert_called()

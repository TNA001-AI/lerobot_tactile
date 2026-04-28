#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import contextlib
import logging
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.sensors.tactile_sensor import TactileSensor
from lerobot.utils.constants import OBS_TACTILE

from .config_so_tactile_follower import SOTactileFollowerConfig


class SOTactileFollower(SOFollower):
    """SO Follower robot (SO-100/101/10X) with tactile sensor support."""

    config_class = SOTactileFollowerConfig
    name = "so_tactile_follower"

    def __init__(self, config: SOTactileFollowerConfig):
        super().__init__(config)

        self._tactile_sensors: dict[str, TactileSensor] = {}

        for name, sensor_cfg in config.tactile_sensors.items():
            try:
                sensor = TactileSensor(
                    port=sensor_cfg.port,
                    baud_rate=sensor_cfg.baud_rate,
                    shape=sensor_cfg.shape,
                    auto_calibrate=sensor_cfg.auto_calibrate,
                    enable_visualization=sensor_cfg.enable_visualization,
                )
                self._tactile_sensors[name] = sensor
                sensor.start_continuous_read()
                logging.info(f"Tactile sensor '{name}' initialized on {sensor_cfg.port}")
            except Exception as e:
                logging.error(f"Failed to initialize tactile sensor '{name}': {e}")

    def get_observation(self) -> dict[str, Any]:
        """Get robot observation including tactile sensor data."""
        observation = super().get_observation()

        for name, sensor in self._tactile_sensors.items():
            obs_key = f"{OBS_TACTILE}.{name}"
            sensor_cfg = self.config.tactile_sensors[name]
            try:
                data = sensor.get_latest_data()
                if data is None:
                    logging.warning(f"Failed to read tactile sensor '{name}', using zeros")
                    data = np.zeros(sensor_cfg.shape, dtype=np.float32)
                observation[obs_key] = data
            except Exception as e:
                logging.error(f"Error reading tactile sensor '{name}': {e}")
                observation[obs_key] = np.zeros(sensor_cfg.shape, dtype=np.float32)

        return observation

    @property
    def observation_features(self) -> dict[str, PolicyFeature]:
        """Define observation features including tactile sensor(s)."""
        features = super().observation_features

        for name, sensor_cfg in self.config.tactile_sensors.items():
            features[f"tactile.{name}"] = PolicyFeature(
                type=FeatureType.TACTILE,
                shape=sensor_cfg.shape,
            )

        return features

    def disconnect(self):
        """Disconnect from robot and all tactile sensors."""
        for name, sensor in self._tactile_sensors.items():
            try:
                sensor.disconnect()
                logging.info(f"Tactile sensor '{name}' disconnected")
            except Exception as e:
                logging.error(f"Error disconnecting tactile sensor '{name}': {e}")

        super().disconnect()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.disconnect()

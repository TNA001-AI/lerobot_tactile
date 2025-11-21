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

import logging
from typing import Any, Optional

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.sensors.tactile_sensor import TactileSensor
from lerobot.utils.constants import OBS_TACTILE, OBS_TACTILES

from .config_so100_tactile_follower import SO100TactileFollowerConfig


class SO100TactileFollower(SO100Follower):
    """SO100 Follower robot with tactile sensor support"""
    
    config_class = SO100TactileFollowerConfig
    name = "so100_tactile_follower"

    def __init__(self, config: SO100TactileFollowerConfig):
        # Initialize base SO100Follower
        super().__init__(config)
        
        self.tactile_sensors: dict[str, TactileSensor] = {}
        self.config = config
        
        # Initialize tactile sensors if enabled
        if config.tactile_enabled:
            if config.tactile_sensors:
                # Initialize multiple named sensors
                for name, sensor_config in config.tactile_sensors.items():
                    try:
                        sensor = TactileSensor(
                            port=sensor_config["port"],
                            baud_rate=sensor_config.get("baud_rate", 2000000),
                            shape=config.tactile_shape,
                            auto_calibrate=config.tactile_auto_calibrate,
                            enable_visualization= True
                        )
                        self.tactile_sensors[name] = sensor
                        logging.info(f"Tactile sensor '{name}' initialized on {sensor_config['port']}")
                    except Exception as e:
                        logging.error(f"Failed to initialize tactile sensor '{name}': {e}")
            else:
                # Legacy mode: single or dual sensor
                try:
                    # Primary sensor
                    sensor = TactileSensor(
                        port=config.tactile_port,
                        baud_rate=config.tactile_baud_rate,
                        shape=config.tactile_shape,
                        auto_calibrate=config.tactile_auto_calibrate,
                        enable_visualization=True,
                    )
                    self.tactile_sensors["primary"] = sensor
                    logging.info(f"Primary tactile sensor initialized on {config.tactile_port}")

                    # Secondary sensor if specified
                    tactile_port_2 = getattr(config, 'tactile_port_2', None)
                    if tactile_port_2:
                        tactile_baud_rate_2 = getattr(config, 'tactile_baud_rate_2', 2000000)
                        sensor_2 = TactileSensor(
                            port=tactile_port_2,
                            baud_rate=tactile_baud_rate_2,
                            shape=config.tactile_shape,
                            auto_calibrate=config.tactile_auto_calibrate,
                        )
                        self.tactile_sensors["secondary"] = sensor_2
                        logging.info(f"Secondary tactile sensor initialized on {tactile_port_2}")
                        
                except Exception as e:
                    logging.error(f"Failed to initialize tactile sensors: {e}")

    def get_observation(self) -> dict[str, Any]:
        """Get robot observation including tactile data from multiple sensors"""
        # Get base observation from SO100Follower
        observation = super().get_observation()
        
        # Add tactile data from multiple sensors
        if self.config.tactile_enabled and self.tactile_sensors:
            if self.config.tactile_sensors:
                # Named sensors: create separate observation keys
                for name, sensor in self.tactile_sensors.items():
                    obs_key = f"{OBS_TACTILE}.{name}"
                    try:
                        tactile_data = sensor.read_data()
                        if tactile_data is not None:
                            observation[obs_key] = tactile_data
                        else:
                            logging.warning(f"Failed to read tactile data from '{name}', using zeros")
                            observation[obs_key] = np.zeros(self.config.tactile_shape, dtype=np.float32)
                    except Exception as e:
                        logging.error(f"Error reading tactile sensor '{name}': {e}")
                        observation[obs_key] = np.zeros(self.config.tactile_shape, dtype=np.float32)
            else:
                # Legacy mode: use list or single sensor
                tactile_data_list = []
                for name, sensor in self.tactile_sensors.items():
                    try:
                        tactile_data = sensor.read_data()
                        if tactile_data is not None:
                            tactile_data_list.append(tactile_data)
                        else:
                            logging.warning(f"Failed to read tactile data from '{name}', using zeros")
                            tactile_data_list.append(np.zeros(self.config.tactile_shape, dtype=np.float32))
                    except Exception as e:
                        logging.error(f"Error reading tactile sensor '{name}': {e}")
                        tactile_data_list.append(np.zeros(self.config.tactile_shape, dtype=np.float32))
                
                if len(tactile_data_list) == 1:
                    # Single sensor: backward compatibility
                    observation[OBS_TACTILE] = tactile_data_list[0]
                else:
                    # Multiple sensors: use list
                    observation[OBS_TACTILES] = tactile_data_list
                    
        elif self.config.tactile_enabled:
            # If tactile is enabled but no sensors initialized, provide zeros
            observation[OBS_TACTILE] = np.zeros(self.config.tactile_shape, dtype=np.float32)
        
        return observation

    @property
    def observation_features(self) -> dict[str, PolicyFeature]:
        """Define observation features including multiple tactile sensors"""
        features = super().observation_features
        
        # Add tactile features if enabled
        # Use simple key "tactile" so prefix is added correctly in hw_to_dataset_features
        if self.config.tactile_enabled:
            if self.config.tactile_sensors:
                # Named sensors: separate features for each
                for name in self.config.tactile_sensors.keys():
                    features[f"tactile.{name}"] = PolicyFeature(
                        type=FeatureType.TACTILE,
                        shape=self.config.tactile_shape,
                    )
            else:
                # Legacy mode: single tactile feature
                features["tactile"] = PolicyFeature(
                    type=FeatureType.TACTILE,
                    shape=self.config.tactile_shape,
                )
        
        return features

    def disconnect(self):
        """Disconnect from robot and all tactile sensors"""
        # Disconnect all tactile sensors
        for name, sensor in self.tactile_sensors.items():
            try:
                sensor.disconnect()
                logging.info(f"Tactile sensor '{name}' disconnected")
            except Exception as e:
                logging.error(f"Error disconnecting tactile sensor '{name}': {e}")
        
        # Disconnect base robot
        super().disconnect()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.disconnect()
        except:
            pass  # Ignore cleanup errors
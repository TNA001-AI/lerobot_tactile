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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so100_tactile_follower")
@dataclass
class SO100TactileFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # Tactile sensor configuration
    tactile_enabled: bool = True
    tactile_shape: tuple[int, int] = (16, 32)
    tactile_auto_calibrate: bool = True
    
    # Multiple tactile sensors support
    # Option 1: Dictionary of named sensors
    tactile_sensors: dict[str, dict] = field(default_factory=lambda: {
        # Example: Two finger sensors
        # "left": {"port": "/dev/ttyUSB0", "baud_rate": 2000000},
        # "right": {"port": "/dev/ttyUSB1", "baud_rate": 2000000}
    })
    
    # Option 2: Simple dual sensor setup (legacy support)
    tactile_port: str = "/dev/ttyUSB0"
    tactile_baud_rate: int = 2000000
    # tactile_port_2: str | None = None  # Second sensor port
    # tactile_baud_rate_2: int = 2000000
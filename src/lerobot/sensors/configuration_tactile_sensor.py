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

from dataclasses import dataclass


@dataclass
class TactileSensorConfig:
    """Configuration for a tactile sensor connected via USB serial.

    Example (single sensor)::

        TactileSensorConfig(port="/dev/ttyUSB0")

    Example (dual sensor in robot config)::

        tactile_sensors = {
            "left":  TactileSensorConfig(port="/dev/ttyUSB0"),
            "right": TactileSensorConfig(port="/dev/ttyUSB1"),
        }
    """

    port: str = "/dev/ttyUSB0"
    baud_rate: int = 2_000_000
    shape: tuple[int, int] = (12, 32)
    # Leave as None to auto-calibrate on first read. Set to a scalar (e.g. 20.0)
    # to skip calibration and use a fixed baseline applied to every pixel.
    baseline: float | None = None
    enable_visualization: bool = True

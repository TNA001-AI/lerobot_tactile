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
"""SO100 follower robot with tactile sensor support"""

from lerobot.robots.so100_tactile_follower.config_so100_tactile_follower import SO100TactileFollowerConfig
from lerobot.robots.so100_tactile_follower.so100_tactile_follower import SO100TactileFollower

__all__ = ["SO100TactileFollowerConfig", "SO100TactileFollower"]
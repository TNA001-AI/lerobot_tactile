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

import draccus

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.robots.so100_tactile_follower import SO100TactileFollowerConfig
from lerobot.scripts.lerobot_record import RecordConfig


def test_record_cli_parses_so100_tactile_follower():
    cfg = draccus.parse(
        RecordConfig,
        args=[
            "--robot.type=so100_tactile_follower",
            "--robot.port=/dev/null",
            "--robot.tactile_port=/dev/null",
            "--dataset.repo_id=test/tactile",
            "--dataset.single_task=pick and place",
            "--teleop.type=so100_leader",
            "--teleop.port=/dev/null",
        ],
    )

    assert isinstance(cfg.robot, SO100TactileFollowerConfig)
    assert cfg.robot.type == "so100_tactile_follower"
    assert cfg.teleop is not None
    assert cfg.teleop.type == "so100_leader"


def test_train_cli_parses_act_with_tactile_enabled():
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=test/tactile",
            "--policy.type=act",
            "--policy.device=cpu",
            "--policy.use_tactile=true",
        ],
    )

    assert cfg.policy is not None
    assert isinstance(cfg.policy, ACTConfig)
    assert cfg.policy.use_tactile is True

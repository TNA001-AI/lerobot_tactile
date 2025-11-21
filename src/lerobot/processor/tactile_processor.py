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
"""Tactile sensor data processing steps"""

import torch
import numpy as np
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.utils.constants import OBS_TACTILE


class TactileNormalizationProcessorStep(ObservationProcessorStep):
    """Normalize tactile sensor data with thresholding and noise filtering"""

    def __init__(self, threshold=30, noise_scale=50, gaussian_sigma=0.1):
        """
        Args:
            threshold: Minimum tactile value threshold
            noise_scale: Scale for noise normalization
            gaussian_sigma: Gaussian noise standard deviation for data augmentation
        """
        self.threshold = threshold
        self.noise_scale = noise_scale
        self.gaussian_sigma = gaussian_sigma

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        if OBS_TACTILE in obs:
            tactile_data = obs[OBS_TACTILE]

            # Convert to tensor if numpy array
            if isinstance(tactile_data, np.ndarray):
                tactile_data = torch.from_numpy(tactile_data).float()

            # Apply threshold
            tactile_data = torch.clamp(tactile_data - self.threshold, min=0)

            # Normalize by noise scale
            tactile_data = tactile_data / self.noise_scale

            # Ensure data is in range [0, 1]
            tactile_data = torch.clamp(tactile_data, 0, 1)

            obs[OBS_TACTILE] = tactile_data

        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Features remain unchanged after normalization"""
        return features


class TactileValidationProcessorStep(ObservationProcessorStep):
    """Validate tactile sensor data format and dimensions"""

    def __init__(self, expected_shape=(16, 32)):
        """
        Args:
            expected_shape: Expected shape of tactile sensor array (H, W)
        """
        self.expected_shape = tuple(expected_shape)

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        if OBS_TACTILE in obs:
            tactile_data = obs[OBS_TACTILE]

            # Convert to tensor if needed
            if isinstance(tactile_data, np.ndarray):
                tactile_data = torch.from_numpy(tactile_data).float()

            # Check dimensions
            if tactile_data.dim() == 2:
                # Keep as (H, W) for single sample
                pass
            elif tactile_data.dim() == 3:
                # (B, H, W) format - keep as is
                pass
            elif tactile_data.dim() == 4 and tactile_data.shape[1] == 1:
                # Remove channel dimension if present: (B, 1, H, W) -> (B, H, W)
                tactile_data = tactile_data.squeeze(1)

            # Validate shape (ignoring batch dimension)
            actual_shape = tuple(tactile_data.shape[-2:])
            if actual_shape != self.expected_shape:
                raise ValueError(
                    f"Tactile data shape mismatch. Expected {self.expected_shape}, "
                    f"got {actual_shape}"
                )

            obs[OBS_TACTILE] = tactile_data

        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Features remain unchanged after validation"""
        return features


class TactileTemporalFilterProcessorStep(ObservationProcessorStep):
    """Apply temporal filtering to tactile sensor data to reduce noise"""

    def __init__(self, alpha=0.2):
        """
        Args:
            alpha: Exponential moving average coefficient (0 < alpha <= 1)
                  Lower values = more smoothing
        """
        self.alpha = alpha
        self.prev_tactile = None

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        if OBS_TACTILE in obs:
            tactile_data = obs[OBS_TACTILE]

            if isinstance(tactile_data, np.ndarray):
                tactile_data = torch.from_numpy(tactile_data).float()

            # Apply temporal filtering
            if self.prev_tactile is not None:
                # Exponential moving average: new = alpha * current + (1-alpha) * previous
                tactile_data = self.alpha * tactile_data + (1 - self.alpha) * self.prev_tactile

            # Store current values for next iteration
            self.prev_tactile = tactile_data.clone().detach()

            obs[OBS_TACTILE] = tactile_data

        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Features remain unchanged after temporal filtering"""
        return features

    def reset(self):
        """Reset temporal filter state"""
        self.prev_tactile = None

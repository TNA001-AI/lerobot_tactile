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
"""Shared tactile encoder modules for use across different policies (ACT, Diffusion, Pi0.5).

Each encoder accepts a tactile map of shape (B, H, W) and produces token embeddings
of shape (B, n_tokens, feature_dim), where n_tokens is configurable at construction time.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class TactileCNN(nn.Module):
    """Tactile CNN backbone — outputs (B, feature_dim)."""

    def __init__(self, input_shape: tuple[int, int] = (16, 32), feature_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        feature_h = input_shape[0] // 8
        feature_w = input_shape[1] // 8
        conv_output_dim = 128 * feature_h * feature_w

        self.fc1 = nn.Linear(conv_output_dim, 512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TactileAttentionCNN(nn.Module):
    """Tactile CNN with spatial attention — outputs (B, feature_dim)."""

    def __init__(self, input_shape: tuple[int, int] = (16, 32), feature_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        attention_weights = self.attention(x)
        x = x * attention_weights
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TactileTokenEncoder(nn.Module):
    """Wraps a tactile CNN backbone to produce multiple token embeddings.

    Args:
        encoder_type: "cnn" or "attention"
        input_shape: (H, W) of the tactile sensor grid
        feature_dim: embedding dimension per token
        n_tokens: number of tokens to produce per tactile sensor observation
        dropout: dropout rate for the backbone

    Forward input:  (B, H, W) tactile map
    Forward output: (B, n_tokens, feature_dim) token embeddings
    """

    def __init__(
        self,
        encoder_type: str,
        input_shape: tuple[int, int],
        feature_dim: int,
        n_tokens: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.feature_dim = feature_dim

        if encoder_type == "cnn":
            self.backbone = TactileCNN(input_shape, feature_dim, dropout)
        elif encoder_type == "attention":
            self.backbone = TactileAttentionCNN(input_shape, feature_dim, dropout)
        else:
            raise ValueError(f"Unknown tactile encoder type: {encoder_type!r}. Choose 'cnn' or 'attention'.")

        # Project to multiple tokens when n_tokens > 1
        self.token_proj = nn.Linear(feature_dim, n_tokens * feature_dim) if n_tokens > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, H, W) tactile sensor map
        Returns:
            (B, n_tokens, feature_dim)
        """
        feat = self.backbone(x)  # (B, feature_dim)
        if self.n_tokens == 1:
            return feat.unsqueeze(1)  # (B, 1, feature_dim)
        feat = self.token_proj(feat)  # (B, n_tokens * feature_dim)
        return feat.view(feat.size(0), self.n_tokens, self.feature_dim)  # (B, n_tokens, feature_dim)

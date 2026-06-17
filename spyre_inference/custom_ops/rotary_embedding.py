# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre OOT replacement for RotaryEmbedding (CPU fallback).

Remove this file once Spyre natively supports rotary embedding ops.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from functools import lru_cache

from .utils import convert

logger = init_logger(__name__)


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """OOT RotaryEmbedding that falls back to CPU execution.

    Keeps cos_sin_cache on CPU via an _apply no-op. Inputs are moved to
    CPU for computation, and outputs are copied back to the original device.
    """

    def _apply(self, fn, recurse=True):
        # Keep cos_sin_cache on CPU so forward_native can use it directly.
        return self

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # positions arrive on Spyre
        target_device = positions.device
        target_dtype = query.dtype

        cpu_positions = convert(positions, device="cpu")
        cpu_query = convert(query, device="cpu")
        cpu_key = convert(key, device="cpu")

        result_query, result_key = RotaryEmbedding.forward_native(
            self,
            cpu_positions,
            cpu_query,
            cpu_key,
        )

        out_query = convert(result_query, device=target_device, dtype=target_dtype)
        out_key = (
            convert(result_key, device=target_device, dtype=target_dtype)
            if result_key is not None
            else None
        )
        return out_query, out_key


@RotaryEmbeddingBase.register_oot(name="Llama3RotaryEmbedding")
class SpyreLlama3RotaryEmbedding(Llama3RotaryEmbedding):
    """OOT Llama3RotaryEmbedding that runs rotary computation on CPU.

    Llama3RotaryEmbedding adds frequency scaling on top of base rotary embedding
    (for Llama-3.1 extended context). The scaling is applied in _compute_inv_freq,
    which computes the cos_sin_cache during __init__. This class:

    1. Inherits _compute_inv_freq from Llama3RotaryEmbedding → scaling preserved
    2. Overrides forward() to run on CPU → avoids index_select on Spyre
    3. Overrides _apply() to keep cos_sin_cache on CPU

    The frequency scaling (factor, low_freq_factor, high_freq_factor) is computed
    once during initialization and stored in the cos_sin_cache, so forward() just
    applies the pre-scaled values.
    """

    def _apply(self, fn, recurse=True):
        # Keep cos_sin_cache on CPU so forward() can use it directly.
        return self

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # positions arrive on Spyre
        target_device = positions.device
        target_dtype = query.dtype

        cpu_positions = convert(positions, device="cpu")
        cpu_query = convert(query, device="cpu")
        cpu_key = convert(key, device="cpu")

        # Llama3RotaryEmbedding.forward_native applies the scaled cos_sin_cache
        result_query, result_key = Llama3RotaryEmbedding.forward_native(
            self,
            cpu_positions,
            cpu_query,
            cpu_key,
        )

        out_query = convert(result_query, device=target_device, dtype=target_dtype)
        out_key = (
            convert(result_key, device=target_device, dtype=target_dtype)
            if result_key is not None
            else None
        )
        return out_query, out_key


@lru_cache(maxsize=1)
def register():
    # No-op: RotaryEmbedding doesn't require custom op registration.

    # Unlike other Spyre layers (RMSNorm, SiluAndMul, etc.), RotaryEmbedding
    # only needs a class replacement that overrides _apply() to keep weights on CPU.
    # This replacement happens at import time via @RotaryEmbedding.register_oot().
    pass

# Copyright 2026 FlagOS Contributors
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

import os

import pytest
import torch
import flaggems_vllm

from . import base


def _default_fp8_dtype():
    try:
        from vllm.platforms import current_platform

        return current_platform.fp8_dtype()
    except ImportError:
        pass

    if getattr(torch.version, "hip", None) is not None and hasattr(
        torch, "float8_e4m3fnuz"
    ):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    pytest.skip("float8_e4m3fn is required for indexer_k_quant_and_cache")


def _is_fp8_fnuz(dtype):
    return hasattr(torch, "float8_e4m3fnuz") and dtype == torch.float8_e4m3fnuz


def torch_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    head_dim = k.shape[-1]
    num_quant_blocks = head_dim // quant_block_size
    fp8_dtype = _default_fp8_dtype()
    scale_divisor = 224.0 if _is_fp8_fnuz(fp8_dtype) else 448.0

    flat_cache = kv_cache.view(num_blocks, -1)
    cache_values = flat_cache[:, : block_size * head_dim].view(fp8_dtype)
    cache_scales = flat_cache[:, block_size * head_dim :].view(torch.float32)

    for token_idx in range(slot_mapping.numel()):
        slot_id = int(slot_mapping[token_idx].item())
        if slot_id < 0:
            continue

        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        for quant_block_id in range(num_quant_blocks):
            start = quant_block_id * quant_block_size
            end = start + quant_block_size
            val = k[token_idx, start:end]
            amax = val.abs().to(torch.float32).amax()

            scale = (
                torch.maximum(
                    amax,
                    torch.tensor(1e-4, dtype=torch.float32, device=k.device),
                )
                / scale_divisor
            )
            if scale_fmt == "ue8m0":
                scale = torch.exp2(torch.ceil(torch.log2(scale)))

            value_start = block_offset * head_dim + start
            value_end = value_start + quant_block_size
            cache_values[block_id, value_start:value_end].copy_(
                (val.to(torch.float32) / scale).to(fp8_dtype)
            )
            cache_scales[
                block_id,
                block_offset * num_quant_blocks + quant_block_id,
            ] = scale


def vllm_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )


try:
    import vllm._custom_ops as ops  # noqa: F401
    if hasattr(torch.ops._C_cache_ops, "indexer_k_quant_and_cache"):
        ref_indexer = vllm_indexer
    else:
        ref_indexer = torch_indexer
except Exception:
    ref_indexer = torch_indexer


class IndexerKQuantAndCacheBenchmark(base.Benchmark):
    def __init__(self, vllm_op):
        super().__init__(
            op_name="indexer_k_quant_and_cache",
            torch_op=vllm_op,
            dtypes=[torch.float16, torch.bfloat16],  # vLLM supports both K dtypes.
        )
        self.set_gems(flaggems_vllm.indexer_k_quant_and_cache)
        self.shape_desc = (
            "num_tokens, num_blocks, block_size, head_dim, quant_block_size"
        )

    def set_shapes(self, shape_file_path=None):
        head_dim = 512
        quant_block_size = 128
        block_size = 16
        token_sweep = (
            1,
            2,
            4,
            8,
            16,
            17,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        )
        self.shapes = [
            (
                num_tokens,
                max(1, (2 * num_tokens + block_size - 1) // block_size),
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in token_sweep
        ]
        block_size = 64
        self.shapes += [
            (
                num_tokens,
                max(1, (2 * num_tokens + block_size - 1) // block_size),
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in (8192, 32768, 65536)
        ]

    def get_input_iter(self, dtype):
        for (
            num_tokens,
            num_blocks,
            block_size,
            head_dim,
            quant_block_size,
        ) in self.shapes:
            k = torch.randn(
                num_tokens,
                head_dim,
                dtype=dtype,
                device=self.device,
            )
            slot_mapping = torch.randperm(
                num_blocks * block_size,
                device=self.device,
            )[
                :num_tokens
            ].to(torch.long)
            cache_stride = head_dim + head_dim * 4 // quant_block_size
            kv_cache = torch.empty(
                num_blocks,
                block_size,
                cache_stride,
                dtype=torch.uint8,
                device=self.device,
            )
            yield k, kv_cache, slot_mapping, quant_block_size, {"scale_fmt": "ue8m0"}


@pytest.mark.indexer_k_quant_and_cache
def test_indexer_k_quant_and_cache_benchmark():
    bench = IndexerKQuantAndCacheBenchmark(ref_indexer)
    bench.run()

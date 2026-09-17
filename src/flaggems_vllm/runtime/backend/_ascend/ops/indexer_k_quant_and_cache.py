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

# Adapted from vLLM v0.20.2:
# csrc/cache_kernels.cu::indexer_k_quant_and_cache_kernel

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _f32_to_fp8_e4m3fn(y):
    b = y.to(tl.int32, bitcast=True)
    a = b & 0x7FFFFFFF
    t = a - 0x3C000000
    t += 0x0007FFFF + ((t >> 20) & 1)
    r_norm = t >> 20
    r_sub = (a.to(tl.float32, bitcast=True) * 512.0 + 8388608.0).to(
        tl.int32, bitcast=True
    ) - 0x4B000000
    r = tl.where(a >= 0x3C800000, r_norm, r_sub)
    return (r | ((b >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_mapping_ptr,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_quant_blocks,
    head_dim: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    GRID_DIM1: tl.constexpr,
    PER_CORE_ROWS: tl.constexpr,
    LAST_CORE_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_len = LAST_CORE_ROWS if pid == tl.num_programs(0) - 1 else PER_CORE_ROWS
    for row_id in tl.range(row_len):
        grid_id = pid * PER_CORE_ROWS + row_id
        tid = grid_id // GRID_DIM1
        quant_block_id = (grid_id % GRID_DIM1) * 4
        quant_block_offsets = tl.arange(0, 4)
        head_offsets = tl.arange(0, QUANT_BLOCK_SIZE)
        offsets = (
            quant_block_id + quant_block_offsets[:, None]
        ) * QUANT_BLOCK_SIZE + head_offsets[None, :]
        mask = offsets < head_dim

        src_ptr = k_ptr + tid * head_dim
        slot_id = tl.load(slot_mapping_ptr + tid)
        if slot_id >= 0:
            block_id = slot_id // block_size
            block_offset = slot_id % block_size

            val = tl.load(src_ptr + offsets, mask=mask, other=0.0)
            amax = tl.max(tl.abs(val).to(tl.float32), axis=1)
            scale = tl.maximum(1e-4, amax) / 448.0

            if USE_UE8M0:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))

            fp8_val = _f32_to_fp8_e4m3fn(val.to(tl.float32) / scale[:, None])
            dst_ptr = kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
            tl.store(dst_ptr + offsets, fp8_val, mask=mask)

            dst_scale_ptr = (
                kv_cache_scale_ptr
                + block_id * kv_cache_scale_stride
                + block_offset * num_quant_blocks
                + quant_block_id
            )
            scale_mask = quant_block_id + quant_block_offsets < num_quant_blocks
            tl.store(dst_scale_ptr + quant_block_offsets, scale, mask=scale_mask)


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size

    kv_cache_flat = kv_cache.view(num_blocks, -1)
    # replace torch.float8_e4m3fn with torch.uint8 to avoid the following MLIRCompilationError:
    #     "unsupported datatype for arith::TruncFOp to hfusion"
    kv_cache_value = kv_cache_flat[:, : block_size * head_dim].view(torch.uint8)
    kv_cache_scale = kv_cache_flat[:, block_size * head_dim :].view(torch.float32)

    grid_dim1 = triton.cdiv(num_quant_blocks, 4)
    grid_size = num_tokens * grid_dim1
    vector_core_num = torch_npu.npu.get_device_properties(k.device.index).vector_core_num
    per_core_rows = (grid_size + vector_core_num - 1) // vector_core_num
    need_core_num = (grid_size + per_core_rows - 1) // per_core_rows
    last_core_rows = per_core_rows if grid_size % per_core_rows == 0 else grid_size % per_core_rows
    _indexer_k_quant_and_cache_kernel[(need_core_num,)](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_quant_blocks,
        head_dim,
        quant_block_size,
        USE_UE8M0=scale_fmt == "ue8m0",
        GRID_DIM1=grid_dim1,
        PER_CORE_ROWS=per_core_rows,
        LAST_CORE_ROWS=last_core_rows,
    )

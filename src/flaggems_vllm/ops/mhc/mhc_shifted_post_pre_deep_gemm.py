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

"""
Triton implementation of DSV4.1 shifted mHC kernel with reference to
https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

NUM_ROUTES = 4  # hc_mult
NUM_HC_OUTPUTS = NUM_ROUTES * (NUM_ROUTES + 2)
NUM_MAX_TOKENS = 1 << 20


def mhc_shifted_post_pre_deep_gemm(
    x: torch.Tensor,  # [T,H], BF16
    residual: torch.Tensor,  # [T,hc_mult,H], BF16
    shifted_prev_mix: torch.Tensor,  # [T,hc_mult], FP32
    post_mix: torch.Tensor,  # [T,hc_mult,1], FP32
    comb_res_mix: torch.Tensor,  # [T,hc_mult,hc_mult], FP32
    fn: torch.Tensor,  # [hc_mult*(hc_mult+2), hc_mult*H], FP32
    mix_scales: torch.Tensor,  # [3], FP32
    mix_bases: torch.Tensor,  # [hc_mult*(hc_mult+2)], FP32
    hc_norm_eps: float,
    hc_pre_eps: float,  
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_weight: torch.Tensor,  # [H], BF16
    rmsnorm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run DSV4.1 shifted post, next pre, and BF16 RMSNorm.

    Return:
        new_residual        [T,hc_mult,H], BF16
        new_post_mix        [T,hc_mult,1], FP32
        new_comb_res_mix    [T,hc_mult,hc_mult], FP32
        y_bf16              [T,H], BF16
        new_prev_mix        [T,hc_mult], FP32
    """
    logger.debug("GEMS MHC_SHIFTED_POST_PRE_DEEP_GEMM")

    T, H = x.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert T, NUM_ROUTES, H == residual.shape and residual.dtype == torch.bfloat16 and residual.is_contiguous()
    assert T, NUM_ROUTES == shifted_prev_mix.shape and shifted_prev_mix.dtype == torch.float and shifted_prev_mix.is_contiguous()
    assert T, NUM_ROUTES, 1 == post_mix.shape and post_mix.dtype == torch.float and post_mix.is_contiguous()
    assert T, NUM_ROUTES, NUM_ROUTES == comb_res_mix.shape and comb_res_mix.dtype == torch.float and comb_res_mix.is_contiguous()
    assert NUM_HC_OUTPUTS, NUM_ROUTES*H == fn.shape and fn.dtype == torch.float and fn.is_contiguous()
    assert 3 == mix_scales.shape and mix_scales.dtype == torch.float and mix_scales.is_contiguous()
    assert NUM_HC_OUTPUTS == mix_bases.shape and mix_bases.dtype == torch.float and mix_bases.is_contiguous()

    new_residual = torch.empty_like(residual)
    new_prev_mix = x.new_empty(num_tokens, hc_mult, 1, dtype=torch.float32)
    new_post_mix = torch.empty_like(post_mix)
    new_comb_res_mix = torch.empty_like(comb_res_mix)
    y_bf16 = x.new_empty(num_tokens, hidden_size, dtype=torch.bfloat16)
    return (
        new_residual,
        new_post_mix,
        new_comb_res_mix,
        y_bf16,
        new_prev_mix.squeeze(-1),
    )

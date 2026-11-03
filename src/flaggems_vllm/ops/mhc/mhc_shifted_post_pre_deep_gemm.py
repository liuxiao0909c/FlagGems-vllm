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

HC_MULT = 4
NUM_HC_OUTPUTS = HC_MULT * (HC_MULT + 2)
NUM_MAX_TOKENS = 1 << 20


NUM_WARPS_AUTOTUNE = [2, 4, 8, 16]
BH_LIST_AUTOTUNE = [1024, 512, 256]
BK_LIST_AUTOTUNE = [1024, 512, 256]
BT_LIST_AUTOTUNE = [1, 8]


#@triton.autotune(
#    configs=[
#        triton.Config({"BLOCK_SIZE_T": bt, "BLOCK_SIZE_H": bh}, num_warps=nw, num_stages=ns)
#        for bt in BT_LIST_AUTOTUNE
#        for bh in BH_LIST_AUTOTUNE
#        for nw in NUM_WARPS_AUTOTUNE
#        for ns in [1]
#    ],
#    key=["T_PAD", "H"],
#)
@triton.jit
def triton_mhc_shifted_post_pre_kernel(
    x_ptr,
    residual_ptr,
    shifted_prev_mix_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    num_sinkhorn_iters,
    rmsnorm_weight_ptr,
    rmsnorm_eps,
    new_residual_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    y_bf16_ptr,
    new_prev_mix_ptr,
    T,
    T_PAD: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    HC_MULT: tl.constexpr = 4
    
    pid_t, pid_h = tl.program_id(0).to(tl.int64), tl.program_id(1)
    t_offs = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    h_offs = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    hc_mult_offs = tl.arange(0, HC_MULT)
    norm_input_ptr = y_bf16_ptr  # both [T, H]
    
    x = tl.load(
        x_ptr + t_offs[:, None] * H + h_offs[None, :],
        mask=(t_offs[:, None] < T) & (h_offs[None, :] < H),
        other=0.0,
    ).to(tl.float32).expand_dims(1)  # [BLOCK_SIZE_T, 1, BLOCK_SIZE_H]
    post_mix = tl.load(
        post_mix_ptr + t_offs[:, None] * HC_MULT + hc_mult_offs[None, :],
        mask=t_offs[:, None] < T,
        other=0.0,
    )[:, :, None]  # [BLOCK_SIZE_T, HC_MULT, 1]
    new_residual = x * post_mix  # [BLOCK_SIZE_T, HC_MULT, BLOCK_SIZE_H]

    # 'tij,tih->tjh'
    for i in tl.static_range(0, HC_MULT):
        comb_res_mix = tl.load(
            comb_res_mix_ptr + t_offs[:, None, None] * HC_MULT * HC_MULT + i * HC_MULT + hc_mult_offs[None, None, :],
            mask=t_offs[:, None, None] < T,
            other=0.0,
        )  # [BLOCK_SIZE_T, 1, HC_MULT]
        residual = tl.load(
            residual_ptr + t_offs[:, None, None] * HC_MULT * H + i * H + h_offs[None, None, :],
            mask=(t_offs[:, None, None] < T) & (h_offs[None, None, :] < H),
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SIZE_T, 1, BLOCK_SIZE_H]
        new_residual += comb_res_mix.trans(0, 2, 1) * residual  # [BLOCK_SIZE_T, HC_MULT, BLOCK_SIZE_H]

    shifted_prev_mix = tl.load(
        shifted_prev_mix_ptr + t_offs[:, None] * HC_MULT + hc_mult_offs[None, :],
        mask=t_offs[:, None] < T,
        other=0.0,
    )[:, :, None]  # [BLOCK_SIZE_T, HC_MULT, 1]
    norm_input = tl.sum(new_residual.to(tl.bfloat16).to(tl.float32) * shifted_prev_mix, axis=1).to(tl.bfloat16)  # [BLOCK_SIZE_T, BLOCK_SIZE_H]
    tl.store(
       norm_input_ptr + t_offs[:, None] * H + h_offs[None, :],
       norm_input,
       mask=(t_offs[:, None] < T) & (h_offs[None, :] < H),
    )
    #if pid_t == 0:
    #    tl.device_print("BLOCK_SIZE_T:", BLOCK_SIZE_T)
    #    tl.device_print("new_residual:", new_residual.to(tl.bfloat16))
    #if pid_t == 0 and pid_h == 0:
    #    tl.device_print("new_residual:", new_residual)
    #    tl.device_print("new_residual_bf16:", new_residual.to(tl.bfloat16))
    tl.store(
       new_residual_ptr + t_offs[:, None, None] * HC_MULT * H + hc_mult_offs[None, :, None] * H + h_offs[None, None, :],
       new_residual.to(tl.bfloat16),
       mask=(t_offs[:, None, None] < T) & (h_offs[None, None, :] < H),
    )


@triton.jit
def _softmax(x, dim):
    max_val = tl.max(x, axis=dim, keep_dims=True)
    x_minus_max = x - max_val
    exp_x = tl.exp(x_minus_max)
    sum_exp = tl.sum(exp_x, axis=dim, keep_dims=True)
    return exp_x / sum_exp


@triton.jit
def _sinkhorn_reference(value, repeat, eps):
    value = _softmax(value, -1) + eps
    value = value / (tl.sum(value, axis=-2, keep_dims=True) + eps)
    for _ in range(repeat - 1):
        value = value / (tl.sum(value, axis=-1, keep_dims=True) + eps)
        value = value / (tl.sum(value, axis=-2, keep_dims=True) + eps)
    return value
   

#@triton.autotune(
#    configs=[
#        triton.Config({"BLOCK_SIZE_T": bt, "BLOCK_SIZE_K": bk}, num_warps=nw, num_stages=ns)
#        for bt in BT_LIST_AUTOTUNE
#        for bk in BK_LIST_AUTOTUNE
#        for nw in NUM_WARPS_AUTOTUNE
#        for ns in [1, 2, 3]
#    ],
#    key=["T_PAD", "H"],
#)
@triton.jit
def triton_mhc_shifted_post_pre_kernel2(
    x_ptr,
    residual_ptr,
    shifted_prev_mix_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    num_sinkhorn_iters,
    rmsnorm_weight_ptr,
    rmsnorm_eps,
    new_residual_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    y_bf16_ptr,
    new_prev_mix_ptr,
    T,
    T_PAD: tl.constexpr,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    HC_MULT: tl.constexpr = 4
    K: tl.constexpr = HC_MULT * H
    norm_input_ptr = y_bf16_ptr  # both [T, H]

    pid_t = tl.program_id(0).to(tl.int64)
    t_offs = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    fn0_offs = tl.arange(0, HC_MULT)
    fn1_offs = fn0_offs
    fn2_offs = tl.arange(0, HC_MULT * HC_MULT)

    mixed0 = tl.zeros([BLOCK_SIZE_T, HC_MULT], dtype=tl.float32)
    mixed1 = tl.zeros([BLOCK_SIZE_T, HC_MULT], dtype=tl.float32)
    mixed2 = tl.zeros([BLOCK_SIZE_T, HC_MULT * HC_MULT], dtype=tl.float32)
    cnt = (HC_MULT * H + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    hc_sqr_sum = tl.zeros([BLOCK_SIZE_T], dtype=tl.float32)
    for split_k in tl.range(0, cnt):
        k_offs = BLOCK_SIZE_K * split_k + tl.arange(0, BLOCK_SIZE_K)
        new_residual = tl.load(
            new_residual_ptr + t_offs[:, None] * K + k_offs[None, :],
            mask=(t_offs[:, None] < T) & (k_offs[None, :] < K),
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SIZE_T, BLOCK_SIZE_K]
        hc_sqr_sum += tl.sum(new_residual * new_residual, axis=1)  # [BLOCK_SIZE_T]

        fn0 = tl.load(
            fn_ptr + fn0_offs[:, None] * K + k_offs[None, :],
            mask=k_offs[None, :] < K,
            other=0.0,
        )  # [HC_MULT, BLOCK_SIZE_K]
        mixed0 = tl.dot(new_residual, tl.trans(fn0), acc=mixed0, input_precision="ieee")  # [BLOCK_SIZE_T, HC_MULT]
        #mixed0 = tl.dot(new_residual, tl.trans(fn0), acc=mixed0)  # [BLOCK_SIZE_T, HC_MULT]

        fn1 = tl.load(
            fn_ptr + (HC_MULT + fn1_offs[:, None]) * K + k_offs[None, :],
            mask=k_offs[None, :] < K,
            other=0.0,
        )  # [HC_MULT, BLOCK_SIZE_K]
        mixed1 = tl.dot(new_residual, tl.trans(fn1), acc=mixed1, input_precision="ieee")  # [BLOCK_SIZE_T, HC_MULT]
        #mixed1 = tl.dot(new_residual, tl.trans(fn1), acc=mixed1)  # [BLOCK_SIZE_T, HC_MULT]

        fn2 = tl.load(
            fn_ptr + (HC_MULT * 2 + fn2_offs[:, None]) * K + k_offs[None, :],
            mask=k_offs[None, :] < K,
            other=0.0,
        )  # [HC_MULT * HC_MULT, BLOCK_SIZE_K]
        mixed2 = tl.dot(new_residual, tl.trans(fn2), acc=mixed2, input_precision="ieee")  # [BLOCK_SIZE_T, HC_MULT * HC_MULT]
        #mixed2 = tl.dot(new_residual, tl.trans(fn2), acc=mixed2)  # [BLOCK_SIZE_T, HC_MULT * HC_MULT]

    rsqrt = tl.rsqrt(hc_sqr_sum / (HC_MULT * H) + hc_norm_eps)
    mixed0 *= rsqrt[:, None]
    mixed1 *= rsqrt[:, None]
    mixed2 *= rsqrt[:, None]
    mix_scale0 = tl.load(mix_scales_ptr)
    mix_base0 = tl.load(mix_bases_ptr + fn0_offs)  # [HC_MULT]
    mixed0 = mixed0 * mix_scale0 + mix_base0[None, :]  # [BLOCK_SIZE_T, HC_MULT]
    mix_scale1 = tl.load(mix_scales_ptr + 1)
    mix_base1 = tl.load(mix_bases_ptr + HC_MULT + fn1_offs)  # [HC_MULT]
    mixed1 = mixed1 * mix_scale1 + mix_base1[None, :]  # [BLOCK_SIZE_T, HC_MULT]
    mix_scale2 = tl.load(mix_scales_ptr + 2)
    mix_base2 = tl.load(mix_bases_ptr + HC_MULT * 2 + fn2_offs)  # [HC_MULT * HC_MULT]
    mixed2 = mixed2 * mix_scale2 + mix_base2[None, :]  # [BLOCK_SIZE_T, HC_MULT * HC_MULT]
     
    new_prev_mix = tl.sigmoid(mixed0) + hc_pre_eps  # [BLOCK_SIZE_T, HC_MULT]
    tl.store(new_prev_mix_ptr + t_offs[:, None] * HC_MULT + fn0_offs[None, :], new_prev_mix, mask=t_offs[:, None] < T)
    new_post_mix = tl.sigmoid(mixed1) * hc_post_scale  # [BLOCK_SIZE_T, HC_MULT]
    tl.store(new_post_mix_ptr + t_offs[:, None] * HC_MULT + fn1_offs[None, :], new_post_mix, mask=t_offs[:, None] < T)
    new_comb_res_mix = _sinkhorn_reference(mixed2.reshape(BLOCK_SIZE_T, HC_MULT, HC_MULT), num_sinkhorn_iters, sinkhorn_eps)  # [BLOCK_SIZE_T, HC_MULT, HC_MULT]
    tl.store(
        new_comb_res_mix_ptr + t_offs[:, None] * HC_MULT * HC_MULT + fn2_offs[None, :],
        new_comb_res_mix.reshape(BLOCK_SIZE_T, HC_MULT * HC_MULT),
        mask=t_offs[:, None] < T,
    )
    h_offs = tl.arange(0, H_PAD)
    norm_input = tl.load(
        norm_input_ptr + t_offs[:, None] * H + h_offs[None, :],
        mask=(t_offs[:, None] < T) & (h_offs < H),
        other=0.0,
    ).to(tl.float32)  # [BLOCK_SIZE_T, H_PAD]
    rmsnorm_weight = tl.load(
        rmsnorm_weight_ptr + h_offs,
        mask=h_offs < H,
        other=0.0,
    ).to(tl.float32)  # [H_PAD]
    y = norm_input * tl.rsqrt(tl.sum(norm_input * norm_input, axis=1) / H + rmsnorm_eps)[:, None] * rmsnorm_weight[None, :]
    tl.store(
        y_bf16_ptr + t_offs[:, None] * H + h_offs[None, :],
        y.to(tl.bfloat16),
        mask=(t_offs[:, None] < T) & (h_offs < H),
    )
 

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
    assert (H > 0) & (H % (64 * 16) == 0) 
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert (T, HC_MULT, H) == residual.shape and residual.dtype == torch.bfloat16 and residual.is_contiguous()
    assert (T, HC_MULT) == shifted_prev_mix.shape and shifted_prev_mix.dtype == torch.float and shifted_prev_mix.is_contiguous()
    assert (T, HC_MULT, 1) == post_mix.shape and post_mix.dtype == torch.float and post_mix.is_contiguous()
    assert (T, HC_MULT, HC_MULT) == comb_res_mix.shape and comb_res_mix.dtype == torch.float and comb_res_mix.is_contiguous()
    assert (NUM_HC_OUTPUTS, HC_MULT*H) == fn.shape and fn.dtype == torch.float and fn.is_contiguous()
    assert (3,) == mix_scales.shape and mix_scales.dtype == torch.float and mix_scales.is_contiguous()
    assert (NUM_HC_OUTPUTS,) == mix_bases.shape and mix_bases.dtype == torch.float and mix_bases.is_contiguous()

    new_residual = torch.empty_like(residual)
    new_prev_mix = x.new_empty(T, HC_MULT, 1, dtype=torch.float32)
    new_post_mix = torch.empty_like(post_mix)
    new_comb_res_mix = torch.empty_like(comb_res_mix)
    y_bf16 = x.new_empty(T, H, dtype=torch.bfloat16)
    if T > 0:
        #def grid1(meta):
        #    return (triton.cdiv(T, meta["BLOCK_SIZE_T"]), triton.cdiv(T, meta["BLOCK_SIZE_H"]))
        #triton_mhc_shifted_post_pre_kernel[grid1](
        triton_mhc_shifted_post_pre_kernel[(T, H // 1024)](
            x,
            residual,
            shifted_prev_mix,
            post_mix,
            comb_res_mix,
            fn,
            mix_scales,
            mix_bases,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
            num_sinkhorn_iters,
            rmsnorm_weight,
            rmsnorm_eps,
            new_residual,
            new_post_mix,
            new_comb_res_mix,
            y_bf16,
            new_prev_mix,
            T,
            T_PAD=triton.next_power_of_2(T),
            H=H,
            BLOCK_SIZE_T=1,
            BLOCK_SIZE_H=1024,
            num_warps=4,
            num_stages=1,
        )
        #def grid2(meta):
        #    return (triton.cdiv(T, meta["BLOCK_SIZE_T"]),)
        #triton_mhc_shifted_post_pre_kernel2[grid2](
        triton_mhc_shifted_post_pre_kernel2[(T,)](
            x,
            residual,
            shifted_prev_mix,
            post_mix,
            comb_res_mix,
            fn,
            mix_scales,
            mix_bases,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
            num_sinkhorn_iters,
            rmsnorm_weight,
            rmsnorm_eps,
            new_residual,
            new_post_mix,
            new_comb_res_mix,
            y_bf16,
            new_prev_mix,
            T,
            T_PAD=triton.next_power_of_2(T),
            H=H,
            H_PAD=triton.next_power_of_2(H),
            BLOCK_SIZE_T=1,
            BLOCK_SIZE_K=1024,
            num_warps=4,
            num_stages=2,
        )
    return (
        new_residual,
        new_post_mix,
        new_comb_res_mix,
        y_bf16,
        new_prev_mix.squeeze(-1),
    )

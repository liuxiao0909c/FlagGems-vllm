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

import importlib

import torch

OP_NAME = "mhc_shifted_post_pre_deep_gemm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False


try:
    from vllm.models.deepseek_v41.nvidia.ops.mega_mhc import (
        is_mega_mhc_supported,
        mhc_shifted_post_pre_deep_gemm as vllm_impl_deepgemm,
    )
except ImportError:
    is_mega_mhc_supported = None
    vllm_impl_deepgemm = None


try:
    import vllm._custom_ops  # noqa: F401
    if hasattr(torch.ops.vllm, "mhc_fused_post_pre_delayed_tilelang"):
        vllm_impl_tilelang = torch.ops.vllm.mhc_fused_post_pre_delayed_tilelang
    else:
        vllm_impl_tilelang = None
except ImportError:
    vllm_impl_tilelang = None


def ref_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    shifted_prev_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_weight: torch.Tensor,
    rmsnorm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    impl = None
    if is_mega_mhc_supported is not None and vllm_impl_deepgemm is not None:
        _, hc_mult, hidden_size = residual.shape
        if is_mega_mhc_supported(hidden_size, hc_mult):
            impl = vllm_impl_deepgemm
    if impl is None and vllm_impl_tilelang is not None:
        return vllm_impl_tilelang(
            x,
            residual,
            post_mix,
            comb_res_mix,
            fn,
            mix_scales,
            mix_bases,
            hc_norm_eps,
            hc_pre_eps,
            sinkhorn_eps,
            hc_post_scale,
            num_sinkhorn_iters,
            pre_mix=shifted_prev_mix,
            norm_weight=rmsnorm_weight,
            norm_eps=rmsnorm_eps,
            capture_aux=False,
        )[:5]
     assert impl is not None
     return impl(
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
    )


def native():
    if is_mega_mhc_supported is not None and vllm_impl_deepgemm is not None:
        if is_mega_mhc_supported(1024, 4):
            return ref_impl
    if vllm_impl_tilelang is not None:
        return ref_impl
    return None


def grid():
    # num_tokens, hidden_state 
    return [
        (num_tokens, hidden_size)
        for num_tokens in [1, 17, 65, 200, 1025, 4096, 32768]
        for hidden_size in [4096, 5120, 7168]
    ]


def build_inputs(binding, dtype, device):
    num_tokens, hidden_size = binding
    hc_mult = 4
    num_hc_outputs = hc_mult * (hc_mult + 2)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=dtype, device=device)
    shifted_prev_mix = torch.rand(num_tokens, hc_mult, device=device)
    post_mix = torch.rand(num_tokens, hc_mult, 1, device=device)
    comb_res_mix = torch.rand(num_tokens, hc_mult, hc_mult, device=device)
    fn = torch.randn(num_hc_outputs, hc_mult * hidden_size, device=device) * 0.01
    mix_scales = torch.randn(3, device=device) * 0.1
    mix_bases = torch.randn(num_hc_outputs, device=device) * 0.1
    rmsnorm_weight = torch.empty(hidden_size, dtype=dtype, device=device).uniform_(0.9, 1.1)
    args = (
        x,
        residual,
        shifted_prev_mix,
        post_mix,
        comb_res_mix,
        fn,
        mix_scales,
        mix_bases,
        2e-5, # hc_norm_eps
        3e-4, # hc_pre_eps
        1.25, # hc_post_scales
        2e-6, # sinkhorn_eps
        10, # num_sinkhorn_iters
        rmsnorm_weight,
        7e-6, # rmsnorm_eps
    )
    return args, {}


def key_shape(binding):
    num_tokens, hidden_size = binding
    return [num_tokenshidden_size]


def config(binding, dtype):
    num_tokens, num_experts, n_group, topk_group, topk, renormalize, scoring_func = (
        binding
    )
    dtype_name = str(dtype)
    return {
        "inputs": {
            "scores": {"shape": [num_tokens, num_experts], "dtype": dtype_name},
            "n_group": {"scalar": n_group},
            "topk_group": {"scalar": topk_group},
            "topk": {"scalar": topk},
            "renormalize": {"scalar": renormalize},
            "routed_scaling_factor": {"scalar": 1.0},
            "bias": {"shape": [num_experts], "dtype": "torch.float32"},
            "scoring_func": {"scalar": scoring_func},
        },
        "outputs": {
            "topk_weights": {"shape": [num_experts, topk], "dtype": "torch.float32"},
            "topk_ids": {"shape": [num_experts, topk], "dtype": "torch.int32"},
        },
        "dims": {
            "T": num_tokens,
            "H": hidden_size,
        },
    }


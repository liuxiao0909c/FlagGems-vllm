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

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    NUM_TOKENS_LIST_VLLM = [0, 1, 1024]
    HIDDEN_SIZE_LIST_VLLM = [7168]
    NUM_TOKENS_LIST_DEEPGEMM = [1, 64]
    HIDDEN_SIZE_LIST_DEEPGEMM = [7168]
else:
    NUM_TOKENS_LIST_VLLM = [0, 1, 17, 128, 1024]
    HIDDEN_SIZE_LIST_VLLM = [5120, 7168]
    NUM_TOKENS_LIST_DEEPGEMM = [1, 64, 65, 200, 1025, 4096, 32768]
    HIDDEN_SIZE_LIST_DEEPGEMM = [4096, 7168]

DEVICE = flaggems_vllm.device


def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


# Reference to https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_mega_mhc.py
def torch_impl(
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
    hc_mult = residual.shape[1]
    new_residual = x.float().unsqueeze(1) * post_mix
    new_residual += torch.einsum(
        'tij,tih->tjh', comb_res_mix, residual.float())
    new_residual = new_residual.bfloat16()
    if shifted_prev_mix is None:
        norm_input = None
    else:
        shifted_prev_mix = shifted_prev_mix.unsqueeze(-1)
        norm_input = (new_residual.float() * shifted_prev_mix).sum(dim=1).bfloat16()

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mixes = new_residual.flatten(1).float() @ fn.mT
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    hc_sqr_sum = new_residual.float().square().sum((1, 2))
    mixes *= torch.rsqrt(
        hc_sqr_sum / (hc_mult * x.size(1)) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((
        mix_scales[0].expand(hc_mult),
        mix_scales[1].expand(hc_mult),
        mix_scales[2].expand(hc_mult * hc_mult),
    ))
    mixes = mixes * scales + mix_bases
    new_prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix = (
        mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale
    ).unsqueeze(2)
    new_comb_res_mix = _sinkhorn_reference(
        mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult),
        num_sinkhorn_iters,
        sinkhorn_eps,
    )
    if norm_input is None:
        norm_input = (new_residual.float() * new_prev_mix).sum(dim=1).bfloat16()

    norm_input_float = norm_input.float()
    y_bf16 = (
        norm_input_float
        * torch.rsqrt(norm_input_float.square().mean(1) + rmsnorm_eps).unsqueeze(1)
        * (rmsnorm_weight.float())
    ).bfloat16()
    return (
        new_residual,
        new_post_mix,
        new_comb_res_mix,
        y_bf16,
        new_prev_mix.squeeze(-1),
    )


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

    impl = impl if impl is not None else torch_impl
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


@pytest.mark.mhc_shifted_post_pre_deep_gemm
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST_VLLM)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE_LIST_VLLM)
def test_vllm_case(num_tokens, hidden_size):
    utils.init_seed(42)
    hc_mult = 4
    num_hc_outputs = hc_mult * (hc_mult + 2)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=DEVICE)
    shifted_prev_mix = torch.rand(num_tokens, hc_mult, device=DEVICE)
    post_mix = torch.rand(num_tokens, hc_mult, 1, device=DEVICE)
    comb_res_mix = torch.rand(num_tokens, hc_mult, hc_mult, device=DEVICE)
    fn = torch.randn(num_hc_outputs, hc_mult * hidden_size, device=DEVICE) * 0.01
    mix_scales = torch.randn(3, device=DEVICE) * 0.1
    mix_bases = torch.randn(num_hc_outputs, device=DEVICE) * 0.1
    rmsnorm_weight = torch.empty(hidden_size, dtype=torch.bfloat16, device=DEVICE).uniform_(0.9, 1.1)
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
    #import pdb; pdb.set_trace()
    expected = ref_impl(*args)
    #result = flaggems_vllm.mhc_shifted_post_pre_deep_gemm(*args)
    result = torch_impl(*args)
    for i, (res, ref) in enumerate(zip(result, expected, strict=True)):
        atol, rtol = (1.6e-2, 1e-2) if i in (0, 3) else (1e-6, 1e-3)
        torch.testing.assert_close(res, ref, atol=atol, rtol=rtol)


@pytest.mark.mhc_shifted_post_pre_deep_gemm
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST_DEEPGEMM)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE_LIST_DEEPGEMM)
def test_deepgemm_case(num_tokens, hidden_size):
    utils.init_seed(42)
    hc_mult = 4
    num_hc_outputs = hc_mult * (hc_mult + 2)
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16, device=DEVICE)
    post_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float, device=DEVICE).sigmoid()
    comb_res_mix = torch.rand((num_tokens, hc_mult, hc_mult), device=DEVICE)
    shifted_prev_mix = torch.randn((num_tokens, hc_mult), dtype=torch.float, device=DEVICE).sigmoid()
    fn = torch.randn((num_hc_outputs, hc_mult * hidden_size), dtype=torch.float, device=DEVICE).mul_(1e-2)
    mix_scales = torch.randn((3,), dtype=torch.float, device=DEVICE).mul_(0.1)
    mix_bases = torch.randn((num_hc_outputs,), dtype=torch.float, device=DEVICE).mul_(0.1)
    rmsnorm_weight = torch.randn((hidden_size,), dtype=torch.float, device=DEVICE).mul_(0.1).add_(1).bfloat16()

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
    #import pdb; pdb.set_trace()
    expected = ref_impl(*args)
    #result = flaggems_vllm.mhc_shifted_post_pre_deep_gemm(*args)
    result = torch_impl(*args)
    for i, (res, ref) in enumerate(zip(result, expected, strict=True)):
        atol, rtol = (1.6e-2, 1e-2) if i in (0, 3) else (1e-6, 1e-3)
        torch.testing.assert_close(res, ref, atol=atol, rtol=rtol)

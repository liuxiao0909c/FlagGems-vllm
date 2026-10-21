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
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm import fused_kda_gate as gems_fused_kda_gate

from . import accuracy_utils as utils
from . import conftest as cfg


def torch_fused_kda_gate(
    g: torch.Tensor,  # [..., H*D]
    A_log: torch.Tensor,  # [H]
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,  # [H*D]
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
) -> torch.Tensor:  # [..., H, D]
    orig_shape = g.shape[:-1]
    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A_log.numel()
    assert H * head_k_dim == HD
    assert g.stride() == (HD, 1)

    g = g.float()
    if g_bias is not None:
        assert g_bias.numel() == HD and g_bias.is_contiguous()
        g += g_bias.float().view(HD)
    A = torch.exp(A_log.float())
    if lower_bound is not None:
        y = lower_bound * torch.sigmoid(A.view(1, H, 1) * g.view(T, H, head_k_dim))
    else:
        g_scaled = g * beta
        use_linear = g_scaled > threshold
        sp = torch.where(use_linear, g, (1.0 / beta) * torch.log(1.0 + torch.exp(g_scaled)))
        y = - A.view(1, H, 1) * sp.view(T, H, head_k_dim)
    return y.view(*orig_shape, H, head_k_dim)


try:
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_kda_gate as ref_fused_kda_gate
except (ImportError, AttributeError):
    ref_fused_kda_gate = torch_fused_kda_gate


if cfg.QUICK_MODE:
    NUM_SEQS = [1, 8, 64]
    H = [8]
    D = [64]
    LOWER_BOUND = [None]
    DTYPES = [torch.bfloat16]
else:
    NUM_SEQS = [1, 8, 32, 64, 300]
    H = [8, 12, 96]
    D = [64, 128]
    LOWER_BOUND = [-5.0, None]
    DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.fused_kda_gate
@pytest.mark.parametrize("num_seqs", NUM_SEQS)
@pytest.mark.parametrize("H", H)
@pytest.mark.parametrize("D", D)
@pytest.mark.parametrize("lower_bound", LOWER_BOUND)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_kda_gate(num_seqs, H, D, lower_bound, dtype):
    utils.init_seed(42)
    T = num_seqs
    raw_g = torch.randn(1, T, H, D, dtype=dtype, device=flaggems_vllm.device)
    g = raw_g.contiguous().view(T, H * D)
    A_log = torch.randn(H, dtype=torch.float32, device=flaggems_vllm.device) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=flaggems_vllm.device) * 0.1
    ref_out = ref_fused_kda_gate(
        g, A_log, D, g_bias=dt_bias, lower_bound=lower_bound,
    )
    res_out = gems_fused_kda_gate(
        g, A_log, D, g_bias=dt_bias, lower_bound=lower_bound,
    )
    ref_out = utils.to_reference(ref_out)
    res_out = utils.to_reference(res_out)
    utils.gems_assert_close(res_out, ref_out, torch.float32)

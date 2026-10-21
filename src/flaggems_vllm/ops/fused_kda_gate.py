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
"""fused_kda_gate kernel token from https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py"""

import os

import logging
import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import device

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    import triton.language.extra.libdevice as tldevice
    log = tldevice.fast_logf
else:
    log = tl.log


logger = logging.getLogger(__name__)
is_amd = device.vendor_name == "hygon"
BT_LIST_AUTOTUNE = [32, 64, 128]
#BD_LIST_AUTOTUNE = [8, 16, 32, 64, 128]
#BD_LIST_AUTOTUNE = [1, 2, 4]
BD_LIST_AUTOTUNE = [8, 16, 32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]

#BT_LIST_AUTOTUNE = [128]
#BD_LIST_AUTOTUNE = [64]
#NUM_WARPS_AUTOTUNE = [8]
#NUM_STAGES_AUTOTUNE = [3]

@triton.autotune(
    configs=[
        triton.Config({"BT": bt, "BD": bd}, num_warps=nw, num_stages=1)
        for bt in BT_LIST_AUTOTUNE
        for bd in BD_LIST_AUTOTUNE
        for nw in NUM_WARPS_AUTOTUNE
    ],
    key=["T_PAD", "H", "D", "USE_LOWER_BOUND"],
)
@triton.jit
def gems_kda_gate_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    lower_bound,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    T_PAD: tl.constexpr,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2)
    n_t = i_t * BT
    o_t = n_t + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t = o_t < T
    m_g = m_t[:, None] & (o_d[None, :] < D)
    g_ptr = g + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]
    y_ptr = y + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]

    b_g = tl.load(g_ptr, mask=m_g, other=0.0).to(tl.float32)

    if HAS_BIAS:
        n_d = i_d * BD + tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias[None, :]

    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_y = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        g_scaled = b_g * beta
        use_linear = g_scaled > threshold
        sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))
        b_y = -b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), mask=m_g)


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """
    Forward pass for KDA gate:
      input g: [..., H*D]
      param A: [H] or [1, 1, H, 1]
      beta: softplus beta parameter
      threshold: softplus threshold parameter
      return  : [..., H, D]
    """
    logger.debug("GEMS FUSED_KDA_GATE")
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD
    assert g.stride() == (HD, 1)

    y = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), H, triton.cdiv(head_k_dim, meta["BD"]))

    gems_kda_gate_fwd_kernel[grid](
        g,
        A,
        y,
        g_bias,
        lower_bound or 0.0,
        beta,
        threshold,
        T,
        triton.next_power_of_2(T),
        H,
        head_k_dim,
        #BD=triton.next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
        USE_LOWER_BOUND=lower_bound is not None,
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y

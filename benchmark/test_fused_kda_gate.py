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

from . import base

try:
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_kda_gate as vllm_fused_kda_gate

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


class FusedKdaGateBenchmark(base.Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        configs = [
            (T, H, D, lower_bound)
            for T in [1, 8, 64, 300]
            #for T in [1, 8, 64, 1024, 4096, 32768, 65536]
            for H in [8, 96]
            #for H in [8]
            for D in [64, 128]
            #for D in [128]
            for lower_bound in [-5.0, None]
            #for lower_bound in [-5.0]
        ]
        self.shapes = configs

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from self.make_input_fn(config, dtype, self.device)

    def make_input_fn(self, config, dtype, device):
        torch.manual_seed(42)
        T, H, D, lower_bound = config
        raw_g = torch.randn(1, T, H, D, dtype=dtype, device=flaggems_vllm.device)
        g = raw_g.contiguous().view(T, H * D)
        A_log = torch.randn(H, dtype=torch.float32, device=flaggems_vllm.device) * 0.5
        dt_bias = torch.randn(H, D, dtype=torch.float32, device=flaggems_vllm.device) * 0.1
        beta = 1.0
        threshold = 20.0
        yield (g, A_log, D, dt_bias, beta, threshold, lower_bound)


@pytest.mark.fused_kda_gate
@pytest.mark.skipif(
    not HAS_VLLM,
    reason="requires vLLM to be installed for reference comparison",
)
def test_fused_kda_gate():
    bench = FusedKdaGateBenchmark(
        op_name="fused_kda_gate",
        torch_op=vllm_fused_kda_gate,
        #dtypes=[torch.bfloat16, torch.float16],
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_fused_kda_gate)
    bench.run()

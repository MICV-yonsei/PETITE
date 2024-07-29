# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from https://docs.monai.io/en/stable/_modules/monai/networks/blocks/selfattention.html
#
# 2023-10-31
# Modified to place LoRA instead of Linear Project in Attention
# nn.Linear -> lora.Linear

from __future__ import annotations
from monai.utils import optional_import

import torch
import torch.nn as nn
import loralib as lora

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class Lora_SABlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        r: int = 4,
        lora_alpha: int = 1,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.q = lora.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, r=r, lora_alpha=lora_alpha, merge_weights=True)
        self.k = lora.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, r=r, lora_alpha=lora_alpha, merge_weights=True)
        self.v = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        
        self.qkv = lora.MergedLinear(hidden_size, 3*hidden_size, bias=qkv_bias, r=r, lora_alpha=lora_alpha, enable_lora=[True, True, False])
        
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()


    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        if self.save_attn:
            # no gradients and new tensor;
            # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x

def init_ssf_scale_shift(dim_out):
        scale = nn.Parameter(torch.ones(dim_out))
        shift = nn.Parameter(torch.zeros(dim_out))
        nn.init.normal_(scale, mean=1, std=.02)
        nn.init.normal_(shift, std=.02)

        return scale, shift

def ssf_ada(x, scale, shift):
        assert scale.shape == shift.shape
        if x.shape[-1] == scale.shape[0]: 
            return x * scale + shift
        elif x.shape[1] == scale.shape[0]:
            return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
        else:
            raise ValueError('the input tensor shape does not match the shape of the scale factor.')
     
class SSF_SABlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")
        
        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_size * 3)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(hidden_size)

    def forward(self, x):
        output = self.input_rearrange(ssf_ada(self.qkv(x), self.ssf_scale_1, self.ssf_shift_1))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        if self.save_attn:
            # no gradients and new tensor;
            # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        x = self.drop_output(x)
        return x
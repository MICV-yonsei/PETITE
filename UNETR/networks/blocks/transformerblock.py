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
# https://docs.monai.io/en/stable/_modules/monai/networks/blocks/transformerblock.html

from __future__ import annotations
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_, _assert

import torch.nn as nn
import torch.nn as nn
import torch
import torch.nn.functional as F

from networks.blocks.mlp import SSF_MLPBlock
from networks.blocks.selfattention import Lora_SABlock, SSF_SABlock

class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Adapted_TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        rf: int = 1,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.adapter_downsample = nn.Linear(hidden_size, int(hidden_size/rf))
        self.adapter_upsample = nn.Linear(int(hidden_size/rf), hidden_size)
        self.adapter_act_fn = nn.GELU ()
        
        nn.init.zeros_(self.adapter_downsample.weight)
        nn.init.zeros_(self.adapter_downsample.bias)
        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)  
    
    def forward(self, x):
        h = x
        x = h + self.attn(self.norm1(x))

        h = x
        x = self.mlp(self.norm2(x))
        adpt = self.adapter_downsample(x)
        adpt = self.adapter_act_fn(adpt)
        adpt = self.adapter_upsample(adpt)
        x = adpt + x

        x = x + h
        return x

class Lora_TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
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
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Lora_SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn, r=r, lora_alpha=lora_alpha)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
   
class SSF_TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        self.mlp = SSF_MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)        
        self.attn = SSF_SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_size)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(hidden_size)


    def forward(self, x):
        x = x + self.attn(ssf_ada(self.norm1(x), self.ssf_scale_1, self.ssf_shift_1))
        x = x + self.mlp(ssf_ada(self.norm2(x), self.ssf_scale_1, self.ssf_shift_1))            
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
     
     
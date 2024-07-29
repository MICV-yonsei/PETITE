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

from __future__ import annotations
from monai.networks.layers import get_act_layer
from monai.utils import look_up_option

import torch 
import torch.nn as nn

SUPPORTED_DROPOUT_MODE = {"vit", "swin"}

class SSF_MLPBlock(nn.Module):
    def __init__(
        self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act: tuple | str = "GELU", dropout_mode="vit"
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim) if act != "GEGLU" else nn.Linear(hidden_size, mlp_dim * 2)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = get_act_layer(act)
        self.drop1 = nn.Dropout(dropout_rate)
        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(mlp_dim)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(hidden_size)

    def forward(self, x):
        x = self.fn(ssf_ada(self.linear1(x), self.ssf_scale_1, self.ssf_shift_1))
        x = self.drop1(x)
        x = self.linear2(x)
        x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        x = self.drop2(x)
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
 
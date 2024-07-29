
from __future__ import annotations
from collections.abc import Sequence
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class De_Adapted_Block(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        tune_mode :bool =False,
        rf : int =1,
    ):
        super().__init__()
        
        # self.resize_conv = nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.tune_mode=tune_mode
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

        self.adapter_downsample = nn.Linear(out_channels, int(out_channels/rf))
        self.adapter_upsample = nn.Linear(int(out_channels/rf), out_channels)
        self.adapter_act_fn = nn.GELU()
        
        nn.init.zeros_(self.adapter_downsample.weight)
        nn.init.zeros_(self.adapter_downsample.bias)
        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)    
            
    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        adpt = self.adapter_downsample(out)
        adpt = self.adapter_act_fn(adpt)
        adpt = self.adapter_upsample(adpt)
        out = adpt + out
        out = self.lrelu(out)
        return out
    

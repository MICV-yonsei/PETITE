from __future__ import annotations

from collections.abc import Sequence
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

import numpy as np
import torch
import torch.nn as nn

from networks.blocks.selfattention import Lora_SABlock, SSF_SABlock
import loralib as lora
class De_SSF_Block(nn.Module):    

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
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
        
        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(out_channels)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_channels)
            
            
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = ssf_ada(out, self.ssf_scale_1, self.ssf_shift_1)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = ssf_ada(out, self.ssf_scale_2, self.ssf_shift_2)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual) 
        out += residual
        out = self.lrelu(out)
        return out


def init_ssf_scale_shift(out_channels):
        scale = nn.Parameter(torch.ones(out_channels))
        shift = nn.Parameter(torch.zeros(out_channels))
        nn.init.normal_(scale, mean=1, std=.02)
        nn.init.normal_(shift, std=.02)
        return scale, shift

def ssf_ada(out, scale, shift):
    assert scale.shape == shift.shape
    if out.shape[-1] == scale.shape[0]:      
        return out * scale + shift
    elif out.shape[1] == scale.shape[0]:
        return out * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')
    



class De_Adpt_Block(nn.Module):    

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        rf: int = 1,
    ):
        super().__init__()
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
        
        # self.adapter_downsample = nn.Conv3d(out_channels, int(out_channels/rf), kernel_size)
        # self.adapter_upsample = nn.Conv3d(int(out_channels/rf), out_channels, kernel_size)
        # self.adapter_act_fn = nn.GELU ()
        
        # self.adapter_downsample = nn.Linear(in_channels, int(out_channels/rf))
        # self.adapter_upsample = nn.Linear(int(in_channels/rf), in_channels)
        # self.adapter_act_fn = nn.GELU ()
        
        self.adapter_downsample = nn.Linear(out_channels, int(out_channels/rf))
        self.adapter_upsample = nn.Linear(int(out_channels/rf), out_channels)
        self.adapter_act_fn = nn.GELU ()
        
        nn.init.zeros_(self.adapter_downsample.weight)
        nn.init.zeros_(self.adapter_downsample.bias)
        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)   
        
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            ) 
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)         
             
    def forward(self, inp):
        residual = inp 
        out = self.conv1(inp)
        out = self.norm1(out) 
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        residual = inp 
        adpt = self.adapter_downsample(inp)
        adpt = self.adapter_act_fn(adpt)
        adpt = self.adapter_upsample(out)
        out = adpt + out
        
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out

class De_LoRA_Block(nn.Module):    

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        qkv_bias: bool = False,
        dropout: tuple | str | float | None = None,
        r: int = 4,
        lora_alpha: int = 1,
    ):
        super().__init__()
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
        self.qkv = lora.MergedLinear(out_channels, 3*out_channels, bias=qkv_bias, r=r, lora_alpha=lora_alpha, enable_lora=[True, True, False])
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
                
    def forward(self, inp):
        residual = inp 
        out = self.conv1(inp)
        out = self.norm1(out) 
        out = self.qkv(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out
    
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from networks.ViT import Prompted_ViT, Adapted_ViT, Lora_ViT, SSF_ViT
from networks.blocks.UnetResBlock import De_SSF_Block, De_Adpt_Block, De_LoRA_Block

class En_Prompted_SSF_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_SSF_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            
        else :    
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out


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
        
        

class En_Prompted_De_LoRA_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_LoRA_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            
        else :    
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out
    
class En_Prompted_De_Adpt_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
        rf:int = 1,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_Adpt_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                rf=rf,
                norm_name=norm_name,
            )
        else :      
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out


class En_Adapted_De_SSF_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_SSF_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            
        else :    
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out


class En_Adapted_De_SSF_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_SSF_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            
        else :    
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out

class En_LoRA_De_Adpt_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
        r: int = 4, 
        lora_alpha: int = 1,
        rf=1,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_Adpt_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            
        else :    
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out


class En_Adapted_De_SSF_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_SSF_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            
        else :    
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out


class En_SSF_De_Adpt_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
        rf=1,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_Adpt_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            
        else :    
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out
    
class En_LoRA_De_SSF_Block(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = De_SSF_Block(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            
        else :    
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        #out torch.Size([24, 32, 96, 96, 96])
        #inp torch.Size([24, 64, 32, 32, 32])
        # breakpoint()
        out = torch.cat((out, skip), dim=1)
        # breakpoint()
        out = self.conv_block(out)
        return out
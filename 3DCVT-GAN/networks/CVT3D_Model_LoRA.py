# 2023-10-31
# Modified to place Lora instead of Convolutional Projection in Attention
# nn.Conv3d -> lora.Conv3d

from functools import partial
from itertools import repeat
#from torch._six import container_abcs

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import SimpleITK as sitk
from timm.models.layers import DropPath, trunc_normal_
from thop import profile
#from .registry import register_model
import loralib as lora

# From PyTorch internals
'''
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
'''

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=2,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False,
                 rank=0,
                 lora_alpha=1,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token
        
        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method, rank=rank, lora_alpha=lora_alpha
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method, rank=rank, lora_alpha=lora_alpha
        )
        self.conv_proj_v = self._build_projection_conv_v(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)


    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method,
                          rank,
                          lora_alpha):
        if method == 'dw_bn':
            """Use LoRA: lora.Conv3d instead of nn.Conv3d"""
            # print(f"Rank {rank} / Alpha {lora_alpha}")
            proj = nn.Sequential(OrderedDict([
                ('conv', lora.Conv3d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in,
                    r=rank,
                    merge_weights=True,
                    lora_alpha=lora_alpha,
                )),
                ('bn', nn.BatchNorm3d(dim_in)),
                ('rearrage', Rearrange('b c h w d -> b (h w d) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool3d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w d-> b (h w d) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def _build_projection_conv_v(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv3d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm3d(dim_in)),
                ('rearrage', Rearrange('b c h w d -> b (h w d) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool3d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w d-> b (h w d) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj
    

    def forward_conv(self, x, h, w, d):
        #print('x:',x.shape)
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w,d=d)
        #print('x:',x.shape)
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w d -> b (h w d) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w d -> b (h w d) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w d -> b (h w d) c')

        return q, k, v

    def forward(self, x, h, w,d):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w, d)

        #print('q:',q.shape)
        #print('k:',q.shape)
        #print('v:',q.shape)
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        #print('q:',q.shape)
        #print('k:',q.shape)
        #print('v:',q.shape)
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
            hasattr(module, 'conv_proj_q')
            and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
            hasattr(module, 'conv_proj_k')
            and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
            hasattr(module, 'conv_proj_v')
            and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops

class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 rank=0,
                 lora_alpha=1,
                 **kwargs):
        super().__init__()

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, rank=rank, lora_alpha=lora_alpha,**kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out) 
        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )
    def forward(self, x, h, w,d):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w,d)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 kernel_size=3,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W,D = x.shape
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D)

        return x
    
class TransposeConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """
    def __init__(self,
                 kernel_size=3,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()

        self.proj = nn.ConvTranspose3d(
            in_chans, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W,D = x.shape
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D)

        return x

class VisionTransformer_up(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='xavier',
                 rank=0,
                 lora_alpha=1,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = TransposeConvEmbed(
            # img_size=img_size,
            #patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=in_chans,
            stride=stride,
            padding=padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            # logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            # logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W ,D= x.size()

        x = rearrange(x, 'b c h w d -> b (h w d) c')

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W,D)
 
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D)

        return x 

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='xavier',
                 rank=0,
                 lora_alpha=1,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            #patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=in_chans,
            stride=stride,
            padding=padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            # logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            # logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x=torch.unsqueeze(x, dim=0)
        x = self.patch_embed(x)
        B, C, H, W ,D= x.size()

        x = rearrange(x, 'b c h w d -> b (h w d) c')

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W,D)
 
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D)

        return x 
#
class ResBlock3D(nn.Module):
    def __init__(self,in_chan,out_chan):
        super().__init__()
        self.in_chan=in_chan
        self.out_chan=out_chan
        self.conv=nn.Conv3d(in_chan,out_chan,kernel_size=1,stride=1,padding=0)
        self.transconv=nn.ConvTranspose3d(in_chan,out_chan,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        return self.conv(x)
#
class Generator(nn.Module):
    def __init__(self, rank=8, lora_alpha=16):
        super().__init__()
        #downsampling part
        #layer0 1->64
        #64*64*64
        print(f"Generator: LoRA Rank {rank}, Alpha {lora_alpha}")
        self.encoder_layer0_down=nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(64),
            )
        self.encoder_layer1_down=nn.Sequential(
            VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=64,embed_dim=128,
                                              depth=1,num_heads=2,mlp_ratio=4,rank=rank,lora_alpha=lora_alpha))
        self.encoder_layer2_down=nn.Sequential(
            VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=128,embed_dim=256,
                                              depth=2,num_heads=4,mlp_ratio=4,drop_rate=0.2,rank=rank,lora_alpha=lora_alpha))

        self.encoder_layer3_down=nn.Sequential(
            VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=256,embed_dim=512,
                                              depth=2,num_heads=4,mlp_ratio=4,rank=rank,lora_alpha=lora_alpha))
        # self.encoder_layer4_down=nn.Sequential(
        #     VisionTransformer(kernel_size=3,stride=1,padding=1,in_chans=512,embed_dim=512,
        #                                       depth=2,num_heads=4,mlp_ratio=4))

        '''
        self.decoder_layer1=nn.Sequential(
            nn.ConvTranspose3d(512*2,512,kernel_size=2,stride=1,padding=1),
            nn.ConvTranspose3d(512,384,kernel_size=2,stride=1,padding=1),
            nn.BatchNorm3d(384),
            nn.LeakyReLU(0.2),)
        self.transres1=ResBlock3D(1024, 384)
        self.decoder_layer1_up=nn.Sequential(
            nn.ConvTranspose3d(384, 256, kernel_size)
            )
        '''
        #4*4*4
        self.decoder_layer1=nn.Sequential(
            nn.Conv3d(512,512,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(512),
            )
        self.resup1=ResBlock3D(512, 512)
        self.decoder_layer1_up=VisionTransformer_up(kernel_size=2,stride=2,in_chans=512,embed_dim=256,depth=2,num_heads=4,rank=rank,lora_alpha=lora_alpha)
        #8*8*8
        self.decoder_layer2=nn.Sequential(
            nn.Conv3d(256*2,256,kernel_size=3,stride=1,padding=1),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(256),
            )
        self.resup2=ResBlock3D(256*2, 256)
        self.decoder_layer2_up=VisionTransformer_up(kernel_size=2,stride=2,in_chans=256,embed_dim=128,depth=1,num_heads=4,rank=rank,lora_alpha=lora_alpha)
        #16*16*16
        #
        self.decoder_layer3=nn.Sequential(
            nn.Conv3d(128*2,128,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(128),
            )
        self.resup3=ResBlock3D(128*2, 128)
        self.decoder_layer3_up=nn.ConvTranspose3d(128,64,kernel_size=2,stride=2)
        #32*32*32
        #
        self.decoder_layer4=nn.Sequential(
            nn.Conv3d(64*2,64,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(64,32,kernel_size=3,stride=1,padding=1),
            nn.ConvTranspose3d(32,1,kernel_size=2,stride=2),
            nn.LeakyReLU(0.2)
            )
        #64*64*64
    def forward(self,x):
        # print("---------Generator---------")
        # print("Gen input x:",x.shape)
        # x=self.encoder_conv(x)
        en0=self.encoder_layer0_down(x)#32*32*32
        # print("Gen en0:",en0.shape)
        en1=self.encoder_layer1_down(en0)#16*16*16
        # print("Gen en1:",en1.shape)
        
        en2=self.encoder_layer2_down(en1)#8*8*8
        # print("Gen en2:",en2.shape)
        en3=self.encoder_layer3_down(en2)#4*4*4
        # print("Gen en3:",en3.shape)
         
        
        
        # print("---------Decoder---------")
        de0=self.decoder_layer1(en3)+self.resup1(en3)
        # print("de0:", de0.shape)#(1, 512, 8, 8, 8)
        
        de0=self.decoder_layer1_up(de0)#8*8*8
        # print("Gen d0:",de0.shape)
        
        cat1=torch.cat([en2,de0],1)
        de1=self.decoder_layer2(cat1)+self.resup2(cat1)
        de1=self.decoder_layer2_up(de1)
        # print("Gen d1:",de1.shape)
        
        cat2=torch.cat([en1,de1],1)
        de2=self.decoder_layer3(cat2)+self.resup3(cat2)
        de2=self.decoder_layer3_up(de2)
        # print("Gen d2:",de2.shape)
        
        cat3=torch.cat([en0,de2],1)
        de3=self.decoder_layer4(cat3)
        
        # print("Gen d3:",de3.shape) 
        return de3+x 
#
class Discriminator(nn.Module):
    def __init__(self, rank=0, lora_alpha=1):
        super().__init__()
        print(f"Discriminator: LoRA Rank {rank}, Alpha {lora_alpha}")
        self.conv0=nn.Sequential(
            nn.Conv3d(2,8,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(8),
            )
        self.conv1=nn.Sequential(
            nn.Conv3d(8,16,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(16),
            )
        self.conv2=nn.Sequential(
            nn.Conv3d(16,32,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(32),
            nn.Dropout3d(0.2),
            )
        self.conv3=VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=32,embed_dim=64,
                                              depth=1,num_heads=2,mlp_ratio=4,drop_rate=0.3,rank=rank,lora_alpha=lora_alpha)
        self.conv4=VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=64,embed_dim=64,
                                              depth=1,num_heads=2,mlp_ratio=4,drop_rate=0.3,rank=rank,lora_alpha=lora_alpha)

        #self.conv5=nn.Conv3d(128,1,kernel_size=1,stride=1,padding=0)
        self.mlp=nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            Rearrange('... () () () -> ...'),
        )
        self.linear=nn.Linear(128, 1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        # print("---------Discriminator---------")
        # print("dis input x:",x.shape)
        x=self.conv0(x)
        # print("dis conv0 x:", x.shape)
        x=self.conv1(x)
        # print("dis conv1 x:", x.shape)
        x=self.conv2(x)
        # print("dis conv2 x:", x.shape)
        x=self.conv3(x)
        # print("dis conv3 x:", x.shape)
        x=self.conv4(x)
        # print("dis conv4 x:", x.shape)
        #x=self.conv5(x)
        # print("dis conv5 x:",x.shape)
        #x=self.mlp(x)
        #print("dis mlp x:",x.shape)
        #x=self.linear(x.view(128))
        #print("dis linear view x:", x.shape)
        x=self.sigmoid(x)
        # print("dis output x:", x.shape)
        return x
    
# (1, 16, 64, 64, 64)
# (1, 32, 32, 32, 32)
# (1, 64, 16, 16, 16)
# (1, 64, 8, 8, 8)
# (1, 64)

#
def D_train(D: Discriminator, G: Generator, X, Y,BCELoss, optimizer_D):
    # Label to object (right to left) x:lpet y:spet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #image_size = X.size(3) // 2
    x = X.to(device)  # label
    y = Y.to(device)  # ground truth
    #x = X[:, :, :, image_size:]  # Label map (right half)
    #y = X[:, :, :, :image_size]   # Physical map (left half)
    xy = torch.cat([x, y], dim=1)  
    #
    D.zero_grad()
    # real data
    D_output_r = D(xy).squeeze()
    #D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()))
    D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
    # fake data
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))
    #D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()))
    # back prop
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    #print('Dloss:',D_loss.item())
    D_loss.backward()
    optimizer_D.step()
    return D_loss.data.item()
#
## training generator ##
def G_train(D: Discriminator, G:Generator, X, Y,BCELoss, L1, optimizer_G, lamb=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #image_size = X.size(3) // 2
    x = X.to(device)   
    y = Y.to(device)   
    #x = X[:, :, :, image_size:]   
    #y = X[:, :, :, :image_size]   
    G.zero_grad()
    # fake data
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    #print(D_output_f.shape)
    G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    #G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()))
    G_L1_Loss = L1(G_output, y)
    #print('G_L1_loss:',G_L1_Loss.item())
    print('bce_loss:',G_BCE_loss.data.item(),' L1_loss:',G_L1_Loss.data.item())
    #
    G_loss = G_BCE_loss + lamb * G_L1_Loss
    #print('cur_g_loss:',G_loss.item())
    G_loss.backward()
    optimizer_G.step()
    return G_loss.data.item() 


def G_val(D, G, X, Y, BCELoss, L1, optimizer_G, lamb=100):
   
              
                  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #image_size = X.size(3) // 2
    
    x = X.to(device)   
    y = Y.to(device)   
    #x = X[:, :, :, image_size:]   
    #y = X[:, :, :, :image_size]   
    G.zero_grad()
    # fake data
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    #print(D_output_f.shape)
    G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    #G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()))
    G_L1_Loss = L1(G_output, y)
    #print('G_L1_loss:',G_L1_Loss.item())
    print('bce_loss:',G_BCE_loss.data.item(),' L1_loss:',G_L1_Loss.data.item())
    #
    G_loss = G_BCE_loss + lamb * G_L1_Loss
    #print('cur_g_loss:',G_loss.item())
    G_loss.backward()
    optimizer_G.step()
    return G_loss.data.item() 



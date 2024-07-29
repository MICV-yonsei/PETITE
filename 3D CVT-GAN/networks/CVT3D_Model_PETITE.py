# CVT3D_Model_PETITE.py
# Tested under 
# main_tuning.py --tuning --tune_mode "petite-cvt" with trainer_tuning.py 
# Encoder 1,2,3 layer - Visual Prompt Tuning (Shallow) 
# Decoder 1,2 layer - LoRA 
# Tuning bitfit across all layers

# LoRA
# Modified to place Lora instead of Convolutional Projection in Attention
# nn.Conv3d -> lora.Conv3d

# 1. 1) Add prompts to encoder 2) Add LoRA to decoder
# 2. ViT_prompted supports self.num_tokens=0 and self.DEEP=False at once
# 3. Contorl whether execute deep mode or shallow mode with argparser

from functools import partial
from itertools import repeat
from timm.models.layers import DropPath, trunc_normal_
from thop import profile
from collections import OrderedDict
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage
from einops import rearrange
from einops.layers.torch import Rearrange

import logging
import warnings
warnings.filterwarnings("ignore")
import os
import math
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk

# LoRA
import loralib as lora


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
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method, rank=rank, lora_alpha=lora_alpha 
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
            if rank==0 and lora_alpha==1 :
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
            else :
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

    def forward_conv(self, x, h, w, d):
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w,d=d)
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
            q, k, v = self.forward_conv(x, h, w,d)
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

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
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, rank=rank, lora_alpha=lora_alpha,
            **kwargs
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
        
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
 
        
        x = x + res
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
                 rank=0,
                 lora_alpha=1,
                 init='xavier',
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
           
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
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
            x = blk(x, H, W, D)
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D) 

        return x 

class VisionTransformer_up_prompted(nn.Module):
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
                 prompt_dropout=0.0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 project=-1,
                 num_tokens=4,
                 deep=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='xavier',
                 INITIATION='random',
                 **kwargs):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  
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
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim+num_tokens,
                    dim_out=embed_dim+num_tokens,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

        self.prompt_dropout = Dropout(p=prompt_dropout)
        self.project = project
        self.num_tokens = num_tokens

        if (embed_dim == 128): hidden_size = 16
        elif (embed_dim == 256): hidden_size = 8
        elif (embed_dim == 512): hidden_size = 4
        else: hidden_size = 2

        if self.project == -1:
            prompt_dim = hidden_size
            self.prompt_proj = nn.Identity()

        self.INITIATION = INITIATION
        self.DEEP = deep

        if self.INITIATION == "random":
            patch_size = kernel_size
            if isinstance(patch_size, int):
                patch_size = (patch_size,)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  
            self.prompt_embeddings1 = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim, prompt_dim, prompt_dim))
            self.prompt_embeddings2 = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim, prompt_dim, int(prompt_dim/2)))            
            nn.init.uniform_(self.prompt_embeddings1.data, -val, val)
            nn.init.uniform_(self.prompt_embeddings2.data, -val, val)
            if self.DEEP:  
                total_d_block = len(self.blocks) 
                self.deep_prompt_embeddings1 = nn.Parameter(torch.zeros(total_d_block, num_tokens, prompt_dim, prompt_dim, prompt_dim)) 
                self.deep_prompt_embeddings2 = nn.Parameter(torch.zeros(total_d_block, num_tokens, prompt_dim, prompt_dim, int(prompt_dim/2))) 
                nn.init.uniform_(self.deep_prompt_embeddings1.data, -val, val)
                nn.init.uniform_(self.deep_prompt_embeddings2.data, -val, val)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def incorporate_prompt(self, x):
        B, C, H, W ,D= x.size()
        
        if self.project == -1:
            if D == W:
                self.prompt_embeddings = self.prompt_embeddings1
            elif D == int(W/2):
                self.prompt_embeddings = self.prompt_embeddings2
            else:
                raise ValueError(f"Unsupported Dimension {B, C, H, W, D}")
            prompt_emb = self.prompt_proj(self.prompt_embeddings)
    
        x = torch.cat((
                self.prompt_dropout(prompt_emb).expand(B, -1, -1, -1, -1),
                x
            ), dim=1) 

        return x

    def forward_deep_prompt(self, x):
        B, C, H, W ,D = x.size()

        if D == W:
            self.deep_prompt_embeddings = self.deep_prompt_embeddings1
        elif D == int(W/2):
            self.deep_prompt_embeddings = self.deep_prompt_embeddings2
        else:
            raise ValueError(f"Unsupported Dimension {B, C, H, W, D}")

        for i, blk in enumerate(self.blocks):
            if i !=0:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_emb = rearrange(self.deep_prompt_embeddings[i-1], 'b h w d -> b (h w d)')
                    deep_emb = self.prompt_proj(deep_emb)
                    deep_emb = rearrange(deep_emb, 'b (h w d) -> b h w d', h=H, w=W, d=D)
                    x = torch.cat((
                        self.prompt_dropout(deep_emb).expand(B, -1, -1, -1, -1),
                        x
                    ), dim=1)

            x = rearrange(x, 'b c h w d -> b (h w d) c') 
            x = self.pos_drop(x)
            x = blk(x, H, W, D)
            x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D)
            x = x[:,self.num_tokens:,:,:,:]

        return x

    def train(self, mode=True):
        if mode:
            self.blocks.eval()
            self.patch_embed.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def forward(self, x):
        if self.INITIATION == "reuse":
            x = x[:,self.num_tokens:,:,:,:]
            self.prompt_embeddings = x[:,:self.num_tokens,:,:,:]

        x = self.patch_embed(x) 
        if self.num_tokens > 0:
            x = self.incorporate_prompt(x)

        if self.DEEP:
            x = self.forward_deep_prompt(x)
        else:
            B, C, H, W ,D = x.size()
            x = rearrange(x, 'b c h w d -> b (h w d) c') 
            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks): 
                x = blk(x, H, W, D)
            x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D)

            x = x[:,self.num_tokens:,:,:,:]

        return x 

class VisionTransformer_prompted(nn.Module):
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
                 prompt_dropout=0.0,
                 DROPOUT=0.0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 project=-1,
                 num_tokens=4,
                 deep=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='xavier',
                 INITIATION='random',
                 **kwargs):
        
        super().__init__()
        
        self.num_features = self.embed_dim = embed_dim  # NUM_TOKENS
        self.rearrage = None
        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            # patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=in_chans,
            stride=stride,
            padding=padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim+num_tokens,
                    dim_out=embed_dim+num_tokens,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)
        
        if init == 'xavier': 
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)
        
        self.prompt_dropout = Dropout(p=prompt_dropout)
        self.project = project
        self.num_tokens = num_tokens 

        if (embed_dim == 128): hidden_size = 16
        elif (in_chans == 32 and embed_dim == 64) or (embed_dim == 256): hidden_size = 8
        elif (in_chans == 64 and embed_dim == 64) or (embed_dim == 512): hidden_size = 4
        else: raise ValueError(f"Unsupported embed dim {embed_dim}")

        if self.project == -1:
            prompt_dim = hidden_size  
            self.prompt_proj = nn.Identity()
            
        self.INITIATION = INITIATION
        self.DEEP = deep
        if self.INITIATION == "random":
            patch_size = kernel_size
            if isinstance(patch_size, int):
                patch_size = (patch_size,)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  
            self.prompt_embeddings1 = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim, prompt_dim, prompt_dim))
            self.prompt_embeddings2 = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim, prompt_dim, int(prompt_dim/2)))            
            nn.init.uniform_(self.prompt_embeddings1.data, -val, val)
            nn.init.uniform_(self.prompt_embeddings2.data, -val, val)
            
            if self.DEEP:  
                total_d_block = len(self.blocks) 
                self.deep_prompt_embeddings1 = nn.Parameter(torch.zeros(total_d_block, num_tokens, prompt_dim, prompt_dim, prompt_dim)) 
                self.deep_prompt_embeddings2 = nn.Parameter(torch.zeros(total_d_block, num_tokens, prompt_dim, prompt_dim, int(prompt_dim/2))) 
                nn.init.uniform_(self.deep_prompt_embeddings1.data, -val, val)
                nn.init.uniform_(self.deep_prompt_embeddings2.data, -val, val)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def incorporate_prompt(self, x):
        B, C, H, W ,D= x.size()

        if self.project == -1:
            if D == W:
                self.prompt_embeddings = self.prompt_embeddings1
            elif D == int(W/2):
                self.prompt_embeddings = self.prompt_embeddings2
            else:
                raise ValueError(f"Unsupported Dimension {B, C, H, W, D}")
        
            prompt_emb = self.prompt_proj(self.prompt_embeddings)
            
        x = torch.cat((
                self.prompt_dropout(prompt_emb).expand(B, -1, -1, -1, -1),
                x
            ), dim=1) 

        return x

    def forward_deep_prompt(self, x):
        B, C, H, W ,D = x.size()

        if D == W:
            self.deep_prompt_embeddings = self.deep_prompt_embeddings1
        elif D == int(W/2):
            self.deep_prompt_embeddings = self.deep_prompt_embeddings2
        else:
            raise ValueError(f"Unsupported Dimension {B, C, H, W, D}")

        for i, blk in enumerate(self.blocks):
            if i !=0:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_emb = rearrange(self.deep_prompt_embeddings[i-1], 'b h w d -> b (h w d)')
                    deep_emb = self.prompt_proj(deep_emb)
                    deep_emb = rearrange(deep_emb, 'b (h w d) -> b h w d', h=H, w=W, d=D)
                    x = torch.cat((
                        self.prompt_dropout(deep_emb).expand(B, -1, -1, -1, -1),
                        x
                    ), dim=1)

            x = rearrange(x, 'b c h w d -> b (h w d) c') 
            x = self.pos_drop(x)
            x = blk(x, H, W, D)
            x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D)
            x = x[:,self.num_tokens:,:,:,:]
            prompt = x[:,:self.num_tokens,:,:,:]
            
        return x, prompt

    def train(self, mode=True):
        if mode:
            self.blocks.eval()
            self.patch_embed.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def forward(self, x): 
        x = self.patch_embed(x) 
        if self.num_tokens > 0:
            x = self.incorporate_prompt(x)

        if self.DEEP:
            x, prompt = self.forward_deep_prompt(x)
        else:
            B, C, H, W ,D = x.size()
            x = rearrange(x, 'b c h w d -> b (h w d) c') 
            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks): 
                x = blk(x, H, W, D)
            x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W,d=D)

            x = x[:,self.num_tokens:,:,:,:]
            prompt = x[:,:self.num_tokens,:,:,:]
            
        if self.INITIATION=="reuse":
            return x, prompt
        return x

class ResBlock3D(nn.Module):
    def __init__(self,in_chan,out_chan):
        super().__init__()
        self.in_chan=in_chan
        self.out_chan=out_chan
        self.conv=nn.Conv3d(in_chan,out_chan,kernel_size=1,stride=1,padding=0)
        self.transconv=nn.ConvTranspose3d(in_chan,out_chan,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, en1_tokens=0, en2_tokens=8, en3_tokens=32, de1_tokens = 0, de2_tokens = 0, rank=8, lora_alpha=16,
                 deep=False, INITIATION="random"):
        super().__init__()
        # VPT
        self.en1_tokens = en1_tokens
        self.en2_tokens = en2_tokens
        self.en3_tokens = en3_tokens
        # LoRA
        self.de1_tokens = de1_tokens
        self.de2_tokens = de2_tokens
        
        print(f"Number of prompt embeddings prepended to Generator: en1 {self.en1_tokens}, en2 {self.en2_tokens}, en3 {self.en3_tokens}")
        print(f"Generator: LoRA Rank {rank}, Alpha {lora_alpha}")
       

        self.encoder_layer0_down=nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(64),
            )
        
        self.encoder_layer1_down=nn.Sequential(
            VisionTransformer_prompted(kernel_size=3,stride=2,padding=1,in_chans=64,embed_dim=128,
                                              depth=1,num_heads=2,mlp_ratio=4,num_tokens=self.en1_tokens, INITIATION=INITIATION))
        self.encoder_layer2_down=nn.Sequential(
            VisionTransformer_prompted(kernel_size=3,stride=2,padding=1,in_chans=128,embed_dim=256,
                                              depth=2,num_heads=4,mlp_ratio=4,drop_rate=0.2,num_tokens=self.en2_tokens,deep=deep, INITIATION=INITIATION))
        self.encoder_layer3_down=nn.Sequential(
            VisionTransformer_prompted(kernel_size=3,stride=2,padding=1,in_chans=256,embed_dim=512,
                                              depth=2,num_heads=4,mlp_ratio=4, num_tokens=self.en3_tokens,deep=deep, INITIATION=INITIATION))
     
        self.decoder_layer1=nn.Sequential(
            nn.Conv3d(512,512,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(512),
            )
        self.resup1=ResBlock3D(512, 512)
        self.decoder_layer1_up=VisionTransformer_up(kernel_size=2,stride=2,in_chans=512,embed_dim=256,
                                                                    depth=2,num_heads=4, rank=rank, lora_alpha=lora_alpha)

        self.decoder_layer2=nn.Sequential(
            nn.Conv3d(256*2,256,kernel_size=3,stride=1,padding=1),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(256),
            )
        self.resup2=ResBlock3D(256*2, 256)
        self.decoder_layer2_up=VisionTransformer_up(kernel_size=2,stride=2,in_chans=256,embed_dim=128,
                                                                    depth=1,num_heads=4, rank=rank, lora_alpha=lora_alpha)
        self.decoder_layer3=nn.Sequential(
            nn.Conv3d(128*2,128,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(128),
            )
        self.resup3=ResBlock3D(128*2, 128)
        self.decoder_layer3_up=nn.ConvTranspose3d(128,64,kernel_size=2,stride=2)
        self.decoder_layer4=nn.Sequential(
            nn.Conv3d(64*2,64,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(64,32,kernel_size=3,stride=1,padding=1),
            nn.ConvTranspose3d(32,1,kernel_size=2,stride=2),
            nn.LeakyReLU(0.2)
            )


    def load_from(self, weight):            
        with torch.no_grad():
            prompted_weight = []

            for layer in [self.encoder_layer1_down, self.encoder_layer2_down, self.encoder_layer3_down]:
                if layer == self.encoder_layer1_down:
                    l = 1
                    num_tokens = self.en1_tokens
                elif layer == self.encoder_layer2_down:
                    l = 2
                    num_tokens = self.en2_tokens
                elif layer == self.encoder_layer3_down:
                    l = 3
                    num_tokens = self.en3_tokens

                for i in range(len(layer[0].blocks)):
                    layer[0].blocks[i].norm1.weight[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.norm1.weight"])
                    layer[0].blocks[i].norm1.bias[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.norm1.bias"])
                    layer[0].blocks[i].attn.conv_proj_q.conv.weight[num_tokens:,...].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.conv.weight"])
                    layer[0].blocks[i].attn.conv_proj_q.bn.weight[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.weight"])
                    layer[0].blocks[i].attn.conv_proj_q.bn.bias[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.bias"])
                    layer[0].blocks[i].attn.conv_proj_q.bn.running_mean[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.running_mean"])
                    layer[0].blocks[i].attn.conv_proj_q.bn.running_var[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.running_var"])
                    # layer[0].blocks[i].attn.conv_proj_q.bn.num_batches_tracked.copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.num_batches_tracked"])
                    layer[0].blocks[i].attn.conv_proj_k.conv.weight[num_tokens:,...].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.conv.weight"])
                    layer[0].blocks[i].attn.conv_proj_k.bn.weight[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.weight"])
                    layer[0].blocks[i].attn.conv_proj_k.bn.bias[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.bias"])
                    layer[0].blocks[i].attn.conv_proj_k.bn.running_mean[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.running_mean"])
                    layer[0].blocks[i].attn.conv_proj_k.bn.running_var[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.running_var"])
                    # layer[0].blocks[i].attn.conv_proj_k.bn.num_batches_tracked.copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.num_batches_tracked"])
                    layer[0].blocks[i].attn.conv_proj_v.conv.weight[num_tokens:,...].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.conv.weight"])
                    layer[0].blocks[i].attn.conv_proj_v.bn.weight[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.weight"])
                    layer[0].blocks[i].attn.conv_proj_v.bn.bias[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.bias"])
                    layer[0].blocks[i].attn.conv_proj_v.bn.running_mean[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.running_mean"])
                    layer[0].blocks[i].attn.conv_proj_v.bn.running_var[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.running_var"])
                    # layer[0].blocks[i].attn.conv_proj_v.bn.num_batches_tracked.copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.num_batches_tracked"])
                    layer[0].blocks[i].attn.proj_q.weight[num_tokens:,num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.proj_q.weight"])
                    layer[0].blocks[i].attn.proj_k.weight[num_tokens:,num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.proj_k.weight"])
                    layer[0].blocks[i].attn.proj_v.weight[num_tokens:,num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.proj_v.weight"])
                    layer[0].blocks[i].attn.proj.weight[num_tokens:,num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.proj.weight"])
                    layer[0].blocks[i].attn.proj.bias[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.attn.proj.bias"])
                    layer[0].blocks[i].norm2.weight[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.norm2.weight"])
                    layer[0].blocks[i].norm2.bias[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.norm2.bias"])
                    layer[0].blocks[i].mlp.fc1.weight[num_tokens*4:,num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.mlp.fc1.weight"])
                    layer[0].blocks[i].mlp.fc1.bias[num_tokens*4:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.mlp.fc1.bias"])
                    layer[0].blocks[i].mlp.fc2.weight[num_tokens:,num_tokens*4:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.mlp.fc2.weight"])
                    layer[0].blocks[i].mlp.fc2.bias[num_tokens:].copy_(weight[f"encoder_layer{l}_down.0.blocks.{i}.mlp.fc2.bias"])

                    prompted_weight.extend([
                        f'encoder_layer{l}_down.0.blocks.{i}.norm1.weight', f'encoder_layer{l}_down.0.blocks.{i}.norm1.bias', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.conv.weight', f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.weight', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.bias', f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.running_mean', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.running_var', #f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_q.bn.num_batches_tracked', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.conv.weight', f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.weight', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.bias', f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.running_mean', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.running_var', #f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_k.bn.num_batches_tracked', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.conv.weight', f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.weight', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.bias', f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.running_mean', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.running_var', #f'encoder_layer{l}_down.0.blocks.{i}.attn.conv_proj_v.bn.num_batches_tracked', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.proj_q.weight', f'encoder_layer{l}_down.0.blocks.{i}.attn.proj_k.weight', f'encoder_layer{l}_down.0.blocks.{i}.attn.proj_v.weight', 
                        f'encoder_layer{l}_down.0.blocks.{i}.attn.proj.weight', f'encoder_layer{l}_down.0.blocks.{i}.attn.proj.bias', 
                        f'encoder_layer{l}_down.0.blocks.{i}.norm2.weight', f'encoder_layer{l}_down.0.blocks.{i}.norm2.bias', 
                        f'encoder_layer{l}_down.0.blocks.{i}.mlp.fc1.weight', f'encoder_layer{l}_down.0.blocks.{i}.mlp.fc1.bias', 
                        f'encoder_layer{l}_down.0.blocks.{i}.mlp.fc2.weight', f'encoder_layer{l}_down.0.blocks.{i}.mlp.fc2.bias'
                    ])

            
            for j in prompted_weight:
                del weight[j]

            return weight

    def forward(self,x):
        en0=self.encoder_layer0_down(x) 
        en1=self.encoder_layer1_down(en0)
        en2=self.encoder_layer2_down(en1)
        en3=self.encoder_layer3_down(en2)
        de0=self.decoder_layer1(en3)+self.resup1(en3)
        de0=self.decoder_layer1_up(de0)
        cat1=torch.cat([en2,de0],1)
        de1=self.decoder_layer2(cat1)+self.resup2(cat1)
        de1=self.decoder_layer2_up(de1)
        cat2=torch.cat([en1,de1],1)
        de2=self.decoder_layer3(cat2)+self.resup3(cat2)
        de2=self.decoder_layer3_up(de2)
        cat3=torch.cat([en0,de2],1)
        de3=self.decoder_layer4(cat3)
        return de3+x

class Discriminator(nn.Module):
    # NOT TUNING DISCRIMINATOR
    def __init__(self, conv3_tokens=0, conv4_tokens=0):
        super().__init__()
        self.conv3_tokens=conv3_tokens
        self.conv4_tokens=conv4_tokens

        print(f"Number of prompt embeddings prepended to Discriminator: conv3 {self.conv3_tokens}, conv4 {self.conv4_tokens}")

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
                                              depth=1,num_heads=2,mlp_ratio=4,drop_rate=0.3,)
        self.conv4=VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=64,embed_dim=64,
                                              depth=1,num_heads=2,mlp_ratio=4,drop_rate=0.3,)

        self.mlp=nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            Rearrange('... () () () -> ...'),
        )
        self.linear=nn.Linear(128, 1)
        self.sigmoid=nn.Sigmoid()

    
    def forward(self,x):
        x=self.conv0(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.sigmoid(x)
        return x
    
def D_train(D: Discriminator, G: Generator, X, Y,BCELoss, optimizer_D):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #image_size = X.size(3) // 2
    x = X.to(device)  
    y = Y.to(device)  
    xy = torch.cat([x, y], dim=1)  

    D.zero_grad()
    D_output_r = D(xy).squeeze()
    D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()
    return D_loss.data.item()

def G_train(D: Discriminator, G:Generator, X, Y,BCELoss, L1, optimizer_G, lamb=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = X.to(device)   
    y = Y.to(device)     
    G.zero_grad()
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    G_L1_Loss = L1(G_output, y)
    G_loss = G_BCE_loss + lamb * G_L1_Loss
    G_loss.backward()
    optimizer_G.step()
    return G_loss.data.item() 

def G_val(D, G, X, Y, BCELoss, L1, optimizer_G, lamb=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = X.to(device)   
    y = Y.to(device)   
    G.zero_grad()
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    G_L1_Loss = L1(G_output, y)
    G_loss = G_BCE_loss + lamb * G_L1_Loss
    G_loss.backward()
    optimizer_G.step()
    return G_loss.data.item() 


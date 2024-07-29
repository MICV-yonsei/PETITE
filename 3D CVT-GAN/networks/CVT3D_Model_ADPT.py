# CVT3D_Model_adapter.py
# Tested under 
# main_cvt.py --tuning --tune_mode "adpt" with trainer_all.py 
# 
# 1. Add the adapter module in class Block
# 2. Use of the adapter can be control with Boolean in Generator
# 3. New hyperparameter 'Reduction Factor' of the adapter is added
#    -- can be control with float in Generator

from functools import partial
from itertools import repeat
from einops import rearrange
from einops.layers.torch import Rearrange
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
from thop import profile


import logging
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk

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
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
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
    """
    Insert Adapter
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm,
                 reduction_factor=1,
                 adpt_mode=False,
                 **kwargs):
        super().__init__()

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
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
        
        self.adpt_mode = adpt_mode
        if adpt_mode:
            self.adapter_downsample = nn.Linear(dim_out, int(dim_out/reduction_factor))
            self.adapter_upsample = nn.Linear(int(dim_out/reduction_factor), dim_out)
            self.adapter_act_fn = act_layer()
            
            nn.init.zeros_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)
            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)   
            
    def forward(self, x, h, w,d):
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w,d)
        x = res + self.drop_path(attn)
        
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        
        if self.adpt_mode:
            adpt = self.adapter_downsample(x)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_upsample(adpt)
            x = adpt + x
        
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
                 init='xavier',
                 reduction_factor=1,
                 adpt_mode=False,
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
                    reduction_factor=reduction_factor,
                    adpt_mode=adpt_mode,
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
#
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
                 reduction_factor=1,
                 adpt_mode=False,
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
                    reduction_factor=reduction_factor,
                    adpt_mode=adpt_mode,
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
    def __init__(self, reduction_factor=1):
        super().__init__()
        print(f"Reduction Factor: {reduction_factor}")
   
        self.encoder_layer0_down=nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(64),
            )
        self.encoder_layer1_down=nn.Sequential(
            VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=64,embed_dim=128,
                                              depth=1,num_heads=2,mlp_ratio=4,reduction_factor=reduction_factor, adpt_mode=True))
        self.encoder_layer2_down=nn.Sequential(
            VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=128,embed_dim=256,
                                              depth=2,num_heads=4,mlp_ratio=4,drop_rate=0.2,reduction_factor=reduction_factor, adpt_mode=True))

        self.encoder_layer3_down=nn.Sequential(
            VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=256,embed_dim=512,
                                              depth=2,num_heads=4,mlp_ratio=4,reduction_factor=reduction_factor, adpt_mode=True))
        
        self.decoder_layer1=nn.Sequential(
            nn.Conv3d(512,512,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(512),
            )
        self.resup1=ResBlock3D(512, 512)
        self.decoder_layer1_up=VisionTransformer_up(kernel_size=2,stride=2,in_chans=512,embed_dim=256,depth=2,num_heads=4,
                                                    reduction_factor=reduction_factor, adpt_mode=True)
        self.decoder_layer2=nn.Sequential(
            nn.Conv3d(256*2,256,kernel_size=3,stride=1,padding=1),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(256),
            )
        self.resup2=ResBlock3D(256*2, 256)
        self.decoder_layer2_up=VisionTransformer_up(kernel_size=2,stride=2,in_chans=256,embed_dim=128,depth=1,num_heads=4,
                                                    reduction_factor=reduction_factor, adpt_mode=True)
  
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
    def __init__(self,reduction_factor=1):
        super().__init__()
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
                                              depth=1,num_heads=2,mlp_ratio=4,drop_rate=0.3,
                                            )
        self.conv4=VisionTransformer(kernel_size=3,stride=2,padding=1,in_chans=64,embed_dim=64,
                                              depth=1,num_heads=2,mlp_ratio=4,drop_rate=0.3,
                                            )

        #self.conv5=nn.Conv3d(128,1,kernel_size=1,stride=1,padding=0)
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

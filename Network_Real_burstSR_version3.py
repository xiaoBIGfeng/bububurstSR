import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import DeformConv2d
from pytorch_lightning import seed_everything
from einops import rearrange
import numbers
from torch import einsum


from utils.metrics import PSNR
psnr_fn = PSNR(boundary_ignore=40)

seed_everything(13)

######################################## Model and Dataset ########################################################


from datasets.burstsr_dataset import BurstSRDataset

from utils.data_format_utils import torch_to_numpy, numpy_to_torch
from utils.data_format_utils import convert_dict
from utils.postprocessing_functions import BurstSRPostProcess

from utils.metrics import AlignedL1, AlignedL1_loss, AlignedL2_loss, AlignedSSIM_loss, AlignedPSNR, AlignedSSIM, AlignedLPIPS, AlignedLPIPS_loss

from pwcnet.pwcnet import PWCNet
from utils.warp import warp

import data_processing.camera_pipeline as rgb2raw
from data_processing.camera_pipeline import *

from collections import OrderedDict




##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(pl.LightningModule):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(pl.LightningModule):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(pl.LightningModule):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(pl.LightningModule):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(pl.LightningModule):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.stride = 1
        self.qk = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=self.stride, padding=1, groups=dim*2, bias=bias)

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qk = self.qk_dwconv(self.qk(x))
        q,k = qk.chunk(2, dim=1)
        
        v = self.v_dwconv(self.v(x))
        
        b, f, h1, w1 = q.size()

        q = rearrange(q, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h
##########################################################################
## Overlapping Cross-Attention (OCA)
class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out


##########################################################################
# class BFA(pl.LightningModule):
#     def __init__(self, dim, num_heads, stride, ffn_expansion_factor, bias, LayerNorm_type):
#         super(BFA, self).__init__()

#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(dim, num_heads, stride, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))

#         return x
    
class BFA(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads, spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type):
        super(BFA, self).__init__()


        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = Attention(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(pl.LightningModule):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        #print("Inside patch embed:::", inp_enc_level1.size())
        x = self.proj(x)

        return x

class alignment(pl.LightningModule):
    def __init__(self, dim=64, memory=False, stride=1, type='group_conv'):
        
        super(alignment, self).__init__()
        
        act = nn.GELU()
        bias = False

        kernel_size = 3
        padding = kernel_size//2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size**2)

        self.offset_conv = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        self.back_projection = ref_back_projection(dim, stride=1)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        
        if memory==True:
            self.bottleneck_o = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
            
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return offset, mask
        
    def forward(self, x, prev_offset_feat=None):
        
        B, f, H, W = x.size()
        ref = x[0].unsqueeze(0)
        ref = torch.repeat_interleave(ref, B, dim=0)

        offset_feat = self.bottleneck(torch.cat([ref, x], dim=1))

        if not prev_offset_feat==None:
            offset_feat = self.bottleneck_o(torch.cat([prev_offset_feat, offset_feat], dim=1))

        offset, mask = self.offset_gen(self.offset_conv(offset_feat)) 

        aligned_feat = self.deform(x, offset, mask)
        aligned_feat[0] = x[0].unsqueeze(0)

        aligned_feat = self.back_projection(aligned_feat)
        
        return aligned_feat, offset_feat


class EDA(pl.LightningModule):
    def __init__(self, in_channels=64):
        super(EDA, self).__init__()
        
        num_blocks = [4,6,6,8] 
        num_refinement_blocks = 4
        heads = [1,2,4,8]
        bias = False
        LayerNorm_type = 'WithBias'

        self.encoder_level1 = nn.Sequential(*[BFA(dim=in_channels, window_size = 4, overlap_ratio=0.5,  num_channel_heads=heads[0], num_spatial_heads=heads[0], 
                                                  spatial_dim_head = 16, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.encoder_level2 = nn.Sequential(*[BFA(dim=in_channels, window_size = 4, overlap_ratio=0.5,  num_channel_heads=heads[1], num_spatial_heads=heads[1], 
                                                  spatial_dim_head = 16, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        # self.encoder_level1 = nn.Sequential(*[BFA(dim=in_channels, num_heads=heads[0], stride=1, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        # self.encoder_level2 = nn.Sequential(*[BFA(dim=in_channels, num_heads=heads[1], stride=1, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
                
        self.down1 = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)        
        self.down2 = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

        self.alignment0 = alignment(in_channels, memory=True)
        self.alignment1 = alignment(in_channels, memory=True)
        self.alignment2 = alignment(in_channels)
        self.cascade_alignment = alignment(in_channels, memory=True)

        self.offset_up1 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        self.offset_up2 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)

        self.up1 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)        
        self.up2 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = self.encoder_level1(x)
        enc1 = self.down1(x)

        enc1 = self.encoder_level2(enc1)
        enc2 = self.down2(enc1)
        enc2, offset_feat_enc2 = self.alignment2(enc2)
        
        dec1 = self.up2(enc2)
        offset_feat_dec1 = self.offset_up2(offset_feat_enc2) * 2
        enc1, offset_feat_enc1 = self.alignment1(enc1, offset_feat_dec1)
        dec1 = dec1 + enc1

        dec0 = self.up1(dec1)
        offset_feat_dec0 = self.offset_up1(offset_feat_enc1) * 2
        x, offset_feat_x = self.alignment0(x, offset_feat_dec0)
        x = x + dec0

        alinged_feat, offset_feat_x = self.cascade_alignment(x, offset_feat_x)    
        
        return alinged_feat

class ref_back_projection(pl.LightningModule):
    def __init__(self, in_channels, stride):

        super(ref_back_projection, self).__init__()

        bias = False
        self.feat_fusion = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1), nn.GELU())
        self.encoder1 = nn.Sequential(*[BFA(dim=in_channels*2, window_size = 4, overlap_ratio=0.5,  num_channel_heads=1, num_spatial_heads=1, 
                                                  spatial_dim_head = 16, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias') for i in range(1)])
        # self.encoder1 = nn.Sequential(*[BFA(dim=in_channels*2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias') for i in range(1)])
        
        self.feat_expand = nn.Sequential(nn.Conv2d(in_channels, in_channels*2, 3, stride=1, padding=1), nn.GELU())
        self.diff_fusion = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1), nn.GELU())
        
    def forward(self, x):
        
        B, f, H, W = x.size()
        #feat = self.encoder1(x)

        ref = x[0].unsqueeze(0)
        ref = torch.repeat_interleave(ref, B, dim=0)
        feat = self.encoder1(torch.cat([ref, x], dim=1))  

        fused_feat = self.feat_fusion(feat)
        exp_feat = self.feat_expand(fused_feat)

        residual = exp_feat - feat
        residual = self.diff_fusion(residual)

        fused_feat = fused_feat + residual

        return fused_feat

class no_ref_back_projection(pl.LightningModule):
    def __init__(self, in_channels, stride):

        super(no_ref_back_projection, self).__init__()

        bias = False
        self.feat_fusion = nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1, bias = False)
        self.feat_expand = nn.Conv2d(in_channels, in_channels*2, 3, stride=1, padding=1, bias = False)
        self.diff_fusion = nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1, bias = False)
        self.encoder1= nn.Sequential(*[BFA(dim=in_channels*2, window_size = 4, overlap_ratio=0.5,  num_channel_heads=1, num_spatial_heads=1, 
                                                  spatial_dim_head = 16, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias') for i in range(1)])
        # self.encoder1 = nn.Sequential(*[BFA(dim=in_channels*2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias') for i in range(1)])

    def burst_fusion(self, x):
        b, f, H, W = x.size()
        x = x.view(-1, f*2, H, W)
        return x

    def forward(self, x):

        B, f, H, W = x.size()
        shifted_x = torch.roll(x, 1, 0)
        feat = x.view(-1, f*2, H, W)
        shifted_feat = shifted_x.view(-1, f*2, H, W)
        feat = torch.cat([feat, shifted_feat], dim=0)

        feat = self.encoder1(feat)
        fused_feat = self.feat_fusion(feat)
        rec_feat = self.feat_expand(fused_feat)

        residual = self.diff_fusion(feat - rec_feat)
        fused_feat = fused_feat + residual
        
        return fused_feat


class adapt_burst_pooling(pl.LightningModule):
    def __init__(self, in_channels, out_burst_num):

        super(adapt_burst_pooling, self).__init__()

        cur_burst_num = out_burst_num - 1
        self.adapt_burst_pool = nn.AdaptiveAvgPool1d(in_channels*cur_burst_num) 

    def forward(self, x):

        B, f, H, W = x.size()
        x_ref = x[0].unsqueeze(0)        
        x = x.view(-1, H, W)
        x = x.permute(1, 2, 0).contiguous()
        x = self.adapt_burst_pool(x)
        x = x.permute(2, 0, 1).contiguous()
        x = x.view(-1, f, H, W)
        x = torch.cat([x_ref, x], dim=0)

        return x


class Base_Model(pl.LightningModule):
    def __init__(self, num_features=48, burst_size=8, reduction=8, bias=False):
        super(Base_Model, self).__init__()
        

        self.train_loss = nn.L1Loss()
        self.valid_psnr = PSNR(boundary_ignore=40)

        bias = False
        
        self.conv1 = nn.Sequential(nn.Conv2d(4, num_features, kernel_size=3, padding=1, bias=bias))
        self.align = EDA(num_features)

        self.back_projection1 = no_ref_back_projection(num_features, stride=1)
        self.back_projection2 = no_ref_back_projection(num_features, stride=1)

        self.up1 = nn.Sequential(nn.Conv2d(num_features*8, num_features*8, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.PixelShuffle(2), nn.GELU())

        self.up2 = nn.Sequential(nn.Conv2d(num_features*2, num_features*4, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.PixelShuffle(2), nn.GELU())

        self.up3 = nn.Sequential(nn.Conv2d(num_features, num_features*4, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.PixelShuffle(2), nn.GELU())
        
        self.out_conv = nn.Sequential(nn.Conv2d(num_features, 3, kernel_size=3, padding=1, bias=bias)) 

        self.adapt_brust_pool = adapt_burst_pooling(num_features, 8)    

        PWCNet_weight_PATH = './pwcnet/pwcnet-network-default.pth'
        self.pwcnet_path = PWCNet_weight_PATH
        self.alignment_net = PWCNet(load_pretrained=True, weights_path=self.pwcnet_path).cuda()
        for param in self.alignment_net.parameters():
            param.requires_grad = False

        self.aligned_l1_loss = AlignedL1(alignment_net=self.alignment_net)
        self.aligned_psnr_fn = AlignedPSNR(alignment_net=self.alignment_net, boundary_ignore=40)
        self.alignedLPIPS_loss = AlignedLPIPS_loss(alignment_net=self.alignment_net)


    def forward(self, burst):
        
        b, n ,c, h, w = burst.size()
        alligned_feature = []
        for i in range (b):

            burst_feat = burst[i]
            burst_feat = self.conv1(burst_feat) # b*t num_features h w
            burst_feat = self.align(burst_feat)
            burst_feat = self.adapt_brust_pool(burst_feat)
            alligned_feature.append(burst_feat)

        alligned_feature = torch.stack(alligned_feature, dim=0)

        burst_feat = alligned_feature

        b, n, f, H, W = burst_feat.size()
        burst_feat = burst_feat.view(-1, f, H, W)

        burst_feat = self.back_projection1(burst_feat)
        burst_feat = burst_feat.view(b, -1, H, W)

        burst_feat = self.up1(burst_feat)
        burst_feat = burst_feat.view(-1, f, 2*H, 2*W)

        burst_feat = self.back_projection2(burst_feat)
        burst_feat = burst_feat.view(b, -1, 2*H, 2*W)

        burst_feat = self.up2(burst_feat)
        burst_feat = burst_feat.view(-1, f, 4*H, 4*W)

        burst_feat = self.up3(burst_feat) 
        burst_feat = self.out_conv(burst_feat)        
        return burst_feat
    
    def training_step(self, train_batch, batch_idx):
        
        #########################################  Torchlighting ########################### 
        x, y, meta_info_burst, meta_info_gt, burst_name = train_batch

        pred = self.forward(x)
        pred = pred.clamp(0.0, 1.0)
        # print("pred shape:", pred.shape)
        # print("y shape:", y.shape)


        # print("x device:", x.device)
        # print("y device:", y.device)
        # print("pred device:", pred.device)
        loss = self.aligned_l1_loss(pred, y, x) + 0.0095*self.alignedLPIPS_loss(pred, y, x)
        # loss = self.train_loss(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, y, meta_info_burst, meta_info_gt, burst_name = val_batch
            pred = self.forward(x)
            pred = pred.clamp(0.0, 1.0)
            PSNR = self.aligned_psnr_fn(pred, y , x)

        return PSNR

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        PSNR = torch.stack(outs).mean()
        self.log('val_psnr', PSNR, on_step=False, on_epoch=True, prog_bar=True)
        torch.cuda.empty_cache()

    def configure_optimizers(self):        
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-6)  
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6),
            'interval': 'epoch',  # 调度器更新的间隔：'epoch' 或 'step'
            'frequency': 1,  # 调度器的调用频率
            'name': 'cosine_annealing_scheduler'  # 可选，调度器的名称
        }          
        return [optimizer], [lr_scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
    
    def on_load_checkpoint(self, checkpoint):
        # 再次冻结alignment_net参数以确保其保持不动
        print("on_load_checkpoint")
        for param in self.alignment_net.parameters():
            param.requires_grad = False

if __name__ == '__main__':
    import torch
    B = 10  
    H = 32  
    W = 32  
    frames = torch.rand(1, B, 4, H, W).cuda()
    # print(frames)
    model = Base_Model().cuda()
    model.eval()
    image = model(frames)
    print('image.shape:',image.shape)

    x = torch.rand(10,4,32,32)
    ref = x[0]
    print(ref.shape)
    ref = ref.unsqueeze(0)
    print(ref.shape)
    ref = torch.repeat_interleave(ref, B, dim=0)
    print(ref.shape)


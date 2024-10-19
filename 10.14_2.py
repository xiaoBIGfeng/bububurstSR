import torch
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(project_root)
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import DeformConv2d
from pytorch_lightning import seed_everything
from einops import rearrange
import numbers
import torchvision
import cv2
import random
from utils.metrics import PSNR
psnr_fn = PSNR(boundary_ignore=40)

seed_everything(13)
from utils.data_format_utils import convert_dict
from utils.postprocessing_functions import SimplePostProcess




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
    def __init__(self, dim, num_heads, stride, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.stride = stride
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



##########################################################################
class BFA(pl.LightningModule):
    def __init__(self, dim, num_heads, stride, ffn_expansion_factor, bias, LayerNorm_type):
        super(BFA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, stride, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

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

        self.encoder_level1 = nn.Sequential(*[BFA(dim=in_channels, num_heads=heads[0], stride=1, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])
        self.encoder_level2 = nn.Sequential(*[BFA(dim=in_channels, num_heads=heads[1], stride=1, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])
                
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
        self.encoder1 = nn.Sequential(*[BFA(dim=in_channels*2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias') for i in range(2)])
        
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
        
        self.encoder1 = nn.Sequential(*[BFA(dim=in_channels*2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias') for i in range(2)])

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

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


'''
        routing_vector = self.routing(x[:, :6]).reshape(batch_size, -1)
        routing_vector = torch.sigmoid(self.l1(routing_vector))
        routing_vector = routing_vector / (routing_vector.sum(1, True) + 1e-6) * 4.5
        routing_vector = torch.clamp(routing_vector, 0, 1)
        ref = RoundSTE.apply(routing_vector)
'''
'''
class SelectiveLayer(nn.Module):
    def __init__(self):
        super(SelectiveLayer, self).__init__()

    def forward(self, burst, v):
        mask = v.squeeze() == 1
        new_burst = burst[mask]
        return new_burst

'''

class SubstitutueLayer(nn.Module):
    def __init__(self):
        super(SubstitutueLayer, self).__init__()

    def forward(self, x, v):
        # x : 14 C H W 
        # v : 13  
        # v[i-1]==0 表示 x[i]要被x[i-1]替换掉
        assert x.shape[0] == v.shape[0]+1
        for i in range(x.shape[0]-1, 0, -1):
            y = x.clone()
            y[i,:,:,:] = v[i-1] * x[i,:,:,:] + (1-v[i-1]) * x[i-1,:,:,:]

        return y

    
class SubstitutueBlock(nn.Module):
    def __init__(self):
        super(SubstitutueBlock, self).__init__()
        self.routing = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=24, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),            
        )
        self.linear = nn.Linear(6, 1)
        self.substitutue_layer = SubstitutueLayer()


    def forward(self, burst):
        
        burst_base = burst[0,:,:,:].repeat(13,1,1,1)
        burst_cur = burst[1:,:,:,:]
        x = torch.concat((burst_base,burst_cur),dim=1)
        routing_vector = self.routing(x)
        routing_vector = torch.sigmoid(self.linear(routing_vector.view(13, -1)))
        # 还要再加点其他限制
        routing_vector = torch.clamp(routing_vector, 0, 1)
        routing_vector = RoundSTE.apply(routing_vector)

        y = self.substitutue_layer(burst, routing_vector)
        return y
    

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

        self.replace = SubstitutueBlock()
        self.adapt_brust_pool = adapt_burst_pooling(num_features, 8)

           
    def forward(self, burst):

        b, n ,c, h, w = burst.size()
        burst = burst[0]
                
              
        burst_feat = self.conv1(burst)
        burst_feat = self.align(burst_feat)


        burst_feat = self.replace(burst_feat)

        burst_feat = self.adapt_brust_pool(burst_feat)        
        b, f, H, W = burst_feat.size()
        burst_feat = self.back_projection1(burst_feat)
        burst_feat = burst_feat.view(1, -1, H, W)

        burst_feat = self.up1(burst_feat)
        burst_feat = burst_feat.view(-1, f, 2*H, 2*W)

        burst_feat = self.back_projection2(burst_feat)
        burst_feat = burst_feat.view(1, -1, 2*H, 2*W)

        burst_feat = self.up2(burst_feat)
        burst_feat = burst_feat.view(-1, f, 4*H, 4*W)

        burst_feat = self.up3(burst_feat) 
        burst_feat = self.out_conv(burst_feat) 

        return burst_feat
    
    def training_step(self, train_batch, batch_idx):
        x, y, flow_vectors, meta_info = train_batch
        pred = self.forward(x)
        pred = pred.clamp(0.0, 1.0)
        loss = self.train_loss(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, flow_vectors, meta_info = val_batch
        pred = self.forward(x)
        pred = pred.clamp(0.0, 1.0)
        PSNR = self.valid_psnr(pred, y)
        self.log('every_psnr', PSNR, on_step=True, on_epoch=True, prog_bar=True)
        return PSNR

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        PSNR = torch.stack(outs).mean()
        self.log('val_psnr', PSNR, on_step=False, on_epoch=True, prog_bar=True)

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
if __name__ == '__main__':

    net = Base_Model().cuda()
    x = torch.randn((1, 14, 4, 24, 24)).cuda()
    y = net(x)
    print(y.shape)


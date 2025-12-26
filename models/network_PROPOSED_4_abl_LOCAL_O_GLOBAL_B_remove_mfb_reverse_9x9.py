import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from collections import OrderedDict
import numbers

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


############################################
# AttentionLayerNorm(Channel)
############################################
class BiasFree_LayerNorm(nn.Module):
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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

############################################
# AttentionLayerNorm(Spatial)
############################################
class Spatial_Attention_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            # channels_first
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

############################################
# SpatialAttention
############################################
class Spatial_Attention(nn.Module):
    """Spatial attention used in Conv2Former"""
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.norm = Spatial_Attention_LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x

############################################
# Simple Channel_Gate
############################################
class Channel_Gate(nn.Module):
    """Channel attention used in CBAM"""
    def __init__(self, in_planes, ratio= 18):
        super(Channel_Gate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


############################################
# Gated-Dconv Feed-Forward Network (GDFN)
############################################
class FeedForward(nn.Module):
    """FFN used in Restormer"""
    def __init__(self, dim, ffn_expansion_factor,outdim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2,  kernel_size=1, stride=1, padding='same', bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding='same',
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, outdim, kernel_size=1, stride=1, padding='same', bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

############################################
# RDSAB(Residual Depth-wise Spatial Attention Block)
############################################
class RDSAB(nn.Module):
    """
    (Multi-scale) Residual Depth-wise Spatial Attention Block(RDSAB)
    """
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,

                 drop_path_rate=0.3,
                 layer_scale_init_value=1e-6,
                 ffn_expansion_factor=3.0,
                 bias=True,
                 ):
        super(RDSAB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        # self.dwconv_5x5 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True),
        #                                 nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=5, stride=1, padding=2, groups=mid_channels // 4, bias=True)
        #                                 )
        # self.dwconv_7x7 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True),
        #                                 nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=7, stride=1, padding=3, groups=mid_channels // 4, bias=True)
        #                                 )
        # self.dwconv_9x9 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True),
        #                                 nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=9, stride=1,padding=4, groups=mid_channels // 4, bias=True)
        #                                 )
        # self.dwconv_11x11 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True),
        #                                   nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=11, stride=1,padding=5, groups=mid_channels // 4, bias=True)
        #                                 )
        # self.channel_gate = Channel_Gate(mid_channels)
        # self.concat_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, groups=in_channels, bias=bias)
        # )

        self.spatial_attn = Spatial_Attention(mid_channels)

        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        # self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # self.l_relu = activation('lrelu', neg_slope=0.05)

        self.ffn = FeedForward(dim = in_channels, ffn_expansion_factor = ffn_expansion_factor, outdim=out_channels, bias = True)

    def forward(self, x):
        out = x
        # out9 = self.dwconv_9x9(out)
        # out7 = self.dwconv_7x7(out)
        # out5 = self.dwconv_5x5(out)
        # out11 = self.dwconv_11x11(out)

        # out = torch.cat((out9, out7, out5, out11), dim=1)
        # out = self.l_relu(out)

        # out = self.channel_gate(out) * out
        # out = self.concat_conv(out)
        # out = self.l_relu(out)

        # out = self.conv1(out)
        # out = self.l_relu(out)

        # Spatial Attention + layer scale
        out = out + self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.spatial_attn(out)
        # out = x + out

        out_ffn = self.ffn(out)
        out = out + out_ffn
        return out

class Channel_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Channel_Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv_pwconv = nn.Conv2d(dim, dim * 3, kernel_size=1, stride=1, padding='same', bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding='same', groups=dim * 3,
                                    bias=bias)

        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding='same', bias=bias)
        self.project_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4, groups=dim, bias=bias),
                                         nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding='same', bias=bias))
    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_pwconv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

############################################
# RDCAB(Residual Depth-wise Channel Attention Block)
############################################
    """
    Residual Depth-wise Channel Attention Block(RDCAB)
    """
class RDCAB(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, outdim, bias, LayerNorm_type='WithBias'):
        super(RDCAB, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Channel_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, outdim, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

############################################
# RDDA-IR(Residual Depth-wise Dual(spatial+channel) Attention - Image Restoration or Image Denoising)
############################################
    """
    Residual Depth-wise Dual(spatial+channel) Attention Network(RDDA)
    """

class RDDA(nn.Module):
    def __init__(self,
                 in_chans=3,
                 feature_channels=72,
                 patch_size=96,
                 upscale=1,
                 norm_type='WithBias',
                 num_heads=8,
                 ffn_expansion_factor=3.0,
                 bias=True):
        super(RDDA, self).__init__()

        self.conv_1 = conv_layer(in_chans, feature_channels, kernel_size=3)

        # self.block_1 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        # self.block_2 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        # self.block_3 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        # self.block_4 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        # self.block_5 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        # self.block_6 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        # self.block_7 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        # self.block_8 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)

        self.block_2 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        self.block_1 = RDCAB(dim=feature_channels,
                                        num_heads=num_heads,
                                        ffn_expansion_factor=ffn_expansion_factor,
                                        outdim=feature_channels,
                                        bias=bias,
                                        LayerNorm_type=norm_type)
        self.block_4 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        self.block_3 = RDCAB(dim=feature_channels,
                                        num_heads=num_heads,
                                        ffn_expansion_factor=ffn_expansion_factor,
                                        outdim=feature_channels,
                                        bias=bias,
                                        LayerNorm_type=norm_type)
        self.block_6 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        self.block_5 = RDCAB(dim=feature_channels,
                                        num_heads=num_heads,
                                        ffn_expansion_factor=ffn_expansion_factor,
                                        outdim=feature_channels,
                                        bias=bias,
                                        LayerNorm_type=norm_type)
        self.block_8 = RDSAB(in_channels=feature_channels, ffn_expansion_factor=ffn_expansion_factor)
        self.block_7 = RDCAB(dim=feature_channels,
                                        num_heads=num_heads,
                                        ffn_expansion_factor=ffn_expansion_factor,
                                        outdim=feature_channels,
                                        bias=bias,
                                        LayerNorm_type=norm_type)
        
        # self.block_1 = RDCAB(dim=feature_channels,
        #                                 num_heads=num_heads,
        #                                 ffn_expansion_factor=ffn_expansion_factor,
        #                                 outdim=feature_channels,
        #                                 bias=bias,
        #                                 LayerNorm_type=norm_type)
        # self.block_2 = RDCAB(dim=feature_channels,
        #                                 num_heads=num_heads,
        #                                 ffn_expansion_factor=ffn_expansion_factor,
        #                                 outdim=feature_channels,
        #                                 bias=bias,
        #                                 LayerNorm_type=norm_type)
        # self.block_3 = RDCAB(dim=feature_channels,
        #                                 num_heads=num_heads,
        #                                 ffn_expansion_factor=ffn_expansion_factor,
        #                                 outdim=feature_channels,
        #                                 bias=bias,
        #                                 LayerNorm_type=norm_type)
        # self.block_4 = RDCAB(dim=feature_channels,
        #                                 num_heads=num_heads,
        #                                 ffn_expansion_factor=ffn_expansion_factor,
        #                                 outdim=feature_channels,
        #                                 bias=bias,
        #                                 LayerNorm_type=norm_type)
        # self.block_5 = RDCAB(dim=feature_channels,
        #                                 num_heads=num_heads,
        #                                 ffn_expansion_factor=ffn_expansion_factor,
        #                                 outdim=feature_channels,
        #                                 bias=bias,
        #                                 LayerNorm_type=norm_type)
        # self.block_6 = RDCAB(dim=feature_channels,
        #                                 num_heads=num_heads,
        #                                 ffn_expansion_factor=ffn_expansion_factor,
        #                                 outdim=feature_channels,
        #                                 bias=bias,
        #                                 LayerNorm_type=norm_type)
        # self.block_7 = RDCAB(dim=feature_channels,
        #                                 num_heads=num_heads,
        #                                 ffn_expansion_factor=ffn_expansion_factor,
        #                                 outdim=feature_channels,
        #                                 bias=bias,
        #                                 LayerNorm_type=norm_type)
        # self.block_8 = RDCAB(dim=feature_channels,
        #                                 num_heads=num_heads,
        #                                 ffn_expansion_factor=ffn_expansion_factor,
        #                                 outdim=feature_channels,
        #                                 bias=bias,
        #                                 LayerNorm_type=norm_type)

        self.conv_2 = conv_layer(feature_channels, in_chans, kernel_size=3)
        # self.upsampler = pixelshuffle_block(feature_channels, in_chans, upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)
        out_b7 = self.block_7(out_b6)
        out_b8 = self.block_8(out_b7)
        output = self.conv_2(out_b8) + x

        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import time
from thop import profile
from thop import clever_format

if __name__ == '__main__':

    upscale = 1
    window_size = 8
    height = 128
    width = 128
    # height = (1024 // upscale // window_size + 1) * window_size
    # width = (720 // upscale // window_size + 1) * window_size


    start_time = time.time()
    # 모델 및 입력 데이터 설정
    model = RDDA().cuda()
    x = torch.ones((1, 3, height, width)).cuda()
    y = model(x)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"input shape : {x.shape}")
    print(f"model shape : {y.shape}")

    ##########################################################################################################
    # GPU 시간 측정 시작
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    # 모델 추론
    yy = model(x)
    # GPU 시간 측정 종료
    end_event.record()
    torch.cuda.synchronize()
    # 소요 시간 계산
    time_taken = start_event.elapsed_time(end_event)
    print(f"Elapsed time on GPU: {time_taken} ms -> {time_taken * 1e-3} s")
    ##########################################################################################################

    print(f"count parameter : {count_parameters(model)}")

    flops, params = profile(model, inputs=(x,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function, ProfilerActivity
from torch.profiler import profile as tprofile

class Up(nn.Module):

    def __init__(self, nc, bias):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=nc, out_channels=nc, kernel_size=2, stride=2, bias=bias)
        self.nc = nc
    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3

    def flops(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W) -- this is x1 in forward()
        Returns:
            int: multiply–add FLOPs for the ConvTranspose2d
        """
        _,_,H,W = x.shape
        flops = 0

        # up
        flops += H * W * self.nc * self.nc * 2 * 2


        return flops


## Spatial Attention
class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.inplanes = in_planes
        self.out_channels = out_planes
        self.kernel_size = kernel_size
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

    def flops(self, x):
        _,_,H,W = x.shape
        flops = 0

        #conv
        flops += H * W * self.inplanes * self.out_channels * self.kernel_size * self.kernel_size

        return flops


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

    def flops(self, x):
        _,C,H,W = x.shape
        flops = 0

        #spatial
        flops += self.spatial.flops(x)

        # scale x x
        flops += H * W * C

        return flops

## Channel Attention Layer
class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
                nn.Sigmoid()
        )
        self.nc =nc
        self.reduction = reduction

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

    def flops(self, x):
        _,_,H,W = x.shape
        flops = 0

        #conv_du
        flops += self.nc * (self.nc // self.reduction) * 1
        flops += (self.nc // self.reduction) * self.nc * 1

        # x * y
        flops += H * W * self.nc

        return flops


class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RAB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = 3
        stride = 1
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)
        self.sab = SAB()

    def forward(self, x):
        x1 = x + self.res(x)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x + x4

        x5 = self.sab(x4_1)
        x5_1 = x + x5

        return x5_1

    def flops(self, x):
        _,_,H,W = x.shape
        flops = 0

        # res(x)
        flops += H * W * self.in_channels * self.out_channels * 3 * 3
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # res(x1)
        flops += H * W * self.in_channels * self.out_channels * 3 * 3
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # res(x2)
        flops += H * W * self.in_channels * self.out_channels * 3 * 3
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # res(x3_1)
        flops += H * W * self.in_channels * self.out_channels * 3 * 3
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        #sab
        flops += self.sab.flops(x)

        return flops


class HDRAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(HDRAB, self).__init__()
        kernel_size = 3
        reduction = 8

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cab = CAB(in_channels, reduction, bias)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)

        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv_tail = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)

    def forward(self, y):
        y1 = self.conv1(y)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y

        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1+y3)
        y6_1 = y6 + y4_1

        y7 = self.conv1_1(y6_1+y2_1)
        y7_1 = self.relu1_1(y7)
        y8 = self.conv_tail(y7_1+y1)
        y8_1 = y8 + y6_1

        y9 = self.cab(y8_1)
        y9_1 = y + y9

        return y9_1

    def flops(self, x):
        _,_,H,W = x.shape
        flops = 0

        # conv1
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # conv2
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # conv3
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # conv4
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # conv3_1
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # conv2_1
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # conv1_1
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # conv_tail
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        # cab
        flops += self.cab.flops(x)


        return flops


class DRANet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, bias=True):
        super(DRANet, self).__init__()
        kernel_size = 3

        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nc = nc

        self.conv_head = nn.Conv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.rab = RAB(nc, nc, bias)

        self.hdrab = HDRAB(nc, nc, bias)

        self.conv_tail = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.dual_tail = nn.Conv2d(2*out_nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.down = nn.Conv2d(nc, nc, kernel_size=2, stride=2, bias=bias)

        self.up = Up(nc, bias)

    def forward(self, x):
        x1 = self.conv_head(x)
        x2 = self.rab(x1)
        x2_1 = self.down(x2)
        x3 = self.rab(x2_1)
        x3_1 = self.down(x3)
        x4 = self.rab(x3_1)
        x4_1 = self.up(x4, x3)
        x5 = self.rab(x4_1 + x3)
        x5_1 = self.up(x5, x2)
        x6 = self.rab(x5_1 + x2)
        x7 = self.conv_tail(x6 + x1)
        X = x - x7

        y1 = self.conv_head(x)
        y2 = self.hdrab(y1)
        y3 = self.hdrab(y2)
        y4 = self.hdrab(y3)
        y5 = self.hdrab(y4 + y3)
        y6 = self.hdrab(y5 + y2)
        y7 = self.conv_tail(y6 + y1)
        Y = x -y7

        z1 = torch.cat([X, Y], dim=1)
        z = self.dual_tail(z1)
        Z = x - z

        return Z

    def flops(self, x):
        _, _, H, W = x.shape
        flops = 0

        # 1) conv_head
        flops += H * W * self.in_nc * self.nc * 3 * 3

        # 2) RAB 레벨1
        feat = torch.empty((1, self.nc, H, W), device='meta')
        flops += self.rab.flops(feat)

        # 3) 다운샘플 → RAB 레벨2
        flops += H * W * self.nc * self.nc
        H2, W2 = H // 2, W // 2
        feat2 = torch.empty((1, self.nc, H2, W2), device='meta')
        flops += self.rab.flops(feat2)

        # 4) 다시 다운 → RAB 레벨3
        flops += H2 * W2 * self.nc * self.nc
        H3, W3 = H2 // 2, W2 // 2
        feat3 = torch.empty((1, self.nc, H3, W3), device='meta')
        flops += self.rab.flops(feat3)

        # 5) 업샘플 → RAB 레벨4
        flops += self.up.flops(feat3)
        flops += self.rab.flops(feat2)

        # 6) 업샘플 → RAB 레벨5
        flops += self.up.flops(feat2)
        flops += self.rab.flops(feat)

        # 7) conv_tail
        flops += H * W * self.nc * self.out_nc * 3 * 3

        # 8) conv_head
        flops += H * W * self.in_nc * self.nc * 3 * 3

        # 9) HDRAB 브랜치
        for _ in range(5):
            flops += self.hdrab.flops(feat)

        # 10) conv_tail
        flops += H * W * self.nc * self.out_nc * 3 * 3

        # --- Dual tail
        flops += H * W * (2 * self.out_nc) * self.out_nc * 3 * 3

        return flops


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import time
from thop import profile
from thop import clever_format

if __name__ == '__main__':


    start_time = time.time()
    # 모델 및 입력 데이터 설정
    model =  DRANet().cuda()
    print(model)
    x = torch.ones((1, 3, 128, 128)).cuda()
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
    print(f"Elapsed time on GPU: {time_taken} ms -> {time_taken * 1e-3} s")  # 밀리초(ms) -> 초(s) 변환
    ##########################################################################################################

    # with tprofile(activities=[
    #         ProfilerActivity.CPU, 
    #         ProfilerActivity.CUDA
    #     ],
    #     profile_memory=True, record_shapes=True) as prof:

    #     with record_function("model_inference"):
    #         model(x)

    # print(f'memory_usage : {prof.key_averages().table(sort_by="self_cuda_memory_usage")}')

    print(f"count parameter : {count_parameters(model)}")

    flops, params = profile(model, inputs=(x,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs2: {model.flops(x) / 1e9}")
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
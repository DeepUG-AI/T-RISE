import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utility
# =========================
def make_gn(num_channels, max_groups=8):
    for g in [8, 4, 2, 1]:
        if num_channels % g == 0 and g <= max_groups:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


# =========================
# Lightweight Blocks
# =========================
class DSConv(nn.Module):
    """Depthwise Separable Conv"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, act=True):
        super().__init__()
        padding = dilation if kernel_size == 3 else 0

        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_ch, bias=False
        )
        self.dw_norm = make_gn(in_ch)
        self.dw_act = nn.GELU()

        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_norm = make_gn(out_ch)
        self.use_act = act
        self.pw_act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.dw_norm(x)
        x = self.dw_act(x)
        x = self.pw(x)
        x = self.pw_norm(x)
        x = self.pw_act(x)
        return x


class DSConvBlock(nn.Module):
    """
    DSConv + DSConv + residual
    """
    def __init__(self, in_ch, out_ch, drop_path=0.0, expand_ratio=1.0):
        super().__init__()
        mid_ch = int(out_ch * expand_ratio)

        self.conv1 = DSConv(in_ch, mid_ch, kernel_size=3, act=True)
        self.conv2 = DSConv(mid_ch, out_ch, kernel_size=3, act=False)

        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                make_gn(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop_path(x)
        x = x + identity
        x = self.act(x)
        return x


class LiteDecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch, drop_path=0.0):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            make_gn(out_ch)
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False),
            make_gn(out_ch)
        )
        self.act = nn.GELU()
        self.block = DSConvBlock(out_ch, out_ch, drop_path=drop_path)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.x_proj(x)
        skip = self.skip_proj(skip)
        x = self.act(x + skip)
        x = self.block(x)
        return x


class LiteContextBlock(nn.Module):
    """
    Lightweight context module:
    - 1x1
    - depthwise dilated 3x3
    - image pooling
    """
    def __init__(self, in_ch, out_ch, dilation=4):
        super().__init__()
        inter_ch = max(out_ch // 4, 16)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False),
            make_gn(inter_ch),
            nn.GELU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=dilation,
                      dilation=dilation, groups=in_ch, bias=False),
            make_gn(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False),
            make_gn(inter_ch),
            nn.GELU()
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False),
            make_gn(inter_ch),
            nn.GELU()
        )

        self.project = nn.Sequential(
            nn.Conv2d(inter_ch * 3, out_ch, kernel_size=1, bias=False),
            make_gn(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        bp = self.pool(x)
        bp = F.interpolate(bp, size=(h, w), mode='bilinear', align_corners=False)
        out = torch.cat([b1, b2, bp], dim=1)
        out = self.project(out)
        return out


class StripPoolingLite(nn.Module):
    def __init__(self, in_ch, pool_ch=None):
        super().__init__()
        pool_ch = pool_ch or max(in_ch // 4, 16)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, pool_ch, kernel_size=1, bias=False),
            make_gn(pool_ch),
            nn.GELU()
        )

        self.conv_h = nn.Sequential(
            nn.Conv2d(pool_ch, pool_ch, kernel_size=(1, 3), padding=(0, 1),
                      groups=pool_ch, bias=False),
            make_gn(pool_ch),
            nn.GELU()
        )

        self.conv_w = nn.Sequential(
            nn.Conv2d(pool_ch, pool_ch, kernel_size=(3, 1), padding=(1, 0),
                      groups=pool_ch, bias=False),
            make_gn(pool_ch),
            nn.GELU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(pool_ch, in_ch, kernel_size=1, bias=False),
            make_gn(in_ch)
        )
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        h, w = x.shape[-2:]

        xr = self.reduce(x)

        fh = F.adaptive_avg_pool2d(xr, (h, 1))
        fh = self.conv_h(fh)
        fh = F.interpolate(fh, size=(h, w), mode='bilinear', align_corners=False)

        fw = F.adaptive_avg_pool2d(xr, (1, w))
        fw = self.conv_w(fw)
        fw = F.interpolate(fw, size=(h, w), mode='bilinear', align_corners=False)

        out = self.fuse(fh + fw)
        return self.act(identity + out)


# =========================
# Main Network
# =========================
class SwiftCrackNetV5(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        c_list= [24, 40, 64, 80, 96],
        pretrained_path='',
        drop_path_rate=0.03,
        use_sigmoid=True
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        ch1, ch2, ch3, ch4, bottleneck_ch = c_list
        dpr = [drop_path_rate * i / 6 for i in range(7)]

        # Encoder
        self.enc1 = DSConvBlock(input_channels, ch1, drop_path=dpr[0])
        self.enc2 = DSConvBlock(ch1, ch2, drop_path=dpr[1])
        self.enc3 = DSConvBlock(ch2, ch3, drop_path=dpr[2])
        self.enc4 = DSConvBlock(ch3, ch4, drop_path=dpr[3])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_in = DSConvBlock(ch4, bottleneck_ch, drop_path=dpr[4])
        self.context = LiteContextBlock(bottleneck_ch, bottleneck_ch, dilation=4)
        self.strip_pool = StripPoolingLite(bottleneck_ch, pool_ch=max(bottleneck_ch // 4, 16))
        self.bottleneck_refine = DSConvBlock(bottleneck_ch, bottleneck_ch, drop_path=dpr[5])

        # Decoder
        self.dec3 = LiteDecoderBlock(bottleneck_ch, ch3, ch3, drop_path=dpr[4])
        self.dec2 = LiteDecoderBlock(ch3, ch2, ch2, drop_path=dpr[3])
        self.dec1 = LiteDecoderBlock(ch2, ch1, ch1, drop_path=dpr[2])

        self.head = nn.Sequential(
            DSConvBlock(ch1, ch1, drop_path=dpr[6]),
            nn.Conv2d(ch1, num_classes, kernel_size=1)
        )

        self.pretrained_path = pretrained_path

    def forward(self, x):
        x1 = self.enc1(x)
        p1 = self.pool(x1)

        x2 = self.enc2(p1)
        p2 = self.pool(x2)

        x3 = self.enc3(p2)
        p3 = self.pool(x3)

        x4 = self.enc4(p3)
        p4 = self.pool(x4)

        b = self.bottleneck_in(p4)
        b = self.context(b)
        b = self.strip_pool(b)
        b = self.bottleneck_refine(b)

        d3 = self.dec3(b, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        out = self.head(d1)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.use_sigmoid and out.shape[1] == 1:
            out = torch.sigmoid(out)
        return out

"""
IFNet — RIFE v4.6 architecture (small variant).

Derived from hzwer/Practical-RIFE (https://github.com/hzwer/Practical-RIFE),
licensed MIT. The class definitions below match the published checkpoints
shipped by Practical-RIFE; the architecture is reproduced here so the .pkl
weights can be loaded without pulling the full Practical-RIFE repo.

Two checkpoint variants are supported:

  flownet_small.pkl  (~12 MB, default for morpheus-cam)
    - 3 IFBlocks with in_planes=11, c=90
    - convblock: 4 conv pairs (convblock0..3)
    - Separate deconv heads for flow (4ch) and mask (1ch)

  flownet.pkl  (~42 MB)
    - The full RIFE_HDv3 model with UNet + ContextNet
    - Heavier; the small variant is enough for morpheus-cam's 9-frame interpolation

License: MIT (c) 2020-2024 hzwer
Reference: https://github.com/hzwer/Practical-RIFE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


_backwarp_grid: dict = {}


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in _backwarp_grid:
        H, W = tenFlow.shape[2], tenFlow.shape[3]
        hg = torch.linspace(-1., 1., W, device=tenFlow.device).view(1, 1, 1, W).expand(tenFlow.shape[0], -1, H, -1)
        vg = torch.linspace(-1., 1., H, device=tenFlow.device).view(1, 1, H, 1).expand(tenFlow.shape[0], -1, -1, W)
        _backwarp_grid[k] = torch.cat([hg, vg], 1)
    f = torch.cat([
        tenFlow[:, 0:1] / ((tenInput.shape[3] - 1.) / 2.),
        tenFlow[:, 1:2] / ((tenInput.shape[2] - 1.) / 2.),
    ], 1)
    g = (_backwarp_grid[k].to(f.dtype) + f).permute(0, 2, 3, 1)
    return F.grid_sample(tenInput, g, mode='bilinear', padding_mode='border', align_corners=True)


def _conv(ip, op, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(ip, op, k, s, p, bias=True), nn.PReLU(op))


class IFBlock(nn.Module):
    """
    Layer layout matching flownet_small.pkl:
      conv0          : Sequential(conv(11,45,3,2,1), conv(45,90,3,2,1))
      convblock0..3  : Sequential(conv(90,90), conv(90,90)) each
      conv1 (flow)   : deconv(90,45) → PReLU → deconv(45,4)
      conv2 (mask)   : deconv(90,45) → PReLU → deconv(45,1)
    """
    def __init__(self, in_planes: int = 11, c: int = 90):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2,    c,      3, 2, 1),
        )
        for i in range(4):
            setattr(self, f"convblock{i}", nn.Sequential(_conv(c, c), _conv(c, c)))

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c,        c // 2, 4, 2, 1), nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2,   4,      4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c,        c // 2, 4, 2, 1), nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2,   1,      4, 2, 1),
        )

    def forward(self, x: torch.Tensor, flow: torch.Tensor, scale: float = 1.) -> tuple:
        x    = F.interpolate(x,    scale_factor=1. / scale, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor=1. / scale, mode='bilinear', align_corners=False, recompute_scale_factor=False) * (1. / scale)
        x = torch.cat([x, flow], 1)

        x = self.conv0(x)
        for i in range(4):
            x = getattr(self, f"convblock{i}")(x) + x

        flow_out = self.conv1(x)
        mask_out = self.conv2(x)

        flow_out = F.interpolate(flow_out, scale_factor=scale, mode='bilinear', align_corners=False, recompute_scale_factor=False) * scale
        mask_out = F.interpolate(mask_out, scale_factor=scale, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return flow_out, mask_out


class IFNet(nn.Module):
    """
    RIFE small model. Input shape: (B, 6, H, W) = cat(img0, img1) in [0,1].
    Output: (merged, flow, mask).
    """
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(11, c=90)
        self.block1 = IFBlock(11, c=90)
        self.block2 = IFBlock(11, c=90)

    def forward(
        self,
        x: torch.Tensor,
        scale_list: list = [4., 2., 1.],
        training: bool = False,
    ):
        B, _, H, W = x.shape
        img0 = x[:, :3]
        img1 = x[:, 3:]

        warped0 = img0.clone()
        warped1 = img1.clone()
        flow = torch.zeros(B, 4, H, W, device=x.device, dtype=x.dtype)
        mask = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        for block, scale in zip([self.block0, self.block1, self.block2], scale_list):
            base = torch.cat([warped0, warped1, mask], 1)
            flow_d, mask_d = block(base, flow, scale)
            flow = flow + flow_d
            mask = mask + mask_d
            warped0 = warp(img0, flow[:, :2])
            warped1 = warp(img1, flow[:, 2:4])

        mask   = torch.sigmoid(mask)
        merged = warped0 * mask + warped1 * (1. - mask)
        return merged, flow, mask

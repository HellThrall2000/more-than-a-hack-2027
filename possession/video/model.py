"""A small fully-convolutional ball locator: heat map + sub-cell offset (CenterNet-style)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import STRIDE


def _block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    """Input (B, C, H, W) of stacked grey frames; output (B, 3, ceil(H/4), ceil(W/4)).

    Channel 0 is the heat-map logit, channels 1-2 the offset of the ball from the cell centre in
    cells. Fully convolutional, so any crop size works -- a larger search window is just a larger
    input.
    """

    def __init__(self, in_ch=3, width=48):
        super().__init__()
        self.body = nn.Sequential(_block(in_ch, 24), _block(24, 32, 2), _block(32, width, 2),
                                  _block(width, width, 1, 2), _block(width, width, 1, 4),
                                  _block(width, width, 1, 1))
        self.head = nn.Conv2d(width, 3, 1)

    def forward(self, x):
        return self.head(self.body(normalise(x)))


def normalise(x):
    """Per-sample standardisation, so exposure and camera brightness do not matter."""
    x = x.float()
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    return (x - mean) / (std + 1e-3)


def out_size(n):
    """Output cells along one axis of a crop n pixels wide (two stride-2 convolutions)."""
    return ((n + 1) // 2 + 1) // 2


def make_targets(uv, present, size, sigma):
    """Gaussian heat-map target (B, 1, h, h) and (cell, offset) targets for the present balls."""
    n, h = len(uv), out_size(size)
    grid = torch.arange(h, dtype=torch.float32)
    heat = torch.zeros(n, 1, h, h)
    cell = torch.zeros(n, 2, dtype=torch.long)
    off = torch.zeros(n, 2)
    if present.any():
        idx = present.nonzero(as_tuple=True)[0]
        u = uv[idx] / STRIDE                                   # continuous position in cells
        c = u.round().clamp(0, h - 1)
        cell[idx], off[idx] = c.long(), u - c
        gx = torch.exp(-(grid[None, :] - u[:, :1]) ** 2 / (2 * sigma ** 2))
        gy = torch.exp(-(grid[None, :] - u[:, 1:]) ** 2 / (2 * sigma ** 2))
        g = gy[:, :, None] * gx[:, None, :]
        g[torch.arange(len(idx)), cell[idx, 1], cell[idx, 0]] = 1.0
        heat[idx, 0] = g
    return heat, cell, off


def loss_fn(out, uv, present, size, sigma):
    """CenterNet focal loss on the heat map + L1 on the offset at the ball cell."""
    heat_t, cell, off_t = make_targets(uv, present, size, sigma)
    heat_t, cell, off_t = heat_t.to(out.device), cell.to(out.device), off_t.to(out.device)
    p = torch.sigmoid(out[:, :1]).clamp(1e-4, 1 - 1e-4)
    pos = heat_t.eq(1.0).float()
    pos_loss = -((1 - p) ** 2 * torch.log(p) * pos).sum()
    neg_loss = -((1 - heat_t) ** 4 * p ** 2 * torch.log(1 - p) * (1 - pos)).sum()
    n_pos = pos.sum().clamp(min=1.0)
    loss = (pos_loss + neg_loss) / n_pos
    if present.any():
        idx = present.to(out.device).nonzero(as_tuple=True)[0]
        pred = out[idx, 1:, cell[idx, 1], cell[idx, 0]]
        loss = loss + F.l1_loss(pred, off_t[idx])
    return loss


@torch.no_grad()
def decode(out):
    """(confidence, u, v) per sample: the strongest cell, refined by its offset, in input pixels."""
    heat = torch.sigmoid(out[:, 0])
    b, h, w = heat.shape
    conf, flat = heat.view(b, -1).max(dim=1)
    cy, cx = flat // w, flat % w
    rows = torch.arange(b)
    ox, oy = out[rows, 1, cy, cx], out[rows, 2, cy, cx]
    return conf.cpu().numpy(), ((cx.float() + ox) * STRIDE).cpu().numpy(), ((cy.float() + oy) * STRIDE).cpu().numpy()


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")      # ROCm builds report cuda
    return torch.device(name)

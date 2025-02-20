from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["QLinear"]


def pack4to8(x: Tensor):
    return 16 * x[..., : x.shape[-1] // 2] + x[..., x.shape[-1] // 2 :]


def pack8to4(x: Tensor):
    return torch.cat([x // 16, x % 16], dim=-1)


def quantize_dequantize(x, scale, zero, maxq, eps=1e-9):
    q = torch.clamp(torch.round(x / scale.clamp_min(eps) + zero), 0, maxq)
    return scale * (q - zero)


def quantize(x, scale, zero, maxq, eps=1e-9):
    q = torch.clamp(torch.round(x / scale.clamp_min(eps) + zero), 0, maxq)
    return q


def dequantize(x, scale, zero):
    return scale * (x - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        norm=2.0,
        grid=100,
        maxshrink=0.8,
        reserved_bins: int = 0,
    ):
        self.bits = bits
        self.maxq = torch.tensor(2**bits - 1 - reserved_bins)
        self.perchannel = perchannel
        self.sym = sym
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        xmin = x.min(1).values
        xmax = x.max(1).values

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = xmin == xmax
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize_dequantize(self, x):
        if self.ready():
            return quantize_dequantize(x, self.scale, self.zero, self.maxq)
        return x

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def dequantize(self, x):
        if self.ready():
            return dequantize(x, self.scale, self.zero)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def dequantize_linear_weight(
    qweight: torch.Tensor, 
    scale: torch.Tensor, 
    zero: torch.Tensor, 
    perm: Optional[torch.Tensor] = None, 
):
    scale = scale.view(qweight.shape[0], -1, 1)
    zero = zero.view(qweight.shape[0], -1, 1)
    num_groups = scale.shape[1]
    weight = dequantize(qweight.view(qweight.shape[0], num_groups, -1), scale, zero).view_as(qweight)
    if perm is not None:
        invperm = perm.argsort()
        weight =weight[:, invperm]
    return weight   


def get_relative_mse_error(q: torch.Tensor, w: torch.Tensor, H: torch.Tensor):
    delta = q - w
    try:
        return (delta).mm(H).mul(delta).mean() / (w.mm(H).mul(w).mean() + 1e-6)
    except:
        return 

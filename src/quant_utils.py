import math
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


FP8_GROUP_SIZE = 128
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


class QuantizationScale(Enum):
    ABSMAX = "absmax"
    MSE = "mse"


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


class Quantizer:

    def __init__(self):
        super().__init__()

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        # Scale search parameters
        quantization_scale: str = "absmax",
        scale_search_iters: int = 100,
    ):
        self.bits = bits
        self.maxq = 2**bits - 1
        self.perchannel = perchannel
        self.sym = sym
        # Scale search parameters
        self.quantization_scale = QuantizationScale(quantization_scale)
        self.scale_search_iters = scale_search_iters

    def get_scale_and_zero(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_rows = x.shape[0]

        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        xmin = x.min(dim=1, keepdim=True).values
        xmax = x.max(dim=1, keepdim=True).values

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = xmin == xmax
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / self.maxq
        if self.sym:
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)

        if self.quantization_scale == QuantizationScale.MSE:
            for _ in range(self.scale_search_iters):
                q = quantize(x, scale, zero, self.maxq)
                delta = q - zero
                scale = x.mul(delta).mean(dim=1, keepdim=True) / delta.pow(2).mean(dim=1, keepdim=True)

        if not self.perchannel:
            scale = scale.expand((num_rows, 1))
            zero = zero.expand((num_rows, 1))

        return scale, zero


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
        weight = weight[:, invperm]
    return weight


def get_relative_mse_error(q: torch.Tensor, w: torch.Tensor, H: Optional[torch.Tensor] = None):
    delta = q - w
    if H is None:
        return delta.pow(2).mean() / w.pow(2).mean()
    else:
        return (delta).mm(H).mul(delta).mean() / (w.mm(H).mul(w).mean() + 1e-6)


def dequantize_weight_from_fp8(W, s):
    g = FP8_GROUP_SIZE
    # Dequantize weight
    d_out, d_in = W.shape
    # Pad weight if needed
    pad_out = math.ceil(d_out / g) * g - d_out
    pad_in = math.ceil(d_in / g) * g - d_in
    W = F.pad(W, (0, pad_in, 0, pad_out))
    d_out_pad, d_in_pad = W.shape

    W = W.view(d_out_pad // g, g, d_in_pad // g, g)
    s = s.view(d_out_pad // g, 1, d_in_pad // g, 1)
    W = (W * s).view(d_out_pad, d_in_pad)

    # Remove padding
    W = W[:d_out, :d_in]
    return W


def dequantize_state_dict(state_dict: dict[str, torch.Tensor], dtype: torch.dtype = torch.float16) -> None:
    state_dict_keys = list(state_dict.keys())
    # Dequantize
    for k in state_dict_keys:
        if k.endswith("scale_inv"):
            layer_name, _ = k.rsplit(".", 1)

            W = state_dict[f"{layer_name}.weight"].to(dtype)
            s = state_dict[f"{layer_name}.weight_scale_inv"].to(dtype)

            state_dict[f"{layer_name}.weight"] = dequantize_weight_from_fp8(W, s)
            del state_dict[f"{layer_name}.weight_scale_inv"]


def can_dequantize_from_fp8(state_dict: dict[str, torch.Tensor]) -> bool:
    for k, v in state_dict.items():
        if v.dtype in FP8_DTYPES and f"{k}_scale_inv" not in state_dict:
            return False
    return True

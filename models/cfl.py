import torch
import torch.nn as nn
import torch.nn.functional as F


class CFL(nn.Module):
    def __init__(self):
        super(CFL, self).__init__()

    def forward(
        self, mask: torch.Tensor, original_img: torch.Tensor, out_img: torch.Tensor
    ) -> torch.Tensor:
        scale_factor = out_img.size(-1) / original_img.size(-1)
        mask_re = F.interpolate(mask, scale_factor=scale_factor)
        origin_re = F.interpolate(original_img, scale_factor=scale_factor)

        return torch.mul(mask_re, out_img) + torch.mul(1 - mask_re, origin_re)

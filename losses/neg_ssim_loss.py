import torch
import torch.nn as nn
from kornia.losses import ssim_loss

from data_model.neg_ssim_param import NegSSIMPram


class NegativeSSIMLoss(nn.Module):
    def __init__(self, neg_ssim_param: NegSSIMPram):
        super(NegativeSSIMLoss, self).__init__()
        self.window_size = neg_ssim_param.window_size

    def forward(self, out_feature: torch.Tensor, gt_feature: torch.Tensor) -> torch.Tensor:
        return ssim_loss(out_feature, gt_feature, self.window_size)

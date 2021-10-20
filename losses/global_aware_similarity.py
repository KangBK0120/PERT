from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_model.global_aware_sim_param import GlobalAwareSimParam


class GlobalAwareSimilarityLoss(nn.Module):
    def __init__(self, gs_loss_param: GlobalAwareSimParam):
        super(GlobalAwareSimilarityLoss, self).__init__()

        self.kernel_sizes = gs_loss_param.kernel_sizes
        self.l2 = nn.MSELoss()

    def forward(self, out_features: List[torch.Tensor], gt_features: List[torch.Tensor]):
        loss = 0

        for i in range(len(out_features)):
            B, C = out_features[i].size(0), out_features[i].size(1)
            for kernel_size in self.kernel_sizes:
                max_pool_size = out_features[i].size(2) // kernel_size
                out_F = F.normalize(
                    F.max_pool2d(out_features[i], max_pool_size, max_pool_size), dim=1
                )
                gt_F = F.normalize(
                    F.max_pool2d(gt_features[i], max_pool_size, max_pool_size), dim=1
                )

                for height in range(kernel_size):
                    for width in range(kernel_size):
                        loss += self.l2(
                            out_F * out_F[:, :, height, width].view(B, C, 1, 1),
                            gt_F * gt_F[:, :, height, width].view(B, C, 1, 1),
                        )

        return loss

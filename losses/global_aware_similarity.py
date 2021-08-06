from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler

from data_model.global_aware_sim_param import GlobalAwareSimParam


class GlobalAwareSimilarityLoss(nn.Module):
    def __init__(self, gs_loss_param: GlobalAwareSimParam):
        super(GlobalAwareSimilarityLoss, self).__init__()

        self.kernel_sizes = gs_loss_param.kernel_sizes
        self.l2 = nn.MSELoss()

    def forward(self, out_features: List[torch.Tensor], gt_features: List[torch.Tensor]):

        loss = 0
        batch_size = out_features[0].size(0)
        for i in range(len(out_features)):
            for kernel_size in self.kernel_sizes:
                sampler = SpatialCorrelationSampler(kernel_size=kernel_size)
                gamma_out = sampler(out_features[i], F.normalize(out_features[i]).clone())
                gamma_gt = sampler(gt_features[i], F.normalize(gt_features[i]))

                loss += self.l2(gamma_out, gamma_gt)
        return loss

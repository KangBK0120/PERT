import torch
import torch.nn as nn

from data_model.region_aware_sim_param import RegionAwareSimilarityParam


class RegionAwareSimilarityLoss(nn.Module):
    def __init__(self, rs_loss_param: RegionAwareSimilarityParam):
        super(RegionAwareSimilarityLoss, self).__init__()
        self.lambda_out = rs_loss_param.lambda_out
        self.lambda_p1 = rs_loss_param.lambda_p1
        self.lambda_p2 = rs_loss_param.lambda_p2

        self.beta_out = rs_loss_param.beta_out
        self.beta_p1 = rs_loss_param.beta_p1
        self.beta_p2 = rs_loss_param.beta_p2

        self.l1_loss = nn.L1Loss()

    def forward(self, p1_out, i1_gt, mask1_gt, p2_out, i2_gt, mask2_gt, i_out, i_gt, mask_gt):
        p1_loss = self.lambda_p1 * self.l1_loss(
            torch.mul(p1_out, mask1_gt), torch.mul(i1_gt, mask1_gt)
        ) + self.beta_p1 * self.l1_loss(
            torch.mul(p1_out, 1 - mask1_gt), torch.mul(i1_gt, 1 - mask1_gt)
        )

        p2_loss = self.lambda_p2 * self.l1_loss(
            torch.mul(p2_out, mask2_gt), torch.mul(i2_gt, mask2_gt)
        ) + self.beta_p2 * self.l1_loss(
            torch.mul(p2_out, 1 - mask2_gt), torch.mul(i2_gt, 1 - mask2_gt)
        )

        i_out_loss = self.lambda_out * self.l1_loss(
            torch.mul(i_out, mask_gt), torch.mul(i_gt, mask_gt)
        ) + self.beta_out * self.l1_loss(
            torch.mul(i_out, 1 - mask_gt), torch.mul(i_gt, 1 - mask_gt)
        )

        return p1_loss + p2_loss + i_out_loss

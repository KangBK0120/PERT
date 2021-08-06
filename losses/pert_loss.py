import torch.nn as nn
import torch.nn.functional as F

from data_model.dice_loss_param import DiceLossPram
from data_model.global_aware_sim_param import GlobalAwareSimParam
from data_model.loss_weight import LossWeight
from data_model.neg_ssim_param import NegSSIMPram
from data_model.region_aware_sim_param import RegionAwareSimilarityParam
from losses.dice_loss import DiceLoss
from losses.global_aware_similarity import GlobalAwareSimilarityLoss
from losses.neg_ssim_loss import NegativeSSIMLoss
from losses.region_aware_similarity import RegionAwareSimilarityLoss
from losses.vgg_loss import VGGLoss
from models.vgg_extractor import VGGExtractor


class PERTLoss(nn.Module):
    def __init__(
        self,
        loss_weight: LossWeight,
        dice_param: DiceLossPram,
        global_sim_param: GlobalAwareSimParam,
        neg_ssim_param: NegSSIMPram,
        region_sim_param: RegionAwareSimilarityParam,
    ):
        super(PERTLoss, self).__init__()

        self.dice_weight = loss_weight.dice_weight
        self.gs_weight = loss_weight.gs_weight
        self.neg_ssim_weight = loss_weight.neg_ssim_weight
        self.rs_weight = loss_weight.rs_weight
        self.vgg_weight = loss_weight.vgg_weight

        self.extractor = VGGExtractor()
        self.dice = DiceLoss(dice_param)
        self.gs = GlobalAwareSimilarityLoss(global_sim_param)
        self.neg_ssim = NegativeSSIMLoss(neg_ssim_param)
        self.rs = RegionAwareSimilarityLoss(region_sim_param)
        self.vgg_loss = VGGLoss()

    def forward(self, mask_out, image_p1_out, image_p2_out, image_out, mask_gt, image_gt):
        out_vgg_feats = self.extractor(image_out)
        gt_vgg_feats = self.extractor(image_gt)

        dice_loss = self.dice_weight * self.dice(mask_out, mask_gt)

        # gs_loss = self.gs_weight * self.gs(out_vgg_feats, gt_vgg_feats)

        neg_sim_loss = self.neg_ssim_weight * self.neg_ssim(image_out, image_gt)

        mask_p1_gt = F.interpolate(mask_gt, size=image_p1_out.shape[2:])
        image_p1_gt = F.interpolate(image_gt, size=image_p1_out.shape[2:])
        mask_p2_gt = F.interpolate(mask_gt, size=image_p2_out.shape[2:])
        image_p2_gt = F.interpolate(image_gt, size=image_p2_out.shape[2:])
        rs_loss = self.rs_weight * self.rs(
            image_p1_out,
            image_p1_gt,
            mask_p1_gt,
            image_p2_out,
            image_p2_gt,
            mask_p2_gt,
            image_out,
            image_gt,
            mask_gt,
        )

        vgg_loss = self.vgg_weight * self.vgg_loss(out_vgg_feats, gt_vgg_feats)

        return dice_loss + neg_sim_loss + rs_loss + vgg_loss  # + gs_loss

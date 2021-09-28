import torch
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
from losses.tv_loss import TVLoss
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
        self.tv_weight = loss_weight.tv_weight

        self.extractor = VGGExtractor()
        self.dice = DiceLoss(dice_param)
        self.gs = GlobalAwareSimilarityLoss(global_sim_param)
        self.neg_ssim = NegativeSSIMLoss(neg_ssim_param)
        self.rs = RegionAwareSimilarityLoss(region_sim_param)
        self.vgg_loss = VGGLoss()
        self.tv_loss = TVLoss()

    def forward(
        self,
        mask_out: torch.Tensor,
        image_p1_out: torch.Tensor,
        image_p2_out: torch.Tensor,
        image_out: torch.Tensor,
        mask_gt: torch.Tensor,
        image_gt: torch.Tensor,
        is_last_stage: bool,
    ):
        loss = self.dice_weight * self.dice(mask_out, mask_gt)
        # print(f"DICE:{loss.item()}")

        if is_last_stage:
            out_vgg_feats = self.extractor(image_out)
            gt_vgg_feats = self.extractor(image_gt)

            gs_loss = self.gs_weight * self.gs(out_vgg_feats, gt_vgg_feats)

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

            tv_loss = self.tv_weight * self.tv_loss(image_out)

            # print(
            #    f"NEG_SSIM:{neg_sim_loss.item()}\tRS:{rs_loss.item()}\tGS:{gs_loss.item()}\tVGG:{vgg_loss.item()}\tTV:{tv_loss.item()}"
            # )
            loss += neg_sim_loss + rs_loss + vgg_loss + gs_loss + tv_loss
        return loss

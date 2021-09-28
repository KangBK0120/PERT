from dataclasses import dataclass


@dataclass
class LossWeight:
    dice_weight: float
    gs_weight: float
    neg_ssim_weight: float
    rs_weight: float
    tv_weight: float
    vgg_weight: float

    @classmethod
    def from_config(cls, config):
        return cls(
            dice_weight=config.dice_weight,
            gs_weight=config.gs_weight,
            neg_ssim_weight=config.neg_ssim_weight,
            rs_weight=config.rs_weight,
            tv_weight=config.tv_weight,
            vgg_weight=config.vgg_weight,
        )

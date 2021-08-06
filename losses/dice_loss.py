import torch
import torch.nn as nn

from data_model.dice_loss_param import DiceLossPram


class DiceLoss(nn.Module):
    def __init__(self, dice_loss_param: DiceLossPram):
        super().__init__()
        self.smooth = dice_loss_param.smooth

    def forward(self, input_feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_flat = input_feat.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (input_flat * target_flat).sum()

        return torch.mean(
            1
            - (
                (2.0 * intersection + self.smooth)
                / (input_flat.sum() + target_flat.sum() + self.smooth)
            )
        )

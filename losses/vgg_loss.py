from typing import List

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, out_feature, gt_feature):
        return self.l2_loss(out_feature, gt_feature)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, out_feature, gt_feature):
        return self.l2_loss(self.gram_matrix(out_feature), self.gram_matrix(gt_feature))

    def gram_matrix(self, feat: torch.Tensor):
        # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
        return gram


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()

    def forward(self, out_features: List[torch.Tensor], gt_features: List[torch.Tensor]):
        assert len(out_features) == len(gt_features)

        loss = 0
        for i in range(len(out_features)):
            loss += self.perceptual_loss(out_features[i], gt_features[i]) + self.style_loss(
                out_features[i], gt_features[i]
            )
        return loss

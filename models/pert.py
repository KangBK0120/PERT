from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BackBone
from .brn import BRN
from .cfl import CFL
from .tln import TLN


class PERT(nn.Module):
    def __init__(self):
        super(PERT, self).__init__()

        self.backbone = BackBone()
        self.brn = BRN()
        self.tln = TLN()
        self.cfl = CFL()

    def forward(
        self,
        input_img: torch.Tensor,
        original_img: torch.Tensor,
    ) -> List[torch.Tensor]:
        conv1_output, res1_output, res2_output, res4_output, out = self.backbone(
            torch.cat([input_img, original_img], dim=1)
        )
        mask_out = self.tln(out, res2_output, res4_output)
        p1_out, p2_out, out = self.brn(out, conv1_output, res1_output, res2_output, res4_output)

        p1_out = self.cfl(mask_out, original_img, p1_out)
        p2_out = self.cfl(mask_out, original_img, p2_out)
        out = self.cfl(mask_out, original_img, out)
        return [mask_out, p1_out, p2_out, out]

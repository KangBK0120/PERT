import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image

from data_model.dice_loss_param import DiceLossPram
from data_model.global_aware_sim_param import GlobalAwareSimParam
from data_model.loss_weight import LossWeight
from data_model.neg_ssim_param import NegSSIMPram
from data_model.region_aware_sim_param import RegionAwareSimilarityParam
from models.pert import PERT
from losses.pert_loss import PERTLoss
from dataset import ErasingData


class Trainer:
    def __init__(self, config):
        self.config = config

        self.device = torch.device(f"cuda:{self.config.gpu}")

        self.pert = PERT().to(self.device)
        # print(sum(p.numel() for p in self.pert.parameters() if p.requires_grad))
        self.pert_loss = PERTLoss(
            loss_weight=LossWeight.from_config(self.config.loss_weight),
            dice_param=DiceLossPram.from_config(self.config.dice_loss_param),
            global_sim_param=GlobalAwareSimParam.from_config(self.config.gs_loss_param),
            neg_ssim_param=NegSSIMPram.from_config(self.config.neg_ssim_loss_param),
            region_sim_param=RegionAwareSimilarityParam.from_config(self.config.rs_loss_param),
        ).to(self.device)

        dataset = ErasingData(self.config.data.train_data_root, self.config.data.input_size)
        self.loader = DataLoader(
            dataset, batch_size=self.config.data.batch_size, shuffle=True, pin_memory=True
        )

        self.optimizer = optim.Adam(
            self.pert.parameters(),
            lr=self.config.optimizer.lr,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
        )

    def train(self):
        for epoch in range(self.config.epoch):
            for iter, (input_image, ground_truth, mask) in enumerate(self.loader):
                self.pert.train()
                self.optimizer.zero_grad()

                input_image, ground_truth, mask = (
                    input_image.to(self.device),
                    ground_truth.to(self.device),
                    mask.to(self.device),
                )
                i_before = ground_truth.clone()
                loss = 0
                for stage in range(self.config.num_iterative_stage):
                    mask_out, p1_out, p2_out, out = self.pert(input_image, i_before)
                    loss += self.pert_loss(mask_out, p1_out, p2_out, out, mask, ground_truth)
                    i_before = out.clone()
                loss.backward()
                self.optimizer.step()

                print(
                    f"Epoch {epoch + 1}/{self.config.epoch} iteration {iter + 1}/{len(self.loader)} Loss: {loss.item()}"
                )
                if (iter % self.config.sample_interval) == 0:
                    save_image(
                        torch.cat([input_image, out, ground_truth], dim=0),
                        os.path.join(
                            self.config.sample_save_path, f"out_{epoch + 1}_{iter + 1}.jpg"
                        ),
                        nrow=self.config.data.batch_size,
                    )
                if (iter % self.config.model_save_interval) == 0:
                    torch.save(
                        self.pert.state_dict(),
                        os.path.join(
                            self.config.model_save_path, f"model_{epoch + 1}_{iter + 1}.pth"
                        ),
                    )

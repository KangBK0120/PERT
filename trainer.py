import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from data_model.dice_loss_param import DiceLossPram
from data_model.global_aware_sim_param import GlobalAwareSimParam
from data_model.loss_weight import LossWeight
from data_model.neg_ssim_param import NegSSIMPram
from data_model.region_aware_sim_param import RegionAwareSimilarityParam
from dataset import ErasingData, Panoplay
from losses.pert_loss import PERTLoss
from models.pert import PERT


class Trainer:
    def __init__(self, config):
        self.config = config

        os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.config.gpu}"
        self.device = torch.device("cuda:0")

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

        self.writer = SummaryWriter(self.config.path.tensorboard_path)

    def train(self):
        for epoch in range(self.config.epoch):
            for iter, (original_image, ground_truth, mask) in enumerate(self.loader):
                self.pert.train()
                self.optimizer.zero_grad()

                original_image, ground_truth, mask = (
                    original_image.to(self.device),
                    ground_truth.to(self.device),
                    mask.to(self.device),
                )
                i_before = original_image.clone()
                loss = 0
                for stage in range(1, self.config.num_iterative_stage + 1):
                    mask_out, p1_out, p2_out, out = self.pert(i_before, original_image)
                    stage_loss = self.pert_loss(
                        mask_out,
                        p1_out,
                        p2_out,
                        out,
                        mask,
                        ground_truth,
                        stage == self.config.num_iterative_stage,
                    )
                    i_before = out.clone()
                    loss += stage_loss["dice_loss"]
                    if stage == self.config.num_iterative_stage:
                        loss += (
                            stage_loss["gs_loss"]
                            + stage_loss["neg_ssim_loss"]
                            + stage_loss["rs_loss"]
                            + stage_loss["vgg_loss"]
                            + stage_loss["tv_loss"]
                        )

                loss.backward()
                self.optimizer.step()

                print(
                    f"Epoch {epoch + 1}/{self.config.epoch} iteration {iter + 1}/{len(self.loader)} Loss: {loss.item()}"
                )

                if (iter % self.config.sample_interval) == 0 or iter + 1 == len(self.loader):

                    img_grid = make_grid(
                        torch.cat(
                            [
                                original_image,
                                out,
                                torch.mul(mask, out) + torch.mul(1 - mask, ground_truth),
                                ground_truth,
                            ],
                            dim=0,
                        ),
                        nrow=original_image.size(0),
                    )
                    save_image(
                        img_grid,
                        os.path.join(
                            self.config.path.sample_save_path, f"out_{epoch + 1}_{iter + 1}.jpg"
                        ),
                    )

                    step = epoch * len(self.loader) + iter
                    self.tensorboard_logging(stage_loss, step)
                    self.writer.flush()

                if (iter % self.config.model_save_interval) == 0 or iter + 1 == len(self.loader):
                    torch.save(
                        self.pert.state_dict(),
                        os.path.join(
                            self.config.path.model_save_path, f"model_{epoch + 1}_{iter + 1}.pth"
                        ),
                    )

    def tensorboard_logging(self, stage_loss, step):
        self.writer.add_scalar(
            "dice_loss",
            stage_loss["dice_loss"],
            step,
        )
        self.writer.add_scalar(
            "gs_loss",
            stage_loss["gs_loss"],
            step,
        )
        self.writer.add_scalar(
            "neg_ssim_loss",
            stage_loss["neg_ssim_loss"],
            step,
        )
        self.writer.add_scalar(
            "rs_loss",
            stage_loss["rs_loss"],
            step,
        )
        self.writer.add_scalar(
            "vgg_loss",
            stage_loss["vgg_loss"],
            step,
        )
        self.writer.add_scalar(
            "tv_loss",
            stage_loss["tv_loss"],
            step,
        )

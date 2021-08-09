import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.pert import PERT
from dataset import ErasingData, OWNData


class Tester:
    def __init__(self, config):
        self.config = config

        self.device = torch.device(f"cuda:{self.config.gpu}")

        self.pert = PERT()
        self.pert.load_state_dict(torch.load(self.config.model_path))
        self.pert = self.pert.to(self.device)

        if self.config.dataset == "SCUT-enstext":
            dataset = ErasingData(
                self.config.data.test_data_root, self.config.data.input_size, False
            )
        else:
            dataset = OWNData(self.config.data.test_data_root, self.config.data.input_size)
        self.loader = DataLoader(
            dataset, batch_size=self.config.data.batch_size, shuffle=True, pin_memory=True
        )

    def test(self):
        for iter, (input_image, ground_truth, mask) in enumerate(self.loader):
            self.pert.eval()

            input_image, ground_truth, mask = (
                input_image.to(self.device),
                ground_truth.to(self.device),
                mask.to(self.device),
            )
            i_before = input_image.clone()
            with torch.no_grad():
                for stage in range(self.config.num_iterative_stage):
                    _, _, _, out = self.pert(input_image, i_before)
                    i_before = out.clone()
                if self.config.dataset == "SCUT-enstext":
                    result = torch.cat([input_image, out, ground_truth], dim=0)
                else:
                    result = torch.cat([input_image, out], dim=0)
                save_image(
                    result,
                    os.path.join(self.config.sample_save_path, f"out_{iter}.jpg"),
                    nrow=self.config.data.batch_size,
                )

import math
import os

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import ErasingData, OWNData, Panoplay
from models.pert import PERT


class Tester:
    def __init__(self, config):
        self.config = config

        self.device = torch.device(f"cuda:{self.config.gpu}")

        self.pert = PERT()
        self.pert.load_state_dict(torch.load(self.config.model_path))
        self.pert = self.pert.to(self.device)

        if self.config.dataset == "scut":
            dataset = ErasingData(
                self.config.data.test_data_root, self.config.data.input_size, "test"
            )
        else:
            dataset = OWNData(self.config.data.test_data_root, self.config.data.input_size)

        self.loader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_siz,
            shuffle=False,
            pin_memory=True,
        )

        if self.config.evaluation:
            self.metrics = dict()
            self.metrics["mse"] = []
            self.metrics["psnr"] = []
            self.metrics["age"] = []
            self.metrics["peps"] = []
            self.metrics["pceps"] = []

    def test(self):
        for iter, (input_image, ground_truth, mask, image_names) in enumerate(self.loader):
            self.pert.eval()

            input_image, ground_truth, mask = (
                input_image.to(self.device),
                ground_truth.to(self.device),
                mask.to(self.device),
            )

            out = self.process_input(input_image)

            if self.config.input_concat:
                result = torch.cat([input_image, out, ground_truth], dim=0)
                save_image(
                    result,
                    os.path.join(self.config.sample_save_path, f"out_{iter}.jpg"),
                    nrow=len(input_image),
                )
            else:
                for image_num in range(out.size(0)):
                    save_image(
                        out[image_num, :, :, :],
                        os.path.join(self.config.sample_save_path, image_names[image_num]),
                    )
            if self.config.evaluation:
                mse_list, psnr_list, age_list, peps_list, pceps_list = self.evaluation(
                    out.cpu(), ground_truth.cpu()
                )
                self.metrics["mse"] += mse_list
                self.metrics["psnr"] += psnr_list
                self.metrics["age"] += age_list
                self.metrics["peps"] += peps_list
                self.metrics["pceps"] += pceps_list

        if self.config.evaluation:
            print(f"MSE: {sum(self.metrics['mse']) / len(self.metrics['mse'])}")
            print(f"PSNR: {sum(self.metrics['psnr']) / len(self.metrics['psnr'])}")
            print(f"AGE: {sum(self.metrics['age']) / len(self.metrics['age'])}")
            print(f"pEPs: {sum(self.metrics['peps']) / len(self.metrics['peps'])}")
            print(f"pCEPs: {sum(self.metrics['pceps']) / len(self.metrics['pceps'])}")

    def process_input(self, input_image):
        i_before = input_image.clone()
        with torch.no_grad():
            for stage in range(self.config.num_iterative_stage):
                _, _, _, out = self.pert(i_before, input_image)
                i_before = out.clone()
        return out

    def evaluation(self, outs, gts):
        assert outs.size(0) == gts.size(0)
        mse_list = []
        psnr_list = []
        age_list = []
        peps_list = []
        pceps_list = []

        for i in range(outs.size(0)):
            mse = ((outs[i, :, :, :] - gts[i, :, :, :]) ** 2).mean()
            psnr = 10 * math.log10(1 / mse)

            gt_R = gts[i, 0, :, :]
            gt_G = gts[i, 1, :, :]
            gt_B = gts[i, 2, :, :]
            y_gt = 0.299 * gt_R + 0.587 * gt_G + 0.114 * gt_B

            out_R = outs[i, 0, :, :]
            out_G = outs[i, 1, :, :]
            out_B = outs[i, 2, :, :]
            y_out = 0.299 * out_R + 0.587 * out_G + 0.114 * out_B

            diff = abs(np.array(y_gt * 255) - np.array(y_out * 255)).round().astype(np.uint8)
            age = np.mean(diff)

            errors = diff > 20
            eps = sum(sum(errors)).astype(float)
            peps = eps / float(512 * 512)

            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            eroded_errors = ndimage.binary_erosion(errors, structure).astype(errors.dtype)
            ceps = sum(sum(eroded_errors))
            pceps = ceps / float(512 * 512)

            mse_list.append(mse)
            psnr_list.append(psnr)
            age_list.append(age)
            peps_list.append(peps)
            pceps_list.append(pceps)

        return mse_list, psnr_list, age_list, peps_list, pceps_list

import glob
import json
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs


def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        # print(angle)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = Image.fromarray(img_rotation)
    return imgs


def image_transforms(loadSize):
    return transforms.Compose(
        [
            transforms.Resize(size=loadSize, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )


class ErasingData(Dataset):
    def __init__(self, data_root, load_size, mode="train"):
        super(ErasingData, self).__init__()
        self.root = data_root
        self.image_names = sorted(
            [x.split("/")[-1] for x in glob.glob(f"{self.root}/all_images/*.jpg")]
        )

        self.load_size = load_size
        self.img_transforms = image_transforms(load_size)
        self.mode = mode

    def __getitem__(self, index):
        img = Image.open(f"{self.root}/all_images/{self.image_names[index]}")
        mask = Image.open(f"{self.root}/mask/{self.image_names[index]}")
        gt = Image.open(f"{self.root}/all_labels/{self.image_names[index]}")
        if self.mode == "train":
            all_input = [img, mask, gt]
            all_input = random_horizontal_flip(all_input)
            all_input = random_rotate(all_input)
            img = all_input[0]
            mask = all_input[1]
            gt = all_input[2]

        input_image = self.img_transforms(img.convert("RGB"))
        mask = self.img_transforms(mask.convert("L"))
        ground_truth = self.img_transforms(gt.convert("RGB"))

        if self.mode == "train":
            return input_image, ground_truth, mask
        else:
            return input_image, ground_truth, mask, self.image_names[index]

    def __len__(self):
        return len(self.image_names)


class OWNData(Dataset):
    def __init__(self, data_root, load_size) -> None:
        super(OWNData, self).__init__()
        self.root = data_root
        self.image_files = glob.glob(f"{self.root}/*.jpg")

        self.load_size = load_size
        self.img_transforms = image_transforms(load_size)

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])
        input_image = self.img_transforms(img.convert("RGB"))

        return input_image, torch.zeros_like(input_image), torch.zeros_like(input_image)

    def __len__(self):
        return len(self.image_files)

import os
import shutil

from omegaconf import OmegaConf


def build_config(is_train: bool):
    if is_train:
        config = OmegaConf.load("./train_config.yaml")
    else:
        config = OmegaConf.load("./test_config.yaml")

    if is_train:
        os.makedirs(config.path.root, exist_ok=True)
        os.makedirs(config.path.model_save_path, exist_ok=True)
        os.makedirs(config.path.sample_save_path, exist_ok=True)
        os.makedirs(config.path.tensorboard_path, exist_ok=True)
        shutil.copy("./train_config.yaml", config.path.root + "/used_config.yaml")
    else:
        os.makedirs(config.sample_save_path, exist_ok=True)
    return config

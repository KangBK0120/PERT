import os
import shutil

from omegaconf import OmegaConf


def build_config(is_train: bool):
    if is_train:
        config = OmegaConf.load("./train_config.yaml")
    else:
        config = OmegaConf.load("./test_config.yaml")

    if "model_save_path" in config:
        os.makedirs(config.model_save_path, exist_ok=True)
        shutil.copy("./train_config.yaml", config.model_save_path + "/used_config.yaml")
    if "sample_save_path" in config:
        os.makedirs(config.sample_save_path, exist_ok=True)

    return config

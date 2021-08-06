from argparse import ArgumentParser
from config import build_config

from tester import Tester
from trainer import Trainer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    mode = parser.parse_args().mode
    if mode == "train":
        trainer = Trainer(build_config(is_train=True))
        trainer.train()
    else:
        tester = Tester(build_config(is_train=False))
        tester.test()

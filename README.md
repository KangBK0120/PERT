# A Simple and Strong Baseline: Progressively Region-based Scene Text Removal Networks

Unofficial Pytorch implementation of PERT | [paper](https://arxiv.org/pdf/2106.13029.pdf) | [Pretrained model](https://drive.google.com/file/d/1hJqsgWjAMVPSegq9KWdyXSrJeAgqomsA/view?usp=sharing), NOTE: Not the latest version of code. I will update the model ASAP.

*NOTE* As the Convolution output channel numbers are not opened in the original paper, I set the numbers arbitrarily.

The models is a little bit larger than the one in the paper (14.0M parameters in the paper, 15M in this codes)

![PERT_output](https://user-images.githubusercontent.com/25279765/128654640-46adae94-7103-4ca9-bac9-dfe9d62395ec.jpg)


The reproduced results from SCUT-enstext test dataset. Input images, network outputs, ground truths.

## Getting Started

### Dependencies

I've trained and tested the codes on a single RTX 3090.

#### Requirements
- pytorch == 1.9.0 (may work with the lower versions, but not tested)
- torchvision
- kornia == 0.5.0
- spatial-correlation-sampler
- opencv-python
- omegaconf

```
pip install -r requirements.txt
```

### Training

I trained the model with [SCUT-enstext](https://github.com/HCIILAB/SCUT-EnsText). As the original dataset does not have mask image file, I created the mask manually with PIL. If you need the codes, please see the utils folder.

Assume the directory structure is same as follows.

```
SCUT-enstext
├── test
│   ├── all_gts
│   ├── all_images
│   ├── all_labels
│   └── mask
└── train
    ├── all_gts
    ├── all_images
    ├── all_labels
    └── mask
```

If you want change training and test config, see {train, test}_config.yaml

You can run the codes as follows

```
python main.py --mode {train, test}
```

## Reference

the dataset code is modified from [EraseNet](https://github.com/lcy0604/EraseNet)
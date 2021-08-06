import torch.nn as nn
import torchvision.models as models


class VGGExtractor(nn.Module):
    def __init__(self):
        super(VGGExtractor, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(1, 4):
            for param in getattr(self, f"enc_{i}").parameters():
                param.requires_grad = False

    def forward(self, image):
        out = self.enc_1(image)
        conv2_1 = out.clone()
        out = self.enc_2(out)
        conv3_1 = out.clone()
        out = self.enc_3(out)
        conv4_1 = out.clone()

        return conv2_1, conv3_1, conv4_1

import sys
import tqdm

import torch
from torch import nn
from torchvision import models

import custom_utils

print('Test pretrained models ...')

dev, _summary_dev = custom_utils.fetch_device()
loss_func = nn.functional.cross_entropy # TODO: Add in weight

# ResNet50
print('Validating deeplabv3_resnet50 ...')
model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=21)
custom_utils.validate(model, dev, loss_func, -1)

# ResNet101
print('Validating deeplabv3_resnet101 ...')
model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)
custom_utils.validate(model, dev, loss_func, -1)

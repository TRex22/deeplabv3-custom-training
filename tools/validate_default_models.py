import sys
import tqdm

import torch
from torch import nn
from torchvision import models

import custom_utils

config = {
  "dataset": "COCO16",
  "coco_path": "/mnt/scratch_disk/data/coco/data_raw/",
  "val_batch_size": 1
}

print('Test pretrained models ...')

category_list = custom_utils.fetch_category_list(config)
dev, _summary_dev = custom_utils.fetch_device()
loss_func = nn.functional.cross_entropy # TODO: Add in weight

# ResNet50
print('Validating deeplabv3_resnet50 ...')
model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=21)
model.to(dev)
custom_utils.validate(model, dev, loss_func, -1, config, category_list=category_list, save=False)

# ResNet101
print('Validating deeplabv3_resnet101 ...')
model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)
model.to(dev)
custom_utils.validate(model, dev, loss_func, -1, config, category_list=category_list, save=False)

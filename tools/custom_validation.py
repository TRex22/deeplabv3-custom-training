import gc
import sys
import time
import tqdm

import torch
from torch import nn

import custom_utils

dev, summary_dev = custom_utils.fetch_device()

path = sys.argv[1]
print(f'Validate: {path}')

config, _start_epoch, model_path = custom_utils.open_config(path)

# Based on reference code
loss_func = nn.functional.cross_entropy # TODO: Add in weight
lr_scheduler = None
epoch = -1

custom_utils.clear_gpu() # Needed to ensure no memory loss

category_list = custom_utils.fetch_category_list(config)
model, opt = custom_utils.initialise_model(dev, config, num_classes=len(category_list))
model, _opt = custom_utils.load(model, opt, dev, path, show_stats=False)

custom_utils.validate(model, dev, loss_func, lr_scheduler, epoch, config, category_list=category_list, save=False)
custom_utils.clear_gpu() # Needed to ensure no memory loss


# compute_iou1(output, target)
# compute_iou2(outputs: torch.Tensor, labels: torch.Tensor)
# compute_iou3(output, target, num_classes)

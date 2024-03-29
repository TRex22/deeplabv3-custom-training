import sys
import tqdm

import torch
from torch import nn
from torchvision import models

import custom_utils

config = {
  "dataset": "COCO21",
  "dataset_path": "/mnt/scratch_disk/data/coco/data_raw/",
  "val_batch_size": 1,
  "sample_percentage": 1.0
}

print('Test pretrained models ...')

category_list = custom_utils.fetch_category_list(config)
device, _summary_dev = custom_utils.fetch_device()
loss_func = nn.functional.cross_entropy # TODO: Add in weight
lr_scheduler = None

# If arg open model
if len(sys.argv) == 2:
  model_path = sys.argv[1]
  print(f'Validating {model_path} ...')
  config, _start_epoch, _model_path = custom_utils.open_config(model_path)
  category_list = custom_utils.fetch_category_list(config)

  model, opt = custom_utils.initialise_model(device, config, num_classes=len(category_list))
  model, _opt, _epoch = custom_utils.load(model, device, model_path, opt=opt) # Load model
  model.to(device)

  config["val_batch_size"] = 1
  config["val_num_workers"] = 1

  custom_utils.validate(model, device, loss_func, lr_scheduler, -1, config, category_list=category_list, save=False)
elif len(sys.argv) == 3:
  model_path = sys.argv[1]
  config_path = sys.argv[2]

  print(f'Validating {model_path} ...')
  config, _start_epoch, _model_path = custom_utils.open_config(config_path)
  category_list = custom_utils.fetch_category_list(config)

  model, opt = custom_utils.initialise_model(device, config, num_classes=len(category_list))
  model, _opt, _epoch = custom_utils.load(model, device, model_path, opt=opt) # Load model
  model.to(device)

  config["val_batch_size"] = 1
  config["val_num_workers"] = 1

  custom_utils.validate(model, device, loss_func, lr_scheduler, -1, config, category_list=category_list, save=False)
else:
  # ResNet50
  print('Validating deeplabv3_resnet50 ...')
  model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=21)
  model.to(device)
  custom_utils.validate(model, device, loss_func, lr_scheduler, -1, config, category_list=category_list, save=False)

  # ResNet101
  print('Validating deeplabv3_resnet101 ...')
  model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)
  model.to(device)
  custom_utils.validate(model, device, loss_func, lr_scheduler, -1, config, category_list=category_list, save=False)

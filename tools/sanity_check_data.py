import sys
import tqdm

import torch
from torch import nn
from torchvision import models

import custom_utils

import sys
sys.path.insert(1, '../references/segmentation/')

from coco_utils import get_coco
import transforms as T

print("Sanity Check Data ...")

config = {
  "dataset": "COCO21",
  "dataset_path": "/mnt/scratch_disk/data/coco/data_raw/",
  "val_batch_size": 1,
  "sample_percentage": 1.0
}

batch_size = 16

print('=== Load COCO21 ===')
train_dataset, train_dataloader = load_dataset(config, config['dataset_path'], 'train', category_list=category_list, batch_size=batch_size, sample=True)
val_dataset, val_dataloader = load_dataset(config, config['dataset_path'], 'val', category_list=category_list, batch_size=1, sample=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('Sample 10%')
config = {
  "dataset": "COCO21",
  "dataset_path": "/mnt/scratch_disk/data/coco/data_raw/",
  "val_batch_size": 1,
  "sample_percentage": 0.1
}

train_dataset, train_dataloader = load_dataset(config, config['dataset_path'], 'train', category_list=category_list, batch_size=batch_size, sample=True)
val_dataset, val_dataloader = load_dataset(config, config['dataset_path'], 'val', category_list=category_list, batch_size=1, sample=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('=== Load COCO16 ===')
config = {
  "dataset": "COCO16",
  "dataset_path": "/mnt/scratch_disk/data/coco/data_raw/",
  "val_batch_size": 1,
  "sample_percentage": 1.0
}

train_dataset, train_dataloader = load_dataset(config, config['dataset_path'], 'train', category_list=category_list, batch_size=batch_size, sample=True)
val_dataset, val_dataloader = load_dataset(config, config['dataset_path'], 'val', category_list=category_list, batch_size=1, sample=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('Sample 10%')
config = {
  "dataset": "COCO16",
  "dataset_path": "/mnt/scratch_disk/data/coco/data_raw/",
  "val_batch_size": 1,
  "sample_percentage": 0.1
}

train_dataset, train_dataloader = load_dataset(config, config['dataset_path'], 'train', category_list=category_list, batch_size=batch_size, sample=True)
val_dataset, val_dataloader = load_dataset(config, config['dataset_path'], 'val', category_list=category_list, batch_size=1, sample=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('=== Load cityscapes ===')
config = {
  "dataset": "cityscapes",
  "dataset_path": "/data/data/cityscapes/data_raw/",
  "val_batch_size": 1,
  "sample_percentage": 1.0
}

train_dataset, train_dataloader = load_dataset(config, config['dataset_path'], 'train', category_list=category_list, batch_size=batch_size, sample=True)
val_dataset, val_dataloader = load_dataset(config, config['dataset_path'], 'val', category_list=category_list, batch_size=1, sample=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('Sample 10%')
config = {
  "dataset": "cityscapes",
  "dataset_path": "/data/data/cityscapes/data_raw/",
  "val_batch_size": 1,
  "sample_percentage": 0.1
}

train_dataset, train_dataloader = load_dataset(config, config['dataset_path'], 'train', category_list=category_list, batch_size=batch_size, sample=True)
val_dataset, val_dataloader = load_dataset(config, config['dataset_path'], 'val', category_list=category_list, batch_size=1, sample=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('=== DataSet Calls ==')
coco_dataset_train = load_coco("/mnt/scratch_disk/data/coco/data_raw/", 'train', category_list=category_list)
coco_dataset_test = load_coco("/mnt/scratch_disk/data/coco/data_raw/", 'test', category_list=category_list)

cityscapes_dataset_train = torchvision.datasets.Cityscapes("/data/data/cityscapes/data_raw/", split='train', mode='fine', target_type='semantic', transforms=custom_utils.cityscapes_transforms()) # TODO: Cityscapes 'test'
cityscapes_dataset_test = torchvision.datasets.Cityscapes("/data/data/cityscapes/data_raw/", split='test', mode='fine', target_type='semantic', transforms=custom_utils.cityscapes_transforms()) # TODO: Cityscapes 'test'

print(f'coco_dataset_train: {len(coco_dataset_train)}')
print(f'coco_dataset_test: {len(coco_dataset_test)}')
print(f'cityscapes_dataset_train: {len(cityscapes_dataset_train)}')
print(f'cityscapes_dataset_test: {len(cityscapes_dataset_test)}')


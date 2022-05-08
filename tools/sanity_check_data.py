import sys
import tqdm

import torch
import torchvision
from torch import nn
from torchvision import models

import custom_utils

import sys
sys.path.insert(1, '../references/segmentation/')

from from_games_dataset import FromGamesDataset

import presets
from coco_utils import get_coco
import transforms as T

print("Sanity Check Data ...")

batch_size = 16

base_path = '/home-mscluster/jchalom/data' #'/data/data'
coco_dataset_path = f"{base_path}/coco/data_raw/"
cityscapes_path = f"{base_path}/cityscapes/data_raw/"
fromgames_path = f"{base_path}/from_games/"

config = {
  "dataset": "COCO21",
  "dataset_path": coco_dataset_path,
  "val_batch_size": 1,
  "sample_percentage": 1.0,
  "shuffle": True,
  "drop_last": True,
  "pin_memory": True,
  "train_num_workers": 4,
  "val_num_workers": 4,
  "cityscapes_mode": "fine"
}

print('=== Load COCO21 ===')
train_dataset, train_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'train', category_list=None, batch_size=batch_size, training=True)
val_dataset, val_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'val', category_list=None, batch_size=1, training=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}, train_dataloader * batch_size: {len(train_dataloader) * batch_size}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('Sample 10%')
config = {
  "dataset": "COCO21",
  "dataset_path": coco_dataset_path,
  "val_batch_size": 1,
  "sample_percentage": 0.1,
  "shuffle": True,
  "drop_last": True,
  "pin_memory": True,
  "train_num_workers": 4,
  "val_num_workers": 4,
  "cityscapes_mode": "fine"
}

train_dataset, train_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'train', category_list=None, batch_size=batch_size, training=True)
val_dataset, val_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'val', category_list=None, batch_size=1, training=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}, train_dataloader * batch_size: {len(train_dataloader) * batch_size}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('=== Load COCO16 ===')
config = {
  "dataset": "COCO16",
  "dataset_path": coco_dataset_path,
  "val_batch_size": 1,
  "sample_percentage": 1.0,
  "shuffle": True,
  "drop_last": True,
  "pin_memory": True,
  "train_num_workers": 4,
  "val_num_workers": 4,
  "cityscapes_mode": "fine"
}

train_dataset, train_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'train', category_list=None, batch_size=batch_size, training=True)
val_dataset, val_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'val', category_list=None, batch_size=1, training=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}, train_dataloader * batch_size: {len(train_dataloader) * batch_size}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('Sample 10%')
config = {
  "dataset": "COCO16",
  "dataset_path": coco_dataset_path,
  "val_batch_size": 1,
  "sample_percentage": 0.1,
  "shuffle": True,
  "drop_last": True,
  "pin_memory": True,
  "train_num_workers": 4,
  "val_num_workers": 4,
  "cityscapes_mode": "fine"
}

train_dataset, train_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'train', category_list=None, batch_size=batch_size, training=True)
val_dataset, val_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'val', category_list=None, batch_size=1, training=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}, train_dataloader * batch_size: {len(train_dataloader) * batch_size}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('=== Load fromgames ===')
config = {
  "dataset": "fromgames",
  "dataset_path": fromgames_path,
  "val_batch_size": 1,
  "sample_percentage": 1.0,
  "shuffle": True,
  "drop_last": True,
  "pin_memory": True,
  "train_num_workers": 4,
  "val_num_workers": 4,
  "cityscapes_mode": "fine"
}

train_dataset, train_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'train', category_list=None, batch_size=batch_size, training=True)
val_dataset, val_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'val', category_list=None, batch_size=1, training=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}, train_dataloader * batch_size: {len(train_dataloader) * batch_size}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('Sample 10%')
config = {
  "dataset": "fromgames",
  "dataset_path": fromgames_path,
  "val_batch_size": 1,
  "sample_percentage": 0.1,
  "shuffle": True,
  "drop_last": True,
  "pin_memory": True,
  "train_num_workers": 4,
  "val_num_workers": 4,
  "cityscapes_mode": "fine"
}

train_dataset, train_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'train', category_list=None, batch_size=batch_size, training=True)
val_dataset, val_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'val', category_list=None, batch_size=1, training=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}, train_dataloader * batch_size: {len(train_dataloader) * batch_size}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('=== Load cityscapes ===')
config = {
  "dataset": "cityscapes",
  "dataset_path": cityscapes_path,
  "val_batch_size": 1,
  "sample_percentage": 1.0,
  "shuffle": True,
  "drop_last": True,
  "pin_memory": True,
  "train_num_workers": 4,
  "val_num_workers": 4,
  "cityscapes_mode": "fine"
}

train_dataset, train_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'train', category_list=None, batch_size=batch_size, training=True)
val_dataset, val_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'val', category_list=None, batch_size=1, training=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}, train_dataloader * batch_size: {len(train_dataloader) * batch_size}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('Sample 10%')
config = {
  "dataset": "cityscapes",
  "dataset_path": cityscapes_path,
  "val_batch_size": 1,
  "sample_percentage": 0.1,
  "shuffle": True,
  "drop_last": True,
  "pin_memory": True,
  "train_num_workers": 4,
  "val_num_workers": 4,
  "cityscapes_mode": "fine"
}

train_dataset, train_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'train', category_list=None, batch_size=batch_size, training=True)
val_dataset, val_dataloader = custom_utils.load_dataset(config, config['dataset_path'], 'val', category_list=None, batch_size=1, training=False)

print(f'train_dataset: {len(train_dataset)}, train_dataloader: {len(train_dataloader)}, train_dataloader * batch_size: {len(train_dataloader) * batch_size}')
print(f'val_dataset: {len(val_dataset)}, val_dataloader: {len(val_dataloader)}')

print('=== DataSet Calls ==')
coco_dataset_train = get_coco(coco_dataset_path, 'train', presets.SegmentationPresetTrain(base_size=520, crop_size=480), category_list=None)
coco_dataset_val = get_coco(coco_dataset_path, 'val', presets.SegmentationPresetEval(base_size=520), category_list=None)

cityscapes_dataset_train = torchvision.datasets.Cityscapes(cityscapes_path, split='train', mode='fine', target_type='semantic', transforms=custom_utils.cityscapes_transforms())
cityscapes_dataset_val = torchvision.datasets.Cityscapes(cityscapes_path, split='val', mode='fine', target_type='semantic', transforms=custom_utils.cityscapes_transforms())

fromgames_dataset_train = FromGamesDataset(fromgames_path, split='train', transforms=custom_utils.cityscapes_transforms())
fromgames_dataset_val = FromGamesDataset(fromgames_path, split='val', transforms=custom_utils.cityscapes_transforms())

print("\nResults:")
print(f'coco_dataset_train: {len(coco_dataset_train)}')
print(f'coco_dataset_val: {len(coco_dataset_val)}\n')

print(f'cityscapes_dataset_train: {len(cityscapes_dataset_train)}')
print(f'cityscapes_dataset_val: {len(cityscapes_dataset_val)}\n')

print(f'fromgames_dataset_train: {len(fromgames_dataset_train)}')
print(f'fromgames_dataset_val: {len(fromgames_dataset_val)}\n')

print('=== Iterate through Data ==')
for xb, yb in fromgames_dataset_train:
  print("Exists!")

breakpoint()

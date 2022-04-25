# Usage: python3 custom_tarining.py ./config.json
# Usage: python3 custom_tarining.py ./model_42.pth
# Usage: python3 custom_tarining.py ./model_42.pth ./config.json # Will use the custom config

# TODO: Early stopping
# TODO: Dynamic lr

# https://stackoverflow.com/questions/63892031/how-to-train-deeplabv3-on-custom-dataset-on-pytorch
import os
import gc
import time
import json
import numpy as np
from pathlib import Path

import torch
from torch import optim
from torch import nn
import torchvision
from torch.utils.data import DataLoader

import tqdm

import torchvision.transforms as transforms
from torchvision import models
from torchinfo import summary

import sys
sys.path.insert(1, '../references/segmentation/')

# Reference Code
import presets
import utils
from coco_utils import get_coco

################################################################################
# Helper Methids                                                               #
################################################################################

# Will either open the config path or get config from model checkpoint
def open_config(path):
  try:
    # Remove comments first
    raw_json = ""
    with open(path) as f:
      for line in f:
        line = line.partition('//')[0]
        line = line.rstrip()
        raw_json += f"{line}\n"

    config = json.loads(raw_json)
    epoch = 0
  except:
    checkpoint = torch.load(path)
    config = checkpoint['args']
    epoch = checkpoint['epoch'] + 1

  create_folder(config["save_path"])
  return [config, epoch]

def save_csv(file_path, csv_data):
  with open(file_path, 'a') as f:
    f.write(f'{csv_data}\n')

# COCO Dataset
# train_image_path = '/data/data/coco/data_raw/train2017'
# val_image_path = '/data/data/coco/data_raw/val2017'
# train_annotation_path = '/mnt/excelsior/data/coco/zips/annotations/annotations/instances_train2017.json'
# val_annotation_path = '/data/data/coco/zips/annotations/annotations/instances_val2017.json'

# train_dataset = torchvision.datasets.CocoDetection(train_image_path, train_annotation_path)
# val_dataset = torchvision.datasets.CocoDetection(val_image_path, val_annotation_path)
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, persistent_workers=False)
def load_coco(root, image_set):
  # Using reference code
  # See Readme.md for new category list
  category_list = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

  if image_set == 'train':
    transforms = presets.SegmentationPresetTrain(base_size=520, crop_size=480)
  else:
    transforms = presets.SegmentationPresetEval(base_size=520)

  return get_coco(root, image_set, transforms, category_list=category_list)

def xavier_uniform_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.xavier_uniform_(layer.weight)

def initialise_model(dev, config, pretrained=False, num_classes=21):
  if config["selected_model"] == 'deeplabv3_resnet101':
    model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, num_classes=num_classes)
  elif config["selected_model"] == 'deeplabv3_resnet50':
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
  else:
    raise RuntimeError("Invalid model selected.")

  # Randomise weights
  model.apply(xavier_uniform_init)

  model = model.to(dev)

  # Reference code uses SGD
  if config["opt_function"] == 'ADAM':
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=config["betas"], eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
  else:
    opt = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

  return [model, opt]

def load(model, opt, path):
  # Load model weights
  # Training crashed when lr dropped to complex numbers

  # TODO: Load optimizer
  print(f'Loading model from: {path}')
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model'], strict=False)
  opt.load_state_dict(checkpoint['optimizer'], strict=False)
  model.to(dev)

  print(f'Model loaded into {summary_dev}!')
  model_stats = summary(model, device=summary_dev)

  return model, opt

def create_folder(path):
  Path(path).mkdir(parents=True, exist_ok=True)

# Built to be compatible with reference code
def save(model, opt, epoch, config, save_path):
  checkpoint = {
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    # "lr_scheduler": lr_scheduler.state_dict(),
    "epoch": epoch,
    "args": config,
  }

  torch.save(checkpoint, os.path.join(save_path, f"model_{epoch}.pth"))

def loss_batch(model, device, scaler, loss_func, xb, yb, opt=None):
  input = xb.to(device)
  prediction = model(input)

  del input

  output = prediction['out']
  target = yb.to(device)

  loss = loss_func(output, target, ignore_index=255)
  dice_loss = dice_coef(target, output)

  sum_batch_iou_score = 0.0
  for i in range(output.shape[0]):
    sum_batch_iou_score += compute_iou(output[i], target[i]).cpu()

  iou_score = sum_batch_iou_score / output.shape[0]

  del output
  del target

  if opt is not None:
    scaler.scale(loss).backward()

    # clip_grad_norm
    # Unscales the gradients of optimizer's assigned config in-place
    scaler.unscale_(opt)

    # Since the gradients of optimizer's assigned config are now unscaled, clips as usual.
    # You may use the same value for max_norm here as you would without gradient scaling.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance

  return [loss.cpu().item(), dice_loss, iou_score, opt]

def run_loop(model, device, dataloader, batch_size, scaler, loss_func, opt=None):
  sum_of_loss = 0.0
  sum_of_iou = 0.0
  sum_of_dice = 0.0

  pbar = tqdm.tqdm(total=len(dataloader))

  # TODO: Allow disabling sub-batches
  if opt is None:
    # Dont use sub-batches
    for xb, yb in tqdm.tqdm(dataloader):
      loss, dice_loss, iou_score, opt = loss_batch(model, device, scaler, loss_func, xb, yb, opt=opt)

      sum_of_loss += loss
      sum_of_iou += iou_score
      sum_of_dice += dice_loss

    final_loss = sum_of_loss / len(dataloader)
    final_iou = sum_of_iou / len(dataloader)
    final_dice = sum_of_dice / len(dataloader)
  else: # Use sub-batches
    for inner_batch in dataloader:
      for i in range(0, inner_batch[0].shape[0], batch_size):
        xb = inner_batch[0][i:i+batch_size]
        yb = inner_batch[1][i:i+batch_size]

        loss, dice_loss, iou_score, opt = loss_batch(model, device, scaler, loss_func, xb, yb, opt=opt)

        sum_of_loss += loss
        sum_of_iou += iou_score
        sum_of_dice += dice_loss

      final_loss = sum_of_loss / (len(dataloader) * batch_size)
      final_iou = sum_of_iou / (len(dataloader) * batch_size)
      final_dice = sum_of_dice / (len(dataloader) * batch_size)

      pbar.update(1)

  if opt is not None:
    pbar.write(f'Epoch {epoch} train loss: {final_loss} train IoU: {final_iou} train dice: {final_dice}')

    train_csv_path = f'{config["save_path"]}/train_loss.csv'
    save_csv(train_csv_path, f'{final_loss},{final_iou},{final_dice}')
  else:
    pbar.write(f'Epoch {epoch} val loss: {final_loss} val IoU: {final_iou} val dice: {final_dice}')

    val_csv_path = f'{config["save_path"]}/val_loss.csv'
    save_csv(val_csv_path, f'{final_loss},{final_iou},{final_dice}')

  return [final_loss, final_iou, opt]

def train(model, device, loss_func, opt, epoch, outer_batch_size):
  # Load Data - in train step to save memory
  train_dataset = load_coco(config['coco_path'], 'train')
  subset_idex = list(range(int(len(train_dataset) * config["sample_percentage"]))) # TODO: Unload others
  train_subset = torch.utils.data.Subset(train_dataset, subset_idex)
  train_dataloader = DataLoader(train_subset, batch_size=outer_batch_size, shuffle=True, drop_last=True, collate_fn=utils.collate_fn)

  model = model.train()
  scaler = torch.cuda.amp.GradScaler(enabled=True)

  with torch.cuda.amp.autocast(enabled=True, cache_enabled=True): # TODO: cache_enabled
    final_loss, final_iou, opt = run_loop(model, device, train_dataloader, config["batch_size"], scaler, loss_func, opt=opt)

  del train_dataloader
  del train_dataset

  return [model, opt]

def validate(model, device, loss_func, epoch, outer_batch_size):
  # Load Data - in val step to save memory
  val_dataset = load_coco(config['coco_path'], 'val')
  val_dataloader = DataLoader(val_dataset, batch_size=config["val_batch_size"], shuffle=False, drop_last=True, collate_fn=utils.collate_fn)

  model = model.eval()
  scaler = torch.cuda.amp.GradScaler(enabled=True)

  sum_of_loss = 0.0
  sum_of_iou = 0.0

  pbar = tqdm.tqdm(total=len(val_dataloader))

  with torch.no_grad():
    final_loss, final_iou, _opt = run_loop(model, device, val_dataloader, config["val_batch_size"], scaler, loss_func, opt=None)

  del val_dataloader
  del val_dataset

# Based on: https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
def compute_iou(output, target):
  intersection = torch.logical_and(output, target)
  union = torch.logical_or(output, target)
  iou_score = torch.sum(intersection) / torch.sum(union)
  # print(f'IoU is {iou_score}')

  return iou_score

# https://towardsdatascience.com/choosing-and-customizing-loss-functions-for-image-processing-a0e4bf665b0a
# https://stackoverflow.com/questions/47084179/how-to-calculate-multi-class-dice-coefficient-for-multiclass-image-segmentation
def dice_coef(y_true, y_pred, epsilon=1e-6):
"""Altered Sorensen–Dice coefficient with epsilon for smoothing."""
    y_true_flatten = np.asarray(y_true).astype(np.bool)
    y_pred_flatten = np.asarray(y_pred).astype(np.bool)

    if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
        return 1.0

    return (2. * np.sum(y_true_flatten * y_pred_flatten)) /\
           (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon)

################################################################################
# Main Thread                                                                  #
################################################################################

print('Custom train deeplabv3 ...')

if len(sys.argv) == 3: # params: model config
  config_path = sys.argv[2]
elif len(sys.argv) == 2: # params: either model or config
  config_path= sys.argv[1]
else:
  raise RuntimeError.new("Invalid Parameters, please add either the model path, config path or both")

print(f'Config/Model path: {config_path}')
config, start_epoch = open_config(config_path)

# Used for pre-fetching
outer_batch_size = config["batch_size"] * config['outer_batch_size_multiplier']
betas = (config["beta_1"], config["beta_2"])
config["betas"] = betas
save_path = config["save_path"]

print(f'Config: {config}')

# Load devices
print(f'Cuda available? {torch.cuda.is_available()}')

dev = torch.device('cpu')
summary_dev = 'cpu'

if torch.cuda.is_available():
  dev = torch.device('cuda')
  summary_dev = 'cuda'

if __name__ == '__main__':
  try:
    torch.multiprocessing.set_start_method('spawn')
  except RuntimeError:
    pass

  model, opt = initialise_model(dev, config)

  if config["load_model"]:
    model = load(model, opt, model_path) # Load model

  # Based on reference code
  loss_func = nn.functional.cross_entropy # TODO: Add in weight
  # Dice Co-Efficient

  pbar = tqdm.tqdm(total=config["epochs"])
  for epoch in range(start_epoch, config["epochs"], 1):
    pbar.write('Training Phase:')
    model, opt = train(model, dev, loss_func, opt, epoch, outer_batch_size)

    pbar.write('Validation Phase:')
    # If you need to purge memory
    # gc.collect() # Force the Training data to be unloaded. Loading data takes ~10 secs
    # time.sleep(30)
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    validate(model, dev, loss_func, epoch, outer_batch_size)

    pbar.write(f'Save epoch {epoch}.')
    save(model, opt, epoch, config, save_path)

    pbar.update(1)

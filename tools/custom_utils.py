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

# import torchvision.transforms as transforms
from torchvision import models
from torchinfo import summary

import sys
sys.path.insert(1, '../references/segmentation/')

from from_games_dataset import FromGamesDataset

# Reference Code
import presets
import utils
from coco_utils import get_coco
import transforms as T

import deeplabv3_metrics

SMOOTH = 1e-4 #1e-6 # Beware Float16!

################################################################################
# Helper Methods                                                               #
################################################################################
def clear_gpu():
  # If you need to purge memory
  gc.collect() # Force the Training data to be unloaded. Loading data takes ~10 secs
  # time.sleep(15) # 30
  # torch.cuda.empty_cache() # Will nuke the model in-memory
  torch.cuda.synchronize() # Force the unload before the next step

def fetch_device():
  print(f'Cuda available? {torch.cuda.is_available()}')
  dev = torch.device('cpu')
  summary_dev = 'cpu'

  if torch.cuda.is_available():
    dev = torch.device('cuda')
    summary_dev = 'cuda'

  return [dev, summary_dev]

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
    model_path = None
  except:
    checkpoint = torch.load(path)
    config = checkpoint['args']
    epoch = checkpoint['epoch'] + 1
    model_path = path

  config["save_path"] = f'{config["save_path"]}/{config["dataset"]}'
  create_folder(config["save_path"])

  return [config, epoch, model_path]

def create_folder(path):
  Path(path).mkdir(parents=True, exist_ok=True)

def save_csv(file_path, csv_data):
  with open(file_path, 'a') as f:
    f.write(f'{csv_data}\n')

################################################################################
# Image Transforms                                                             #
################################################################################
# https://discuss.pytorch.org/t/normalising-images-in-cityscapes-using-mean-and-std-of-imagenet/120556
def cityscapes_transforms():
  mean = (0.485, 0.456, 0.406) # Taken from COCO reference
  std = (0.229, 0.224, 0.225)

  # https://stackoverflow.com/questions/49356743/how-to-train-tensorflows-deeplab-model-on-cityscapes
  transforms_arr = T.Compose(
    [
      T.RandomCrop(460), # 480 # 513 # 520
      T.PILToTensor(),
      T.ConvertImageDtype(torch.float),
      T.Normalize(mean=mean, std=std),
    ]
  )

  return transforms_arr

def rescale_pixels(image, scale=1./255.):
  return image * scale

def undo_rescale_pixels(image, scale=1./255.):
  return image * (1/scale)

################################################################################
# Dataset Loading                                                              #
################################################################################
def cityscapes_collate(batch):
  images, targets = list(zip(*batch))

  # images = np.array([(rescale_pixels(i.numpy())) for i in images]) # Cant rescale if want to maintain compatibility
  images = np.array([(i.numpy()) for i in images])
  targets = np.array([(t.numpy()) for t in targets])

  return torch.from_numpy(images), torch.from_numpy(targets)

def load_dataset(config, root, image_set, category_list=None, batch_size=1, training=False):
  if config["dataset"] == "COCO16" or config["dataset"] == "COCO21":
    dataset = load_coco(root, image_set, category_list=category_list)
  elif config["dataset"] == "cityscapes":
    dataset = torchvision.datasets.Cityscapes(root, split=image_set, mode=config["cityscapes_mode"], target_type='semantic', transforms=cityscapes_transforms()) # TODO: Cityscapes 'test'
  elif config["dataset"] == "fromgames":
    dataset = FromGamesDataset(root, split=image_set, transforms=cityscapes_transforms())
  # elif config["dataset"] == "mixed":
    # TODO: Mix here

  sample_size = len(dataset) * config["sample_percentage"]
  if training:
    if sample_size < batch_size:
      sample_size = len(dataset)

    subset_idex = list(range(int(sample_size))) # TODO: Unload others
    subset = torch.utils.data.Subset(dataset, subset_idex)

    if config["dataset"] == "cityscapes":
      dataloader = DataLoader(subset, batch_size=batch_size, shuffle=config["shuffle"], drop_last=config["drop_last"], collate_fn=cityscapes_collate, num_workers=config["train_num_workers"], pin_memory=config["pin_memory"])
    else:
      dataloader = DataLoader(subset, batch_size=batch_size, shuffle=config["shuffle"], drop_last=config["drop_last"], collate_fn=utils.collate_fn, num_workers=config["train_num_workers"], pin_memory=config["pin_memory"])
  else:
    if config["dataset"] == "cityscapes":
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=config["shuffle"], drop_last=config["drop_last"], collate_fn=cityscapes_collate, num_workers=config["val_num_workers"], pin_memory=config["pin_memory"])
    else:
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=config["shuffle"], drop_last=config["drop_last"], collate_fn=utils.collate_fn, num_workers=config["val_num_workers"], pin_memory=config["pin_memory"])

  # print(f'Number of data points for {image_set}: {len(dataloader)}')
  return [dataset, dataloader]

def fetch_category_list(config):
  if config["dataset"] == "COCO16":
    return  [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16] # New list of categories
  elif config["dataset"] == "COCO21":
    return [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72] # Original List
  elif config["dataset"] == "cityscapes" or config["dataset"] == "fromgames":
    # return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, -1] # Cityscapes
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33] # Cityscapes

# COCO Dataset
# train_image_path = '/data/data/coco/data_raw/train2017'
# val_image_path = '/data/data/coco/data_raw/val2017'
# train_annotation_path = '/data/data/coco/zips/annotations/annotations/instances_train2017.json'
# val_annotation_path = '/data/data/coco/zips/annotations/annotations/instances_val2017.json'

# train_dataset = torchvision.datasets.CocoDetection(train_image_path, train_annotation_path)
# val_dataset = torchvision.datasets.CocoDetection(val_image_path, val_annotation_path)
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, persistent_workers=False)
def load_coco(root, image_set, category_list=None):
  # Using reference code
  # See Readme.md for new category list

  if category_list is None:
    category_list = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72] # Default
    # category_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21] # New

  if image_set == 'train':
    transforms = presets.SegmentationPresetTrain(base_size=520, crop_size=480)
  else:
    transforms = presets.SegmentationPresetEval(base_size=520)

  return get_coco(root, image_set, transforms, category_list=category_list)

################################################################################
# Model Functions                                                              #
################################################################################
# Used to load pre-trained models or return None (So we can construct the new ones)
def fetch(model):
  if model == 'ResNet50':
    return models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=21)
  elif model == 'ResNet101':
    return models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)

  return None

def xavier_uniform_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_uniform_(layer.weight, gain=gain)

def initialise_model(dev, config, pretrained=False, num_classes=21, randomise=True):
  if config["selected_model"] == 'deeplabv3_resnet101':
    model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, num_classes=num_classes)
  elif config["selected_model"] == 'deeplabv3_resnet50':
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
  else:
    raise RuntimeError("Invalid model selected.")

  # Randomise weights
  if randomise:
    model.apply(xavier_uniform_init)

  model = model.to(dev)

  # Reference code uses SGD
  # https://www.programmersought.com/article/22245145270/
  if config["opt_function"] == 'ADAM':
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=config["betas"], eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    print('ADAM Optimizer is selected!')
  # if config["opt_function"] == 'RMSprop':
    # https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
  else:
    opt = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    print('SGD Optimizer is selected!')

  return [model, opt]

def load(model, device, path, opt=None, show_stats=True):
  # Load model weights
  # Training crashed when lr dropped to complex numbers

  # TODO: Load optimizer
  print(f'Loading model from: {path}')
  checkpoint = torch.load(path)
  epoch = checkpoint['epoch']
  model.load_state_dict(checkpoint['model'], strict=False)

  if opt:
    opt.load_state_dict(checkpoint['optimizer'])

  model.to(device)

  print(f'Model loaded into {device}!')

  if show_stats:
    model_stats = summary(model, device=device)

  return model, opt, epoch

# Built to be compatible with reference code
def save(model, opt, lr_scheduler, epoch, config, save_path):
  checkpoint = {
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "epoch": epoch,
    "args": config,
  }

  if lr_scheduler is not None:
    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()

  torch.save(checkpoint, os.path.join(save_path, f"model_{epoch}.pth"))

################################################################################
# Loss Functions                                                               #
################################################################################
# Based on: https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
def compute_iou1(output, target):
  intersection = torch.logical_and(output, target)
  union = torch.logical_or(output, target)
  iou_score = torch.sum(intersection) / torch.sum(union)
  # print(f'IoU is {iou_score}')

  return iou_score

# Based on: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy/script
def compute_iou2(outputs: torch.Tensor, labels: torch.Tensor):
  # You can comment out this line if you are passing tensors of equal shape
  # But if you are passing output from UNet or something it will most probably
  # be with the BATCH x 1 x H x W shape
  outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

  intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
  union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

  iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

  thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

  return thresholded.mean() #thresholded  # Or thresholded.mean() if you are interested in average across the batch

# TODO: Convert to Tensor
# Based on: https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
def compute_iou3(output, target, num_classes):
  pred = np.asarray(output.cpu(), dtype=np.uint8).copy()
  mask = np.asarray(target.cpu(), dtype=np.uint8).copy()
  # 255 -> 0
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_classes, range=(1, num_classes))
  (area_pred, _) = np.histogram(pred, bins=num_classes, range=(1, num_classes))
  (area_mask, _) = np.histogram(mask, bins=num_classes, range=(1, num_classes))
  area_union = area_pred + area_mask - area_inter

  # return (area_inter, area_union)
  return ((area_inter + SMOOTH) / (area_union + SMOOTH)).mean()

# TODO: IOU SKLearn

# https://towardsdatascience.com/choosing-and-customizing-loss-functions-for-image-processing-a0e4bf665b0a
# https://stackoverflow.com/questions/47084179/how-to-calculate-multi-class-dice-coefficient-for-multiclass-image-segmentation
# Dice Co-Efficient
def dice_coef(y_true, y_pred, epsilon=SMOOTH): # 1e-6 wont work for float16
  # Altered Sorensen–Dice coefficient with epsilon for smoothing.
  y_true_flatten = y_true.to(torch.bool)
  y_pred_flatten = y_pred.to(torch.bool)

  if not torch.sum(y_true_flatten) + torch.sum(y_pred_flatten):
    return 1.0

  return (2. * torch.sum(y_true_flatten * y_pred_flatten)) / (torch.sum(y_true_flatten) + torch.sum(y_pred_flatten) + epsilon)

################################################################################
# Train and Eval                                                               #
################################################################################
def loss_batch(model, device, scaler, loss_func, xb, yb, opt=None, num_classes=None):
  if opt is not None:
    # opt.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
    for param in model.parameters(): # Optimisation to save n operations
      param.grad = None

  with torch.cuda.amp.autocast(enabled=True, cache_enabled=True): # TODO: cache_enabled
    input = xb.to(device)
    prediction = model(input)

    # Compute Loss
    output = prediction['out']
    target = yb.to(device)

    loss = loss_func(output, target, ignore_index=255)

    max_output = output.argmax(1)
    dice_loss = dice_coef(target, max_output)

    iou_score1 = compute_iou1(max_output, target)
    iou_score2 = compute_iou2(max_output, target)
    iou_score3 = compute_iou3(max_output, target, num_classes)

    # Test other calcs
    # sum_pixel_accuracy = 0.0
    # sum_mean_accuracy = 0.0
    # sum_mean_IU = 0.0
    # sum_frequency_weighted_IU = 0.0

    # # TODO: Turn into Torch
    # result_batch_size = output.shape[0]
    # np_max_output = max_output.cpu().numpy()
    # np_target = target.cpu().numpy()

    # # Iterate through batch
    # for i in range(result_batch_size):
    #   pixel_accuracy = deeplabv3_metrics.pixel_accuracy(np_max_output[i], np_target[i])
    #   mean_accuracy = deeplabv3_metrics.mean_accuracy(np_max_output[i], np_target[i])
    #   mean_IU = deeplabv3_metrics.pixel_accuracy(np_max_output[i], np_target[i])
    #   frequency_weighted_IU = deeplabv3_metrics.pixel_accuracy(np_max_output[i], np_target[i])

    #   sum_pixel_accuracy += pixel_accuracy
    #   sum_mean_accuracy += mean_accuracy
    #   sum_mean_IU += mean_IU
    #   sum_frequency_weighted_IU += frequency_weighted_IU

    # pixel_accuracy = sum_pixel_accuracy / result_batch_size
    # mean_accuracy = sum_mean_accuracy / result_batch_size
    # mean_IU = sum_mean_IU / result_batch_size
    # frequency_weighted_IU = sum_frequency_weighted_IU / result_batch_size

    del input
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

  # Check if loss.detach() is better
  # cpu().item()
  return [loss.detach(), dice_loss, iou_score1, iou_score2, iou_score3, opt]

def run_loop(model, device, dataloader, batch_size, scaler, loss_func, epoch, config, opt=None, save=True):
  sum_of_loss = 0.0

  sum_of_iou1 = 0.0
  sum_of_iou2 = 0.0
  sum_of_iou3 = 0.0

  sum_of_dice = 0.0

  category_list = fetch_category_list(config)
  num_classes = len(category_list)

  pbar = tqdm.tqdm(total=len(dataloader))

  # TODO: Allow disabling sub-batches
  if opt is None: # Validation
    # Dont use sub-batches
    for xb, yb in dataloader:
      loss, dice_loss, iou_score1, iou_score2, iou_score3, _opt = loss_batch(model, device, scaler, loss_func, xb, yb, opt=opt, num_classes=num_classes)

      sum_of_loss += loss

      sum_of_iou1 += iou_score1
      sum_of_iou2 += iou_score2
      sum_of_iou3 += iou_score3

      sum_of_dice += dice_loss

      pbar.update(1)

    final_loss = sum_of_loss / len(dataloader)

    final_iou1 = sum_of_iou1 / len(dataloader)
    final_iou2 = sum_of_iou2 / len(dataloader)
    final_iou3 = sum_of_iou3 / len(dataloader)

    final_dice = sum_of_dice / len(dataloader)

    pbar.write(f'Epoch {epoch} val loss: {final_loss} val IoU1: {final_iou1} val IoU2: {final_iou2} val IoU3: {final_iou3} val dice: {final_dice}')

    if save:
      val_csv_path = f'{config["save_path"]}/val_loss.csv'
      save_csv(val_csv_path, f'{epoch},{final_loss},{final_iou1},{final_iou2},{final_iou3},{final_dice}')

  else: # Use sub-batches / Training
    curr_lr = opt.param_groups[0]["lr"]

    for inner_batch in dataloader:
      for i in range(0, inner_batch[0].shape[0], batch_size):
        xb = inner_batch[0][i:i+batch_size]
        yb = inner_batch[1][i:i+batch_size]

        loss, dice_loss, iou_score1, iou_score2, iou_score3, opt = loss_batch(model, device, scaler, loss_func, xb, yb, opt=opt, num_classes=num_classes)

        sum_of_loss += loss

        sum_of_iou1 += iou_score1
        sum_of_iou2 += iou_score2
        sum_of_iou3 += iou_score3

        sum_of_dice += dice_loss

      pbar.update(1)

    final_loss = sum_of_loss / (len(dataloader) * batch_size)

    final_iou1 = sum_of_iou1 / (len(dataloader) * batch_size)
    final_iou2 = sum_of_iou2 / (len(dataloader) * batch_size)
    final_iou3 = sum_of_iou3 / (len(dataloader) * batch_size)

    final_dice = sum_of_dice / (len(dataloader) * batch_size)

    pbar.write(f'Epoch {epoch} train loss: {final_loss} train IoU1: {final_iou1} train IoU2: {final_iou2} train IoU3: {final_iou3} train dice: {final_dice} lr: {curr_lr}')

    train_csv_path = f'{config["save_path"]}/train_loss.csv'

    if save:
      save_csv(train_csv_path, f'{epoch},{final_loss},{final_iou1},{final_iou2},{final_iou3},{final_dice},{curr_lr}')

  return [final_loss, final_dice, final_iou1, final_iou2, final_iou3, opt]

def train(model, device, loss_func, opt, epoch, config, outer_batch_size, category_list=None):
  # Load Data - in train step to save memory
  train_dataset, train_dataloader = load_dataset(config, config['dataset_path'], 'train', category_list=category_list, batch_size=outer_batch_size, training=True)

  model = model.train()
  scaler = torch.cuda.amp.GradScaler(enabled=True)

  final_loss, final_dice, final_iou1, final_iou2, final_iou3, opt = run_loop(model, device, train_dataloader, config["batch_size"], scaler, loss_func, epoch, config, opt=opt)

  del train_dataloader
  del train_dataset

  return [model, opt]

def validate(model, device, loss_func, lr_scheduler, epoch, config, category_list=None, save=True):
  # Load Data - in val step to save memory
  val_dataset, val_dataloader = load_dataset(config, config['dataset_path'], 'val', category_list=category_list, batch_size=config["val_batch_size"], training=False)

  model = model.eval()
  scaler = torch.cuda.amp.GradScaler(enabled=True)

  sum_of_loss = 0.0
  sum_of_iou = 0.0

  with torch.no_grad():
    final_loss, final_dice, final_iou1, final_iou2, final_iou3, _opt = run_loop(model, device, val_dataloader, config["val_batch_size"], scaler, loss_func, epoch, config, opt=None, save=save)

  del val_dataloader
  del val_dataset

  # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
  if lr_scheduler is not None and config["opt_function"] == 'SGD':
    lr_scheduler.step(final_loss) # Use the average val loss for the batch

  return final_loss, final_dice, final_iou1, final_iou2, final_iou3

# https://stackoverflow.com/questions/63892031/how-to-train-deeplabv3-on-custom-dataset-on-pytorch
import time
import numpy as np

import torch
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

print('Custom train deeplabv3 ...')

# TODO: Add in custom configuration / args
selected_model = 'deeplabv3_resnet50'
# selected_model = 'deeplabv3_resnet101'
print(f'Selected Model: {selected_model}')

# TODO: Sub batch
batch_size = 16
print(f'Batch Size: {batch_size}')

epochs = 1
print(f'Epochs: {epochs}')

sample_percentage = 0.01 # 0.1 # 1.0
print(f'Data sample percent: {sample_percentage}')

load_model = False

# ResNet50
# model_path = '/mnt/excelsior/trained_models/deeplabv3_resnet50/model_2.pth'
# model_path = '/mnt/excelsior/trained_models/deeplabv3_resnet50/model_25.pth' # NaN output ;()

# ResNet101
model_path = '/mnt/excelsior/trained_models/deeplabv3_resnet101/model_2.pth'
# model_path = '/mnt/excelsior/trained_models/deeplabv3_resnet101/model_18.pth'

# Load devices
print(f'Cuda available? {torch.cuda.is_available()}')

dev = torch.device('cpu')
summary_dev = 'cpu'

if torch.cuda.is_available():
  dev = torch.device('cuda')
  summary_dev = 'cuda'

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

def initialise_model(selected_model, dev, pretrained=False, num_classes=21):
  if selected_model == 'deeplabv3_resnet101':
    model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, num_classes=num_classes)
  elif selected_model == 'deeplabv3_resnet50':
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
  else:
    raise RuntimeError("Invalid model selected.")

  # Randomise weights
  model.apply(xavier_uniform_init)
  return model.to(dev)

# https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html
def load(model, path):
  # Load model weights
  # Training crashed when lr dropped to complex numbers

  # TODO: Load optimizer
  print(f'Loading model from: {path}')
  model.load_state_dict(torch.load(path)['model'], strict=False)
  model.to(dev)

  print(f'Model loaded into {summary_dev}!')
  model_stats = summary(model, device=summary_dev)

  return model

def loss_batch(model, device, scaler, loss_func, xb, yb):
  input = xb.to(device)
  prediction = model(input)

  del input

  output = prediction['out']
  target = yb.to(device)

  loss = loss_func(output, target, ignore_index=255)

  sum_batch_iou_score = 0.0
  for i in range(output.shape[0]):
    sum_batch_iou_score += compute_iou(output[i], target[i]).cpu()

  iou_score = sum_batch_iou_score / output.shape[0]

  del output
  del target

  return [loss.cpu().item(), iou_score]

def train(model, dev, train_dataloader, loss_func, epoch):
  model = model.train()
  scaler = torch.cuda.amp.GradScaler(enabled=True)

  sum_of_loss = 0.0
  sum_of_iou = 0.0

  with torch.cuda.amp.autocast(enabled=True, cache_enabled=True): # TODO: cache_enabled
    pbar = tqdm.tqdm(total=len(train_dataloader))

    for xb, yb in train_dataloader:
      loss, iou_score = loss_batch(model, dev, scaler, loss_func, xb, yb)

      sum_of_loss += loss
      sum_of_iou += iou_score
      pbar.update(1)

  # TODO: Save loss
  final_loss = sum_of_loss / len(val_dataloader)
  final_iou = sum_of_iou / len(val_dataloader)

  pbar.write(f'Epoch {epoch} train loss: {final_loss} train IoU: {final_iou}')
  return model

# TODO: Save results
# TODO: Early stopping
# TODO: Dynamic lr
def validate(model, dev, val_dataloader, loss_func, epoch):
  model = model.eval()
  scaler = torch.cuda.amp.GradScaler(enabled=True)

  sum_of_loss = 0.0
  sum_of_iou = 0.0

  pbar = tqdm.tqdm(total=len(val_dataloader))

  with torch.no_grad():
    for xb, yb in val_dataloader:
      loss, iou_score = loss_batch(model, dev, scaler, loss_func, xb, yb)

      sum_of_loss += loss
      sum_of_iou += iou_score
      pbar.update(1)

  final_loss = sum_of_loss / len(val_dataloader)
  final_iou = sum_of_iou / len(val_dataloader)

  pbar.write(f'Epoch {epoch} val loss: {final_loss} val IoU: {final_iou}')

def process_image(image, device, size=()):
  preprocess = transforms.Compose([
    # transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  # image = image.convert('RGB')
  image_batch = preprocess(image).unsqueeze(0)
  # image.close()

  return image_batch.to(device)

# Based on: https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
def compute_iou(output, target):
  intersection = torch.logical_and(output, target)
  union = torch.logical_or(output, target)
  iou_score = torch.sum(intersection) / torch.sum(union)
  # print(f'IoU is {iou_score}')

  return iou_score

def test_IOU(model, dataset):
  model = model.eval()

  # Super inefficient to do this in-memory but will use less HDD
  # Since we only run one epoch/run
  sum_of_iou = 0.0
  sum_of_data_load_time = 0.0

  with torch.no_grad():
    for image, target in tqdm.tqdm(dataset):
      start_data_time = time.time()
      image_batch = image.unsqueeze(0).to(dev) #process_image(image, dev)

      data_load_time = time.time() - start_data_time
      sum_of_data_load_time += data_load_time

      # Make prediction
      output = model(image_batch)
      output = output["out"]

      iou_score = compute_iou(output, target.to(dev))
      sum_of_iou += iou_score.cpu()

  average_iou = sum_of_iou / len(dataset)
  average_data_load_time = sum_of_data_load_time / len(dataset)

  # print(f'Total IoU: {sum_of_iou}')
  print(f'Average IOU: {average_iou}')

  # print(f'Total Data Load Time: {sum_of_data_load_time}')
  # print(f'Average Data Load Time: {average_data_load_time}')

  return average_iou

# Setup Data
train_dataset = load_coco('/mnt/scratch_disk/data/coco/data_raw/', 'train')
subset_idex = list(range(int(len(train_dataset) * sample_percentage))) # TODO: Unload others
train_subset = torch.utils.data.Subset(train_dataset, subset_idex)
train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=utils.collate_fn)

# TODO: Load separately
val_dataset = load_coco('/mnt/scratch_disk/data/coco/data_raw/', 'val')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=utils.collate_fn)

model = initialise_model(selected_model, dev)

if load_model:
  model = load(model, model_path) # Load model
else: # Train model to be better
  print("Train model ...")

  loss_func = nn.functional.cross_entropy
  for epoch in tqdm.tqdm(range(epochs)):
    model = train(model, dev, train_dataloader, loss_func, epoch)
    validate(model, dev, val_dataloader, loss_func, epoch)

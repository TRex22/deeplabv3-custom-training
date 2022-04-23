# https://stackoverflow.com/questions/63892031/how-to-train-deeplabv3-on-custom-dataset-on-pytorch
import time
import numpy as np

import torch
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
from coco_utils import get_coco

# selected_model = 'deeplabv3_resnet50'
selected_model = 'deeplabv3_resnet101'
print(selected_model)

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
  # Use reference
  transforms = presets.SegmentationPresetEval(base_size=520)
  return get_coco(root, image_set, transforms)

def initialise_model(selected_model, pretrained=False, num_classes=21):
  if selected_model == 'deeplabv3_resnet101':
    return models.segmentation.deeplabv3_resnet101(pretrained=pretrained, num_classes=num_classes)
  elif selected_model == 'deeplabv3_resnet50':
    return models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
  else:
    raise RuntimeError("Invalid model selected.")

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

def

# def train(model, train_dataset):


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

def test_IOU(model, dataset)
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

  print(f'Total IoU: {sum_of_iou}')
  print(f'Average IOU: {average_iou}')

  print(f'Total Data Load Time: {sum_of_data_load_time}')
  print(f'Average Data Load Time: {average_data_load_time}')

train_dataset = load_coco('/mnt/scratch_disk/data/coco/data_raw/', 'train')
val_dataset = load_coco('/mnt/scratch_disk/data/coco/data_raw/', 'val')

model = initialise_model(selected_model)

if load_model:
  model = load(model, model_path) # Load model
else: # Train model to be better
  model = train(model, train_dataset)

# Run test on COCO
test_IOU(model, val_dataset)

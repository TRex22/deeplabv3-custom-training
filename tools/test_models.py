# https://stackoverflow.com/questions/63892031/how-to-train-deeplabv3-on-custom-dataset-on-pytorch
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader

import tqdm

import torchvision.transforms as transforms
from torchvision import models
from torchinfo import summary

import sys
sys.path.insert(1, '../')

selected_model = 'deeplabv3_resnet50'
# selected_model = 'deeplabv3_resnet101'
print(selected_model)

# COCO Dataset
# train_image_path = '/data/data/coco/data_raw/train2017'
val_image_path = '/data/data/coco/data_raw/val2017'
# train_annotation_path = '/mnt/excelsior/data/coco/zips/annotations/annotations/instances_train2017.json'
val_annotation_path = '/data/data/coco/zips/annotations/annotations/instances_val2017.json'

# train_dataset = torchvision.datasets.CocoDetection(train_image_path, train_annotation_path)
val_dataset = torchvision.datasets.CocoDetection(val_image_path, val_annotation_path)
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, persistent_workers=False)

# Load devices
print(f'Cuda available? {torch.cuda.is_available()}')

dev = torch.device('cpu')
summary_dev = 'cpu'

if torch.cuda.is_available():
  dev = torch.device('cuda:0')
  summary_dev = 'cuda'

# Load model
model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=21)
# model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)

# Load model weights
# Training crashed when lr dropped to complex numbers
model_path = '/mnt/excelsior/trained_models/deeplabv3_resnet50/model_25.pth'
# model_path = '/mnt/excelsior/trained_models/deeplabv3_resnet101/model_18.pth'
model.load_state_dict(torch.load(model_path)['model'], strict=False)
model.to(dev)

print(f"Model loaded into {summary_dev}!")
model_stats = summary(model, device=summary_dev)

# Model evaluate
model = model.eval()

# Super inefficient to do this in-memory but will use less HDD
# Since we only run one epoch/run
def process_image(image, device, size=()):
  preprocess = transforms.Compose([
    # transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  image = image.convert('RGB')
  image_batch = preprocess(image).unsqueeze(0)
  image.close()

  return image_batch.to(device)

# Run test on COCO
with torch.no_grad():
  for image, target in tqdm.tqdm(val_dataset):
    image_batch = process_image(image, dev)
    # target = torch.Tensor(np.array(target)).to(dev) # process_image(target, dev)

    output = model(image_batch)
    output = output["out"]

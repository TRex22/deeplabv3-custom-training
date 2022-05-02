import sys
import cv2
import re

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

import custom_utils

# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
cityscapes_colours = [
  [  0,  0,  0],  # "unlabeled"
  [  0,  0,  0],  # "ego vehicle"
  [  0,  0,  0],  # "rectification border"
  [  0,  0,  0],  # "out of roi"
  [  0,  0,  0],  # "static"
  [111, 74,  0],  # "dynamic"
  [ 81,  0, 81],  # "ground"
  [128, 64,128],  # "road"
  [244, 35,232],  # "sidewalk"
  [250,170,160],  # "parking"
  [230,150,140],  # "rail track"
  [ 70, 70, 70],  # "building"
  [102,102,156],  # "wall"
  [190,153,153],  # "fence"
  [180,165,180],  # "guard rail"
  [150,100,100],  # "bridge"
  [150,120, 90],  # "tunnel"
  [153,153,153],  # "pole"
  [153,153,153],  # "polegroup"
  [250,170, 30],  # "traffic light"
  [220,220,  0],  # "traffic sign"
  [107,142, 35],  # "vegetation"
  [152,251,152],  # "terrain"
  [ 70,130,180],  # "sky"
  [220, 20, 60],  # "person"
  [255,  0,  0],  # "rider"
  [  0,  0,142],  # "car"
  [  0,  0, 70],  # "truck"
  [  0, 60,100],  # "bus"
  [  0,  0, 90],  # "caravan"
  [  0,  0,110],  # "trailer"
  [  0, 80,100],  # "train"
  [  0,  0,230],  # "motorcycle"
  [119, 11, 32],  # "bicycle"
  [  0,  0,142]   # "license plate" ... Unused
]

coco_21_colours = [
  (0, 0, 0),       # __background__
  (128, 0, 0),     # person
  (0, 128, 0),     # bicycle
  (128, 128, 0),   # car
  (0, 0, 128),     # motorcycle
  (128, 0, 128),   # airplane
  (0, 128, 128),   # bus
  (128, 128, 128), # train
  (64, 0, 0),      # truck
  (192, 0, 0),     # boat
  (64, 128, 0),    # traffic light
  (192, 128, 0),   # fire hydrant
  (64, 0, 128),    # N/A
  (192, 0, 128),   # stop sign
  (64, 128, 128),  # parking meter
  (192, 128, 128), # bench
  (0, 64, 0),      # bird
  (128, 64, 0),    # cat
  (0, 192, 0),     # dog
  (128, 192, 0),   # horse
  (0, 64, 128),    # sheep
]

def convert_segmentation_to_colour(segmentation_map, label_set='cityscapes'):
  width, height = segmentation_map.shape[0:2]
  colour_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3))

  for w in range(width):
    for h in range(height):
      if label_set == 'cityscapes':
        colour_segmentation_map[w][h] = cityscapes_colours[segmentation_map[w][h][0]]
      else:
        colour_segmentation_map[w][h] = coco_21_colours[segmentation_map[w][h][0]]

  return colour_segmentation_map


print('Generate Segmentation ...')

if len(sys.argv) != 4:
  raise RuntimeError("Invalid Parameters, model_path, input_image_path and save_path are required.")

model_path = sys.argv[1]
input_image_path = sys.argv[2]
save_path = sys.argv[3]

# Attempt to load pre-trained model
print(f'Model path: {model_path}')
label_set = 'COCO21'
model = custom_utils.fetch(model_path)
device, summary_dev = custom_utils.fetch_device()

if model is None:
  config, _start_epoch, _model_path = custom_utils.open_config(model_path)
  category_list = custom_utils.fetch_category_list(config)

  config["batch_size"] = 1
  config["val_batch_size"] = 1
  config["val_num_workers"] = 1
  config["train_num_workers"] = 0
  config["val_num_workers"] = 0

  label_set = 'cityscapes'
  model, opt = custom_utils.initialise_model(device, config, num_classes=len(category_list))
  model, _opt = custom_utils.load(model, opt, device, model_path) # Load model

model = model.to(device)
model = model.eval()

preprocess = transforms.Compose([
  transforms.Resize(460), # 480 # 513 # 520
  # transforms.ToTensor(),
  transforms.PILToTensor(),
  transforms.ConvertImageDtype(torch.float),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open(input_image_path).convert('RGB')
image_shape = np.array(img).shape
input = preprocess(img).to(device).unsqueeze(0)

img.close()

# input = torchvision.transforms.functional.to_tensor(image).to(device).unsqueeze(0)
prediction = model(input)

output = np.transpose(prediction['out'].argmax(1).cpu().numpy())
segmentation = convert_segmentation_to_colour(output, label_set=label_set)

torch.save(output, f'{save_path}/raw_output.pth')

output_image = cv2.rotate(segmentation, cv2.ROTATE_90_COUNTERCLOCKWISE)
output_image = cv2.flip(output_image, 0)
output_image = cv2.resize(output_image, (image_shape[1], image_shape[0]), cv2.INTER_NEAREST) # INTER_NEAREST for segmentations

filename = input_image_path.split('.')[0].split('/')[-1]
cv2.imwrite(f'{save_path}/{filename}_segmentation.png', output_image)

print('Complete!')

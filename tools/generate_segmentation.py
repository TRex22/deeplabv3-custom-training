import sys
import cv2
import torch
import torchvision

import numpy as np
from PIL import Image

import custom_utils

# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
cityscapes_colours = [
  [  0,  0,  0],
  [  0,  0,  0],
  [  0,  0,  0],
  [  0,  0,  0],
  [  0,  0,  0],
  [  0,  0,  0],
  [111, 74,  0],
  [ 81,  0, 81],
  [128, 64,128],
  [244, 35,232],
  [250,170,160],
  [230,150,140],
  [ 70, 70, 70],
  [102,102,156],
  [190,153,153],
  [180,165,180],
  [150,100,100],
  [150,120, 90],
  [153,153,153],
  [153,153,153],
  [250,170, 30],
  [220,220,  0],
  [107,142, 35],
  [152,251,152],
  [ 70,130,180],
  [220, 20, 60],
  [255,  0,  0],
  [  0,  0,142],
  [  0,  0, 70],
  [  0, 60,100],
  [  0,  0, 90],
  [  0,  0,110],
  [  0, 80,100],
  [  0,  0,230],
  [119, 11, 32],
  [  0,  0,142]
]

def convert_segmentation_to_colour(segmentation_map):
  width, height = segmentation_map.shape[0:2]
  colour_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3))

  for w in range(width):
    for h in range(height):
      colour_segmentation_map[w][h] = cityscapes_colours[segmentation_map[w][h][0]]

  return colour_segmentation_map


print('Generate Segmentation ...')

if len(sys.argv) != 4:
  raise RuntimeError("Invalid Parameters, model_path, input_image_path and save_path are required.")

model_path = sys.argv[1]
input_image_path = sys.argv[2]
save_path = sys.argv[3]

print(f'Model path: {model_path}')
config, _start_epoch = custom_utils.open_config(model_path)

device, summary_dev = custom_utils.fetch_device()

category_list = custom_utils.fetch_category_list(config)
model, opt = custom_utils.initialise_model(device, config, num_classes=len(category_list))

model, _opt = custom_utils.load(model, opt, device, model_path) # Load model
model = model.eval()

image = Image.open(input_image_path)
input = torchvision.transforms.functional.to_tensor(image).to(device).unsqueeze(0)
prediction = model(input)

output = np.transpose(prediction['out'].argmax(1).cpu().numpy())
segmentation = convert_segmentation_to_colour(output)

torch.save(output, f'{save_path}/raw_output.pth')
cv2.imwrite(f'{save_path}/segmentation.png', cv2.rotate(segmentation, cv2.ROTATE_180))

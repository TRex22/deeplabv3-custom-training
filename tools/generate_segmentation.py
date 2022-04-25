import sys
import torch
import torchvision
from PIL import Image

import custom_utils

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

model = custom_utils.load(model, opt, model_path) # Load model
model = model.eval()

image = Image.open(input_image_path)
input = torchvision.transforms.functional.to_tensor(image).to(device)
prediction = model(input)

output = prediction['out']

torch.save(output, f'{save_path}/raw_output.pth')

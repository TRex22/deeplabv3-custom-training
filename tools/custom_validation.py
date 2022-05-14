# Can take in 1 argument which is the model checkpoint path
# Will run that one model and print out the results

# Can also take in 2 arguments, where the first is the path to models and the
# second is path and filename of the validation results csv
import os
import gc
import sys
import time
import tqdm

import torch
from torch import nn

import custom_utils

################################################################################
# Optimisations                                                                #
################################################################################
# https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
torch.backends.cudnn.benchmark = True # Initial training steps will be slower
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

################################################################################
# Validate Strategies                                                          #
################################################################################
def validate(model_path, model, dev, loss_func, config, category_list, save_path=None):
  model, opt, epoch = custom_utils.load(model, opt, dev, model_path, show_stats=False)
  final_loss, final_iou = custom_utils.validate(model, dev, loss_func, None, epoch, config, category_list=category_list, save=False)

  if save_path:
    csv_data = f'{final_loss},{final_iou}'
    custom_utils.save_csv(save_path, csv_data)

################################################################################
# Main Thread                                                                  #
################################################################################
if __name__ == '__main__':
  print('Custom validate deeplabv3 ...')
  if len(sys.argv) < 2 and len(sys.argv) > 3:
    raise RuntimeError("Please provide the path to the model or the path to the folder of models and csv save path.")

  dev, summary_dev = custom_utils.fetch_device()

  model_path = sys.argv[1]
  print(f'Validate: { model_path}')

  config, _start_epoch, model_path = custom_utils.open_config( model_path)

  # Based on reference code
  loss_func = nn.functional.cross_entropy # TODO: Add in weight
  category_list = custom_utils.fetch_category_list(config)

  custom_utils.clear_gpu() # Needed to ensure no memory loss

  model, _opt = custom_utils.initialise_model(dev, config, num_classes=len(category_list), randomise=False)

  if len(sys.argv) == 2:
    validate(model_path, model, dev, loss_func, config, category_list)
  elif len(sys.argv) == 3:
    save_path = sys.argv[2]
    models = os.listdir(f'{model_path}/*.pth')

    for specific_model_path in tqdm.tqdm(models):
      validate(specific_model_path, model, dev, loss_func, config, category_list, save_path=save_path)

  print("Complete!")

  # TODO:
  # compute_iou1(output, target)
  # compute_iou2(outputs: torch.Tensor, labels: torch.Tensor)
  # compute_iou3(output, target, num_classes)

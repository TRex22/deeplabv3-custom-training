# Can take in 1 argument which is the model checkpoint path
# Will run that one model and print out the results

# Can also take in 2 arguments, where the first is the path to models and the
# second is path and filename of the validation results csv

# Can also take in 3 arguments, where the first is the path to models, the
# second is path and filename of the validation results csv, and the third
# is the number of runs to average

# TODO: Create a training evaluator

import os
import gc
import sys
import time
import tqdm

from natsort import natsorted

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
def validate(model_path, dev, loss_func, save_path=False, iteration=None):
  custom_utils.clear_gpu() # Needed to ensure no memory loss

  config, _start_epoch, model_path = custom_utils.open_config(model_path)
  category_list = custom_utils.fetch_category_list(config)

  model, _opt = custom_utils.initialise_model(dev, config, num_classes=len(category_list), randomise=False)

  model, _opt, epoch = custom_utils.load(model, dev, model_path, show_stats=False)
  final_loss, final_dice, final_iou1, final_iou2, final_iou3 = custom_utils.validate(model, dev, loss_func, None, epoch, config, category_list=category_list, save=False)

  if iteration:
    epoch = iteration

  if save_path:
    csv_data = f'{epoch},{final_loss},{final_iou1},{final_iou2},{final_iou3},{final_dice}'
    custom_utils.save_csv(save_path, csv_data)

################################################################################
# Main Thread                                                                  #
################################################################################
if __name__ == '__main__':
  print('Custom validate deeplabv3 ...')
  if len(sys.argv) < 2 or len(sys.argv) > 4:
    raise RuntimeError("Please provide: \n \
      The path to the model \n \
      The path to the folder of models and csv save path \n \
      The path to the model, csv save path and number of iterations to average")

  dev, summary_dev = custom_utils.fetch_device()

  model_path = sys.argv[1]
  print(f'Validate: {model_path}')

  # Based on reference code
  loss_func = nn.functional.cross_entropy # TODO: Add in weight

  if len(sys.argv) == 2:
    validate(model_path, dev, loss_func)
  elif len(sys.argv) == 3:
    save_path = sys.argv[2]
    models = os.listdir(f'{model_path}/')
    models = natsorted(models)

    for specific_model_path_name in tqdm.tqdm(models):
      _name, ext = os.path.splitext(specific_model_path_name)

      if ext == '.pth':
        validate(f'{model_path}/{specific_model_path_name}', dev, loss_func, save_path=save_path)
  elif len(sys.argv) == 4:
    save_path = sys.argv[2]
    number_of_iterations = Int(sys.argv[3])

    for i in tqdm.tqdm(range(number_of_iterations)):
      validate(model_path, dev, loss_func, save_path=save_path, iteration=i)

  print("Complete!")

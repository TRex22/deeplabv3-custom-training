# Usage: python3 custom_tarining.py ./config.json
# Usage: python3 custom_tarining.py ./model_42.pth
# Usage: python3 custom_tarining.py ./model_42.pth ./config.json # Will use the custom config

# TODO: Early stopping
# TODO: Dynamic lr

# https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_deeplabv3_resnet101.ipynb
# https://stackoverflow.com/questions/63892031/how-to-train-deeplabv3-on-custom-dataset-on-pytorch

import sys
import tqdm

import torch
from torch import nn

import custom_utils

################################################################################
# Main Thread                                                                  #
################################################################################

print('Custom train deeplabv3 ...')

if len(sys.argv) == 3: # params: model config
  config_path = sys.argv[2]
  model_path = sys.argv[1]
  config, start_epoch, _model_path = custom_utils.open_config(config_path)
  start_epoch = torch.load(model_path)['epoch'] + 1
elif len(sys.argv) == 2: # params: either model or config
  config_path = sys.argv[1]
  config, start_epoch, model_path = custom_utils.open_config(config_path)
else:
  raise RuntimeError("Invalid Parameters, please add either the model path, config path or both")

print(f'Config/Model path: {config_path}')

# Used for pre-fetching
outer_batch_size = config["batch_size"] * config['outer_batch_size_multiplier']
betas = (config["beta_1"], config["beta_2"])
config["betas"] = betas
save_path = config["save_path"]

print(f'Config: {config}')
print(f'start_epoch: {start_epoch}')

# Load devices
dev, summary_dev = custom_utils.fetch_device()

if __name__ == '__main__':
  try:
    torch.multiprocessing.set_start_method('spawn')
  except RuntimeError:
    pass

  category_list = custom_utils.fetch_category_list(config)
  model, opt = custom_utils.initialise_model(dev, config, num_classes=len(category_list))

  if model_path is not None:
    model, opt = custom_utils.load(model, opt, dev, model_path) # Load model

  # Based on reference code
  loss_func = nn.functional.cross_entropy # TODO: Add in weight
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience = 5, min_lr = 0.00001) # TODO: Make configurable

  pbar = tqdm.tqdm(total=config["epochs"] - start_epoch)
  for epoch in range(start_epoch, config["epochs"], 1):
    pbar.write('Training Phase:')
    model, opt = custom_utils.train(model, dev, loss_func, lr_scheduler, opt, epoch, config, outer_batch_size, category_list=category_list)

    pbar.write('Validation Phase:')
    # If you need to purge memory
    # gc.collect() # Force the Training data to be unloaded. Loading data takes ~10 secs
    # time.sleep(30)
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    custom_utils.validate(model, dev, loss_func, epoch, config, category_list=category_list)

    pbar.write(f'Save epoch {epoch}.')
    custom_utils.save(model, opt, epoch, config, save_path)

    pbar.update(1)

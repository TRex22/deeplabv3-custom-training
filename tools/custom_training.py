# Usage: python3 custom_tarining.py ./config.json
# Usage: python3 custom_tarining.py ./model_42.pth
# Usage: python3 custom_tarining.py ./model_42.pth ./config.json # Will use the custom config

# TODO: Early stopping
# TODO: Dynamic lr

# https://stackoverflow.com/questions/63892031/how-to-train-deeplabv3-on-custom-dataset-on-pytorch
import sys
import torch
import tqdm

import custom_utils

################################################################################
# Main Thread                                                                  #
################################################################################

print('Custom train deeplabv3 ...')

if len(sys.argv) == 3: # params: model config
  config_path = sys.argv[2]
elif len(sys.argv) == 2: # params: either model or config
  config_path= sys.argv[1]
else:
  raise RuntimeError.new("Invalid Parameters, please add either the model path, config path or both")

print(f'Config/Model path: {config_path}')
config, start_epoch = custom_utils.open_config(config_path)

# Used for pre-fetching
outer_batch_size = config["batch_size"] * config['outer_batch_size_multiplier']
betas = (config["beta_1"], config["beta_2"])
config["betas"] = betas
save_path = config["save_path"]

print(f'Config: {config}')

# Load devices
print(f'Cuda available? {torch.cuda.is_available()}')

dev = torch.device('cpu')
summary_dev = 'cpu'

if torch.cuda.is_available():
  dev = torch.device('cuda')
  summary_dev = 'cuda'

if __name__ == '__main__':
  try:
    torch.multiprocessing.set_start_method('spawn')
  except RuntimeError:
    pass

  model, opt = custom_utils.initialise_model(dev, config)

  if config["load_model"]:
    model = custom_utils.load(model, opt, model_path) # Load model

  # Based on reference code
  loss_func = nn.functional.cross_entropy # TODO: Add in weight
  # Dice Co-Efficient

  pbar = tqdm.tqdm(total=config["epochs"])
  for epoch in range(start_epoch, config["epochs"], 1):
    pbar.write('Training Phase:')
    model, opt = custom_utils.train(model, dev, loss_func, opt, epoch, outer_batch_size)

    pbar.write('Validation Phase:')
    # If you need to purge memory
    # gc.collect() # Force the Training data to be unloaded. Loading data takes ~10 secs
    # time.sleep(30)
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    custom_utils.validate(model, dev, loss_func, epoch, outer_batch_size)

    pbar.write(f'Save epoch {epoch}.')
    custom_utils.save(model, opt, epoch, config, save_path)

    pbar.update(1)

import os
from torchvision.io import read_image

# https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec # VaporWave
# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
class FromGamesDataset:
  def __init__(self, base_path, split=None, transforms=None):
    self.base_path = base_path
    self.transforms = transforms

    self.image_paths = os.listdir(f'{base_path}/images/')
    self.label_paths = os.listdir(f'{base_path}/labels/')

    self.count = len(self.image_paths)
    self.split = 0.8 # 80% for training, hardcode for now
    train_start_idex = int(self.count - (self.count * self.split))

    if split == 'train':
      self.image_paths = self.image_paths[train_start_idex:-1]
      self.label_paths = self.label_paths[train_start_idex:-1]
    elif split == 'val': # TODO: Test?
      self.image_paths = self.image_paths[0:train_start_idex]
      self.label_paths = self.label_paths[0:train_start_idex]

    self.count = len(self.image_paths)

  def __len__(self):
    return self.count

  def __getitem__(self, idx):
    img_path = f'{self.base_path}/{self.image_paths[idx]}'
    label_path = f'{self.base_path}/{self.label_paths[idx]}'

    image = read_image(img_path).permute(0, 2, 1).float() # .permute(0, 2, 1)
    label = read_image(label_path).permute(0, 2, 1).int() # .permute(0, 2, 1)

    if self.transforms:
      image = self.transforms(image)

    return (image, label) # Return as a tuple

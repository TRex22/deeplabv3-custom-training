# deeplabv3-custom-training
This repo exists to train the pyTorch Torchvision deeplabv3 models with custom labels and datasets.

I could not find a good set of trained models online so I decided to train my own.

My aim is to train on the main cityscapes train dataset as well as combine it with the from_games dataset to attempt to improve and generalise the performance of the models.

# Usage of pre-trained models
1. Download the trained model `.pth`
2. In your code:

```python
import torch
from torchvision import models

models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
```

# TODO
- [] Train on CityScapes (±600 epochs)
- [] Combine from_games Dataset
- [] Make clearer notes
- [] Upload models
- [] Make reference to mixed precision
- [] Make notes for cluster compute
- [] Attempt multi-node if required
- [] Add in evaluation code based on reference but using multiple models
- [] Add in links to reference code
- [√] Add in required dependencies and an environment

# Thanks
Computations were performed using High Performance Computing infrastructure provided by the Mathematical Sciences Support unit at the University of the Witwatersrand.

# References
## Cityscapes Dataset
Home Page: https://www.cityscapes-dataset.com/
```
@inproceedings{Cordts2016Cityscapes,
    title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
    author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
    booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016}
}
```

## from_games Dataset
Home Page: https://download.visinf.tu-darmstadt.de/data/from_games/

```
@InProceedings{Richter_2016_ECCV,
    author = {Stephan R. Richter and Vibhav Vineet and Stefan Roth and Vladlen Koltun},
    title = {Playing for Data: {G}round Truth from Computer Games},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2016},
    editor = {Bastian Leibe and Jiri Matas and Nicu Sebe and Max Welling},
    series = {LNCS},
    volume = {9906},
    publisher = {Springer International Publishing},
    pages = {102--118}
}
```

# Notes
Modify the hardcoded labels in the coco_utils and then train them: https://github.com/pytorch/vision/blob/main/references/segmentation/coco_utils.py

COCO Labels:
https://gist.github.com/salihkaragoz/8c3291b55abee745389de150df94bbca

## Old labels
[   '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

## New Labels
## TODO: List reduced classes
CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16]
CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]

0: __background__
1: person
2: bicycle
3: car
4: motorcycle
5: airplane
6: bus
7: train
8: truck
10: traffic light
11: fire hydrant
13: stop sign
14: parking meter
15: bench
16: bird
17: cat
18: dog
19: horse
20: sheep
21: cow

## Train Command
`torchrun train.py -h`

### Resnet50
torchrun train.py --data-path /mnt/excelsior/data/coco/data_raw/ --device cuda --lr 0.02 --dataset coco -b 12 -j 12 --epochs 30 --model deeplabv3_resnet50 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /data/models/vision/

torchrun train.py --data-path /mnt/scratch_disk/data/coco/data_raw/ --device cuda --lr 0.02 --dataset coco -b 12 -j 12 --epochs 30 --model deeplabv3_resnet50 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /data/models/vision/

### Resnet101
torchrun train.py --data-path /mnt/excelsior/data/coco/data_raw/ --device cuda --lr 0.02 --dataset coco -b 6 -j 12 --epochs 30 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /data/models/vision/

torchrun train.py --data-path /mnt/scratch_disk/data/coco/data_raw/ --device cuda --lr 0.02 --dataset coco -b 6 -j 12 --epochs 30 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /data/models/vision/

## Resume
### Resnet50
torchrun train.py --data-path /mnt/scratch_disk/data/coco/data_raw/ --device cuda --lr 0.02 --dataset coco -b 12 -j 12 --epochs 30 --model deeplabv3_resnet50 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /data/models/vision/ --start-epoch 5 --resume /mnt/excelsior/models/vision/checkpoint.pth

### Resnet101
torchrun train.py --data-path /mnt/scratch_disk/data/coco/data_raw/ --device cuda --lr 0.02 --dataset coco -b 6 -j 12 --epochs 30 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /data/models/vision/ --start-epoch 5 --resume /mnt/excelsior/models/vision/checkpoint.pth

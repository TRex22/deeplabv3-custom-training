#!/bin/bash
#SBATCH --job-name=deeplabv3
#SBATCH --output=/home-mscluster/jchalom/deeplabv3_result.txt
#SBATCH --ntasks=1
#SBATCH --time=600:00
#SBATCH --nodes=1 
#SBATCH --partition=biggpu

# torchrun train.py --data-path /mnt/excelsior/data/coco/data_raw/ --device cuda --lr 0.02 --dataset coco -b 48 -j 28 --epochs 30 --model deeplabv3_resnet50 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /data/models/vision/
torchrun train.py --data-path /mnt/excelsior/data/coco/data_raw/ --device cuda --lr 0.02 --dataset coco -b 24 -j 28 --epochs 30 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /data/models/vision/

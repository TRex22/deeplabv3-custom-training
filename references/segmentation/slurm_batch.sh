#!/bin/bash
#SBATCH --job-name=deeplabv3
#SBATCH --output=/home-mscluster/jchalom/deeplabv3_result.txt
#SBATCH --ntasks=1
#SBATCH --time=2550:00
#SBATCH --nodes=1 
#SBATCH --partition=biggpu

# torchrun train.py --data-path /home-mscluster/jchalom/data/coco/zips --device cuda --lr 0.02 --dataset coco -b 48 -j 28 --epochs 30 --model deeplabv3_resnet50 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /home-mscluster/jchalom/trained_models/deeplabv3
# torchrun train.py --data-path /home-mscluster/jchalom/data/coco/zips --device cuda --lr 0.02 --dataset coco -b 42 -j 28 --epochs 30 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /home-mscluster/jchalom/trained_models/deeplabv3

# Continue - set as needed
torchrun --nproc_per_node=2 train.py --data-path /home-mscluster/jchalom/data/coco/zips --device cuda --lr 0.02 --dataset coco -b 48 -j 48 --epochs 30 --model deeplabv3_resnet101 --aux-loss --weights-backbone ResNet101_Weights.IMAGENET1K_V1 --output-dir /home-mscluster/jchalom/trained_models/deeplabv3 --start-epoch 18 --resume /home-mscluster/jchalom/trained_models/deeplabv3/checkpoint.pth

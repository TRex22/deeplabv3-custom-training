#!/bin/bash
#SBATCH --job-name=dlv3SDG
#SBATCH --output=/home-mscluster/jchalom/deeplabv3_sgd_result.txt
#SBATCH --ntasks=1
#SBATCH --time=2550:00
#SBATCH --nodes=1 
#SBATCH --partition=biggpu

python custom_tarining.py /home-mscluster/jchalom/deeplabv3-custom-training/tools/config_48gb_SGD.json

#!/bin/bash
# conda env create --file deeplabv3.yml
conda env export | grep -v "^prefix: " > deeplabv3.yml

#!/bin/bash
conda env export | grep -v "^prefix: " > deeplabv3.yml

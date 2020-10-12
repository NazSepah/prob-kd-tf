#!/usr/bin/env bash
PYTHONPATH='./':$PYTHONPATH
PYTHONPATH=${PYTHONPATH} python ./src/training/test3d_unet.py "$@"

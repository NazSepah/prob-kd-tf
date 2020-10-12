#!/usr/bin/env bash
PYTHONPATH=.
PYTHONPATH=${PYTHONPATH} python ./src/training/train3d_unet.py "$@"

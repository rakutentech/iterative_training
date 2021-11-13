#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 4 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [binarization order] [learning rate] [gpu]"
  exit 1
fi

NAME=$1
MODEL=vgg9_layer
EPOCHS=450
EPOCHS_PER_LAYER=450
LAYERS_INDICES=$2
LEARNING_RATE=$3
GPU=$4

./scripts/cifar10/vgg9/val.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $GPU

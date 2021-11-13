#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 4 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [binarization order] [learning rate] [gpu]"
  exit 1
fi

NAME=$1
MODEL=vgg5_layer
EPOCHS=750
EPOCHS_PER_LAYER=150
LAYERS_INDICES=$2
LEARNING_RATE=$3
GPU=$4

./scripts/cifar10/vgg5/val.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $GPU

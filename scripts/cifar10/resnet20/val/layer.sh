#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 4 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [binarization order] [learning rate] [gpu]"
  exit 1
fi

NAME=$1
MODEL=resnet20_layer
LAYERS_INDICES=$2
LEARNING_RATE=$3
GPU=$4

./scripts/cifar10/resnet20/val_ep1200.sh $NAME $MODEL $LAYERS_INDICES $LEARNING_RATE $GPU

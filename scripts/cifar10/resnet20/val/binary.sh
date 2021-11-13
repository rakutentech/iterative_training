#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 3 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [learning rate] [gpu]"
  exit 1
fi

NAME=$1
MODEL=resnet20_binary
LAYERS_INDICES=forward
LEARNING_RATE=$2
GPU=$3

./scripts/cifar10/resnet20/val_ep1200.sh $NAME $MODEL $LAYERS_INDICES $LEARNING_RATE $GPU

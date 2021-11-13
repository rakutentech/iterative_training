#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 5 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [binarization order] [learning rate] [seed] [gpu]"
  exit 1
fi

NAME=$1
MODEL=vgg9_layer
EPOCHS=1350
EPOCHS_PER_LAYER=150
LAYERS_INDICES=$2
LEARNING_RATE=$3
SEED=$4
GPU=$5

./scripts/cifar10/vgg9/run.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $SEED $GPU

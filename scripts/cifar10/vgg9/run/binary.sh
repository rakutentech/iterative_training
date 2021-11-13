#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 4 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [learning rate] [seed] [gpu]"
  exit 1
fi

NAME=$1
MODEL=vgg9_binary
EPOCHS=1350
EPOCHS_PER_LAYER=150
LAYERS_INDICES=forward
LEARNING_RATE=$2
SEED=$3
GPU=$4

./scripts/cifar10/vgg9/run.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $SEED $GPU

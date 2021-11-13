#!/bin/bash -x

# check if command line argument is empty or not present
if [ "$#" -ne 5 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [binarization order] [learning rate] [seed] [gpu]"
  exit 1
fi

NAME=$1
MODEL=resnet20_layer
LAYERS_INDICES=$2
LEARNING_RATE=$3
SEED=$4
GPU=$5

./scripts/cifar10/resnet20/run_ep1200.sh $NAME $MODEL $LAYERS_INDICES $LEARNING_RATE $SEED $GPU

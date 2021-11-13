#!/bin/bash -x

# check if command line argument is empty or not present
if [ "$#" -ne 4 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [learning rate] [seed] [gpu]"
  exit 1
fi

NAME=$1
MODEL=resnet20
LAYERS_INDICES=forward
LEARNING_RATE=$2
SEED=$3
GPU=$4

./scripts/cifar10/resnet20/run_ep1200.sh $NAME $MODEL $LAYERS_INDICES $LEARNING_RATE $SEED $GPU

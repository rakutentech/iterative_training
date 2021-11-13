#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 5 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [model] [binarization order] [learning rate] [gpu]"
  exit 1
fi

NAME=$1
MODEL=$2
EPOCHS=1200
EPOCHS_PER_LAYER=50
#EPOCHS=24
#EPOCHS_PER_LAYER=1
LAYERS_INDICES=resnet20_layers_$3.json
MILESTONES=[1000,1100]
#MILESTONES=[20,22]
LEARNING_RATE=$4
DATASET_INDICES=cifar10_split0613.json
GPU=$5

./scripts/cifar10/resnet20_val.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $MILESTONES $LEARNING_RATE $DATASET_INDICES $GPU

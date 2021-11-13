#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 7 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [model] [# of epochs] [epochs/layer] [binarization order] [learning rate] [gpu]"
  exit 1
fi

NAME=$1
MODEL=$2
EPOCHS=$3
EPOCHS_PER_LAYER=$4
LAYERS_INDICES=vgg5_layers_$5.json
LEARNING_RATE=$6
DATASET_INDICES=cifar10_split0613.json
GPU=$7

./scripts/cifar10/vgg_val.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $DATASET_INDICES $GPU

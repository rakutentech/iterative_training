#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 8 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [model] [# of epochs] [epochs/layer] [binarization order] [learning rate] [seed] [gpu]"
  exit 1
fi

NAME=$1
MODEL=$2
EPOCHS=$3
EPOCHS_PER_LAYER=$4
LAYERS_INDICES=vgg9_layers_$5.json
LEARNING_RATE=$6
DATASET_INDICES=cifar10_split0613.json
SEED=$7
GPU=$8

./scripts/cifar10/vgg_run.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $DATASET_INDICES $SEED $GPU

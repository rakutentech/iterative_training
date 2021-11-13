#!/bin/bash -x

# check if command line argument is empty or not present
if [ "$#" -ne 6 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [model] [binarization order] [learning rate] [seed] [gpu]"
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
SEED=$5
DATASET_INDICES=cifar10_split0613.json
GPU=$6

./scripts/cifar10/resnet20_run.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $MILESTONES $LEARNING_RATE $SEED $DATASET_INDICES $GPU

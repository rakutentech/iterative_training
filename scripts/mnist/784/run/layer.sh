#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 5 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [binarization order] [learning rate] [seed] [gpu]"
  exit 1
fi

NAME=$1
MODEL=784_layer
EPOCHS=450
EPOCHS_PER_LAYER=150
LAYERS_INDICES=mlp3_layers_$2.json
LEARNING_RATE=$3
SEED=$4
GPU=$5

./scripts/mnist/mlp3_run.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $SEED $GPU

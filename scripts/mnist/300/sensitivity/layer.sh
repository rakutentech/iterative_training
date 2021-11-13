#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 4 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [binarization order] [learning rate] [gpu]"
  exit 1
fi


NAME=$1
MODEL=300_layer
EPOCHS=150
EPOCHS_PER_LAYER=150
LAYERS_INDICES=mlp3_layers_$2.json
LEARNING_RATE=$3
GPU=$4

./scripts/mnist/mlp3_val.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $GPU

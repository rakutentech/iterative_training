#!/bin/bash

# check if command line argument is empty or not present
if [ "$#" -ne 3 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [learning rate] [gpu]"
  exit 1
fi


NAME=$1
MODEL=300
EPOCHS=450
EPOCHS_PER_LAYER=150
LAYERS_INDICES=mlp3_layers_forward.json
LEARNING_RATE=$2
GPU=$3

./scripts/mnist/mlp3_val.sh $NAME $MODEL $EPOCHS $EPOCHS_PER_LAYER $LAYERS_INDICES $LEARNING_RATE $GPU

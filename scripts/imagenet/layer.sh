#!/bin/bash -x

# check if command line argument is empty or not present
if [ "$#" -ne 7 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [ImageNet folder] [# of epochs] [LR milestones] [epochs/layer] [layer indices] [learning rate]"
  exit 1
fi

NAME=$1
IMAGENET_DIR=$2
EPOCHS=$3
LR_MILESTONES=$4
EPOCHS_PER_LAYER=$5
LAYERS_INDICES=$6
LEARNING_RATE=$7

WORKERS=8

function run() {
  FOLDER=./logs/imagenet/$NAME
  mkdir -p $FOLDER
  python -u main_imagenet_binary.py \
    --gpu 0 \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --lr-milestones $LR_MILESTONES \
    --binarize \
    --epochs-per-layer $EPOCHS_PER_LAYER \
    --layers-json ./models/resnet21_layers_$LAYERS_INDICES.json \
    $IMAGENET_DIR \
    $FOLDER \
  2>&1 | tee $FOLDER/lr$LEARNING_RATE-ep$EPOCHS-epl$EPOCHS_PER_LAYER-resnet21_layers_$LAYERS_INDICES.txt
}


run

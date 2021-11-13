#!/bin/bash -x

# check if command line argument is empty or not present
if [ "$#" -ne 5 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [ImageNet folder] [# of epochs] [LR milestones] [learning rate]"
  exit 1
fi

NAME=$1
IMAGENET_DIR=$2
EPOCHS=$3
LR_MILESTONES=$4
LEARNING_RATE=$5

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
    --all-binary \
    --layers-json ./models/resnet21_layers_1.json \
    $IMAGENET_DIR \
    $FOLDER \
  2>&1 | tee $FOLDER/lr$LEARNING_RATE-ep$EPOCHS-all_binary.txt
}


run

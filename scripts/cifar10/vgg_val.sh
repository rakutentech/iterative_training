#!/bin/bash -x

# check if command line argument is empty or not present
if [ "$#" -ne 8 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [model] [# of epochs] [epochs/layer] [binarization order] [learning rate] [dataset indices] [gpu]"
  exit 1
fi

NAME=$1
MODEL=$2
#EPOCHS=10
#EPOCHS_PER_LAYER=1
EPOCHS=$3
EPOCHS_PER_LAYER=$4
LAYERS_INDICES=$5
LEARNING_RATE=$6
DATASET_INDICES=$7
GPU=$8

DATASET=cifar10
BATCH_SIZE=100
TEST_BATCH_SIZE=1000
LOG_INTERVAL=150
PATIENCE=0

#    --test-training \
function run() {
  FOLDER=./logs/$DATASET/val/$NAME/$MODEL-lr$LEARNING_RATE-ep$EPOCHS-eplayer$EPOCHS_PER_LAYER-$LAYERS_INDICES-$DATASET_INDICES
  mkdir -p $FOLDER
  CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --gpu 0 \
    --dataset $DATASET \
    --dataset-indices $DATASET_INDICES \
    --use-data-augmentation \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --test-batch-size $TEST_BATCH_SIZE \
    --lr $LEARNING_RATE \
    --log-interval $LOG_INTERVAL \
    --epochs $EPOCHS \
    --epochs-per-layer $EPOCHS_PER_LAYER \
    --layers-indices ./models/$LAYERS_INDICES \
    --patience $PATIENCE \
    --logdir $FOLDER \
    --test-validation \
  2>&1 | tee $FOLDER/log.txt
}


run

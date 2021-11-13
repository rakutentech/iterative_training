#!/bin/bash -x

# check if command line argument is empty or not present
if [ "$#" -ne 8 ]; then
  echo "Invalid parameters!"
  echo ""
  echo "Usage: $0 [name] [model] [# of epochs] [epochs/layer] [layer indices] [learning rate] [seed] [gpu]"
  exit 1
fi

NAME=$1
MODEL=$2
#EPOCHS=6
#EPOCHS_PER_LAYER=2
EPOCHS=$3
EPOCHS_PER_LAYER=$4
LAYERS_INDICES=$5
LEARNING_RATE=$6
SEED=$7
GPU=$8

DATASET=mnist
BATCH_SIZE=100
TEST_BATCH_SIZE=1000
LOG_INTERVAL=150
PATIENCE=0

function run() {
  FOLDER=./logs/$DATASET/run/$NAME/$MODEL-lr$LEARNING_RATE-seed$SEED-ep$EPOCHS-eplayer$EPOCHS_PER_LAYER-$LAYERS_INDICES
  mkdir -p $FOLDER
  CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --gpu 0 \
    --dataset $DATASET \
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
    --seed $SEED \
    --test-training \
  2>&1 | tee $FOLDER/log.txt
}


run

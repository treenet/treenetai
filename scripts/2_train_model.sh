#!/bin/bash

export EXPERIMENTS=/storage/lukovic/Data/FORWARDS/treenet/experiments/
export DATA=/storage/lukovic/Data/FORWARDS/treenet/processed_data/tfrecords/

EXP_DESC="test_GPU_node04"

python training.py \
  --timeseries_label \
  --proj_name 'TNT' \
  --experiment_type 'reconstruction' \
  --exp_description $EXP_DESC \
  --model_name CNN_LSTM_reconstruction \
  --test_batch 64 \
  --train_batch 64 \
  --optimizer 'adam' \
  --verbose 2 \
  --lr 0.00001 \
  --epochs 5 \
  --file_id 9   # TODO: construct a test that makes sure the correct input data file is used

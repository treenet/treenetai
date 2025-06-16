#!/bin/bash

#export EXPERIMENTS=/home/lukovic/data/treenet/experiments
#export DATA=/storage/lukovic/Data/FORWARDS/treenet/tfrecords/

EXP_DESC="test_GPU_node05_1"

python3 ~/codes/treenetai/training.py \
  --timeseries_label \
  --proj_name 'TNT-climate-data' \
  --experiment_type 'climate-processing' \
  --exp_description $EXP_DESC \
  --model_name climate_processing_LSTM_CNN \
  --test_batch 12 \
  --train_batch 12 \
  --optimizer 'adam' \
  --verbose 2 \
  --lr 0.00001 \
  --epochs 10 \
  --data_file_id 1   # TODO: construct a test that makes sure the correct input data file is used

#!/bin/bash

export EXPERIMENTS=/Users/lukovic/Data/FORWARDS/TNT/experiments/
export DATA=/Users/lukovic/Data/FORWARDS/TNT/tfrecords/

python evaluation.py 20240625-193439-CNN_LSTM_reconstruction 0 0
python evaluation.py 20240626-012054-CNN_LSTM_reconstruction 0 0
python evaluation.py 20240626-012111-CNN_LSTM_reconstruction 0 0

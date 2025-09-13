#!/bin/bash

export EXPERIMENTS=/home/lukovic/data/treenet/experiments/
export DATA=/storage/lukovic/Data/FORWARDS/treenet/tfrecords/

ID=20250611-141849
task=climate-processing

python3 ~/codes/treenetai/evaluation.py $ID $task


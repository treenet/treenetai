#!/bin/bash

meta_path="/home/lukovic/data/treenet/server_data/clima.pkl"
dendro_path="/home/lukovic/data/treenet/server_data/data_dendro_lm.pkl"
clima_path="/home/lukovic/data/treenet/server_data/clima.pkl"
out_folder="/home/lukovic/data/treenet/server_data/"

python3 ~/codes/treenetai/raw_data_elaboration/rearrange_raw_data.py $meta_path $dendro_path $clima_path $out_folder


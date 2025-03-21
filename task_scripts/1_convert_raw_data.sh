#!/bin/bash

meta_path="/home/lukovic/data/treenet/server_data/metadata.pkl"
dendro_path="/home/lukovic/data/treenet/server_data/data_dendro_lm.pkl"
clima_path="/home/lukovic/data/treenet/server_data/climate.pkl"
out_folder="/home/lukovic/data/treenet/server_data/"

python3 ~/codes/treenetai/raw_data_elaboration/rearrange_raw_data.py $meta_path $dendro_path $clima_path $out_folder


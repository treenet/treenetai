#!/bin/bash

meta_path="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_dendrometer.pkl"
dendro_path="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_dendro_lm_dictionary.pkl"
clima_path="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_meteo_lm_dictionary.pkl"
out_folder="/home/lukovic/data/treenet/server_data/"

python3 ~/codes/treenetai/raw_data_elaboration/combine_dendro_clima_data.py $meta_path $dendro_path $clima_path $out_folder

mv /home/lukovic/data/treenet/server_data/* /storage/lukovic/Data/FORWARDS/treenet/server_data/
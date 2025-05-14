#!/bin/bash

clima_l2="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_meteo_l2_dictionary.pkl"
clima_lm="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_meteo_lm_dictionary.pkl"
temperature_l0="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_all_l0_temperature_dictionary.pkl"
temperature_l1="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_all_l1_temperature_dictionary.pkl"
humidity_l0="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_all_l0_humidity_dictionary.pkl"
humidity_l1="/storage/lukovic/Data/FORWARDS/treenet/server_data/data_all_l1_humidity_dictionary.pkl"

meta="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_all.pkl"
meta_temperature_l0="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_data_all_l0_temperature.pkl"
meta_temperature_l1="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_data_all_l1_temperature.pkl"
meta_humidity_l0="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_data_all_l0_humidity.pkl"
meta_humidity_l1="/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_data_all_l1_humidity.pkl"


out_folder="/home/lukovic/data/treenet/server_data/" 

python3 ~/codes/treenetai/raw_data_elaboration/combine_dendro_clima_data.py $meta_path $dendro_path $clima_path $out_folder

mv /home/lukovic/data/treenet/server_data/* /storage/lukovic/Data/FORWARDS/treenet/server_data/